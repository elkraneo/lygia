#!/usr/bin/env python3
"""Checks each claim in test/msl/proof/claims on the GPU.

A claim is a snippet that calls LYGIA functions and writes float4 results. It
runs three times on the same GPU:

  GLSL    LYGIA's GLSL files, compiled with glslangValidator and translated to
          Metal with spirv-cross. This is the reference, unless the claim gives
          expected values because GLSL has the bug too.
  before  LYGIA's Metal files at another commit (upstream main by default).
  after   LYGIA's Metal files in this checkout.

A claim is shown when "after" matches the reference and "before" doesn't. The
snippet is written with GLSL types (vec2, mat3, ...), which test/msl/proof/prelude.h
maps to Metal types.

Claim files (claims/*.claim):

  // @claim text                 what the claim says (one line)
  // @include math/rotate2d      a LYGIA file, without extension (.glsl / .msl is added)
  // @define NAME [value]        defined before the includes, in every version
  // @count 2                    number of float4 results (default 1)
  // @expect i x [y [z [w]]]     expected values for result i; makes the values the
  //                             reference instead of GLSL
  // @eps 0.001                  tolerance (default 0.0001)
  // @before <git ref>           commit for "before" (default origin/main)
  // @note text                  shown in the report
  results[0] = vec4(rotate2d(1.0) * vec2(1.0, 0.0), 0.0, 0.0);

Use `#ifdef __METAL_VERSION__` in a snippet for code that must differ between
the languages. Requires macOS with Xcode, glslang and spirv-cross (brew).

usage: test/msl/proof/proof.py [--markdown FILE] [claim ...]

proof.sh runs this with compile.sh and linking.sh and writes RESULTS.md.
"""

import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROOF = Path(__file__).resolve().parent
ROOT = PROOF.parent.parent.parent
TMP = Path(tempfile.mkdtemp(prefix="lygia-proof-"))


def sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def parse(path):
    claim = {"name": path.stem, "claim": "", "include": [], "define": [], "count": 1,
             "expect": {}, "eps": 1e-4, "before": "origin/main", "note": "", "body": []}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*//\s*@(\w+)\s*(.*)", line)
        if not m:
            claim["body"].append(line)
            continue
        key, value = m.group(1), m.group(2).strip()
        if key in ("include", "define"):
            claim[key].append(value)
        elif key == "expect":
            i, *xs = value.split()
            claim["expect"][int(i)] = [float(x) for x in xs]
        elif key == "count":
            claim["count"] = int(value)
        elif key == "eps":
            claim["eps"] = float(value)
        elif key in claim:
            claim[key] = value
        else:
            sys.exit(f"{path.name}: unknown annotation @{key}")
    claim["body"] = "\n".join(claim["body"]).strip()
    return claim


def tree(ref):
    """LYGIA's .msl files at a commit, extracted once per commit."""
    d = TMP / "trees" / re.sub(r"[^\w.-]", "_", ref)
    if not d.exists():
        d.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", ref, "--", ":(glob)**/*.msl"], cwd=ROOT, capture_output=True)
        if archive.returncode != 0:
            sys.exit(f"git archive {ref}: {archive.stderr.decode().strip()}")
        subprocess.run(["tar", "-x", "-C", str(d)], input=archive.stdout, check=True)
    return d


def include_root(lygia):
    """A folder with a `lygia` symlink, so "lygia/..." includes resolve."""
    d = TMP / "roots" / str(abs(hash(str(lygia))))
    if not d.exists():
        d.mkdir(parents=True)
        (d / "lygia").symlink_to(lygia)
    return d


def first_error(text):
    for line in text.splitlines():
        if re.search(r"\berror\b", line, re.I) and "compilation terminated" not in line:
            line = re.sub(r"\S*/build/[^/]+/k\.(comp|metal)", "snippet", line)
            line = re.sub(r"\S*/(roots/[^/]+|trees/[^/]+)/(lygia/)?", "", line).replace(str(ROOT) + "/", "")
            return re.sub(r"^.*?\b(error)\b:?\s*", "", line, flags=re.I).strip()[:160]
    return text.strip().splitlines()[-1][:160] if text.strip() else "failed"


def metallib(src, name, lygia):
    """Compiles Metal source to a metallib; returns (path, None) or (None, error)."""
    out = TMP / "build" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "k.metal").write_text(src)
    r = sh("xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-w", "-I", str(include_root(lygia)),
           str(out / "k.metal"), "-o", str(out / "k.metallib"))
    return (out / "k.metallib", None) if r.returncode == 0 else (None, first_error(r.stderr))


def run(lib, kernel, count):
    r = sh(str(TMP / "run"), str(lib), kernel, str(count))
    if r.returncode != 0:
        return None, first_error(r.stdout + r.stderr)
    return [[float(x) for x in line.split()] for line in r.stdout.splitlines()], None


def defines(claim):
    return "".join(f"#define {d}\n" for d in claim["define"])


def glsl(claim):
    """GLSL version: GLSL -> SPIR-V -> Metal. Returns (results, error)."""
    missing = [i for i in claim["include"] if not (ROOT / f"{i}.glsl").exists()]
    if missing:
        return None, f"no {missing[0]}.glsl"
    out = TMP / "build" / f"{claim['name']}-glsl"
    out.mkdir(parents=True, exist_ok=True)
    includes = "".join(f'#include "lygia/{i}.glsl"\n' for i in claim["include"])
    (out / "k.comp").write_text(
        "#version 450\n#extension GL_GOOGLE_include_directive : require\n" + defines(claim) + includes +
        "layout(std430, binding = 0) buffer Results { vec4 results[]; };\n"
        "layout(local_size_x = 1) in;\n"
        f"void main() {{\n{claim['body']}\n}}\n")
    r = sh("glslangValidator", "-V", "-I" + str(include_root(ROOT)), str(out / "k.comp"), "-o", str(out / "k.spv"))
    if r.returncode != 0:
        return None, first_error(r.stdout + r.stderr)
    r = sh("spirv-cross", str(out / "k.spv"), "--msl", "--msl-version", "30100", "--output", str(out / "k.metal"))
    if r.returncode != 0:
        return None, "spirv-cross: " + first_error(r.stderr)
    lib, error = metallib((out / "k.metal").read_text(), f"{claim['name']}-glsl-msl", ROOT)
    return run(lib, "main0", claim["count"]) if lib else (None, error)


def msl(claim, lygia):
    missing = [i for i in claim["include"] if not (lygia / f"{i}.msl").exists()]
    if missing:
        return None, f"no {missing[0]}.msl"
    includes = "".join(f'#include "lygia/{i}.msl"\n' for i in claim["include"])
    src = ("#include <metal_stdlib>\nusing namespace metal;\n" + defines(claim) + includes +
           f'#include "{PROOF}/prelude.h"\n'
           f"kernel void proof(device float4* results [[buffer(0)]]) {{\n{claim['body']}\n}}\n")
    lib, error = metallib(src, f"{claim['name']}-{abs(hash(str(lygia)))}", lygia)
    return run(lib, "proof", claim["count"]) if lib else (None, error)


def compare(results, reference, eps):
    """Returns None if they match, or a description of the first difference."""
    for i, want in reference.items():
        got = results[i]
        for c, w in enumerate(want):
            if not (abs(got[c] - w) <= eps):  # NaN never matches
                return f"[{i}] {fmt(got[:len(want)])}, expected {fmt(want)}"
    return None


def fmt(v):
    return "(" + ", ".join("nan" if math.isnan(x) else f"{x:.6g}" for x in v) + ")"


def check(claim):
    g, g_error = glsl(claim)
    after, a_error = msl(claim, ROOT)
    before, b_error = msl(claim, tree(claim["before"]))

    if claim["expect"]:
        reference, source = claim["expect"], "expected values"
    elif g is not None:
        reference, source = {i: v for i, v in enumerate(g)}, "GLSL"
    else:
        reference, source = None, "GLSL"

    def status(results, error):
        if error:
            return "error", error
        if reference is None:
            return "error", f"no reference: {g_error}"
        diff = compare(results, reference, claim["eps"])
        return ("match", "") if diff is None else ("differs", diff)

    row = {"claim": claim, "source": source,
           "after": status(after, a_error), "before": status(before, b_error),
           "glsl": status(g, g_error) if claim["expect"] else ("reference", "") if g else ("error", g_error)}
    # Shown: this checkout matches the reference, and "before" didn't (differs or didn't compile).
    row["shown"] = row["after"][0] == "match" and row["before"][0] != "match"
    return row


def main():
    args = sys.argv[1:]
    markdown = None
    if args[:1] == ["--markdown"]:
        markdown, args = Path(args[1]), args[2:]
    paths = [Path(a) for a in args] or sorted((PROOF / "claims").glob("*.claim"))
    r = sh("swiftc", "-O", str(PROOF / "run.swift"), "-o", str(TMP / "run"))
    if r.returncode != 0:
        sys.exit(r.stderr)

    rows = []
    for path in paths:
        row = check(parse(path))
        rows.append(row)
        c = row["claim"]
        mark = "SHOWN" if row["shown"] else "NOT SHOWN"
        print(f"{mark:9} {c['name']}: {c['claim']}")
        for key in ("glsl", "before", "after"):
            state, detail = row[key]
            label = {"before": f"before ({c['before']})", "after": "after", "glsl": "GLSL"}[key]
            print(f"          {label}: {state}{' ' + detail if detail else ''}")

    shown = sum(r["shown"] for r in rows)
    print(f"\nMSL proof: {shown}/{len(rows)} claims shown")
    if markdown:
        write_markdown(markdown, rows)
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(0 if shown == len(rows) else 1)


def cell(state):
    s, detail = state
    if s == "match":
        return "✓"
    if s == "reference":
        return "reference"
    if s == "differs":
        return "✗ differs"
    return "✗ missing" if detail.startswith("no ") and detail.endswith(".msl") else "✗ doesn't compile"


def write_markdown(path, rows):
    """The claims section of RESULTS.md (proof.sh adds the rest)."""
    lines = [
        "| Claim | Reference | GLSL | Metal before | Metal after |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["claim"]
        lines.append(f"| [{c['name']}](#{c['name'].lower()}) | {r['source']} | {cell(r['glsl'])} | "
                     f"{cell(r['before'])} | {cell(r['after'])} |")
    lines += ["", f"**{sum(r['shown'] for r in rows)}/{len(rows)} claims shown.**", ""]
    for r in rows:
        c = r["claim"]
        lines += [f"#### {c['name']}", "", f"{c['claim']}. {c['note']}", "",
                  f"Snippet: [`claims/{c['name']}.claim`](claims/{c['name']}.claim)", ""]
        for key, label in (("glsl", "GLSL"), ("before", f"Metal before ({c['before']})"), ("after", "Metal after")):
            state, detail = r[key]
            if detail:
                lines.append(f"- {label}: `{detail}`")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
