#!/usr/bin/env bash
# Compiles every *.msl file in LYGIA as a standalone translation unit
# using the Metal compiler. Requires macOS with the Metal toolchain.
#
# usage: test/msl/compile.sh [file.msl ...]

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STD="${METAL_STD:-metal3.1}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"

if [ "$#" -gt 0 ]; then
    printf '%s\n' "$@" | sed 's|^\./||' > "$TMP/list"
else
    find . -name '*.msl' -not -path './.git/*' -not -path './node_modules/*' | sed 's|^\./||' | sort > "$TMP/list"
fi

compile_one() {
    f="$1"
    id="$(echo "$f" | tr '/.' '__')"
    printf '#include <metal_stdlib>\nusing namespace metal;\n#include "%s/%s"\n' "$ROOT" "$f" > "$TMP/$id.metal"
    # print each result in a single write so parallel jobs don't interleave
    if xcrun -sdk macosx metal -std="$STD" -c "$TMP/$id.metal" -o "$TMP/$id.air" 2> "$TMP/$id.err"; then
        printf 'OK   %s\n' "$f"
    else
        printf 'FAIL %s\n%s\n' "$f" "$(grep -m3 'error:' "$TMP/$id.err" | sed "s|$ROOT/||g; s|^|     |")"
    fi
}
export -f compile_one
export ROOT STD TMP

xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {} < "$TMP/list" > "$TMP/results"

awk '/^FAIL/{p=1; print; next} /^OK/{p=0} p' "$TMP/results"
total=$(grep -c '' "$TMP/list")
failed=$(grep -c '^FAIL' "$TMP/results")
echo "MSL: $((total - failed))/$total compiled"

# All files together in one translation unit, to catch clashes between modules
# (names that shadow Metal types, ...). Each file is included twice, to catch
# missing include guards.
{
    printf '#include <metal_stdlib>\nusing namespace metal;\n'
    sed "s|.*|#include \"$ROOT/&\"|" "$TMP/list"
    sed "s|.*|#include \"$ROOT/&\"|" "$TMP/list"
} > "$TMP/all.metal"
if xcrun -sdk macosx metal -std="$STD" -c "$TMP/all.metal" -o "$TMP/all.air" 2> "$TMP/all.err"; then
    echo "MSL: combined translation unit compiled"
    combined=0
else
    echo "FAIL combined translation unit"
    grep 'error:' "$TMP/all.err" | sed "s|$ROOT/||g; s|^|     |" | head -20
    combined=1
fi

# Link two translation units that include everything, like an app with two
# .metal files that share LYGIA includes. Fails with duplicate symbols unless
# every function is inline.
linked=0
if [ "$combined" -eq 0 ]; then
    cp "$TMP/all.metal" "$TMP/all2.metal"
    if xcrun -sdk macosx metal -std="$STD" -c "$TMP/all2.metal" -o "$TMP/all2.air" 2>> "$TMP/all.err" &&
       xcrun -sdk macosx metallib "$TMP/all.air" "$TMP/all2.air" -o "$TMP/all.metallib" 2> "$TMP/link.err"; then
        echo "MSL: two translation units linked"
    else
        echo "FAIL linking two translation units"
        head -3 "$TMP/link.err" | sed 's|^|     |'
        linked=1
    fi
fi

# The lighting API uses templates for its optional environment texture, which
# only get checked when called. test/msl/instantiate/*.metal call every
# overload under different options; compile them all and link them together.
instantiated=0
if [ "$#" -eq 0 ] && [ -d "$ROOT/test/msl/instantiate" ]; then
    airs=""
    for f in "$ROOT"/test/msl/instantiate/*.metal; do
        air="$TMP/inst_$(basename "$f" .metal).air"
        if xcrun -sdk macosx metal -std="$STD" -c "$f" -o "$air" 2> "$TMP/inst.err"; then
            airs="$airs $air"
        else
            echo "FAIL test/msl/instantiate/$(basename "$f")"
            grep -m3 'error:' "$TMP/inst.err" | sed "s|$ROOT/||g; s|^|     |"
            instantiated=1
        fi
    done
    if [ "$instantiated" -eq 0 ]; then
        if xcrun -sdk macosx metallib $airs -o "$TMP/inst.metallib" 2> "$TMP/inst.err"; then
            echo "MSL: lighting overloads instantiated ($(ls "$ROOT"/test/msl/instantiate/*.metal | wc -l | tr -d ' ') configurations)"
        else
            echo "FAIL linking test/msl/instantiate"
            head -3 "$TMP/inst.err" | sed 's|^|     |'
            instantiated=1
        fi
    fi
fi

[ "$failed" -eq 0 ] && [ "$combined" -eq 0 ] && [ "$linked" -eq 0 ] && [ "$instantiated" -eq 0 ]
