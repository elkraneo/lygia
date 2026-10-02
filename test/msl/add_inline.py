#!/usr/bin/env python3
"""Prefix every top-level function definition in LYGIA .msl files with `LYGIA_FNC`,
and include lygia.msl, which defines it (`static inline` by default).

Without it, an app with two .metal files that include the same LYGIA file
fails to link with duplicate symbols. Plain `inline` links, but when two files
include a function with different options (e.g. FBM_OCTAVES), the linker keeps
one definition for both; `static` gives each file its own copy.

Skips block/line comments and preprocessor lines (including macro
continuations), function prototypes (ending in `;`), definitions that are
already inline, and anything inside braces (struct members, function bodies).

usage: test/msl/add_inline.py file.msl [...]
"""
import re
import sys
from pathlib import Path

SKIP_WORDS = {'inline', 'LYGIA_FNC', 'return', 'if', 'for', 'while', 'else', 'do', 'switch', 'case',
              'constant', 'const', 'struct', 'typedef', 'using', 'namespace', 'static',
              'kernel', 'vertex', 'fragment', 'template', 'enum', 'union'}
HEAD = re.compile(r'^([A-Za-z_][\w<>:,]*(?:\s+[A-Za-z_][\w<>:,&*]*)*?)\s+[&*]?\s*([A-Za-z_]\w*)\s*\(')


def strip_comments_state(line, in_block):
    """Return (code_without_comments, in_block_after)."""
    out = []
    i = 0
    while i < len(line):
        if in_block:
            end = line.find('*/', i)
            if end < 0:
                return ''.join(out), True
            i = end + 2
            in_block = False
        elif line.startswith('/*', i):
            in_block = True
            i += 2
        elif line.startswith('//', i):
            break
        else:
            out.append(line[i])
            i += 1
    return ''.join(out), in_block


def process(text):
    lines = text.split('\n')
    code = []
    in_block = False
    for line in lines:
        c, in_block_after = strip_comments_state(line, in_block)
        # a line that starts inside a block comment is not code
        code.append(None if in_block else c)
        in_block = in_block_after

    changed = 0
    in_macro = False
    depth_at = []
    d = 0
    for c in code:
        depth_at.append(d)
        for ch in (c or ''):
            d += ch == '{'
            d -= ch == '}'
    for i, c in enumerate(code):
        prev_macro = in_macro
        raw = lines[i]
        in_macro = raw.rstrip().endswith('\\') and (prev_macro or raw.lstrip().startswith('#'))
        if c is None or prev_macro or depth_at[i] != 0:
            continue
        indent = len(c) - len(c.lstrip())
        c = c.lstrip()
        if not c or c.startswith('#'):
            continue
        m = HEAD.match(c)
        if not m:
            continue
        first = m.group(1).split()[0]
        if first in SKIP_WORDS:
            continue
        # scan forward to the end of the parameter list, then look for { or ;
        depth = 0
        seen_paren = False
        verdict = None
        for j in range(i, min(i + 40, len(code))):
            s = code[j] or ''
            start = indent + m.end() - 1 if j == i else 0
            for ch in s[start:]:
                if ch == '(':
                    depth += 1
                    seen_paren = True
                elif ch == ')':
                    depth -= 1
                elif seen_paren and depth == 0 and ch in '{;':
                    verdict = ch
                    break
            if verdict:
                break
        if verdict == '{':
            lines[i] = raw[:indent] + 'LYGIA_FNC ' + raw[indent:]
            changed += 1
    return '\n'.join(lines), changed


def add_include(text, path):
    """Insert `#include ".../lygia.msl"` after the doc comment, unless present."""
    if 'lygia.msl"' in text:
        return text, 0
    root = Path(__file__).resolve().parents[2]
    depth = len(path.resolve().relative_to(root).parts) - 1
    inc = '#include "' + '../' * depth + 'lygia.msl"'
    lines = text.split('\n')
    i = 0
    if lines and lines[0].startswith('/*'):
        while i < len(lines) and '*/' not in lines[i]:
            i += 1
        i += 1
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    lines[i:i] = [inc, '']
    return '\n'.join(lines), 1


def main(paths):
    total = 0
    for p in paths:
        path = Path(p)
        new, n = process(path.read_text())
        new, added = add_include(new, path)
        n += added
        if n:
            path.write_text(new)
            total += n
    print(f'{total} definitions marked LYGIA_FNC')


if __name__ == '__main__':
    main(sys.argv[1:])
