import os
import sys
import re

def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def generate_bundle(root_dir, output_dir):
    header_path = os.path.join(output_dir, 'lygia.h')
    source_path = os.path.join(output_dir, 'lygia.cpp')

    files_map = {}

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude hidden directories/files and build artifacts if present
        if '/.' in dirpath or '\\.' in dirpath:
            continue

        for filename in filenames:
            if filename.endswith('.glsl'):
                full_path = os.path.join(dirpath, filename)
                # Calculate relative path from root_dir
                rel_path = os.path.relpath(full_path, root_dir)
                # Ensure forward slashes and prepend 'lygia/'
                key_path = 'lygia/' + rel_path.replace(os.path.sep, '/')

                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content = remove_comments(content)
                        files_map[key_path] = content
                except Exception as e:
                    print(f"Skipping {full_path}: {e}")

    # Sort by key so the generated table can be binary-searched
    sorted_items = sorted(files_map.items())

    # Write Header
    with open(header_path, 'w') as f:
        f.write('#pragma once\n')
        f.write('#include <string>\n\n')
        f.write('// LYGIA, Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0\n')
        f.write('// LYGIA, Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license\n')
        f.write('std::string getLygiaFile(const std::string& _path);\n')

    # Write Source
    #
    # The table below is a plain, pre-sorted array of C-string literal pairs
    # (not a std::map<std::string,std::string>). This matters for size: a
    # std::map initializer would heap-allocate a std::string for every single
    # key AND value up front, for every one of the ~1000+ files, the moment
    # any one of them is requested. Plain string literal pointers cost nothing
    # at startup - the text lives in .rodata and is only ever copied into a
    # std::string for the one file actually requested, on demand.
    #
    # Files larger than the per-literal chunk size are split into adjacent
    # raw string literals (e.g. R"a(...)a" R"b(...)b"), which the compiler
    # concatenates into a single literal at compile time with no runtime
    # cost - this only exists to stay under MSVC's ~16k-64k limit on a single
    # string literal token (C2026).
    with open(source_path, 'w') as f:
        f.write('#include "lygia.h"\n')
        f.write('#include <cstring>\n\n')

        f.write('struct LygiaFile { const char* path; const char* content; };\n\n')

        f.write('static const LygiaFile LYGIA_FILES[] = {\n')

        for key, content in sorted_items:
            delimiter = "LYGIA_CONTENT"
            while delimiter in content:
                delimiter += "_"

            # Split content into chunks to avoid C2026 on MSVC (limit around 16k-64k).
            # Adjacent raw string literals are concatenated by the compiler at
            # compile time, so this costs nothing at runtime.
            chunk_size = 2048
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            if len(chunks) == 0:
                f.write(f'    {{"{key}", ""}},\n')
            else:
                literal = " ".join(f'R"{delimiter}({chunk}){delimiter}"' for chunk in chunks)
                f.write(f'    {{"{key}", {literal}}},\n')

        f.write('};\n\n')
        f.write('static const size_t LYGIA_FILES_COUNT = sizeof(LYGIA_FILES) / sizeof(LYGIA_FILES[0]);\n\n')

        f.write('std::string getLygiaFile(const std::string& _path) {\n')
        f.write('    // LYGIA_FILES is sorted by path, so we can binary search it directly\n')
        f.write('    // instead of paying for a std::map<std::string, std::string> of every file.\n')
        f.write('    size_t lo = 0;\n')
        f.write('    size_t hi = LYGIA_FILES_COUNT;\n')
        f.write('    while (lo < hi) {\n')
        f.write('        size_t mid = lo + (hi - lo) / 2;\n')
        f.write('        int cmp = std::strcmp(_path.c_str(), LYGIA_FILES[mid].path);\n')
        f.write('        if (cmp == 0)\n')
        f.write('            return LYGIA_FILES[mid].content;\n')
        f.write('        else if (cmp < 0)\n')
        f.write('            hi = mid;\n')
        f.write('        else\n')
        f.write('            lo = mid + 1;\n')
        f.write('    }\n')
        f.write('    return "";\n')
        f.write('}\n')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bundle.py <root_dir> <output_dir>")
        sys.exit(1)

    generate_bundle(sys.argv[1], sys.argv[2])
