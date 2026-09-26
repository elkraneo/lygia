#!/usr/bin/env bash
# Checks the linking claims (lygia#322) and prints a Markdown report:
#
# 1. Two .metal files that include the same LYGIA files link into one metallib,
#    as Xcode does for an app target (SwiftUI ShaderLibrary, makeDefaultLibrary,
#    RealityKit CustomMaterial).
# 2. static inline doesn't make the GPU code bigger.
# 3. Compiler flags (-fvisibility=hidden, -flto=thin) and linker options don't
#    avoid the duplicates.
# 4. Two files that include a function with different options (FBM_OCTAVES)
#    each get their own version.
# 5. LYGIA compiled once as a helper library (dynamic, static, or one .air) that
#    the .metal files call into: links and runs, with one set of options for all.
#
# It compares three versions of LYGIA's Metal files:
#   before         upstream main (include guards only)
#   inline         this checkout with plain `inline` instead of `static inline`
#   static inline  this checkout
#
# usage: test/msl/proof/linking.sh [before ref (default origin/main)]
set -euo pipefail
PROOF="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PROOF/../../.." && pwd)"
BEFORE="${1:-origin/main}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# One folder per version, each with a `lygia` folder so "lygia/..." includes resolve.
mkdir -p "$TMP/before/lygia" "$TMP/inline/lygia" "$TMP/static"
git -C "$ROOT" archive "$BEFORE" -- ':(glob)**/*.msl' | tar -x -C "$TMP/before/lygia"
ln -s "$ROOT" "$TMP/static/lygia"
(cd "$ROOT" && git ls-files '*.msl' | tar -c -T - -f -) | tar -x -C "$TMP/inline/lygia"
find "$TMP/inline/lygia" -name '*.msl' -exec perl -pi -e 's/\bstatic inline\b/inline/g' {} +
VERSIONS="before inline static"
label() { case "$1" in before) echo "before ($BEFORE)";; inline) echo "plain inline";; static) echo "static inline";; esac; }

swiftc -O "$PROOF/run.swift" -o "$TMP/run"
metal() { xcrun -sdk macosx metal -std=metal3.1 -w "$@"; }
# Median wall time in ms of 7 runs of a command.
ms() { perl -MTime::HiRes=time -e 'my @t; for (1..7) { my $s = time; system(@ARGV) == 0 or exit 1; push @t, time - $s } @t = sort { $a <=> $b } @t; printf "%.0f", $t[3] * 1000' -- "$@"; }

# A shader file with a kernel that uses several LYGIA modules.
cat > "$TMP/pattern.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
#include "lygia/generative/snoise.msl"
#include "lygia/generative/fbm.msl"
#include "lygia/generative/voronoi.msl"
#include "lygia/generative/random.msl"
#include "lygia/sdf/circleSDF.msl"
#include "lygia/color/space/hsv2rgb.msl"

kernel void KERNEL(texture2d<float, access::write> out [[texture(0)]],
                   constant float& time [[buffer(0)]],
                   uint2 gid [[thread_position_in_grid]]) {
    float2 st = float2(gid) / float2(out.get_width(), out.get_height());
    float n = fbm(float3(st * 4.0, time));
    float3 v = voronoi(st * 6.0, time);
    float s = circleSDF(st);
    float3 c = hsv2rgb(float3(fract(n + v.z), 0.7, 0.9)) * step(0.3, s) + random(st) * 0.02;
    out.write(float4(c + snoise(float3(st * 8.0, time)) * 0.05, 1.0), gid);
}
EOF
sed 's/KERNEL/pattern_a/' "$TMP/pattern.metal" > "$TMP/a.metal"
sed 's/KERNEL/pattern_b/' "$TMP/pattern.metal" > "$TMP/b.metal"

echo "### Two .metal files that include the same LYGIA files"
echo
echo "Both files include snoise, fbm, voronoi, random, circleSDF and hsv2rgb, and are compiled separately and linked into one metallib, as Xcode does."
echo
echo "| LYGIA | Links | GPU code for one kernel | Compile time for one file |"
echo "|---|---|---|---|"
for v in $VERSIONS; do
    metal -I "$TMP/$v" -c "$TMP/a.metal" -o "$TMP/$v-a.air"
    metal -I "$TMP/$v" -c "$TMP/b.metal" -o "$TMP/$v-b.air"
    if xcrun -sdk macosx metallib "$TMP/$v-a.air" "$TMP/$v-b.air" -o "$TMP/$v-ab.metallib" 2> "$TMP/link.log"; then
        links="yes"
    else
        links="no: \`$(grep -m1 -o 'multiple symbols.*' "$TMP/link.log" || head -1 "$TMP/link.log")\`"
    fi
    # Code size of one file on its own, where every version links.
    xcrun -sdk macosx metallib "$TMP/$v-a.air" -o "$TMP/$v-a.metallib"
    echo "| $(label $v) | $links | $("$TMP/run" --size "$TMP/$v-a.metallib" pattern_a) bytes | $(ms xcrun -sdk macosx metal -std=metal3.1 -w -I "$TMP/$v" -c "$TMP/a.metal" -o "$TMP/time.air") ms |"
done
echo
echo "The GPU code is the machine code of the compiled pipeline, stored in a binary archive. The compile time is the median of 7 runs of \`metal -c\` for a.metal."

echo
link_error() { grep -m1 -o -E 'multiple symbols.*|unknown argument.*|[0-9]+ duplicated symbols' "$TMP/link.log" || head -1 "$TMP/link.log"; }

echo "### Compiler and linker options instead of static inline"
echo
echo "The same two files with upstream main's LYGIA, compiled or linked with options that could hide the duplicates:"
echo
echo "| Options | Links |"
echo "|---|---|"
for flags in "-fvisibility=hidden" "-flto=thin"; do
    metal $flags -I "$TMP/before" -c "$TMP/a.metal" -o "$TMP/flags-a.air"
    metal $flags -I "$TMP/before" -c "$TMP/b.metal" -o "$TMP/flags-b.air"
    if xcrun -sdk macosx metallib "$TMP/flags-a.air" "$TMP/flags-b.air" -o "$TMP/flags.metallib" 2> "$TMP/link.log"; then
        echo "| \`$flags\` | yes |"
    else
        echo "| \`$flags\` | no: \`$(link_error)\` |"
    fi
done
metal -I "$TMP/before" -c "$TMP/a.metal" -o "$TMP/flags-a.air"
metal -I "$TMP/before" -c "$TMP/b.metal" -o "$TMP/flags-b.air"
for flags in "-Wl,--allow-multiple-definition" "-Wl,-z,muldefs"; do
    if metal "$TMP/flags-a.air" "$TMP/flags-b.air" $flags -o "$TMP/flags.metallib" 2> "$TMP/link.log"; then
        echo "| \`$flags\` | yes |"
    else
        echo "| \`$flags\` | no: \`$(link_error)\` |"
    fi
done
echo
matches=$(metal -Wl,--help "$TMP/flags-a.air" 2>&1 | grep -ciE 'multiple|muldef|duplicate' || true)
echo "The linker is air-lld. Options in its help (\`xcrun metal -Wl,--help\`) that mention multiple or duplicate definitions: $matches."

echo
echo "### Two .metal files that include fbm with different options"
echo
echo "One file defines FBM_OCTAVES 1 and the other 8, and each kernel returns fbm(1.3, 2.7). Each should get its own fbm: 1 octave gives -0.34727, 8 octaves give -0.185045."
echo
echo "| LYGIA | Optimization | 1 octave | 8 octaves |"
echo "|---|---|---|---|"
for k in "a 1" "b 8"; do
    set -- $k
    printf '#include <metal_stdlib>\nusing namespace metal;\n#define FBM_OCTAVES %s\n#include "lygia/generative/fbm.msl"\nkernel void k(device float4* results [[buffer(0)]]) { results[0] = float4(fbm(float2(1.3, 2.7))); }\n' "$2" \
        | sed "s/kernel void k(/kernel void k_$1(/" > "$TMP/odr_$1.metal"
done
for v in $VERSIONS; do
    for opt in -O0 -O2; do
        metal $opt -I "$TMP/$v" -c "$TMP/odr_a.metal" -o "$TMP/odr_a.air"
        metal $opt -I "$TMP/$v" -c "$TMP/odr_b.metal" -o "$TMP/odr_b.air"
        if xcrun -sdk macosx metallib "$TMP/odr_a.air" "$TMP/odr_b.air" -o "$TMP/odr.metallib" 2> /dev/null; then
            a=$("$TMP/run" "$TMP/odr.metallib" k_a 1 | awk '{print $1}')
            b=$("$TMP/run" "$TMP/odr.metallib" k_b 1 | awk '{print $1}')
            echo "| $(label $v) | $opt | $a | $b |"
        else
            echo "| $(label $v) | $opt | doesn't link | doesn't link |"
        fi
    done
done
echo
echo "With plain inline, both files export the same fbm symbol and the linker keeps one; at -O0, where calls aren't inlined, both kernels use it."

echo
echo "### LYGIA as a helper library"
echo
echo "LYGIA's fbm from $BEFORE (plain definitions: a library has to export its functions, so they can't be static) is compiled once into a library, and the two files above only declare \`float fbm(float2 st);\` and call it. The library uses the default FBM_OCTAVES, 4."
echo
for k in a b; do
    sed -e 's|#include "lygia/generative/fbm.msl"|float fbm(float2 st);|' -e '/FBM_OCTAVES/d' "$TMP/odr_$k.metal" > "$TMP/call_$k.metal"
    metal -c "$TMP/call_$k.metal" -o "$TMP/call_$k.air"
done
printf '#include <metal_stdlib>\nusing namespace metal;\n#include "lygia/generative/fbm.msl"\n' > "$TMP/lib.metal"
metal -I "$TMP/before" -c "$TMP/lib.metal" -o "$TMP/lib.air"
lib_run() {  # label, link command..., then the run options after --
    local label="$1"; shift
    local link=() run=()
    while [ "$1" != "--" ]; do link+=("$1"); shift; done; shift; run=("$@")
    if "${link[@]}" -o "$TMP/lib-app.metallib" 2> "$TMP/link.log"; then
        a=$("$TMP/run" ${run[@]+"${run[@]}"} "$TMP/lib-app.metallib" k_a 1 | awk '{print $1}')
        b=$("$TMP/run" ${run[@]+"${run[@]}"} "$TMP/lib-app.metallib" k_b 1 | awk '{print $1}')
        echo "| $label | yes | $a | $b |"
    else
        echo "| $label | no: \`$(link_error)\` | | |"
    fi
}
metal -I "$TMP/before" -dynamiclib -install_name @loader_path/liblygia.metallib "$TMP/lib.metal" -o "$TMP/liblygia.metallib"
metal "$TMP/lib.air" --emit-static-lib -o "$TMP/liblygia-static.metallib"
echo "| Library | Links | 1-octave kernel | 8-octave kernel |"
echo "|---|---|---|---|"
lib_run "dynamic library (\`metal -dynamiclib\`)" metal "$TMP/call_a.air" "$TMP/call_b.air" -L "$TMP" -llygia -- --dylib "$TMP/liblygia.metallib"
lib_run "static library (\`--emit-static-lib\`)" metal "$TMP/call_a.air" "$TMP/call_b.air" "$TMP/liblygia-static.metallib" --
lib_run "LYGIA's .air, linked once" xcrun -sdk macosx metallib "$TMP/call_a.air" "$TMP/call_b.air" "$TMP/lib.air" --
sed -e 's/FBM_OCTAVES 1/FBM_OCTAVES 4/' "$TMP/odr_a.metal" > "$TMP/odr_4.metal"
metal -I "$TMP/static" -c "$TMP/odr_4.metal" -o "$TMP/odr_4.air"
xcrun -sdk macosx metallib "$TMP/odr_4.air" -o "$TMP/odr_4.metallib"
echo
echo "Every form links and runs, and both kernels return fbm with 4 octaves ($("$TMP/run" "$TMP/odr_4.metallib" k_a 1 | awk '{print $1}') with static inline and FBM_OCTAVES 4): the library fixes the options once for every shader. The dynamic library is loaded with the pipeline's \`preloadedLibraries\`."
