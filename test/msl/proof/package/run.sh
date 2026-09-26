#!/bin/bash
# A Swift package with LYGIA's .msl files and two .metal files that include
# generative/fbm.msl, built with xcodebuild, with upstream main's LYGIA and
# with this branch (static inline). Also builds for iOS, the simulator and visionOS.
# Output: RESULTS.txt. usage: test/msl/proof/package/run.sh
set -uo pipefail
E="$(cd "$(dirname "$0")" && pwd)"
LYGIA="$(cd "$E/../../../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
T="$TMP/LygiaKit"; S="$T/Sources/LygiaKit"
mkdir -p "$S/Shaders" "$S/lygia"
cp "$E/Package.swift" "$T/"; cp "$E/LygiaKit.swift" "$S/"
cp "$E/Effects.metal" "$S/Shaders/"; sed 's/lygiaFbm/lygiaFbm2/' "$E/Effects.metal" > "$S/Shaders/Effects2.metal"
n=0
build() {  # a clean build in its own folder; prints the result and the first error
    n=$((n + 1))
    (cd "$T" && xcodebuild -scheme LygiaKit -destination "$1" -derivedDataPath "$T/dd$n" build 2>&1 \
        | grep -E 'error:|\*\* BUILD' | sort -u | head -2 | tr '\n' ' ')
    echo
}
for ref in origin/main HEAD; do
    rm -rf "$S/lygia" && mkdir "$S/lygia"
    git -C "$LYGIA" archive "$ref" -- ':(glob)**/*.msl' | tar -x -C "$S/lygia"
    echo "$ref, macOS: $(build 'generic/platform=macOS')"
done
for d in 'generic/platform=iOS' 'generic/platform=iOS Simulator' 'generic/platform=visionOS'; do
    echo "HEAD, $d: $(build "$d")"
done
xcrun metal-nm "$(find "$T/dd1" -path '*macosx*' -o -name default.metallib | grep -m1 default.metallib)" 2>/dev/null | grep -o 'lygiaFbm2\?' | sort -u | sed 's/^/in the bundle metallib: /'
