#!/bin/bash
# A Swift package with LYGIA's .msl files and two .metal files that include
# generative/fbm.msl, built with xcodebuild, with upstream main's LYGIA and
# with this branch (static inline). Also builds for iOS, the simulator and visionOS.
# Used by proof.sh. usage: test/msl/proof/package/run.sh [before ref (default origin/main)]
set -uo pipefail
E="$(cd "$(dirname "$0")" && pwd)"
LYGIA="$(cd "$E/../../../.." && pwd)"
BEFORE="${1:-origin/main}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
T="$TMP/LygiaKit"; S="$T/Sources/LygiaKit"
mkdir -p "$S/Shaders" "$S/lygia"
cp "$E/Package.swift" "$T/"; cp "$E/LygiaKit.swift" "$S/"
cp "$E/Effects.metal" "$S/Shaders/"; sed 's/lygiaFbm/lygiaFbm2/' "$E/Effects.metal" > "$S/Shaders/Effects2.metal"
build() {  # a clean build in its own folder ($2); prints the result and the first error
    (cd "$T" && xcodebuild -scheme LygiaKit -destination "$1" -derivedDataPath "$T/$2" build 2>&1 \
        | grep -E 'error:|\*\* BUILD' | sort -u | head -2 | tr '\n' ' ')
    echo
}
for ref in "$BEFORE" HEAD; do
    rm -rf "$S/lygia" && mkdir "$S/lygia"
    git -C "$LYGIA" archive "$ref" -- ':(glob)**/*.msl' | tar -x -C "$S/lygia"
    echo "$ref, macOS: $(build 'generic/platform=macOS' "dd-$ref")"
done
for d in iOS 'iOS Simulator' visionOS; do
    echo "HEAD, $d: $(build "generic/platform=$d" "dd-$d")"
done
lib=$(find "$T/dd-HEAD" -name default.metallib -path '*LygiaKit*' | head -1)
echo "functions in the package's default.metallib: $(xcrun metal-nm "$lib" 2>/dev/null | grep -o 'lygiaFbm2\?$' | sort -u | tr '\n' ' ')"
