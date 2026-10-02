#!/usr/bin/env bash
# Copies the repository's .msl files into the Swift package's resource folder,
# swift/Sources/Lygia/Resources/lygia. SwiftPM copies a resource folder as-is
# and can't filter it, so the Metal files are kept there as a plain copy.
# Run after changing any .msl file; `swift test` fails when the copy is stale.
#
# usage: swift/sync.sh [--check]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/swift/Sources/Lygia/Resources/lygia"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/lygia"
(cd "$ROOT" && git ls-files '*.msl' | tar -c -T - -f -) | tar -x -C "$TMP/lygia"
git -C "$ROOT" describe --tags --always > "$TMP/lygia/VERSION"

if [ "${1:-}" = "--check" ]; then
    if diff -rq "$TMP/lygia" "$DEST" > "$TMP/diff" 2>&1; then
        echo "swift package resources are up to date"
    else
        echo "swift package resources are stale; run swift/sync.sh:"
        head -5 "$TMP/diff"
        exit 1
    fi
else
    rm -rf "$DEST"
    mkdir -p "$(dirname "$DEST")"
    mv "$TMP/lygia" "$DEST"
    echo "copied $(find "$DEST" -name '*.msl' | wc -l | tr -d ' ') .msl files to swift/Sources/Lygia/Resources/lygia"
fi
