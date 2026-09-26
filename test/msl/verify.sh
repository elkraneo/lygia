#!/usr/bin/env bash
# Runs the Metal verification cases (test/msl/verify/cases/*.metal) on the GPU and
# checks them against the expected values from the WGSL tests. Requires macOS.
#
# usage: test/msl/verify.sh [case.metal ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$(mktemp -d)/verify"
trap 'rm -rf "$(dirname "$BIN")"' EXIT
swiftc -O "$ROOT/test/msl/verify/verify.swift" -o "$BIN"
"$BIN" "$ROOT" "$@"
