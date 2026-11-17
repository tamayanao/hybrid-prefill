#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root="${1:-dist}/Hybrid Prefill.app/Contents"
mkdir -p "$root/MacOS" "$root/Resources"
cache=$(mktemp -d)
trap 'rm -rf "$cache"' EXIT
xcrun swiftc -module-cache-path "$cache" -target arm64-apple-macos14.0 -O -parse-as-library app/Workflow.swift \
    -o "$root/MacOS/HybridPrefill"
cp app/Info.plist "$root/"
cp launcher/*.py launcher/launcher.example.json README.md README.en.md "$root/Resources/"
codesign --force --sign - "$root/.."
