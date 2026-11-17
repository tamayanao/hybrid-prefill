#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
version=0.0.2
name="Hybrid-Prefill-v$version"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
package="$scratch/$name"
mkdir -p "$package"/{launcher,docs} dist
bash build-mac.sh "$package"
cp README.md README.en.md LICENSE omlx.patch build-mac.sh "$package/"
cp -R app windows-helper benchmark "$package/"
cp launcher/*.py launcher/windows-memory.ps1 launcher/launcher.example.json "$package/launcher/"
cp "docs/hybrid-prefill-v$version-ja.png" "docs/hybrid-prefill-v$version-en.png" "docs/release-v$version.md" "$package/docs/"
codesign --verify --strict "$package/Hybrid Prefill.app"
ditto -c -k --sequesterRsrc --keepParent "$package" "$scratch/$name.zip"
mv "$scratch/$name.zip" "dist/$name.zip"
(cd dist && shasum -a 256 "$name.zip" > "$name.zip.sha256")
