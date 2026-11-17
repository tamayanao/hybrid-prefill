#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# != 1 || ! $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo 'Usage: bash install.sh WINDOWS_WSL_GATEWAY_IPV4' >&2
    exit 2
fi
name=hybrid-prefill-helper
image="$name:0.0.1"
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists. Its state has been preserved." >&2
    exit 1
fi
docker build --platform linux/amd64 -t "$image" .
docker create --gpus all --network host --name "$name" --label io.hybridprefill.package=1 \
    -v "$name:/data" -e "HYBRID_PREFILL_PAIR_HOST=$1" "$image"
