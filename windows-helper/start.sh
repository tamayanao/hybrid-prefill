#!/bin/sh
set -eu
pair_host=${HYBRID_PREFILL_PAIR_HOST:-}
if [ -n "$pair_host" ] && ! python -c 'from omlx_cuda_helper.state import State; raise SystemExit(not State("/data").hosts())'; then
    set -- --pair "$pair_host:18880" "$@"
fi
exec omlx-cuda-helper --state-dir /data serve --no-mdns --name 'Hybrid Prefill' "$@"
