# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 (model type ``deepseek_v41``): planning without a GPU library.

Engram tables stay on the host, which sends each chunk's rows.
"""

from __future__ import annotations

import os
from pathlib import Path

from ... import checkpoint
from .config import Config, expert_bits, load_config

CONTRACT = 1
# A later stage may start at any layer: it gets the shared state of the layers
# before it from the stage before.
FIRST_STAGE_MIN = 1

_NOT_LOADED = (".engram.embed.", ".mtp.")
_AFTER_LAYERS = ("language_model.head.", "language_model.norm.")


def resident(tensor: str) -> bool:
    """Whether a helper loads this tensor; the host keeps the head and final norm."""
    return (
        tensor.startswith("language_model.")
        and not any(part in tensor for part in _NOT_LOADED)
        and not tensor.startswith(_AFTER_LAYERS)
    )


def streams_layer(layer: int | None) -> bool:
    """Opt-in contiguous prefix beyond the three resident layers."""
    requested = int(os.getenv("OMLX_CUDA_STREAM_EXPERTS_THROUGH", "0"))
    if not requested and os.getenv("OMLX_CUDA_STREAM_LAYER4") == "1":
        requested = 4
    return layer is not None and 3 <= layer < requested


def _held_bytes(config: Config, name: str, tensor: checkpoint.Tensor) -> int:
    """GPU bytes of a checkpoint tensor as the stage holds it."""
    module, _, kind = name.rpartition(".")
    spec = config.quantized.get(module)
    if spec is None:
        return tensor.nbytes
    if ".ffn.experts." in name:
        # Expert banks stay packed, some with wider codes.
        if kind == "weight":
            return tensor.nbytes * expert_bits(spec) // spec["bits"]
        return tensor.nbytes
    # Other quantized weights are unpacked to bfloat16; scales are dropped.
    return tensor.nbytes * 8 // spec["bits"] * 2 if kind == "weight" else 0


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds on a GPU."""
    config = load_config(model_dir)
    shared, per_layer, staging = 0, {}, {}
    for name, tensor in checkpoint.tensors(model_dir).items():
        if not resident(name):
            continue
        layer = checkpoint.layer_of(name)
        if streams_layer(layer) and ".ffn.experts." in name:
            staging[layer] = staging.get(layer, 0) + tensor.nbytes * 3 * 32 // tensor.shape[0]
            per_layer.setdefault(layer, 0)
            continue
        size = _held_bytes(config, name, tensor)
        if layer is None:
            shared += size
        else:
            per_layer[layer] = per_layer.get(layer, 0) + size
    shared += max(staging.values(), default=0)
    count = max(per_layer) + 1 if per_layer else 0
    return shared, [per_layer.get(i, 0) for i in range(count)]
