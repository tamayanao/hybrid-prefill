# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded expert access for the DeepSeek V4.1 oQ4e helper."""

import json
from pathlib import Path
import struct

import numpy as np


_DTYPES = {"U32": "<u4", "U8": "u1"}


class HostExpertWeights:
    """Keep the packed expert tensors in read-only host mappings."""

    def __init__(self, model_dir: str | Path, names: list[str], expert_count: int):
        root = Path(model_dir)
        weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.tensors = {}
        headers = {}
        for name in names:
            filename = weights[name]
            if filename not in headers:
                with (root / filename).open("rb") as shard:
                    size, = struct.unpack("<Q", shard.read(8))
                    header = json.loads(shard.read(size))
                headers[filename] = (8 + size, header)
            base, header = headers[filename]
            spec = header[name]
            start, end = spec["data_offsets"]
            shape = tuple(spec["shape"])
            dtype = np.dtype(_DTYPES[spec["dtype"]])
            if not shape or shape[0] != expert_count or dtype.itemsize * np.prod(shape) != end - start:
                raise ValueError(f"Invalid packed expert tensor: {name}")
            self.tensors[name] = np.memmap(
                root / filename, dtype=dtype, mode="r", offset=base + start, shape=shape
            )

    def subset(self, bank_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Materialize only the required packed weights and scales."""
        return {name: tensor[bank_ids] for name, tensor in self.tensors.items()}
