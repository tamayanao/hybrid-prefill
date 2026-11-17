"""Choose the largest validated prefix that fits the helper's RAM."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct


GIB = 1 << 30
RESERVE = 12 * GIB
RESIDENT_LAYERS = 3
MAX_VALIDATED_LAYERS = 11
HELPER_MEMORY = 68 * GIB
_EXPERT = re.compile(r"^language_model\.layers\.(\d+)\.ffn\.experts\.(w1|w3|w2)\.(weight|scales)$")
_EXPECTED = {(part, field) for part in ("w1","w3","w2") for field in ("weight","scales")}


@dataclass(frozen=True)
class Plan:
    layers: int
    pinned_bytes: int
    effective_ram_bytes: int


def read_meminfo(source):
    fields = {}
    for line in source.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in ("MemTotal", "MemAvailable"):
            size, unit = value.split()
            if unit != "kB":
                raise ValueError("Unexpected WSL memory unit")
            fields[key] = int(size) * 1024
    if not 0 < fields.get("MemAvailable", 0) <= fields.get("MemTotal", 0):
        raise ValueError("WSL memory counters are unavailable or invalid")
    return fields


def expert_sizes(checkpoint, maximum):
    root = Path(checkpoint)
    index = json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    headers, by_layer = {}, {}
    for tensor, shard in index.items():
        match = _EXPERT.fullmatch(tensor)
        if match is None:
            continue
        layer = int(match[1])
        if not RESIDENT_LAYERS <= layer < maximum:
            continue
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("Invalid checkpoint shard path")
        if shard not in headers:
            with (root/shard).open("rb") as stream:
                size, = struct.unpack("<Q",stream.read(8))
                if not 0 < size <= 64 * (1 << 20):
                    raise ValueError("Invalid safetensors header size")
                headers[shard] = json.loads(stream.read(size))
        dtype = "U32" if match[3] == "weight" else "U8"
        spec = headers[shard][tensor]
        start, end = spec["data_offsets"]
        if spec["dtype"] != dtype or not spec["shape"] or start < 0 or end <= start:
            raise ValueError(f"Unsupported packed expert tensor in layer {layer + 1}")
        by_layer.setdefault(layer, {})[match[2], match[3]] = end - start
    if any(set(by_layer.get(layer, {})) != _EXPECTED
           for layer in range(RESIDENT_LAYERS, maximum)):
        raise ValueError("The checkpoint lacks one or more expected expert banks")
    return {layer: sum(entries.values()) for layer, entries in by_layer.items()}


def choose_layers(*, physical_bytes, wsl_total_bytes, wsl_available_bytes,
                  layer_bytes, maximum=MAX_VALIDATED_LAYERS):
    if not RESIDENT_LAYERS <= maximum <= MAX_VALIDATED_LAYERS:
        raise ValueError("max_layers must be between 3 and 9")
    if min(physical_bytes, wsl_total_bytes, wsl_available_bytes) <= 0:
        raise ValueError("Memory capacity must be positive")
    effective = min(physical_bytes, wsl_total_bytes, wsl_available_bytes)
    room = max(0, effective - RESERVE)
    layers, used = RESIDENT_LAYERS, 0
    for layer in range(RESIDENT_LAYERS, maximum):
        size = layer_bytes[layer]
        if size <= 0:
            raise ValueError("Invalid expert weight size")
        if used + size > room:
            break
        used += size
        layers += 1
    return Plan(layers, used, effective)
