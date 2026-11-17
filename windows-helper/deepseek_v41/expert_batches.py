# SPDX-License-Identifier: Apache-2.0
"""Plan bounded expert-bank batches without changing token routing."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertBatch:
    bank_ids: np.ndarray
    route_positions: np.ndarray
    slot_ids: np.ndarray


def plan_expert_batches(
    chosen: np.ndarray, *, expert_count: int, slots: int
) -> list[ExpertBatch]:
    """Group routes by the experts that fit in a single weight buffer.

    Positions index the original flattened token/route grid. Slot IDs index
    the sorted bank IDs in the batch, so the existing sorted gather kernels
    can consume them. The caller scatters outputs back before reducing routes.
    """
    if chosen.ndim != 2 or not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer router grid")
    if expert_count <= 0 or not 0 < slots <= expert_count:
        raise ValueError("Invalid expert slot count")
    routes = chosen.reshape(-1)
    if not routes.size:
        return []
    if routes.min() < 0 or routes.max() >= expert_count:
        raise ValueError("Route references an invalid expert")
    positions = np.argsort(routes, kind="stable")
    bank_ids, starts = np.unique(routes[positions], return_index=True)
    ends = np.append(starts[1:], routes.size)
    batches = []
    for start in range(0, bank_ids.size, slots):
        end = min(start + slots, bank_ids.size)
        first_route, last_route = starts[start], ends[end - 1]
        slot_ids = np.repeat(np.arange(end - start, dtype=np.uint32),
                             ends[start:end] - starts[start:end])
        batches.append(ExpertBatch(bank_ids[start:end],
                                   positions[first_route:last_route], slot_ids))
    return batches
