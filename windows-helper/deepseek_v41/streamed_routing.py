# SPDX-License-Identifier: Apache-2.0
"""Sequential reference path for a bounded host-backed expert bank."""

import mlx.core as mx
import numpy as np

from .expert_batches import plan_expert_batches
from .layers import ExpertBank, _swiglu


def run_routed(flat, xq, chosen, weights, host_weights, names, config, slots):
    """Return a tensor with one result per original token/route position."""
    grid = np.asarray(chosen)
    batches = plan_expert_batches(grid, expert_count=config.n_routed_experts, slots=slots)
    out = mx.zeros((grid.size, flat.shape[-1]), dtype=flat.dtype)
    route_weights = weights.reshape(-1)
    k = grid.shape[1]
    for index, batch in enumerate(batches):
        if not hasattr(host_weights, "enqueue"):
            tensors = {name: mx.array(value)
                       for name, value in host_weights.subset(batch.bank_ids).items()}
        else:
            slot = index % 2
            if index == 0:
                host_weights.enqueue(slot, batch.bank_ids)
            if index + 1 < len(batches):
                host_weights.enqueue(1 - slot, batches[index + 1].bank_ids)
            tensors = host_weights.receive(slot, len(batch.bank_ids))
        w1, w3, w2 = (ExpertBank(tensors, name, config) for name in names)
        positions = mx.array(batch.route_positions.astype(np.int32))
        tokens = positions // k
        experts = mx.array(batch.slot_ids)
        gate = w1((xq if w1.quantize_input else flat)[tokens][:, None], experts)
        up = w3((xq if w3.quantize_input else flat)[tokens][:, None], experts)
        act = _swiglu(gate, up, route_weights[positions], config.swiglu_limit, w2.quantize_input)
        routed = w2(act, experts).reshape(len(batch.route_positions), -1)
        out[positions] = routed
        # Finish consuming this batch's bank before loading the next.
        mx.eval(out)
        del tensors, w1, w3, w2, routed, act, gate, up
    return out
