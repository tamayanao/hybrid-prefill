# SPDX-License-Identifier: Apache-2.0
"""Routed execution for bounded host-backed expert banks."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import os
from time import perf_counter_ns

import mlx.core as mx
import numpy as np

from . import kernels
from .expert_batches import plan_expert_batches
from .layers import ExpertBank, _swiglu


logger = logging.getLogger(__name__)


@contextmanager
def _load_batches(host, batches):
    """Keep host-copy workers within the lifetime of the CUDA staging slots."""
    if not hasattr(host, "enqueue"):
        yield lambda _, batch: {name: mx.array(value)
                                for name, value in host.subset(batch.bank_ids).items()}
        return

    if not (getattr(host, "host_slots", None)
            and os.environ.get("HYBRID_PREFILL_MMAP_HOST_PREFETCH") == "1"):
        def load(index, batch):
            slot = index % 2
            if index == 0:
                host.enqueue(slot, batch.bank_ids)
            if index + 1 < len(batches):
                host.enqueue(1 - slot, batches[index + 1].bank_ids)
            return host.receive(slot, len(batch.bank_ids))

        yield load
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None

        def load_prefetched(index, batch):
            nonlocal pending
            slot = index % 2
            prepared_ns = pending.result() if pending is not None else None
            host.enqueue(slot, batch.bank_ids, prepared_ns=prepared_ns)
            # The opposite slot is either unused or its prior receive has
            # synchronized H2D and finished copying to MLX-owned arrays.
            pending = (executor.submit(host.prepare_host_slot, 1 - slot,
                                       batches[index + 1].bank_ids)
                       if index + 1 < len(batches) else None)
            return host.receive(slot, len(batch.bank_ids))

        try:
            yield load_prefetched
        except BaseException:
            # A failed copy submission can leave work in flight.
            host.synchronize_slots()
            raise


def run_routed(flat, xq, chosen, weights, host_weights, names, config, slots):
    """Return a tensor with one result per original token/route position."""
    profile = os.environ.get("HYBRID_PREFILL_CUDA_PROFILE") == "1"
    start = perf_counter_ns() if profile else 0
    grid = np.asarray(chosen)
    routing_end = perf_counter_ns() if profile else 0
    batches = plan_expert_batches(grid, expert_count=config.n_routed_experts, slots=slots)
    plan_end = perf_counter_ns() if profile else 0
    batch_records = []
    out = mx.zeros((grid.size, flat.shape[-1]), dtype=flat.dtype)
    route_weights = weights.reshape(-1)
    k = grid.shape[1]
    use_tile = os.environ.get("HYBRID_PREFILL_EXPERT_TILE") == "1"
    codes = None
    with _load_batches(host_weights, batches) as load:
        for index, batch in enumerate(batches):
            load_start = perf_counter_ns() if profile else 0
            tensors = load(index, batch)
            load_end = perf_counter_ns() if profile else 0
            w1, w3, w2 = (ExpertBank(tensors, name, config) for name in names)
            positions = mx.array(batch.route_positions.astype(np.int32))
            tokens = positions // k
            experts = mx.array(batch.slot_ids)
            if use_tile:
                if codes is None:
                    if not all(bank.quantize_input for bank in (w1, w3, w2)) or not kernels.mx_experts_fit((w1, w3, w2)):
                        raise RuntimeError("Experimental MX tile requires compatible MXFP4 banks")
                    codes = kernels.fp8_codes(flat)
                activation = kernels.expert_gate_up_mx(
                    *codes, tokens.astype(mx.uint32), experts, w1, w3,
                    route_weights[positions], config.swiglu_limit,
                )
                routed = kernels.expert_down_mx(*activation, experts, w2)
            else:
                gate = w1((xq if w1.quantize_input else flat)[tokens][:, None], experts)
                up = w3((xq if w3.quantize_input else flat)[tokens][:, None], experts)
                act = _swiglu(gate, up, route_weights[positions], config.swiglu_limit, w2.quantize_input)
                routed = w2(act, experts)
            routed = routed.reshape(len(batch.route_positions), -1)
            out[positions] = routed
            # Finish consuming this batch's bank before loading the next.
            compute_eval_start = perf_counter_ns() if profile else 0
            mx.eval(out)
            if profile:
                batch_records.append({
                    "experts": len(batch.bank_ids), "routes": len(batch.route_positions),
                    "load_wall_ns": load_end - load_start,
                    "compute_graph_ns": compute_eval_start - load_end,
                    "eval_wall_ns": perf_counter_ns() - compute_eval_start,
                })
            del tensors, w1, w3, w2, routed
            if use_tile:
                del activation
            else:
                del act, gate, up
    if profile:
        logger.info("prefill_routed %s", json.dumps({
            "expert_bank": names[0],
            "start_ns": start, "end_ns": perf_counter_ns(),
            "routing_to_host_wall_ns": routing_end - start,
            "batch_planning_wall_ns": plan_end - routing_end,
            "batches": batch_records,
            "routed_wall_ns": perf_counter_ns() - start,
        }, separators=(",", ":")))
    return out
