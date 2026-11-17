# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded H2D staging for DeepSeek's packed expert tensors."""

from contextlib import ExitStack
import ctypes as c
from math import prod
import json
import logging
import os
from pathlib import Path
from resource import RUSAGE_THREAD, getrusage
from time import perf_counter_ns

import mlx.core as mx
import numpy as np

from .cuda_staged_copy import copy_staged_tensor


_POINTER = c.c_void_p
_SIZE = c.c_size_t
logger = logging.getLogger(__name__)


class PinnedExpertWeights:
    """Own a pinned host bank and two bounded CUDA staging slots."""

    def __init__(self, host_weights, capacity, *, staging=None, pin_bank=True):
        if capacity <= 0:
            raise ValueError("Expected a positive staging capacity")
        self.capacity = capacity
        self.device_bytes = 0
        self.closed = False
        self.cuda = c.CDLL("libcudart.so.13")
        for name, args in {
            "cudaHostAlloc": [c.POINTER(_POINTER), _SIZE, c.c_uint],
            "cudaFreeHost": [_POINTER],
            "cudaMalloc": [c.POINTER(_POINTER), _SIZE],
            "cudaFree": [_POINTER],
            "cudaStreamCreateWithFlags": [c.POINTER(_POINTER), c.c_uint],
            "cudaStreamSynchronize": [_POINTER],
            "cudaStreamDestroy": [_POINTER],
            "cudaMemcpyAsync": [_POINTER, _POINTER, _SIZE, c.c_int, _POINTER],
        }.items():
            getattr(self.cuda, name).argtypes = args
        self.resources = ExitStack()
        self.tensors = {}
        self.streams = []
        self.buffers = []
        self.events = []
        self.scheduled = {}
        # A memory-mapped bank must stay live while its raw pointers are in use.
        self.mapped = None if pin_bank else host_weights
        self.host_slots = []
        try:
            weights = host_weights.tensors
            total = sum(array.nbytes if pin_bank else 2 * capacity * array[0].nbytes
                        for array in weights.values())
            memory = Path("/proc/meminfo").read_text().splitlines()
            available = next(int(line.split()[1]) * 1024 for line in memory
                             if line.startswith("MemAvailable:"))
            if total + (12 << 30) > available:
                raise MemoryError("Insufficient RAM for pinned DeepSeek expert weights")
            for name, array in weights.items():
                if pin_bank:
                    host = self._allocate("cudaHostAlloc", "cudaFreeHost", array.nbytes, 0)
                    c.memmove(host, array.ctypes.data, array.nbytes)
                else:
                    host = _POINTER(array.ctypes.data)
                dtype = mx.uint32 if array.dtype.itemsize == 4 else mx.uint8
                self.tensors[name] = (host, array.shape[1:], array.dtype.itemsize, dtype)
            if not pin_bank:
                allocation_start = perf_counter_ns()
                self.host_slots = [
                    {name: self._allocate("cudaHostAlloc", "cudaFreeHost",
                                          capacity * array[0].nbytes, 0)
                     for name, array in weights.items()}
                    for _ in range(2)
                ]
                logger.info("Memory-mapped expert weights: pinned staging bytes=%d", total)
                if os.environ.get("HYBRID_PREFILL_V41_TRACE") == "1":
                    logger.info("prefill_pinned_allocate %s", json.dumps({
                        "start_ns": allocation_start, "end_ns": perf_counter_ns(), "bytes": total
                    }, separators=(",", ":")))
            if staging is None:
                for _ in range(2):
                    self.streams.append(self._allocate("cudaStreamCreateWithFlags",
                                                       "cudaStreamDestroy", 1))
                    self.buffers.append({
                        name: self._allocate("cudaMalloc", "cudaFree",
                                             capacity * prod(shape) * itemsize)
                        for name, (_, shape, itemsize, _) in self.tensors.items()
                    })
                self.device_bytes = 2 * capacity * sum(
                    prod(shape) * itemsize for _, shape, itemsize, _ in self.tensors.values()
                )
            else:
                if staging.closed or capacity != staging.capacity:
                    raise ValueError("Incompatible shared CUDA capacity")
                keys = {name.split(".ffn.experts.", 1)[1]: name for name in staging.tensors}
                aliases = {name: keys[name.split(".ffn.experts.", 1)[1]]
                           for name in self.tensors}
                if any(self.tensors[name][1:] != staging.tensors[other][1:]
                       for name, other in aliases.items()):
                    raise ValueError("Incompatible expert tensor layout")
                self.streams = staging.streams
                self.buffers = [{name: slot[other] for name, other in aliases.items()}
                                for slot in staging.buffers]
            if os.environ.get("HYBRID_PREFILL_CUDA_PROFILE") == "1":
                self._init_events()
        except BaseException:
            self.resources.close()
            raise

    def _init_events(self):
        for name, args in {
            "cudaEventCreate": [c.POINTER(_POINTER)],
            "cudaEventDestroy": [_POINTER],
            "cudaEventRecord": [_POINTER, _POINTER],
            "cudaEventElapsedTime": [c.POINTER(c.c_float), _POINTER, _POINTER],
        }.items():
            getattr(self.cuda, name).argtypes = args
        self.events = [
            tuple(self._allocate("cudaEventCreate", "cudaEventDestroy") for _ in range(2))
            for _ in self.streams
        ]

    def _call(self, name, *args):
        result = getattr(self.cuda, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with CUDA status {result}")

    def _allocate(self, allocate, free, *args):
        pointer = _POINTER()
        self._call(allocate, c.byref(pointer), *args)
        self.resources.callback(self._call, free, pointer)
        return pointer

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.device_bytes = 0
        try:
            self.synchronize_slots()
        finally:
            self.resources.close()

    def synchronize_slots(self):
        for stream in self.streams:
            self._call("cudaStreamSynchronize", stream)

    def prepare_host_slot(self, slot, bank_ids):
        """Pack the selected mmap rows; callers must own the host slot."""
        if not self.host_slots or len(bank_ids) > self.capacity:
            raise ValueError("Invalid mmap staging batch")
        runs = np.split(bank_ids, np.flatnonzero(np.diff(bank_ids) != 1) + 1)
        start = perf_counter_ns() if self.events else 0
        usage = getrusage(RUSAGE_THREAD) if self.events else None
        for name, (host, shape, itemsize, _) in self.tensors.items():
            row_bytes = prod(shape) * itemsize
            target = self.host_slots[slot][name].value
            offset = 0
            for run in runs:
                size = len(run) * row_bytes
                c.memmove(target + offset, host.value + int(run[0]) * row_bytes, size)
                offset += size
        return self._record_host_copy(slot, len(bank_ids), start, usage) if self.events else 0

    def _record_host_copy(self, slot, experts, start, usage):
        end = perf_counter_ns()
        current = getrusage(RUSAGE_THREAD)
        logger.info("prefill_host_copy %s", json.dumps({
            "slot": slot, "experts": experts, "start_ns": start, "end_ns": end,
            "minor_faults": current.ru_minflt - usage.ru_minflt,
            "major_faults": current.ru_majflt - usage.ru_majflt,
            "thread_cpu_ns": int(1e9 * (current.ru_utime + current.ru_stime
                                       - usage.ru_utime - usage.ru_stime)),
        }, separators=(",", ":")))
        return end - start

    def enqueue(self, slot, bank_ids, *, prepared_ns=None):
        """Copy contiguous runs of selected expert rows into a packed GPU slot."""
        if len(bank_ids) > self.capacity:
            raise ValueError("Expert batch exceeds CUDA slot")
        runs = (np.split(bank_ids, np.flatnonzero(np.diff(bank_ids) != 1) + 1)
                if prepared_ns is None else ())
        if self.events:
            enqueue_start = perf_counter_ns()
            host_gather_ns = prepared_ns or 0
            self._call("cudaEventRecord", self.events[slot][0], self.streams[slot])
        gather_start_ns = perf_counter_ns() if self.events and self.host_slots and prepared_ns is None else None
        gather_usage = getrusage(RUSAGE_THREAD) if gather_start_ns is not None else None
        for name, (host, shape, itemsize, _) in self.tensors.items():
            row_bytes = prod(shape) * itemsize
            device = self.buffers[slot][name].value
            packed_host = self.host_slots[slot][name].value if self.host_slots else None
            offset = 0 if prepared_ns is None else len(bank_ids) * row_bytes
            for run in runs:
                nbytes = len(run) * row_bytes
                source = host.value + int(run[0]) * row_bytes
                if packed_host is not None:
                    if self.events:
                        gather_start = perf_counter_ns()
                    c.memmove(packed_host + offset, source, nbytes)
                    if self.events:
                        host_gather_ns += perf_counter_ns() - gather_start
                else:
                    self._call("cudaMemcpyAsync", device + offset, source, nbytes, 1,
                               self.streams[slot])
                offset += nbytes
            if packed_host is not None:
                self._call("cudaMemcpyAsync", device, packed_host, offset, 1, self.streams[slot])
        if gather_start_ns is not None:
            self._record_host_copy(slot, len(bank_ids), gather_start_ns, gather_usage)
        if self.events:
            self._call("cudaEventRecord", self.events[slot][1], self.streams[slot])
            transferred = len(bank_ids) * sum(prod(shape) * itemsize
                                             for _, shape, itemsize, _ in self.tensors.values())
            self.scheduled[slot] = (transferred, perf_counter_ns() - enqueue_start, host_gather_ns)

    def receive(self, slot, count):
        if self.events:
            wait_start = perf_counter_ns()
        self._call("cudaStreamSynchronize", self.streams[slot])
        if self.events:
            wait_end = perf_counter_ns()
            elapsed = c.c_float()
            self._call("cudaEventElapsedTime", c.byref(elapsed), *self.events[slot])
            copy_start = perf_counter_ns()
        tensors = {
            name: copy_staged_tensor(self.buffers[slot][name].value, (count, *shape), dtype)
            for name, (_, shape, _, dtype) in self.tensors.items()
        }
        # All MLX outputs own their storage. No reference to a staging slot
        # escapes before every device-side read of that slot is finished.
        mx.eval(*tensors.values())
        if self.events:
            copy_end = perf_counter_ns()
            transferred, enqueue_ns, host_gather_ns = self.scheduled.pop(slot)
            logger.info("prefill_h2d %s", json.dumps({
                "slot": slot, "experts": count, "bytes": transferred,
                "wait_start_ns": wait_start, "wait_end_ns": wait_end,
                "staging_copy_start_ns": copy_start, "staging_copy_end_ns": copy_end,
                "expert_tensor": next(iter(self.tensors)), "host_gather_ns": host_gather_ns,
                "h2d_stream_span_ms": elapsed.value, "cpu_enqueue_ns": enqueue_ns,
                "cpu_stream_wait_ns": wait_end - wait_start,
                "staging_to_mlx_wall_ns": copy_end - copy_start,
            }, separators=(",", ":")))
        return tensors
