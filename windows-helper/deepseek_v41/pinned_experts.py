# SPDX-License-Identifier: Apache-2.0
"""Experimental bounded H2D staging for DeepSeek's packed expert tensors."""

from contextlib import ExitStack
import ctypes as c
from math import prod
from pathlib import Path

import mlx.core as mx
import numpy as np

from .cuda_staged_copy import copy_staged_tensor


_POINTER = c.c_void_p
_SIZE = c.c_size_t


class PinnedExpertWeights:
    """Own a pinned host bank and two bounded CUDA staging slots."""

    def __init__(self, host_weights, capacity, *, staging=None):
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
        try:
            total = sum(array.nbytes for array in host_weights.tensors.values())
            memory = Path("/proc/meminfo").read_text().splitlines()
            available = next(int(line.split()[1]) * 1024 for line in memory
                             if line.startswith("MemAvailable:"))
            if total + (12 << 30) > available:
                raise MemoryError("Insufficient RAM for pinned DeepSeek expert weights")
            for name, array in host_weights.tensors.items():
                host = self._allocate("cudaHostAlloc", "cudaFreeHost", array.nbytes, 0)
                c.memmove(host, array.ctypes.data, array.nbytes)
                dtype = mx.uint32 if array.dtype.itemsize == 4 else mx.uint8
                self.tensors[name] = (host, array.shape[1:], array.dtype.itemsize, dtype)
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
        except BaseException:
            self.resources.close()
            raise

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
            for stream in self.streams:
                self._call("cudaStreamSynchronize", stream)
        finally:
            self.resources.close()

    def enqueue(self, slot, bank_ids):
        """Copy contiguous runs of selected expert rows into a packed GPU slot."""
        if len(bank_ids) > self.capacity:
            raise ValueError("Expert batch exceeds CUDA slot")
        runs = np.split(bank_ids, np.flatnonzero(np.diff(bank_ids) != 1) + 1)
        for name, (host, shape, itemsize, _) in self.tensors.items():
            row_bytes = prod(shape) * itemsize
            device = self.buffers[slot][name].value
            offset = 0
            for run in runs:
                nbytes = len(run) * row_bytes
                self._call("cudaMemcpyAsync", device + offset,
                           host.value + int(run[0]) * row_bytes, nbytes, 1,
                           self.streams[slot])
                offset += nbytes

    def receive(self, slot, count):
        self._call("cudaStreamSynchronize", self.streams[slot])
        tensors = {
            name: copy_staged_tensor(self.buffers[slot][name].value, (count, *shape), dtype)
            for name, (_, shape, _, dtype) in self.tensors.items()
        }
        # All MLX outputs own their storage. No reference to a staging slot
        # escapes before every device-side read of that slot is finished.
        mx.eval(*tensors.values())
        return tensors
