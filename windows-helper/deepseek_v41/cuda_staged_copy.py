# SPDX-License-Identifier: Apache-2.0
"""Copy an already-staged CUDA buffer to an MLX-owned tensor."""

from functools import cache
from math import prod

import mlx.core as mx


@cache
def _copy_kernel():
    return mx.fast.cuda_kernel(
        name="deepseek_v41_copy_staged_weights",
        input_names=["address", "count"],
        output_names=["weights"],
        source="""
            auto index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < count[0]) {
                const auto* source = reinterpret_cast<const T*>(address[0]);
                weights[index] = source[index];
            }
        """,
    )


def copy_staged_tensor(device_pointer, shape, dtype):
    """The caller must synchronize H2D and own the source until this evaluates."""
    count = prod(shape)
    if dtype not in (mx.uint8, mx.uint32) or count <= 0 or device_pointer <= 0:
        raise ValueError("Unsupported packed CUDA tensor")
    width = 256
    return _copy_kernel()(
        inputs=[mx.array([device_pointer], dtype=mx.uint64),
                mx.array([count], dtype=mx.uint64)],
        template=[("T", dtype)],
        grid=((count + width - 1) // width * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[tuple(shape)],
        output_dtypes=[dtype],
    )[0]
