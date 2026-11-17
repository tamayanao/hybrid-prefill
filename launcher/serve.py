"""Run the pinned oMLX release with the selected helper-layer plan."""

import os

from omlx.cli import main
from omlx.prefill_helper import manager
from omlx.prefill_helper.adapters.deepseek_v41 import DeepseekV41Adapter
from omlx.scheduler import Scheduler

from ram_plan import RESIDENT_LAYERS, MAX_VALIDATED_LAYERS, expert_sizes


def configure():
    maximum = int(os.environ["HYBRID_PREFILL_LAYERS"])
    cap = int(os.environ["HYBRID_PREFILL_CHUNK_TOKENS"])
    if not RESIDENT_LAYERS <= maximum <= MAX_VALIDATED_LAYERS or cap not in (2048, 4096, 8192):
        raise ValueError("Invalid helper plan")
    if maximum > RESIDENT_LAYERS:
        manager._RESERVE_BYTES = 1 << 30
    original_plan = DeepseekV41Adapter.memory_plan
    original_step = Scheduler._base_prefill_step_size

    def memory_plan(self, model_path):
        shared, layers = original_plan(self, model_path)
        sizes = expert_sizes(model_path, maximum)
        for layer, size in sizes.items():
            layers[layer] -= size
            if layers[layer] < 0:
                raise ValueError("Expert footprint exceeds the resident layer plan")
        # Two staging buffers and one MLX-owned batch, 32 of 384 experts.
        return shared + max(sizes.values(), default=0) // 4, layers

    def step(self, processed_tokens, remaining_tokens):
        return min(cap, original_step(self, processed_tokens, remaining_tokens))

    DeepseekV41Adapter.memory_plan = memory_plan
    Scheduler._base_prefill_step_size = step


if __name__ == "__main__":
    configure()
    raise SystemExit(main())
