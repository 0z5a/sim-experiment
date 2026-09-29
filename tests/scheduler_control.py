"""Integration fixture: original vLLM scheduler, two ranks, no CUDA computation."""

import argparse
from pathlib import Path

from simulator.runtime import StepModel
from simulator.workload import Workload
from tools.simulate_serving import run


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
samples = [{"shape": {"phase": phase, "batch": batch, "context": context}, "duration_ns": 1000}
           for phase in ("prefill", "decode") for batch in (1, 8, 16, 32) for context in (32, 64, 128, 256)]
models = [StepModel(samples), StepModel(samples)]
result = run(Workload("fixture", 40, 64, 4), args.config.read_bytes(), models, 1)
assert result["metrics"]["completed_requests"] == 40
assert result["metrics"]["generated_tokens"] == 160
assert result["metrics"]["finish_ns"] == 8000
assert result["token_visible_ns"]["0"] == [1000, 2000, 3000, 4000]
assert result["token_visible_ns"]["39"] == [5000, 6000, 7000, 8000]
print("PASS: two original scheduler instances, queueing, release/reuse, fixed token counts, no initialized CUDA")
