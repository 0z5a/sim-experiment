"""CPU scheduler-boundary simulation, explicitly separate from U0 ABI execution."""

import time
PROCESS_START = time.perf_counter_ns()

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle

import torch
from vllm import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from simulator.runtime import RankClock, StepModel
from simulator.workload import metrics, scheduled_shape, validation_workloads, Workload


def run(workload: Workload, config_bytes: bytes, models: list[StepModel], scale: float) -> dict:
    started = time.perf_counter_ns()
    schedulers = []
    for rank in range(2):
        config, kv_config, block_size, hash_block_size = pickle.loads(config_bytes)
        assert config.parallel_config.tensor_parallel_size == 2
        assert not config.scheduler_config.async_scheduling
        assert not config.cache_config.enable_prefix_caching
        assert config.speculative_config is None
        assert not config.scheduler_config.enable_chunked_prefill
        config.cache_config.num_gpu_blocks = kv_config.num_blocks
        config.cache_config.block_size = block_size
        schedulers.append(Scheduler(config, kv_config, StructuredOutputManager(config), block_size, hash_block_size))
    clock = RankClock(2)
    arrivals = workload.arrivals()
    stamps: dict[str, list[int]] = {}
    submitted = 0
    now = 0
    steps = []
    while submitted < workload.requests or schedulers[0].get_num_unfinished_requests():
        if not schedulers[0].get_num_unfinished_requests() and submitted < workload.requests:
            now = max(now, arrivals[submitted])
        while submitted < workload.requests and arrivals[submitted] <= now:
            for scheduler in schedulers:
                scheduler.add_request(Request(
                    str(submitted), [100 + submitted % 8] * workload.prompt,
                    SamplingParams(temperature=0, ignore_eos=True, max_tokens=workload.output, detokenize=False),
                    None, arrival_time=arrivals[submitted] / 1e9,
                ))
            submitted += 1
        scheduled = [scheduler.schedule() for scheduler in schedulers]
        if scheduled[0].num_scheduled_tokens != scheduled[1].num_scheduled_tokens:
            raise ValueError("Cross-rank scheduling mismatch")
        shapes = [scheduled_shape(s, out) for s, out in zip(schedulers, scheduled)]
        if shapes[0] != shapes[1]:
            raise ValueError("Cross-rank control-state mismatch")
        costs = [round(model.predict(shape) * scale) for model, shape in zip(models, shapes)]
        now = clock.complete(now, costs)
        outputs = []
        for scheduler, sched_output in zip(schedulers, scheduled):
            ids = list(sched_output.num_scheduled_tokens)
            # This adapter implements fixed-length token control, not language content.
            tokens = [[100] if scheduler.requests[rid].num_computed_tokens >= scheduler.requests[rid].num_tokens else [] for rid in ids]
            runner_output = ModelRunnerOutput(ids, {rid: i for i, rid in enumerate(ids)}, tokens)
            outputs.append(scheduler.update_from_output(sched_output, runner_output))
        left = [(o.request_id, o.new_token_ids, o.finished) for batch in outputs[0].values() for o in batch.outputs]
        right = [(o.request_id, o.new_token_ids, o.finished) for batch in outputs[1].values() for o in batch.outputs]
        if left != right:
            raise ValueError("Cross-rank completion mismatch")
        for rid, tokens, finished in left:
            stamps.setdefault(rid, []).extend([now] * len(tokens))
        steps.append({"shape": asdict(shapes[0]), "ids": list(scheduled[0].num_scheduled_tokens),
                      "waiting": len(schedulers[0].waiting), "end_ns": now, "rank_costs_ns": costs})
    result = {"workload": asdict(workload), "metrics": metrics(workload, stamps, now),
              "simulator_wall_ns": time.perf_counter_ns() - started, "steps": steps,
              "token_visible_ns": stamps, "duration_scale": scale}
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU simulation unexpectedly initialized CUDA")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    config_bytes = (args.root / "scheduler-config.pkl").read_bytes()
    models = []
    for rank in range(2):
        calibration = json.loads((args.root / f"calibration-rank{rank}.json").read_text())
        models.append(StepModel([step for case in calibration for step in case["steps"]]))
    setup_ns = time.perf_counter_ns() - PROCESS_START
    results = [run(case, config_bytes, models, 1.0) for repeat in range(3) for case in validation_workloads()]
    feedback = run(Workload("two-waves-192", 192, 64, 64, 500_000_000), config_bytes, models, 0.2)
    document = {"mode": "scheduler_boundary_adapter", "abi_simulation": False,
                "cuda_initialized": torch.cuda.is_initialized(), "results": results,
                "feedback_faster_device": feedback, "setup_ns": setup_ns,
                "process_wall_ns": time.perf_counter_ns() - PROCESS_START,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest()}
    (args.root / "simulated.json").write_text(json.dumps(document))
    for result in results:
        print(json.dumps({"name": result["workload"]["name"], "wall_ms": result["simulator_wall_ns"] / 1e6, **result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
