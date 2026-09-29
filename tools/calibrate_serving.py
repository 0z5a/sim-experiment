"""Real TP2 calibration and held-out serving runs in one resident engine."""

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
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.distributed.parallel_state import get_world_group

from simulator.workload import calibration_workloads, metrics, scheduled_shape, validation_workloads, Workload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    config = EngineArgs(
        model=args.model, dtype="bfloat16", tensor_parallel_size=2,
        enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
        enable_chunked_prefill=False, disable_custom_all_reduce=True,
        distributed_executor_backend="external_launcher", skip_tokenizer_init=True,
        kv_cache_memory_bytes=1024 * 1024 * 1024, max_model_len=512,
        gpu_memory_utilization=0.15,
        max_num_seqs=32, max_num_batched_tokens=4096, seed=0,
        compilation_config={"mode": 0}, attention_backend="TRITON_ATTN",
    )
    scheduler_config = config.create_engine_config()
    engine = LLMEngine.from_engine_args(config)
    scheduler = engine.engine_core.engine_core.scheduler
    if rank == 0:
        assert not scheduler_config.compilation_config.static_forward_context
        with (args.root / "scheduler-config.pkl").open("wb") as stream:
            pickle.dump((scheduler_config, scheduler.kv_cache_config,
                         scheduler.block_size, scheduler.hash_block_size), stream)
    original_schedule = scheduler.schedule
    observed = []

    def record_schedule(throttle_new_prefills: bool = False):
        output = original_schedule(throttle_new_prefills)
        if output.total_num_scheduled_tokens:
            observed.append({"shape": asdict(scheduled_shape(scheduler, output)),
                             "ids": list(output.num_scheduled_tokens),
                             "waiting": len(scheduler.waiting)})
        return output

    scheduler.schedule = record_schedule
    group = get_world_group().cpu_group

    def run(workload: Workload) -> dict:
        torch.distributed.barrier(group=group)
        observed.clear()
        arrivals = workload.arrivals()
        submitted = 0
        stamps: dict[str, list[int]] = {}
        started = time.perf_counter_ns()
        steps = []
        while submitted < workload.requests or engine.has_unfinished_requests():
            tick = time.perf_counter_ns()
            due = submitted
            if rank == 0:
                if not engine.has_unfinished_requests() and submitted < workload.requests:
                    wait = arrivals[submitted] - (tick - started)
                    if wait > 0:
                        time.sleep(wait / 1e9)
                        tick = time.perf_counter_ns()
                while due < workload.requests and arrivals[due] <= time.perf_counter_ns() - started:
                    due += 1
            admission = [due]
            torch.distributed.broadcast_object_list(admission, src=0, group=group)
            due = admission[0]
            for index in range(submitted, due):
                engine.add_request(str(index), {"prompt_token_ids": [100 + index % 8] * workload.prompt},
                                   SamplingParams(temperature=0, ignore_eos=True,
                                                  max_tokens=workload.output, detokenize=False))
            submitted = due
            outputs = engine.step()
            now = time.perf_counter_ns()
            for output in outputs:
                current = len(output.outputs[0].token_ids)
                times = stamps.setdefault(output.request_id, [])
                times.extend([now - started] * (current - len(times)))
            step = dict(observed[-1], duration_ns=now - tick, end_ns=now - started)
            steps.append(step)
        finish = time.perf_counter_ns() - started
        result = {"workload": asdict(workload), "steps": steps,
                  "metrics": metrics(workload, stamps, finish), "token_visible_ns": stamps}
        print(json.dumps({"rank": rank, "name": workload.name, **result["metrics"]}), flush=True)
        return result

    # Warm every calibration shape before recording any calibration sample.
    for case in calibration_workloads()[:12]:
        run(Workload("warm-" + case.name, case.requests, case.prompt, case.output))
    calibration = [run(case) for case in calibration_workloads()]
    (args.root / f"calibration-rank{rank}.json").write_text(json.dumps(calibration))
    ready = time.perf_counter_ns()
    validation = [run(case) for repeat in range(3) for case in validation_workloads()]
    document = {"rank": rank, "model": args.model, "validation": validation,
                "initialization_and_calibration_ns": ready - PROCESS_START,
                "process_wall_ns": time.perf_counter_ns() - PROCESS_START,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.root / f"real-rank{rank}.json").write_text(json.dumps(document))
    engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
