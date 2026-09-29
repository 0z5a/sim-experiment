"""Stock, in-process vLLM baseline; all workers finish without process signals."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput


def run_batch(engine: LLMEngine, batch: int, prompt: int, output: int, tag: str) -> dict:
    params = SamplingParams(temperature=0, max_tokens=output, ignore_eos=True, detokenize=False)
    started = time.perf_counter_ns()
    for i in range(batch):
        engine.add_request(f"{tag}-{i}", {"prompt_token_ids": [100 + i] * prompt}, params)
    tokens: dict[str, list[int]] = {}
    timestamps: dict[str, list[int]] = {}
    steps = 0
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        now = time.perf_counter_ns()
        for item in step_outputs:
            assert isinstance(item, RequestOutput)
            current = list(item.outputs[0].token_ids)
            previous = tokens.get(item.request_id, [])
            assert len(current) >= len(previous)
            timestamps.setdefault(item.request_id, []).extend([now - started] * (len(current) - len(previous)))
            tokens[item.request_id] = current
        steps += 1
    duration = time.perf_counter_ns() - started
    assert len(tokens) == batch and all(len(t) == output for t in tokens.values()), tokens
    ttft = [t[0] / 1e6 for t in timestamps.values()]
    itl = [(b - a) / 1e6 for ts in timestamps.values() for a, b in zip(ts, ts[1:])]
    return {
        "batch": batch, "prompt_tokens": prompt, "output_tokens": output,
        "wall_ns": duration, "tokens_per_second": batch * output * 1e9 / duration,
        "ttft_median_ms": statistics.median(ttft),
        "itl_median_ms": statistics.median(itl) if itl else None,
        "steps": steps, "tokens": tokens, "token_visible_ns": timestamps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] in (
        "GPU-739fcff8-0a90-fd63-8d91-65847ad665d2",
        "GPU-c1e2d922-6fb4-92a2-c9e3-9f4ef9d9c079",
        "GPU-739fcff8-0a90-fd63-8d91-65847ad665d2,GPU-c1e2d922-6fb4-92a2-c9e3-9f4ef9d9c079",
    )
    before = subprocess.check_output([
        "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
        "--query-gpu=uuid,name,driver_version,memory.used,temperature.gpu,clocks.sm,power.draw",
        "--format=csv", ], text=True)
    config = EngineArgs(
        model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
        enforce_eager=True, async_scheduling=False,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        disable_custom_all_reduce=True,
        distributed_executor_backend="uni" if args.tp == 1 else "external_launcher",
        kv_cache_memory_bytes=512 * 1024 * 1024, max_model_len=1024,
        max_num_seqs=8, max_num_batched_tokens=1024,
        skip_tokenizer_init=True, seed=0, compilation_config={"mode": 0},
        attention_backend="TRITON_ATTN",
    )
    initialized = time.perf_counter_ns()
    engine = LLMEngine.from_engine_args(config)
    startup_ns = time.perf_counter_ns() - initialized
    cases = [(1, 32, 4), (4, 128, 16), (8, 64, 32)]
    for index, case in enumerate(cases):
        run_batch(engine, *case, f"warm-{index}")
    runs = []
    for repeat in range(args.runs):
        for index, case in enumerate(cases):
            result = run_batch(engine, *case, f"r{repeat}-c{index}")
            runs.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("tokens", "token_visible_ns")}), flush=True)
    if args.profile:
        # Separate diagnosis from every reported timing sample.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            engine.add_request("pause-probe", {"prompt_token_ids": [101] * 32},
                               SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, detokenize=False))
            engine.step()
            torch.cuda.synchronize()
            assert engine.get_num_unfinished_requests() == 1
            with torch.profiler.record_function("harness_paused_with_pending_request"):
                time.sleep(0.25)
            assert engine.get_num_unfinished_requests() == 1
            while engine.has_unfinished_requests():
                engine.step()
            run_batch(engine, 1, 32, 4, "arrived-after-pause")
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.out.with_suffix(".trace.json")))
    sources = {}
    import vllm
    vllm_root = Path(vllm.__file__).parent
    for relative in ("v1/engine/llm_engine.py", "v1/engine/core_client.py", "v1/worker/gpu_model_runner.py", "v1/worker/gpu_worker.py", "v1/sample/sampler.py", "v1/worker/gpu/model_runner.py", "v1/worker/gpu/states.py", "v1/worker/gpu/async_utils.py"):
        source = vllm_root / relative
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "mode": "real_gpu", "compatibility": "U0", "pid": os.getpid(),
        "python": sys.executable, "rank": int(os.environ.get("RANK", "0")),
        "tp": args.tp, "host_load": os.getloadavg(), "device_before": before,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "vllm", "triton", "transformers")},
        "device": torch.cuda.get_device_name(0), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "engine_core_type": type(engine.engine_core).__name__, "sources": sources,
        "resolved_config": str(engine.vllm_config), "startup_ns": startup_ns,
        "libraries": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if any(n in line for n in ("libcuda", "libnccl", "libnvidia", "libcupti"))}),
        "runs": runs,
    }
    args.out.write_text(json.dumps(result, indent=2))
    engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
