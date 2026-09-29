# Cross-rank scheduler-boundary prototype

This research path runs two independent instances of the installed vLLM scheduler on CPU and replaces model execution with fixed-length token-control outputs and calibrated step costs. **It is not the U0 CUDA/NCCL ABI simulator.** It does not execute the original model runner, validate generated text, or simulate individual CUDA kernels.

## Experiment contract

- Qwen2.5-0.5B, BF16, TP2 on the recorded pair of RTX 5090s; eager synchronous execution, Triton attention, PyNccl all-reduce, no custom all-reduce.
- Fixed prompt IDs, greedy generation, ignore EOS, no detokenization, no prefix caching, no chunked prefill, no speculation, no LoRA, no connectors.
- At most 32 running requests, 4096 scheduled tokens, 512-token context cap, 1 GiB KV budget per rank. The CPU schedulers inherit the actual KV block count and layout.
- Calibration: batches 1/8/16/32, prompt lengths 32/64/128, 96 generated tokens, two repetitions after warming every shape. Calibration and held-out results are separate files.
- Held out: bursts of 128/256/512 requests, plus two 96-request waves separated by 500 ms; P=64/O=64; three repetitions in the same resident real engine.

Initial accuracy target: absolute median error within 10% for cohort duration, throughput, TTFT, and ITL in each held-out workload. The report retains all results, including misses of this target; no held-out sample is used to refit the model.

The real harness uses external ranks with deterministic admission coordinated over the existing CPU process group. This coordination is included in measured rank-step cost. A read-only wrapper around `Scheduler.schedule` records shapes and request membership. No package files are edited.

## Timing and causality

For each rank, fit median costs indexed by prefill/decode phase, batch size, and mean active context length. Decode contexts use 16-token fitting bins. Prediction uses bilinear interpolation and rejects missing phases or features outside the recorded domain. Mixed prefill/decode steps are deliberately unsupported by this first model.

Each rank submits one positive, **all-inclusive** step cost at the same safe scheduler boundary. The completion frontier is the maximum rank completion time. NCCL and host cost are already inside the measured duration; no separate collective duration is added. This does not predict asynchronous overlap, collective algorithms, rank-local compute phases, or a different GPU/rank topology.

Requests enter each scheduler only after their declared arrival. The original schedulers allocate/free KV blocks and choose the next batch. Simulated output is released only at the cross-rank completion frontier. Rank scheduling and completions must agree. Output token values are synthetic; only fixed-length request control is represented. Prefix cache and content-dependent features remain off.

The queue-feedback test changes modeled durations while keeping arrivals fixed, then checks that the original scheduler's waiting queue changes. This is a sensitivity fixture, not a measured optimization claim.

## Cost accounting

Workload speedup divides real cohort wall time by CPU simulation wall time including scheduler construction and execution. It excludes process startup and calibration. A separate subprocess measurement includes Python imports, model fitting, all validation simulations, feedback validation, result serialization, and normal process exit.

The report additionally charges the first campaign the complete real initialization, warmup, and calibration cost. The initial calibration is not free, and performance-model reuse across changed stacks or topologies is not valid without new validation. Download time is excluded from both sides.

## Reproduce in the existing environment

```bash
/home/gongji/0z5a/bin/python tools/launch_calibration.py
/home/gongji/0z5a/bin/python tools/run_cpu_simulation.py --root artifacts/crossrank-v2
/home/gongji/0z5a/bin/python tools/report_crossrank.py --root artifacts/crossrank-v2
```

The launch script records GPU/process snapshots, binds ranks to GPUs 4 and 7 by UUID, and waits for natural process exit without sending termination signals. The CPU wrapper hides all CUDA devices. The scheduler snapshot is a small configuration-only pickle produced locally by the calibration tool; load only snapshots produced by this trusted experiment.

## Failures and limits

The first high-load startup used the environment's default 92% memory-utilization reservation. A newly started workload occupied GPU 7, so rank 1 failed the reservation check and rank 0 remained in device initialization. It was not terminated. Its process and the other GPU workload are recorded as background contention in the subsequent calibration. The retry uses an explicit 15% startup reservation and the same 1 GiB KV budget; this is a run configuration, not an environment change.

A subsequent preparation attempt exposed two harness issues: snapshotting the live compilation context included runtime tensors, and this installed `schedule` method takes a prefill-throttling argument. Both ranks exited naturally. The unused large snapshot was deleted. The current version snapshots a fresh configuration with an empty forward context and preserves the scheduler argument. Failed attempts are excluded from scoring.

Results therefore describe the recorded shared machine and workload domain. They do not establish isolated RTX 5090 latency, arbitrary serving accuracy, TP4/TP8 behavior, or the ABI-level semantic closure required by the original U0 roadmap.
