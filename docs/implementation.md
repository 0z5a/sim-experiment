# Execution-driven simulator: initial checkpoint

## Scope

The target is an unmodified installed vLLM/PyTorch stack with CUDA/NCCL ABI interposition, device/control semantics, and calibrated causal time. This document covers the M0 real-GPU baseline and feasibility probes. A separate [CPU scheduler-boundary prototype](crossrank.md) implements calibrated cross-rank timing and serving predictions; it does not satisfy the unmodified-engine ABI acceptance gates below.

The model workload uses explicit prompt IDs, greedy decoding, fixed output lengths, no detokenization, eager execution, synchronous scheduling, no prefix cache, and a fixed 512 MiB KV budget. Timing includes request submission, the original scheduler, model execution, and output processing. It excludes model loading, warmup, profiling, and HTTP transport.

## Implemented probes

- `tools/baseline.py`: a real-weight, TP1, in-process `LLMEngine` generation matrix. It checks every request's output length and saves actual token IDs and host-visible token timestamps. Independent invocations provide fresh engine instances.
- `tools/cuda_loader_audit.c`: glibc `LD_AUDIT` library that records CUDA/NCCL/NVML library loads, dynamic symbol binding, and the two `cuGetProcAddress` ABIs. It wraps resolvers returned by resolvers too. Other function pointers remain real driver pointers. **Resolution is not proof that a function was called.**
- `tools/cupti_api_trace.c`: an independent CUPTI driver-API callback trace of completed calls, including correlation IDs, return values, and context IDs. It is diagnostic instrumentation, not a replacement CUDA backend.
- `tests/driver_probe.py`: real-device tests of both resolver ABIs, optional missing-symbol queries, and tensor round-trip correctness.

Neither probe modifies installed packages or system CUDA libraries. The loader audit uses the glibc audit namespace; it is Linux/glibc-specific. CUPTI and PyTorch profiling run in separate processes.

## Reproduce

On the test host, the existing interpreter is `/home/gongji/0z5a/bin/python`. Build the probes with `bash tools/build_probes.sh`. The CUDA Toolkit headers and CUPTI are already installed under `/usr/local/cuda`.

`tools/run_baseline.sh` records the target machine's permitted GPU UUID and existing environment paths. Adjust those explicitly for a different machine. Run from the experiment checkout:

```bash
bash tools/run_baseline.sh --runs 3 --out artifacts/A0.json
SIM_ENABLE_LOADER=1 SIM_LOADER_TRACE="$PWD/artifacts/P0-loader.jsonl" \
  bash tools/run_baseline.sh --runs 3 --out artifacts/P0.json
SIM_ENABLE_LOADER=1 SIM_LOADER_TRACE="$PWD/artifacts/P1-loader.jsonl" \
  bash tools/run_baseline.sh --runs 3 --out artifacts/P1.json
bash tools/run_baseline.sh --runs 3 --out artifacts/A1.json
bash tools/run_baseline.sh --runs 1 --profile --out artifacts/profile.json
SIM_ENABLE_CUPTI=1 SIM_CUPTI_TRACE="$PWD/artifacts/cupti.jsonl" \
  bash tools/run_baseline.sh --runs 1 --out artifacts/cupti-run.json
```

Do not attach the profiler to scored runs. Model assets are private experiment downloads or copies of an existing local cache; remove only those experiment-owned files after validation finishes.

## Remaining acceptance gates

M1 needs typed forwarding wrappers and evidence that actual calls, including resolver-returned entrypoints, are intercepted. The current loader probe does not satisfy this requirement.

M2 needs device/memory/stream semantics and a verified metadata producer-consumer closure. A callback trace cannot supply these semantics. GPU-hidden execution must reject unknown control reads and audit zero real-backend calls.

M3–M5 need held-out kernel/step calibration, causal time and host cost, NCCL matching/TP2 validation, and closed-loop arrivals through the original scheduler. Fixed trace replay cannot replace execution-driven simulation.

CUDA Graph, async scheduling, and HTTP each need independent validation beyond this initial scope.

## References

- [NVIDIA CUDA Driver entrypoint access](https://docs.nvidia.com/cuda/cuda-driver-api/driver-entry-point-access.html)
- [NVIDIA CUPTI callback API](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html)
- [NVIDIA skills catalog](https://github.com/NVIDIA/skills): catalog checked for a CUDA ABI simulator workflow; no dedicated matching skill was found.

ABI declarations are checked against the installed CUDA 13 headers. Source hashes and runtime versions are recorded with the actual runs.
