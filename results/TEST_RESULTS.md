# Test results

The comparison is real vLLM with native CUDA versus the same stack with loader auditing. It measures instrumentation overhead, not simulator acceleration.

TinyLlama-1.1B, RTX 5090 GPU 4, BF16, TP1, eager, synchronous in-process engine, Triton attention, fixed 512 MiB KV cache. Four fresh processes in A0/P0/P1/A1 order; three repetitions per workload per process. Each arm contributes its median, then paired arm medians are combined geometrically. One quartet provides descriptive results, not statistical confirmation.

| Batch | Input/output tokens per request | Native ms | Audited ms | Speed ratio | Throughput change | Token identity |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 32/4 | 56.860 | 57.997 | 0.9804× | -1.96% | Exact match |
| 4 | 128/16 | 228.543 | 234.884 | 0.9730× | -2.70% | Exact match |
| 8 | 64/32 | 461.612 | 472.891 | 0.9761× | -2.39% | Exact match |

Pending-request pause: 251.257 ms, zero GPU kernels during the pause; GPU work observed before and after (2152 kernels in the complete diagnostic trace).

GPU-free serving, prediction error, TP2, and simulator wall-clock speedup: **not implemented or measured in this checkpoint**. GPU 5 and 6 had existing workloads; host-wide isolation was not available. GPUs 0–3 were not used.
