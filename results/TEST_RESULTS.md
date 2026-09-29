# Real-device E2E and loader overhead

Native vLLM versus loader auditing, using the same real GPU execution. These rows measure probe overhead, not simulator acceleration. Four fresh processes per model, A0/P0/P1/A1, three repeats per arm; arm medians combined geometrically. One quartet is descriptive evidence, without a confidence claim.

| Model | Batch | P/O | Native ms | Audited ms | Speed ratio | Throughput change | Tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyLlama-1.1B | 1 | 32/4 | 57.843 | 57.521 | 1.0056× | +0.56% | Exact per-request match |
| TinyLlama-1.1B | 4 | 128/16 | 233.865 | 232.943 | 1.0040× | +0.40% | Exact per-request match |
| TinyLlama-1.1B | 8 | 64/32 | 467.880 | 468.747 | 0.9981× | -0.19% | Exact per-request match |
| Qwen2.5-0.5B | 1 | 32/4 | 62.202 | 63.161 | 0.9848× | -1.52% | Exact per-request match |
| Qwen2.5-0.5B | 4 | 128/16 | 254.426 | 258.296 | 0.9850× | -1.50% | Exact per-request match |
| Qwen2.5-0.5B | 8 | 64/32 | 512.184 | 522.868 | 0.9796× | -2.04% | Exact per-request match |


Both models also completed the same real-weight matrix with TP2 on GPUs 4/7, in native and audited fresh sessions; both ranks agreed. TP2 sessions are smoke comparisons, not independent quartets.

Fixed stack: existing 0z5a environment, RTX 5090, BF16, eager, synchronous engine, Triton attention, fixed 512 MiB KV budget. Other machine workloads remain running. GPUs 0–3 were not used. See the cross-rank report for the separate CPU simulator experiment.
TinyLlama-1.1B: pending-request pause 251.783 ms, no kernels during pause, 2152 kernels observed in the complete diagnostic trace.
Qwen2.5-0.5B: pending-request pause 251.429 ms, no kernels during pause, 2332 kernels observed in the complete diagnostic trace.
