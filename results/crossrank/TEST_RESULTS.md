# Cross-rank high-load E2E results

CPU scheduler-boundary adapter; this is not U0 CUDA/NCCL ABI simulation. Qwen2.5-0.5B, TP2, GPUs 4/7, max 32 active requests, P=64/O=64. Three workload repeats in one resident real-engine session; two independent scheduler instances in the CPU simulation.

| Workload | Real workload s | Simulator s | Workload speedup | Duration error | Throughput error | TTFT median error | ITL median error |
|---|---:|---:|---:|---:|---:|---:|---:|
| burst-128 | 11.075 | 0.301 | 35.9× | -2.57% | +2.64% | -1.97% | -0.86% |
| burst-256 | 22.230 | 0.547 | 40.5× | -2.93% | +3.02% | -2.36% | -2.90% |
| burst-512 | 43.670 | 1.050 | 41.9× | -1.17% | +1.19% | -4.80% | +0.19% |
| two-waves-192 | 17.026 | 0.430 | 40.0× | -4.94% | +5.20% | -10.66% | -8.54% |

Initial accuracy target (absolute median errors ≤10% for all four metrics in every workload): NOT MET.

| Workload | CPU prediction vs GPU oracle: duration hits | Throughput hits | TTFT hits | ITL hits | All four hit |
|---|---:|---:|---:|---:|---:|
| burst-128 | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) |
| burst-256 | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) |
| burst-512 | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) | 3/3 (100.00%) |
| two-waves-192 | 3/3 (100.00%) | 3/3 (100.00%) | 0/3 (0.00%) | 3/3 (100.00%) | 0/3 (0.00%) |
| Total | 12/12 (100.00%) | 12/12 (100.00%) | 9/12 (75.00%) | 12/12 (100.00%) | 9/12 (75.00%) |

A prediction hits when its absolute relative error against the GPU run is ≤10% (inclusive). Counts use individual held-out workload repeats, not the median error and not lookup/cache coverage. Duration uses the slower real rank; throughput uses that duration; TTFT and ITL use rank 0 host-visible token timestamps, matching the error table above. CPU refers to the simulator producing the prediction, not a separately modeled CPU execution cost. The 12 trials share one real-engine session and are not independent hardware restarts.

One-time real initialization, warmup and calibration: 168.10 s. Complete CPU process measurement (imports, fit, 12 validation workloads and two feedback scenarios): 22.17 s. Real validation workload total: 281.62 s.

Campaign speedup with an existing calibration: 12.70×. Charging this campaign the entire one-time calibration cost: 1.48×. Downloads are excluded. Workload rows exclude process startup and calibration; campaign figures include simulator startup.

Feedback check: reducing modeled step durations to 20% changes the original scheduler's maximum waiting queue from 160 to 128. This is a causal sensitivity fixture, not a measured device optimization.

The cost model is fit only from calibration files (batches 1/8/16/32, prompts 32/64/128, O=96). Held-out requests/arrivals/output lengths are not used for fitting. Mixed prefill/decode steps and features outside the calibration domain are rejected. Rank costs include communication and host overhead; NCCL is not added again.

Fixed token counts and request lifetimes are validated. Synthetic token values do not reproduce model text. No CUDA context is initialized by the simulation. GPU 4/7 were shared during calibration/validation; see the saved GPU/process snapshots and failed-run notes. These results are conditional on that measured stack and contention, and are not isolated-hardware estimates or statistical confidence intervals.
