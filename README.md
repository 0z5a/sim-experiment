# sim-experiment

Execution-driven LLM serving simulation experiments using the stock vLLM scheduler.

The initial implementation contains real-device baselines, CUDA ABI feasibility probes, and a CPU cross-rank scheduler-boundary prototype. The prototype uses the installed vLLM scheduler with synthetic fixed-length token control and calibrated TP2 step costs; it is not an unmodified-engine CUDA/NCCL ABI simulator.

See [cross-rank results](results/crossrank/TEST_RESULTS.md), [methodology](docs/crossrank.md), [ABI feasibility scope](docs/implementation.md), and [real-device probe results](results/TEST_RESULTS.md).

On the measured Qwen2.5-0.5B TP2 stack, CPU simulation takes 22.17 seconds for a campaign whose real validation workloads take 281.62 seconds: 12.70× with existing calibration, or 1.48× after charging 168.10 seconds of initial calibration. Cohort duration errors are 1.17–4.94%; the two-wave workload's median TTFT error is −10.66%, so the predefined all-metric 10% accuracy gate is not met. Results reflect a shared machine, fixed-length generation, and the calibrated workload domain.
