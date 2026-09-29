# sim-experiment

Execution-driven LLM serving simulation experiments using the stock vLLM scheduler.

The initial implementation contains real-device baselines, CUDA ABI feasibility probes, and a CPU cross-rank scheduler-boundary prototype. The prototype uses the installed vLLM scheduler with synthetic fixed-length token control and calibrated TP2 step costs; it is not an unmodified-engine CUDA/NCCL ABI simulator.

See [cross-rank methodology](docs/crossrank.md), [ABI feasibility scope](docs/implementation.md), and [real-device test results](results/TEST_RESULTS.md). Predictions and simulator wall-clock speedup are reported separately.
