# Reproducible evidence

`evidence.tar.gz` contains the raw measured baselines, token IDs, TP2 rank outputs, independent profiler/driver traces, cross-rank calibration, held-out validation, simulation outputs, configuration-only scheduler snapshot, source hashes, process inventories, and failed-start logs. Its digest is in `evidence-sha256.json`. It contains no model weights or credentials. Extract into a fresh directory; load the pickle only as this experiment's trusted configuration artifact.

`crossrank/TEST_RESULTS.md` is the final performance table. `crossrank/report.json` retains all three repeats, prediction errors, batch-membership checks, and accounting inputs. The archive's report has identical values; the checked-in Markdown clarifies that the feedback check contains two scenarios. Row values are medians; speedups and errors are medians of paired ratios, not ratios of separately rounded medians.

All 12 validation cases completed the requested token counts. Both real ranks and the CPU schedulers have identical batch membership after validating the engine's random request-ID suffix mapping. The all-metric accuracy gate is **not met** because two-wave median TTFT error is −10.66%. No held-out observation was used for fitting.

Validation: 14 unit tests, remote Ruff checks, original-scheduler CPU integration with CUDA hidden, real TP1 fresh-process comparisons and TP2 generation on TinyLlama/Qwen, separate pause profiling, and C11 compilation with warnings as errors. Three high-load repeats share one resident engine; they do not establish cross-startup confidence intervals.

The successful high-load ranks exited naturally with codes `[0, 0]`. The first failed startup left PID 953574 in GPU initialization on GPU 4; it was still present at the final snapshot and was not signaled. GPU 7 was free in that snapshot. GPU 0–3 processes, the installed environment, and lcpu NFS were not modified. GPU 5/6 had pre-existing workloads and were not used.

Task-owned TinyLlama and Qwen assets were removed, freeing 2,200,120,472 and 999,588,359 bytes respectively. Cleanup records are committed; shared model caches were preserved. The remaining failed process had no open file descriptor or mapped file referencing the Qwen task directory when cleanup occurred.

The results support a calibrated scheduler-boundary CPU prototype. CUDA/NCCL ABI execution, arbitrary control-payload semantics, asynchronous overlap, and generated-text equivalence remain outside this implementation.
