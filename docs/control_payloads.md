# Control payload map: installed vLLM 0.29.0

The observed runner is `vllm/v1/worker/gpu/model_runner.py` (V2), not the older `gpu_model_runner.py`. Paths below are relative to the installed `vllm` package. The experiment saves file hashes; these locations are specific to that stack.

| Payload | Producer / path | Consumer / visibility | Current evidence |
|---|---|---|---|
| Sampled token IDs | `GPUModelRunner.sample`, then `sample_tokens` in `worker/gpu/model_runner.py:1861` | `AsyncOutput.__init__` copies IDs to host (`worker/gpu/async_utils.py:140`); `get_output` waits on `copy_event` at line 168 before list conversion | Source chain and actual generated token IDs; kernel argument schema remains unimplemented |
| Valid sampled-token counts | `sample` returns `num_sampled` alongside sampled IDs and rejected counts | Separate D2H in `async_utils.py:149`; list conversion at line 175; each request's IDs are truncated to its count | Required independently of token payload; cannot replace every D2H with one synthetic token |
| Request index mapping and sampled/rejected counts | `postprocess_sampled` in `model_runner.py:1477`, called after output-copy submission at line 1921 | GPU request state feeds the next model step; includes last token, computed-token progress, and mapping | Required device-side control state even when it is not read back by the CPU |
| Attention lengths / slot mappings | V2 input batch and block table preparation | Attention kernels and subsequent request state | Dynamic fields must be tracked before a kernel performance key is valid |
| Request-to-output mapping | `ModelRunnerOutput.req_ids` constructed from `input_batch.req_ids`, `model_runner.py:1886` | Original engine output processor | Preserved by using the installed engine; compared per request rather than as an unordered token multiset |

The asynchronous D2H path remains present with `async_scheduling=False`. Disabling asynchronous scheduling therefore does not remove stream/event or host-visibility requirements.

This is a source-and-trace map, **not a complete semantic closure**. Shapes, dtypes, generations, write ranges, and parameter schemas must be established for the actual kernels before M2. The profiler verifies the synchronous engine boundary; it does not prove that all metadata can already be simulated.
