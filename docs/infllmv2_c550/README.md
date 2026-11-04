# InfLLM-V2 on MetaX C550

The final implementation retains the existing `infllmv2_attention` pytest mark,
public prefill/decode facade and operator inventory. C550 kernel dispatch lives
in `src/flag_attn/infllmv2`; other accelerators retain the existing generic/TLE paths.

## What changed

- Use C550-compatible dense and sparse MMA layouts and launch configurations;
  avoid the original CUDA tiles exceeding C550's 64 KiB shared-memory budget.
- Fuse compressed-K preparation and pooling/selection, preserve deterministic
  block ordering, and remove host reads of device length metadata where possible.
- Reuse selected local KV across queries and GQA heads; mask padded heads and
  dimensions and choose tiles by resource usage. Explicit Selected inputs retain
  their general path, including duplicate-ID semantics.
- Split wide backward head groups and accumulate dK/dV in FP32. Preserve causal
  suffix alignment, packed lengths, custom scales and Q/K/V gradients.

## Baseline and measurement

**V0 means our initial working C550 adaptation from 2026-11-02, before the later
performance optimization rounds.** It is archived in [baseline_v0](baseline_v0/).
All six SHA256 hashes match the original adaptation and first optimization-round
baseline manifests. V5 is the final generalized implementation in this PR.
Version numbers are explanatory labels, not Git tags.

Both versions were measured directly on physical C550 GPU1, using identical
inputs (seed 7411), three warmup calls, eleven alternating A/B rounds, and CUDA
Event medians for the complete API. CUDA Graph and profiler instrumentation were
not used for these latency numbers. The exact sources, software version and
timestamps are in [measurement_manifest.json](measurement_manifest.json).
No cross-round speedups are multiplied together.

Forward: **230 comparable cases out of 230**, equal-weight geometric
mean **6.92x**; 230 above 1.05x,
0 within 0.95–1.05x, 0 below 0.95x.
V0 could not run 0 forward cases.
Forward+backward: 6 comparable cases out of 18,
12 unsupported by V0.

| Case | Hq/Hkv | D | BF16 V0 -> V5 ms | Speedup | FP16 V0 -> V5 ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| 8K prefill | 4/4 | 64 | 20.014 -> 5.286 | 3.79x | 19.896 -> 4.671 | 4.26x |
| 8K prefill | 16/4 | 128 | 371.011 -> 11.424 | 32.48x | 367.172 -> 11.060 | 33.20x |
| 8K prefill | 16/4 | 96 | 371.886 -> 11.391 | 32.65x | 371.907 -> 11.068 | 33.60x |
| 8K prefill | 16/4 | 256 | 1262.532 -> 37.698 | 33.49x | 1244.920 -> 37.003 | 33.64x |
| 8K prefill | 32/2 | 64 | 11.988 -> 4.720 | 2.54x | 11.566 -> 4.190 | 2.76x |
| 8K prefill | 32/2 | 128 | 185.381 -> 9.193 | 20.16x | 183.564 -> 8.872 | 20.69x |
| Dense 1K | 64/2 | 256 | 18.760 -> 1.384 | 13.56x | 19.161 -> 1.342 | 14.28x |
| Chunk 128/8K | 16/4 | 96 | 9.649 -> 0.878 | 10.99x | 9.739 -> 0.875 | 11.13x |

All representative rows use B=1. The 8K prefill rows use Q=KV=8192,
Block=64, TopK=64, init/local=1/32. The complete 248-row table, including
unchanged/slower and unsupported cases, is in [performance.md](performance.md).
[comparison_results.json](comparison_results.json) preserves every sample.

The matrix covers selected combinations of BF16/FP16, GQA1–64 including
non-power-of-two groups, D32–256, batch sizes 1/2/4, sequence boundaries,
Block/TopK and windows. It is not the Cartesian product of all parameters.
Unsupported-baseline rows have no speedup; final-version correctness is covered
by the independent reference tests rather than inferred from a failed baseline.

## Validation and registered mark

The integrated branch passed **151 tests with 4 skips** using strict marker
validation. This includes the existing operator suite, C550 regression tests and
44 independent generalized-shape reference tests. Seven selected existing
benchmark cases also passed under the same registered mark. The four skips are
NVIDIA/Hopper-specific tests. Logs are in [validation.json](validation.json).

```bash
pytest tests/flag_attn/test_infllmv2_attention.py tests/flag_attn/test_infllmv2_c550.py tests/flag_attn/test_infllmv2_c550_generalization.py -m infllmv2_attention --strict-markers -q
pytest benchmark/test_infllmv2_attention_benchmark.py -m infllmv2_attention --strict-markers -q
python benchmark/infllmv2_c550_ab.py --baseline-dir docs/infllmv2_c550/baseline_v0 --optimized-dir src/flag_attn/infllmv2 --cases docs/infllmv2_c550/comparison_cases.json --output results/infllmv2_v0_v5_repeat.json
```

Run from the repository root in a C550/MACA environment with `PYTHONPATH=src`.
The full registered benchmark command also includes longer sequences than the
seven selected integration cases. It is not the command used for the A/B table.

## Remaining limits

Wide-dimension backward kernels still have register spills; FP32 atomic gradient
accumulation is not bitwise deterministic. The two attention paths require FP32
partial buffers, and some cases reduce instructions while increasing memory
traffic. Short queries, explicit Selected inputs and unsupported configurations
use general fallbacks. Benefits are limited to the measured shapes and types.
