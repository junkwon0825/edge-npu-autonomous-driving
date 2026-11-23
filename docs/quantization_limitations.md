# Quantization on Furiosa RNGD (furiosa-torch 2026.4.0): what works, what does not, and why

Status: **INT8 compute (W8A8) — BLOCKED by the toolchain** · **INT8 weight-only (W8, BF16 compute) — TESTED: no speed-up,
accuracy loss (§4)** · **BF16 — TESTED, recommended** · FP8 conv — compiles, numerically not evaluated.

## 1. API level (2026-11-22)
- `furiosa.torch` exposes no `quantize` / `calibrate` / QAT entry point; `get_amp_supported_dtype()` returns `[]`.
- Quantization code in the installed packages exists only in LLM kernels (`kernels/qwen3`, `kernels/exaone4`,
  `kernels/common/dynamic_quantization.py`) used by `furiosa-llm` (FP8 path). No vision/CNN path, no
  documented PyTorch-CNN INT8 workflow. `furiosa-models` vision modules are reported empty after the RNGD
  transition (knowledge-base note, unverified).

## 2. Compiler level (2026-11-23) — asked the compiler directly
The typecast table of the SDK maps `torch.int8 → tcl.i8` and `torch.float8_e4m3fn → tcl.f8_e4`, so
"no API" did not prove "no INT8 capability". `remote/int8_probe.py` compiles a tiny Conv(16→32, 3×3)
block on `furiosa:0` in each formulation, one subprocess per case
(`reports/evidence/npu_int8_probe.json`, `npu_int8_probe2.json`):

| Formulation | Compiles / runs | Result |
|---|---|---|
| FP32 control | PASS | max abs diff vs CPU 0.0065 |
| INT8 weights stored in the graph, dequantised to BF16, BF16 conv (**weight-only W8**) | PASS | max abs diff vs CPU 2.4e-4 |
| `conv2d(int8 activations, int8 weights)` | PASS, output dtype int8 | **wrap-around accumulation**: device output equals `((exact + 128) mod 256) − 128` for 100 % of outputs, equals the exact integer result for 0.4 % (exact values reach ±115 687). Bit-identical to PyTorch-CPU int8 semantics, but useless for inference: there is no int32 accumulator to requantize from |
| `conv2d` on int32-cast int8 tensors (int32 accumulation) | **FAIL** | `O117 is not an operator that is yet supported by the compiler` / `TensorOperation must not have non-tile operations` |
| `torch._int_mm(int8, int8) → int32` (1×1 conv as GEMM) | **FAIL** | `aten::_int_mm` "Not importable" in the DFG importer |
| `conv2d` in `float8_e4m3fn` | PASS, output fp8 | no CPU reference kernel (`slow_conv2d_cpu` not implemented for fp8); accuracy not evaluated |

## 3. Conclusion
Static INT8 quantization of a CNN needs int8×int8 products **accumulated in int32**, then requantized
with per-layer scales from calibration data. With the public `furiosa.torch` 2026.4.0 graph API, none
of the three ways to express that (int8 conv with wider output, int32 conv, `_int_mm`) is lowerable,
and the one int8 conv that compiles keeps the result in int8 with modular overflow. The hardware may
support INT8 internally (the LLM stack uses low-precision kernels), but there is no supported way to
reach it from a PyTorch vision graph in this SDK. **INT8 compute is therefore BLOCKED; nothing in this
repository is claimed as INT8-accelerated.**

What is possible and was done:
- **BF16** (precision reduction, no calibration): 2.000 ms compute vs 2.107 ms FP32; recall vs GPU 0.972.
- **W8 weight-only**: all 102 convolutions of the fused YOLO26n with per-output-channel symmetric INT8
  weights (absmax/127), dequantised to BF16 inside the graph. This reduces the stored weight bytes
  (int8 instead of bf16/fp32) but compute stays BF16, so no INT8 speed-up is expected. Results below.

Calibration frames (18, `split == "calib"`) remain reserved and unused because no activation
quantization path exists.

## 4. W8 weight-only results (measured 2026-11-23)
**This is weight-only INT8, not INT8 inference**: weights are stored as int8 and converted to BF16
inside the graph; every multiply-accumulate runs in BF16.

| Metric | FP32 | BF16 | W8 (weight-only) | Source |
|---|---|---|---|---|
| Weight bytes handed to the compiler | 9.64 MB | 4.82 MB | **2.44 MB** (2.40 int8 + 0.035 bf16 scales/bias) | `evidence/model_size.json` |
| AOT compile (shared host) | 1901.9 s | 1169.6 s | 1161.7 s | `npu_compile*.json` |
| `bench` compute, mean / p95 / p99 (n=216, per-frame H2D before) | 2.107 / 2.119 / 2.125 ms | 2.000 / 2.020 / 2.071 ms | **2.352 / 2.371 / 2.682 ms** | `npu_bench*.json` |
| `perf` sequential, inputs resident, mean / p99 (n=500) | 2.149 / 2.203 ms | 1.854 / 1.905 ms | 2.155 / 2.175 ms | `npu_perf*.json` |
| Sustained 20 s, sync every frame | 470 fps | 543 fps | 446 fps | `npu_perf*.json` |
| Board power under that load (idle 39 W) | 53.3 W | 53.6 W | 51.2 W | `npu_perf*.json` (furiosa-smi, 1 W resolution, n≈28) |
| NPU vs same graph on host CPU, boxes mean abs | 0.032 | 0.034 | 0.029 | `npu_bench*.json` |
| Recall / precision / mean IoU vs GPU FP32 (54 eval frames) | 0.988 / 0.992 / 0.991 | 0.972 / 0.992 / 0.989 | **0.957 / 0.980 / 0.966** | `gpu_vs_npu*.json` |
| Mean / max abs Δconf vs GPU | 0.005 / 0.161 | 0.007 / 0.032 | 0.022 / 0.152 | `gpu_vs_npu*.json` |

Interpretation:
- W8 is **not faster**. It is slower than BF16 in both protocols, and slower than or equal to FP32
  depending on the protocol (2.35 vs 2.11 ms in `bench`, 2.16 vs 2.15 ms in `perf`). This is
  consistent with the dequantisation (int8→bf16 multiply by scale) being executed in the graph at
  every inference instead of being constant-folded.
- The NPU executes the W8 graph faithfully (0.029 mean abs vs the same graph on CPU); the accuracy
  drop vs GPU (recall 0.957, IoU 0.966) is caused by the quantisation itself (per-output-channel
  absmax on fused Conv+BN weights, no calibration, no per-layer exclusion).
- The only benefit measured is weight storage (−49 % vs BF16). Device memory use could not be
  observed: `furiosa-smi status` reported 0.00 GiB during all runs.
- The two NPU timing protocols differ by up to 0.2 ms for the same compiled graph (shared host,
  load average 19–34); differences below that are not significant.

Verdict: on this SDK, **BF16 is the best supported precision** for this detector (fastest, smallest
accuracy loss after FP32). W8 is kept as a documented negative result.

## 5. Alternatives (not executed)
- Ask Furiosa for the RNGD vision quantization workflow / a release with `furiosa-models` vision support.
- Export INT8 via a vendor tool once documented; do not install mismatched SDK versions (the firmware
  is 2026.4.0 and the SDK must match).
- For throughput rather than precision: batched inputs or several `furiosa:N` devices (8 visible).
