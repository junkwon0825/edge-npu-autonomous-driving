# Quantization on Furiosa RNGD (furiosa-torch 2026.4.0): what works, what does not, and why

Status: **INT8 compute (W8A8) — BLOCKED by the toolchain** · **INT8 weight-only (W8, BF16 compute) —
see results table** · **BF16 — TESTED** · FP8 conv — compiles, numerically not evaluated.

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

## 4. W8 weight-only results
See `reports/npu_compilation.md` § "W8 weight-only" (filled from `reports/evidence/npu_compile_w8.json`,
`npu_bench_w8.json`, `gpu_vs_npu_w8.json`).

## 5. Alternatives (not executed)
- Ask Furiosa for the RNGD vision quantization workflow / a release with `furiosa-models` vision support.
- Export INT8 via a vendor tool once documented; do not install mismatched SDK versions (the firmware
  is 2026.4.0 and the SDK must match).
- For throughput rather than precision: batched inputs or several `furiosa:N` devices (8 visible).
