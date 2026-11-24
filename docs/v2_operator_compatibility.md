# V2: NPU operator compatibility, model porting and quantization lab

SDK: `furiosa-torch` 2026.4.0 (rev 5800e2ba24), firmware 2026.4.0. Furiosa RNGD on a shared host.

- The per-operator database is `docs/npu_operator_compatibility.md`. It is generated from evidence by
  `scripts/v2_op_table.py`.
- Raw results: `reports/evidence/v2_npu_port_lab_*.json`.
- Code: `remote/port_lab.py`. Each case is compiled and run in its own subprocess with a timeout, and
  compared with the same module on the host CPU.

## 1. Operator probes (tiny BF16 blocks, 1×16×40×40)
**Compile and run, cosine vs CPU ≥ 0.99999:**
- Conv2d + ReLU
- Hardswish, Hardsigmoid
- SE block (adaptive average pool, 1×1 convs, broadcast multiply)
- depthwise conv
- MaxPool2d
- nearest ×2 upsample
- channel concat, channel softmax
- PixelShuffle
- slicing + concat

argmax compiles and runs, but BF16 ties flip some pixels (cosine 0.998).

**Fail, with the compiler's own reason:**

| Operator | Compiler message | Category |
|---|---|---|
| BatchNorm2d (unfused, eval) | `failed to import dfg: … _native_batch_norm_legit_no_training … Not importable` | needs fusion before export |
| ConvTranspose2d | `transposed convolution is not implemented yet` | needs graph rewrite |
| bilinear interpolate (×2 or fixed size) | `from_table_lookup expects a 1D table; use from_batched_table_lookup` | needs graph rewrite |
| dilated Conv2d (d = 4) | `verification of operator failed: Primitive Mul …` (shape mismatch in the lowered kernel) | needs graph rewrite |
| `torch.round` | `Lowering Deadline Violation: Aten(round) is allowed up to dfg, found in primitive` | not lowerable |
| `fake_quantize_per_tensor_affine` / `_per_channel_` | `failed to import dfg: … Not importable` | not supported |

## 2. Exact graph rewrites (validated against the original operator)
Each rewrite was first checked on CPU in FP32 against the original operator (max abs difference
≤ 1e-6, i.e. float rounding). It then compiles and runs on RNGD with cosine 1.00000.

| Original | Rewrite | Why it is exact |
|---|---|---|
| bilinear ×2, `align_corners=False` | nearest ×2, then a depthwise 3×3 conv with kernel [1,2,1]⊗[1,2,1]/16 and edge replication (slice + concat) | each output is 0.75·x_i + 0.25·x_{i±1}, which equals the [1,2,1]/4 smoothing of the nearest-upsampled signal; replicate padding reproduces the border clamp |
| ConvTranspose2d(k = 2, s = 2) | 1×1 conv to 4·C_out channels + PixelShuffle(2) | every input pixel produces one independent 2×2 output block |
| dilated conv (k, d) | dense conv with kernel (k−1)·d+1 and the weights at every d-th position (zeros elsewhere) | identical sums; costs more MACs (81 vs 9 taps for k = 3, d = 4) |

## 3. Segmentation model porting (lane-segmentation candidates, 1×3×256×512)
Two architectures were tried: a small 4-level U-Net (16–128 channels, 0.45–0.48 M parameters) and
torchvision LR-ASPP MobileNetV3-Large (3.2 M parameters). Weights are seeded random. Porting
correctness means NPU vs CPU on the same weights; segmentation accuracy is not evaluated.

| Model | Step | Result |
|---|---|---|
| U-Net, ConvTranspose up | as-is (BN folded) | FAIL: `transposed convolution is not implemented yet` |
| U-Net, bilinear up | as-is (BN folded) | FAIL: `from_table_lookup expects a 1D table` |
| U-Net, ConvTranspose → conv + PixelShuffle | ported (CPU difference vs original 4.5e-8) | FAIL: `failed to lower the operator … TacticKernel EinsumByDpe … no populated tactics`; the history names `max_pool2d_with_indices` on a 16×256×512 tensor |
| U-Net, bilinear → exact rewrite | ported (CPU difference 6e-8) | FAIL: same `max_pool2d` tactic failure at 16×256×512 |
| LR-ASPP MobileNetV3-L | as-is | FAIL: unfused BatchNorm not importable |
| LR-ASPP, BN folded | fused | FAIL: `Primitive Mul` verification (dilated depthwise convs in the backbone) |
| LR-ASPP, dense dilated kernels + exact bilinear ×2 steps (final ×8 as 3 × ×2) | ported (CPU difference vs original 0.0014; the ×8 replacement is not exact) | FAIL after 272 s of compiling: `Inconsistent tensor group offsets for T8510 and T7679` |

**Status: lane segmentation does not run on RNGD yet.** This is a documented engineering result. Each
rewrite removed the error it targeted, and a deeper one appeared:
- **U-Net.** A max-pool that compiles at 40×40 has no lowering tactic at 256×512 with 16 channels.
  This is a size or tiling limit, not an unsupported operator.
  - Next candidates: a strided convolution instead of max-pool, or a smaller input or first stage.
- **LR-ASPP.** It reaches the backend scheduler and fails on tensor grouping. That error is internal
  and has no public explanation.

Continued in V3 Phase 1 (`experiments/npu_porting/lane_segmentation/`).

## 4. Quantization lab: what the SDK allows
V1 established three points:
- BF16 is the recommended precision.
- W8 weight-only quantization works: weights take 49 % less storage, there is no speed-up, and
  agreement is lower.
- True INT8 compute (W8A8) is blocked: int8 conv accumulates with wrap-around, an int32-accumulating
  conv is not supported, and `aten._int_mm` cannot be imported.

V2 does not repeat those attempts. It asks what *activation* quantization would need:

| Question | Probe | Result |
|---|---|---|
| Is there a quantization or calibration API? | introspection of every `furiosa.torch` submodule for quant/calib/amp/int8/fp8 names | none; only typecast tables (`_DTYPE_ETYPE_MAP`); `get_amp_supported_dtype()` returns `[]` |
| Per-tensor activation fake-quant | `fake_quantize_per_tensor_affine` | not importable |
| Per-channel activation fake-quant | `fake_quantize_per_channel_affine` | not importable |
| Explicit round/clamp quantization | `clamp(round(x/s)) * s` | `round` not lowerable |
| Round-trip through an int8 tensor | `… .to(int8).to(bf16)` | `round` not lowerable |

**Conclusion.** With `furiosa-torch` 2026.4.0 as used here, activations cannot be quantized or even
quantization-simulated inside the compiled graph. Calibration is therefore moot.

The only working low-precision options are:
- BF16 compute, and
- weight-only INT8 (W8): int8 storage, cast to BF16 in the graph.

This answers the question more strongly than V1's "INT8 blocked". It is not a claim about the
Furiosa toolchain in general. Other SDK components (for example an LLM or ONNX path) were not
available in this environment and were not tested.

## 5. Multi-stream / multi-PE behaviour (YOLO26n BF16, 8 s per configuration)
The SDK exposes `furiosa:0..7` on one RNGD. The device configuration is
`Fusion(num_pe_per_cluster=1)`, so each device is a single PE.

| Configuration | Throughput | Latency p50 / p95 | Health check after |
|---|---|---|---|
| 1 thread, PE 0 | 537 fps | 1.86 / 1.88 ms | PASS |
| 2 threads, same PE | 680 fps | 2.94 / 2.97 ms | PASS |
| 2 threads, PE 0 and 1 | 1052 fps | 1.90 / 2.00 ms | PASS |
| 4 threads, PE 0–3 | 1762.5 fps | 2.10 / 3.08 ms | PASS |
| 8 threads, PE 0–7 | 947 fps | 8.19 / 12.96 ms | PASS |

- **Scaling.** It is nearly linear up to 2 PEs and gives 3.3× at 4 PEs. It collapses at 8: the Python
  host side (8 threads, GIL, a loaded 16-core host) becomes the limit, not the NPU.
- **Two threads on one PE.** They overlap host overhead (+27 % throughput) at a higher per-call
  latency.
- **Usage error found on the way.** Moving one compiled instance from `furiosa:0` to `furiosa:1` fails
  with `peer D2D copy requires source and destination devices in the same mesh`. Each device needs
  its own `CompileModule` instance; loading one from the cache takes 3.2–5.0 s.
- **For multi-model V3 work.** Separate PEs per model are a measured, viable layout.

Evidence: `v2_npu_multistream.json`, `v2_npu_multistream2.json`.
