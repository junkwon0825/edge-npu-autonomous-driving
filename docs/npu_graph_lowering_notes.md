# NPU graph-lowering notes (Furiosa RNGD, furiosa-torch 2026.4.0)

These notes contain only behaviour observed in this project, through compiler messages, results and
public API names. They do not claim knowledge of compiler internals beyond the error text.

Evidence: `docs/npu_operator_compatibility.md` (74 generated rows) and the JSON files it cites.

## Observed pipeline stages (from error messages)
| Stage name seen in messages | Example of what failed there | Interpretation (hedged) |
|---|---|---|
| `failed to decompose` | `Input spec name input does not match node name input_1`; `Cannot view a tensor … with strides …` | export / decomposition of the PyTorch graph before compilation. The first was our harness (`nn.Sequential.forward(input)`); the second needs `.contiguous()` before a reshape in attention |
| `failed to import dfg … Not importable` | `_native_batch_norm_legit_no_training`, `fake_quantize_per_tensor_affine`, `fake_quantize_per_channel_affine`, `aten::_int_mm` | the aten op has no import rule into the compiler's dataflow graph: fold or rewrite before export |
| `Lowering Deadline Violation: Aten(round) is allowed up to dfg, found in primitive` | `torch.round` | the op must be lowered before the primitive level and no rule exists |
| `while verifying Primitive … verification of operator failed: Primitive Mul` | dilated convolution, dilated depthwise conv in MobileNetV3 | a lowered primitive got inconsistent tensor shapes |
| `transposed convolution is not implemented yet` | ConvTranspose2d | explicit "not implemented" |
| `from_table_lookup expects a 1D table` | bilinear interpolation (×2 and fixed size) | the interpolation lowering uses a lookup that does not support this case |
| `while lowering TacticKernel … no populated tactics` (kind `EinsumByDpe`, history `max_pool2d_with_indices` / `avg_pool2d`) | max/avg pooling at 256×512, and at 128×256 with ≤ 16 channels | no kernel tactic for that size and channel layout |
| `O… is not an operator that is yet supported … TensorOperation must not have non-tile operations` | Conv2d 2×2 stride 2 | operator form not supported |
| `TransposedLower[bf16] … LowLevelTailCrop … size_in_bytes_with_length(...) <= u16::MAX` | 3×3 conv 256→512 (YOLO26s), 512→512 and 512→256 (ResNet18) | a lowered weight slice must fit 64 KiB |
| `Inconsistent tensor group offsets for T… and T…` | ported LR-ASPP | backend scheduling / grouping; no workaround found |
| `Failed to deserialize cached output` | recompiling a graph that failed earlier | compiler-cache robustness issue |
| `peer D2D copy requires source and destination devices in the same mesh` | moving one compiled instance from `furiosa:0` to `furiosa:1` | our usage error: one `CompileModule` per device |

## Fusion and rewrites that worked (each validated against the original on CPU first)
| Pattern | Rewrite | Exact? |
|---|---|---|
| Conv → BatchNorm | fold BN into the conv (`fuse_conv_bn_eval`; structural in `Sequential`, or `torch.fx` fuse for torchvision models) | yes |
| bilinear ×2 (`align_corners=False`) | nearest ×2 + depthwise [1,2,1]⊗[1,2,1]/16 with edge replication via slice + concat | yes (≤ 5e-7) |
| ConvTranspose2d(k2, s2) | 1×1 conv to 4·C_out + PixelShuffle(2) | yes |
| dilated conv | dense kernel with inserted zeros | yes (more MACs) |
| max-pool / 2×2 s2 conv downsampling | 3×3 stride-2 conv (model change; requires training) | no, architectural |
| large 3×3 conv (C_out > 256) | `SplitConv`: split output channels into ≤ 256 + concat | yes; YOLO26s then compiles and runs (4.67 ms) |
| large C_in (512) | **not yet**: would need an input-channel split with a summed result (ResNet18 still fails) | — |
| SDPA on strided q/k/v | `.contiguous()` before the call | yes |
| YOLO decode / NMS | keep on host (shared code for GPU and NPU) | — |

## Precision
- **What works:** BF16 compute, and W8 weight-only (int8 storage, cast to BF16 in the graph).
- **What fails:**
  - int8 convolution wraps around (V1)
  - no in-graph activation quantization (fake-quant is not importable; `round` is not lowerable)
  - `get_amp_supported_dtype()` returns `[]`
- **Not tested:** FP8 compiles but its accuracy was not evaluated.

## Performance observations (qualitative; no hardware counters)
- YOLO26n exec is 1.75–2.04 ms from 320 to 640 input: overhead-bound, not compute-bound, at batch 1.
- YOLO26s 4.67 ms vs YOLO26n 2.0 ms, at about 4× the parameters.
- Multi-PE scaling is linear to 2 PEs and 3.3× at 4 PEs; 8 Python threads become host-bound.
- Two models on two PEs overlap: detection + lane parallel section 3.43 ms vs 2.24 + 2.75 ms
  sequential.
