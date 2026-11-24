# NPU model porting summary (Furiosa RNGD, furiosa-torch 2026.4.0)

Every row is a recorded compile/run on the real RNGD. Batch 1. Latency is the on-device execution time
unless stated otherwise. Details and failure messages: `docs/npu_operator_compatibility.md` (74 rows),
`docs/npu_graph_lowering_notes.md`, `docs/v3_lane_npu_porting.md`.

| Model | Task | Compile | Runtime on RNGD | Main issue | Workaround / status | Evidence |
|---|---|---|---|---|---|---|
| YOLO26n | object detection (driving loop) | PASS after porting | BF16 2.0 ms; agreement with GPU FP32: recall 0.974, precision 0.988 | unfused BatchNorm not importable; head concat lowering | fuse Conv+BN; split at the raw head; decode/NMS shared on the host | `npu_bench_bf16.json`, `v2_transport_profile_r2.json` |
| YOLO26n W8 | weight-only INT8 | PASS | 2.35 ms (slower than BF16), recall 0.957 | INT8 compute unavailable: int8 conv wraps modulo 256, int32-accumulating conv `O117 not yet supported` | weights stored in int8, cast to BF16 in the graph; **no W8A8** | `npu_int8_probe*.json`, `docs/quantization_limitations.md` |
| Lane U-Net (V3, base 32, 4 levels, 2.12 M params) | lane-marking segmentation (driving loop) | PASS after redesign (94 s) | BF16 2.8 ms; masks equal to GPU BF16 (IoU 0.991–0.998 vs GPU FP32 on the same input) | ConvTranspose, bilinear and dilated conv unsupported; max/avg-pool and 2×2 s2 conv fail at 256×512 | exact rewrites (ConvT → 1×1 conv + PixelShuffle, bilinear → nearest + [1,2,1] filter); **3×3 stride-2 conv downsampling** chosen from a pooling sweep | `v3_npu_lane_port_*.json`, `v3_lane_correctness_*.json` |
| LR-ASPP MobileNetV3 | segmentation (candidate) | FAIL | — | `Inconsistent tensor group offsets` (backend scheduling) after rewrites | not pursued; replaced by the U-Net | `v2_npu_port_lab_seg*.json` |
| YOLO26s | larger detector | FAIL as-is → PASS with rewrite | BF16 4.67 ms (vs YOLO26n 2.0 ms) | `TransposedLower … <= u16::MAX`: a lowered weight slice must fit 64 KiB (3×3 conv 256→512) | exact **output-channel SplitConv** (concat of conv slices), validated vs the original on CPU | `v3_npu_compile_yolo26s_bf16.json`, `v3_npu_split_conv.json` |
| ResNet18 | classification backbone | FAIL | — | same 64 KiB limit on 512→512 and 512→256 convs; output split is not enough | needs an input-channel split with a summed result: **not done** | `v3_npu_model_lab*.json` |
| MobileNetV3-Small | classification | PASS | 0.50 ms | — | none needed | `v3_npu_model_lab.json` |
| Tiny ViT (small test configuration) | classification (transformer) | PASS | 0.44–0.48 ms | SDPA needs contiguous q/k/v; LayerNorm / GELU first looked unsupported because of a harness bug (`nn.Sequential` input naming) | `.contiguous()` before attention; wrapper module in the test harness | `v3_npu_model_lab_rerun.json` |

## Multi-model runtime and scaling (measured)
| Item | Result | Evidence |
|---|---|---|
| Multi-PE scaling, YOLO26n | 537 → 1,052 → 1,762 inferences/s on 1 / 2 / 4 PEs; 8 PEs 947/s (host-bound) | `v2_npu_multistream2.json` |
| Two models, one server | detection on `furiosa:0`, lane on `furiosa:1`, one upload per frame; parallel section 3.43 ms vs 2.24 + 2.75 ms | `v3_npu_multimodel_bench.json` |
| Scheduling (on-host, no network) | separate PEs keep detection at 400 Hz under stress; two workers on one PE are worse than sequential; priority starves the lane model | `v3_npu_multimodel.json` |
| Scheduling (closed loop, remote) | all strategies give the same driving outcome; adaptive priority cuts dropped frames 170 → 21 and queue delay 15 → 3 ms; same PE adds ~6–8 ms latency p95 | `v3_closed_loop.json` |
| Failover | stall → GPU 0.54 s, crash → GPU 0.11 s, 0 frames lost (offline); in CARLA: stall and crash handled, logged, stall recovered to RNGD | `v3_failure_engineering.json`, `v3_closed_loop.json` |

## What limits the system (not the NPU)
- The model execution is 2–3 ms. The remote round trip in the closed loop was 63–65 ms (mean) and
  92–103 ms (p95), because of SSH transport and the shared host. That is why the detection age in the loop
  rises from 30 ms (GPU) to 121–126 ms (RNGD), p95 (runs with failed requests excluded).
- The NPU-client path letterboxes the frame to 640×360 before the lane model resizes it to 512×256. This
  double resize, not BF16, accounts for the 2–6 % mask IoU difference against the GPU path.
