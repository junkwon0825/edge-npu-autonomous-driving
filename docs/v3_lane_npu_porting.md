# V3: porting a lane-segmentation network to Furiosa RNGD

SDK: `furiosa-torch` 2026.4.0. All results are recorded compiles and runs:
- `reports/evidence/v2_npu_port_lab_*.json`
- `v3_npu_lane_port_*.json`
- `v3_lane_training.json`
- `v3_npu_multimodel_bench.json`

Code:
- `remote/port_lab.py`, `remote/lane_port_v3.py`
- `src/npu_ad/perception/lane_net.py`, `src/npu_ad/npu/rewrites.py`

**Goal.** Gain porting experience on a model other than YOLO, and end with a lane model that actually
drives LKA. The pipeline:

PyTorch → graph inspection → rewrite → AOT compile → unsupported-op analysis → model change → run →
correctness vs CPU → task accuracy.

## Timeline of failures and fixes
| # | Attempt | Compiler result | Learned / next step |
|---|---|---|---|
| 1 | U-Net (ConvTranspose up), LR-ASPP MobileNetV3 as-is | `transposed convolution is not implemented yet`; unfused BatchNorm `Not importable`; bilinear `from_table_lookup expects a 1D table`; dilated conv `Primitive Mul` verification | four operator gaps |
| 2 | Exact rewrites in isolation (V2): bilinear ×2 → nearest + [1,2,1]/4 depthwise; ConvT(k2, s2) → 1×1 conv + PixelShuffle; dilated → dense kernel | all compile and run, cosine 1.00000 vs CPU | the rewrites are correct |
| 3 | Ported U-Net (rewrites applied) at 1×3×256×512 | `max_pool2d … no populated tactics` on 16×256×512 | the operator is supported, but not at this size |
| 3b | Ported LR-ASPP | `Inconsistent tensor group offsets` after 272 s | internal scheduler error; not pursued |
| 4 | Max-pool sweep (V3) | passes at 16×40×40, 24/32/48×128×256; fails at 16×64×128, 16×128×256 and every 256×512 case (4–64 channels) | the limit depends on spatial size **and** channels; no simple rule. Avg-pool behaves the same |
| 5 | Downsampling with a 2×2 stride-2 conv | `O2987 is not an operator that is yet supported … TensorOperation must not have non-tile operations` | 2×2/s2 conv is unsupported |
| 6 | **Downsampling with a 3×3 stride-2 conv (YOLO-style)** | **PASS**: 256×512 base 16, 0.82 ms, cosine 0.99998, pixel agreement 100 %; base 32, 2.70 ms, cosine 0.99996 | the architecture to train |
| 7 | Trained V3 model (`LaneUNet`, base 32, 4 levels, 2.12 M parameters) | compiles in 94 s (first time; cached afterwards), runs on `furiosa:1` | deployed |

## Final architecture and correctness
**`LaneUNet`:**
- 3×3 stride-2 Conv-BN-ReLU for downsampling
- ConvTranspose(2, 2) for training, exported as 1×1 conv + PixelShuffle
- BatchNorm only directly after a conv inside a `Sequential`, folded on export
- `export_for_npu` changes outputs by at most 2.1e-5 on CPU; a unit test covers it

**Training** (`scripts/v3_train_lane.py`):
- **Data:** 2,700 CARLA frames with RoadLine masks from the semantic camera (`scripts/v3_collect_lane_data.py`), 3 maps, 7 weathers.
- **Split by map:** Town10HD_Opt + Town04 for train/val (1620/180); **Town03 held out entirely** (900).
- **Training setup:** BCE + Dice loss, as in the author's earlier Kaggle lane U-Net; 30 epochs, AdamW, AMP, 31.8 min on the RTX 3060 (sharing the GPU with CARLA).

| Metric (threshold logit > 0) | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation (Town10 / Town04) | 0.817 | 0.899 | 0.882 | 0.917 |
| **Held-out Town03, GPU FP32** | **0.727** | **0.842** | 0.873 | 0.813 |
| Held-out Town03, 150 frames upsampled to 800×450 — GPU FP32 | 0.688 | 0.815 | 0.869 | 0.768 |
| same 150 frames — **RNGD BF16** | **0.664** | **0.798** | 0.862 | 0.742 |
| RNGD vs GPU mask agreement (pixel IoU), 72 camera frames / 150 Town03 frames | 0.906 / 0.932 mean | — | — | — |

- **Weather.** The Town03 test set contains no night frames, so night generalisation is not measured.
  Held-out IoU per weather ranges from 0.69 (ClearNoon) to 0.76 (WetCloudySunset).
- **BF16 cost.** BF16 lowers lane IoU by about 0.025 against GT. The NPU and GPU masks differ mostly
  at marking edges.

## Deployment on RNGD (`remote/npu_server_v3.py`)
| Request (remote, V2 lossless transport, n = 144) | e2e mean / p95 | NPU exec p50 |
|---|---|---|
| detection only (`furiosa:0`) | 55.0 / 64.2 ms | 2.05 ms |
| lane only (`furiosa:1`) | 59.2 / 64.4 ms | 2.80 ms |
| **detection + lane, one upload, concurrent on two PEs** | **61.2 / 64.3 ms** | 2.24 + 2.75 ms; parallel section 3.43 ms |

**Lessons:**
1. **Read the compiler's own error text.** It was specific enough each time to choose the next
   experiment.
2. **Validate every rewrite against the original op on CPU before compiling.** Each rewrite in this
   document was checked that way first.
3. **Supported is not the same as supported at your shape.** Max-pool compiles small and fails large.
4. **Change the architecture once you know the constraints, then train.** The model was trained for
   the hardware rather than patched after training.

## Update after the V3 closed loop: where the RNGD vs GPU difference comes from
`scripts/v3_lane_correctness.py` runs the same frames through six paths (`v3_lane_correctness_*.json`).
It used 240 lane-set frames and 200 real 800×450 closed-loop frames, covering night, rain, sunset and day.

| Comparison | Mask IoU vs GPU FP32 | What it isolates |
|---|---|---|
| RNGD BF16, same preprocessing as the GPU path | 0.991–0.998 | NPU numerics: identical to GPU BF16 |
| Deployed RNGD path | 0.937–0.980 | preprocessing: a double resize, letterbox content 640×360 → 512×256 |

The mask agreement of 0.906 / 0.932 reported above therefore reflects the preprocessing difference,
not BF16. Making both paths use the same preprocessing is the fix (roadmap).
