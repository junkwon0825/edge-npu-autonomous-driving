# YOLO GPU Baseline (Phase 2, Gate C)

**Status: PASS** — evidence `reports/evidence/gpu_benchmark_{fp32_clean,fp16_clean,fp32_carla_running}.json`,
`reports/evidence/raw_split_validation_{cpu,cuda}.json`.

## Model choice: YOLO26n (`models/yolo26n.pt`, SHA-256 `9b09cc8b…a8ad4fef`)
Chosen for NPU operator compatibility over novelty, based on prior RNGD logs re-read in Phase 0:
`yolo11n` failed (unfused BatchNorm unsupported; fused head concat → `select_scatter` lowering error),
while a fused YOLO26n split at the raw head had compiled. This checkpoint has `end2end = False`
(one-to-many head, `reg_max = 1`, so no DFL), i.e. it needs NMS like YOLOv8.

## Pipeline split (`src/npu_ad/perception/yolo_raw.py`)
- Accelerator part `YoloRaw`: fused backbone + neck + head convs → raw `boxes (1,4,8400)` and `scores (1,80,8400)` logits.
- Host part `RawDecoder`: anchors/strides, `dist2bbox`, sigmoid, NMS (`conf 0.25`, `iou 0.7`, `max_det 300`), letterbox inverse.
- Fixed 640×640 letterbox (static shape needed by the NPU), identical for GPU and NPU.

**Validation of the split** against the unmodified ultralytics forward on the same tensor (`scripts/validate_raw_split.py`):

| Device | Decoded max abs diff | Detections (ours / ref) | Post-NMS max abs diff | Tolerance | Result |
|---|---|---|---|---|---|
| CPU | 0.0 | 1 / 1 | 0.0 | 1e-4 | PASS |
| CUDA | 2.98e-8 | 1 / 1 | 0.0 | 1e-4 | PASS |

## Benchmark protocol (`scripts/gpu_benchmark.py`)
72 dataset frames × 5 repeats = 360 measured frames, 20 warm-up frames excluded, batch 1, sequential.
`model` = CUDA-event time around the raw module (synchronized); `postprocess` = D2H + decode + NMS + rescale on CPU;
`e2e` = wall clock from BGR frame to detections. GPU util/memory/temperature sampled with NVML every 50 ms.

## Results (ms: mean / p50 / p95)

| Condition | Preprocess | H2D | Model-only | Postprocess | End-to-end | Seq. FPS | GPU util mean | Peak torch alloc |
|---|---|---|---|---|---|---|---|---|
| FP32, CARLA stopped | 1.59 / 1.50 / 2.02 | 0.39 / 0.37 / 0.53 | **13.54 / 12.64 / 18.43** | 2.15 / 2.11 / 3.03 | **17.73 / 16.85 / 22.72** | 56.0 | 24.3 % | 64.7 MiB |
| FP16, CARLA stopped | 1.60 / 1.51 / 2.02 | 0.63 / 0.63 / 0.91 | 14.99 / 13.82 / 22.23 | 2.33 / 2.32 / 3.17 | 19.62 / 18.50 / 27.30 | 50.7 | 24.9 % | 48.4 MiB |
| FP32, CARLA free-running | 2.12 / 2.15 / 2.69 | 0.48 / 0.47 / 0.63 | 21.39 / 21.35 / 22.37 | 2.76 / 2.70 / 3.72 | 26.83 / 26.80 / 28.47 | 37.0 | 99 % | 64.7 MiB |

Environment: RTX 3060 12 GB, driver 616.92, torch 2.14.1+cu126, Python 3.12.10, Windows 11.

## Interpretation (measured facts only)
- GPU utilisation was only ~24 % during the clean runs and FP16 was **not** faster than FP32. Batch-1
  eager PyTorch execution of this 2.4 M-parameter model is therefore bound by per-layer launch/dispatch
  overhead rather than GPU arithmetic. The "model-only" CUDA-event time includes those launch gaps.
  Reducing it (CUDA Graphs, TensorRT, batching) is untested here and listed as future work.
- A free-running CARLA server occupies the same GPU (99 % util, ~9.5 GB) and raised model-only latency by
  58 % (13.5 → 21.4 ms). Closed-loop scenarios use synchronous mode so the server renders only on tick.
- Classes used downstream: pedestrian {0 person}, cyclist {1 bicycle}, vehicle {2 car, 3 motorcycle,
  5 bus, 7 truck} (COCO). No road-obstacle class (cones, debris) exists in this COCO model; such obstacles
  are **not** claimed to be detected.
- Absolute detection accuracy (mAP against ground truth) was **not** measured: no labelled CARLA set was
  produced. GPU FP32 output is used as the reference for NPU agreement.

Sample visualisations: `assets/vis_*.jpg`.
