# Portfolio summary (English)

**CARLA → YOLO26n → RTX 3060 / Furiosa RNGD NPU → perception-driven AEB** — personal project,
simulation only. Evidence for every number: `reports/evidence/`.

## Verified implementation
- Reproducible CARLA 0.9.16 pipeline: synchronous mode, frame-matched sensor queue, tracked actor
  teardown that restores the server even when start-up or destroy fails, process-level exclusive
  server lock, resumable atomic result files, structured JSON logs. 33 unit tests pass (+1 opt-in live CARLA test), ruff clean.
- NPU deployment: diagnosed earlier RNGD compile failures (unfused BatchNorm, head concat lowering),
  redesigned the boundary (fused model split at the raw head, decode/NMS on host), validated against
  the unfused original on 72/72 frames (348/348 detections), AOT-compiled FP32, BF16 and W8 variants
  with `furiosa-torch` 2026.4.0 and ran them on `furiosa:0`.
- Three AEB input modes kept strictly apart: GT oracle, GT-assisted (GT range gated by detection) and
  perception-driven (flat-ground monocular range + closest-in-path tracker; ground truth only for
  scoring).

## Measured results
- Model-only, batch 1: GPU eager 13.5 ms, **GPU CUDA Graph 2.60 ms**, **RNGD FP32 2.11 ms**, BF16 1.85–2.0 ms.
  RNGD jitter std 0.007 ms. Board energy/frame ≈ 113 mJ (RNGD) vs ≈ 351 mJ (GPU CUDA Graph).
- NPU vs GPU detection agreement: recall 0.988 / precision 0.992 (FP32); identical first-brake
  decisions in 7/7 open-loop AEB replays on 738 identical frames.
- Closed-loop AEB at 30 km/h, 0–200 ms latency: no AEB → every hazard collides; GT-assisted 0/30 and
  perception-driven 0/30 collisions; 0 false brakes in 45 no-hazard runs. Perception range error MAE
  0.11–0.79 m; closing-speed MAE 1.2–3.2 m/s (the weak link).

## Negative results reported as such
- INT8 compute blocked by the toolchain (proved with a compiler probe, not assumed); INT8 weight-only
  saves 49 % weight memory but is not faster and loses accuracy (recall 0.957).
- Remote NPU from the desktop costs 121 ms per raw frame → not real-time for AEB.
- Speed sweep: pedestrian crossing at 40 km/h collides in all modes; at 50 km/h the AEB itself causes
  a side collision (no "target will clear" logic) — documented, not tuned away.
- Experiment integrity: a live test touched the shared simulator during a matrix; 6 overlapping runs
  were quarantined and re-run (1 had changed), and a server lock now prevents it.

## Limitations / future work
Simulation only, one map/seed, no GT labels (agreement, not mAP), NPU not in the closed loop, NPU
batch > 1 not compiled. Next: labelled set for mAP, co-located NPU, AEB path prediction, Kalman range
rate, TensorRT comparison.
