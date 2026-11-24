# V3 report: multi-model NPU perception, lane keeping, fusion (STATUS: IN PROGRESS)

Branch `feature/v3-multimodel-npu` (from `v2.0.0-rc1`). Local commits only; nothing pushed.

**Status.** All RNGD-dependent work, the perception and control implementation, the tests and the docs
are done. The **closed-loop V3 matrix and the V3 demo video are not run yet.** The CARLA server was
stopped by the system because memory was critically low (2026-11-24). It is restarted only on request,
because memory may still be short. Smoke runs (Mode A lane keeping, 3 routes) were done before that.

## Done (with evidence)
| Area | Result | Evidence |
|---|---|---|
| Phase 0 V2 recovery | V2 tag present; 62 tests passing at V2; V2 limitations carried over (pedestrian/cyclist tracking, lateral velocity from ego yaw, remote path near-real-time) | `reports/v2_final_report.md` |
| Lane-segmentation port | 7-step porting record; final U-Net (3×3 s2 down, conv + PixelShuffle up) runs on RNGD | `docs/v3_lane_npu_porting.md`, `v3_npu_lane_port_*.json` |
| Lane model training | CARLA dataset of 2,700 frames; val IoU 0.817; **held-out Town03 IoU 0.727** (GPU); RNGD BF16 0.664 vs GPU 0.688 on the same 150 frames | `v3_lane_training.json`, `v3_npu_multimodel_bench.json` |
| Operator database | 74 generated rows (YOLO, U-Net, LR-ASPP, ResNet18, MobileNetV3, tiny ViT, transformer ops, pooling sweep, quantization) | `docs/npu_operator_compatibility.md`, `docs/npu_graph_lowering_notes.md` |
| Model complexity | YOLO26s fails (64 KiB weight-slice limit); after the exact SplitConv rewrite it runs at 4.67 ms vs YOLO26n 2.0 ms | `v3_npu_compile_yolo26s_bf16.json`, `v3_npu_split_conv.json` |
| Non-driving models | MobileNetV3-S 0.50 ms, tiny ViT 0.44–0.48 ms run; ResNet18 fails even with output splits (input-channel limit) | `v3_npu_model_lab*.json` |
| Multi-model runtime | `npu_server_v3` with detection on `furiosa:0` and lane on `furiosa:1`; detection + lane 61.2 ms vs detection only 55.0 ms remote; parallel section 3.4 ms | `docs/v3_multimodel_runtime.md` |
| Scheduling A–D | separate PEs best under stress (detection 400 Hz, 0 drops); same-PE workers worse than sequential; priority starves lane | `v3_npu_multimodel.json` |
| Failure engineering | stall → GPU in 0.54 s, crash → GPU in 0.11 s, 0 lost frames, logged, recovered; bad shape / load failure / missing device / both down handled | `v3_failure_engineering.json` |
| Lane geometry, LKA, arbitration | implemented and unit-tested; smoke lane keeping (Mode A, 30 km/h): lateral RMSE 0.008 / 0.014 / 0.108 m (straight / mild / strong), 0 departures | `docs/v3_perception_control.md`, `v3_smoke_lka_modeA.json` |
| AEB on curves | path-relative corridor + ego yaw-rate compensation (unit-tested) | same |
| Radar fusion stage 1 | association + Kalman fusion implemented and unit-tested | same |
| Scene state, adaptive scheduler | implemented and unit-tested | `src/npu_ad/scene.py`, `runtime/multimodel.py` |
| Real-time budget | capture → result: GPU ≈ 22 / 28 ms, RNGD remote (both models) 61 / 64 ms mean / p95 | `docs/v3_multimodel_runtime.md` |
| Tests | 77 passed, 1 skipped (live CARLA opt-in); ruff clean | — |

## Pending (needs CARLA)
1. **V3 matrix** (`scripts/v3_run_scenarios.py`, about 100 runs):
   - Modes A/B/C/D on lane keeping, AEB, combined (curve + obstacle, curve + lead braking, pedestrian
     while cornering, degraded lane) and fault scenarios, at several speeds
   - camera-only vs camera + radar
   - fixed vs adaptive scheduler
2. **V3 demo video** (Mode D overlay: detections, lane, tracks, TTC, AEB, LKA, backends, frame ages).
3. Closed-loop numbers for the README, the key results and this report.

## Not done / limitations already known
- `road_condition` is not perceived.
- No traffic-sign/light model (optional third model).
- No LiDAR (optional stage 2).
- ResNet18 not ported.
- The Town03 test set contains no night frames.
- The remote NPU loop is near-real-time.
- The controller was developed and evaluated on the same scenario family.
- Frozen V1 evidence (local commits) contains the remote container hostname; early V2 commits contain
  the remote home path. A decision is needed before any publication.
