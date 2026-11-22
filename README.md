# NPU Auto Driving Project 2 — CARLA → YOLO → GPU / Furiosa RNGD → AEB

Personal research prototype (not a road-ready system): a reproducible camera-perception pipeline in
CARLA 0.9.16, a YOLO26n detector executed on an RTX 3060 and on a Furiosa **RNGD** NPU, and a
conservative, perception-gated emergency-braking (AEB) controller evaluated in closed-loop scenarios.

Every number in this repository comes from a recorded run under `reports/evidence/`.
Status labels: **TESTED** (executed with evidence) · **IMPLEMENTED** (code exists, not fully exercised) ·
**PARTIAL** · **BLOCKED** · **PLANNED**.

## Results at a glance (measured 2026-11-22)

| Area | Result | Status |
|---|---|---|
| CARLA capture | 3/3 runs × 100 contiguous frames, 0 orphaned actors | TESTED |
| Raw-head split vs ultralytics | max abs diff 0.0 (CPU), 2.98e-8 (CUDA) | TESTED |
| GPU (RTX 3060, eager FP32, batch 1) | model 13.54 ms mean, e2e 17.73 ms mean / 22.72 ms p95, 56 FPS | TESTED |
| RNGD compile (furiosa-torch 2026.4.0) | AOT `CompileModule` in 1901.9 s | TESTED |
| RNGD inference (`furiosa:0`) | compute 2.107 ms mean; H2D+compute+D2H 3.81 ms mean | TESTED |
| NPU vs GPU detections (54 eval frames) | recall 0.988, precision 0.992, mean IoU 0.991 | TESTED (agreement, not GT mAP) |
| INT8 quantization | no vision quantization API in installed SDK | BLOCKED |
| BF16 on RNGD | compute 2.000 ms; recall vs GPU 0.972 | TESTED |
| Closed-loop AEB, 5 hazard scenarios | AEB off: 5/5 collide · AEB on: 0/25 collide for 0–200 ms delay | TESTED (simulation) |
| Latency effect (pedestrian crossing) | min gap 2.19 → 2.05 → 1.89 → 1.36 → 0.53 m at 0/10/30/100/200 ms | TESTED (NPU latency injected) |
| NPU-in-the-loop driving, GT mAP, power | not done | PLANNED |

The GPU/NPU latency ratio is not a hardware comparison: the GPU path is eager PyTorch, the NPU path an
AOT-compiled graph on a different (shared, remote) host.

## Architecture

```text
 CARLA 0.9.16 (sync, dt fixed)            host (Windows, RTX 3060)                     remote RNGD host
 ┌──────────────────────┐   BGRA    ┌───────────────────────────────────────┐
 │ ego + RGB camera     │──frame───▶│ letterbox 640² → tensor (shared code) │──inputs──▶ furiosa:0
 │ NPC vehicles/walkers │           │   ├─ CUDA: YoloRaw (fused, raw head)  │           CompileModule
 │ collision sensor     │           │   └─ decode + NMS on CPU (shared)     │◀─raw out── (YoloRaw)
 └─────────▲────────────┘           │ GT association (projected centre ∈ box)
           │ VehicleControl         │ AEB: TTC/gap from CARLA ground truth │
           └────────────────────────│ actuation delay = measured latency   │
                                    └───────────────────────────────────────┘
```

- `src/npu_ad/simulation/` — `CarlaSession` (sync mode, frame-matched sensor queue, tracked teardown),
  seeded traffic, ground-truth geometry and camera projection.
- `src/npu_ad/perception/` — `YoloRaw` (accelerator part) + `RawDecoder` (host part), GPU runner with
  CUDA-event timing. The same `yolo_raw.py` file is copied to the NPU host.
- `src/npu_ad/control/aeb.py` — TTC/gap AEB that only brakes on perception-confirmed targets; `DelayLine`.
- `src/npu_ad/evaluation/metrics.py` — latency statistics, detection agreement.
- `remote/npu_job.py` — runs on the RNGD host (`dryrun`, `compile`, `bench`).
- `scripts/` — runnable steps (below). `reports/` — results. `reports/agent_runs/` — durable task state.

## Reproduce

```powershell
# 0. environment (existing venv: Python 3.12, torch 2.14.1+cu126, carla 0.9.16, ultralytics 8.4.202)
cd <LOCAL_PROJECT_ROOT>
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q              # unit tests; live CARLA test runs if a server is up

# 1. CARLA server (separate terminal)
C:\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2000

# 2. Phase 1: capture smoke test and seeded dataset
.\.venv\Scripts\python.exe scripts\capture_smoke.py --runs 3 --frames 100
.\.venv\Scripts\python.exe scripts\collect_dataset.py

# 3. Phase 2: validate the raw-head split, GPU baseline (stop the CARLA server for the clean run)
.\.venv\Scripts\python.exe scripts\validate_raw_split.py --device cuda
.\.venv\Scripts\python.exe scripts\gpu_benchmark.py --tag fp32_clean --export-npu-inputs

# 4. Phase 3–5: RNGD host via an existing ~/.ssh/config alias (default "furiosa-npu")
.\.venv\Scripts\python.exe scripts\npu_remote.py sync
.\.venv\Scripts\python.exe scripts\npu_remote.py start compile
.\.venv\Scripts\python.exe scripts\npu_remote.py put outputs\npu\inputs.pt
.\.venv\Scripts\python.exe scripts\npu_remote.py start bench
.\.venv\Scripts\python.exe scripts\npu_remote.py status bench
.\.venv\Scripts\python.exe scripts\npu_remote.py fetch npu_outputs_fp32.pt outputs\npu
.\.venv\Scripts\python.exe scripts\compare_gpu_npu.py

# 5. Phase 6: closed-loop scenarios (CARLA server running)
.\.venv\Scripts\python.exe scripts\make_latency_conditions.py
.\.venv\Scripts\python.exe scripts\run_scenarios.py
```

Weights (`models/yolo26n.pt`, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`) and
generated data (`outputs/`) are not committed.

## Reports
- `reports/environment_audit.md` — measured environment, historical issues re-checked
- `reports/carla_baseline.md` — Phase 1 reproducibility
- `reports/gpu_baseline.md` — Phase 2 GPU baseline
- `reports/npu_compilation.md` — Phase 3/4 RNGD compilation, quantization status, failures
- `reports/benchmark_gpu_vs_npu.md` — Phase 5
- `reports/scenario_evaluation.md` — Phase 6 closed-loop AEB
- `reports/review_findings.md` — independent review and verification
- `reports/final_report.md` — phase table, approvals, next steps
- `docs/troubleshooting.md`, `docs/portfolio_summary.md` (KO/EN)
