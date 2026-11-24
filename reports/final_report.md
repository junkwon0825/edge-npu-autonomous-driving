# Final Report — NPU Auto Driving Project 2

> **V1 release (2026-11-24):** see `reports/v1_final_report.md` and `README.md`. This report covers the
> sessions before V1 and is kept unchanged below.

Sessions 2026-11-22 (pipeline, first NPU results, branch `feature/npu-pipeline`) and 2026-11-23
(audit and portfolio hardening, branch `feature/portfolio-hardening`). Local commits only; nothing
pushed.

## Executive summary
The CARLA → YOLO26n → GPU / Furiosa RNGD → AEB pipeline runs end to end with recorded evidence.
The detector runs on a real RNGD (2.11 ms FP32 / 1.85–2.0 ms BF16) and agrees with the GPU
(recall 0.988, precision 0.992); in an open-loop replay the NPU never changed an AEB decision. A fair
GPU baseline (CUDA Graphs, 2.60 ms) shows that most of the earlier 6× gap was eager-PyTorch overhead.
Perception-driven AEB (no ground truth in the controller) avoided collision in 30/30 hazard runs at
30 km/h and 0–200 ms latency, with documented failure cases at 40–50 km/h. INT8 compute is blocked
by the toolchain (proved by a compiler probe); weight-only INT8 works but is not faster. The project
is **not COMPLETE**: the NPU is not in the closed loop (remote transport 121 ms/frame) and there is
no ground-truth accuracy (mAP).

## Phase table
| Phase | Status | Evidence |
|---|---|---|
| 0 State recovery / environment | PASS | `reports/environment_audit.md`, Git history |
| 1 Technical audit (Codex, Kiro, Claude) | PASS | `reports/technical_audit.md` |
| 2 GPU vs NPU validation | PASS (NPU batch > 1 NOT_RUN; pipelined NPU execution FAILED on device) | `docs/gpu_npu_comparison.md`, `evidence/gpu_benchmark_ext.json`, `evidence/npu_perf*.json`, `evidence/npu_link.json` |
| 3 BF16 / INT8 investigation | PARTIAL — INT8 compute BLOCKED (SDK), W8 weight-only TESTED, BF16 TESTED | `docs/quantization_limitations.md`, `evidence/npu_int8_probe*.json` |
| 4 Closed-loop AEB | PASS (simulation; NPU-in-the-loop BLOCKED by transport) | `docs/aeb_evaluation.md`, `evidence/scenarios_v3.json`, `evidence/replay_npu_vs_gpu.json` |
| 5 Software engineering | PASS | 32 tests (31 unit + 1 opt-in live), ruff clean, server lock, JSON logs, lock file |
| 6 Portfolio | PASS | `README.md`, `docs/*`, `assets/plots/*` |
| 7 Final verification | PASS | see below |

## Final verification (2026-11-23)
- `pytest`: 32 passed with `NPU_AD_LIVE_CARLA=1` and a CARLA server owned by the test only; 31 passed
  + 1 skipped without it. `ruff check` and `ruff format --check`: clean.
- Server lock demonstrated live: a second process got `ServerBusy: CARLA localhost:2000 is owned by pid …`.
- GPU inference determinism: 738 frames twice, max abs diff 0.0.
- NPU health check before re-running after the device error: PASS.
- Benchmark calculations re-derived from raw timing arrays (mean/p50/p95/p99/std in the docs match
  the evidence files); plots regenerated only from evidence files.
- CARLA server stopped at the end; no NPU job of this project running.

## Failures and incidents
| Failure | Root cause | Resolution |
|---|---|---|
| Live pytest connected to the simulator during the v3 matrix | live test ran by default when a server was up | 6 overlapping runs quarantined and re-run (1 differed: min gap 4.78 → 9.04 m); live tests opt-in; process-level server lock |
| NPU `perf` with 8 in-flight executions | HAL sync timeout, then `PXI-122 Duplicate submission` in following processes | burst default 1; device health check PASS; partial results saved per phase |
| Codex could not read the project through the orchestrator | Codex sandbox limited to its worktree | ran `codex exec -s read-only -C <project>` directly |
| Monocular range 35 % too long | camera height calibrated while the car was still falling after spawn | 1 s settle phase |
| Scenario runner truncated during an edit | shell command opened the target file before the source failed to open | restored from Git, patch applied from a script file |
| Earlier session failures | see 2026-11-22 entries in `reports/agent_runs/failures.jsonl` | fixed |

## Approval requests (not performed)
1. Push `feature/npu-pipeline` / `feature/portfolio-hardening` to a GitHub remote (none configured).
2. Publish the knowledge-base handoff (draft prepared, see `reports/agent_runs/latest_status.md`).
3. Optional TensorRT/ONNX installation for a compiled-GPU comparison beyond CUDA Graphs.
4. Deleting the large replay files this project left on the shared RNGD host
   (`~/npu_auto_driving_p2/replay_inputs.pt` 0.9 GB, `npu_outputs_fp32_replay.pt` 2.1 GB).

## Must not be claimed
INT8-accelerated inference; NPU-in-the-loop driving; mAP or accuracy (only GPU agreement); real-vehicle
results; "NPU is 6× faster than GPU" (that compares against eager PyTorch); a false-positive rate in
traffic (only 45 simple no-hazard runs).
