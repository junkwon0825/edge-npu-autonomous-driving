# Final Report — NPU Auto Driving Project 2 (session 2026-11-22)

## Executive summary
The CARLA → YOLO26n → GPU / Furiosa RNGD → AEB pipeline runs end to end with recorded evidence.
The YOLO26n detector was compiled for and executed on a real Furiosa RNGD NPU (2.11 ms mean compute per
640×640 frame) and agrees with the RTX 3060 FP32 reference on held-out CARLA frames (recall 0.988,
precision 0.992, mean IoU 0.991). A perception-gated AEB prototype avoided collision in 25/25 hazard runs
for 0–200 ms latency, while 5/5 hazard runs without AEB collided. INT8 quantization was not possible with
the installed SDK. Gates A–F pass with stated caveats; Gate G is complete for this scope. The project is
**not COMPLETE** in the master-prompt sense because quantization (Phase 4) is blocked and the NPU is not in
the control loop.

## Phase table
| Phase | Status | Evidence |
|---|---|---|
| 0 Environment audit | PASS | `reports/environment_audit.md` |
| 1 CARLA baseline (Gate B) | PASS | `reports/carla_baseline.md`, `evidence/capture_smoke.json`, live test |
| 2 GPU baseline (Gates A, C) | PASS | `reports/gpu_baseline.md`, `evidence/gpu_benchmark_*.json` |
| 3 NPU compile + runtime (Gate D) | PASS | `reports/npu_compilation.md`, `evidence/npu_compile.json`, `evidence/npu_bench.json` |
| 4 Quantization | PARTIAL — INT8 BLOCKED (no vision quantization API in SDK 2026.4.0); BF16 compiled and measured (recall vs GPU 0.972) | `reports/npu_compilation.md` § Quantization |
| 5 GPU vs NPU (Gate E) | PASS (author-set tolerances) | `reports/benchmark_gpu_vs_npu.md`, `evidence/gpu_vs_npu.json` |
| 6 Integration (Gate F) | PASS (simulation; NPU latency injected) | `reports/scenario_evaluation.md`, `evidence/scenarios_v{1,2}.json` |
| 7 Portfolio (Gate G) | PASS for this scope | `README.md`, `docs/`, `reports/` |
| Orchestration | PARTIAL | Kiro review ran for real via `agent_lab`; implementation was done by Claude alone, no parallel multi-agent writes |

## Implementation evidence
- Repo `<LOCAL_PROJECT_ROOT>`, local branch `feature/npu-pipeline` (Git initialised this session; never pushed).
- Tests: `python -m pytest` → 21 passed (incl. live CARLA test); `ruff check` / `ruff format --check` clean.
- Durable state: `reports/agent_runs/{task_state.json,latest_status.md,failures.jsonl}`.
- NPU host work dir: `~/npu_auto_driving_p2` (job scripts, status JSON, raw outputs).

## Environment
Windows 11, RTX 3060 12 GB, driver 616.92, Python 3.12.10, torch 2.14.1+cu126, ultralytics 8.4.202,
CARLA 0.9.16. RNGD host: furiosa-torch 2026.4.0, firmware 2026.4.0, Python 3.11.18, torch 2.10.0+cpu.

## Benchmarks (measured)
| Metric | GPU RTX 3060 (eager FP32) | RNGD (AOT compiled) |
|---|---|---|
| Model-only mean / p95 | 13.54 / 18.43 ms | 2.107 / 2.119 ms |
| Device-side incl. transfers, mean / p95 | 0.39 H2D + 13.54 model | 3.81 / 4.80 ms |
| End-to-end p95 | 22.72 ms (measured) | 9.85 ms (composed: device p95 + host pre/post p95) |
| Detection agreement vs GPU (eval) | reference | recall 0.988, precision 0.992, IoU 0.991 |
Not a like-for-like hardware comparison (eager vs compiled; different hosts). Power: N/A.

## Failures
| Failure | Root cause | Resolution |
|---|---|---|
| Remote start blocked ssh | backgrounded and-list | fixed in `npu_remote.py` |
| NPU bench `TypeError current_stream()` | API signature assumed | fixed; 1 retry |
| GPU benchmark `TypeError: 'Event' object is not callable` | shadowed `Thread._stop` | renamed attribute |
| Full `dryrun` too slow on shared host | per-node compile probe | stopped; AOT compile used instead |
| AEB v1 stop-and-go creep | brake released when TTC recovered | v2 hysteresis; both result sets kept |
| Orchestrator marked Kiro review `failed` | evidence paths resolved inside agent worktree only | findings used after manual verification; orchestrator limitation recorded |

## Approval requests (not performed)
1. Publishing: pushing `feature/npu-pipeline` to a GitHub remote (no remote configured).
2. Knowledge-base update: a handoff entry in `jun-ai-knowledge/handoffs/` (draft text in the session report).
3. Optional: installing TensorRT / ONNX tooling locally for a compiled-GPU comparison (new dependencies).

## Next commands
```powershell
cd <LOCAL_PROJECT_ROOT>; $env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\task_state.py show
.\.venv\Scripts\python.exe scripts\npu_remote.py status bench_bf16
.\.venv\Scripts\python.exe scripts\compare_gpu_npu.py --dtype bf16
.\.venv\Scripts\python.exe scripts\render_scenario_tables.py
```
Then: labelled CARLA set for GT mAP; repeat scenarios over several seeds/speeds; NPU-in-the-loop via a
streaming client on the RNGD host; CUDA Graphs/TensorRT GPU path.

## Portfolio readiness
Can be shown truthfully: the reproducible CARLA pipeline and tests; the RNGD compile/run with failure
analysis; GPU-vs-NPU agreement numbers; closed-loop AEB results including the v1 defect and fix.
Must not be claimed: INT8 quantization, NPU-in-the-loop driving, mAP, power efficiency, real-vehicle results,
"NPU is N× faster than GPU" as a hardware statement.
