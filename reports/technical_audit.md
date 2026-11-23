# Technical Audit (Phase 1, session 2026-11-23)

Three independent read-only reviews of `feature/npu-pipeline` @ `2fdf039`, then verification of every
finding against the code and, where possible, by execution. Raw reviewer outputs:
`outputs/audit/codex_arch_review.md`, `outputs/audit/kiro_code_test_review.md` (git-ignored, local).

| Reviewer | Scope | How it ran |
|---|---|---|
| A — Codex (`codex exec -s read-only -C carla_project`) | architecture | the `agent_lab` orchestrator run `wf-20261123-220950-785e15` failed for Codex (its sandbox only reads its own worktree), so the same prompt was re-run directly with a read-only sandbox in the project dir |
| B — Kiro (`agent_lab.cli_agents.workflow_cli`, read mode) | code and tests | run `wf-20261123-220950-785e15`; marked `failed` by the orchestrator only because evidence paths are resolved inside Kiro's worktree (known limitation, see `review_findings.md`) |
| C — Claude (author) | experiment design, measured evidence | re-read every `reports/evidence/*.json` and the scenario logs |

Note: Codex read the tree while the v3 runner was being edited; its two CRITICAL findings
(`EGO_TARGET_MPS` undefined, `lane_follow(…, carla)`) and the missing `--mode`/FP-scenario branches refer
to that transient state and were already fixed in commit `1678298` before any run.

## Findings and disposition

| # | Source | Finding | Verdict | Action |
|---|---|---|---|---|
| A1 | Codex | `CarlaSession.__enter__` failure after enabling sync mode leaves the server in sync mode; an exception in `apply_batch_sync` skips settings restore | **Confirmed** | fixed (`carla_env.py`: try/except in `__enter__`, destroy wrapped); tests `test_failed_enter_restores_settings`, `test_destroy_error_still_restores_settings` |
| A2 | Codex | `gt_gated` "perception" = GT centre inside *any* same-class box; a box of another object can confirm the target | **Confirmed (design limitation)** | documented; new `mono` mode uses no GT at all; `gt_gated` renamed "GT-assisted" everywhere |
| A3 | Codex | no common GPU/NPU detector interface; NPU is offline only, NPU conditions are latency injection | **Confirmed** | NPU stays offline by measurement: a raw frame costs 121 ms over SSH (`evidence/npu_link.json`); added open-loop GPU-vs-NPU replay of the AEB chain (`scripts/replay_aeb.py`) and a `npu_remote_p95` latency condition |
| A4 | Codex | collision callbacks are async; a contact in the last tick may be missed | **Plausible** | 3 extra ticks before teardown in `run_scenarios.py` |
| A5 | Codex | no lock on remote jobs; two starts overwrite status/output files | **Confirmed** | `npu_remote.py start` refuses with `BUSY` while a job runs (verified live: second start → exit 3) |
| A6 | Codex | lead vehicle `enable_constant_velocity` uses world X | **Rejected** | CARLA 0.9.16 documents the vector in the actor's local frame; logged lateral offset of the lead stays 0.07–0.20 m (same finding rejected in `review_findings.md`) |
| B1 | Kiro | duplicate `camera.listen()` | **Not present** | one `listen` per sensor (`carla_env.py` `attach_rgb`, `attach_collision`) |
| B2 | Kiro | letterbox rounding not inverted in `scale_boxes` | **Rejected** | `scale_boxes` subtracts the actual top/left pad, identical to ultralytics; full-dataset check below shows 348/348 matches |
| B3 | Kiro | NVML GPU utilisation averaged over the whole loop incl. host gaps | **Confirmed (reporting caveat)** | stated in `docs/gpu_npu_comparison.md`: utilisation is device-wide over the run, not kernel occupancy |
| B4 | Kiro | `compare_gpu_npu.py` lacks a length check | **Low** | tensor subtraction would raise on mismatch; replay script checks names and shapes explicitly |
| B5 | Kiro | `get_image` aborts on a newer frame instead of resyncing | **Rejected (by design)** | fail-fast is intended in synchronous mode; a newer frame means the loop is broken and the run must not continue silently |
| B6 | Kiro | missing tests: fp16 path, letterbox round-trip, NPU job logic, collision payload | **Partly addressed** | added mono-range, cleanup tests; NPU job logic remains tested only on the host by execution |
| C1 | Claude | raw-split validation used **1 image** and compared fused-vs-fused (`raw.model` is the fused model), so "max diff 0.0" did not validate Conv+BN fusion | **Confirmed** | `validate_raw_split.py --all`: 72 frames vs. the *unfused* original: 348/348 detections, decoded max diff 0.017 px, conf Δ ≤ 2.9e-6 (`evidence/raw_split_validation_all_cpu.json`) |
| C2 | Claude | v2 `low_light` and `rain` results are bit-identical to their clear-weather counterparts: in `gt_gated` mode braking timing is set by GT TTC; perception only gates. The weather "robustness" claim was therefore vacuous | **Confirmed** | v3 `mono` mode makes timing depend on perception; weather is now a real variable |
| C3 | Claude | perception runs every 10 ms simulation tick (100 Hz camera) | **Confirmed (modelling choice)** | documented; latency is modelled as actuation delay only |
| C4 | Claude | each scenario/condition ran once; determinism not shown | **Confirmed** | v3 repeats a subset (`tag=rep2`) and compares logs |
| C5 | Claude | `npu_p95` (9.85 ms) excludes the transport to the remote NPU host | **Confirmed** | measured link; `npu_remote_p95` = 134.7 ms added |
| C6 | Claude | NPU-vs-GPU agreement is not accuracy (no GT labels) | **Confirmed** | unchanged; stated wherever the numbers appear |
| C7 | Claude | "INT8 BLOCKED" was concluded from API absence only, without asking the compiler | **Confirmed** | compiler probe (`remote/int8_probe.py`): int8 conv lowers but wraps modulo 256; int32-accumulating conv and `_int_mm` do not compile → see `docs/quantization_limitations.md` |
| C8 | Claude | the scenario spawn transient (vehicles dropped from elevated spawn points) overlaps t=0 | **Confirmed** | 1 s settle phase in v3; first mono smoke run had measured camera height 2.30 m instead of 1.70 m because of it |

## Verified-unchanged claims
- CARLA capture 3/3 × 100 frames, 0 orphaned actors (`evidence/capture_smoke.json`).
- GPU 13.54 ms model / 17.73 ms e2e mean (`evidence/gpu_benchmark_fp32_clean.json`, 360 samples).
- RNGD FP32 compute 2.107 ms mean over 216 samples (`evidence/npu_bench.json`), BF16 2.000 ms.
- NPU vs GPU detection agreement (eval split) recall 0.988 / precision 0.992 (`evidence/gpu_vs_npu.json`).
