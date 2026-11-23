# Handoff — NPU Auto Driving Project 2 (2026-11-23, Claude Code)

Repo `<LOCAL_PROJECT_ROOT>`, branch `feature/portfolio-hardening` (from `feature/npu-pipeline` @ `2fdf039`).
Local commits only, nothing pushed. Machine-readable state: `task_state.json`, `task_table.md`, `failures.jsonl`.

## 1. Implemented
- Perception-driven AEB mode `mono` (`src/npu_ad/perception/mono_range.py`: flat-ground range, CIPO tracker,
  residual-gated range rate) next to `gt_oracle` and `gt_gated` (GT-assisted); speed sweep, no-hazard probes,
  settle phase, per-run world/spawn/seed metadata, replay frame dump (`scripts/run_scenarios.py`).
- CARLA integrity: `CarlaSession` restores settings on failed start/destroy; process-level exclusive server lock
  (`src/npu_ad/simulation/server_lock.py`); live tests opt-in (`NPU_AD_LIVE_CARLA=1`); quarantine tool.
- RNGD: `remote/int8_probe.py` (compiler-level INT8/FP8 probe), `npu_job.py --dtype w8` (weight-only INT8),
  `perf` (cold start, throughput, board power), `--tag`; job lock in `npu_remote.py start`.
- Benchmarks/analysis: `gpu_benchmark_ext.py` (batch, CUDA Graphs, power), `measure_link.py`, `model_size.py`,
  `replay_aeb.py` (open-loop GPU vs NPU AEB), `validate_raw_split.py --all`, `render_v3_tables.py`, `make_plots.py`.
- Structured JSON logging (`src/npu_ad/logutil.py`), `requirements-lock-windows.txt`.
- Docs: README, `docs/{architecture,npu_compilation,quantization_limitations,gpu_npu_comparison,aeb_evaluation,
  troubleshooting,portfolio_summary_en,portfolio_summary_ko}.md`, `reports/{technical_audit,final_report,scenario_tables_v3}.md`.

## 2. Actually tested
- pytest: 32 passed (incl. live CARLA test with exclusive server); 31 passed + 1 skipped without live flag; ruff clean.
- Server lock live demo: second process refused (`ServerBusy`).
- CARLA v3: 156 matrix runs + 23 follow-up runs (6 re-runs of quarantined, 10 remote-latency, 7 replay dumps).
- RNGD: INT8 probe (2 rounds), W8 compile + bench, perf ×3 (after one failed attempt), health check, replay bench (738 frames).
- GPU: extended benchmark with CARLA stopped; determinism (738 frames twice, diff 0.0).

## 3. Measured GPU / NPU results (batch 1; n and spread in `docs/gpu_npu_comparison.md`)
| | mean (ms) | p99 (ms) | throughput | board power |
|---|---|---|---|---|
| GPU eager FP32 | 13.54 (n=360) | 25.19 | 72 fps | 52 W |
| GPU CUDA Graph FP32 | 2.60 (n=300, std 0.18) | 3.23 | 387 fps | 136 W |
| RNGD FP32 | 2.107 (n=216, std 0.007) | 2.125 | 470 fps | 53 W (idle 39 W) |
| RNGD BF16 | 1.85–2.00 | 1.91–2.07 | 543 fps | 54 W |
| RNGD W8 weight-only | 2.16–2.35 | 2.18–2.68 | 446 fps | 51 W |

Transport desktop → RNGD host: 121.2 ms mean / 124.8 ms p95 per raw 1.23 MB frame (n=30).
Control delays used: GPU p95 22.7 ms, NPU on-vehicle p95 9.85 ms, NPU remote p95 134.7 ms.
Agreement vs GPU (54 frames): FP32 recall 0.988 / precision 0.992; BF16 0.972 / 0.992; W8 0.957 / 0.980.

## 4. CARLA AEB results (30 km/h unless noted)
- AEB off: every hazard scenario collides. GT oracle 0/25, GT-assisted 0/30, perception-driven 0/30 collisions
  over 0–200 ms delays (incl. 134.7 ms remote-NPU); 0 false brakes in 45 no-hazard runs.
- Perception range MAE 0.11–0.79 m; closing-speed MAE 1.2–3.2 m/s (p95 up to 14 m/s beyond 30 m) → early or
  premature stops (static obstacle at 200 ms stopped 23.4 m short).
- Speed sweep: pedestrian 40 km/h collides in all modes; 50 km/h AEB-induced side collision (GT-assisted).
- Open-loop replay: GPU and NPU give the identical first-brake decision in 7/7 runs.
- Repeats: same outcome, logs not bit-identical; GPU is deterministic, so variation comes from rendered frames.

## 5. INT8 conclusion
INT8 compute is **BLOCKED** in furiosa-torch 2026.4.0: int8 conv compiles but accumulates with int8 wrap-around
(100 % of outputs), int32-accumulating conv (`O117 not yet supported`) and `aten::_int_mm` (not importable) fail.
Weight-only INT8 (W8) compiles and runs: −49 % weight bytes vs BF16, not faster, recall 0.957. BF16 is the
recommended precision. Nothing is claimed as INT8 inference.

## 6. Modified files (this session)
67+ files changed vs `feature/npu-pipeline` (`git diff --stat feature/npu-pipeline..HEAD`). Key: `src/npu_ad/
{perception/mono_range.py, simulation/server_lock.py, simulation/carla_env.py, logutil.py}`, `remote/{npu_job.py,
int8_probe.py}`, `scripts/{run_scenarios.py, npu_remote.py, replay_aeb.py, gpu_benchmark_ext.py, measure_link.py,
model_size.py, quarantine_runs.py, render_v3_tables.py, make_plots.py, validate_raw_split.py, compare_gpu_npu.py,
make_latency_conditions.py, task_state.py, run_v3_matrix.sh, run_v3_followup.sh, npu_chain.sh}`, `tests/
{conftest.py, test_mono_range.py, test_server_lock.py, test_logutil.py, test_carla_env_fake.py}`, docs, reports, evidence, plots.

## 7. Local commits (oldest → newest)
`1678298 bf872a2 e397a89 2c655e7 ff438ff c422714 9090bc6 607a0e6 daff8e0 0b1a4d2 6402f5b dea5bf3 f142c37 7c605c3
8254763 a09600a 5258a82 5782dca 3ade696 1da65a5` + the commit that adds this file (`git log -1`).

## 8. Failed / blocked tasks
- BLOCKED: INT8 compute (SDK); NPU-in-the-loop (transport 121 ms/frame).
- FAILED (not retried): pipelined NPU execution with 8 in-flight submissions (HAL timeout, PXI-122).
- NOT_RUN: NPU batch > 1; GT-labelled mAP; multiple seeds/maps; TensorRT (needs approval).
- Incident handled: live pytest during the matrix → 6 runs quarantined and re-run (1 differed).

## 9. Remaining technical risks
- Perception-driven range rate is noisy; results depend on it (premature stops). No Kalman filter yet.
- AEB latch has no escape / "target will clear" logic (50 km/h side collision).
- Single seed/map per cell; rendering nondeterminism makes repeats differ slightly.
- Shared RNGD host: timings vary up to ~0.2 ms between protocols; compile times load-dependent.
- Scenario start transient (speed dip after hand-brake release) affects all modes equally.
- 3 GB of replay files left in `~/npu_auto_driving_p2` on the shared host (deletion awaits approval).

## 10. Exact next recommended action
1. Review `README.md` and `docs/gpu_npu_comparison.md`; decide on publishing (push needs approval).
2. Approve or edit the KB handoff draft, then publish it with
   `python -m lab_handoff publish <draft> --approve --approved-by <you>` from `<AGENT_LAB_ROOT>`.
3. Next engineering step: add a Kalman range-rate filter and a "target will clear" rule to the AEB, then re-run
   `bash scripts/run_v3_matrix.sh` into a new results file (do not overwrite v3).
