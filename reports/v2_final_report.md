# V2 Final Report: frame-age-aware NPU perception runtime and robust AEB

Release candidate `v2.0.0-rc1` (local tag), branch `feature/v2-npu-closed-loop` (from `v1.0.0-rc1`),
2026-11-24. Local commits only; nothing pushed. Evidence index: `reports/v2/evidence_manifest.json`
(SHA-256 of all V2 evidence, plots, demo assets and docs). V1 evidence was not modified (V1 manifest:
46/46 files match).

**V2 question.** How should an NPU perception runtime be designed so that low model latency becomes
low and predictable control latency?

**Answer.** In this setup, the 2 ms NPU kernel was 1 % of the V1-equivalent remote path (203 ms) and
1–3 % of the V2 path. The system-level losses were:
- network transport (bandwidth)
- remote-host CPU contention
- treating stale results as current

V2 fixed what software could fix:
- transport 203 → 60 ms, lossless
- a latest-frame queue
- age-gated results with prediction to the control time
- explicit failover

It then quantified the effect on closed-loop braking. The remote NPU loop is now near-real-time and
collision-free in the tested cases, but not real-time.

## 1. V2 architecture
`docs/v2_architecture.md`, `assets/diagrams/v2_architecture.png`:
- unified `InferenceBackend`: GPU / remote RNGD / replay / mock, plus a failover wrapper
- latest-frame queue and inference worker
- result gate (stale > 0.3 s rejected; 0.5 s without data → degraded hold)
- SORT-style tracker with per-track range and lateral Kalman filters
- gated risk model with path prediction
- CLEAR/WATCH/WARNING/BRAKE/RELEASE state machine

## 2. Implemented NPU runtime
`docs/v2_npu_runtime.md`:
- a persistent BF16 inference server on the RNGD host (binary protocol, per-stage timing, health and
  fault-injection commands)
- a client with reader and writer threads, deadline-bounded requests, backpressure and stale-answer
  accounting
- `FailoverBackend` with logged transitions, and `HealthMonitor` with bounded sampling
- multi-PE behaviour measured: 1 → 4 PEs gives 537 → 1762.5 fps (3.3×); 8 threads are host-bound

## 3. Transport results
`docs/v2_npu_transport.md`:

| Path | e2e mean / p95 | Detections |
|---|---|---|
| V1-equivalent (1.23 MB letterboxed raw, default threads) | 202.8 / 277.0 ms (437 / 643 ms on a loaded host) | reference |
| V2 lossless (content-only, 2 pinned threads, exact pre-filter, SSH compression) | **60.2 / 65.5 ms** (71 / 104 ms loaded) | identical (recall 0.974 / precision 0.988 vs GPU, same as V1-equivalent) |
| JPEG q95 | 30.8 / 33.1 ms | recall 0.908 (lossy, not used for safety) |
| new SSH process per frame | 700.9 ms | — |

**Bottleneck.** Bandwidth (≈ 11 MB/s) plus CPU oversubscription on the shared host. Serialization and
NPU execution were not the problem.

## 4. Async-pipeline results
`docs/v2_async_pipeline.md`, `docs/v2_benchmark.md` § 4B:
- With latency also limiting the perception rate (the realistic model), V1 collides from 100 ms on.
- **V2 is collision-free from 0 to 200 ms,** and under jitter, bursts, 30 % dropped and 20 % stale
  results.
- Stale results were rejected (12–14 per stale run), and degraded ticks were held safely.

## 5. Tracking results
`docs/v2_tracking.md`, `reports/v2_scenario_tables.md` § C:
- Vehicles: 98–99 % coverage, 1 track id.
- Pedestrians: 79 % coverage, median 2 ids (rain 4).
- Cyclist: 55 % coverage.

Pedestrian and cyclist identity stability is the tracker's weak point.

## 6. Filter comparison
`docs/v2_range_rate_filtering.md`:
- **Offline, 33 Hz, 39 v3 logs:** the Kalman filter has rate MAE 1.27 m/s vs V1's LSQ 1.51, p95 2.98
  vs 4.04, MAE beyond 30 m 2.07 vs 3.44, and the same reaction delay (0.48 vs 0.45 s).
- **Closed loop (out of sample):** range-rate MAE 0.64–0.97 m/s on vehicles and 1.34 m/s on
  pedestrians.

## 7. AEB improvements
`docs/v2_aeb_controller.md`, `docs/v2_benchmark.md` § 4A:

| Measured GPU latency, 30/40/50 km/h | V2 | V1 | AEB off |
|---|---|---|---|
| Hazard collisions | **0 / 18** | 2 / 18 | 14 / 18 |
| Runs with unneeded braking | 1 / 12 | 2 / 12 | 0 / 12 |

What V2 fixes:
- the 40 km/h pedestrian collision (stops 9.1 m short)
- the unnecessary stop for a pedestrian who clears the path
- the permanent latch, replaced by a standstill hold with release

What it does not fix or makes worse:
- premature stops at a static obstacle (16–17 m short)
- smaller lead-vehicle margins than V1
- one short unneeded brake for an adjacent parked car at 50 km/h
- the 50 km/h cyclist, a near miss for both controllers

## 8. NPU live-loop status
`docs/v2_benchmark.md` § 6. CARLA → persistent SSH → RNGD BF16 → V2 stack → CARLA works in real time
(paced):
- **No collision** in pedestrian crossing (min gap **1.48 m**) or lead braking (8.87 m).
- **Classification: near-real-time.** Frame age p95 165–212 ms, perception 12–15 Hz. The GPU runs had
  66 ms and 20 Hz.
- The host was loaded during the runs (load average 18–21).
- **Not real-time; no real-time claim is made.**

In simulation-time stress runs with measured distributions, remote RNGD with the V2 transport was
collision-free, and the V1-equivalent transport collided.

## 9. Fallback behaviour
**Live.** An RNGD stall injected at 1.5 s was detected by the 0.25 s timeout. The system switched
PRIMARY → FALLBACK (GPU, 19 frames), logged both transitions, and was restored to PRIMARY at 3.5 s.
No collision; min gap 3.08 m.

**Unit tests** cover the DEGRADED state (both backends fail), the stale late answer and backpressure.

## 10. Additional model porting result
`docs/v2_operator_compatibility.md`. Lane segmentation (U-Net, LR-ASPP MobileNetV3) was **not running
on RNGD at the end of V2.**
- Exact graph rewrites for bilinear upsampling, ConvTranspose and dilated conv each compile and match
  the CPU (cosine 1.00000).
- At model scale, the ported U-Net failed in `max_pool2d` and LR-ASPP in tensor-group offsets.

Follow-up work, done in V3 Phase 1 while waiting for the V2 matrix, found an architecture that runs:
3×3 stride-2 downsampling and conv + PixelShuffle upsampling. This is reported in V3.

## 11. Operator compatibility findings
`docs/npu_operator_compatibility.md` (39 rows, generated from evidence):

| Group | Operators |
|---|---|
| Supported directly | conv, depthwise conv, SiLU / ReLU / Hardswish / Hardsigmoid, SE blocks, max-pool (small sizes), nearest upsample, concat, softmax, PixelShuffle, slicing |
| Need fusion | BatchNorm |
| Need a graph rewrite | ConvTranspose, bilinear interpolate, dilated conv |
| Not lowerable | `round`, `fake_quantize_*` |
| Host execution | YOLO decode / NMS |

## 12. Benchmark table
`docs/v2_benchmark.md` § 1: GPU CUDA Graph 2.60 ms; RNGD BF16 2.0–2.3 ms on device; end to end to the
desktop loop 22 ms (GPU in loop) vs 60–71 ms (RNGD remote, V2) vs 203–437 ms (V1-equivalent).

Resolution study:
- 640 is the operating point; RNGD exec stays flat from 320 to 640 (1.75–2.04 ms).
- Lower resolution only shortens transport and costs recall (0.42 at 320).

**Quantization.** No activation quantization is possible in-graph with this SDK path: no API,
fake-quant not importable, `round` not lowerable. BF16 and W8 weight-only remain the only options.

## 13. Failures (kept)
`reports/agent_runs/failures.jsonl`:
- **Blocking send.** The remote client's send could block past its deadline; fixed with a writer
  thread.
- **Contamination 1, orphan process.** Stopping the matrix wrapper left an orphan run with pre-fix code
  holding the CARLA lock. The server lock made the next start refuse.
- **Contamination 2, leftover actors.** Force-killed clients left actors on the road, and the cleanup
  missed them because the synchronous snapshot was not ticked. Two pilot rounds were contaminated, and
  one controller change based on them was reverted.
- **Response to both:**
  - a reset tool
  - a clean-world guard in the runner
  - a code commit recorded per run
  - contaminated outputs archived and kept out of evidence
- **Multi-PE usage error.** Moving one compiled instance between devices.
- **Controller defects found and fixed in pilots:**
  - side-truncated boxes biasing lateral estimates
  - stop-and-go under partial braking
  - spurious lateral velocity from ego yaw
- **Method caveat.** The controller was developed and evaluated on the same scenario family; there is
  no held-out scenario set.

## 14. Tests
- `pytest`: **62 passed, 1 skipped** (V1 33 + 1 live opt-in; V2 runtime 13, control 16).
- `ruff check` / `ruff format --check`: clean.
- The live CARLA test was not re-run after V2. V2 CARLA runs (142 matrix + 5 live) exercised the
  session, lock and clean-world guard.

## 15. Commits
`git log --oneline v1.0.0-rc1..v2.0.0-rc1` on `feature/v2-npu-closed-loop`.

## 16. V3 recommendations
- Use the ported lane U-Net (3×3 stride-2, conv + PixelShuffle, base width ≥ 32).
- Put each model on its own PE (V3 scheduling data: separate PEs gave the lowest age under load).
- Add ego-motion compensation from vehicle odometry (fixes the remaining adjacent-vehicle brake).
- Add radar for range rate.
- Keep the age gate and the failover.
- Report the remote path as near-real-time; a real-time claim needs a co-located NPU.

## Known limitations
- simulation only; one map; one seed per cell
- monocular range
- no ego-motion compensation
- pedestrian and cyclist tracking weaker than vehicle tracking
- remote NPU is near-real-time only
- shared RNGD host: load varied between 5 and 31 during measurements
- power and memory not re-measured in V2
- **V1 evidence contains the remote container hostname:** frozen, unpushed, decision needed before
  publishing
