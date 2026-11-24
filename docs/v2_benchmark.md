# V2 benchmark

Every number comes from a file in `reports/evidence/` (named in each section).

**Conditions vary by section and are stated each time:**
- the remote RNGD host is shared: its load average is recorded
- desktop GPU timings run with CARLA stopped unless stated
- the closed-loop matrix runs in simulation time, with measured latency distributions

## 1. Canonical comparison (YOLO26n, batch 1)
| Runtime | Precision | Model latency | p95 | Throughput | End-to-end to a desktop control loop (mean / p95) | Frame age in the CARLA loop | Collisions (V2 controller, 30 km/h hazards) |
|---|---|---|---|---|---|---|---|
| GPU RTX 3060, CUDA Graph | FP32 | 2.60 ms (V1) | 3.01 | 387 fps | — | — | — |
| GPU RTX 3060, eager, in the CARLA loop | FP32 | 13.5 ms clean / 22.4 ms e2e with CARLA sharing the GPU | 27.8 (e2e) | — | 22.4 / 27.8 ms (`gpu_local`) | see § 4 | see § 4 |
| RNGD (remote), V2 transport | BF16 | 2.0–2.3 ms exec | 2.3 | 537 fps per PE, 1762.5 fps on 4 PEs | 60.2 / 65.5 ms (quiet host), 71.2 / 104.4 ms (load average ~30) | see § 4 | see § 4 |
| RNGD (remote), V1-equivalent transport | BF16 | same | same | same | 202.8 / 277.0 ms (quiet), 437 / 643 ms (loaded) | see § 4 | see § 4 |
| RNGD | W8 weight-only | 2.16–2.35 ms (V1) | 2.37 | 446 fps | not re-measured in V2 | — | — |

**Sources:**
- `gpu_benchmark_ext.json`, `npu_bench*.json`, `npu_perf*.json` (V1, unchanged)
- `v2_transport_profile*.json`, `v2_latency_samples.json`, `v2_npu_multistream*.json`

**Power (V1, indicative only).** GPU 136 W (CUDA Graph) vs RNGD 53–54 W board power, from different
meters. It was not re-measured in V2.

**Memory.** Per-variant device memory is not observable (`furiosa-smi status` reports 0.00 GiB in
use), so no memory comparison is made.

## 2. Profiling (where the time goes)
See `docs/v2_npu_transport.md`. In short:
- **NPU execution is 1–3 % of the remote end-to-end time.**
- **V1 configuration:** 51 % network upload, 39 % remote-host CPU (thread oversubscription).
- **V2 configuration:** about 28 % network and 8 % server.
- **Classification:** transport-bound (bandwidth), plus CPU contention on the shared host. The
  analysis is qualitative; no hardware counters are available.

## 3. Resolution study (`v2_resolution_npu.json`, `v2_resolution_gpu.json`)
| Input | RNGD exec p50 | RNGD remote e2e mean / p95 | Payload | Recall / precision vs GPU-640 (RNGD) | Recall / precision vs GPU-640 (GPU at that size) |
|---|---|---|---|---|---|
| 320 | 1.89 ms | 31.2 / 36.3 ms | 173 kB | 0.422 / 0.754 | 0.448 / 0.788 |
| 480 | 1.75 ms | 46.5 / 54.6 ms | 389 kB | 0.647 / 0.818 | 0.661 / 0.833 |
| 640 | 2.04 ms | 61.8 / 70.0 ms | 691 kB | 0.974 / 0.988 | 1.000 / 1.000 (reference) |

**Reading the table:**
- **NPU compute.** It barely changes with resolution for this nano model (fixed-overhead-bound).
- **Remote latency.** It scales with the number of pixels sent (transport-bound).
- **Accuracy cost.** At lower resolution, many small distant objects are lost (recall 0.42 at 320).

**Operating point: 640.** A lower resolution only helps the transport, and costs more accuracy than
lossless compression does. GPU latency per resolution: § 3a.

### 3a. GPU per resolution (`v2_resolution_gpu.json`, CARLA stopped, n = 216)
| Input | Eager mean / p95 | CUDA Graph mean / p95 |
|---|---|---|
| 320 | 16.71 / 22.80 ms | 1.87 / 3.20 ms |
| 480 | 14.69 / 18.06 ms | 2.27 / 3.40 ms |
| 640 | 13.16 / 17.89 ms | 2.89 / 3.95 ms |

**Reading the table:**
- **Eager does not get faster at lower resolution.** It is bound by kernel launches, not compute.
- **CUDA Graph scales with pixels.** The RNGD does not.

## 4. Closed loop (V2 matrix, `reports/v2_scenario_tables.md`, `v2_scenarios.json`)
**Setup:**
- 142 runs: Town10HD_Opt, dt 0.01 s, camera 33 Hz, seed 42.
- Controller code is identical in all runs (commits 933a1fe and 9d2afb7 differ only in the unused
  overlay module).
- Perception is GPU YOLO26n; the latency model sets when each result is available.

**A. Measured in-loop GPU latency (`gpu_local`), 30/40/50 km/h, 10 scenarios.**

| Controller | Hazard runs | Collisions | No-brake-needed runs | Runs with unneeded braking |
|---|---|---|---|---|
| **V2** | 18 | **0** | 12 | 1 (adjacent vehicle, 50 km/h, short partial brake, no stop) |
| V1 (perception-driven) | 18 | 2 (pedestrian crossing and rain at 40 km/h) | 12 | 2 (pedestrian clearing, 30 and 40 km/h: stopped) |
| AEB off | 18 | 14 | 12 | 0 |

- **V2 avoids V1's documented 40 km/h pedestrian failure.** It stops 9.1 m short (rain: 10.0 m). Its
  path prediction brakes on the pedestrian's predicted entry instead of waiting for the corridor.
- **At 50 km/h, V2 stops 2.3 m before the crossing pedestrian.** V1 and AEB-off pass alongside.
- **V2 does not brake for the pedestrian who clears the path.**
- **Margins are not always larger.**
  - At a static obstacle, V2 stops 16–17 m short and V1 18.5–25 m short. Both are premature, caused by
    partial braking at TTC 2.6 s.
  - In lead braking, V2 keeps less distance than V1 (5.2–8.9 m vs 7.7–10.8 m).
  - The cyclist at 50 km/h is a near miss for both (gap below 0.05 m, no contact).

**B. Latency and runtime stress (30 km/h, pedestrian crossing and lead braking, 13 conditions).**

| | V2 collisions | V1 collisions |
|---|---|---|
| fixed 0 / 50 ms | 0 / 4 | 0 / 4 |
| fixed 100 / 150 / 200 ms | 0 / 6 | **6 / 6** |
| jitter ±40 ms, bursts +250 ms, 30 % dropped results | 0 / 6 | 0 / 6 |
| 20 % stale results (+400 ms) | 0 / 2 | 1 / 2 |
| NPU co-located (measured server time) | 0 / 2 | 0 / 2 |
| NPU remote, V2 transport (measured, loaded host) | 0 / 2 | 1 / 2 |
| NPU remote, JPEG q95 | 0 / 2 | 0 / 2 |
| NPU remote, V1-equivalent transport (437 ms mean) | **2 / 2** | **2 / 2** |

**Reading the stress table:**
- **The rate-limiting latency model is harsher than V1's actuation delay.** A slow backend also gets
  fewer frames. V1 survived 100–200 ms as a pure delay in the v3 matrix; here it collides from 100 ms
  on.
- **V2 holds up to 200 ms** because of frame-age-aware tracking and prediction to the control time.
- **V2 cannot survive the V1-equivalent remote transport.** Its results are mostly older than the
  0.3 s limit, so they are rejected: 572 degraded ticks in lead braking. Without fresh perception it
  cannot start braking. The age gate makes the failure explicit (degraded hold); it cannot create
  perception that is not there.
- **The V2 transport changed the outcome of a safety function in simulation.** With the same NPU, the
  V1-equivalent remote path collides and the V2 path stops 9.3–11.1 m short.

**C. Tracking quality (V2 runs, GT target visible; medians).**
- Vehicles: 98–99 % coverage, 1 track id, range-rate MAE 0.64–0.97 m/s.
- Pedestrians: 79 % coverage, 2 ids (rain: 4), range-rate MAE 1.34 m/s.
- Cyclist: 55 % coverage.

These are out-of-sample checks of the Kalman settings. The closed-loop range-rate error is lower than
V1's 1.2–3.2 m/s, except for the crossing cyclist (2.85 m/s).

## 5. NPU replay, Mode B (`v2_npu_replay.json`)
The setup:
- 738 recorded frames from 7 v3 closed-loop runs
- the RNGD BF16 remote server (V2 transport, full 640² frames, `-C`) vs local GPU FP32
- both streams drive the identical V2 stack, open loop

| Metric | Value |
|---|---|
| Detection agreement (recall / precision / matched IoU) | 0.956 / 0.949 / 0.990 |
| First BRAKE time identical | 7 / 7 runs (including 2 runs with no BRAKE) |
| First WARNING time identical | 5 / 7 (adjacent vehicle: RNGD-only WARNING at 0.95 s; static obstacle: 50 ms earlier on RNGD) |
| Per-tick AEB state agreement | 86–99 % per run |
| RNGD request latency | 69.8 ms mean / 84.9 ms p95 / 94.2 ms p99 (n = 738; host load average 9–10) |

## 6. Live loop, Modes A and C (`v2_live_loop.json`)
**Setup:**
- Real threads; the simulation is paced to wall-clock time with a 50 ms control period.
- Latest-frame queue of size 1. Frame age is control time minus capture time, so it includes up to
  one control period.
- **The RNGD host was loaded during these runs** (load average 18–21, a parallel YOLO26s compile).

| Run | Collision | Min gap | First brake | Frame age at use, mean / p95 | Backend latency, mean / p95 | Perception rate | Dropped frames | Class |
|---|---|---|---|---|---|---|---|---|
| A: GPU, pedestrian crossing | no | 11.34 m | 1.65 s | 66 / 67 ms | 22 / 25 ms | 19.9 Hz | 1 | real-time |
| A: GPU, lead braking | no | 9.28 m | 3.70 s | 65 / 66 ms | 22 / 26 ms | 20.0 Hz | 0 | real-time |
| C: RNGD remote, pedestrian crossing | no | **1.48 m** | 2.85 s | 136 / 195 ms | 70 / 110 ms | 13.0 Hz | 62 | near-real-time |
| C: RNGD remote, lead braking | no | 8.87 m | 3.75 s | 131 / 165 ms | 65 / 81 ms | 15.2 Hz | 42 | near-real-time |
| C + failover (RNGD stalled 1.5 s at t = 1.5 s) | no | 3.08 m | 2.65 s | 137 / 212 ms | 70 / 107 ms | 12.0 Hz | 71 | near-real-time |

**Classes.** "Real-time" means frame-age p95 ≤ 100 ms and fewer than 10 % paced-tick overruns.
"Near-real-time" means ≤ 250 ms. Overruns were 0–4 %.

**Remote NPU pedestrian run.** It avoided the collision with only 1.48 m left. It had 6 timeouts and 16
backpressure refusals, and 52 ticks without fresh perception at the start, so it started braking 1.2 s
later than the GPU run. **The live remote NPU loop works, but it is not real-time and its safety
margin is the smallest of all runs.**

**Failover.**
- The injected stall (t = 1.5 s) was detected by the 0.25 s timeout. The first switch,
  PRIMARY → FALLBACK, was logged at t = 1.83 s.
- 19 frames ran on the GPU.
- The RNGD was probed and restored at t = 3.50 s (stall end 3.0 s + 0.5 s probe interval).
- Every transition was logged, and the overlay shows `Fallback: FALLBACK`.

Videos: `assets/demo/v2_demo_npu_live.mp4`, `assets/demo/v2_demo_npu_failover.mp4`.
