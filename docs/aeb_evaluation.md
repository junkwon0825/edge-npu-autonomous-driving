# Closed-loop AEB evaluation in CARLA (v3, 2026-11-23)

All numbers: `reports/evidence/scenarios_v3.json`, rendered tables `reports/scenario_tables_v3.md`
(generated, not hand-edited). Earlier matrices (v1/v2, GT-assisted only) remain in
`reports/scenario_evaluation.md` and are not mixed with v3.

## What drives the brake? (three explicitly separated baselines)
| Mode | Range / TTC input | Role of the camera detector | Label used in this repo |
|---|---|---|---|
| `gt_oracle` | CARLA ground truth | none | **GT oracle** (upper bound) |
| `gt_gated` | CARLA ground truth | brake only while the GT target's projected centre lies inside a same-class box | **GT-assisted** (v1/v2 results were this mode) |
| `mono` | camera only: flat-ground range from the box bottom edge, closing speed from a 0.3 s least-squares fit of the range history, closest in-path object | everything | **perception-driven** (GPU detector in the loop) |

Ground truth is logged in every mode for evaluation only. 2D box size is never used as distance; the
ground-plane range is validated per tick against GT (below). NPU perception in the loop: see
"NPU" at the end.

## Setup
- Town10HD_Opt, fixed straight spawn (first spawn point with ≥ 90 m straight lane), synchronous mode
  dt = 0.01 s, TM seed 42, 1 s settle phase, ego target speed 30 km/h unless noted, AEB v2
  (TTC full 1.6 s / partial 2.6 s, min gap 4 m, corridor ±1.8 m, perception hold 3 ticks, hysteresis).
- Scenarios: stationary vehicle (40 m), lead vehicle 5 m/s braking hard at t = 3 s (25 m),
  pedestrian crossing (1.6 m/s, triggered at a time-to-arrival of 2.88 s), the last two again in
  `ClearNight` (`low_light`) and `HardRainNoon` (`rain`); no-hazard probes: empty road, parked
  vehicle in the adjacent lane, pedestrian standing at the roadside.
- Latency conditions (actuation delay): 0 ms, NPU on-vehicle p95 9.85 ms, GPU p95 22.7 ms,
  100 ms, remote NPU p95 134.7 ms (NPU + measured SSH frame transfer), 200 ms.
- One run per cell; repeatability checked separately (below).

## Results (30 km/h)
| Mode | Hazard runs (AEB on) | Collisions | Median min gap | FP brake events in no-hazard runs | Median GT TTC at first brake |
|---|---|---|---|---|---|
| GT oracle | see table | 0 | see table | 0 / 15 | see table |
| GT-assisted | 24 | 0 | 8.84 m | 0 / 15 | 2.14 s |
| perception-driven | 25 | 0 | 9.53 m | 0 / 15 | 2.51 s |

AEB off: every hazard scenario collides. Per-cell values: `reports/scenario_tables_v3.md`.

Findings:
1. **Perception-driven AEB works in these scenarios** (0 collisions at 0–200 ms) and brakes *earlier*
   than GT-assisted (median GT TTC at first brake 2.51 s vs 2.14 s). The earlier onset is not better
   perception; it comes from **overestimated closing speed** (next point).
2. **Range is good, range rate is not.** Range error vs GT (MAE): 0.11 m pedestrian, 0.26 m lead
   vehicle, 0.46 m lead at night, 0.79 m static vehicle at 15–40 m (p95 ≤ 2 m).
   Closing-speed error (MAE): 1.2–1.5 m/s below 30 m, 3.2 m/s beyond 30 m with p95 14 m/s
   (`evidence/mono_closing_speed_error.json`). In `static_obstacle` at 200 ms the overestimate
   (12–13 m/s vs 8.4 m/s true at ~30 m) started partial braking at 30 m; the hysteresis held it, the
   car stopped and latched 23.4 m short of the obstacle — a premature stop, not a collision.
3. **Weather now matters.** In GT-assisted mode `low_light`/`rain` are bit-identical to their clear
   counterparts (v2 and v3), because timing is set by GT TTC. In perception-driven mode they differ:
   rain pedestrian min gap 2.81 → 1.12 m (0 → 200 ms) vs 2.98 → 1.29 m in clear weather.
4. **Latency sensitivity** is monotone for the pedestrian case in all modes (e.g. perception-driven
   2.98 / 2.84 / 2.68 / 2.15 / 1.29 m at 0 / 9.85 / 22.7 / 100 / 200 ms). Lead-vehicle braking is
   non-monotone around 100–200 ms because the staged partial/full braking thresholds are crossed at
   different ticks.
5. **No false-positive braking** in 45 no-hazard runs (3 modes × 15). The probes are simple (one
   static object outside the corridor); this does not establish a false-positive rate in traffic.

## Speed sweep (NPU on-vehicle latency, 9.85 ms)
| Scenario | 20 km/h | 30 km/h | 40 km/h | 50 km/h |
|---|---|---|---|---|
| static obstacle | no collision (both modes) | no collision | no collision | no collision |
| lead vehicle braking | no collision | no collision | no collision | no collision |
| pedestrian crossing | no collision | no collision | **collision in both modes** | see finding 7 |

6. **40 km/h pedestrian: AEB cannot avoid** in GT-assisted or perception-driven mode. With the
   trigger scaled to the same time-to-arrival, the pedestrian enters the corridor about 1.2 s before
   impact; at 11.1 m/s the ego needs more than that to stop with this controller (staged 0.5/1.0
   brake). This is the physical limit of a corridor-only AEB, not a perception failure (no misses).
7. **50 km/h pedestrian: AEB-induced collision.** With AEB off the ego passes before the pedestrian
   reaches its lane (no contact). GT-assisted AEB detects the pedestrian at the corridor edge just as
   the ego reaches it (gap ≈ 1.7 m), latches full braking, stops in the pedestrian's path and is hit
   on the side. The controller has no "target already alongside / will clear" logic and the latch is
   permanent. Perception-driven mode did not brake (target never classified in-path ahead) and had
   no contact. This is a real controller defect and is listed as future work; the controller was not
   changed after seeing it, so all v3 cells use the same logic.

## Repeatability
The same configuration run twice (`tag = rep2`) gives the same outcome (min gap within 0.03 m,
identical brake onset tick), but the logs are not bit-identical: detections differ from the first
tick (`n_dets`, mono range up to 0.47 m), and in `static_obstacle` this changed one partial-brake
decision and hence the trajectory slightly. Vehicle kinematics are identical when decisions are
identical (pedestrian case). The non-determinism is in the perception input, see
`docs/gpu_npu_comparison.md` § determinism.

## Data integrity
- Six runs overlapped an unauthorized second CARLA client (a live pytest, 22:34:17–22:36:16). They
  were moved to `evidence/scenarios_v3_quarantined.json` (kept, not deleted) and re-run.
  Since then a process-level server lock (`npu_ad/simulation/server_lock.py`) refuses a second
  client and live tests need `NPU_AD_LIVE_CARLA=1`.

## Limitations
Simulation only, one map, one weather per scenario, one seed, single target per scenario, 100 Hz
perception, latency modelled as actuation delay, no sensor noise model beyond rendering, flat-ground
range assumption (fails on slopes/pitch), simple corridor logic (no path prediction).
