# Closed-loop CARLA Scenario Evaluation (Phase 6, Gate F)

Full tables: `reports/scenario_tables.md` (generated). Raw evidence: `reports/evidence/scenarios_{v1,v2}.json`;
per-tick logs (ego speed, gap, lateral offset, closing speed, perception flag, detection count, inference
latency, decision, applied brake/throttle/steer, collision count) in `outputs/scenario_logs/*.jsonl` (not committed).

**This is a controlled research prototype. Nothing here demonstrates road-ready autonomous driving.**

## Setup
- CARLA 0.9.16, Town10HD_Opt, synchronous mode, **dt = 0.01 s** (10 ms latency resolution), seed 42.
- Ego: Tesla Model 3 spawned on the first spawn point with ≥ 90 m of straight, junction-free lane;
  initial speed 30 km/h; simple waypoint pursuit + P speed control (not a planner).
- Perception: YOLO26n raw split on the RTX 3060, **every tick, in the loop** (in-loop GPU e2e p95 ≈ 22–28 ms).
- Target association: the ground-truth target's bounding-box centre is projected into the camera; the target
  counts as *perceived* if it falls inside a detection box of the matching class group.
- Distance/TTC: **CARLA ground truth** (bumper gap along ego heading, closing speed). A 2-D box alone is not
  used for metric distance.
- AEB (`src/npu_ad/control/aeb.py`): corridor ±1.8 m; partial 0.5 brake if TTC < 2.6 s; full brake (latched)
  if TTC < 1.6 s or gap < 4 m; brakes only on perception-confirmed targets (3-tick hold).
  v2 adds partial-brake hold and standstill hold (see below).
- Latency model: a decision computed at tick *k* is applied at tick *k + ⌈delay/dt⌉*.

| Condition | Delay (ms) | Origin |
|---|---|---|
| aeb_off | — | AEB disabled (danger baseline) |
| ideal_0ms | 0 | idealised |
| npu_p95 | 9.85 → applied 10 | **measured** RNGD device-side p95 (4.80) + host pre/post p95 — *latency injection* |
| gpu_p95 | 22.72 → applied 30 | **measured** clean GPU e2e p95 |
| stress_100ms / stress_200ms | 100 / 200 | sensitivity sweep |

**Important:** in every condition detections come from the in-loop GPU model. The NPU condition injects the
measured NPU latency; it is *not* NPU-in-the-loop (the RNGD is a remote host). This is justified for
detection content by the 98.8 % / 99.2 % GPU–NPU detection agreement (Phase 5), but it remains a simulation.

## Scenarios
| Scenario | Weather | Hazard |
|---|---|---|
| straight_road | ClearNoon | none (10 s cruise) |
| lead_vehicle_braking | ClearNoon | lead car 25 m ahead at 5 m/s, full brake at t = 3 s |
| pedestrian_crossing | ClearNoon | pedestrian 4.5 m to the side, 30 m ahead, crosses at 1.6 m/s when gap ≤ 24 m |
| static_obstacle | ClearNoon | stopped car in lane 40 m ahead (COCO has no cone/debris class) |
| low_light | ClearNight | lead-vehicle braking |
| rain | HardRainNoon | pedestrian crossing |

## Results (AEB v2)

| Scenario | AEB off | ideal 0 ms | NPU 10 ms | GPU 30 ms | 100 ms | 200 ms |
|---|---|---|---|---|---|---|
| lead_vehicle_braking | **collision** | 8.21 m | 8.19 m | 8.90 m | 8.37 m | 7.55 m |
| pedestrian_crossing | **collision** | 2.19 m | 2.05 m | 1.89 m | 1.36 m | 0.53 m |
| static_obstacle | **collision** | 15.44 m | 15.43 m | 15.26 m | 14.62 m | 13.79 m |
| low_light | **collision** | 8.21 m | 8.19 m | 8.90 m | 8.37 m | 7.55 m |
| rain | **collision** | 2.19 m | 2.05 m | 1.89 m | 1.36 m | 0.53 m |
| straight_road | no braking | no braking | no braking | no braking | no braking | no braking |

(values = minimum bumper gap before stop; all AEB-on runs ended with the ego stopped, 0 collisions)

### Findings
1. Without AEB the ego collides in all five hazard scenarios; with AEB it stops without collision in all
   25 AEB-on hazard runs (5 scenarios × 5 delay conditions) per controller version, for delays up to 200 ms.
2. **Latency visibly reduces the safety margin in the pedestrian case**: 2.19 → 2.05 → 1.89 → 1.36 → 0.53 m
   for 0 / 10 / 30 / 100 / 200 ms. At 200 ms only 0.53 m remained, so the margin at this speed is small.
   The measured NPU pipeline latency (10 ms applied) kept 0.16 m more margin than the measured GPU latency
   (30 ms applied) — a consequence of the injected delays, not an NPU-in-the-loop measurement.
3. Lead-vehicle braking is **non-monotonic** between 0 and 30 ms (8.21 vs 8.90 m). Logs show why: with 0 ms
   the partial brake alone stopped the car and TTC never fell below 1.6 s; with 30 ms the extra delay pushed
   TTC below 1.6 s at t = 3.82 s, triggering full braking earlier. This is a threshold effect of staged braking.
4. `low_light` produced the same trajectories as daylight lead braking: the weather was applied (mean
   detections/frame 3.67 day vs 2.61 night), but the lead car 25 m ahead was perceived in 100 % of frames
   in both. **This setup did not reveal a low-light perception failure**; a harder night scenario is needed.
5. `rain` changed perception timing (first pedestrian detection 0.35–0.69 s vs 0.38 s in clear weather, varying
   between runs because rain rendering is not deterministic) but braking was triggered later by the
   corridor-entry condition, so outcomes matched the clear-weather case.
6. `straight_road` shows no spurious braking, but by construction the AEB only reacts to ground-truth actors
   confirmed by perception, so **detector false positives cannot trigger braking in this design** — this scenario
   does not test false-positive robustness.

## AEB v1 → v2 (defect found by evaluation)
v1 released the brake as soon as TTC rose above the threshold; cruise control then re-accelerated towards
the stopped obstacle (partial ↔ clear oscillation, logged in `lead_vehicle_braking__ideal_0ms__v1.jsonl`).

| Scenario (ideal 0 ms) | v1 min gap | v2 min gap |
|---|---|---|
| static_obstacle | 3.89 m | 15.44 m |
| lead_vehicle_braking | 3.87 m | 8.21 m |
| pedestrian_crossing | 2.19 m | 2.19 m (full brake path, unaffected) |

v1 also had zero collisions; v2 is kept as default. Both result sets are preserved.

## Limitations
Single map, single seed per scenario, single ego speed (30 km/h), one run per cell (no repeat statistics);
GT-based distance; perception association via projected centre; latency modelled as pure actuation delay
(no frame dropping or queueing); NPU latency injected, not measured in the loop.
