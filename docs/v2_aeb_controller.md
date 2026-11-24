# V2: tracking-based risk model, target-clear logic and AEB state machine

Code: `src/npu_ad/control/aeb_v2.py`, `src/npu_ad/perception/tracker.py`. Tests:
`tests/test_v2_control.py` (16 tests). Closed-loop results: `reports/v2_scenario_tables.md` (generated)
and `docs/v2_benchmark.md`.

## Inputs (per confirmed track, predicted to the control time)
| Quantity | Source |
|---|---|
| range, range rate | per-track `RangeKalman` on the flat-ground gap (`docs/v2_range_rate_filtering.md`) |
| lateral offset, lateral velocity | per-track `Kalman1D` on the flat-ground lateral offset |
| class, confidence, hits, age | tracker (`docs/v2_tracking.md`) |
| ego speed | vehicle state |

Ground truth is never used by the controller.

## Risk model (every threshold documented)
| Step | Rule | Value |
|---|---|---|
| Confirmation | hits ≥ `min_hits` | 3 |
| Confidence | smoothed confidence ≥ `min_conf` | 0.35 |
| Range validity | 0 < range < `max_range_m`, filter valid | 60 m |
| Closing plausibility | −30 m/s ≤ closing ≤ ego speed + 6 m/s (oncoming traffic out of scope) | — |
| Lateral plausibility | \|v_lat\| ≤ 8 m/s | — |
| TTC | range / closing, only if closing > 0.3 m/s; otherwise ∞ | — |
| Path envelope W | ego half width + class half width + margin | 1.0 + {ped 0.3, cyclist 0.4, vehicle 0.9} + 0.3 m |
| Lateral motion used | track age ≥ 0.25 s and \|v_lat\| ≥ 0.5 m/s (below: 0) | see "development history" |
| Ego conflict window | [TTC, TTC + (ego length 4.8 m + target length) / v_ego] | — |
| Pass margin | 0.3 s | — |

## Path prediction and target-clear logic (constant velocity, both axes)
The target overlaps the ego path while |y + v_y·t| ≤ W, over the interval [t_in, t_out]. With the ego
conflict window [t_arrive, t_exit], each track is classified as:
- **`collision_path`:** the intervals overlap (± pass margin).
- **`will_clear`:** the target leaves the path before the ego arrives. No braking. This fixes V1's
  unnecessary stop in "pedestrian clearing".
- **`ego_passes`:** the ego leaves the conflict zone before the target enters. No braking. This
  targets V1's 50 km/h failure, where braking stopped the car in the pedestrian's way.
- **`not_in_path`, `in_path_not_closing`.**
- **`uncertain`:** lateral motion is not yet known. It counts as `collision_path` only if the target is
  inside W now, and otherwise as WATCH.

## State machine
| State | Brake | Entered when | Left when |
|---|---|---|---|
| CLEAR | 0 | nothing relevant | — |
| WATCH | 0 | collision course with TTC ≤ 5 s, or target near the path (will clear / ego passes / uncertain / in path but not closing within 30 m) | risk rises or vanishes |
| WARNING | 0.5 | collision course and TTC ≤ 2.6 s | escalates immediately; releases only when risk is lower for 0.3 s **and** TTC > 2.6 + 1.0 s, or the target is off the collision course |
| BRAKE | 1.0 | collision course and (TTC ≤ 1.6 s or gap < 4 m while closing) | after ≥ 0.4 s in BRAKE, same release rule |
| RELEASE | ramps 2.0/s to 0 | from WARNING/BRAKE | risk returns → escalate again |

**Standstill hold.** If the AEB brought the car to a stop (in any braking state), the brake is held
while a target lies inside W within 30 m. It is released 1.0 s after the path is clear.

There is no permanent latch: the car drives on after a crossing pedestrian has left the path.

**Degraded perception.** With no accepted result for 0.5 s, the last command is held. A brake is never
released on missing data.

The TTC thresholds (1.6 / 2.6 s) and the partial brake (0.5) are V1's values. Differences between V1
and V2 therefore come from estimation and decision logic, not from retuned thresholds.

## Development history (kept because it is part of the result)
**1. Replay false brake.** The NPU replay (`v2_npu_replay.json`, first version) showed V2 braking on a
parked car in the adjacent lane, on GPU and NPU detections alike.
- Cause: the car's box was clipped at the right image border. The visible centre slid inward, so the
  estimated lateral offset fell from 3.1 to 2.4 m (GT 3.5 m), with a fake inward velocity of
  −1.6 m/s.
- Fix: boxes touching a side border do not update the lateral filter, and their lateral velocity is
  treated as 0.

**2. Stop-and-go at a static obstacle.** In the first matrix run, WARNING's partial brake stopped the
car 16.5 m short. As the car slowed, TTC rose above 2.6 s, WARNING released, the car crept forward,
and the cycle repeated (4 brake onsets).
- Fixes:
  - a standstill hold for every braking state
  - a 1.0 s TTC margin before releasing
  - an in-path target that is not closing counts as WATCH
- A regression test covers it.

**3. Contaminated pilots.**
- What happened: force-killed matrix processes left 3 ego vehicles, 2 pedestrians and an Audi on the
  test road. The cleanup tool listed 0 actors, because in synchronous mode the actor snapshot is only
  refreshed by a tick.
- Effect: two pilot rounds braked on and hit these leftovers. One path-envelope change was made on
  that data and then reverted.
- Response:
  - the world was reset with `scripts/carla_reset.py`, which ticks before listing
  - the V2 runner refuses to start if any vehicle or walker already exists
  - every run records its code commit
- Contaminated outputs are kept under `outputs/v2/*contaminated*` / `*prefix*`. None are in the
  evidence files.

**4. Spurious lateral velocity on a parked car.** In a clean pilot, a parked car in the adjacent lane
(lateral 3.3–3.4 m, GT 3.38 m) showed an estimated lateral velocity of −0.3 to −0.44 m/s.
- Cause: the ego's own small heading changes during lane following move distant objects laterally in
  the ego frame (Δlat ≈ range · Δyaw).
- Response: lateral motion below 0.5 m/s is treated as zero; pedestrians walk at 1.6 m/s and the
  cyclist rides at 2.5 m/s.
- After this, the adjacent vehicle still caused **one short unneeded partial brake** (no stop) at
  30 and 50 km/h. Tuning stopped there to avoid fitting the evaluation scenarios.
- The proper fix, ego-motion compensation from vehicle odometry, is listed for V3.

Development and evaluation used the same scenario family. There is no held-out scenario set; this is
the main methodological limitation of the V2 controller numbers.
