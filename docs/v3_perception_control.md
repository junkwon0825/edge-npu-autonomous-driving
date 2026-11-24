# V3: lane perception, LKA, AEB on curves, sensor fusion, scene state

Code:
- `src/npu_ad/perception/{lane_net,lane_geometry,radar_fusion,tracker,filters}.py`
- `src/npu_ad/control/{lka,aeb_v2}.py`, `src/npu_ad/scene.py`
- `scripts/v3_run_scenarios.py`

Tests: `tests/test_v3_lane_control.py` (9).

## Lane perception
1. **Lane-marking segmentation.** `LaneUNet` on GPU or RNGD; see `docs/v3_lane_npu_porting.md` for
   accuracy.
2. **Geometry** (`LaneGeometry`):
   1. Mask pixels → 800×450 camera pixels.
   2. Flat-ground projection with the camera height calibrated at rest: Z = f_y·h / (v − c_y),
      Y = (u − c_x)·Z / f_x. Points from 4 to 30 m ahead are used.
   3. Left/right boundary seeds come from a lateral histogram at 4–15 m, nearest the previous lane
      centre on each side.
   4. 2 m sliding windows (±0.6 m + 2 % of range) follow each boundary.
   5. Weighted least squares per boundary: **y(x) = a·x² + b·x + c** (linear below 8 m of extent).
      Weights are 1/x, because flat-ground error grows with x².
   6. Centre = mean of both boundaries. With one side only, the centre is offset by the running lane
      width. A pair is rejected if its width is outside 2.4–5.0 m.
   7. Outputs: **lateral offset** e = −c_centre, **heading error** ψ = atan(b_centre), curvature 2a,
      width and confidence.
   - Unit tests render synthetic masks from known geometry and recover the offset to ±0.15 m, the
     heading to ±0.02 rad and the width to ±0.3 m.

## LKA (pure pursuit) and arbitration
| Parameter | Value | Why |
|---|---|---|
| Look-ahead | 0.8 s × v, clamped to 6–20 m | standard pure-pursuit scaling |
| Latency compensation | target = y_c(L + v·age) − L·Δψ (Δψ from odometry since the lane frame's capture) | the lane estimate is up to ~100 ms old in Mode D |
| Steering | δ = atan(wheelbase · 2·y / L²), wheelbase 2.875 m; command δ / 70° | Tesla Model 3 geometry in CARLA |
| Saturation / rate limit | ±0.5, 1.5 per second | smoothness |
| Lane timeout | 0.5 s → LKA unavailable | never steer on stale geometry |
| Control frequency | 100 Hz (lane updates 15 Hz in the fixed schedule) | — |

**Arbitration** (`Arbiter`), in order of precedence:
1. **AEB braking** (WARNING/BRAKE): throttle 0, AEB brake, steering = LKA clamped to ±0.1. There is no
   evasive steering while braking.
2. **LKA unavailable:** steering decays to 0 at 0.5 per second and the car coasts (throttle 0).
3. **Normal:** LKA steering + speed controller.

The V1/V2 runs steered with a map-waypoint follower (ground-truth lane). **In V3, steering comes only
from camera lane perception.**

## AEB on curves and ego-motion compensation
- **Path-relative corridor.** Tracks are re-expressed relative to the predicted ego path: the perceived
  lane centre y = a·x² + b·x, or yaw-rate / speed when no lane is available. Without this, an in-lane
  obstacle 45 m ahead on a 147 m-radius curve sits 6.9 m to the side and is ignored. A unit test
  covers it.
- **Ego yaw-rate compensation.** The tracked lateral velocity gets + range × yaw rate added. This was
  V2's spurious lateral velocity source on the adjacent parked car.
- A first version also added a geometric path term. A unit test showed this double-counts motion, and
  it was removed.

## Sensor fusion, stage 1 (radar)
- **Sensor.** A CARLA `sensor.other.radar` at the front bumper: 30° × 10°, 80 m, 3000 points/s, 20 Hz.
- **Association.**
  - Radar points within the camera track's box azimuth (±1.5°).
  - Range gate ±max(3 m, 35 %) around the camera range.
  - The nearest-surface cluster (30th percentile + 0.5 m); its median gives the gap and the closing
    speed (radial velocity / cos azimuth).
- **Fusion.** Scalar Kalman updates of the same track filter: range σ 0.25 m, rate σ 0.3 m/s, NIS-gated
  (`kalman_measure`). Radar refines tracks the camera created; it does not create tracks.
- **Camera-only vs camera + radar** in closed loop: § Results (pending).

## Scene state
`SceneState`: the control layer reads only this:
- ego speed, yaw, yaw rate
- lane estimate, its capture time and the yaw at capture
- tracked objects (path-relative)
- closest hazard and TTC
- per-source health: backend, ages, stale/failed counts, fallback state
- `road_condition`: not perceived in V3, so it stays "unknown"

Raw boxes and masks never reach the controllers.

## Results so far
**Smoke runs** (`reports/evidence/v3_smoke_lka_modeA.json`):
- Mode A (GPU detection + GPU lane), 30 km/h, 12 s each.
- They were run with V3 code that was not yet committed. It was committed afterwards in `d44ba7e`;
  runs since then record `+dirty` when that happens.

| Route | Lateral RMSE (GT) | Max lateral error | Heading error (mean) | Lane departures | Lane-estimate offset MAE | Steering rate RMS | Lane geometry ok / one side / fail |
|---|---|---|---|---|---|---|---|
| Town04 straight | 0.008 m | 0.010 m | 0.01° | 0 | 0.016 m | 0.03 /s | 199 / 0 / 0 |
| Town04 mild curve (R 147 m) | 0.014 m | 0.032 m | 0.21° | 0 | 0.053 m | 0.06 /s | 200 / 0 / 0 |
| Town10 strong curve (R 39 m) | 0.108 m | 0.200 m | 0.94° | 0 | 0.296 m | 0.32 /s | 135 / 65 / 0 |

On the strong curve the geometry often sees only one boundary (65 of 200 frames), and the offset
estimate error grows to 0.30 m.

**Closed-loop results** (V3 matrix, all tables in `reports/v3_closed_loop_tables.md`, report §7–§10):
- **Lane keeping at 30 km/h:**
  - 0.8–2.6 cm RMSE on straight, mild, night and rain routes
  - 10–13 cm on the 39 m-radius curve
  - 0 departures in healthy runs
- **Lane-age tolerance:** about 0.25 s, set by the 0.3 s stale gate.
- **AEB + LKA:**
  - 0 collisions in 36 hazard runs
  - all 29,618 braking ticks in an AEB arbitration mode, with |steer| ≤ 0.10
  - lateral error while braking ≤ 0.12 m
- **Camera-only vs camera + radar:** mixed.
  - Better for a lead vehicle at 50 km/h.
  - Worse closing-speed error for a pedestrian at 30 km/h and an obstacle on a curve.
- **Open issues:**
  - unneeded braking on the tight curve (4/10 runs) from path-entry prediction
  - early braking for a kerbside pedestrian while cornering
