# V2 CARLA scenario tables (generated)

Source: `reports/evidence/v2_scenarios.json` (142 runs; code commit(s) 933a1fe, 9d2afb7).
Town10HD_Opt, dt 0.01 s, camera 33 Hz, perception = GPU YOLO26n detections released by the async pipeline
after the condition's latency. Min gap = approach phase only (until stop / contact / passing). `aeb_off` = no AEB.

## A. Controllers at the measured in-loop GPU latency (`gpu_local`)

### 30 km/h

| Scenario | V2 | V1 (perception-driven) | AEB off |
|---|---|---|---|
| static_obstacle | gap 16.5 m, stop | gap 24.7 m, stop | **COLLISION** (vehicle) |
| lead_vehicle_braking | gap 8.9 m, stop | gap 9.9 m, stop | **COLLISION** (vehicle) |
| pedestrian_crossing | gap 11.9 m, stop | gap 2.0 m, stop | **COLLISION** (walker) |
| cyclist_crossing | gap 8.0 m, stop, resumed | gap 4.5 m, stop | gap 0.1 m, no stop |
| rain | gap 12.4 m, stop | gap 1.7 m, stop | **COLLISION** (walker) |
| low_light | gap 8.0 m, stop | gap 8.3 m, stop | **COLLISION** (vehicle) |
| pedestrian_clearing | gap 0.1 m, no stop | **unneeded brake** ×2, gap 11.5 m, stop | gap 0.1 m, no stop |
| pedestrian_curbside | no stop | no stop | no stop |
| adjacent_vehicle | no stop | no stop | no stop |
| straight_road | no stop | no stop | no stop |

### 40 km/h

| Scenario | V2 | V1 (perception-driven) | AEB off |
|---|---|---|---|
| static_obstacle | gap 16.9 m, stop | gap 24.0 m, stop | **COLLISION** (vehicle) |
| lead_vehicle_braking | gap 6.8 m, stop, 2 onsets | gap 10.8 m, stop | **COLLISION** (vehicle) |
| pedestrian_crossing | gap 9.1 m, stop, 2 onsets, resumed | **COLLISION** (walker) | **COLLISION** (walker) |
| cyclist_crossing | gap 2.3 m, stop, resumed | gap 3.5 m, stop | gap 0.1 m, no stop |
| rain | gap 10.0 m, stop, resumed | **COLLISION** (walker) | **COLLISION** (walker) |
| low_light | gap 4.9 m, stop, 2 onsets | gap 15.6 m, stop | **COLLISION** (vehicle) |
| pedestrian_clearing | gap 0.1 m, no stop | **unneeded brake** ×1, gap 8.6 m, stop | gap 0.1 m, no stop |
| pedestrian_curbside | no stop | no stop | no stop |
| adjacent_vehicle | no stop | no stop | no stop |
| straight_road | no stop | no stop | no stop |

### 50 km/h

| Scenario | V2 | V1 (perception-driven) | AEB off |
|---|---|---|---|
| static_obstacle | gap 16.0 m, stop | gap 18.5 m, stop | **COLLISION** (vehicle) |
| lead_vehicle_braking | gap 5.2 m, stop, 2 onsets | gap 7.7 m, stop, 2 onsets | **COLLISION** (vehicle) |
| pedestrian_crossing | gap 2.3 m, stop, resumed | gap 0.1 m, no stop | gap 0.1 m, no stop |
| cyclist_crossing | gap 0.0 m, stop, resumed | gap 0.0 m, stop | **COLLISION** (vehicle) |
| rain | gap 9.1 m, stop, resumed | gap 0.1 m, no stop | gap 0.1 m, no stop |
| low_light | gap 3.8 m, stop, 2 onsets | gap 16.1 m, stop, 2 onsets | **COLLISION** (vehicle) |
| pedestrian_clearing | gap 0.1 m, no stop | gap 0.1 m, no stop | gap 0.1 m, no stop |
| pedestrian_curbside | no stop | no stop | no stop |
| adjacent_vehicle | **unneeded brake** ×1, no stop | no stop | no stop |
| straight_road | no stop | no stop | no stop |

### Totals (gpu_local, all speeds)

| Controller | Hazard runs | Collisions | No-brake-needed runs | Runs with unneeded braking | Stopped and resumed |
|---|---|---|---|---|---|
| v2 | 18 | 0 | 12 | 1 | 7 |
| v1 | 18 | 2 | 12 | 2 | 0 |
| aeb_off | 18 | 14 | 12 | 0 | 0 |

## B. Latency / runtime stress (30 km/h)

| Scenario | Condition | V2 | V1 | V2 frame age mean / p95 (ms) | V2 stale rejected | V2 dropped frames | V2 degraded ticks |
|---|---|---|---|---|---|---|---|
| pedestrian_crossing | fixed_0ms | gap 12.1 m, stop | gap 2.2 m, stop | 0 / 0 | 0 | 0 | 0 |
| pedestrian_crossing | fixed_50ms | gap 11.3 m, stop | gap 1.0 m, stop | 70 / 80 | 0 | 84 | 5 |
| pedestrian_crossing | fixed_100ms | gap 11.0 m, stop | **COLLISION** (walker) | 119 / 130 | 0 | 149 | 10 |
| pedestrian_crossing | fixed_150ms | gap 10.1 m, stop | **COLLISION** (walker) | 174 / 180 | 0 | 173 | 15 |
| pedestrian_crossing | fixed_200ms | gap 8.5 m, stop, resumed | **COLLISION** (walker) | 219 / 230 | 0 | 283 | 20 |
| pedestrian_crossing | jitter_50pm40ms | gap 11.6 m, stop | gap 0.4 m, stop | 62 / 100 | 0 | 78 | 4 |
| pedestrian_crossing | burst_25ms_250ms | gap 11.6 m, stop, resumed | gap 2.0 m, stop | 60 / 280 | 0 | 160 | 71 |
| pedestrian_crossing | drop30_25ms | gap 11.6 m, stop, resumed | gap 1.0 m, stop | 30 / 30 | 0 | 0 | 9 |
| pedestrian_crossing | stale20_25ms | gap 9.1 m, stop | **COLLISION** (walker) | 31 / 40 | 12 | 156 | 129 |
| pedestrian_crossing | npu_colocated | gap 12.0 m, stop | gap 1.2 m, stop | 21 / 40 | 0 | 0 | 2 |
| pedestrian_crossing | npu_remote_v2 | gap 11.1 m, stop | **COLLISION** (walker) | 92 / 120 | 0 | 122 | 7 |
| pedestrian_crossing | npu_remote_jpeg95 | gap 11.4 m, stop | gap 1.7 m, stop | 54 / 80 | 0 | 32 | 3 |
| pedestrian_crossing | npu_remote_v1equiv | **COLLISION** (walker) | **COLLISION** (walker) | — | 7 | 127 | 407 |
| lead_vehicle_braking | fixed_0ms | gap 9.2 m, stop | gap 10.1 m, stop | 0 / 0 | 0 | 0 | 0 |
| lead_vehicle_braking | fixed_50ms | gap 9.2 m, stop | gap 9.5 m, stop | 70 / 80 | 0 | 113 | 5 |
| lead_vehicle_braking | fixed_100ms | gap 9.0 m, stop | **COLLISION** (vehicle) | 119 / 130 | 0 | 199 | 10 |
| lead_vehicle_braking | fixed_150ms | gap 8.9 m, stop | **COLLISION** (vehicle) | 171 / 180 | 0 | 228 | 15 |
| lead_vehicle_braking | fixed_200ms | gap 8.6 m, stop | **COLLISION** (vehicle) | 219 / 230 | 0 | 243 | 20 |
| lead_vehicle_braking | jitter_50pm40ms | gap 8.9 m, stop | gap 9.8 m, stop | 69 / 110 | 0 | 118 | 8 |
| lead_vehicle_braking | burst_25ms_250ms | gap 9.3 m, stop | gap 2.9 m, stop, 4 onsets | 62 / 280 | 0 | 144 | 67 |
| lead_vehicle_braking | drop30_25ms | gap 9.2 m, stop | gap 9.9 m, stop | 30 / 30 | 0 | 0 | 3 |
| lead_vehicle_braking | stale20_25ms | gap 8.8 m, stop | gap 0.7 m, stop | 31 / 40 | 14 | 193 | 158 |
| lead_vehicle_braking | npu_colocated | gap 8.9 m, stop | gap 9.9 m, stop | 23 / 50 | 0 | 4 | 1 |
| lead_vehicle_braking | npu_remote_v2 | gap 9.3 m, stop | gap 5.3 m, stop, 2 onsets | 88 / 119 | 0 | 160 | 7 |
| lead_vehicle_braking | npu_remote_jpeg95 | gap 9.3 m, stop | gap 9.8 m, stop | 56 / 80 | 0 | 50 | 5 |
| lead_vehicle_braking | npu_remote_v1equiv | **COLLISION** (vehicle) | **COLLISION** (vehicle) | 280 / 280 | 11 | 185 | 572 |

## C. Tracking and estimation quality (V2 runs, GT target visible)

| Scenario | runs | covered by a confirmed track (median) | distinct track ids for the target (median / max) | longest uncovered gap, frames (median) | range MAE m (median) | rate MAE m/s (median) | rate p95 m/s (median) |
|---|---|---|---|---|---|---|---|
| static_obstacle | 3 | 0.99 | 1 / 1 | 3 | 0.63 | 0.97 | 3.45 |
| lead_vehicle_braking | 16 | 0.98 | 1 / 1 | 6 | 0.31 | 0.64 | 2.49 |
| pedestrian_crossing | 16 | 0.79 | 2 / 6 | 38 | 0.64 | 1.34 | 5.32 |
| cyclist_crossing | 3 | 0.55 | 1 / 2 | 49 | 0.92 | 2.85 | 7.74 |
| rain | 3 | 0.89 | 4 / 5 | 23 | 0.72 | 1.91 | 8.22 |
| low_light | 3 | 0.99 | 1 / 1 | 3 | 0.34 | 0.74 | 2.74 |
| pedestrian_clearing | 3 | 0.18 | 1 / 1 | 148 | 0.49 | 1.51 | 8.09 |
| pedestrian_curbside | 3 | 0.42 | 1 / 1 | 46 | 0.20 | 0.87 | 2.79 |
| adjacent_vehicle | 3 | 0.90 | 1 / 1 | 12 | 1.05 | 1.33 | 3.71 |
