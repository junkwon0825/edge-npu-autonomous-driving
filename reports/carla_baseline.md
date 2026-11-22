# CARLA Baseline and Reproducibility (Phase 1, Gate B)

**Status: PASS** — evidence `reports/evidence/capture_smoke.json`, test `tests/test_carla_live.py`.

## Setup
- CARLA 0.9.16 server, started with `CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2000`; client 0.9.16.
- Map `Town10HD_Opt` (server default; no map reload).
- Synchronous mode, `fixed_delta_seconds = 0.05` (capture/dataset) or `0.01` (closed-loop scenarios),
  Traffic Manager synchronous with `set_random_device_seed(42)`, Python `random.seed(42)`.
- Ego `vehicle.tesla.model3`; RGB camera 800×450, FOV 90°, mounted at x=1.5 m, z=1.7 m,
  `sensor_tick = 0` (one image per tick).

## Implementation (`src/npu_ad/simulation/carla_env.py`)
- `CarlaSession` context manager: connects, checks server/client versions, applies synchronous
  settings, seeds TM, and **always** restores the original settings on exit (also on exceptions).
- Every spawned actor is tracked; teardown stops sensors / walker controllers first, then destroys
  all tracked actors in reverse spawn order with one `apply_batch_sync` call.
- `get_image(sensor, frame)` returns only the image whose `frame` equals the ticked frame, drops
  older ones and fails fast on a newer one (desynchronisation).

## Smoke test results (3 runs × 100 frames, 5 warm-up ticks excluded)

| Run | Spawn index | Frames | Contiguous frame ids | Saved (every 10th) | Re-read OK | Orphans | Wall time |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 100 | yes | 10 | yes | 0 | 4.64 s |
| 1 | 7 | 100 | yes | 10 | yes | 0 | 4.38 s |
| 2 | 14 | 100 | yes | 10 | yes | 0 | 4.26 s |

Image shape (450, 800, 3) for every frame; saved PNGs are non-degenerate (mean 110–130, std 40–51).

## Seeded evaluation / calibration dataset (`scripts/collect_dataset.py`)
- 4 weather presets × 18 frames = 72 frames: `ClearNoon`, `WetCloudySunset`, `HardRainNoon`, `ClearNight`.
- 40 NPC vehicles (autopilot) + 30 AI walkers per preset, seed `42 + preset index`; frames every
  20 ticks after a 40-tick settling period.
- Deterministic split stored in `outputs/dataset/manifest.json`: every 4th frame → `calib` (18), rest → `eval` (54).
  Calibration frames are never used for reported agreement metrics.
- Two `destroy error: unable to destroy actor: not found` warnings occurred in the first preset
  (actors already removed server-side); a post-run query confirmed 0 vehicles, 0 walkers, 0 sensors left
  and synchronous mode off.

## Known limitations
- Determinism: synchronous mode + fixed seeds make actor placement reproducible, but CARLA physics and
  Traffic Manager behaviour are not guaranteed bit-identical across runs/GPUs. Frame-id ranges differ
  between runs (server frame counter is global).
- The pre-existing learning scripts in `code/` were left unchanged; `code/09_yolo_detection.py` still
  contains the duplicated `camera.listen()` call.
