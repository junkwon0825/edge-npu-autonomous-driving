# Architecture

## Data flow

```text
 CARLA 0.9.16 server (synchronous, dt = 0.01 s scenarios / 0.05 s capture)
   │  world.tick() -> frame id F
   ▼
 CarlaSession.get_image(cam, F)     exactly one listener per sensor; older frames dropped,
   │  BGRA 800x450                   a newer frame raises FrameTimeout (fail fast)
   ▼
 letterbox 640x640 (yolo_raw.letterbox, shared by GPU, NPU and replay paths)
   │
   ├── GPU path (in the loop) ───────── GpuDetector: fused YoloRaw on cuda:0 (CUDA-event timing)
   └── NPU path (offline) ──────────── remote/npu_job.py on the RNGD host: CompileModule on furiosa:0
   │   raw outputs: boxes (1,4,8400) + score logits (1,80,8400)
   ▼
 RawDecoder (host): anchors, dist2bbox, sigmoid, NMS (conf 0.25, IoU 0.7)  -> scale_boxes to 800x450
   │   detections [x1,y1,x2,y2,conf,cls]
   ▼
 AEB target (one of three input modes, scripts/run_scenarios.py --modes)
   ├── gt_oracle : CARLA ground-truth gap / closing speed, perception ignored
   ├── gt_gated  : ground truth, braking only while the GT target is inside a same-class box
   └── mono      : GroundPlaneRanger (flat-ground pinhole range from the box bottom edge)
   │               + CipoTracker (closest in-path object, range-rate by least squares)
   ▼
 AebController (TTC/gap thresholds, perception hold, v2 hysteresis)  -> Decision(brake)
   ▼
 DelayLine(delay_ticks)  — perception/transport latency modelled as actuation delay
   ▼
 ego.apply_control(throttle, steer from lane follower, brake)
```

## Modules

| Module | Responsibility | Depends on CARLA? |
|---|---|---|
| `npu_ad/simulation/carla_env.py` | session lifecycle: sync mode, TM seed, actor tracking, sensor queues, teardown that restores settings even when spawning or destroying fails | yes (injected, fake in tests) |
| `npu_ad/simulation/geometry.py` | GT geometry: ego frame, bumper gap, closing speed, intrinsics, projection | no |
| `npu_ad/perception/yolo_raw.py` | `YoloRaw` (accelerator part), `RawDecoder` (host part), letterbox | no |
| `npu_ad/perception/gpu_runner.py` | CUDA execution with stage timing | no |
| `npu_ad/perception/mono_range.py` | perception-only range and closing speed | no |
| `npu_ad/control/aeb.py` | AEB decision logic, delay line | no |
| `npu_ad/evaluation/metrics.py` | latency statistics, detection matching | no |
| `remote/npu_job.py`, `remote/int8_probe.py` | run on the RNGD host only | no |
| `scripts/*` | runnable experiments; all write evidence to `reports/evidence/` | some |

Control and evaluation code has no CARLA import; ground truth reaches the controller only in the
two GT modes, and the `mono` mode receives detections only. Ground truth is always *logged* so that
every perception-driven run can be scored (range error, TTC at brake onset, misses).

## Why the model is split at the raw head
Earlier RNGD compiles of YOLO failed on BatchNorm (not importable) and on the head's concat
(`select_scatter` lowering). The deployed unit is therefore the fused backbone+neck+head convs with a
static 1×3×640×640 input; anchor decode, sigmoid and NMS (dynamic shapes, data-dependent control
flow) run on the host with the same code for both accelerators. The split + fusion reproduces the
unfused ultralytics model on all 72 dataset frames (348/348 detections,
`reports/evidence/raw_split_validation_all_cpu.json`).

## Timing model and its limits
- The detector runs on every 10 ms simulation tick (a 100 Hz camera). Inference wall time does not
  advance simulation time; latency enters only through `DelayLine` (actuation delay). Latency
  conditions come from measurements (`configs/latency_conditions.json`).
- `npu_p95` assumes the NPU sits next to the camera (device-side p95 + host pre/post p95).
  `npu_remote_p95` adds the measured SSH round trip of one raw frame to the remote RNGD host.
- The NPU is not in the closed loop. NPU perception is compared with GPU perception on the same frames
  by open-loop replay (`scripts/replay_aeb.py`).

## Failure handling
| Failure | Behaviour |
|---|---|
| CARLA unreachable / TM port busy during `__enter__` | session closes itself, restores settings, re-raises |
| exception inside a run | `with CarlaSession` tears down actors (sensors first) and restores settings |
| batch destroy RPC error | logged; settings restore still runs |
| missing / out-of-order camera frame | `FrameTimeout` aborts the run (no silent desync) |
| scenario matrix interrupted | results JSON is written atomically after each run; rerun skips finished keys |
| second NPU job started while one runs | `npu_remote.py start` refuses (`BUSY`) |
| remote job failure | `status_<cmd>.json` records `FAIL`, error type, traceback tail |
