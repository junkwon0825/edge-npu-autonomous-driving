# V2: asynchronous, frame-age-aware perception pipeline

**Engineering question.** How should an NPU perception runtime be designed so that low model latency
becomes low and predictable *control* latency?

**V2 answer.**
- Never queue: process only the newest frame.
- Stamp every result with its capture time.
- Let the controller reject results that are too old.
- Predict the tracked state to the control time.
- Fail safe when perception is missing.

Code: `src/npu_ad/runtime/pipeline.py`.

## Structure
```text
camera (33 Hz) ──► LatestFrameQueue(maxsize=1) ──► inference worker ──► newest result
                     drops the oldest frame            (one backend         │
                     when full (counted)                call at a time)     ▼
control loop (100 Hz) ◄── ResultGate: new / repeat / stale / failed ◄── poll()
        │
        └── tracker.update(capture_t, dets); views predicted to control time; AEB state machine
```

**Timestamps per result:**
- capture
- enqueue
- inference start
- inference end
- control (first use)

**Derived quantities:**
- queue delay = start − enqueue
- inference latency = end − start (the backend's own per-stage timing is kept too)
- frame age at use = control − capture

## Two executions of the same design
| | `AsyncInferenceWorker` (live) | `SimulatedAsyncPipeline` (matrix) |
|---|---|---|
| Clock | wall clock (`perf_counter`), simulation paced to it | simulation time |
| Worker | real thread, real backend (GPU or remote RNGD) | detector runs when a job starts; result released at start + latency |
| Latency | real | seeded `LatencyModel`: constant, uniform jitter, periodic bursts, drop probability, stale injection, or an empirical distribution of measured samples |
| Used for | Mode A/C live runs, failover tests, demo | deterministic CARLA matrix and latency stress |

In both, a busy worker cannot start a new frame. A slow backend therefore also lowers the perception
*rate*. V1 modelled latency as a pure actuation delay at 100 Hz, which hid this. A newer frame
replaces a waiting one, so queueing never accumulates latency.

## Control-side rules (`ResultGate`, controller)
- **Stale rejection.** A result older than `stale_s` = 0.3 s when it arrives is rejected and counted.
  It is never used as current perception.
- **Repeat.** A result already consumed is not "new perception" on later ticks. The tracker updates
  only once per result, using its *capture* time.
- **Failed results.** They are counted and ignored.
- **Perception timeout.** If no accepted result was captured within `perception_timeout_s` = 0.5 s,
  the controller runs in `degraded` mode and holds its last command. It never releases a brake on
  missing data, and it cannot start a new brake without data.
- **Latency compensation.** Track range is predicted from the last measurement time to the control
  time with the Kalman velocity. Lateral position is predicted with the lateral filter.

**Thresholds and their reasons:**
- **0.3 s stale reject.** At 30 km/h that is 2.5 m of travel, about the camera-to-bumper calibration
  scale. The measured remote-V2 p99 is 116 ms.
- **0.5 s timeout.** Longer than one dropped burst (0.25 s) plus the stale limit.

Neither threshold was tuned on the matrix.

## Tests (`tests/test_v2_runtime.py`)
- the queue drops the oldest frame (size 1 and 2)
- the gate's new / repeat / stale / failed logic
- the simulated pipeline:
  - limits the rate at 100 ms latency
  - drops frames
  - never shows a result before capture + latency
- the seeded latency model is reproducible (drop, stale, burst)
- the threaded worker processes the newest frame and its queue delay is non-negative
- failover switches, logs and recovers, and reports DEGRADED when both backends fail
- health sampling is bounded
- the remote protocol, deadline-bounded timeouts, backpressure, and a late-answer stale count against
  a fake server

Closed-loop results: `docs/v2_benchmark.md` and `docs/v2_aeb_controller.md`.
