# Resume project summary (English)

**Project:** Edge-NPU Autonomous Driving Perception & Control System. A personal research prototype, in
simulation only (CARLA 0.9.16). Every number comes from recorded runs in `reports/evidence/`.

## One-line version
I deployed detection and lane-segmentation models on a Furiosa RNGD NPU and built a multi-model,
failover-capable runtime. I then integrated it into closed-loop CARLA emergency braking and lane keeping,
and quantified how system latency, not model latency, changes vehicle behaviour.

## Three-bullet resume version
- **NPU porting:** ported YOLO26n (2.0 ms BF16) and a compiler-aware lane U-Net (2.8 ms) to Furiosa
  RNGD.
  - Diagnosed unsupported operators with a 74-entry compatibility database and six exact graph
    rewrites, which also made YOLO26s compile.
  - Proved by compiler probes that INT8 compute was unavailable on the SDK.
- **Runtime and transport:** built an asynchronous, frame-age-aware runtime: latest-frame queues,
  stale-result gates, a two-PE multi-model server and logged GPU failover.
  - Cut the remote NPU round trip from 203 to 60 ms losslessly by stage profiling.
  - Scaled to 1,762 inferences/s on 4 PEs.
- **Closed-loop evaluation:** evaluated camera-based AEB + LKA (with radar fusion) in 80 CARLA
  closed-loop runs: 36 hazard runs with 0 collisions, and lane RMSE 0.8–2.6 cm on straight, mild-curve, night and rain routes.
  - Reported the unsafe cases too: remote data age cut a 50 km/h braking margin from 13.7 m to
    2.1 m.
  - Traced a night/rain lane failure to request timeouts, not NPU precision.

## Detailed portfolio version
**Problem:** a fast NPU model does not by itself make a safe driving loop. The project measures where
time goes between the camera and the actuator, and what that does to braking and steering.

**V1: GPU closed loop + NPU validation**
- **Closed loop:** CARLA camera → YOLO26n → monocular range / TTC → AEB, with 0 collisions in 30 hazard
  runs at 30 km/h.
- **NPU deployment:** YOLO26n on RNGD, after diagnosing compile failures (unfused BatchNorm, head
  concat). The model was cut at the raw head, with decode/NMS on the host, and validated 72/72 frames
  against the original.
- **Fair GPU baseline:** CUDA Graph at 2.60 ms vs eager at 13.54 ms. This reduced an apparent 6× NPU
  advantage to 1.2–1.4×.
- **INT8:** W8 is weight-only. It shrinks weights 4.82 → 2.44 MB but is slower (2.35 vs 2.00 ms), and
  true INT8 compute is blocked by the toolchain.

**V2: async runtime + robust AEB**
- **Transport:** stage profiling of the remote path (203 → 60 ms). The fixes were a smaller payload,
  thread pinning, an exact pre-filter and SSH compression.
- **Runtime:**
  - backend abstraction (GPU / RNGD / replay)
  - latest-frame queues
  - frame-age gates
  - explicit failover
- **Perception and braking:**
  - a SORT-style tracker with a range Kalman filter
  - a path-prediction AEB, with 0 / 18 hazard collisions (V1: 2 / 18) and collision-free behaviour up
    to 200 ms of injected latency
- **Live loop:** remote RNGD runs in the live CARLA loop, but only near-real-time (frame age p95
  165–212 ms).

**V3: multi-model NPU perception + LKA + sensor fusion**
- **Lane model:** a lane U-Net designed around measured compiler limits. Pooling fails at 256×512, so
  it uses 3×3 stride-2 downsampling and exact upsampling rewrites.
- **Multi-model server:** detection on PE 0 and lane on PE 1, from one upload (61 ms vs 55 ms for
  detection alone, remote). A request-coalescing dispatcher fixed a lane-starvation bug.
- **Control:**
  - camera-only pure-pursuit LKA, with lane-age compensation and a 0.5 s timeout
  - an arbiter that clamps steering to ±0.1 while braking
  - radar fusion stage 1
- **Evaluation:** 80 closed-loop runs. They included lane keeping (A–D modes × routes), combined AEB +
  LKA on curves, radar pairs, scheduling strategies, fault injection and a lane-age sweep.
- **Results:**
  - 36 hazard runs, 0 collisions
  - healthy-session runs, 0 lane departures
  - lane-age tolerance about 0.25 s
  - 29,618 braking ticks, all with |steer| ≤ 0.10
- **Failure analysis:** a night/rain lane failure looked like a BF16 issue.
  - A same-frame GPU vs NPU test on 440 frames showed RNGD = GPU BF16 (IoU 0.991–0.998).
  - The cause was remote request timeouts under host memory pressure, combined with no lane fallback.
  - Healthy reruns recovered (0 departures).

**Honest limits:**
- remote data age at 50 km/h (2.1 m margin)
- 20 % stale detections (0.33 m gap)
- radar fusion worse in two scenario types
- unneeded braking on a tight curve
- one run per cell
- simulation only

## Technical interview version
- **Model latency vs system latency:**
  - RNGD executes YOLO26n in 2.0 ms.
  - In the loop the remote request takes 63–65 ms (mean), and detection data age is 121–126 ms (p95),
    vs 30 ms on the local GPU.
  - Control depends on data age.
- **Porting method:**
  1. Read the compiler error.
  2. Write an exact rewrite, validated against the original op on CPU.
  3. Recompile.
  4. Where a constraint is shape-dependent (pooling), change the architecture before training.

  This came to six rewrites in total. YOLO26s needed an output-channel SplitConv for a 64 KiB
  weight-slice limit; ResNet18 is still open (it needs an input-channel split).
- **Scheduling:**
  - Offline, separate PEs per model are best; two workers on one PE are worse than sequential.
  - In the loop, transport dominates, so all strategies give the same driving outcome. Adaptive
    priority still cuts dropped frames from 170 to 21.
- **Failover:** PRIMARY / FALLBACK / DEGRADED, every switch logged; stall → GPU in 0.54 s, crash →
  GPU in 0.11 s. Fallback latency includes the failed NPU wait (an earlier bug hid it).
- **Debugging discipline:** three simulator/runtime bugs were found by questioning results that looked
  too good:
  - Mode D lane aliasing
  - the hidden failover wait
  - ineffective stale injection

  Each was fixed with a unit test, and the affected runs were re-run separately.
- **Next steps:**
  - co-locate the NPU with the camera (zero-copy, C++ client)
  - lane fallback by default
  - unified preprocessing
  - AEB target logging
  - repeated runs with confidence intervals
