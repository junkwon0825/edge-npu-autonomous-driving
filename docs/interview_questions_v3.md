# Interview questions and answers (based on what this project actually implemented)

**Why an NPU instead of a GPU?**
For this model at batch 1, the RNGD is faster and much more stable than an *optimised* GPU path:
- 2.0–2.1 ms vs 2.6 ms with CUDA Graphs
- jitter 0.007 ms vs 0.18 ms std
- about 53 W vs 136 W board power (different meters, indicative only)

Against eager PyTorch the gap looks like 6×, but that is launch overhead, not hardware. The real
argument is predictable latency and power per frame. In this setup the NPU's advantage disappears
behind transport, because it sits on a remote host.

**Why BF16?**
- It was the fastest variant (1.85–2.0 ms).
- It keeps detection agreement high (recall 0.972–0.974).
- The SDK path offered no usable INT8 compute.
- BF16 cost about 0.025 IoU on the lane model.

**Why did W8 not improve latency?**
W8 stores int8 weights and casts them to BF16 inside the graph, so the compute stays BF16 and the cast
adds work. At batch 1 YOLO26n is overhead-bound, not weight-bandwidth-bound, so the 49 % smaller
weights do not help. True W8A8 failed: int8 conv wraps around, and the int32-accumulating forms do
not compile.

**Why was SSH transport the bottleneck?**
Stage profiling of the V1 path (203 ms) showed:
- 100 ms network upload of a 1.23 MB letterboxed frame at about 11 MB/s
- 70–85 ms of remote CPU work: an oversubscribed thread pool on a loaded shared host
- 2 ms of NPU execution

The fixes:
- send only the content region
- pin 2 threads
- decode only anchors above the threshold (exactly the same detections)
- SSH compression

Together they reached 60 ms losslessly. A new SSH process per frame would cost 0.7 s.

**What is frame age?**
Control time minus capture time of the frame behind the result in use. It includes queueing,
transport, inference and waiting for the next control tick. Model latency alone understates it.
Example: GPU 22 ms of inference gave 66 ms age in a 20 Hz live loop.

**Why bounded latest-frame queues?**
An unbounded queue turns a slow backend into ever-growing latency. A size-1 queue that drops the
oldest frame keeps the age bounded by roughly one inference time, at the cost of a lower perception
rate. Drops are counted, never hidden.

**Why tracking?**
To get per-object identity, range rate and lateral motion over time: the inputs to TTC and to "will
it clear the path". Vehicles kept one track id with 98–99 % coverage. Pedestrians had 2 ids and 79 %
coverage, a known weakness.

**Why Kalman or α-β filtering?**
Finite differences of monocular range are useless (rate MAE 5.2 m/s). V1 used a 0.3 s least-squares
fit; the Kalman filter with range-dependent measurement noise lowered the error:

| Estimator | Rate MAE (33 Hz) | MAE beyond 30 m |
|---|---|---|
| V1 least squares | 1.51 m/s | 3.44 m/s |
| V2 Kalman | 1.27 m/s | 2.07 m/s |

Its reaction to hard braking was as fast as V1's.

The first Kalman settings were too smooth: the outlier gate treated a real brake as noise. That was
fixed by measuring with a sustained-crossing delay metric.

**How does TTC work here?**
TTC = range / closing speed, only when the closing speed exceeds 0.3 m/s and the track passed the
gates: confirmed, confidence, plausibility. Path prediction then checks whether the object will be in
the ego corridor during the ego's arrival window, and labels it collision_path, will_clear or
ego_passes. Thresholds: 1.6 s full brake, 2.6 s partial.

**How is this AEB different from a production AEB?**
- simulation only and camera monocular range (radar fusion is V3 stage 1)
- no driver model or warning HMI, and no NCAP test catalogue
- a single map family; no functional-safety process (ISO 26262, SOTIF)
- the controller was tuned and evaluated on the same scenario family

It is a research prototype for studying perception latency.

**How did sensor fusion improve robustness?**
In general it did not, and the measurements say where it helps and where it hurts.
- **Helps:** a lead vehicle braking at 50 km/h. Closing-speed MAE fell from 0.40 to 0.24 m/s and the
  minimum gap grew from 5.5 to 10.4 m.
- **Hurts:** a pedestrian at 30 km/h (0.55 → 1.14 m/s) and a static obstacle on a curve (0.34 → 2.85 m/s).
- **Likely cause:** association picks returns from other surfaces inside the gate. Not proven yet.
- **Next step:** log the radar clusters behind each fused update.

**Your NPU lane mode failed at night and in rain. Was it the NPU?**
No.
- **Same frames:** RNGD output equals GPU BF16 (IoU 0.991–0.998 against FP32).
- **Logs:** the failing runs had RNGD requests (detection too) timing out at 1 s. They were the last
  runs before the machine ran out of memory.
- **Rerun in a healthy session:** the same cases had 0 departures.
- **Real defects:** a missing lane fallback in those modes, and a double resize in the NPU client's
  preprocessing (2–6 % IoU).

**How old can perception be before control fails?**
- **Lane keeping:** about 0.25 s. Up to +150 ms of injected lane latency still kept the lane. At +250 ms
  the 0.3 s stale gate rejects about half the results and LKA is unavailable 57–65 % of the time.
- **Braking:** the remote NPU's extra ~90 ms of detection age cut the margin at 50 km/h on a curve from
  13.7 m to 2.1 m.

**How are multiple models scheduled on the NPU?**
One model per PE (`furiosa:0` detection, `furiosa:1` lane), each with its own worker, fed by one upload
per frame. Measured:
- **At 30 Hz:** every strategy serves every frame within 6 ms.
- **Under a 400 Hz stress:** separate PEs keep detection at full rate. Two workers on the same PE are
  worse than sequential. Priority scheduling protects detection and starves the lane model.

**What causes accelerator utilisation bottlenecks here?**
- at batch 1, fixed overhead (exec time barely changes from 320 to 640 input)
- the host side: transport, CPU pre/post-processing, Python threads; 8 threads on 8 PEs became
  host-bound
- not compute

Without hardware counters these statements are qualitative.

**What unsupported operators did you hit?**
- ConvTranspose, bilinear interpolation, dilated conv
- unfused BatchNorm, `round`, fake-quant
- max/avg pooling at large sizes, 2×2 stride-2 conv
- large 3×3 convs (a 64 KiB weight-slice limit in a lowering pass)

Most have exact rewrites. Pooling needed an architecture change (3×3 stride-2 conv), and ResNet18
still needs an input-channel split.

**What would you change for production?**
- co-locate the NPU with the camera; drop SSH and use zero-copy capture into device memory
- C++ runtime with a real-time scheduler, no Python GIL
- static memory and a watchdog
- validated sensor fusion with radar and LiDAR
- a held-out scenario catalogue, and tuning separated from evaluation
- functional-safety work
- fault injection in CI
- calibrated quantization once the toolchain supports it
