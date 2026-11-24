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
- BF16 itself barely changes the lane masks: on the same input, BF16 masks agree with FP32 at IoU
  0.991–0.998. The first held-out measurement (GT IoU 0.688 GPU FP32 vs 0.664 RNGD) also included a
  preprocessing difference (double resize), so that 0.025 gap should not be read as the cost of BF16.

**What does CUDA Graph change, and why use it as the baseline?**
Eager PyTorch at batch 1 is dominated by kernel-launch overhead: 13.54 ms mean for YOLO26n on the RTX
3060. Capturing the forward pass once as a CUDA Graph and replaying it removes most of that overhead:
2.60 ms mean, with outputs bit-identical to eager. Comparing the NPU against eager would have claimed a
6× advantage. Against the optimised GPU path the honest figure is 1.2–1.4×.

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

**What graph rewrites did you use, and how did you know they were safe?**
Six exact rewrites, each compared against the original op on CPU before compiling:
- BatchNorm folding into the preceding convolution
- bilinear ×2 upsampling → nearest + a fixed [1, 2, 1] filter
- ConvTranspose → 1×1 conv + PixelShuffle
- dilated conv → a dense kernel with inserted zeros
- SplitConv: a large conv split by output channels and concatenated (this made YOLO26s compile)
- contiguous q/k/v before attention

For YOLO26n the deployment boundary also moved: the fused model is cut at the raw head, and decode/NMS
run on the host. The split model matched the unfused original on 72/72 frames.

**How did you port the lane-segmentation model?**
The compiler constraints were measured first, then the network was designed around them:
- **Probing:** small probe models showed that ConvTranspose, bilinear and dilated conv were
  unsupported, and that pooling compiled at small sizes but failed at 256×512.
- **Design and training:** a 2.12 M-parameter U-Net was built from supported pieces only (3×3 stride-2
  downsampling, exact upsampling rewrites) and trained on CARLA data.
- **Result on RNGD:** 2.8 ms in BF16, and the same output as GPU BF16 on the same input.

The lesson: "supported" is not the same as "supported at your shape".

**What is the difference between model latency, end-to-end latency and data age?**
- **Model latency:** accelerator execution. 2.0 ms for YOLO26n BF16 on RNGD.
- **End-to-end request latency:** the client round trip, including transport. 63–65 ms (mean) for the
  remote RNGD in the loop.
- **Data age:** how old the result is when control uses it. 121–126 ms (p95) for remote detection,
  vs 30 ms on the local GPU.

Control quality depends on data age, so a 2 ms model can still give a 2.1 m braking margin.

**What happens when perception results are stale?**
Every result carries its capture time. Two gates use it:
- **Result gate:** rejects anything older than 0.3 s.
- **LKA:** becomes unavailable if no lane result arrives for 0.5 s, so it never steers on stale geometry.

Stale results that pass the gate are still dangerous. With 20 % stale detections and RNGD latency, the
car stopped 0.33 m from a pedestrian. The first version of that test was ineffective: the
measured-latency simulation silently replaced the injected delay. It was found because the result looked
too good, then fixed and unit-tested.

**How does runtime failover work?**
`FailoverBackend` has three states: PRIMARY (RNGD), FALLBACK (GPU) and DEGRADED.
- **Switching:** a timeout or an error moves it to FALLBACK, and every switch is logged.
- **Offline:** stall → GPU in 0.54 s, crash → GPU in 0.11 s, 0 frames lost.
- **In CARLA:** after a stall it returned to RNGD. After a server crash it did not return within the
  12 s run.
- **Latency:** the fallback result carries the time spent waiting for the failed NPU request. An earlier
  version hid that wait, which made failover look free.

**How are AEB and LKA arbitrated?**
A fixed precedence in `Arbiter`:
1. **AEB braking:** throttle 0, AEB brake, and steering = LKA clamped to ±0.1. There is no evasive
   steering.
2. **LKA unavailable:** steering decays to 0 and the car coasts.
3. **Normal:** LKA steering plus the speed controller.

In 29,618 logged braking ticks the arbiter was always in an AEB mode with |steer| ≤ 0.10. Lateral
error while braking stayed ≤ 12 cm on curves.

**What safety limitations remain?**
- **Margin:** remote NPU data age cuts the 50 km/h curve braking margin from 13.7 m to 2.1 m.
- **Stale detections** can bring the car within 0.33 m of a pedestrian.
- **Unneeded braking:** on a 39 m-radius curve in 4 of 10 runs, and early braking for a kerbside
  pedestrian. The root cause is open because the target state was not logged.
- **Radar fusion** worsens closing speed for a pedestrian and on a curve.
- **No lane fallback:** RNGD lane modes have none by default.
- **Evaluation:** one run per cell; developed and evaluated on the same scenario family; simulation only.

Zero collisions in 36 hazard runs does not show general safety.

**What would you change for production?**
- co-locate the NPU with the camera; drop SSH and use zero-copy capture into device memory
- C++ runtime with a real-time scheduler, no Python GIL
- static memory and a watchdog
- validated sensor fusion with radar and LiDAR
- a held-out scenario catalogue, and tuning separated from evaluation
- functional-safety work
- fault injection in CI
- calibrated quantization once the toolchain supports it
