# Key results (verified only)

Simulation-based research prototype, not a production system. Each line points to its evidence in
`reports/evidence/`.

## Accelerator deployment (Furiosa RNGD, furiosa-torch 2026.4.0)
- **YOLO26n:**
  - Runs on RNGD in BF16 at 2.0 ms per frame (batch 1), with recall 0.974 / precision 0.988 against
    GPU FP32. *(npu_bench_bf16, v2_transport_profile_r2)*
  - Scales to 1,762 inferences/s on 4 PEs (3.3×). *(v2_npu_multistream2)*
- **Lane-segmentation network:** designed around the measured compiler limits, trained on CARLA data,
  and running on RNGD.
  - IoU 0.664 on a held-out map (GPU FP32 0.688).
  - 2.8 ms per frame.
  *(v3_lane_training, v3_npu_multimodel_bench)*
- **YOLO26s:** failed to compile (a 64 KiB weight-slice limit) and runs after an exact split-conv
  rewrite (4.67 ms). *(v3_npu_compile_yolo26s_bf16, v3_npu_split_conv)*

## NPU engineering
- **Operator compatibility database:** 74 recorded compile/run results with the compiler's own
  failure reasons. *(docs/npu_operator_compatibility.md)*
- **Six exact graph rewrites** validated against the original op on CPU: BN folding, bilinear ×2,
  ConvTranspose, dilated conv, split conv, contiguous attention.
- **INT8 compute and activation quantization:** shown to be unavailable on this SDK path, by compiler
  probes rather than assumption. *(npu_int8_probe*, v2_npu_port_lab_quant*)*

## System-level latency
- **Remote NPU path: 203 → 60 ms** (lossless). The causes were found by stage profiling: oversized
  payload, oversubscribed host threads, decode work. *(v2_transport_profile*)*
- **Two models on two PEs:** detection + lane from one upload costs 61 ms vs 55 ms for detection
  alone. *(v3_npu_multimodel_bench)*
- **Scheduling:** separate PEs per model keep detection at 400 Hz under stress; two workers on one PE
  do worse than sequential execution. *(v3_npu_multimodel)*

## Safety logic (CARLA, 30–50 km/h, measured in-loop GPU latency)
- **V2 AEB:** 0 / 18 hazard collisions (V1: 2 / 18, AEB off: 14 / 18). It avoids V1's 40 km/h
  pedestrian collision. *(v2_scenarios)*
- **Frame-age-aware runtime:** collision-free from 0 to 200 ms perception latency, and under jitter,
  bursts, 30 % dropped and 20 % stale results. V1 collides from 100 ms. *(v2_scenarios)*
- **Range-rate filtering:** MAE 1.51 → 1.27 m/s (offline), 0.64–1.34 m/s in closed loop.
  *(v2_filter_eval, v2_scenarios)*

## Robustness
- **Explicit GPU failover:** 0.54 s on an NPU stall and 0.11 s on a server crash, with 0 frames lost,
  every switch logged, and recovery to the NPU. *(v3_failure_engineering)*
- **Live CARLA → remote RNGD → AEB loop:** works but is near-real-time (frame-age p95 165–212 ms);
  no real-time claim. *(v2_live_loop)*
