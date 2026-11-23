# GPU vs NPU: what was measured and what it means

Same model semantics (fused YOLO26n raw module, static 1×3×640×640), same letterboxed CARLA frames
(72 frames: 18 calibration + 54 evaluation split), same host decode/NMS code for both. Batch 1 unless
stated. Hardware: RTX 3060 12 GB (desktop, Windows) vs Furiosa RNGD (remote shared host).

> **Headline.** The NPU (2.1 ms FP32 / 1.85–2.0 ms BF16) is about 6× faster than the GPU's *eager*
> PyTorch path (13.5 ms) — but eager PyTorch at batch 1 is launch-bound. With CUDA Graphs the same GPU
> model runs in **2.60 ms** (bit-identical outputs). Against that, the RNGD is 1.2–1.4× faster per
> frame at batch 1 and draws ~53 W board power vs ~136 W, i.e. roughly **3× less energy per frame**
> (≈ 99–113 mJ vs 350 mJ, two different power meters, see caveats). For closed-loop use from this
> desktop the NPU is **not** usable in real time, because moving a frame to the remote host costs
> 121 ms. No claim of "NPU is N× faster than GPU" is made as a hardware statement.

## 1. Model-only latency (accelerator time, inputs already prepared)
| Path | n | mean | p50 | p95 | p99 | std | Source |
|---|---|---|---|---|---|---|---|
| GPU eager FP32, CUDA events (baseline 2026-11-22) | 360 | 13.54 | 12.64 | 18.43 | 25.19 | 2.62 | `gpu_benchmark_fp32_clean.json` |
| GPU eager FP32, CUDA events (re-measured 2026-11-23) | 300 | 13.79 | 12.17 | 21.76 | 25.23 | 3.50 | `gpu_benchmark_ext.json` |
| **GPU CUDA Graph FP32** | 300 | **2.60** | 2.50 | 3.01 | 3.23 | 0.18 | `gpu_benchmark_ext.json` (max abs diff vs eager 0.0) |
| RNGD FP32, `bench` (per-frame H2D before, stream sync) | 216 | 2.107 | 2.106 | 2.119 | 2.125 | 0.007 | `npu_bench.json` |
| RNGD FP32, `perf` (inputs resident, wall clock incl. sync) | 500 | 2.149 | 2.139 | — | 2.203 | — | `npu_perf.json` |
| RNGD BF16, `bench` / `perf` | 216 / 500 | 2.000 / 1.854 | 1.988 / 1.848 | 2.020 / — | 2.071 / 1.905 | 0.153 / — | `npu_bench_bf16.json`, `npu_perf_bf16.json` |
| RNGD W8 weight-only, `bench` / `perf` | 216 / 500 | 2.352 / 2.155 | 2.339 / 2.155 | 2.371 / — | 2.682 / 2.175 | 0.060 / — | `npu_bench_w8.json`, `npu_perf_w8.json` |

GPU eager timing varies run to run (p95 18.4 → 21.8 ms between two sessions); NPU timing is very
stable (std 0.007 ms FP32) even on a shared host with load average 19–34.

## 2. Throughput and batch size (GPU)
| GPU eager FP32 batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| latency mean (ms) | 13.79 | 14.82 | 13.64 | 19.59 |
| frames/s | 72.5 | 135.0 | 293.2 | 408.4 |
| per-frame ms | 13.79 | 7.41 | 3.41 | 2.45 |
| peak torch allocation (MiB) | 398 | 417 | 455 | 531 |

RNGD throughput (batch 1, synchronise after every frame, 20 s): FP32 470 fps, BF16 543 fps,
W8 446 fps (`npu_perf*.json`). NPU batch > 1 was not compiled (each static shape is a new
~20–30 min compile on the shared host). An attempt to pipeline 8 executions before one
`synchronize()` failed on the device (HAL timeout, then `PXI-122 Duplicate submission`), so no
pipelined NPU number exists (`reports/agent_runs/failures.jsonl`, signature P2).

## 3. Warm-up / cold start
| | GPU (fresh process) | RNGD |
|---|---|---|
| load / compile | import + load 2.15 s | AOT compile 1162–1902 s once; cached reload 3.1–5.3 s; `.to(device)` 0.05 s |
| first inference | 388 ms (CUDA/cuDNN lazy init) | 2.47 ms FP32, 2.11 ms BF16 |
| second inference | 13.7 ms | — |

## 4. Power and energy (board level)
| | idle | under load | fps | energy / frame |
|---|---|---|---|---|
| RTX 3060 eager b1 (NVML, 50 ms sampling, n=398) | — ¹ | 52.3 W | 72.0 | 727 mJ |
| RTX 3060 CUDA Graph b1 (n=399) | — ¹ | 135.8 W | 387.4 | 351 mJ |
| RNGD FP32 (`furiosa-smi`, 1 W resolution, ~0.7 s sampling, n=28) | 39 W | 53.3 W | 470 | 113 mJ |
| RNGD BF16 (n=29) | 39 W | 53.6 W | 543 | 99 mJ |

¹ The GPU "idle" window directly after the load loop averaged 58 W with a 141 W maximum (the card had
not settled); it is not a valid idle figure and is not used. Energy per frame = mean board power /
fps; it includes static power. The two meters differ (NVML vs `furiosa-smi`), so the ratio is
indicative only.

## 5. End-to-end and transport latency — three separate quantities
| Quantity | GPU (desktop) | NPU on-vehicle (assumed co-located) | NPU remote (this setup) |
|---|---|---|---|
| model-only | 13.54 ms eager / 2.60 ms CUDA Graph | 2.11 ms | 2.11 ms |
| device-side incl. H2D/D2H (mean / p95) | 0.39 + 13.54 | 3.81 / 4.80 ms | same |
| host pre + post (mean) | 1.59 + 2.15 ms | same host code | same |
| **frame transport desktop ↔ RNGD host** (1.23 MB, persistent SSH, n=30) | — | — | **121.2 ms mean, 124.8 ms p95** (≈ 10 MB/s); a 60 kB payload 8.5 ms; 1 B ping 3.6 ms |
| **perception e2e p95 used as control delay** | 22.7 ms (`gpu_p95`, measured) | 9.85 ms (`npu_p95`, composed) | 134.7 ms (`npu_remote_p95`, composed) |

The control delay is applied in simulation as actuation delay (`DelayLine`); simulation time does not
advance during inference.

## 6. Is remote NPU-in-the-loop AEB real-time feasible?
- **Latency:** 134.7 ms p95 per frame. At 30 km/h the car travels 1.12 m during that time, at 50 km/h
  1.87 m. In the CARLA matrix with this delay (`npu_remote_p95`) all 10 hazard runs (5 scenarios ×
  GT-assisted / perception-driven) avoided collision, with smaller margins: pedestrian crossing
  2.01 m (GT-assisted) and 1.79 m (perception-driven) vs 3.14 / 2.84 m with the on-vehicle NPU delay.
- **Throughput:** transport is serial over one session: at 121 ms per raw frame the loop runs at
  ≤ 8 Hz, not 30–100 Hz. This is *not* modelled by the simulation (which perceives every 10 ms and
  only delays the action), so the simulated margins above are optimistic for a real remote loop.
- **Conclusion:** remote NPU over this SSH link is not a real-time option for AEB. Options (not
  executed): JPEG transfer (60 kB ≈ 8.5 ms, but changes the input and needs re-validation), a
  co-located NPU, or running the camera simulation on the NPU host.

## 7. Detection correctness (agreement with the GPU FP32 output — not accuracy)
| | recall | precision | mean IoU | mean / max abs Δconf |
|---|---|---|---|---|
| RNGD FP32 | 0.988 | 0.992 | 0.991 | 0.005 / 0.161 |
| RNGD BF16 | 0.972 | 0.992 | 0.989 | 0.007 / 0.032 |
| RNGD W8 (weight-only) | 0.957 | 0.980 | 0.966 | 0.022 / 0.152 |

54 evaluation frames, 254 reference detections. No ground-truth labels exist for these frames, so
mAP is not reported. Open-loop AEB replay on identical frames (GPU vs NPU): see § 8.

## 8. Open-loop AEB replay: GPU vs NPU perception on identical frames
Seven perception-driven runs (`tag = replay`) dumped every 5th letterboxed frame (20 Hz, 738 frames).
The RNGD (FP32, `furiosa:0`) and the GPU processed the identical frames; the same decode, ground-plane
ranger, CIPO tracker and AEB controller then replayed the decisions open-loop up to the first brake
decision (`scripts/replay_aeb.py`, `reports/evidence/replay_npu_vs_gpu.json`).

| Run | Frames compared | First brake decision GPU / NPU (s) | GT TTC at that moment |
|---|---|---|---|
| static obstacle | 28 | 1.35 / 1.35 | 3.67 s |
| lead vehicle braking | 72 | 3.55 / 3.55 | 2.67 s |
| low light (lead braking at night) | 80 | 3.95 / 3.95 | 1.83 s |
| pedestrian crossing | 58 | 2.85 / 2.85 | 0.72 s |
| rain (pedestrian) | 58 | 2.85 / 2.85 | 0.72 s |
| adjacent vehicle (no hazard) | 160 | no brake / no brake | — |
| pedestrian at roadside (no hazard) | 160 | no brake / no brake | — |

Detection agreement over the 616 compared frames: recall 0.987, precision 0.989, mean IoU 0.994,
max |Δconf| 0.030. **The NPU's small numerical differences never changed an AEB decision in these
runs.** Caveats: open-loop (frames come from GPU-driven closed-loop runs), 20 Hz instead of the 100 Hz
closed loop (so brake times differ from the closed-loop values), FP32 graph only.

## 9. Determinism
GPU inference is bit-deterministic on identical inputs (738 replay frames run twice: max abs diff
0.0). Therefore the run-to-run differences seen in repeated CARLA scenarios (`docs/aeb_evaluation.md`)
come from the rendered camera frames, not from inference.

## Verdict
Under the tested conditions the RNGD gives a **real but modest latency advantage** over a
CUDA-Graph GPU path at batch 1 (2.1 vs 2.6 ms), **much better stability** (std 0.007 vs 0.18 ms),
and **~3× lower energy per frame** (board level, different meters). The 6× advantage over the eager
baseline is mostly a software artefact of eager PyTorch. In the closed loop that matters here, the
remote deployment is dominated by transport (121 ms), so the NPU's latency advantage only matters in a
co-located deployment.
