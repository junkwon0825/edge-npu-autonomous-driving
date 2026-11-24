# V3: multi-model NPU runtime, scheduling and failure engineering

Code:
- `src/npu_ad/runtime/multimodel.py`
- `remote/npu_server_v3.py`, `remote/multimodel_v3.py`
- `src/npu_ad/runtime/fallback.py`, `src/npu_ad/runtime/pipeline.py`

Tests: `tests/test_v3_runtime.py`, `tests/test_v2_runtime.py`.

## Runtime layer (small interfaces, no monolith)
| Piece | Responsibility |
|---|---|
| `InferenceBackend` (detection) / `LaneBackend` (lane) | one frame → result with frame id, capture time, backend, precision, latency, ok/error |
| `GpuInferenceBackend`, `GpuLaneBackend` | local CUDA models: baseline and fallback |
| `RngdMultiModelClient` | one persistent session to the V3 server; `detection()`, `lane()`, `both()` views satisfy the interfaces above over one transport |
| `npu_server_v3.ModelRunner` | one compiled model on one `furiosa:N` with its own worker thread; detection on `furiosa:0`, lane on `furiosa:1` |
| `SimulatedAsyncPipeline` / `AsyncInferenceWorker` | latest-frame queue + worker per model (V2), with an optional *measured-latency* hardware-in-the-loop mode |
| `RateScheduler` | which model runs on which camera frame (fixed or adaptive) |
| `FailoverBackend`, `HealthMonitor` | explicit RNGD → GPU failover, bounded health sampling (V2) |

**Request accounting.** Every request records frame id and capture, enqueue, start, end and receive
times. Per-model server stages are H2D, exec and D2H; the transport is estimated as RTT minus server
time.

**Memory.** Per-model device memory is not observable (`furiosa-smi` reports 0.00 GiB), so no memory
numbers are claimed.

## Scheduling experiment
On-host, device-resident inputs, no network (`remote/multimodel_v3.py`, `v3_npu_multimodel.json`).
YOLO26n BF16 takes 2.00 ms and the lane U-Net 2.72 ms (isolated p50, one PE).

| Strategy | 30 Hz: served Hz (YOLO / lane) | 30 Hz: age p95 (YOLO / lane) | 400 Hz stress: served Hz (YOLO / lane) | 400 Hz: drops (YOLO / lane) |
|---|---|---|---|---|
| A sequential (one worker, one PE, YOLO then lane per frame) | 30 / 30 | 2.35 / 5.06 ms | 217 / 217 | 1100 / 1101 |
| B independent workers, **same PE** | 30 / 30 | 4.91 / 4.93 ms | 251 / 251 | 894 / 894 |
| B independent workers, **separate PEs** | 30 / 30 | **2.51 / 3.45 ms** | **400 / 367** | **0 / 197** |
| C rate-decoupled (lane every 2nd frame) | 30 / 15 | 2.29 / 5.03 ms | 235 / 200 | 992 / 0 |
| D priority, hazard phase (YOLO first) | 22.5 / 11.3 (average of cruise and hazard phases) | 2.56 / 5.53 ms | 287 / 95 | 80 / 333 |

**Answers to the V3 questions** (in this setup, on a shared host):
- **Can both models share one NPU?** Yes. At a single 30 Hz camera, RNGD is lightly loaded (about
  5 ms of compute per frame for both models), and every strategy serves every frame within 6 ms.
- **Does concurrent execution help?** Only with **separate PEs**. Two workers on the same PE are
  *worse* than sequential execution (contention). Separate PEs keep YOLO at full rate even at 400 Hz.
- **How should resources be allocated?** One PE per model. The scheduler then only matters under
  overload, where priority protects detection at the lane model's expense.
- **How much latency does the second model add?** In the deployed remote server: detection only
  55.0 ms mean, detection + lane 61.2 ms (p95 64.2 → 64.3 ms). One upload feeds both models, and the
  parallel section is 3.43 ms vs 2.24 + 2.75 ms.

## Failure engineering
Bounded fault cases against a real 30 Hz stream (`scripts/v3_failure_engineering.py`,
`v3_failure_engineering.json`). Nothing was damaged: the faults were a sleep, a malformed request, a
process exit and invalid start arguments.

| Fault | Detection | Behaviour | Failover | Frames lost | Recovery |
|---|---|---|---|---|---|
| Server stall 1.5 s | 0.25 s timeout | failing frame re-run on the GPU, logged PRIMARY → FALLBACK | 0.54 s | 0 | RNGD again after 1.61 s (stall end + 0.5 s probe) |
| Malformed request (payload ≠ w×h×3) | server error answer `payload … does not match` | request fails, server keeps serving | — | 1 request | immediate |
| Late answer after a timeout | frame-id mismatch | counted as stale, never used | — | — | next frame OK |
| Server process crash | connection closed | GPU takes over | 0.11 s | 0 | new session after 2 s; ready after 7.5 s (two-model cache load); RNGD again 9.6 s after the crash |
| Missing lane weights | start-up error | GPU-only configuration, logged | — | — | — |
| Non-existent device `furiosa:9` | start-up error `Failed to initialize device for index 9` | GPU-only configuration, logged | — | — | — |
| NPU and GPU both failing | both results `ok=False` | `DEGRADED`; the controller holds its last command (V2 tests) | — | — | — |

## Real-time budget (capture → control), measured components
| Stage | GPU local (Mode A) | RNGD remote, detection + lane (Mode D) | Source |
|---|---|---|---|
| Camera frame period | 33 ms (33 Hz) | 33 ms | design |
| Pre-processing (letterbox / resize, encode) | in the GPU e2e | 0.6 ms encode (client) | `v3_npu_multimodel_bench.json` |
| Transport up + down | — | ≈ 55 ms mean (RTT − server; 34 ms receive on the server side) | same |
| Detection inference | 13–22 ms eager in the loop (CARLA shares the GPU) | 2.0–2.2 ms exec + 1.5 ms H2D/D2H | V2 `gpu_local`, V3 bench |
| Lane inference | measured per run (in-loop GPU, shares the GPU with CARLA and YOLO) | 2.8 ms exec, concurrent with detection | V3 bench |
| Host decode / NMS / mask encode | ≈ 2 ms | 0.9–1.5 ms (server) | V3 bench |
| Tracking, filters, lane geometry, AEB, LKA, arbitration | < 2 ms per tick (Python) | same | profiling during the runs |
| Control period | 10 ms | 10 ms | design |
| **Capture → result available (mean / p95)** | ≈ 22 / 28 ms (detection) | **61 / 64 ms** (detection + lane) | measured |
| Frame age at control, worst case | result + up to 10 ms control period | same | — |

**Conclusion.** With a co-located NPU, the model stages would be about 7 ms for both models. The
remote setup is transport-bound, so a remote RNGD closed loop is *near-real-time*, not real-time.
This document makes no real-time guarantee.
