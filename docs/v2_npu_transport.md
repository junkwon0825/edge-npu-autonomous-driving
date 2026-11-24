# V2: where the remote-NPU latency goes (transport and profiling)

**Question.** V1 measured about 121 ms to move one raw frame from the desktop to the remote RNGD host.
Is the bottleneck network bandwidth, SSH process overhead, serialization, image size, or remote
synchronization?

**Answer, measured.** In the V1-equivalent path, two things cost the most:
- Network upload of an oversized payload (~100 ms).
- The remote host's CPU stages (pre-processing, D2H copy and decode/NMS: 70–85 ms). The default
  PyTorch thread pool oversubscribed a shared 16-core host running at load average 9–30.

NPU execution itself is 2.0–2.3 ms.

Four changes cut the end-to-end time from **203 ms to 60 ms mean** (p95 277 → 66 ms), with no
change to the detections:
1. Send only the letterbox content.
2. Pin the remote threads.
3. Use an exact score pre-filter.
4. Turn on SSH compression.

Under heavy host load, the V1-equivalent path degrades to 437 ms mean / 643 ms p95, while the V2 path
stays at 71 / 104 ms. JPEG reaches 30 ms, but it is lossy and lowers detection agreement.

Evidence:
- `reports/evidence/v2_transport_profile.json` (round 1, n = 72 per variant)
- `reports/evidence/v2_transport_profile_r2.json` (round 2, n = 216 per variant)
- `reports/evidence/v2_latency_samples.json` (n = 200 per variant, host under load)

Scripts: `scripts/v2_transport_profile.py`, `scripts/v2_latency_samples.py`.

## Setup
**Client.** `npu_ad.runtime.backends.FuriosaRemoteBackend` on the desktop.

**Server.** `remote/npu_server.py` runs on the RNGD host.
- It runs as one persistent SSH session.
- It uses a binary length-prefixed protocol over stdin/stdout.
- The BF16 YOLO26n comes from the SDK compile cache (load 5 s, no recompile).
- Every response carries per-stage server durations.

**Network estimate.** The client RTT minus the server's total time minus the client's send time.

**Frames.** The 72 CARLA frames of the V1 dataset. Agreement is measured against local GPU FP32
detections on the same frames; it is not mAP.

## 1. Transport only (ECHO, no inference; persistent session)
| Payload | p50 RTT | Comment |
|---|---|---|
| 1 B | 11.0 ms | protocol and SSH floor |
| 60 kB | 8.2 ms | |
| 691 kB (640×360×3 content) | 63.7 ms | ≈ 11 MB/s |
| 1.23 MB (640×640×3 letterboxed, V1) | 110.8 ms | ≈ 11 MB/s |
| **new `ssh` process per request**, 691 kB | **700.9 ms** | connection + auth + remote shell |

The link is bandwidth-limited at about 11 MB/s. Starting a new SSH process per frame would cost 0.7 s,
so a persistent session is mandatory.

## 2. Inference variants (BF16, batch 1; stage p50 in ms)
| Variant | e2e mean / p95 | upload (rx) | pre | NPU exec | D2H | decode+NMS | Server total | Recall / precision vs GPU |
|---|---|---|---|---|---|---|---|---|
| V1-equivalent: letterboxed raw 1.23 MB, default threads | 202.8 / 277.0 | 101.5 | 19.1 | 2.3 | 13.0 | 38.4 | 77.1 | 0.974 / 0.988 |
| content only (691 kB), default threads | 146.2 / 158.3 | 55.9 | 19.2 | 2.3 | 12.7 | 37.1 | 75.2 | 0.974 / 0.988 |
| + 4 threads + exact pre-filter | 97.7 / 128.4 | 55.9 | 1.4 | 2.1 | 0.9 | 1.1 | 6.2 | 0.974 / 0.988 |
| + 2 threads + exact pre-filter | 77.7 / 79.5 | 56.0 | 0.7 | 2.1 | 0.7 | 0.9 | 5.4 | 0.974 / 0.988 |
| **+ SSH compression (`-C`), 2 threads** (round 2, n = 216) | **60.2 / 65.5** (p99 67.5, std 4.8) | net ≈ 16.9 | — | 2.05 | — | — | 5.1 | **0.974 / 0.988** |
| PNG level 1 (lossless), 2 threads (round 2) | 74.8 / 79.5 | net ≈ 33.6 | client encode 17.1 | 2.28 | — | — | 12.9 | 0.974 / 0.988 |
| JPEG q95, 2 threads (round 2) | 30.8 / 33.1 | net ≈ 10.3 | — | 1.97 | — | — | 5.8 | 0.908 / 0.924 |
| JPEG q90, 4 threads (round 1) | 29.9 / 33.4 | 1.6 | 0.7 | 2.1 | 0.7 | 0.9 | 6.2 | 0.891 / 0.923 |
| JPEG q75, 4 threads (round 1) | 23.7 / 32.0 | 0.0 | 0.9 | 2.0 | 0.7 | 0.7 | 5.6 | 0.810 / 0.892 |

**What each change did:**
- **Content-only payload.** The letterbox padding (rows of value 114) is removed before sending and
  re-added on the server. This is lossless and halves the bytes.
- **Thread pinning (`torch.set_num_threads(2)`).** The default pool used every core on a host at load
  average 9–30. Pinning cut pre-processing 19 → 0.7 ms and D2H 13 → 0.7 ms. More threads were
  *slower* (4 threads: p95 128 ms).
- **Exact pre-filter** (`RawDecoder.decode_prefiltered`). It decodes only anchors whose best class
  score exceeds the confidence threshold, which NMS would discard anyway. Locally, 72/72 frames gave
  identical detections (348/348, max diff 0.0). On the remote host, decode+NMS fell from 37–38 to
  under 1 ms (with 2 threads).
- **SSH compression.** CARLA frames compress well with zlib, so the network estimate falls from 59 to
  17 ms. This is lossless.
- **PNG.** It is lossless but slower: client encoding costs 17 ms.
- **JPEG.** It is the fastest, but it changes the detections. Recall drops to 0.89–0.91 at q90–95 and
  0.81 at q75, so it is not used for the safety function.

## 3. Pipelining (requests in flight)
| Configuration | 1 in flight | 2 in flight |
|---|---|---|
| content raw, 4 threads, pre-filter (round 1) | 12.8 fps | 16.9 fps |
| content raw, 2 threads, pre-filter, `-C` (round 2) | 16.5 fps | 27.3 fps |
| PNG level 1 (round 2) | 16.6 fps | 30.9 fps |

Overlapping the upload of frame N+1 with the inference of frame N raises throughput. It does not
lower the age of each individual frame.

The V2 client (`FuriosaRemoteBackend`) has a writer thread that allows **one** request in flight.
- A send blocked by a stalled server or a congested link is refused with `TransportBusy`, so
  `infer()` cannot hang past its deadline.
- The pipelining measurement above used the earlier blocking-send client. The protocol is unchanged.
- The writer thread was added after a unit test showed that the original send could block for the
  full server stall.

## 4. Distributions under load (used by the CARLA matrix)
Measured with `scripts/v2_latency_samples.py`, n = 200 per variant. The host's load average was 27–31
at the time (a parallel compile job).

| Variant | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| `npu_remote_v1equiv` | 437.3 | 435.9 | 643.3 | 752.8 |
| `npu_remote_v2` (content raw, `-C`, 2 threads, pre-filter) | 71.2 | 66.9 | 104.4 | 115.5 |
| `npu_remote_jpeg95` | 36.1 | 34.1 | 55.5 | 72.7 |
| `npu_colocated` (server-side service time only, no network) | 16.0 | 11.3 | 40.7 | 57.5 |
| `gpu_local` (in-loop GPU e2e from the V1 demo log, CARLA sharing the GPU, n = 342) | 22.4 | 22.1 | 27.8 | 32.2 |

The V1-equivalent configuration doubles under load, because its unpinned thread pool competes with
the other job. The V2 configuration moves by about 10 ms.

`npu_colocated` is the server's own processing time on a loaded shared host. It estimates what a
co-located NPU would add; nothing was measured on a vehicle.

## 5. Bottleneck classification (qualitative; no hardware counters available)
| Stage | Bound | Evidence |
|---|---|---|
| NPU execution (2.0–2.3 ms) | not the bottleneck; exec time barely changes from 320 to 640 input (1.8–2.0 ms, `v2_resolution_npu.json`) → fixed-overhead-bound for this nano model | stage timing |
| Remote host CPU pre/post (V1 config) | CPU contention (oversubscribed thread pool on a shared host) | 37–85 ms → 1–5 ms with 2 threads |
| Network upload | bandwidth-bound (≈ 11 MB/s), payload-proportional | ECHO sweep; e2e ∝ bytes in the resolution study |
| SSH | per-process setup cost (0.7 s); a persistent session removes it | ECHO vs new process |
| Serialization (raw bytes) | negligible (client encode 0.6–0.9 ms; PNG 17 ms is the exception) | stage timing |

No roofline or utilisation claims are made: `furiosa-smi` exposes power and temperature, but no
compute or memory counters.

## Answer to the V2 question
- **Remote NPU-in-the-loop over SSH: near-real-time at best.** The lossless V2 path is 60 ms mean /
  66 ms p95 on a quiet host and 71 / 104 ms under load. It is transport-bound (bandwidth).
- **What would make it real-time:**
  - an NPU co-located with the camera (service time 5–16 ms mean on this host), or
  - lossy compression with an accuracy cost, or
  - a faster link.
- **The low model latency translated into system latency only after the host-side stages were
  fixed.** In the V1 configuration, the NPU's 2 ms were about 1 % of the end-to-end time.
