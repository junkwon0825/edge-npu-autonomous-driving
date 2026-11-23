# GPU → Furiosa RNGD deployment workflow

Detailed log, failures and per-run numbers: `reports/npu_compilation.md`. Quantization:
`docs/quantization_limitations.md`.

## Environment (measured)
| | Desktop (GPU) | RNGD host (shared, remote) |
|---|---|---|
| Hardware | RTX 3060 12 GB, Ryzen 5 5600 | Furiosa RNGD `npu0` (`furiosa-smi`: 47.5 GiB device memory; torch sees `furiosa:0…7`; compiler config `num_pe: 8`) |
| Software | Windows 11, Python 3.12.10, torch 2.14.1+cu126, ultralytics 8.4.202 | conda `furiosa-npu`: Python 3.11.18, torch 2.10.0+cpu, furiosa-torch 2026.4.0, firmware 2026.4.0 |
| Access | local | existing `~/.ssh/config` alias `furiosa-npu`; host and key are not stored in this repo |

## Steps
1. **Prepare the module on the desktop side of the contract** (`src/npu_ad/perception/yolo_raw.py`):
   `model.fuse()` (Conv+BN folding, BN is not importable) and `YoloRaw` returning raw
   `boxes (1,4,8400)` and score logits `(1,80,8400)`. Validated against the unfused original on all
   72 frames: 348/348 detections (`evidence/raw_split_validation_all_cpu.json`).
2. **Freeze inputs**: `gpu_benchmark.py --export-npu-inputs` writes the letterboxed uint8 frames
   (`outputs/npu/inputs.pt`); both accelerators consume the same bytes through the same `to_tensor`.
3. **Sync code** (`npu_remote.py sync`): `npu_job.py`, `int8_probe.py`, `yolo_raw.py`, weights;
   SHA-256 printed for each.
4. **Compile AOT** (`npu_remote.py start -- compile --dtype {fp32|bf16|w8}`):
   `furiosa.torch.CompileModule.from_module(raw, (x,))`, static 1×3×640×640. The SDK caches the
   compiled artifact; later runs reload it in 3–5 s.
5. **Run and time** (`start -- bench`): `compiled.to("furiosa:0")`; H2D, compute
   (`current_stream(dev).synchronize()`), D2H timed separately; raw outputs saved.
6. **Compare** (`compare_gpu_npu.py --dtype …`): identical host decode/NMS on GPU and NPU raw outputs;
   raw-tensor diff + detection agreement on the held-out split.
7. **Performance / power** (`start -- perf --dtype …`): cold start, sequential vs pipelined
   throughput, `furiosa-smi` board power and device memory.

Operational rules applied: one NPU job at a time (`BUSY` lock), `nice -n 10`, bounded waits, only the
project work dir `~/npu_auto_driving_p2` is written.

## Results per variant
| Variant | Compile (s, shared host) | Compute mean / p95 / p99 (ms, n=216) | Recall / precision / IoU vs GPU FP32 (54 frames) | Weights in graph |
|---|---|---|---|---|
| FP32 | 1901.9 | 2.107 / 2.119 / 2.125 | 0.988 / 0.992 / 0.991 | 9.64 MB fp32 |
| BF16 | 1169.6 | 2.000 / 2.020 / 2.071 | 0.972 / 0.992 / 0.989 | 4.82 MB bf16 |
| W8 weight-only (BF16 compute) | 1161.7 | 2.352 / 2.371 / 2.682 | 0.957 / 0.980 / 0.966 | 2.44 MB (int8 + bf16 scales) |

Compile times depend on the shared host's load (load average 14–42 during these runs) and are not a
stable metric. The compiled artifact size is not reported per variant: the compiler cache on the
shared host is written by several users and cannot be attributed reliably.
