# Furiosa RNGD Compilation, Runtime and Quantization (Phases 3–4, Gate D)

## Hardware / SDK gate — PASS
| Item | Value |
|---|---|
| Access | existing `~/.ssh/config` alias `furiosa-npu` (key auth); host `<REMOTE_NPU_HOST>` (shared) |
| Device | `furiosa-smi info`: `rngd` `npu0`, firmware 2026.4.0 (93d67f3); torch devices `furiosa:0…7` |
| SDK | `furiosa-torch 2026.4.0 (rev 5800e2ba24, built 2026-10-12)`, `furiosa-kernels 2026.4.0`, compiler revision `fc43f6a779` |
| Python env | conda `furiosa-npu`: Python 3.11.18, torch 2.10.0+cpu, ultralytics 8.4.202 |
| Work dir | `~/npu_auto_driving_p2` only; jobs under `nice -n 10`; one NPU job at a time |

## Model preparation
Operator history (prior logs on the host, 2026-11-04, re-read in Phase 0):

| Attempt | Result | Root cause |
|---|---|---|
| yolo11n, unfused, `torch.compile(backend=furiosa.torch.backend)` | FAIL | `aten::_native_batch_norm_legit_no_training` not importable |
| yolo11n, fused | FAIL | `Lowering Deadline Violation: Aten(select_scatter_1)` from head concat (`_custom_ops/cat_stack.py`) |
| yolo26n raw head, AOT (`CompileModule`, SDK 2026.3.0) | compiled (3056 s) | — |

Applied transformations in this project (all justified and numerically checked):
1. `model.fuse()` — folds Conv+BN (removes the unsupported BN op).
2. Split at the detection head: the module returns raw `boxes (1,4,8400)` and `scores (1,80,8400)` logits;
   anchor decode, sigmoid and NMS run on the host. Verified identical to ultralytics' own decoded output
   (max abs diff 0.0 on CPU, 2.98e-8 on CUDA).
3. Static input shape 1×3×640×640 (fixed letterbox), FP32.
NMS, dynamic shapes and data-dependent ops are therefore not on the NPU.

Operator probe: `furiosa.torch.dryrun.dryrun` (full per-node compile probe) was started but stopped by the
author after ~7 min on the loaded shared host in favour of the real AOT compile; no per-op report was produced
(`status_dryrun.json` state `ABORTED_BY_OPERATOR`). The successful full compile shows no unsupported op
remained in the raw module.

## Compilation — PASS
- API: `furiosa.torch.CompileModule.from_module(raw_module, (x,))` (signature read from the installed SDK).
- Result: `CompileModule`, **1901.9 s** wall time (host load average 25–37), `status_compile.json` PASS
  (`reports/evidence/npu_compile.json`).
- Compiler cache entry (identified by creation/finish timestamps 02:31/03:02 UTC): `output.bin` 2,862,469 B;
  `compile_params.yaml` shows `implicit_type_casting: true`, `lowering_mode: Optimal`, `num_pe: 8`.

## Runtime inference — PASS
- `compiled.to("furiosa:0")`, timing with `furiosa.torch.current_stream(device).synchronize()`.
- Reload from cache 3.1 s, `.to(device)` 0.05 s; 10 warm-up + 216 measured inferences on 72 CARLA frames.
- Compute mean **2.107 ms** (p50 2.106, p95 2.119, std 0.007); H2D mean 0.80 ms; D2H mean 0.90 ms.
- Numerical check vs the same module on the host CPU: boxes max/mean abs 2.958 / 0.032, score logits
  5.566 / 0.050. Detection-level agreement vs GPU FP32: recall 0.988, precision 0.992 (see
  `benchmark_gpu_vs_npu.md`). The internal precision used by the compiler was not confirmed.

## Failures in this session
| Time (UTC) | Step | Error | Cause | Fix |
|---|---|---|---|---|
| 03:03 | bench | `TypeError: current_stream() missing 1 required positional argument: 'device'` | author assumed the signature | `current_stream(dev)`; logged in `reports/agent_runs/failures.jsonl` |
| 02:23 | remote start | local ssh call hung | backgrounded and-list kept stdout open | only `nohup` is backgrounded |

## Quantization (Phase 4)
<!-- QUANT_SECTION -->
