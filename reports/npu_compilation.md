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
**Status: PARTIAL — INT8 quantization BLOCKED; BF16 precision reduction executed.**

### INT8 / static quantization — BLOCKED (SDK capability)
- `furiosa.torch` 2026.4.0 exposes no `quantize`/`calibrate` API; `get_amp_supported_dtype()` returns `[]`.
- A source search of the installed `furiosa` packages (envs `furiosa-npu`, `furiosa02`) found quantization
  code only in LLM kernels (`kernels/qwen3`, `kernels/qwen3_moe`, `kernels/exaone4`,
  `kernels/common/dynamic_quantization.py`), i.e. the FP8 path used by `furiosa-llm`, not a vision/CNN path.
- The KB records a student report that `furiosa-models` vision modules became empty after the RNGD
  transition (2026-11-03, unverified). No officially documented INT8 path for a PyTorch CNN was found
  in the installed SDK, so none was invented.
- Prepared for later use: 18 calibration frames (`split == "calib"` in `outputs/dataset/manifest.json`),
  never used for reported agreement metrics.

### BF16 precision reduction — executed (`--dtype bf16`)
Same raw module and inputs cast to `torch.bfloat16` before `CompileModule.from_module`.
This is a precision change, **not** quantization, and needs no calibration.

| Metric | NPU FP32 graph | NPU BF16 graph |
|---|---|---|
| AOT compile time | 1901.9 s | 1169.6 s |
| Compute mean / p50 / p95 | 2.107 / 2.106 / 2.119 ms | 2.000 / 1.988 / 2.020 ms |
| H2D mean | 0.80 ms | 0.74 ms |
| D2H mean / p95 (includes host `.float()` of the outputs) | 0.90 / 1.76 ms | 11.16 / 21.36 ms |
| Raw diff vs GPU FP32: boxes max / mean | 2.965 / 0.032 | 3.863 / 0.039 |
| Raw diff vs GPU FP32: logits max / mean | 5.352 / 0.051 | 7.087 / 0.094 |
| Eval split vs GPU FP32: recall / precision / mean IoU | 0.988 / 0.992 / 0.991 | 0.972 / 0.992 / 0.989 |
| Mean / max abs Δconf (eval) | 0.0051 / 0.161 | 0.0074 / 0.032 |

Observations: BF16 compute was 5 % faster and compile 38 % shorter (compile times on the shared host depend on its load and are not a stable metric), while recall vs GPU dropped from 0.988
to 0.972 (4 more missed detections out of 254). The BF16 D2H figure is dominated by something on the host
side of the output copy/conversion (~10× FP32); its cause was not isolated, so the BF16 device-side total is
not used for any latency claim. Evidence: `reports/evidence/npu_{compile,bench}_bf16.json`,
`reports/evidence/gpu_vs_npu_bf16.json`.
