# Troubleshooting

Each entry was observed in this project (date 2026-11-22 unless noted). Symptoms → cause → fix.

## CARLA
| Symptom | Cause | Fix |
|---|---|---|
| GPU at 91–99 % and YOLO 58 % slower while "nothing is running" | CARLA server free-runs and renders continuously when no client holds synchronous mode | Use `CarlaSession` (synchronous; server renders only on `tick()`), or stop the server for model-only benchmarks. Benchmarks record `carla_server_running` |
| Camera frames missing / duplicated / frame and detection out of step | `code/09_yolo_detection.py` registers `camera.listen()` twice; a single queue then receives duplicate images | One listener per sensor; consume only the image whose `frame` equals the value returned by `world.tick()` (`CarlaSession.get_image`) |
| `destroy error: unable to destroy actor: not found` at teardown | Actor already removed server-side (observed with walkers) | Harmless; verify with a world actor query (done in `capture_smoke.py`) |
| Scenario pedestrian spawn fails | Spawn location collides with kerb/props | Runner retries 3 lateral offsets (4.5, 4.0, 3.5 m) |
| Simulator left in synchronous mode after a crash (client hangs on next connect) | Process killed before settings restored | `CarlaSession.__exit__` restores settings; after a hard kill, connect once and `world.apply_settings` with `synchronous_mode=False` |

## YOLO / PyTorch
| Symptom | Cause | Fix |
|---|---|---|
| `forward_head(..., **m.one2many)` returns `{}` / KeyError on some YOLO26 checkpoints | `Detect.fuse()` removes the unused branch; for end-to-end checkpoints the one-to-many head is deleted | `YoloRaw` selects `m.one2one if m.end2end else m.one2many` at run time |
| FP16 not faster than FP32 | Batch-1 eager execution is launch-bound (~24 % GPU util) | Expected; see `reports/gpu_baseline.md` |

## Furiosa RNGD host
| Symptom | Cause | Fix |
|---|---|---|
| `aten::_native_batch_norm_legit_no_training` not importable (2026-11-04 log) | Unfused BatchNorm is not supported by the RNGD importer | Call `model.fuse()` before compiling |
| `Lowering Deadline Violation: Aten(select_scatter_1)` (2026-11-04 log, yolo11n fused) | Head concat lowered through `select_scatter` | Split at the raw head; do decode/NMS on the host |
| Local `ssh` call never returns after starting a remote job | `a && b && nohup cmd &` backgrounds a subshell that keeps ssh's stdout open | Only background the `nohup` command (`scripts/npu_remote.py start`) |
| `furiosa.torch.dryrun` full probe takes > 7 min on a loaded host | Compile-probes every node; host load 14–37 on 16 cores (shared) | Use `--trace-only`, or rely on the AOT compile result |
| `ModuleNotFoundError: furiosa.*` | Wrong conda env (envs differ in installed SDK packages) | Use env `furiosa-npu` (furiosa-torch 2026.4.0 matching firmware 2026.4.0) |
