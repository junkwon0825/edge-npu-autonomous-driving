# Environment Audit (Phase 0)

Audit date: 2026-11-22 (KST). All values below were measured in this session; historical notes
from the knowledge base were re-checked, not copied.

## Local workstation (Windows)

| Item | Measured | How |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB | `nvidia-smi` |
| Driver | 616.92 (CUDA UMD 13.4) | `nvidia-smi` |
| Python (project venv) | 3.12.10 | `.venv\Scripts\python.exe` |
| PyTorch | **2.14.1+cu126**, `torch.cuda.is_available() == True`, CUDA runtime 12.6 | import check |
| torchvision | 0.29.1+cu126 | import check |
| ultralytics | 8.4.202 | import check |
| OpenCV | 5.0.0 | import check |
| numpy | 2.5.4 | import check |
| CARLA server | 0.9.16 (`C:\CARLA_0.9.16`) | `client.get_server_version()` |
| CARLA Python client | 0.9.16 (wheel in project venv) | `client.get_client_version()` |
| Default map | `Carla/Maps/Town10HD_Opt` | `world.get_map().name` |
| Disk C: | 195 GB free of 465 GB | `df -h` |
| Other venvs | `<LOCAL_VENV_DIR>\carla310` (Python 3.10, empty: no torch/carla), `<LOCAL_VENV_DIR>\carla312` | listing |

### Historical issues re-checked

| Historical observation | Current state | Action |
|---|---|---|
| PyTorch `2.15.0+cpu`, CUDA unavailable | **Resolved** in `carla_project\.venv` (2.14.1+cu126, CUDA available). The `+cpu` build exists only on the NPU host env `furiosa-project` | none — no reinstall performed |
| CARLA GPU utilisation reached 100% | **Reproduced**: with the server free-running (asynchronous) the GPU sat at 91–99% util, 75–76 °C, ~8.9 GB used | Benchmarks record whether CARLA was running; scenario runs use synchronous mode, where the server renders only on `tick()` |
| Camera frames sometimes failed to save | Not reproduced in 3×100-frame runs. Likely cause found in `code/09_yolo_detection.py`: `camera.listen()` is registered **twice**, which can enqueue two images per tick | New `CarlaSession` registers one listener and accepts only the image whose `frame` equals the ticked frame |
| Project directory not under Git | Confirmed | `git init`, branch `feature/npu-pipeline` (local only, never pushed) |
| Furiosa BatchNorm / operator issues | Confirmed in prior server logs: unfused `yolo11n` fails on `aten::_native_batch_norm_legit_no_training`; fused `yolo11n` fails with `select_scatter` lowering violation in the head concat | Use fused model and split at the raw head (see `npu_compilation.md`) |

### Pre-existing project contents (preserved, unchanged)
`code/01_world.py … 09_yolo_detection.py`, `data/rgb/*.png`, `data/lidar/*.ply`, `yolo11n.pt`, `.venv/`.

### Changes made to the environment
- Installed `pytest` into `carla_project\.venv` (test runner). No other package, driver or system change.
- Ruff is used from the existing lab venv (`<AGENT_LAB_ROOT>\.venv`), not installed here.
- Copied `yolo26n.pt` (5.5 MB) from the user's own NPU host home directory to `models/`
  (SHA-256 `9b09cc8b…a8ad4fef`, identical on both machines). No internet download.

## Furiosa RNGD NPU host (remote)

Reached through the pre-existing `~/.ssh/config` alias `furiosa-npu` (key auth, `BatchMode`, strict
host-key checking). Address, user and key are not recorded here. The desktop manual
(`개발Tool\Hermes_Furiosa_접속_매뉴얼.txt`) describes the Hermes WSL agent and the SSH procedure; it does
not contain the server address — the existing SSH config does.

| Item | Measured |
|---|---|
| Hostname | `<REMOTE_NPU_HOST>`, kernel 6.8.0-117-generic, 16 vCPU, 1 TB RAM |
| NPU | `furiosa-smi info`: arch `rngd`, device `npu0`, firmware 2026.4.0 (93d67f3), 37.7 °C, 39 W idle |
| `furiosa.torch.get_available_devices()` | `furiosa:0` … `furiosa:7` |
| SDK (env `furiosa-npu`) | `furiosa-torch 2026.4.0 (rev 5800e2ba24, built 2026-10-12)`, `furiosa-kernels 2026.4.0`, torch 2.10.0+cpu, Python 3.11.18, ultralytics 8.4.202 |
| Other envs | `furiosa-yolo` (SDK 2026.3.0), `furiosa02` (furiosa-llm 2026.4.0), `furiosa-project`, `furiosa` |
| Public API found | `CompileModule.from_module(m, args, *, compiler_config=None, …)`, `.to("furiosa:N")`, `current_stream().synchronize()`, `Event`, `dryrun.dryrun(fn, *inputs, trace_only=…)` |
| Quantization API | **None found** in `furiosa.torch` (no `quantize`/`calibrate`); `get_amp_supported_dtype() == []` |
| Shared use | Load average 14–25 on 16 cores and compile-cache entries dated the same day ⇒ the host is shared. All work is confined to `~/npu_auto_driving_p2`, jobs run under `nice -n 10`, one NPU job at a time |

## Hermes (WSL)
`wsl -l -v` lists `hermes01` (Stopped) and `furiosa01` (Stopped). Hermes is a Telegram-gateway agent,
not the NPU; it was not started or modified.
