"""Phase 2 extension: GPU model-only latency vs. batch size, CUDA Graphs, cold start and power.

All cases run the same fused raw YOLO26n module on the same letterboxed CARLA frames as the
baseline (outputs/npu/inputs.pt, written by gpu_benchmark.py --export-npu-inputs). Model-only time
is measured with CUDA events around the forward (inputs already on the device), so it is directly
comparable with the NPU "compute" figure. Board power comes from NVML (``nvmlDeviceGetPowerUsage``)
sampled during a sustained loop. Refuses to run while a CARLA server is up unless --allow-carla.

  python scripts/gpu_benchmark_ext.py            # -> reports/evidence/gpu_benchmark_ext.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import latency_stats
from npu_ad.perception.yolo_raw import load_raw_model, to_tensor

COLD_START_CHILD = r"""
import sys, time, json
t0 = time.perf_counter()
import torch
sys.path.insert(0, 'src')
from npu_ad.perception.yolo_raw import load_raw_model
raw, _ = load_raw_model(sys.argv[1]); raw = raw.cuda()
t1 = time.perf_counter()
x = torch.zeros(1, 3, 640, 640, device='cuda')
with torch.no_grad():
    raw(x)
torch.cuda.synchronize()
t2 = time.perf_counter()
with torch.no_grad():
    raw(x)
torch.cuda.synchronize()
t3 = time.perf_counter()
print(json.dumps({'import_and_load_s': t1 - t0, 'first_inference_ms': (t2 - t1) * 1e3,
                  'second_inference_ms': (t3 - t2) * 1e3}))
"""


class PowerSampler(threading.Thread):
    def __init__(self, period: float = 0.05):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period, self.w, self.util = period, [], []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.w.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
            self.util.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            time.sleep(self.period)

    def stop(self) -> dict:
        self._halt.set()
        self.join()
        return {
            "power_w_mean": float(np.mean(self.w)),
            "power_w_max": float(np.max(self.w)),
            "gpu_util_mean_pct": float(np.mean(self.util)),
            "samples": len(self.w),
        }


def _events_ms(fn, n: int) -> list[float]:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(n):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--power-s", type=float, default=20.0)
    ap.add_argument("--allow-carla", action="store_true")
    a = ap.parse_args()
    import psutil

    carla = any((p.info["name"] or "").lower().startswith("carlaue4") for p in psutil.process_iter(["name"]))
    if carla and not a.allow_carla:
        print("CARLA server is running (shares the GPU); stop it or pass --allow-carla", file=sys.stderr)
        return 2
    cfg = load_config()
    weights = str(resolve(cfg["model"]["weights"]))
    res: dict = {"carla_server_running": carla, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    # cold start in a fresh process (import + load + first CUDA inference incl. cuDNN autotune/lazy init)
    p = subprocess.run(
        [sys.executable, "-c", COLD_START_CHILD, weights], capture_output=True, text=True, timeout=300, cwd=resolve(".")
    )
    res["cold_start"] = (
        json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-500:]}
    )

    raw, _ = load_raw_model(weights)
    raw = raw.cuda().eval()
    data = torch.load(resolve("outputs/npu/inputs.pt"))
    xs = torch.cat([to_tensor(img.numpy()) for img in data["lb_bgr_uint8"]]).cuda()
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for _ in range(20):
            raw(xs[:1])
        torch.cuda.synchronize()
        res["batch"] = {}
        for b in a.batches:
            xb = xs[:b].contiguous()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(10):
                raw(xb)
            st = latency_stats(_events_ms(lambda xb=xb: raw(xb), a.n))
            st["frames_per_s"] = b * 1000.0 / st["mean_ms"]
            st["per_frame_ms"] = st["mean_ms"] / b
            st["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            res["batch"][str(b)] = st
            print("batch", b, round(st["mean_ms"], 2), "ms", round(st["frames_per_s"], 1), "fps")

        # CUDA Graphs (batch 1): removes per-kernel launch overhead of eager PyTorch
        static_x = xs[:1].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                raw(static_x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = raw(static_x)
        diffs = []
        for i in range(len(xs)):
            static_x.copy_(xs[i : i + 1])
            g.replay()
            ref = raw(xs[i : i + 1])
            diffs.append(max((o - r).abs().max().item() for o, r in zip(static_out, ref, strict=True)))
        st = latency_stats(_events_ms(g.replay, a.n))
        st["max_abs_diff_vs_eager"] = max(diffs)
        res["cuda_graph_b1"] = st
        print("cuda graph b1", round(st["mean_ms"], 2), "ms")

        # power under sustained load: eager b1 and CUDA graph b1
        for name, fn in (("eager_b1", lambda: raw(static_x)), ("cuda_graph_b1", g.replay)):
            ps = PowerSampler()
            ps.start()
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < a.power_s:
                fn()
                n += 1
                if n % 8 == 0:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            pw = ps.stop()
            pw["fps"] = n / el
            pw["energy_per_frame_mj"] = pw["power_w_mean"] / pw["fps"] * 1000.0
            res.setdefault("power", {})[name] = pw
            print("power", name, round(pw["power_w_mean"], 1), "W", round(pw["fps"], 1), "fps")
        ps = PowerSampler()
        ps.start()
        time.sleep(5)
        res.setdefault("power", {})["idle"] = ps.stop()
    out = resolve("reports/evidence/gpu_benchmark_ext.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
