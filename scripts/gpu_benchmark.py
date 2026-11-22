"""Phase 2 / Gate C: YOLO26n raw-split GPU baseline on the fixed CARLA dataset.

Measures preprocess / H2D / model-only (CUDA events) / postprocess / end-to-end per frame after
warm-up, peak allocated GPU memory and sampled GPU utilization (NVML). Saves detections, raw
head outputs (for NPU comparison), NPU input tensors with identical preprocessing, and a few
visualizations. Record whether the CARLA server was running: it competes for the same GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time

import cv2
import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import latency_stats
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.yolo_raw import letterbox


class NvmlSampler(threading.Thread):
    def __init__(self, period: float = 0.05):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period, self.util, self.mem, self.temp = period, [], [], []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.util.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            self.mem.append(self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20)
            self.temp.append(self.nv.nvmlDeviceGetTemperature(self.h, 0))
            time.sleep(self.period)

    def stop(self) -> dict:
        self._halt.set()
        self.join()
        u = np.asarray(self.util or [0])
        return {"gpu_util_mean_pct": float(u.mean()), "gpu_util_max_pct": float(u.max()),
                "device_mem_used_max_mib": float(max(self.mem or [0])),
                "temp_max_c": float(max(self.temp or [0])), "samples": len(self.util)}


def carla_running() -> bool:
    import psutil

    return any((p.info["name"] or "").lower().startswith("carlaue4") for p in psutil.process_iter(["name"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/dataset/manifest.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tag", default="fp32")
    ap.add_argument("--export-npu-inputs", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    m = cfg["model"]
    manifest = json.loads(resolve(a.manifest).read_text(encoding="utf-8"))
    frames = manifest["frames"]
    images = [cv2.imread(str(resolve(f["path"]))) for f in frames]
    assert all(im is not None for im in images)
    torch.manual_seed(0)
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"], half=a.half)
    out_dir = resolve(f"outputs/gpu/{a.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(a.warmup):
        det(images[i % len(images)])
    torch.cuda.reset_peak_memory_stats()
    sampler = NvmlSampler()
    sampler.start()
    stages: dict[str, list[float]] = {k: [] for k in ("preprocess", "h2d", "model", "postprocess", "e2e")}
    dets, raw_b, raw_s = {}, [], []
    t_start = time.perf_counter()
    for r in range(a.repeats):
        for f, im in zip(frames, images, strict=True):
            out, t, (b, s) = det(im)
            for k in stages:
                stages[k].append(getattr(t, k))
            if r == 0:
                dets[f["path"]] = out.tolist()
                raw_b.append(b)
                raw_s.append(s)
    wall = time.perf_counter() - t_start
    nv = sampler.stop()
    n = a.repeats * len(frames)
    result = {
        "tag": a.tag,
        "precision": "fp16" if a.half else "fp32",
        "model": m,
        "frames": len(frames),
        "repeats": a.repeats,
        "warmup": a.warmup,
        "carla_server_running": carla_running(),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()},
        "latency": {k: latency_stats(v) for k, v in stages.items()},
        "throughput_fps_sequential": n / wall,
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "nvml": nv,
        "detections_total_first_pass": sum(len(v) for v in dets.values()),
    }
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "detections.json").write_text(json.dumps(dets), encoding="utf-8")
    torch.save({"names": [f["path"] for f in frames], "boxes": torch.cat(raw_b), "scores": torch.cat(raw_s)},
               out_dir / "raw_outputs.pt")
    names = det.raw.model.names
    for f, im in list(zip(frames, images, strict=True))[::12]:
        vis = im.copy()
        for x1, y1, x2, y2, c, k in dets[f["path"]]:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, f"{names[int(k)]} {c:.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"vis_{f['weather']}_{f['frame']}.jpg"), vis)
    if a.export_npu_inputs:
        lbs = np.stack([letterbox(im, m["imgsz"])[0] for im in images])
        npu_dir = resolve("outputs/npu")
        npu_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"names": [f["path"] for f in frames], "lb_bgr_uint8": torch.from_numpy(lbs)},
                   npu_dir / "inputs.pt")
    print(json.dumps({k: result[k] for k in ("tag", "carla_server_running", "throughput_fps_sequential",
                                             "peak_torch_allocated_mib")}, indent=2))
    for k, v in result["latency"].items():
        print(f"{k:12s} mean {v['mean_ms']:.3f} ms  p50 {v['p50_ms']:.3f}  p95 {v['p95_ms']:.3f}")
    print("nvml", nv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
