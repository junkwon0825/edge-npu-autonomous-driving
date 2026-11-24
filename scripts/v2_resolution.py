"""V2 resolution experiment: YOLO26n at 320 / 480 / 640 input (same weights, no retraining).

GPU (FP32 eager and CUDA Graph, CARLA stopped) and RNGD BF16 (persistent server, one AOT compile per
static shape, from the SDK cache) on the same 72 CARLA frames. Agreement is measured against the
GPU FP32 640 detections (the deployed V1 reference), not against ground truth.
Output: reports/evidence/v2_resolution.json
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess

import cv2
import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import aggregate_agreement, latency_stats, match_detections
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.yolo_raw import letterbox, to_tensor
from npu_ad.runtime.backends import FuriosaRemoteBackend


def carla_running() -> bool:
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    return "CarlaUE4" in out


def gpu_timing(det: GpuDetector, frames, imgsz: int) -> dict:
    xs = [to_tensor(letterbox(f, imgsz)[0]).cuda() for f in frames]
    ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    with torch.no_grad():
        for x in xs[:10]:
            det.raw(x)
        eager = []
        for _ in range(3):
            for x in xs:
                ev[0].record()
                det.raw(x)
                ev[1].record()
                torch.cuda.synchronize()
                eager.append(ev[0].elapsed_time(ev[1]) / 1000)
        static = xs[0].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                det.raw(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            det.raw(static)
        graph = []
        for _ in range(3):
            for x in xs:
                static.copy_(x)
                ev[0].record()
                g.replay()
                ev[1].record()
                torch.cuda.synchronize()
                graph.append(ev[0].elapsed_time(ev[1]) / 1000)
    return {"eager_ms": latency_stats(eager), "cuda_graph_ms": latency_stats(graph)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[320, 480, 640])
    ap.add_argument("--skip-npu", action="store_true")
    a = ap.parse_args()
    if carla_running():
        raise SystemExit("stop the CARLA server first (it shares the GPU)")
    cfg = load_config()
    m = cfg["model"]
    frames = [cv2.imread(p) for p in sorted(glob.glob(str(resolve("outputs/dataset/*/*.png"))))]
    ref_det = GpuDetector(str(resolve(m["weights"])), 640, m["conf"], m["iou"], m["max_det"])
    ref = [ref_det(f)[0] for f in frames]
    res: dict = {"frames": len(frames), "reference": "GPU FP32 at 640 (V1 deployment)", "sizes": {}}
    for sz in a.sizes:
        det = GpuDetector(str(resolve(m["weights"])), sz, m["conf"], m["iou"], m["max_det"])
        g = [det(f)[0] for f in frames]
        agree = aggregate_agreement([match_detections(r, d) for r, d in zip(ref, g, strict=True)])
        e: dict = {"gpu_fp32": {"agreement_vs_640": agree,
                                **gpu_timing(det, frames, sz)}}  # fmt: skip
        if not a.skip_npu:
            b = FuriosaRemoteBackend(dtype="bf16", imgsz=sz, encoding="content_raw", threads=2, prefilter=True,
                                     timeout_s=5.0, startup_timeout_s=900,
                                     ssh_opts=("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-C"))  # fmt: skip
            try:
                for i in range(5):
                    b.infer(900 + i, 0.0, frames[i])
                rs = [b.infer(i, 0.0, f) for i, f in enumerate(frames * 2)]
                ok = [r for r in rs if r.ok]
                e["npu_bf16"] = {
                    "server": b.ready,
                    "ok": len(ok),
                    "agreement_vs_640": aggregate_agreement(
                        [match_detections(ref[i % len(frames)], r.dets) for i, r in enumerate(rs) if r.ok][
                            : len(frames)
                        ]
                    ),
                    "exec_ms": latency_stats([r.stages["exec"] for r in ok]),
                    "server_total_ms": latency_stats([r.stages["server_total"] for r in ok]),
                    "e2e_ms": latency_stats([r.latency_s for r in ok]),
                    "payload_bytes": int(np.mean([r.stages["payload_bytes"] for r in ok])),
                    "remote_health": b.remote_health(),
                }
            except RuntimeError as ex:
                e["npu_bf16"] = {"error": repr(ex)}
            finally:
                b.close()
        res["sizes"][str(sz)] = e
        gn = e["gpu_fp32"]
        print(sz, f"gpu recall {gn['agreement_vs_640']['recall_vs_ref']:.3f} eager {gn['eager_ms']['mean_ms']:.2f}"
              f" graph {gn['cuda_graph_ms']['mean_ms']:.2f}",
              "npu", {k: (round(v["mean_ms"], 2) if isinstance(v, dict) and "mean_ms" in v else None)
                      for k, v in e.get("npu_bf16", {}).items()}, flush=True)  # fmt: skip
    out = resolve("reports/evidence/v2_resolution.json")
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
