"""Measured latency samples (seconds) used as empirical distributions by the V2 matrix.

* ``gpu_local``          in-loop GPU perception e2e from the V1 demo log (CARLA sharing the GPU)
* ``npu_remote_v1equiv`` remote RNGD, V1-equivalent transport (letterboxed raw, default threads)
* ``npu_remote_v2``      remote RNGD, best lossless V2 transport (content raw, ssh -C, 2 threads, prefilter)
* ``npu_remote_jpeg95``  remote RNGD, JPEG q95 (lossy)
* ``npu_colocated``      server-side service time of the V2 variant (decode..post on the RNGD host,
                         no network): what a co-located NPU would add, measured on the shared host
Output: reports/evidence/v2_latency_samples.json
"""

from __future__ import annotations

import glob
import json
import os
import time

import cv2

from npu_ad.config import resolve
from npu_ad.evaluation.metrics import latency_stats
from npu_ad.runtime.backends import FuriosaRemoteBackend

VARIANTS = {
    "npu_remote_v1equiv": dict(encoding="letterbox_raw", threads=0, prefilter=False, comp=False),
    "npu_remote_v2": dict(encoding="content_raw", threads=2, prefilter=True, comp=True),
    "npu_remote_jpeg95": dict(encoding="content_jpeg", jpeg_quality=95, threads=2, prefilter=True, comp=False),
}


def main(n: int = 200) -> int:
    demo = glob.glob(str(resolve("outputs/v1/demo/logs/*.jsonl")))[0]
    gpu = [json.loads(ln)["infer_e2e_ms"] / 1000 for ln in open(demo, encoding="utf-8")]
    frames = [cv2.imread(p) for p in sorted(glob.glob(str(resolve("outputs/dataset/*/*.png"))))]
    res = {"samples_s": {"gpu_local": gpu}, "sources": {"gpu_local": os.path.relpath(demo, resolve("."))},
           "experiment_id": "v2_latency_samples_" + time.strftime("%Y%m%dT%H%M%S")}  # fmt: skip
    for name, v in VARIANTS.items():
        opts = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20") + (("-C",) if v.pop("comp") else ())
        b = FuriosaRemoteBackend(dtype="bf16", timeout_s=5.0, ssh_opts=opts, **v)
        try:
            for i in range(5):
                b.infer(900 + i, 0.0, frames[i])
            rs = [b.infer(i, 0.0, frames[i % len(frames)]) for i in range(n)]
            res["samples_s"][name] = [r.latency_s for r in rs if r.ok]
            if name == "npu_remote_v2":
                res["samples_s"]["npu_colocated"] = [r.stages["server_total"] for r in rs if r.ok]
            res["sources"][name] = {"server": b.ready, "failures": sum(not r.ok for r in rs),
                                    "remote_health": b.remote_health()}  # fmt: skip
        finally:
            b.close()
    res["stats_ms"] = {k: latency_stats(v) for k, v in res["samples_s"].items()}
    out = resolve("reports/evidence/v2_latency_samples.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, s in res["stats_ms"].items():
        print(
            f"{k:20s} n {s['n']:4d} mean {s['mean_ms']:7.1f} p50 {s['p50_ms']:7.1f}"
            f" p95 {s['p95_ms']:7.1f} p99 {s['p99_ms']:7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
