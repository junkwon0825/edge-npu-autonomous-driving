"""V2 P0: where is time lost between a CARLA frame on the desktop and detections from the RNGD?

Runs bounded experiments against ``remote/npu_server.py`` (one persistent SSH session per variant):

1. transport only: ECHO round trips for 1 B ... 1.23 MB payloads
2. SSH process overhead: a new ``ssh`` process per request vs the persistent session
3. inference variants: frame encoding (letterboxed raw / content raw / JPEG q90 / q75),
   remote CPU threads, exact pre-NMS score filter, SSH compression
4. pipelining: 2 requests in flight vs 1 (throughput)

Each inference response carries per-stage remote durations; detections are compared with the local
GPU FP32 detector on the same frames (agreement, not mAP). Output:
``reports/evidence/v2_transport_profile.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import aggregate_agreement, latency_stats, match_detections
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.runtime.backends import OP_INFER, FuriosaRemoteBackend

VARIANTS = [
    # name, encoding, jpeg_q, threads, prefilter, ssh compression
    ("v1_equiv_letterbox_raw", "letterbox_raw", 90, 0, False, False),
    ("content_raw", "content_raw", 90, 0, False, False),
    ("content_raw_t4_prefilter", "content_raw", 90, 4, True, False),
    ("content_raw_t2_prefilter", "content_raw", 90, 2, True, False),
    ("jpeg90_t4_prefilter", "content_jpeg", 90, 4, True, False),
    ("jpeg75_t4_prefilter", "content_jpeg", 75, 4, True, False),
    ("content_raw_t4_prefilter_sshC", "content_raw", 90, 4, True, True),
]
VARIANTS_R2 = [  # round 2: best lossless candidates, more samples
    ("content_raw_t2_prefilter_sshC", "content_raw", 90, 2, True, True),
    ("png1_t2_prefilter", "content_png", 90, 2, True, False),
    ("jpeg95_t2_prefilter", "content_jpeg", 95, 2, True, False),
    ("content_raw_t2_prefilter", "content_raw", 90, 2, True, False),
]


def stage_stats(results) -> dict:
    keys = ["encode", "send", "rx", "decode", "pre", "h2d", "exec", "d2h", "post", "server_total", "network_est", "rtt"]
    out = {}
    for k in keys:
        v = [r.stages[k] for r in results if r.ok and k in r.stages]
        if v:
            out[k] = latency_stats(v)
    out["e2e"] = latency_stats([r.latency_s for r in results if r.ok])
    out["payload_bytes_mean"] = float(np.mean([r.stages.get("payload_bytes", 0) for r in results if r.ok]))
    return out


def pipelined(b: FuriosaRemoteBackend, frames, n: int, depth: int) -> dict:
    """Throughput with ``depth`` requests outstanding (server processes them in order)."""
    payloads = [b.encode(f)[:4] for f in frames]
    sent = got = 0
    t0 = time.perf_counter()
    while got < n:
        while sent < n and sent - got < depth:
            enc, w, h, p = payloads[sent % len(payloads)]
            b._send(OP_INFER, 10_000 + sent, enc, w, h, p)
            sent += 1
        b._wait(10_000 + got, 10.0)
        got += 1
    dt = time.perf_counter() - t0
    return {"depth": depth, "n": n, "wall_s": dt, "fps": n / dt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="furiosa-npu")
    ap.add_argument("--n", type=int, default=72, help="inferences per variant")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default="reports/evidence/v2_transport_profile.json")
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config()
    paths = sorted(glob.glob(str(resolve("outputs/dataset/*/*.png"))))
    frames = [cv2.imread(p) for p in paths]
    m = cfg["model"]
    gpu = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    ref = [gpu(f)[0] for f in frames]
    res: dict = {
        "experiment_id": "v2_transport_profile_" + time.strftime("%Y%m%dT%H%M%S"),
        "host_alias": a.host,
        "dtype": a.dtype,
        "frames": len(frames),
        "frame_shape": list(frames[0].shape),
        "n_per_variant": a.n,
        "reference": "local GPU FP32 detections on the same frames (agreement, not mAP)",
        "variants": {},
    }
    # 2. SSH process overhead: new process per request (connect + auth + remote shell) vs persistent
    payload = os.urandom(691_200)
    t_new = []
    for _ in range(5 if a.round == 1 else 0):
        t0 = time.perf_counter()
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", a.host, "cat > /dev/null"], input=payload, check=True, timeout=60
        )  # fmt: skip
        t_new.append(time.perf_counter() - t0)
    if t_new:
        res["ssh_new_process_per_request_691kB"] = latency_stats(t_new)
    for name, enc, q, threads, pre, comp in VARIANTS if a.round == 1 else VARIANTS_R2:
        opts = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20") + (("-C",) if comp else ())
        b = FuriosaRemoteBackend(
            host=a.host, dtype=a.dtype, encoding=enc, jpeg_quality=q, timeout_s=10.0, ssh_opts=opts,
            threads=threads, prefilter=pre,
        )  # fmt: skip
        try:
            v: dict = {
                "encoding": enc,
                "jpeg_quality": q,
                "threads": threads,
                "prefilter": pre,
                "ssh_compression": comp,
            }
            v["server"] = b.ready
            if name == VARIANTS[0][0] and a.round == 1:
                v["echo_rtt"] = {
                    str(sz): latency_stats([b.echo(sz, fid=900 + i) for i in range(20)])
                    for sz in (1, 60_000, 691_200, 1_228_800)
                }
            for i in range(3):  # warm-up
                b.infer(800 + i, 0.0, frames[i])
            rs = [b.infer(i, 0.0, frames[i % len(frames)]) for i in range(a.n)]
            v["ok"] = sum(r.ok for r in rs)
            v["errors"] = [r.error for r in rs if not r.ok][:3]
            v["stages"] = stage_stats(rs)
            per = [match_detections(ref[i % len(frames)], r.dets) for i, r in enumerate(rs) if r.ok]
            v["agreement_vs_gpu"] = aggregate_agreement(per)
            if name in ("content_raw_t4_prefilter", "content_raw_t2_prefilter_sshC", "png1_t2_prefilter"):
                v["pipelined"] = [pipelined(b, frames, 40, d) for d in (1, 2)]
            v["remote_health"] = b.remote_health()
        finally:
            b.close()
        res["variants"][name] = v
        s = v["stages"]
        print(
            f"{name:34s} e2e p50 {s['e2e']['p50_ms']:7.1f} ms  rx {s['rx']['p50_ms']:5.1f}"
            f"  pre {s['pre']['p50_ms']:5.1f}"
            f"  exec {s['exec']['p50_ms']:4.1f}  d2h {s['d2h']['p50_ms']:5.1f}  post {s['post']['p50_ms']:5.1f}"
            f"  recall {v['agreement_vs_gpu'].get('recall_vs_ref', float('nan')):.3f}",
            flush=True,
        )
    out = resolve(a.out)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
