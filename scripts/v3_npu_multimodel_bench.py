# ruff: noqa: E501  (long result/row dict literals kept on one line)
"""V3: RNGD multi-model server benchmark and lane correctness (detection on furiosa:0, lane U-Net on furiosa:1).

1. Latency per request type (persistent server, V2 lossless transport): detection only, lane only, both
   (one upload, models run concurrently on separate PEs); server stage times per model.
2. Lane correctness: RNGD BF16 mask vs local GPU FP32 mask on the same frames (pixel IoU = agreement) and
   both vs ground truth on the held-out Town03 lane set (upsampled to the camera size so both paths see the
   identical 800x450 input).
3. Detection agreement of the V3 server vs local GPU (sanity check against V2).
Output: reports/evidence/v3_npu_multimodel_bench.json
"""

from __future__ import annotations

import glob
import json
import random

import cv2
import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import aggregate_agreement, latency_stats, match_detections
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.lane_net import load_lane_model
from npu_ad.runtime.multimodel import GpuLaneBackend, RngdMultiModelClient


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main() -> int:
    cfg = load_config()
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    gl = GpuLaneBackend(load_lane_model(str(resolve("models/lane_unet_s3c32.pt")), "cuda:0"))
    cam = [cv2.imread(p) for p in sorted(glob.glob(str(resolve("outputs/dataset/*/*.png"))))]
    man = json.loads(resolve("outputs/v3/lane_dataset/manifest.json").read_text(encoding="utf-8"))
    test = [s["id"] for s in man["samples"] if s["map"] == "Town03"]
    test = random.Random(0).sample(test, 150)
    root = resolve("outputs/v3/lane_dataset")
    c = RngdMultiModelClient(timeout_s=5.0, startup_timeout_s=1200)
    res: dict = {"server": c.ready, "frames_camera": len(cam), "lane_test_frames": len(test)}
    try:
        for op, fn in (("det", c.infer_det), ("lane", c.infer_lane), ("both", c.infer_both)):
            for i in range(5):
                fn(900 + i, 0.0, cam[i])
            rs = [fn(i, 0.0, cam[i % len(cam)]) for i in range(144)]
            ok = [r for r in rs if r.ok]
            stages = {}
            for k in {k for r in ok for k in getattr(r, "stages", getattr(getattr(r, "det", None), "stages", {}))}:
                vals = [getattr(r, "stages", getattr(getattr(r, "det", None), "stages", {})).get(k) for r in ok]
                vals = [v for v in vals if isinstance(v, float)]
                if vals:
                    stages[k] = latency_stats(vals)
            res[op] = {"ok": len(ok), "e2e_ms": latency_stats([r.latency_s for r in ok]), "stages_ms": stages}
            print(
                op, len(ok), round(res[op]["e2e_ms"]["mean_ms"], 1), round(res[op]["e2e_ms"]["p95_ms"], 1), flush=True
            )
        # detection agreement (V3 server vs local GPU)
        per = [match_detections(det(f)[0], c.infer_det(i, 0.0, f).dets) for i, f in enumerate(cam)]
        res["det_agreement_vs_gpu"] = aggregate_agreement(per)
        # lane: NPU vs GPU agreement on camera frames
        agree = [iou(c.infer_lane(i, 0.0, f).mask, gl.infer(i, 0.0, f).mask) for i, f in enumerate(cam)]
        res["lane_npu_vs_gpu_iou_camera_frames"] = {"mean": float(np.mean(agree)), "min": float(np.min(agree)),
                                                   "n": len(agree)}  # fmt: skip
        # lane vs ground truth on held-out Town03 frames (identical 800x450 input for both paths)
        tp = {"npu": [0, 0, 0], "gpu": [0, 0, 0]}
        agree_t = []
        for i, sid in enumerate(test):
            img = cv2.resize(cv2.imread(str(root / "rgb" / f"{sid}.jpg")), (800, 450), interpolation=cv2.INTER_LINEAR)
            gt = cv2.imread(str(root / "mask" / f"{sid}.png"), 0) > 127
            mn = c.infer_lane(10_000 + i, 0.0, img).mask
            mg = gl.infer(i, 0.0, img).mask
            agree_t.append(iou(mn, mg))
            for k, mm in (("npu", mn), ("gpu", mg)):
                tp[k][0] += int((mm & gt).sum())
                tp[k][1] += int((mm & ~gt).sum())
                tp[k][2] += int((~mm & gt).sum())
        for k, (t, f, n) in tp.items():
            pr, rc = t / max(t + f, 1), t / max(t + n, 1)
            res[f"lane_{k}_vs_gt_town03"] = {"iou": t / max(t + f + n, 1), "f1": 2 * pr * rc / max(pr + rc, 1e-9),
                                             "precision": pr, "recall": rc, "n": len(test)}  # fmt: skip
        res["lane_npu_vs_gpu_iou_town03"] = {"mean": float(np.mean(agree_t)), "min": float(np.min(agree_t))}
        res["remote_health"] = c.remote_health()
    finally:
        c.close()
    out = resolve("reports/evidence/v3_npu_multimodel_bench.json")
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    for k in ("det_agreement_vs_gpu", "lane_npu_vs_gpu_iou_camera_frames", "lane_npu_vs_gt_town03", "lane_gpu_vs_gt_town03",
              "lane_npu_vs_gpu_iou_town03"):  # fmt: skip
        print(k, json.dumps(res[k])[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
