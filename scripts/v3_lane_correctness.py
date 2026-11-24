# ruff: noqa: E501  (long result/row dict literals kept on one line)
"""V3: same-frame GPU vs RNGD lane-segmentation correctness, separating the causes of the night/rain failure.

The same weights (``models/lane_unet_s3c32.pt``) on the same frames, through six paths:

| path | device / precision | preprocessing |
|---|---|---|
| ``gpu_fp32``          | local CUDA, FP32 (closed-loop GPU path, reference) | 800x450 -> INTER_AREA 512x256 |
| ``gpu_fp32_npuprep``  | local CUDA, FP32 | NPU-client path: letterbox content 640x360 (INTER_LINEAR) -> INTER_AREA 512x256 |
| ``gpu_bf16``          | local CUDA, BF16, NPU-exported graph (BN folded, ConvT rewritten) | direct |
| ``gpu_bf16_npuprep``  | local CUDA, BF16, NPU-exported graph | NPU-client path |
| ``rngd_live``         | remote RNGD BF16 via the V3 server (exact closed-loop request: ``content_raw``) | NPU-client path |
| ``rngd_directprep``    | remote RNGD BF16, frame resized locally 800x450 -> INTER_AREA 512x256 and sent at that size | direct (the server's resize is then the identity; the server rejects frames wider than 640) |

Hypotheses (night/rain lane failure of modes C/D):
- A numerical precision: ``gpu_bf16`` / ``rngd_directprep`` vs ``gpu_fp32`` (same preprocessing)
- C preprocessing: ``gpu_fp32_npuprep`` vs ``gpu_fp32`` (same precision); ``rngd_live`` vs ``rngd_directprep``
- RNGD-specific numerics: ``rngd_directprep`` vs ``gpu_bf16``
- B transport / frame age is not present here (offline, no time): if offline RNGD masks are good, the
  closed-loop failure is temporal; the closed-loop logs and the lane-delay runs answer B.

Sources: ``dataset`` (512x256 lane set with ground-truth masks, upsampled to 800x450; per map x weather)
and ``frames`` (real 800x450 closed-loop frames from ``scripts/v3_capture_frames.py`` with the ground-truth
lane offset). Metrics per path: mask IoU vs ``gpu_fp32`` and vs ground truth, pixel disagreement, positive
fraction, logit statistics (local paths), lane-geometry validity and offset error.
Output: reports/evidence/v3_lane_correctness.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.perception.lane_geometry import CameraModel, LaneGeometry
from npu_ad.perception.lane_net import LANE_IN_H, LANE_IN_W, export_for_npu, load_lane_model, preprocess
from npu_ad.perception.yolo_raw import letterbox
from npu_ad.runtime.multimodel import RngdMultiModelClient
from npu_ad.simulation import geometry as geo

DATASET_GROUPS = [("Town04", "ClearNight"), ("Town10HD_Opt", "ClearNight"), ("Town04", "HardRainNoon"),
                  ("Town03", "HardRainNoon"), ("Town03", "ClearNoon"), ("Town04", "WetCloudySunset"),
                  ("Town04", "ClearSunset"), ("Town10HD_Opt", "ClearNoon")]  # fmt: skip
PATHS = ["gpu_fp32", "gpu_fp32_npuprep", "gpu_bf16", "gpu_bf16_npuprep", "rngd_live", "rngd_directprep"]


def npu_client_input(bgr: np.ndarray) -> np.ndarray:
    """What the RNGD client sends for ``content_raw``: the letterbox content (640x360 for 800x450)."""
    lb_img, lb = letterbox(bgr, 640)
    y0, x0 = int(lb.pad_y), int(lb.pad_x)
    return lb_img[y0 : 640 - y0, x0 : 640 - x0]


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


class FullFrameClient(RngdMultiModelClient):
    """Sends the frame already resized to the lane input (direct preprocessing); server resize = identity."""

    full = False

    def encode(self, bgr):
        if not self.full:
            return super().encode(bgr)
        t0 = time.perf_counter()
        small = cv2.resize(bgr, (LANE_IN_W, LANE_IN_H), interpolation=cv2.INTER_AREA)
        lb = letterbox(bgr, self.imgsz)[1]  # only used for detection boxes, not for the lane mask
        return 1, LANE_IN_W, LANE_IN_H, np.ascontiguousarray(small).tobytes(), lb, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["dataset", "frames"])
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--frames-dir", default="outputs/v3/lane_frames")
    ap.add_argument("--out", default="reports/evidence/v3_lane_correctness.json")
    a = ap.parse_args()
    cfg = load_config()
    model = load_lane_model(str(resolve("models/lane_unet_s3c32.pt")), "cuda:0")
    bf16 = export_for_npu(model.cpu()).to("cuda:0", torch.bfloat16)
    model = model.to("cuda:0")
    k = geo.intrinsics(800, 450, cfg["camera"]["fov"])

    def cam(h):
        return CameraModel(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), h, cfg["camera"]["x"])

    @torch.no_grad()
    def local(bgr, half: bool, npuprep: bool):
        x = preprocess(npu_client_input(bgr) if npuprep else bgr).to("cuda:0")
        lo = (bf16(x.to(torch.bfloat16)) if half else model(x))[0, 0].float().cpu().numpy()
        return lo > 0, lo

    client = FullFrameClient(models="lane", timeout_s=5.0, startup_timeout_s=1200)
    res: dict = {"server": client.ready, "opencv_local": cv2.__version__, "torch_local": torch.__version__,
                 "lane_input": [LANE_IN_H, LANE_IN_W], "paths": PATHS, "sources": {}}  # fmt: skip

    def remote(fid, bgr, full: bool):
        client.full = full
        r = client.infer_lane(fid, 0.0, bgr)
        return (r.mask if r.ok else None), r.latency_s

    try:
        for _ in range(3):  # warm-up
            remote(0, np.zeros((450, 800, 3), np.uint8), False)
        items = []  # (group, sequence id, bgr, gt_mask | None, gt_offset | None, cam height)
        if "dataset" in a.sources:
            root = resolve("outputs/v3/lane_dataset")
            man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            split = json.loads(resolve("reports/evidence/v3_lane_training.json").read_text(encoding="utf-8"))
            for town, wx in DATASET_GROUPS:
                ids = sorted(s["id"] for s in man["samples"] if s["map"] == town and s["weather"] == wx)
                for sid in random.Random(0).sample(ids, min(a.per_group, len(ids))):
                    img = cv2.resize(
                        cv2.imread(str(root / "rgb" / f"{sid}.jpg")), (800, 450), interpolation=cv2.INTER_LINEAR
                    )
                    gt = cv2.imread(str(root / "mask" / f"{sid}.png"), 0) > 127
                    items.append((f"dataset:{town}:{wx}", sid, img, gt, None, cfg["camera"]["z"]))
            res["dataset_note"] = ("512x256 JPEG lane-set frames upsampled to 800x450; Town04/Town10 frames are training/validation "
                                   f"maps, Town03 is held out (split sizes {split.get('split')}). Agreement between paths is "
                                   "independent of the split; ground-truth IoU on training maps is optimistic.")  # fmt: skip
        if "frames" in a.sources:
            for fj in sorted(resolve(a.frames_dir).glob("*/frames.json")):
                meta = json.loads(fj.read_text(encoding="utf-8"))
                for fr in meta["frames"]:
                    img = cv2.imread(str(fj.parent / fr["file"]))
                    items.append((f"frames:{meta['scenario']}:{meta['weather']}", meta["scenario"], img, None,
                                  fr["gt_offset"], fr["cam_height_m"]))  # fmt: skip
        per = defaultdict(lambda: defaultdict(list))
        geoms: dict = {}
        for i, (grp, seq, img, gt, gt_off, h) in enumerate(items):
            masks, logits, lat, offs = {}, {}, {}, {}
            masks["gpu_fp32"], logits["gpu_fp32"] = local(img, False, False)
            masks["gpu_fp32_npuprep"], logits["gpu_fp32_npuprep"] = local(img, False, True)
            masks["gpu_bf16"], logits["gpu_bf16"] = local(img, True, False)
            masks["gpu_bf16_npuprep"], logits["gpu_bf16_npuprep"] = local(img, True, True)
            masks["rngd_live"], lat["rngd_live"] = remote(1000 + 2 * i, img, False)
            masks["rngd_directprep"], lat["rngd_directprep"] = remote(1001 + 2 * i, img, True)
            ref, ref_lo = masks["gpu_fp32"], logits["gpu_fp32"]
            g = per[grp]
            g["ref_pos_frac"].append(float(ref.mean()))
            g["ref_low_margin_frac"].append(float((np.abs(ref_lo[ref]) < 1.0).mean()) if ref.any() else None)
            for p in PATHS:
                mk = masks[p]
                if mk is None:
                    g[f"{p}:failed"].append(1)
                    continue
                g[f"{p}:iou_vs_gpu_fp32"].append(iou(mk, ref))
                g[f"{p}:pixel_disagree"].append(float((mk != ref).mean()))
                g[f"{p}:pos_frac"].append(float(mk.mean()))
                if gt is not None:
                    g[f"{p}:iou_vs_gt"].append(iou(mk, gt))
                if p in logits:
                    g[f"{p}:logit_mae_vs_fp32"].append(float(np.abs(logits[p] - ref_lo).mean()))
                # lane geometry: sequences keep state (as in closed loop), dataset frames are independent
                key = (p, seq) if gt is None else (p, i)
                lg = geoms.setdefault(key, LaneGeometry(cam(h)))
                est = lg.estimate(mk)
                g[f"{p}:geom_valid"].append(int(est.valid))
                g[f"{p}:geom_two_sides"].append(int(est.valid and est.n_left > 0 and est.n_right > 0))
                if est.valid:
                    offs[p] = est.offset_m
                    if gt_off is not None:
                        g[f"{p}:offset_err_vs_gt"].append(abs(est.offset_m - gt_off))
            for p in PATHS[1:]:  # lane-centre offset difference vs the reference, frames where both are valid
                if p in offs and "gpu_fp32" in offs:
                    g[f"{p}:offset_diff_vs_gpu_fp32"].append(abs(offs[p] - offs["gpu_fp32"]))
            for p in ("rngd_live", "rngd_directprep"):
                g[f"{p}:latency_ms"].append(lat[p] * 1000)
            if i % 40 == 0:
                print(
                    i,
                    len(items),
                    grp,
                    {
                        p: round(np.mean(per[grp][f"{p}:iou_vs_gpu_fp32"]), 3)
                        for p in PATHS
                        if per[grp][f"{p}:iou_vs_gpu_fp32"]
                    },
                    flush=True,
                )
        for grp, g in per.items():
            out = {"n": len(g["ref_pos_frac"])}
            for kk, v in g.items():
                v = [x for x in v if x is not None]
                if not v:
                    continue
                if kk.endswith(":failed"):
                    out[kk] = len(v)
                else:
                    out[kk] = {"mean": float(np.mean(v)), "p5": float(np.percentile(v, 5)), "min": float(np.min(v))}
            res["sources"][grp] = out
    finally:
        client.close()
    path = resolve(a.out)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    for grp, o in res["sources"].items():
        print(grp, o["n"], {p: (round(o[f"{p}:iou_vs_gpu_fp32"]["mean"], 3), round(o[f"{p}:geom_valid"]["mean"], 2))
                            for p in PATHS if f"{p}:iou_vs_gpu_fp32" in o})  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
