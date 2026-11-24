"""Capture real closed-loop camera frames (800x450, lossless PNG) for the GPU-vs-RNGD lane correctness test.

Drives the V3 runner in Mode A (GPU detection + GPU lane) on the lane-keeping routes and saves every
``--every``-th camera frame with the ground-truth lane offset / heading / width of that tick (evaluation
only). The runner is unchanged; frames are taken by its read-only ``on_tick`` observer.
Output: outputs/v3/lane_frames/<scenario>/<tick>.png + frames.json (not committed; large).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import v3_run_scenarios as v3  # noqa: E402

from npu_ad.config import load_config, resolve  # noqa: E402
from npu_ad.logutil import setup_logging  # noqa: E402
from npu_ad.perception.gpu_runner import GpuDetector  # noqa: E402
from npu_ad.perception.lane_net import load_lane_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=["lk_night", "lk_rain", "lk_mild", "lk_strong"])
    ap.add_argument("--every", type=int, default=8, help="save every n-th camera frame")
    ap.add_argument("--out", default="outputs/v3/lane_frames")
    a = ap.parse_args()
    setup_logging("v3_capture_frames")
    cfg = load_config()
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    lane_model = load_lane_model(str(resolve("models/lane_unet_s3c32.pt")), "cuda:0")
    for sc in a.scenarios:
        d = resolve(a.out) / sc
        d.mkdir(parents=True, exist_ok=True)
        meta, n = [], [0]

        def grab(bgr, info, d=d, meta=meta, n=n):
            n[0] += 1
            if n[0] % a.every:
                return
            row = info["row"]
            name = f"{int(round(info['t'] * 100)):05d}.png"
            cv2.imwrite(str(d / name), bgr)
            meta.append({"file": name, "t": info["t"], "gt_offset": row["gt_offset"], "gt_heading": row["gt_heading"],
                         "lane_width": row["lane_width"], "cam_height_m": info["cam"].height_m})  # fmt: skip

        r = v3.run(cfg, sc, v3.SCENARIOS[sc], "A", 30.0, det, lane_model, None, False, "fixed", "",
                   resolve("outputs/v3/capture_logs"), on_tick=grab)  # fmt: skip
        sce = v3.SCENARIOS[sc]
        info = {"scenario": sc, "route": sce["route"], "map": r["map"], "weather": sce["weather"], "frames": meta,
                "run_lat_rmse_m": r["lat_rmse_m"]}  # fmt: skip
        (d / "frames.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(sc, len(meta), "frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
