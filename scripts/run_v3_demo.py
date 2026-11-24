# ruff: noqa: E501  (long call kept on one line)
"""V3 portfolio demo: Mode D (RNGD detection + RNGD lane segmentation) closed loop with overlay.

The same code path as the V3 matrix (``v3_run_scenarios.run``), with a read-only ``on_tick`` observer
that draws the overlay; it never influences control. Clips, in one video:
1. ``ped_cornering``: lane keeping on the 39 m-radius curve, pedestrian crossing -> AEB while cornering
2. ``curve_lead_brake``: lane keeping on the 147 m-radius curve, lead vehicle brakes hard
3. ``lead_vehicle_braking`` with ``npu_failure``: RNGD server crash -> explicit GPU failover, logged

Prerequisites: a CARLA 0.9.16 server (``CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2000``), the lane
weights ``models/lane_unet_s3c32.pt``, and the ``furiosa-npu`` SSH alias for the RNGD clips.

Usage: python scripts/run_v3_demo.py [--out assets/demo/v3] [--clips ped_cornering curve_lead_brake npu_failover]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import v3_run_scenarios as v3  # noqa: E402

from npu_ad.config import load_config, resolve  # noqa: E402
from npu_ad.logutil import setup_logging  # noqa: E402
from npu_ad.perception.gpu_runner import GpuDetector  # noqa: E402
from npu_ad.perception.lane_net import load_lane_model  # noqa: E402
from npu_ad.runtime.multimodel import RngdMultiModelClient  # noqa: E402
from npu_ad.viz.overlay_v3 import OverlayV3  # noqa: E402

CLIPS = {  # name -> (scenario, mode, km/h, radar, scheduler, fault)
    "ped_cornering": ("ped_cornering", "D", 30.0, False, "fixed", ""),
    "curve_lead_brake": ("curve_lead_brake", "D", 30.0, False, "fixed", ""),
    "npu_failover": ("lead_vehicle_braking", "B", 30.0, False, "fixed", "npu_failure"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/demo/v3")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS), choices=list(CLIPS))
    a = ap.parse_args()
    setup_logging("run_v3_demo")
    cfg = load_config()
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    lane_model = load_lane_model(str(resolve("models/lane_unet_s3c32.pt")), "cuda:0")
    out = resolve(a.out)
    ov = OverlayV3(out, fps=33.3, name="demo_v3.mp4")
    summary = {"clips": [], "note": "overlay values are the controller's inputs; ground truth only in the summary"}
    for clip in a.clips:
        sc, mode, kmh, radar, sched, fault = CLIPS[clip]
        client = RngdMultiModelClient(timeout_s=1.0, startup_timeout_s=900)
        try:
            t0 = time.perf_counter()
            r = v3.run(cfg, sc, v3.SCENARIOS[sc], mode, kmh, det, lane_model, client, radar, sched, fault,
                       resolve("outputs/v3/demo_logs"), on_tick=lambda bgr, info, c=clip: ov.add(bgr, info, c))  # fmt: skip
        finally:
            client.close()
        keys = ("scenario", "mode", "ego_kmh", "collision", "min_gap_m", "first_brake_t", "lat_rmse_m", "lat_max_m",
                "lane_departures", "lka_unavailable_frac", "det_age_ms", "lane_age_ms", "failover_events")  # fmt: skip
        summary["clips"].append({"clip": clip, "wall_s": time.perf_counter() - t0} | {k: r.get(k) for k in keys})
        print(clip, {k: r.get(k) for k in ("collision", "min_gap_m", "lat_rmse_m", "lane_departures")}, flush=True)
    shots = ov.close()
    summary["screenshots"] = shots
    (out / "demo_v3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("screenshots:", shots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
