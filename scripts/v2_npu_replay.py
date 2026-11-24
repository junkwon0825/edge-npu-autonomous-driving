"""V2 Mode B - NPU replay: recorded CARLA frames -> RNGD BF16 (persistent server) -> V2 decisions.

The 738 letterboxed frames dumped during the frozen v3 closed-loop runs (``outputs/replay/*.npz``,
every 5th tick = 20 Hz, read only) are sent through the remote RNGD server (best lossless V2
transport) and through the local GPU detector. Both detection streams then drive the identical V2
stack (tracker, range Kalman, AEB state machine) open loop, with the ego speed taken from the
recorded run log. Compared: detection agreement, per-tick AEB state, first WARNING/BRAKE time, TTC.
Output: reports/evidence/v2_npu_replay.json
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb_v2 import BRAKE, WARNING, AebStateMachine, view_of
from npu_ad.evaluation.metrics import aggregate_agreement, latency_stats, match_detections
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.mono_range import GroundPlaneRanger
from npu_ad.perception.tracker import MultiObjectTracker
from npu_ad.runtime.backends import FuriosaRemoteBackend
from npu_ad.simulation import geometry as geo

GROUPS = {"vehicle": {2, 3, 5, 7}, "pedestrian": {0}, "cyclist": {1}}
GAIN, PAD_X, PAD_Y = 0.8, 0.0, 140.0  # letterbox 800x450 -> 640x640


def to_image(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, np.float64).reshape(-1, 6).copy()
    d[:, [0, 2]] = ((d[:, [0, 2]] - PAD_X) / GAIN).clip(0, 800)
    d[:, [1, 3]] = ((d[:, [1, 3]] - PAD_Y) / GAIN).clip(0, 450)
    return d


def drive(dets_seq, steps, speeds, dt, cam_h, k) -> list[dict]:
    ranger = GroundPlaneRanger(k=k, image_h=450, cam_height_m=cam_h, cam_x_m=1.5, ego_half_len_m=2.4)
    mot, sm, out = MultiObjectTracker(ranger, GROUPS), AebStateMachine(), []
    for dets, st in zip(dets_seq, steps, strict=True):
        t = float(st) * dt
        mot.update(t, dets)
        o = sm.step(t, [view_of(tr, t) for tr in mot.confirmed()], speeds.get(int(st), 0.0), True)
        out.append({"t": t, "state": o.state, "brake": o.brake, "ttc": o.ttc_s, "range": o.range_m})
    return out


def first(seq, states) -> float | None:
    return next((r["t"] for r in seq if r["state"] in states), None)


def main() -> int:
    cfg = load_config()
    m = cfg["model"]
    gpu = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    npu = FuriosaRemoteBackend(dtype="bf16", encoding="content_raw", threads=2, prefilter=True, timeout_s=5.0,
                               ssh_opts=("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-C"))  # fmt: skip
    k = geo.intrinsics(800, 450, 90.0)
    res: dict = {"server": npu.ready, "runs": {}, "note": "open loop on frames recorded in v3 (GPU closed loop)"}
    all_per, lat = [], []
    try:
        for p in sorted(glob.glob(str(resolve("outputs/replay/*.npz")))):
            name = Path(p).stem
            d = np.load(p)
            frames, steps, dt, cam_h = d["frames"], d["steps"], float(d["dt"]), float(d["cam_height_m"])
            log = resolve(f"outputs/scenario_logs/{name}.jsonl")
            speeds = {json.loads(ln)["step"]: json.loads(ln)["ego_mps"] for ln in open(log, encoding="utf-8")}
            g_seq, n_seq, per = [], [], []
            for i, f in enumerate(frames):
                g = to_image(gpu(f)[0])  # 640x640 input: letterbox is the identity
                r = npu.infer(i, 0.0, f)
                if not r.ok:
                    raise RuntimeError(r.error)
                lat.append(r.latency_s)
                n = to_image(r.dets)  # backend un-letterboxed with identity params -> 640 coords
                g_seq.append(g)
                n_seq.append(n)
                per.append(match_detections(g, n))
            all_per += per
            dg = drive(g_seq, steps, speeds, dt, cam_h, k)
            dn = drive(n_seq, steps, speeds, dt, cam_h, k)
            same = sum(a["state"] == b["state"] for a, b in zip(dg, dn, strict=True)) / len(dg)
            ttc_d = [abs(a["ttc"] - b["ttc"]) for a, b in zip(dg, dn, strict=True)
                     if a["ttc"] is not None and b["ttc"] is not None
                     and math.isfinite(a["ttc"]) and math.isfinite(b["ttc"])]  # fmt: skip
            res["runs"][name] = {
                "frames": len(frames),
                "agreement": aggregate_agreement(per),
                "state_agreement_frac": same,
                "gpu_first_warning_t": first(dg, (WARNING, BRAKE)),
                "npu_first_warning_t": first(dn, (WARNING, BRAKE)),
                "gpu_first_brake_t": first(dg, (BRAKE,)),
                "npu_first_brake_t": first(dn, (BRAKE,)),
                "ttc_abs_diff_median_s": float(np.median(ttc_d)) if ttc_d else None,
                "ttc_abs_diff_max_s": float(np.max(ttc_d)) if ttc_d else None,
            }
            print(
                name[:40], json.dumps({k2: v for k2, v in res["runs"][name].items() if k2 != "agreement"}), flush=True
            )
        res["agreement_all"] = aggregate_agreement(all_per)
        res["npu_e2e_latency_ms"] = latency_stats(lat)
        res["remote_health_end"] = npu.remote_health()
    finally:
        npu.close()
    out = resolve("reports/evidence/v2_npu_replay.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["agreement_all"]), json.dumps(res["npu_e2e_latency_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
