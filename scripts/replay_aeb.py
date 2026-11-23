"""Open-loop AEB replay on identical frames: GPU vs. NPU perception (perception-driven ``mono`` chain).

Closed-loop NPU-in-the-loop is not possible from this desktop at useful rates (a raw frame takes
~121 ms over SSH, reports/evidence/npu_link.json). Instead, scenario runs dump the exact letterboxed
detector inputs (``run_scenarios.py --dump-every 5``, i.e. 20 Hz). Both accelerators process the same
frames; the same host decode/NMS, ground-plane ranger, CIPO tracker and AEB controller then replay
the decision sequence. Comparison stops at the first brake decision, because after that the logged
trajectory (driven by the in-loop GPU) no longer reflects what the replayed controller would do.

  python scripts/replay_aeb.py prepare            # outputs/npu/replay_inputs.pt (upload to the NPU host)
  python scripts/replay_aeb.py gpu                # raw GPU outputs for the same inputs
  python scripts/replay_aeb.py replay             # needs outputs/npu/npu_outputs_fp32_replay.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb import AebController, AebParams
from npu_ad.evaluation.metrics import aggregate_agreement, match_detections
from npu_ad.perception.mono_range import CipoTracker, GroundPlaneRanger
from npu_ad.perception.yolo_raw import RawDecoder, letterbox, load_raw_model, scale_boxes, to_tensor
from npu_ad.simulation import geometry as geo

INPUTS = "outputs/npu/replay_inputs.pt"
GPU_OUT = "outputs/npu/gpu_outputs_replay.pt"
NPU_OUT = "outputs/npu/npu_outputs_fp32_replay.pt"


def _dumps() -> list[Path]:
    return sorted(resolve("outputs/replay").glob("*.npz"))


def prepare() -> None:
    names, frames = [], []
    for p in _dumps():
        d = np.load(p)
        for st, fr in zip(d["steps"], d["frames"], strict=True):
            names.append(f"{p.stem}:{int(st)}")
            frames.append(torch.from_numpy(fr))
    out = resolve(INPUTS)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "lb_bgr_uint8": torch.stack(frames)}, out)
    print(len(names), "frames from", len(_dumps()), "runs ->", out)


@torch.no_grad()
def gpu() -> None:
    cfg = load_config()
    raw, _ = load_raw_model(str(resolve(cfg["model"]["weights"])))
    raw = raw.cuda()
    data = torch.load(resolve(INPUTS))
    boxes, scores = [], []
    for img in data["lb_bgr_uint8"]:
        b, s = raw(to_tensor(img.numpy()).cuda())
        boxes.append(b.float().cpu())
        scores.append(s.float().cpu())
    torch.save({"names": data["names"], "boxes": torch.cat(boxes), "scores": torch.cat(scores)}, resolve(GPU_OUT))
    print("gpu outputs ->", GPU_OUT)


def _replay_run(stem: str, idx: list[int], out: dict, dec, lb, cfg, groups) -> dict:
    rows = {
        r["step"]: r for r in map(json.loads, resolve(f"outputs/scenario_logs/{stem}.jsonl").read_text().splitlines())
    }
    w, h = cfg["camera"]["width"], cfg["camera"]["height"]
    k = geo.intrinsics(w, h, cfg["camera"]["fov"])
    cal = np.load(resolve(f"outputs/replay/{stem}.npz"))  # calibration measured by the dumped run
    ranger = GroundPlaneRanger(k, h, float(cal["cam_height_m"]), cfg["camera"]["x"], float(cal["ego_half_len_m"]))
    tracker, aeb = CipoTracker(ranger, groups), AebController(AebParams())
    dt = float(cal["dt"])
    res = {"first_brake_t": None, "gt_ttc_at_first_brake_s": None, "dets": []}
    for i in idx:
        step = int(out["names"][i].split(":")[1])
        det = dec(out["boxes"][i : i + 1], out["scores"][i : i + 1], 0.25, 0.7, 300)[0]
        d = scale_boxes(det, lb, w, h)
        res["dets"].append(d)
        cipo, _ = tracker.update(step * dt, d)
        r = rows.get(step)
        if r is None:
            break
        decision = aeb.decide([cipo] if cipo else [], r["ego_mps"])
        if decision.brake > 0 and res["first_brake_t"] is None:
            res["first_brake_t"] = step * dt
            res["gt_ttc_at_first_brake_s"] = r["gt_ttc_s"]
            res["decision"] = decision.reason
            break
    return res


def replay() -> None:
    cfg = load_config()
    _, head = load_raw_model(str(resolve(cfg["model"]["weights"])))
    dec = RawDecoder(head, cfg["model"]["imgsz"])
    lb = letterbox(np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8), cfg["model"]["imgsz"])[1]
    groups = {"vehicle": set(cfg["classes"]["vehicle"]), "pedestrian": set(cfg["classes"]["pedestrian"])}
    g, n = torch.load(resolve(GPU_OUT)), torch.load(resolve(NPU_OUT))
    if g["names"] != n["names"] or len(g["names"]) != g["boxes"].shape[0] or n["boxes"].shape != g["boxes"].shape:
        raise SystemExit("GPU/NPU replay outputs are not aligned (names or shapes differ)")
    stems: dict[str, list[int]] = {}
    for i, name in enumerate(g["names"]):
        stems.setdefault(name.split(":")[0], []).append(i)
    runs, per_frame = {}, []
    for stem, idx in stems.items():
        rg = _replay_run(stem, idx, g, dec, lb, cfg, groups)
        rn = _replay_run(stem, idx, n, dec, lb, cfg, groups)
        m = min(len(rg["dets"]), len(rn["dets"]))
        matches = [match_detections(rg["dets"][j], rn["dets"][j]) for j in range(m)]
        per_frame += matches
        runs[stem] = {
            "frames_compared": m,
            "gpu_first_brake_t": rg["first_brake_t"],
            "npu_first_brake_t": rn["first_brake_t"],
            "gpu_gt_ttc_at_first_brake_s": rg["gt_ttc_at_first_brake_s"],
            "npu_gt_ttc_at_first_brake_s": rn["gt_ttc_at_first_brake_s"],
            "detection_agreement": aggregate_agreement(matches),
        }
        print(stem, json.dumps({k: v for k, v in runs[stem].items() if k != "detection_agreement"}))
    out = {
        "method": "open-loop replay at 20 Hz on identical letterboxed frames; perception-driven mono chain; "
        "compared up to the first brake decision; NPU = furiosa:0 FP32 compiled raw module",
        "runs": runs,
        "detection_agreement_all": aggregate_agreement(per_frame),
    }
    resolve("reports/evidence/replay_npu_vs_gpu.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "gpu", "replay"])
    a = ap.parse_args()
    {"prepare": prepare, "gpu": gpu, "replay": replay}[a.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
