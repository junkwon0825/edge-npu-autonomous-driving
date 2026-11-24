"""V3 overlay: detection + lane perception, LKA / AEB / arbitration, backends and freshness.

Everything drawn comes from the controller's own inputs (never ground truth):
- the confirmed tracks of the newest accepted detection result, on the current frame
- the newest accepted lane mask
- the estimated lane centre (exact inverse of ``LaneGeometry.ground_points``)

So the visible lag is the real frame age. Unavailable values are printed as ``n/a``.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, AMBER, RED, CYAN, MAGENTA = (
    (255, 255, 255),
    (80, 200, 80),
    (0, 190, 255),
    (60, 60, 230),
    (230, 200, 40),
    (220, 80, 220),
)
STATE_COL = {"CLEAR": GREEN, "WATCH": GREEN, "WARNING": AMBER, "BRAKE": RED, "RELEASE": AMBER}


def _f(v, spec: str, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:{spec}}{unit}"


def project_centre(est, cam, x_from: float = 5.0, x_to: float = 30.0, n: int = 26) -> np.ndarray:
    """Image points (u, v) of the estimated lane centre, the inverse of the flat-ground projection."""
    pts = []
    for x in np.linspace(x_from, x_to, n):
        z = x - cam.x_m
        if z <= 0.5:
            continue
        y = est.lateral_at(float(x))
        pts.append((cam.cx + cam.fx * y / z, cam.cy + cam.fy * cam.height_m / z))
    return np.asarray(pts, np.int32)


class OverlayV3:
    def __init__(self, out_dir: Path, fps: float, name: str = "demo_v3.mp4"):
        self.dir, self.fps, self.name = out_dir, fps, name
        self.writer = None
        self.shots: dict[str, np.ndarray] = {}

    def draw(self, bgr: np.ndarray, i: dict) -> np.ndarray:
        img = bgr.copy()
        h, w = img.shape[:2]
        if i.get("lane_mask") is not None:  # lane-marking segmentation (model output)
            m = cv2.resize(i["lane_mask"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            img[m] = (0.45 * img[m] + 0.55 * np.array(CYAN)).astype(np.uint8)
        if i.get("lane_est") is not None and i.get("cam") is not None:
            pts = project_centre(i["lane_est"], i["cam"])
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, MAGENTA, 2, cv2.LINE_AA)
        out = i["aeb"]
        for tr, vw in zip(i["tracks"], i["views"], strict=False):
            x1, y1, x2, y2 = (int(v) for v in tr.box)
            crit = out.target_id == tr.track_id
            col = STATE_COL.get(out.state, GREEN) if crit else GREEN
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            rng = _f(vw.range_m, ".1f", "m")
            radar = " R" if getattr(tr, "radar_hits", 0) else ""
            cv2.putText(img, f"#{tr.track_id} {tr.kind} {rng}{radar}", (x1, max(12, y1 - 4)), FONT, 0.42, col, 1)
        closing = out.closing_mps
        lines = [
            f"Mode {i['mode']}  Detection: {i['det_backend']}  Lane: {i['lane_backend']}  Fallback: {i['fallback']}",
            f"Det latency {_f(i['det_lat_ms'], '.0f', ' ms')}  age {_f(i['det_age_ms'], '.0f', ' ms')}"
            f"   Lane latency {_f(i['lane_lat_ms'], '.0f', ' ms')}  age {_f(i['lane_age_ms'], '.0f', ' ms')}",
            f"Scheduler: {i['sched']}   Dropped frames: {i['dropped']}",
            f"Track {_f(out.target_id, 'd')}  Range {_f(out.range_m, '.1f', ' m')}"
            f"  Closing {_f(closing, '.1f', ' m/s')}  TTC {_f(out.ttc_s, '.2f', ' s')}",
            f"Radar: {i['radar']}",
            f"AEB: {out.state} ({out.reason}) brake {out.brake:.2f}",
            f"LKA: {i['lka'].reason}  steer {i['lka'].steer:+.3f}  offset {_f(i['offset_m'], '+.2f', ' m')}",
            f"Arbiter: {i['arb_mode']}  steer cmd {i['steer']:+.3f}   Speed {i['speed_kmh']:.1f} km/h"
            f"   t={i['t']:.2f} s",
        ]
        cv2.rectangle(img, (0, 0), (640, 10 + 17 * len(lines)), (20, 20, 20), -1)
        for k, t in enumerate(lines):
            col = STATE_COL.get(out.state, WHITE) if t.startswith("AEB") else WHITE
            if t.startswith("LKA") and i["lka"].reason != "tracking":
                col = AMBER
            cv2.putText(img, t, (8, 20 + 17 * k), FONT, 0.43, col, 1, cv2.LINE_AA)
        if out.brake > 0:
            cv2.rectangle(img, (w - 170, h - 46), (w - 8, h - 8), RED, -1)
            cv2.putText(img, "AEB BRAKE", (w - 160, h - 20), FONT, 0.7, WHITE, 2)
        cv2.putText(img, "cyan: lane mask  magenta: est. lane centre", (8, h - 10), FONT, 0.4, WHITE, 1)
        return img

    def add(self, bgr: np.ndarray, info: dict, tag: str = "") -> None:
        img = self.draw(bgr, info)
        if self.writer is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            hh, ww = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.dir / self.name), fourcc, self.fps, (ww, hh))
        self.writer.write(img)
        st = info["aeb"].state
        pre = f"{tag}_" if tag else ""
        if info.get("lane_mask") is not None and f"{pre}lane" not in self.shots and info["t"] > 2.0:
            self.shots[f"{pre}lane"] = img
        if st in ("WARNING", "BRAKE") and f"{pre}{st.lower()}" not in self.shots:
            self.shots[f"{pre}{st.lower()}"] = img
        if "FALLBACK" in info["fallback"] and f"{pre}fallback" not in self.shots:
            self.shots[f"{pre}fallback"] = img

    def close(self) -> list[str]:
        if self.writer is not None:
            self.writer.release()
        for k, im in self.shots.items():
            cv2.imwrite(str(self.dir / f"{k}.png"), im)
        return sorted(self.shots)
