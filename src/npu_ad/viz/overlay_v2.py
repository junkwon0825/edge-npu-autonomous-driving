"""V2 runtime overlay: makes backend, frame age, queueing, tracking and AEB state visible.

Boxes are the confirmed tracks of the newest *accepted* result drawn on the *current* frame, so the
visible lag between box and object is the real frame age. Unavailable values are printed as n/a.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, AMBER, RED, GREY = (255, 255, 255), (80, 200, 80), (0, 190, 255), (60, 60, 230), (170, 170, 170)
STATE_COL = {"CLEAR": GREEN, "WATCH": GREEN, "WARNING": AMBER, "BRAKE": RED, "RELEASE": AMBER}


def _f(v, spec: str, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:{spec}}{unit}"


class OverlayV2:
    def __init__(self, out_dir: Path, fps: float):
        self.dir, self.fps = out_dir, fps
        self.writer = None
        self.shots: dict[str, np.ndarray] = {}

    def draw(self, bgr: np.ndarray, info: dict) -> np.ndarray:
        img = bgr.copy()
        out = info["aeb"]
        for tr in info["tracks"]:
            x1, y1, x2, y2 = (int(v) for v in tr.box)
            crit = out.target_id == tr.track_id
            col = STATE_COL.get(out.state, GREEN) if crit else GREEN
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"#{tr.track_id} {tr.kind} {tr.conf:.2f}", (x1, max(12, y1 - 4)), FONT, 0.42, col, 1)
        res = info["result"]
        lines = [
            f"Backend: {info['backend']}   Fallback: {info['fallback']}",
            f"Frame ID: {info['frame_id']}   Result frame: {res.frame_id if res else 'n/a'}",
            f"Frame age: {_f(info['age_ms'], '.0f', ' ms')}"
            f"   Inference: {_f(res.latency_s * 1e3 if res else None, '.1f', ' ms')}",
            f"Queue depth: {info['queue']}   Dropped frames: {info['dropped']}   Stale rejected: {info['stale']}",
            f"Track ID: {_f(out.target_id, 'd')}   Range: {_f(out.range_m, '.1f', ' m')}"
            f"   Range rate: {_f(-out.closing_mps if out.closing_mps is not None else None, '.1f', ' m/s')}",
            f"TTC: {_f(out.ttc_s, '.2f', ' s')}   Risk/path: {out.path or 'n/a'}",
            f"AEB: {out.state} ({out.reason}) brake {out.brake:.2f}{'  DEGRADED' if out.degraded else ''}",
            f"Vehicle speed: {info['speed_kmh']:.1f} km/h   t={info['t']:.2f} s",
        ]
        cv2.rectangle(img, (0, 0), (560, 10 + 17 * len(lines)), (20, 20, 20), -1)
        for k, t in enumerate(lines):
            col = STATE_COL.get(out.state, WHITE) if t.startswith("AEB") else WHITE
            cv2.putText(img, t, (8, 20 + 17 * k), FONT, 0.43, col, 1, cv2.LINE_AA)
        if out.brake > 0:
            h, w = img.shape[:2]
            cv2.rectangle(img, (w - 170, h - 46), (w - 8, h - 8), RED, -1)
            cv2.putText(img, "AEB BRAKE", (w - 160, h - 20), FONT, 0.7, WHITE, 2)
        return img

    def add(self, bgr: np.ndarray, info: dict) -> None:
        img = self.draw(bgr, info)
        if self.writer is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.dir / "demo_v2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(img)
        st = info["aeb"].state
        if info["tracks"] and "tracking" not in self.shots:
            self.shots["tracking"] = img
        if st in ("WARNING", "BRAKE") and st.lower() not in self.shots:
            self.shots[st.lower()] = img
        if info["fallback"] not in ("n/a", "PRIMARY") and "fallback" not in self.shots:
            self.shots["fallback"] = img
        self.shots["final"] = img

    def close(self) -> list[str]:
        if self.writer is not None:
            self.writer.release()
        for k, im in self.shots.items():
            cv2.imwrite(str(self.dir / f"{k}.png"), im)
        return sorted(self.shots)
