"""Lane geometry from a lane-marking mask (V3).

mask (256x512, network input space)
  -> pixel coordinates in the 800x450 camera image
  -> flat-ground projection with the calibrated camera (forward x from the ego origin, lateral y > 0 right)
  -> left / right boundary of the ego lane: lateral peaks at near range (5-15 m) closest to the previous
     lane centre, then sliding windows along x (2 m bins, +-0.6 m around the running boundary estimate)
  -> weighted least squares per boundary: y(x) = a*x^2 + b*x + c  (x in metres ahead of the ego origin)
  -> centre line = mean of both boundaries (one boundary + previous width if only one side is found)
  -> lateral offset of the ego from the centre e = -c_centre (m, > 0 = ego right of centre),
     heading error psi = atan(b_centre) (rad, lane direction relative to ego heading), curvature 2a.

Ground truth (CARLA waypoints) is only used by evaluation code, never here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float  # camera height above the road
    x_m: float  # camera position ahead of the ego origin
    image_w: int = 800
    image_h: int = 450


@dataclass
class LaneEstimate:
    valid: bool
    offset_m: float = 0.0  # ego lateral position relative to the lane centre (> 0: right of centre)
    heading_rad: float = 0.0  # lane direction minus ego heading
    curvature: float = 0.0  # 1/m, d2y/dx2 of the centre line
    width_m: float = 0.0
    confidence: float = 0.0
    n_left: int = 0
    n_right: int = 0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # a, b, c of the centre line
    reason: str = ""

    def lateral_at(self, x: float) -> float:
        a, b, c = self.center
        return a * x * x + b * x + c


@dataclass
class LaneGeometry:
    cam: CameraModel
    mask_w: int = 512
    mask_h: int = 256
    x_min: float = 4.0
    x_max: float = 30.0
    min_points: int = 25
    nominal_width_m: float = 3.5
    win_halfwidth_m: float = 0.6
    _prev_center: float = 0.0
    _prev_width: float = 3.5
    stats: dict = field(default_factory=lambda: {"ok": 0, "one_side": 0, "fail": 0})

    def ground_points(self, mask: np.ndarray) -> np.ndarray:
        """(N, 2) array of (x forward, y lateral) metres for lane pixels below the horizon."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return np.zeros((0, 2))
        u = (xs + 0.5) * self.cam.image_w / self.mask_w
        v = (ys + 0.5) * self.cam.image_h / self.mask_h
        dv = v - self.cam.cy
        keep = dv > 4.0
        u, dv = u[keep], dv[keep]
        z = self.cam.fy * self.cam.height_m / dv  # camera depth
        lat = (u - self.cam.cx) * z / self.cam.fx
        x = z + self.cam.x_m
        sel = (x >= self.x_min) & (x <= self.x_max)
        return np.stack([x[sel], lat[sel]], 1)

    def _track_boundary(self, pts: np.ndarray, y0: float) -> np.ndarray:
        out, y = [], y0
        for x0 in np.arange(self.x_min, self.x_max, 2.0):
            band = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x0 + 2.0)]
            near = band[np.abs(band[:, 1] - y) < self.win_halfwidth_m + 0.02 * x0]
            if len(near) >= 3:
                y = float(np.median(near[:, 1]))
                out.append(near)
        return np.concatenate(out) if out else np.zeros((0, 2))

    @staticmethod
    def _fit(p: np.ndarray) -> np.ndarray:
        w = 1.0 / np.maximum(p[:, 0], 1.0)  # near points are more accurate (range error ~ x^2)
        return np.polyfit(p[:, 0], p[:, 1], 2 if np.ptp(p[:, 0]) > 8 else 1, w=w)

    def estimate(self, mask: np.ndarray) -> LaneEstimate:
        pts = self.ground_points(mask)
        if len(pts) < self.min_points:
            self.stats["fail"] += 1
            return LaneEstimate(False, reason="too few lane pixels")
        near = pts[(pts[:, 0] < 15)]
        if len(near) < 10:
            near = pts
        hist, edges = np.histogram(near[:, 1], bins=np.arange(-8, 8.01, 0.2))
        centres = (edges[:-1] + edges[1:]) / 2
        c0 = self._prev_center
        left_bins = [(h, c) for h, c in zip(hist, centres, strict=True) if c < c0 - 0.5 and h >= 3]
        right_bins = [(h, c) for h, c in zip(hist, centres, strict=True) if c > c0 + 0.5 and h >= 3]
        left = max(left_bins, key=lambda hc: (hc[0] >= 0.3 * max(hist), hc[1]))[1] if left_bins else None
        right = min(right_bins, key=lambda hc: (hc[0] < 0.3 * max(hist), hc[1]))[1] if right_bins else None
        lp = self._track_boundary(pts, left) if left is not None else np.zeros((0, 2))
        rp = self._track_boundary(pts, right) if right is not None else np.zeros((0, 2))
        fl = self._fit(lp) if len(lp) >= self.min_points else None
        fr = self._fit(rp) if len(rp) >= self.min_points else None

        def coef(f):
            f = np.asarray(f)
            return np.pad(f, (3 - len(f), 0))

        if fl is not None and fr is not None:
            cl, cr = coef(fl), coef(fr)
            width = float(cr[2] - cl[2])
            if not 2.4 <= width <= 5.0:  # implausible pair: keep the side closer to the previous boundary
                fl = fl if abs(cl[2] - (c0 - self._prev_width / 2)) < abs(cr[2] - (c0 + self._prev_width / 2)) else None
                fr = None if fl is not None else fr
        if fl is not None and fr is not None:
            centre = (coef(fl) + coef(fr)) / 2
            width = float(coef(fr)[2] - coef(fl)[2])
            self._prev_width = 0.8 * self._prev_width + 0.2 * width
            self.stats["ok"] += 1
            conf = min(1.0, (len(lp) + len(rp)) / 400)
        elif fl is not None or fr is not None:
            c = coef(fl if fl is not None else fr)
            sign = 1.0 if fl is not None else -1.0
            centre = c + np.array([0.0, 0.0, sign * self._prev_width / 2])
            width = self._prev_width
            self.stats["one_side"] += 1
            conf = min(0.6, len(lp if fl is not None else rp) / 400)
        else:
            self.stats["fail"] += 1
            return LaneEstimate(False, reason="no boundary fit")
        self._prev_center = float(centre[2])
        return LaneEstimate(True, offset_m=-float(centre[2]), heading_rad=math.atan(float(centre[1])),
                            curvature=2 * float(centre[0]), width_m=width, confidence=conf, n_left=len(lp),
                            n_right=len(rp), center=(float(centre[0]), float(centre[1]), float(centre[2])))  # fmt: skip
