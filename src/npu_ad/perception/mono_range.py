"""Perception-only range estimation for AEB (no simulator ground truth).

``GroundPlaneRanger`` turns a 2D detection box into a metric (gap, lateral) estimate with the
flat-ground pinhole model: the bottom edge of the box is assumed to touch the road plane, so for a
camera at height ``h`` with zero pitch, depth ``Z = f_y * h / (v_bottom - c_y)``. This assumption
breaks for slopes, pitch changes under braking, boxes truncated by the image border and occlusion;
its error is therefore *measured* against CARLA ground truth in every scenario run
(``reports/aeb_perception_eval.md``) instead of being assumed.

``CipoTracker`` selects the closest in-path object (CIPO) per frame and estimates its closing speed
from a least-squares slope of the range history, i.e. only from perception outputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from npu_ad.control.aeb import TargetState


@dataclass(frozen=True)
class RangeEstimate:
    gap_m: float  # from ego front bumper to the box's ground-contact point, along the optical axis
    lateral_m: float  # >0 right of the optical axis
    depth_m: float  # camera depth Z
    truncated: bool  # box touches the bottom image border -> range is an upper bound only
    box: tuple[float, float, float, float]
    cls: int
    conf: float


@dataclass(frozen=True)
class GroundPlaneRanger:
    k: np.ndarray  # 3x3 intrinsics
    image_h: int
    cam_height_m: float  # camera height above the road plane
    cam_x_m: float  # camera position ahead of the ego origin
    ego_half_len_m: float  # ego origin -> front bumper
    min_rows_below_horizon: float = 2.0  # rows; closer to the horizon the range is ill-conditioned

    def estimate(self, det: np.ndarray) -> RangeEstimate | None:
        x1, y1, x2, y2, conf, cls = (float(v) for v in det[:6])
        fx, fy, cx, cy = self.k[0, 0], self.k[1, 1], self.k[0, 2], self.k[1, 2]
        dv = y2 - cy
        if dv < self.min_rows_below_horizon:
            return None
        z = fy * self.cam_height_m / dv
        u = 0.5 * (x1 + x2)
        lat = (u - cx) * z / fx
        gap = z + self.cam_x_m - self.ego_half_len_m
        return RangeEstimate(gap, lat, z, y2 >= self.image_h - 2, (x1, y1, x2, y2), int(cls), conf)


@dataclass
class CipoTracker:
    """Closest in-path object with a range-rate estimate. One track at a time (sufficient for AEB)."""

    ranger: GroundPlaneRanger
    class_groups: dict[str, set[int]]  # e.g. {"vehicle": {2,3,5,7}, "pedestrian": {0}}
    corridor_half_width_m: float = 1.8
    window_s: float = 0.3
    min_samples: int = 5
    gate_m: float = 3.0  # max range jump (beyond expected motion) to keep the same track
    max_range_m: float = 60.0
    max_fit_residual_m: float = 0.5  # noisy range history (pitch, box jitter) -> no closing estimate
    _hist: deque = field(default_factory=deque)
    _track_id: int = 0
    _kind: str | None = None
    _last: RangeEstimate | None = None

    def _kind_of(self, cls: int) -> str | None:
        return next((k for k, s in self.class_groups.items() if cls in s), None)

    def update(self, t_s: float, dets: np.ndarray) -> tuple[TargetState | None, RangeEstimate | None]:
        cands: list[tuple[RangeEstimate, str]] = []
        for d in np.asarray(dets).reshape(-1, 6):
            kind = self._kind_of(int(d[5]))
            if kind is None:
                continue
            r = self.ranger.estimate(d)
            if r is None or r.gap_m > self.max_range_m or r.gap_m < -1.0:
                continue
            if abs(r.lateral_m) <= self.corridor_half_width_m:
                cands.append((r, kind))
        if not cands:
            # keep history (a 1-2 frame miss must not reset the rate estimate) but report nothing
            while self._hist and t_s - self._hist[0][0] > self.window_s * 2:
                self._hist.popleft()
            return None, None
        r, kind = min(cands, key=lambda c: c[0].gap_m)
        if self._hist:
            t_prev, g_prev = self._hist[-1]
            expected_jump = 40.0 * (t_s - t_prev)  # <= 144 km/h relative motion
            if kind != self._kind or abs(r.gap_m - g_prev) > self.gate_m + expected_jump:
                self._hist.clear()
                self._track_id += 1
        self._kind = kind
        self._last = r
        self._hist.append((t_s, r.gap_m))
        while self._hist and t_s - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        closing = 0.0
        if len(self._hist) >= self.min_samples:
            ts = np.array([h[0] for h in self._hist])
            gs = np.array([h[1] for h in self._hist])
            coef = np.polyfit(ts - ts[0], gs, 1)
            resid = gs - np.polyval(coef, ts - ts[0])
            if float(np.std(resid)) <= self.max_fit_residual_m:
                closing = -float(coef[0])
        # actor_id is the perception track id (negative so it cannot collide with CARLA actor ids)
        return TargetState(-1 - self._track_id, kind, r.gap_m, r.lateral_m, closing, True), r
