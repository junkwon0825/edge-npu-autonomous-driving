"""Multi-object tracker feeding the V2 motion / risk estimator.

SORT-style, kept deliberately simple (no appearance model, greedy association, no extra dependency):

* prediction: constant-velocity motion of the box corners in pixels (alpha-beta per coordinate),
* association: same class group, greedy on IoU with the predicted box (>= ``iou_gate``), then on
  centre distance normalised by box height (<= ``dist_gate``) for small/fast boxes,
* life cycle: tentative until ``min_hits`` updates, coasted (prediction only) for at most
  ``max_coast_s`` without a detection, then deleted.

Every track carries a ``RangeKalman`` on the flat-ground gap and a ``Kalman1D`` on the lateral
offset, so the risk model gets range, range rate, lateral position and lateral velocity per object.
Boxes touching the bottom image border (range only an upper bound) do not update the range filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from npu_ad.evaluation.metrics import box_iou
from npu_ad.perception.filters import RangeKalman
from npu_ad.perception.mono_range import GroundPlaneRanger


@dataclass
class Kalman1D:
    """Constant-velocity Kalman filter for a position with a per-update measurement std."""

    sigma_a: float = 3.0
    max_dt: float = 0.5
    x: np.ndarray | None = None
    P: np.ndarray | None = None
    t: float | None = None
    n: int = 0

    def update(self, t: float, z: float, std: float) -> None:
        if self.x is None or self.t is None or t - self.t > self.max_dt:
            self.x, self.P, self.t, self.n = np.array([z, 0.0]), np.diag([std * std, 4.0]), t, 1
            return
        dt = max(t - self.t, 1e-3)
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.sigma_a**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        x, p = f @ self.x, f @ self.P @ f.T + q
        s = p[0, 0] + std * std
        k = p[:, 0] / s
        self.x = x + k * (z - x[0])
        self.P = p - np.outer(k, p[0, :])
        self.t, self.n = t, self.n + 1

    def predict(self, t: float) -> float | None:
        return None if self.x is None or self.t is None else float(self.x[0] + self.x[1] * (t - self.t))


@dataclass
class Track:
    track_id: int
    kind: str
    cls: int
    box: np.ndarray
    conf: float
    born_t: float
    last_t: float
    hits: int = 1
    misses: int = 0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))
    rng: RangeKalman = field(default_factory=RangeKalman)
    lat: Kalman1D = field(default_factory=Kalman1D)
    truncated: bool = False
    side_truncated: bool = False  # box touches the left/right image border: lateral estimate biased
    updated: bool = True  # matched a detection in the latest update
    radar_hits: int = 0

    @property
    def age_s(self) -> float:
        return self.last_t - self.born_t

    @property
    def center(self) -> tuple[float, float]:
        return float((self.box[0] + self.box[2]) / 2), float((self.box[1] + self.box[3]) / 2)

    def predicted_box(self, t: float) -> np.ndarray:
        return self.box + self.vel * (t - self.last_t)


class MultiObjectTracker:
    def __init__(
        self,
        ranger: GroundPlaneRanger,
        class_groups: dict[str, set[int]],
        iou_gate: float = 0.2,
        dist_gate: float = 1.0,
        min_hits: int = 3,
        max_coast_s: float = 0.4,
        min_conf: float = 0.25,
        image_w: int | None = None,
        side_margin_px: float = 2.0,
        box_alpha: float = 0.6,
        box_beta: float = 0.2,
    ):
        self.ranger, self.groups = ranger, class_groups
        self.iou_gate, self.dist_gate, self.min_hits = iou_gate, dist_gate, min_hits
        self.max_coast_s, self.min_conf = max_coast_s, min_conf
        self.box_alpha, self.box_beta = box_alpha, box_beta
        self.image_w = image_w or int(round(2 * float(ranger.k[0, 2])))
        self.side_margin_px = side_margin_px
        self.tracks: list[Track] = []
        self._next_id = 1
        self.n_created = 0

    def kind_of(self, cls: int) -> str | None:
        return next((k for k, s in self.groups.items() if cls in s), None)

    def _update_motion(self, tr: Track, t: float, d: np.ndarray) -> None:
        dt = max(t - tr.last_t, 1e-3)
        pred = tr.box + tr.vel * dt
        r = d[:4] - pred
        tr.box = pred + self.box_alpha * r
        tr.vel = tr.vel + self.box_beta / dt * r
        tr.conf = 0.7 * tr.conf + 0.3 * float(d[4])
        tr.cls = int(d[5])
        tr.last_t, tr.hits, tr.misses, tr.updated = t, tr.hits + 1, 0, True

    def _update_range(self, tr: Track, t: float, d: np.ndarray) -> None:
        est = self.ranger.estimate(d)
        if est is None:
            return
        tr.truncated = est.truncated
        if not est.truncated:
            tr.rng.update(t, est.gap_m)
        # a box clipped at a side border shrinks towards the image centre while the object leaves the
        # view: its centre (hence lateral offset) moves inward although the object does not (V2 replay
        # false brake on a parked car in the adjacent lane). Such boxes do not update the lateral filter.
        tr.side_truncated = bool(d[0] <= self.side_margin_px or d[2] >= self.image_w - self.side_margin_px)
        if tr.side_truncated:
            return
        depth = max(est.depth_m, 1.0)
        fx = float(self.ranger.k[0, 0])
        std = math.sqrt(0.1**2 + (2.0 * depth / fx) ** 2 + (0.05 * est.lateral_m) ** 2)
        tr.lat.update(t, est.lateral_m, std)

    def update(self, t: float, dets: np.ndarray) -> list[Track]:
        dets = np.asarray(dets, np.float64).reshape(-1, 6)
        for tr in self.tracks:
            tr.updated = False
        cand = [i for i, d in enumerate(dets) if d[4] >= self.min_conf and self.kind_of(int(d[5])) is not None]
        used_d: set[int] = set()
        used_t: set[int] = set()
        pairs = []
        for ti, tr in enumerate(self.tracks):
            pb = tr.predicted_box(t)
            for di in cand:
                if self.kind_of(int(dets[di, 5])) != tr.kind:
                    continue
                iou = float(box_iou(pb[None], dets[di, :4][None])[0, 0])
                h = max(pb[3] - pb[1], 1.0)
                c1 = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
                c2 = np.array([(dets[di, 0] + dets[di, 2]) / 2, (dets[di, 1] + dets[di, 3]) / 2])
                dist = float(np.linalg.norm(c1 - c2) / h)
                if iou >= self.iou_gate or dist <= self.dist_gate:
                    pairs.append((-iou, dist, ti, di))
        for _, _, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[ti]
            self._update_motion(tr, t, dets[di])
            self._update_range(tr, t, dets[di])
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_coast_s]
        for di in cand:
            if di in used_d:
                continue
            d = dets[di]
            tr = Track(self._next_id, self.kind_of(int(d[5])), int(d[5]), d[:4].copy(), float(d[4]), t, t)
            tr.rng = RangeKalman(fy=float(self.ranger.k[1, 1]), cam_h=self.ranger.cam_height_m)
            self._next_id += 1
            self.n_created += 1
            self._update_range(tr, t, d)
            self.tracks.append(tr)
        return self.confirmed()

    def update_radar(self, t: float, pts, ego_front_m: float, cfg) -> int:
        """Fuse radar points into the confirmed tracks updated by the camera in this cycle (V3)."""
        from npu_ad.perception.filters import kalman_measure
        from npu_ad.perception.radar_fusion import associate

        fx, cx = float(self.ranger.k[0, 0]), float(self.ranger.k[0, 2])
        n = 0
        for tr in self.confirmed():
            cam_rng = tr.rng.range
            a = associate(tr.box, fx, cx, cam_rng, pts, ego_front_m, cfg)
            if a is None:
                continue
            gap, closing, _ = a
            if tr.rng.x is None:
                continue
            ok_r = kalman_measure(tr.rng, t, gap, 0, cfg.range_std_m)
            ok_v = kalman_measure(tr.rng, t, -closing, 1, cfg.rate_std_mps)
            if ok_r or ok_v:
                tr.radar_hits += 1
                n += 1
        return n

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]
