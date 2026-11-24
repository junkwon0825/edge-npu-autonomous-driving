"""Camera + radar association and fusion (V3 sensor fusion, stage 1).

CARLA ``sensor.other.radar`` detections (depth, azimuth, altitude, radial velocity) are associated with
camera tracks by azimuth (the track's box columns, +- margin) and gated by range against the camera
range estimate. The associated radar range and closing speed are fused into the track's Kalman filter as
additional measurements (sequential scalar updates), so the controller keeps one state per object.

Radar is mounted at the front bumper; its range is converted to the bumper-to-target gap as
depth * cos(azimuth) - (ego_front - radar_x). Ground truth is not used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RadarConfig:
    x_m: float = 2.4  # mount position ahead of the ego origin
    z_m: float = 1.0
    hfov_deg: float = 30.0
    vfov_deg: float = 10.0
    range_m: float = 80.0
    points_per_second: int = 3000
    az_margin_rad: float = math.radians(1.5)
    range_gate_frac: float = 0.35
    range_gate_min_m: float = 3.0
    range_std_m: float = 0.25
    rate_std_mps: float = 0.3


def radar_points(meas) -> np.ndarray:
    """CARLA RadarMeasurement -> (N, 4) [depth, azimuth, altitude, velocity] (velocity < 0: approaching)."""
    a = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)  # velocity, azimuth, altitude, depth
    return np.stack([a[:, 3], a[:, 1], a[:, 2], a[:, 0]], 1).astype(np.float64)


def associate(box, cam_fx: float, cam_cx: float, cam_range_m: float | None, pts: np.ndarray, ego_front_m: float,
              cfg: RadarConfig) -> tuple[float, float, int] | None:  # fmt: skip
    """Radar gap (m), closing speed (m/s, > 0 approaching) and point count for one camera box, or None."""
    if pts.size == 0:
        return None
    az_l = math.atan((box[0] - cam_cx) / cam_fx) - cfg.az_margin_rad
    az_r = math.atan((box[2] - cam_cx) / cam_fx) + cfg.az_margin_rad
    sel = pts[(pts[:, 1] >= az_l) & (pts[:, 1] <= az_r)]
    if sel.size == 0:
        return None
    gap = sel[:, 0] * np.cos(sel[:, 1]) - (ego_front_m - cfg.x_m)
    if cam_range_m is not None:
        tol = max(cfg.range_gate_min_m, cfg.range_gate_frac * cam_range_m)
        keep = np.abs(gap - cam_range_m) <= tol
        sel, gap = sel[keep], gap[keep]
    if sel.size == 0:
        return None
    near = gap <= np.percentile(gap, 30) + 0.5  # nearest surface cluster of the object
    closing = -sel[near, 3] / np.maximum(np.cos(sel[near, 1]), 0.5)
    return float(np.median(gap[near])), float(np.median(closing)), int(near.sum())
