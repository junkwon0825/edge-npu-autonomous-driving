"""Ground-truth geometry helpers (pure numpy; CARLA objects are converted by the caller).

CARLA/UE4 world frame: x forward, y right, z up (left-handed), yaw in degrees.
"""

from __future__ import annotations

import math

import numpy as np


def to_ego_frame(px: float, py: float, ex: float, ey: float, yaw_deg: float) -> tuple[float, float]:
    """World point -> (longitudinal, lateral) in the ego frame (lateral > 0 = right)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - ex, py - ey
    return dx * c + dy * s, -dx * s + dy * c


def half_length_along(ext_x: float, ext_y: float, rel_yaw_deg: float) -> float:
    """Half-extent of a box projected on a direction rotated by ``rel_yaw_deg`` from its x-axis."""
    r = math.radians(rel_yaw_deg)
    return abs(ext_x * math.cos(r)) + abs(ext_y * math.sin(r))


def longitudinal_gap(long_m: float, ego_half_len: float, target_half_len: float) -> float:
    return long_m - ego_half_len - target_half_len


def closing_speed(ego_v: tuple[float, float], tgt_v: tuple[float, float], yaw_deg: float) -> float:
    """Rate at which the longitudinal gap shrinks (m/s), from world-frame xy velocities."""
    fx, fy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (ego_v[0] - tgt_v[0]) * fx + (ego_v[1] - tgt_v[1]) * fy


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def project(world_xyz: np.ndarray, world_to_camera: np.ndarray, k: np.ndarray) -> np.ndarray | None:
    """Project a world point with a 4x4 world->camera(UE) matrix; None if behind the camera."""
    p = world_to_camera @ np.array([*world_xyz, 1.0])
    x_cam, y_cam, z_cam = p[1], -p[2], p[0]  # UE (fwd, right, up) -> camera (right, down, fwd)
    if z_cam <= 0.1:
        return None
    uvw = k @ np.array([x_cam, y_cam, z_cam])
    return uvw[:2] / uvw[2]


def point_in_any_box(uv: np.ndarray, dets: np.ndarray, classes: set[int], pad_px: float = 4.0) -> bool:
    for x1, y1, x2, y2, _conf, cls in np.asarray(dets).reshape(-1, 6):
        if int(cls) in classes and x1 - pad_px <= uv[0] <= x2 + pad_px and y1 - pad_px <= uv[1] <= y2 + pad_px:
            return True
    return False
