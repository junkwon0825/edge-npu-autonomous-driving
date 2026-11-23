import numpy as np
import pytest

from npu_ad.perception.mono_range import CipoTracker, GroundPlaneRanger
from npu_ad.simulation import geometry as g

K = g.intrinsics(800, 450, 90.0)  # f = 400 px, cy = 225
RANGER = GroundPlaneRanger(k=K, image_h=450, cam_height_m=1.7, cam_x_m=1.5, ego_half_len_m=2.4)
GROUPS = {"vehicle": {2}, "pedestrian": {0}}


def box_at(depth_m: float, lateral_m: float = 0.0, half_w_px: float = 30.0, cls: int = 2) -> np.ndarray:
    v = 225 + 400 * 1.7 / depth_m
    u = 400 + 400 * lateral_m / depth_m
    return np.array([u - half_w_px, v - 50, u + half_w_px, v, 0.9, cls])


def test_ground_plane_inverts_projection():
    r = RANGER.estimate(box_at(20.0, lateral_m=1.0))
    assert r.depth_m == pytest.approx(20.0)
    assert r.lateral_m == pytest.approx(1.0)
    assert r.gap_m == pytest.approx(20.0 + 1.5 - 2.4)
    assert not r.truncated


def test_box_at_or_above_horizon_has_no_range():
    d = np.array([380, 150, 420, 226, 0.9, 2])  # bottom 1 px below the principal row
    assert RANGER.estimate(d) is None


def test_truncated_box_is_flagged():
    d = np.array([300, 300, 500, 449, 0.9, 2])
    assert RANGER.estimate(d).truncated


def test_cipo_picks_closest_in_corridor_and_estimates_closing_speed():
    tr = CipoTracker(RANGER, GROUPS)
    state = None
    for i in range(30):  # target approaches at 8 m/s, sampled at 100 Hz
        t = i * 0.01
        depth = 30.0 - 8.0 * t
        dets = np.stack([box_at(depth), box_at(10.0, lateral_m=4.0), box_at(12.0, cls=9)])  # off-path + other class
        state, rng = tr.update(t, dets)
    assert state.kind == "vehicle"
    assert state.gap_m == pytest.approx(30.0 - 8.0 * 0.29 + 1.5 - 2.4, abs=1e-6)
    assert state.closing_mps == pytest.approx(8.0, abs=0.05)
    assert state.actor_id < 0  # perception track ids never collide with CARLA actor ids


def test_cipo_resets_track_on_range_jump_and_reports_nothing_without_detection():
    tr = CipoTracker(RANGER, GROUPS)
    for i in range(10):
        s1, _ = tr.update(i * 0.01, box_at(30.0)[None])
    s2, _ = tr.update(0.10, box_at(12.0)[None])  # different object
    assert s2.actor_id != s1.actor_id
    assert s2.closing_mps == 0.0  # not enough history on the new track
    assert tr.update(0.11, np.zeros((0, 6))) == (None, None)


def test_noisy_range_history_gives_no_closing_estimate():
    tr = CipoTracker(RANGER, GROUPS)
    rng = np.random.default_rng(0)
    for i in range(30):
        state, _ = tr.update(i * 0.01, box_at(20.0 + rng.normal(0, 1.5))[None])
    assert state.closing_mps == 0.0
