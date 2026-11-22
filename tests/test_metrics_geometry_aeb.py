import math

import numpy as np
import pytest

from npu_ad.control.aeb import AebController, AebParams, Decision, DelayLine, TargetState
from npu_ad.evaluation.metrics import aggregate_agreement, box_iou, latency_stats, match_detections
from npu_ad.simulation import geometry as g


def test_latency_stats_ms_and_fps():
    s = latency_stats([0.010, 0.020, 0.030])
    assert s["n"] == 3
    assert s["mean_ms"] == pytest.approx(20.0)
    assert s["p50_ms"] == pytest.approx(20.0)
    assert s["fps_from_mean"] == pytest.approx(50.0)
    assert latency_stats([]) == {"n": 0}


def test_box_iou_basic():
    a = np.array([[0, 0, 10, 10]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou(a, np.array([[5, 0, 15, 10]]))[0, 0] == pytest.approx(50 / 150)
    assert box_iou(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_requires_same_class_and_iou():
    ref = np.array([[0, 0, 10, 10, 0.9, 2], [20, 20, 30, 30, 0.8, 0]])
    test = np.array([[0, 0, 10, 11, 0.85, 2], [20, 20, 30, 30, 0.8, 2]])  # 2nd: wrong class
    m = match_detections(ref, test)
    assert m["matched"] == 1
    agg = aggregate_agreement([m])
    assert agg["recall_vs_ref"] == pytest.approx(0.5)
    assert agg["precision_vs_ref"] == pytest.approx(0.5)
    assert agg["mean_abs_conf_delta"] == pytest.approx(0.05)


def test_ego_frame_and_closing_speed():
    lon, lat = g.to_ego_frame(10, 0, 0, 0, 0)
    assert (lon, lat) == pytest.approx((10, 0))
    lon, lat = g.to_ego_frame(0, 10, 0, 0, 90)  # facing +y
    assert lon == pytest.approx(10) and lat == pytest.approx(0, abs=1e-9)
    lon, lat = g.to_ego_frame(10, 2, 0, 0, 0)
    assert lat == pytest.approx(2)  # +y is to the right in CARLA
    assert g.closing_speed((10, 0), (4, 0), 0) == pytest.approx(6)
    assert g.half_length_along(2.0, 1.0, 90) == pytest.approx(1.0)


def test_projection_center_and_behind():
    k = g.intrinsics(800, 450, 90)
    assert k[0, 0] == pytest.approx(400)
    eye = np.eye(4)  # camera at origin looking along +x
    uv = g.project(np.array([10.0, 0.0, 0.0]), eye, k)
    assert uv == pytest.approx([400, 225])
    assert g.project(np.array([-5.0, 0.0, 0.0]), eye, k) is None
    right = g.project(np.array([10.0, 2.0, 0.0]), eye, k)
    assert right[0] > 400
    dets = np.array([[390, 215, 410, 235, 0.9, 2]])
    assert g.point_in_any_box(uv, dets, {2})
    assert not g.point_in_any_box(uv, dets, {0})


def _t(gap, closing, perceived=True, lat=0.0, aid=1):
    return TargetState(aid, "vehicle", gap, lat, closing, perceived)


def test_aeb_full_brake_on_low_ttc_and_latches():
    c = AebController()
    d = c.decide([_t(10.0, 8.0)], 8.0)  # ttc 1.25 s
    assert d.brake == 1.0 and d.reason == "full" and d.ttc_s == pytest.approx(1.25)
    assert c.decide([], 5.0).reason == "latched"
    assert c.decide([], 0.0).reason == "latched_stopped"


def test_aeb_partial_and_clear():
    c = AebController()
    assert c.decide([_t(16.0, 8.0)], 8.0).reason == "partial"  # ttc 2.0
    c = AebController()
    assert c.decide([_t(40.0, 8.0)], 8.0).brake == 0.0


def test_aeb_ignores_unperceived_out_of_corridor_and_opening():
    c = AebController()
    assert c.decide([_t(5.0, 8.0, perceived=False)], 8.0).brake == 0.0
    assert c.decide([_t(5.0, 8.0, lat=3.0)], 8.0).brake == 0.0
    assert c.decide([_t(5.0, -1.0)], 8.0).brake == 0.0


def test_aeb_perception_hold_bridges_short_miss():
    c = AebController(AebParams(perception_hold_ticks=2))
    c.decide([_t(40.0, 8.0, perceived=True)], 8.0)
    d = c.decide([_t(9.0, 8.0, perceived=False)], 8.0)
    assert d.reason == "full"


def test_aeb_hold_expires_for_targets_absent_from_input():
    # review finding (Kiro): a target missing from the list must not keep a stale perception hold
    c = AebController(AebParams(perception_hold_ticks=2))
    c.decide([_t(40.0, 8.0, perceived=True)], 8.0)
    c.decide([], 8.0)
    c.decide([], 8.0)
    assert c.decide([_t(9.0, 8.0, perceived=False)], 8.0).brake == 0.0


def test_aeb_v2_holds_partial_until_not_closing_and_holds_standstill():
    c = AebController()
    assert c.decide([_t(16.0, 8.0)], 8.0).reason == "partial"
    d = c.decide([_t(15.0, 3.0)], 3.0)  # ttc 5 s: v1 would release here
    assert d.reason == "partial_hold" and d.brake == 0.5
    assert c.decide([_t(14.9, 0.3)], 0.3).reason == "standstill_hold"
    assert c.decide([], 0.0).reason == "latched_stopped"


def test_aeb_v1_releases_partial_when_ttc_recovers():
    c = AebController(AebParams(hysteresis=False))
    assert c.decide([_t(16.0, 8.0)], 8.0).reason == "partial"
    assert c.decide([_t(15.0, 3.0)], 3.0).reason == "clear"


def test_aeb_v2_releases_hold_when_target_leaves_corridor():
    c = AebController()
    c.decide([_t(16.0, 8.0)], 8.0)
    assert c.decide([_t(15.0, 3.0, lat=3.0)], 3.0).reason == "clear"


def test_delay_line():
    dl = DelayLine(2)
    out = [dl.push(Decision(float(i), str(i))).brake for i in range(1, 5)]
    assert out == [0.0, 0.0, 1.0, 2.0]
    assert DelayLine(0).push(Decision(1.0, "x")).brake == 1.0
    assert not math.isnan(out[-1])
