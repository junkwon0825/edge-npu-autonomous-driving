"""V2 estimation and control: filters, tracker, TTC / path risk, AEB state machine."""

import math

import numpy as np

from npu_ad.control.aeb_v2 import (
    BRAKE,
    CLEAR,
    RELEASE,
    WARNING,
    WATCH,
    AebStateMachine,
    RiskParams,
    TrackView,
    assess,
    path_interval,
)
from npu_ad.perception.filters import AlphaBeta, FiniteDifference, RangeKalman, WindowLsq
from npu_ad.perception.mono_range import GroundPlaneRanger
from npu_ad.perception.tracker import MultiObjectTracker
from npu_ad.simulation import geometry as geo


def _ramp(filt, rate=-8.0, noise=0.0, n=60, dt=1 / 30, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(n):
        filt.update(k * dt, 40.0 + rate * k * dt + noise * rng.standard_normal())
    return filt


def test_filters_track_constant_closing_without_noise():
    for f in (FiniteDifference(), WindowLsq(), AlphaBeta(), RangeKalman()):
        _ramp(f)
        assert f.valid and abs(f.rate + 8.0) < 0.3, type(f).__name__


def test_kalman_beats_finite_difference_under_noise():
    fd = _ramp(FiniteDifference(), noise=0.3, n=90)
    kf = _ramp(RangeKalman(), noise=0.3, n=90)
    assert abs(kf.rate + 8.0) < abs(fd.rate + 8.0) or abs(kf.rate + 8.0) < 1.0
    assert kf.rate_std is not None and kf.rate_std < 3.0


def test_kalman_rejects_gross_outlier():
    kf = _ramp(RangeKalman(), n=40)
    before = kf.rate
    kf.update(40 / 30, 5.0)  # wrong box: 5 m instead of ~29 m
    assert kf.n_rejected == 1 and abs(kf.rate - before) < 0.5


def test_kalman_handles_irregular_dt_and_reinit_after_gap():
    kf = RangeKalman()
    for t in (0.0, 0.03, 0.11, 0.13, 0.3, 0.33):
        kf.update(t, 30.0 - 5.0 * t)
    assert kf.valid
    kf.update(2.0, 20.0)  # > max_dt: restart
    assert kf.n == 1 and kf.range == 20.0


def test_path_interval():
    assert path_interval(0.0, 0.0, 1.5) == (0.0, math.inf)
    assert path_interval(3.0, 0.0, 1.5) is None
    t_in, t_out = path_interval(3.0, -1.5, 1.5)  # walking into the path from the right
    assert abs(t_in - 1.0) < 1e-9 and abs(t_out - 3.0) < 1e-9
    assert path_interval(3.0, 1.0, 1.5) is None  # walking away


def _tv(rng=20.0, rate=-8.0, lat=0.0, vlat=0.0, kind="vehicle", hits=10, age=1.0, conf=0.8, lat_valid=True):
    return TrackView(1, kind, conf, hits, age, rng, rate, True, lat, vlat, lat_valid)


def test_assess_ttc_and_gating():
    p = RiskParams()
    a = assess(_tv(), 8.3, p)
    assert a.gated == "ok" and abs(a.ttc_s - 2.5) < 1e-9 and a.path == "collision_path"
    assert assess(_tv(hits=1), 8.3, p).gated == "tentative"
    assert assess(_tv(conf=0.1), 8.3, p).gated == "low_conf"
    assert assess(_tv(rate=-30.0), 8.3, p).gated == "implausible_closing"  # 30 > 8.3 + 6
    assert assess(_tv(rate=+2.0), 8.3, p).ttc_s == math.inf  # opening: no TTC


def test_assess_target_clear_and_ego_passes():
    p = RiskParams()
    # pedestrian 1.0 m right of centre walking out at 1.6 m/s, TTC 2.5 s: leaves W=1.6 at 0.375 s
    a = assess(_tv(kind="pedestrian", lat=1.0, vlat=1.6), 8.3, p)
    assert a.path == "will_clear"
    # pedestrian 3.5 m away walking in, ego arrives in 0.25 s and exits before it enters
    b = assess(_tv(kind="pedestrian", rng=3.5, rate=-13.9, lat=3.5, vlat=-1.6), 13.9, p)
    assert b.path == "ego_passes"
    # same pedestrian but ego slower: conflict
    c = assess(_tv(kind="pedestrian", rng=12.0, rate=-8.3, lat=3.0, vlat=-1.6), 8.3, p)
    assert c.path == "collision_path"
    # lateral motion not yet known -> uncertain, conservative only inside the path
    d = assess(_tv(lat=0.5, age=0.1), 8.3, p)
    assert d.path == "uncertain" and d.t_in == 0.0


def test_state_machine_escalates_brakes_and_releases():
    sm = AebStateMachine()
    t, out = 0.0, None
    seq = []
    for rng in (30.0, 18.0, 12.0, 8.0):  # closing 8 m/s: TTC 3.75, 2.25, 1.5, 1.0
        out = sm.step(t, [_tv(rng=rng)], 8.0, True)
        seq.append(out.state)
        t += 0.1
    assert seq == [WATCH, WARNING, BRAKE, BRAKE] and out.brake == 1.0
    # target disappears while still moving: hold >= min_brake_s and release_hold_s, then ramp down
    states = []
    for _ in range(15):
        out = sm.step(t, [], 5.0, True)
        states.append((out.state, round(out.brake, 2)))
        t += 0.1
    assert states[0][0] == BRAKE and RELEASE in [s for s, _ in states]
    assert states[-1][0] in (CLEAR, WATCH) and states[-1][1] == 0.0
    brakes = [b for _, b in states]
    assert all(a >= b for a, b in zip(brakes, brakes[1:], strict=False))  # monotone release, no oscillation


def test_state_machine_standstill_hold_until_path_clear():
    sm = AebStateMachine()
    t = 0.0
    sm.step(t, [_tv(rng=3.0, rate=-8.0, kind="pedestrian")], 8.0, True)
    assert sm.state == BRAKE
    for _ in range(20):  # stopped, pedestrian still in front (not closing): keep braking
        t += 0.1
        out = sm.step(t, [_tv(rng=3.0, rate=0.0, kind="pedestrian")], 0.0, True)
    assert out.state == BRAKE and out.reason == "standstill_hold"
    for _ in range(9):  # pedestrian has left the path: wait standstill_clear_s before releasing
        t += 0.1
        out = sm.step(t, [_tv(rng=3.0, rate=0.0, lat=3.0, kind="pedestrian")], 0.0, True)
    assert out.state == BRAKE
    for _ in range(12):
        t += 0.1
        out = sm.step(t, [_tv(rng=3.0, rate=0.0, lat=3.0, kind="pedestrian")], 0.0, True)
    assert out.state in (RELEASE, CLEAR, WATCH)


def test_state_machine_holds_command_without_fresh_perception():
    sm = AebStateMachine()
    sm.step(0.0, [_tv(rng=8.0)], 8.0, True)
    out = sm.step(0.1, [], 8.0, perception_ok=False)
    assert out.degraded and out.state == BRAKE and out.brake == 1.0


def test_will_clear_target_does_not_trigger_braking():
    sm = AebStateMachine()
    for k in range(10):
        out = sm.step(k * 0.05, [_tv(kind="pedestrian", rng=15.0 - k * 0.4, lat=1.0 + 0.08 * k, vlat=1.6)], 8.3, True)
        assert out.brake == 0.0


def _ranger():
    k = geo.intrinsics(800, 450, 90.0)
    return GroundPlaneRanger(k=k, image_h=450, cam_height_m=1.7, cam_x_m=1.5, ego_half_len_m=2.4)


def test_tracker_keeps_id_and_coasts_through_short_gap():
    tr = MultiObjectTracker(_ranger(), {"vehicle": {2}, "pedestrian": {0}})
    ids = set()
    for k in range(20):
        t = k / 30
        y2 = 300 + 2 * k
        dets = np.array([[380 - k, 250, 420 + k, y2, 0.9, 2]]) if k not in (8, 9) else np.zeros((0, 6))
        for x in tr.update(t, dets):
            ids.add(x.track_id)
    assert len(ids) == 1 and tr.n_created == 1
    t0 = tr.confirmed()[0]
    assert t0.rng.valid and t0.rng.rate < 0  # box bottom moving down -> range decreasing


def test_tracker_separates_classes_and_deletes_lost_tracks():
    tr = MultiObjectTracker(_ranger(), {"vehicle": {2}, "pedestrian": {0}}, max_coast_s=0.2)
    dets = np.array([[100, 250, 140, 330, 0.9, 0], [100, 250, 140, 330, 0.9, 2]])
    for k in range(4):
        tr.update(k / 30, dets)
    assert len(tr.confirmed()) == 2
    for k in range(4, 15):
        tr.update(k / 30, np.zeros((0, 6)))
    assert tr.tracks == []


def test_side_truncated_box_does_not_update_lateral():
    tr = MultiObjectTracker(_ranger(), {"vehicle": {2}})
    for k in range(6):
        tr.update(k / 30, np.array([[600, 250, 700, 330, 0.9, 2]]))
    lat_before = tr.confirmed()[0].lat.x[0]
    for k in range(6, 12):  # box now clipped at the right border and shrinking: centre moves inward
        tr.update(k / 30, np.array([[640 + 20 * (k - 6), 250, 800, 330, 0.9, 2]]))
    t0 = tr.confirmed()[0]
    assert t0.side_truncated and abs(t0.lat.x[0] - lat_before) < 0.5
    from npu_ad.control.aeb_v2 import view_of

    assert view_of(t0, 12 / 30).lat_rate_mps == 0.0


def test_standstill_holds_for_static_obstacle_in_path():
    sm = AebStateMachine()
    sm.step(0.0, [_tv(rng=10.0, rate=-8.0)], 8.0, True)
    t = 0.0
    for _ in range(40):  # stopped 16 m before a static car in the lane: must not release and creep on
        t += 0.1
        out = sm.step(t, [_tv(rng=16.0, rate=0.0)], 0.0, True)
    assert out.state == BRAKE and out.reason == "standstill_hold"


def test_partial_brake_stop_is_held_and_no_creeping():
    """Regression (first V2 matrix run): WARNING's partial brake stopped the car 16 m short, then TTC
    rose, WARNING released and the car crept on (4 brake onsets)."""
    sm = AebStateMachine()
    out = sm.step(0.0, [_tv(rng=21.0, rate=-8.3)], 8.3, True)  # TTC 2.53 -> WARNING
    assert out.state == WARNING
    t, onsets, prev = 0.0, 0, out.brake
    for v, rng, rate in [(6.0, 19.0, -6.0), (4.0, 17.5, -4.0), (2.0, 16.8, -2.0), (0.0, 16.5, 0.0)] + [
        (0.0, 16.5, 0.0)
    ] * 30:
        t += 0.1
        out = sm.step(t, [_tv(rng=rng, rate=rate)], v, True)
        onsets += prev == 0 and out.brake > 0
        prev = out.brake
    assert out.brake > 0 and onsets == 0  # still holding, never released in between
