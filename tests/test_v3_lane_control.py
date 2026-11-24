"""V3: lane geometry, LKA, arbitration, radar association / fusion, scene state, NPU rewrites."""

import math

import numpy as np
import torch

from npu_ad.control.lka import Arbiter, LaneKeeping, LkaOutput
from npu_ad.perception.filters import RangeKalman, kalman_measure
from npu_ad.perception.lane_geometry import CameraModel, LaneEstimate, LaneGeometry
from npu_ad.perception.lane_net import LaneUNet, export_for_npu
from npu_ad.perception.radar_fusion import RadarConfig, associate
from npu_ad.scene import SceneState

CAM = CameraModel(fx=400.0, fy=400.0, cx=400.0, cy=225.0, height_m=1.7, x_m=1.5)


def synth_mask(offset=0.0, heading=0.0, curv=0.0, width=3.5, mw=512, mh=256):
    """Render the two lane boundaries of a lane whose centre is y_c(x) = curv/2 x^2 + tan(h) x - offset."""
    m = np.zeros((mh, mw), np.uint8)
    for side in (-1, 1):
        for x in np.arange(4.0, 40.0, 0.02):
            y = curv / 2 * x * x + math.tan(heading) * x - offset + side * width / 2
            z = x - CAM.x_m
            u, v = CAM.cx + CAM.fx * y / z, CAM.cy + CAM.fy * CAM.height_m / z
            px, py = int(u * mw / 800), int(v * mh / 450)
            for dx in (-1, 0, 1):
                if 0 <= px + dx < mw and 0 <= py < mh:
                    m[py, px + dx] = 255
    return m


def test_lane_geometry_recovers_offset_heading_width():
    g = LaneGeometry(CAM)
    for off, hd in [(0.0, 0.0), (0.4, 0.0), (-0.5, 0.03), (0.2, -0.04)]:
        g._prev_center = 0.0
        est = g.estimate(synth_mask(off, hd))
        assert est.valid, est.reason
        assert abs(est.offset_m - off) < 0.15, (off, est.offset_m)
        assert abs(est.heading_rad - hd) < 0.02, (hd, est.heading_rad)
        assert abs(est.width_m - 3.5) < 0.3


def test_lane_geometry_one_side_and_failure():
    g = LaneGeometry(CAM)
    m = synth_mask(0.3)
    m[:, 256:] = 0  # right boundary missing (only the left marking is visible)
    est = g.estimate(m)
    assert est.valid and est.n_right == 0 and abs(est.offset_m - 0.3) < 0.25
    assert not g.estimate(np.zeros((256, 512), np.uint8)).valid


def test_lka_pure_pursuit_sign_rate_limit_and_timeout():
    lka = LaneKeeping()
    lane_right = LaneEstimate(True, offset_m=-1.0, center=(0.0, 0.0, 1.0), confidence=1.0)  # centre 1 m to the right
    out = lka.step(0.0, lane_right, 0.0, 8.0, 0.0, 0.0)
    assert out.active and out.steer > 0  # steer right towards the centre
    lka2 = LaneKeeping()
    o1 = lka2.step(0.0, lane_right, 0.0, 8.0, 0.0, 0.0)
    o2 = lka2.step(0.01, lane_right, 0.0, 8.0, 0.0, 0.0)
    assert abs(o2.steer - o1.steer) <= 1.5 * 0.01 + 1e-9  # rate limit
    stale = lka2.step(1.0, lane_right, 0.0, 8.0, 0.0, 0.0)
    assert not stale.active and stale.reason == "lane_unavailable"


def test_lka_compensates_heading_change_since_capture():
    lka = LaneKeeping()
    lane = LaneEstimate(True, center=(0.0, 0.0, 0.0), confidence=1.0)  # straight lane, ego centred at capture
    out = lka.step(0.0, lane, 0.0, 8.0, yaw_now=0.05, yaw_at_capture=0.0)  # ego turned right since capture
    assert out.target_lateral_m < 0 and out.steer < 0  # steer back left


def test_arbitration_precedence():
    arb = Arbiter()
    lka = LkaOutput(0.4, True, "tracking")
    c = arb.combine(0.0, 1.0, lka, 0.5)
    assert c.mode == "AEB_PRIORITY" and c.throttle == 0 and c.brake == 1.0 and abs(c.steer) <= 0.1
    c = arb.combine(0.1, 0.0, lka, 0.5)
    assert c.mode == "NORMAL" and c.steer == 0.4 and c.throttle == 0.5
    c = arb.combine(0.2, 0.0, LkaOutput(0.4, False, "lane_unavailable"), 0.5)
    assert c.mode == "LKA_DEGRADED" and c.throttle == 0 and c.steer < 0.4  # decaying, coasting


def test_radar_association_and_kalman_fusion():
    cfg = RadarConfig()
    box = (380.0, 200.0, 420.0, 260.0)  # centred object
    pts = np.array([[20.0, 0.01, 0.0, -8.0], [20.3, -0.02, 0.01, -8.1], [45.0, 0.0, 0.0, 0.0], [20.0, 0.4, 0.0, -8.0]])
    a = associate(box, 400.0, 400.0, 19.0, pts, 2.4, cfg)
    assert a is not None and abs(a[0] - 20.0) < 0.5 and abs(a[1] - 8.0) < 0.3 and a[2] == 2
    assert associate(box, 400.0, 400.0, 19.0, pts[2:3], 2.4, cfg) is None  # out of range gate
    kf = RangeKalman()
    for k in range(5):
        kf.update(k * 0.03, 30.0 - 0.03 * k * 2.0)  # camera says ~2 m/s
    assert kalman_measure(kf, 0.15, -8.0, 1, 0.3)  # radar says 8 m/s closing
    assert kf.rate < -5.0


def test_scene_state_health():
    s = SceneState(t=1.0, ego_speed_mps=8.0, ego_yaw_rad=0.0, ego_yaw_rate_rps=0.0)
    assert not s.perception_ok
    s.detection.age_s = 0.1
    s.lane_capture_t = 0.8
    assert s.perception_ok and abs(s.lane_age() - 0.2) < 1e-9


def test_lane_unet_export_is_equivalent():
    torch.manual_seed(0)
    m = LaneUNet(base=8, levels=3).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
    x = torch.rand(1, 3, 64, 128)
    e = export_for_npu(m)
    assert not any(isinstance(s, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)) for s in e.modules())
    with torch.no_grad():
        assert (e(x) - m(x)).abs().max().item() < 1e-4


def test_path_relative_keeps_in_lane_object_in_corridor_on_curve():
    from npu_ad.control.aeb_v2 import RiskParams, TrackView, assess, path_relative

    R, x = 147.0, 45.0
    a = 1 / (2 * R)  # lane centre y = x^2 / (2R) (curving right)
    tv = TrackView(1, "vehicle", 0.9, 10, 1.0, x, -8.0, True, a * x * x, 0.0, True)  # static car in lane
    assert assess(tv, 8.0, RiskParams()).path == "not_in_path"  # straight corridor misses it
    assert assess(path_relative(tv, (a, 0.0)), 8.0, RiskParams()).path == "collision_path"
