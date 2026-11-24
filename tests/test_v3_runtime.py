"""V3 multi-model runtime: scheduler policies, Mode D result split, lane backends, multi-model protocol."""

import sys
from pathlib import Path

import numpy as np

from npu_ad.perception.lane_net import LaneUNet
from npu_ad.runtime.multimodel import GpuLaneBackend, LaneResult, MultiResult, RateScheduler, RngdMultiModelClient

FAKE = [sys.executable, str(Path(__file__).with_name("fake_npu_server.py"))]


def test_fixed_scheduler_rate_decoupling():
    s = RateScheduler(mode="fixed", det_every=1, lane_every=2)
    plans = [s.plan(n, None, None, None) for n in range(10)]
    assert sum(d for d, _ in plans) == 10 and sum(ln for _, ln in plans) == 5


def test_adaptive_scheduler_policies():
    s = RateScheduler(mode="adaptive")
    stable = [s.plan(n, None, 0.9, 0.1) for n in range(8)]
    assert sum(d for d, _ in stable) == 4 and sum(ln for _, ln in stable) == 2  # cruise: det 1/2, lane 1/4
    hazard = [s.plan(n, 1.5, 0.9, 0.1) for n in range(8)]
    assert sum(d for d, _ in hazard) == 8  # hazard: detection every frame
    uncertain = [s.plan(n, None, 0.1, 0.0) for n in range(8)]
    assert sum(ln for _, ln in uncertain) == 8  # low lane confidence: lane every frame


def test_mode_d_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from v3_run_scenarios import split_result

    lane = LaneResult(1, 0.0, "rngd", "bf16", True, np.zeros((2, 2), bool), 0.05)
    det = type("D", (), {"dets": np.zeros((0, 6))})()
    both = MultiResult(1, 0.0, "rngd", "bf16", True, det, lane, 0.05)
    assert split_result(both) == (det, lane)
    assert split_result(lane) == (None, lane)
    assert split_result(None) == (None, None)


def test_gpu_lane_backend_on_cpu():
    b = GpuLaneBackend(LaneUNet(base=8, levels=3), device="cpu")
    r = b.infer(3, 0.1, np.zeros((450, 800, 3), np.uint8))
    assert r.ok and r.mask.shape == (256, 512) and r.mask.dtype == bool and r.frame_id == 3


def test_multimodel_client_protocol_det_lane_both():
    c = RngdMultiModelClient(command=FAKE, timeout_s=2.0, startup_timeout_s=20)
    try:
        img = np.zeros((450, 800, 3), np.uint8)
        d = c.infer_det(1, 0.0, img)
        ln = c.infer_lane(2, 0.0, img)
        b = c.infer_both(3, 0.0, img)
        assert d.ok and d.dets.shape == (1, 6)
        assert ln.ok and ln.mask.shape == (256, 512) and ln.mask[200, 201]
        assert b.ok and b.det.dets.shape == (1, 6) and b.lane.mask.sum() == ln.mask.sum()
        c.crash()
        r = c.infer_det(4, 0.0, img)  # worker gone: explicit failure, never a hang
        assert not r.ok
    finally:
        c.close()
