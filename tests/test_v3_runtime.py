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


def test_coalescing_dispatcher_prevents_lane_starvation():
    """53 ms requests, 30 ms frames, lane every 2nd frame: capture-time op selection starves the lane
    model through aliasing; coalescing the wanted models until dispatch serves it."""
    from npu_ad.runtime.multimodel import CoalescingDispatcher
    from npu_ad.runtime.pipeline import Frame, LatencyModel, SimulatedAsyncPipeline

    class FakeView:
        name, precision, method = "rngd", "bf16", "infer_both"

        def infer(self, fid, cap, bgr):
            lane = self.method in ("infer_both", "infer_lane")
            det = self.method in ("infer_both", "infer_det")
            return MultiResult(fid, cap, "rngd", "bf16", True, det, lane, 0.053)

        def health(self):
            return {}

    def run(coalesce: bool) -> int:
        view = FakeView()
        disp = CoalescingDispatcher(view)
        pipe = SimulatedAsyncPipeline(disp if coalesce else view, LatencyModel(use_measured=True))
        sched, lanes = RateScheduler(mode="fixed", lane_every=2), set()
        for step in range(900):  # 9 s at 10 ms
            t = step * 0.01
            if step % 3 == 0:
                d, ln = sched.plan(step // 3 + 1, None, None, None)
                if coalesce:
                    if disp.request(d, ln):
                        pipe.on_frame(t, Frame(step, t, None))
                else:
                    view.method = "infer_both" if (d and ln) else ("infer_det" if d else "infer_lane")
                    pipe.on_frame(t, Frame(step, t, None))
            r = pipe.poll(t)
            if r is not None and r.lane:
                lanes.add(r.frame_id)
        return len(lanes)

    aliased, coalesced = run(False), run(True)
    assert coalesced > 120  # ~every request carries the lane model (9 s / 53 ms = 170 requests)
    assert aliased < 0.7 * coalesced  # measured in this model: 85 vs 149
