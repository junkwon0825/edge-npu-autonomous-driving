"""V2 runtime: queue, frame-age gate, simulated async pipeline, failover, remote transport client."""

import sys
import time
from pathlib import Path

import numpy as np

from npu_ad.runtime.backends import FuriosaRemoteBackend, InferenceResult, MockBackend
from npu_ad.runtime.fallback import DEGRADED, FALLBACK, PRIMARY, FailoverBackend, HealthMonitor
from npu_ad.runtime.pipeline import (
    AsyncInferenceWorker,
    Frame,
    LatencyModel,
    LatestFrameQueue,
    ResultGate,
    SimulatedAsyncPipeline,
)

FAKE = [sys.executable, str(Path(__file__).with_name("fake_npu_server.py"))]


def _res(fid, t, ok=True):
    return InferenceResult(fid, t, "mock", "none", ok, np.zeros((0, 6)), 0.0)


def test_latest_frame_queue_drops_oldest():
    q = LatestFrameQueue(maxsize=1)
    for i in range(5):
        q.put(Frame(i, i * 0.1, None))
    assert q.n_dropped == 4 and q.depth() == 1
    assert q.get(timeout=0).frame_id == 4


def test_latest_frame_queue_size_two_keeps_two_newest():
    q = LatestFrameQueue(maxsize=2)
    for i in range(4):
        q.put(Frame(i, 0.0, None))
    assert [q.get(0).frame_id, q.get(0).frame_id] == [2, 3]


def test_result_gate_rejects_stale_failed_and_repeat():
    g = ResultGate(max_age_s=0.2)
    assert g.accept(None, 0.0)[1] == "none"
    assert g.accept(_res(1, 1.0), 1.1)[1] == "new"
    assert g.accept(_res(1, 1.0), 1.15)[1] == "repeat"  # same result is not "new perception" twice
    assert g.accept(_res(2, 1.0), 1.5)[1] == "stale"  # 0.5 s old > 0.2 s
    assert g.accept(_res(3, 1.4, ok=False), 1.5)[1] == "failed"
    assert g.stats.n_stale == 1 and g.stats.n_failed == 1 and g.stats.n_new == 1


def test_simulated_pipeline_latency_limits_rate_and_drops_frames():
    # 30 Hz camera, 100 ms latency -> one job per 100 ms, intermediate frames replaced in the queue
    pipe = SimulatedAsyncPipeline(MockBackend(sleep=False), LatencyModel(base_s=0.1), maxsize=1)
    got = []
    for k in range(31):
        t = k / 30
        pipe.on_frame(t, Frame(k, t, None))
        r = pipe.poll(t)
        if r is not None and (not got or r.frame_id != got[-1][0]):
            got.append((r.frame_id, t, r.capture_t))
    assert len(pipe.records) <= 11  # ~1 s / 0.1 s
    assert pipe.n_dropped_frames > 0
    for _fid, t_seen, cap in got:  # results are never visible before capture + latency
        assert t_seen - cap >= 0.1 - 1e-9


def test_simulated_pipeline_zero_latency_returns_same_frame():
    pipe = SimulatedAsyncPipeline(MockBackend(sleep=False), LatencyModel(base_s=0.0))
    pipe.on_frame(0.0, Frame(7, 0.0, None))
    assert pipe.poll(0.0).frame_id == 7


def test_latency_model_drop_and_stale_injection_are_seeded():
    m1 = LatencyModel(base_s=0.02, drop_p=0.3, stale_p=0.2, stale_extra_s=0.5, seed=3)
    m2 = LatencyModel(base_s=0.02, drop_p=0.3, stale_p=0.2, stale_extra_s=0.5, seed=3)
    a = [m1.draw(i * 0.03) for i in range(200)]
    assert a == [m2.draw(i * 0.03) for i in range(200)]
    tags = [t for _, t in a]
    assert 0.2 < tags.count("dropped") / 200 < 0.4 and tags.count("stale_injected") > 10


def test_latency_model_burst():
    m = LatencyModel(base_s=0.02, burst_period_s=1.0, burst_len_s=0.2, burst_extra_s=0.3)
    assert m.draw(0.1) == (0.32, "burst") and m.draw(0.5) == (0.02, "ok")


def test_async_worker_thread_processes_newest_frame():
    q = LatestFrameQueue(1)
    w = AsyncInferenceWorker(MockBackend(latency_s=0.05), q).start()
    try:
        for i in range(10):
            q.put(Frame(i, time.perf_counter(), None, enqueue_t=time.perf_counter()))
            time.sleep(0.01)
        time.sleep(0.15)
        assert w.latest() is not None and w.latest().frame_id == 9
        assert q.n_dropped > 0
        assert all(r["queue_delay_s"] >= 0 for r in w.records)
    finally:
        w.stop()


def test_failover_switches_logs_and_recovers():
    clock = [0.0]
    prim = MockBackend(name="rngd", fail_ids={2, 3, 4}, sleep=False)
    sec = MockBackend(name="gpu", sleep=False)
    fb = FailoverBackend(prim, sec, fail_threshold=1, probe_interval_s=0.5, clock=lambda: clock[0])
    out = []
    for fid in range(1, 8):
        clock[0] = fid * 0.2
        out.append(fb.infer(fid, clock[0], None).backend)
    assert out[0] == "rngd" and out[1] == "gpu"  # frame 2 failed on the NPU and was re-run on the GPU
    assert fb.events[0]["from"] == PRIMARY and fb.events[0]["to"] == FALLBACK
    assert out[-1] == "rngd" and fb.state == PRIMARY  # probe after 0.5 s succeeded
    assert any(e["to"] == PRIMARY for e in fb.events)


def test_failover_degraded_when_both_fail():
    fb = FailoverBackend(MockBackend(fail_ids={1}, sleep=False), MockBackend(fail_ids={1}, sleep=False))
    r = fb.infer(1, 0.0, None)
    assert not r.ok and fb.state == DEGRADED


def test_health_monitor_bounded_sampling():
    clock = [0.0]
    mon = HealthMonitor(MockBackend(sleep=False), remote_every_s=10.0, clock=lambda: clock[0])
    for i in range(50):
        clock[0] = i * 0.1
        mon.record(_res(i, clock[0]), queue_depth=1)
    assert len(mon.samples) == 1  # 5 s of calls -> one sample (bounded)


def test_remote_backend_protocol_and_timeout():
    b = FuriosaRemoteBackend(command=FAKE, timeout_s=0.5, startup_timeout_s=20)
    try:
        img = np.zeros((450, 800, 3), np.uint8)
        r = b.infer(1, 0.0, img)
        assert r.ok and r.dets.shape == (1, 6)
        # letterbox 800x450 -> 640: gain 0.8, pad_y 140 -> y 200 maps to (200-140)/0.8 = 75
        assert abs(r.dets[0, 1] - 75.0) < 1e-6
        b.stall(1.0)
        r2 = b.infer(2, 0.0, img)
        assert not r2.ok and "no response" in r2.error and b.n_timeout == 1
        r_busy = b.infer(3, 0.0, img)  # frame 2 is still being written into the stalled pipe
        assert not r_busy.ok and b.n_busy == 1
        time.sleep(1.5)
        r4 = b.infer(4, 0.0, img)  # the late answer to frame 2 is counted as stale and skipped
        assert r4.ok and r4.frame_id == 4 and b.n_stale == 1
    finally:
        b.close()


def test_encodings_payload_sizes():
    b = FuriosaRemoteBackend(command=FAKE, startup_timeout_s=20)
    try:
        img = np.random.default_rng(0).integers(0, 255, (450, 800, 3), dtype=np.uint8)
        sizes = {}
        for enc in ("letterbox_raw", "content_raw", "content_jpeg", "content_png"):
            b.encoding = enc
            sizes[enc] = len(b.encode(img)[3])
        assert sizes["letterbox_raw"] == 640 * 640 * 3 and sizes["content_raw"] == 640 * 360 * 3
        assert sizes["content_jpeg"] < sizes["content_raw"]
    finally:
        b.close()
