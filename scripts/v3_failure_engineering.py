# ruff: noqa: E501  (long result dict literals kept on one line)
"""V3: bounded NPU failure engineering (no CARLA; a real 30 Hz frame stream of recorded camera frames).

Each case drives ``FailoverBackend(RNGD multi-model detection, local GPU detection)`` and records whether the
fault was detected, how long failover took (fault -> first successful fallback result), how many frames were
lost, whether every switch was logged, and how the system recovered. Cases:
  timeout        server stalls 1.5 s (requests time out at 0.25 s)
  invalid_shape  one malformed request (payload size does not match w x h) -> error answer, server keeps serving
  stale_answer   a late answer to a timed-out frame arrives later -> counted as stale, never used
  worker_crash   server process exits without answering -> connection closed -> GPU; reconnect after 2 s
  load_failure   server started with missing lane weights -> start-up error -> explicit GPU-only configuration
  npu_unavailable server started on a non-existent device (furiosa:9) -> start-up error -> GPU-only
  both_down      NPU and GPU fail -> DEGRADED (controller holds its last command, see test_v2_control)
Nothing is damaged: faults are a sleep, a malformed request, a process exit, and invalid start arguments.
Output: reports/evidence/v3_failure_engineering.json
"""

from __future__ import annotations

import glob
import json
import time

import cv2

from npu_ad.config import load_config, resolve
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.runtime.backends import OP_INFER, GpuInferenceBackend, MockBackend
from npu_ad.runtime.fallback import FailoverBackend
from npu_ad.runtime.multimodel import RngdMultiModelClient

PERIOD = 1 / 30


def stream(fb, frames, secs, fault_at=None, fault=None):
    """Run a paced stream; returns per-frame records (t, backend, ok)."""
    recs, t0, k, fired = [], time.perf_counter(), 0, False
    while (now := time.perf_counter() - t0) < secs:
        if fault and not fired and now >= fault_at:
            fault()
            fired = True
            t_fault = now
        r = fb.infer(k, now, frames[k % len(frames)])
        recs.append({"t": time.perf_counter() - t0, "frame": k, "backend": r.backend, "ok": r.ok, "lat": r.latency_s})
        k += 1
        sl = t0 + k * PERIOD - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
    return recs, (t_fault if fired else None)


def summarize(recs, t_fault, fb) -> dict:
    after = [r for r in recs if t_fault is not None and r["t"] >= t_fault]
    first_fb = next((r for r in after if r["ok"] and r["backend"] == "gpu"), None)
    lost = sum(1 for r in after[: after.index(first_fb)] if not r["ok"]) if first_fb in after else None
    back = next((r for r in after if r["ok"] and r["backend"] == "rngd"), None) if first_fb else None
    return {"frames": len(recs), "failed_frames": sum(not r["ok"] for r in recs),
            "gpu_frames": sum(r["backend"] == "gpu" for r in recs),
            "failover_s": (first_fb["t"] - t_fault) if first_fb else None, "frames_lost_before_failover": lost,
            "recovered_to_npu_s": (back["t"] - t_fault) if back else None,
            "events": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in fb.events]}  # fmt: skip


def main() -> int:
    cfg = load_config()
    m = cfg["model"]
    gpu = GpuInferenceBackend(GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"]))
    frames = [cv2.imread(p) for p in sorted(glob.glob(str(resolve("outputs/dataset/*/*.png"))))]
    res: dict = {"period_s": PERIOD, "cases": {}}
    log = resolve("outputs/v3/failure_server.log")

    def client(**kw):
        return RngdMultiModelClient(timeout_s=0.25, startup_timeout_s=600, log_path=log, **kw)

    # timeout
    c = client()
    fb = FailoverBackend(c.detection(), gpu, probe_interval_s=0.5)
    recs, tf = stream(fb, frames, 6.0, 2.0, lambda: c.stall(1.5))
    res["cases"]["timeout"] = summarize(recs, tf, fb) | {"client": c.health()}
    # invalid shape: malformed request straight on the transport, then normal frames
    enc, w, h, payload, lb, _ = c.encode(frames[0])
    with c._lock:
        c._send(OP_INFER, 4242, enc, w + 8, h, payload)  # size mismatch
        _, status, body, _ = c._wait(4242, 5.0)
    ok_after = c.infer_det(4243, 0.0, frames[1]).ok
    res["cases"]["invalid_shape"] = {"status": status, "error": body.get("error"), "server_alive_after": ok_after}
    # stale answer: timeout then the late answer must be dropped
    c.stall(0.6)
    r1 = c.infer_det(5000, 0.0, frames[2])  # times out
    time.sleep(0.8)
    r2 = c.infer_det(5001, 0.0, frames[3])
    res["cases"]["stale_answer"] = {"first_ok": r1.ok, "first_error": r1.error, "next_ok": r2.ok,
                                    "next_frame_id": r2.frame_id, "n_stale": c.n_stale}  # fmt: skip
    # worker crash with reconnect
    holder = {"c": c}
    fb = FailoverBackend(c.detection(), gpu, probe_interval_s=0.5)
    t_rec = {}

    def crash():
        holder["c"].crash()

    recs, t0w = [], time.perf_counter()
    crashed = reconnected = False
    k = 0
    while (now := time.perf_counter() - t0w) < 12.0:
        if not crashed and now >= 2.0:
            crash()
            crashed, t_crash = True, now
        if crashed and not reconnected and now >= t_crash + 2.0:
            t_rec["start"] = now
            nc = client()  # new session: compile cache hit
            holder["c"] = nc
            fb.primary = nc.detection()
            reconnected = True
            t_rec["ready"] = time.perf_counter() - t0w
        r = fb.infer(k, now, frames[k % len(frames)])
        recs.append({"t": time.perf_counter() - t0w, "frame": k, "backend": r.backend, "ok": r.ok, "lat": r.latency_s})
        k += 1
        sl = t0w + k * PERIOD - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
    res["cases"]["worker_crash"] = summarize(recs, t_crash, fb) | {"reconnect": t_rec}
    holder["c"].close()
    # load failure / NPU unavailable: start-up must fail loudly, the system configures GPU-only
    t0 = time.perf_counter()
    try:
        bad = client(lane_weights="does_not_exist.pt")
        bad.close()
        res["cases"]["load_failure"] = {"startup": "unexpected success"}
    except RuntimeError as e:
        res["cases"]["load_failure"] = {"startup": "failed (expected)", "error": str(e)[:200],
                                        "detect_s": time.perf_counter() - t0, "action": "GPU-only, logged"}  # fmt: skip
    # non-existent device
    t0 = time.perf_counter()
    cmd = ("cd npu_auto_driving_p2 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate furiosa-npu && "
           "exec nice -n 10 python -u npu_server_v3.py --models det --det-device furiosa:9")  # fmt: skip
    try:
        bad = RngdMultiModelClient(timeout_s=0.25, startup_timeout_s=600, log_path=log,
                                   command=["ssh", "-o", "BatchMode=yes", "furiosa-npu", f"bash -lc '{cmd}'"])  # fmt: skip
        bad.close()
        res["cases"]["npu_unavailable"] = {"startup": "unexpected success"}
    except RuntimeError as e:
        res["cases"]["npu_unavailable"] = {"startup": "failed (expected)", "error": str(e)[:200],
                                           "detect_s": time.perf_counter() - t0, "action": "GPU-only, logged"}  # fmt: skip
    # both down -> DEGRADED
    fb = FailoverBackend(MockBackend(fail_ids=set(range(100)), sleep=False, name="rngd"),
                         MockBackend(fail_ids=set(range(100)), sleep=False, name="gpu"))  # fmt: skip
    r = fb.infer(1, 0.0, None)
    res["cases"]["both_down"] = {"ok": r.ok, "state": fb.state, "events": [e["to"] for e in fb.events]}
    out = resolve("reports/evidence/v3_failure_engineering.json")
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    for k2, v in res["cases"].items():
        print(k2, json.dumps({kk: vv for kk, vv in v.items() if kk != "events"}, default=str)[:260])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
