"""V2 Mode A / Mode C - live closed loop with real asynchronous inference (wall clock).

CARLA runs synchronously with dt = ``--dt`` and the loop is paced so that simulation time follows
wall-clock time (tick k is released at t0 + k*dt; overruns are counted). Every tick: camera frame ->
``LatestFrameQueue`` (size 1) -> inference worker thread (real backend) -> newest result ->
``ResultGate`` (stale results rejected) -> V2 tracker / range Kalman / AEB state machine -> control.

Backends:  gpu            local RTX 3060 (Mode A)
           npu            remote RNGD BF16 through the persistent server, best lossless transport (Mode C)
           npu_failover   NPU with explicit GPU failover; ``--fault-at`` stalls the NPU server for
                          ``--fault-s`` seconds (fault injection) to measure failover and recovery
Frame age = control time - capture time (simulation clock == wall clock while the loop keeps pace).
Output: reports/evidence/v2_live_loop.json (appends one entry per run).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import run_scenarios as v1
import v2_run_scenarios as v2

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb_v2 import AebStateMachine, view_of
from npu_ad.evaluation.metrics import latency_stats
from npu_ad.logutil import setup_logging
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.mono_range import GroundPlaneRanger
from npu_ad.perception.tracker import MultiObjectTracker
from npu_ad.runtime.backends import FuriosaRemoteBackend, GpuInferenceBackend
from npu_ad.runtime.fallback import FailoverBackend, HealthMonitor
from npu_ad.runtime.pipeline import AsyncInferenceWorker, Frame, LatestFrameQueue, ResultGate
from npu_ad.simulation import geometry as geo
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr


def classify(age_p95_s: float | None, overrun_frac: float) -> str:
    """real-time: p95 frame age <= 100 ms and < 10 % paced-tick overruns; near-real-time: <= 250 ms."""
    if age_p95_s is None:
        return "no_results"
    if age_p95_s <= 0.10 and overrun_frac < 0.10:
        return "real-time"
    if age_p95_s <= 0.25 and overrun_frac < 0.25:
        return "near-real-time"
    return "non-real-time"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="pedestrian_crossing", choices=list(v2.SCENARIOS))
    ap.add_argument("--backend", default="gpu", choices=["gpu", "npu", "npu_failover"])
    ap.add_argument("--speed-kmh", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--stale-s", type=float, default=0.3)
    ap.add_argument("--perception-timeout-s", type=float, default=0.5)
    ap.add_argument("--npu-timeout-s", type=float, default=0.25)
    ap.add_argument("--fault-at", type=float, default=None, help="sim time of the injected NPU stall")
    ap.add_argument("--fault-s", type=float, default=1.5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    setup_logging("v2_live_loop")
    import carla

    cfg = load_config(overrides={"carla": {"fixed_delta_seconds": a.dt}})
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    for _ in range(10):
        det(np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8))
    gpu_b = GpuInferenceBackend(det)
    npu_b = None
    if a.backend != "gpu":
        npu_b = FuriosaRemoteBackend(
            dtype="bf16", encoding="content_raw", threads=2, prefilter=True, timeout_s=a.npu_timeout_s,
            ssh_opts=("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-C"),
            log_path=resolve("outputs/v2/npu_server_live.log"),
        )  # fmt: skip
    sc = v2.SCENARIOS[a.scenario]
    sc = dict(sc, duration_s=min(sc["duration_s"], 9.0))
    ego_mps = a.speed_kmh / 3.6
    rows, collisions = [], []
    with CarlaSession(cfg) as s:
        clock0 = [None]

        def sim_clock() -> float:  # pipeline clock = paced simulation time (wall clock since start)
            return 0.0 if clock0[0] is None else time.perf_counter() - clock0[0]

        if a.backend == "gpu":
            backend = gpu_b
        elif a.backend == "npu":
            backend = npu_b
        else:
            backend = FailoverBackend(npu_b, gpu_b, fail_threshold=1, probe_interval_s=0.5, clock=sim_clock)
        health = HealthMonitor(npu_b or gpu_b, remote_every_s=3.0, clock=sim_clock)
        s.world.set_weather(getattr(carla.WeatherParameters, sc["weather"]))
        cmap = s.world.get_map()
        sp = s.spawn_points()[v1.find_straight_spawn(s.world)]
        ego = s.spawn_ego(sp)
        cam = s.attach_rgb(ego)
        s.attach_collision(ego, collisions)
        k_mat = geo.intrinsics(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov"])
        target, tkind, walk_dir = v2.spawn_target(s, carla, cmap.get_waypoint(sp.location), sc, ego_mps)
        for _ in range(int(v1.SETTLE_S / a.dt)):
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            s.tick()
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        fwd = sp.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * ego_mps, fwd.y * ego_mps, 0))
        if sc["kind"] == "lead_brake":
            target.enable_constant_velocity(carla.Vector3D(sc["lead_speed_mps"], 0, 0))
        trigger_gap = sc.get("trigger_gap_m", 0.0) * ego_mps / (v1.DEFAULT_EGO_KMH / 3.6)
        cam_tf = cam.get_transform()
        ranger = GroundPlaneRanger(
            k=k_mat, image_h=cfg["camera"]["height"],
            cam_height_m=cam_tf.location.z - cmap.get_waypoint(cam_tf.location).transform.location.z,
            cam_x_m=cfg["camera"]["x"], ego_half_len_m=ego.bounding_box.extent.x + ego.bounding_box.location.x,
        )  # fmt: skip
        q = LatestFrameQueue(1)
        worker = AsyncInferenceWorker(backend, q, clock=sim_clock)
        gate, mot, sm = ResultGate(a.stale_s), MultiObjectTracker(ranger, v2.CLASS_GROUPS), AebStateMachine()
        last_ok, triggered, faulted, overruns = -1e9, False, False, 0
        worker.start()
        clock0[0] = time.perf_counter()
        try:
            for step in range(int(sc["duration_s"] / a.dt)):
                t = step * a.dt
                lag = sim_clock() - t
                if lag < 0:
                    time.sleep(-lag)
                elif lag > a.dt:
                    overruns += 1
                fid = s.tick()
                img = s.get_image(cam, fid)
                q.put(Frame(fid, t, image_to_bgr(img), enqueue_t=sim_clock()))
                if a.fault_at is not None and not faulted and t >= a.fault_at and npu_b is not None:
                    npu_b.stall(a.fault_s)
                    faulted = True
                t_ctrl = sim_clock()
                res, why = gate.accept(worker.latest(), t_ctrl)
                if res is not None:
                    mot.update(res.capture_t, res.dets)
                    last_ok = res.capture_t
                    health.record(res, q.depth())
                ok = (t_ctrl - last_ok) <= a.perception_timeout_s
                v = ego.get_velocity().length()
                out = sm.step(t_ctrl, [view_of(tr, t_ctrl) for tr in mot.confirmed()], v, ok)
                ts = v1.target_state(ego, target, tkind if tkind != "cyclist" else "vehicle", False)
                if sc["kind"] == "lead_brake" and t >= sc["brake_at_s"]:
                    target.disable_constant_velocity()
                    target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if sc["kind"] == "ped_cross" and (triggered or ts.gap_m <= trigger_gap):
                    triggered = True
                    target.apply_control(carla.WalkerControl(direction=walk_dir, speed=sc["walk_mps"]))
                steer, throttle = v1.lane_follow(ego, cmap, ego_mps)
                if out.brake > 0:
                    throttle = 0.0
                ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=out.brake))
                rows.append({"t": t, "t_ctrl": t_ctrl, "gate": why, "ok": ok, "state": out.state, "brake": out.brake,
                             "ego_mps": v, "gt_gap": ts.gap_m, "gt_closing": ts.closing_mps,
                             "backend": res.backend if res is not None else None,
                             "age_s": (t_ctrl - res.capture_t) if res is not None else None,
                             "collisions": len(collisions)})  # fmt: skip
                if collisions and t - next(r["t"] for r in rows if r["collisions"]) > 0.5:
                    break
        finally:
            worker.stop()
            for _ in range(3):
                s.tick()
    recs = worker.records
    ages = [r["age_s"] for r in rows if r["age_s"] is not None]
    n_ticks = len(rows)
    lat = [r["latency_s"] for r in recs if r["ok"]]
    first_brake = next((r["t"] for r in rows if r["brake"] > 0), None)
    contact = next((i for i, r in enumerate(rows) if r["collisions"]), None)
    gaps = [r["gt_gap"] for r in rows[: contact if contact is not None else len(rows)]]
    stop_i = next(
        (i for i, r in enumerate(rows) if first_brake is not None and r["t"] >= first_brake and r["ego_mps"] < 0.05),
        None,
    )
    if stop_i is not None:
        gaps = gaps[: stop_i + 1]
    gaps = [g for g in gaps if g >= 0] or gaps
    entry = {
        "experiment_id": f"v2_live__{a.scenario}__{a.backend}{a.tag}__" + time.strftime("%Y%m%dT%H%M%S"),
        "scenario": a.scenario, "backend": a.backend, "speed_kmh": a.speed_kmh, "dt_s": a.dt,
        "fault_at_s": a.fault_at, "fault_s": a.fault_s if a.fault_at is not None else None,
        "collision": bool(collisions), "min_gap_m": min(gaps) if gaps else None, "first_brake_t": first_brake,
        "ticks": n_ticks, "overruns": overruns, "overrun_frac": overruns / max(n_ticks, 1),
        "frames": q.n_put, "dropped_frames": q.n_dropped, "inferences": len(recs),
        "inference_ok": sum(r["ok"] for r in recs),
        "gate": {k: v for k, v in vars(gate.stats).items() if k != "ages_s"},
        "frame_age_at_use_ms": latency_stats([a_ for a_ in gate.stats.ages_s]),
        "age_at_control_ms": latency_stats(ages),
        "backend_latency_ms": latency_stats(lat),
        "queue_delay_ms": latency_stats([r["queue_delay_s"] for r in recs]),
        "perception_rate_hz": len(recs) / (n_ticks * a.dt) if n_ticks else None,
        "degraded_ticks": sum(1 for r in rows if not r["ok"]),
        "backend_share": {b: sum(1 for r in recs if r["backend"] == b) for b in {r["backend"] for r in recs}},
        "failover_events": getattr(backend, "events", []),
        "health_samples": health.samples[:6],
        "classification": classify(
            (np.percentile(gate.stats.ages_s, 95) if gate.stats.ages_s else None), overruns / max(n_ticks, 1)
        ),
    }  # fmt: skip
    if npu_b is not None:
        entry["npu_client"] = npu_b.health()
        npu_b.close()
    out = resolve("reports/evidence/v2_live_loop.json")
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"runs": []}
    data["runs"].append(entry)
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    keys = ("scenario", "backend", "collision", "min_gap_m", "first_brake_t", "classification", "overrun_frac",
            "dropped_frames", "perception_rate_hz", "degraded_ticks", "backend_share")  # fmt: skip
    print(json.dumps({k: entry[k] for k in keys}, default=str))
    print(
        "age at use ms",
        {k: round(v, 1) for k, v in entry["frame_age_at_use_ms"].items() if k in ("mean_ms", "p95_ms", "n")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
