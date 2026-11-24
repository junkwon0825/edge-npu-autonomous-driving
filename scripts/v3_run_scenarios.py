# ruff: noqa: E501  (long result/row dict literals kept on one line)
"""V3 closed loop: multi-model perception (detection + lane), LKA + AEB + arbitration, optional radar fusion.

Per 10 ms tick: camera frame every 3 ticks (33 Hz) -> RateScheduler decides which models run ->
detection / lane pipelines (``SimulatedAsyncPipeline`` with ``use_measured`` latency: every frame is
really processed by its backend - local GPU or the remote RNGD server - and the measured call latency is
applied in simulation time; CARLA waits during the call = time-dilated hardware-in-the-loop) ->
age-gated results -> tracker (+ radar) and lane geometry -> SceneState -> AEB state machine (path-relative,
ego-yaw compensated) + LKA -> Arbiter -> CARLA control.

Modes: A = GPU detection + GPU lane, B = RNGD detection + GPU lane, C = GPU detection + RNGD lane,
D = RNGD detection + RNGD lane (one request per frame carries both models, separate PEs).
Ground truth (lane centre from map waypoints, target gaps) is logged for evaluation only.
Output: reports/evidence/v3_scenarios.json (resumable), per-tick logs outputs/v3/scenario_logs.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import time
from pathlib import Path

import numpy as np
import run_scenarios as v1
import v2_run_scenarios as v2

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb_v2 import AebStateMachine, path_relative, view_of
from npu_ad.control.lka import Arbiter, LaneKeeping
from npu_ad.logutil import setup_logging
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.lane_geometry import CameraModel, LaneGeometry
from npu_ad.perception.lane_net import load_lane_model
from npu_ad.perception.mono_range import GroundPlaneRanger
from npu_ad.perception.radar_fusion import RadarConfig, radar_points
from npu_ad.perception.tracker import MultiObjectTracker
from npu_ad.runtime.backends import GpuInferenceBackend
from npu_ad.runtime.fallback import FailoverBackend
from npu_ad.runtime.multimodel import GpuLaneBackend, LaneResult, RateScheduler, RngdMultiModelClient
from npu_ad.runtime.pipeline import Frame, LatencyModel, ResultGate, SimulatedAsyncPipeline
from npu_ad.scene import SceneState
from npu_ad.simulation import geometry as geo
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr

ROUTES = {"straight": ("Town04", 298), "mild": ("Town04", 208), "strong": ("Town10HD_Opt", 154),
          "strong_heldout": ("Town03", 263), "aeb": ("Town10HD_Opt", None)}  # fmt: skip
LK = {"kind": "none", "weather": "ClearNoon", "duration_s": 12.0}
SCENARIOS = {
    "lk_straight": LK | {"route": "straight"},
    "lk_mild": LK | {"route": "mild"},
    "lk_strong": LK | {"route": "strong"},
    "lk_strong_heldout": LK | {"route": "strong_heldout"},
    "lk_offset": LK | {"route": "straight", "offset_m": 0.8},
    "lk_night": LK | {"route": "mild", "weather": "ClearNight"},
    "lk_rain": LK | {"route": "mild", "weather": "HardRainNoon"},
    "lead_vehicle_braking": v2.SCENARIOS["lead_vehicle_braking"] | {"route": "aeb"},
    "static_obstacle": v2.SCENARIOS["static_obstacle"] | {"route": "aeb"},
    "pedestrian_crossing": v2.SCENARIOS["pedestrian_crossing"] | {"route": "aeb"},
    "pedestrian_clearing": v2.SCENARIOS["pedestrian_clearing"] | {"route": "aeb"},
    "cyclist_crossing": v2.SCENARIOS["cyclist_crossing"] | {"route": "aeb"},
    "curve_obstacle": {"kind": "static_vehicle", "weather": "ClearNoon", "duration_s": 12.0, "gap_m": 45.0,
                       "route": "mild"},
    "curve_lead_brake": v2.SCENARIOS["lead_vehicle_braking"] | {"route": "mild"},
    "ped_cornering": v2.SCENARIOS["pedestrian_crossing"] | {"route": "strong"},
    "degraded_lane": LK | {"route": "mild", "fault": ("lane_fail", 2.0, 4.0)},
}  # fmt: skip
FAULTS = {  # applied to the detection path unless stated
    "det_drop30": {"drop_p": 0.3},
    "det_stale20": {"stale_p": 0.2, "stale_extra_s": 0.4},
    "det_timeout": ("det_delay", 1.5, 2.5, 0.6),
    "seg_failure": ("lane_fail", 1.5, 3.5),
    "npu_failure": ("npu_crash", 1.5),
}
YAW_HIST_MAX = 400


class FaultWindow:
    """Wraps a backend: failures or extra latency for frames captured inside [t0, t1)."""

    def __init__(self, backend, t0: float, t1: float, kind: str, extra_s: float = 0.0):
        self.b, self.t0, self.t1, self.kind, self.extra = backend, t0, t1, kind, extra_s
        self.name, self.precision = getattr(backend, "name", "?"), getattr(backend, "precision", "?")

    def infer(self, fid, cap_t, bgr):
        r = self.b.infer(fid, cap_t, bgr)
        if self.t0 <= cap_t < self.t1:
            if self.kind == "fail":
                r.ok, r.error = False, "injected failure"
                if hasattr(r, "mask"):
                    r.mask = None
                elif hasattr(r, "dets"):
                    r.dets = np.zeros((0, 6))
            else:
                r.latency_s += self.extra
        return r

    def health(self):
        return {}

    def close(self):
        pass


def lane_gt(cmap, ego):
    tf = ego.get_transform()
    wp = cmap.get_waypoint(tf.location, project_to_road=True)
    r = wp.transform.get_right_vector()
    d = tf.location - wp.transform.location
    off = d.x * r.x + d.y * r.y  # > 0: ego right of the lane centre
    he = math.radians(wp.transform.rotation.yaw - tf.rotation.yaw)
    he = math.atan2(math.sin(he), math.cos(he))  # lane direction minus ego heading
    return off, he, wp.lane_width


def run(cfg, name, sc, mode, kmh, det, lane_model, client, radar_on, sched_mode, fault, out_logs: Path) -> dict:
    import carla

    dt = 0.01
    ego_mps = kmh / 3.6
    town, sp_idx = ROUTES[sc["route"]]
    cfg = cfg | {"carla": cfg["carla"] | {"map": town, "fixed_delta_seconds": dt}}
    collisions, rows = [], []
    gpu_det, gpu_lane = GpuInferenceBackend(det), GpuLaneBackend(lane_model)
    det_b = {"A": gpu_det, "B": client.detection() if client else None, "C": gpu_det}.get(mode)
    lane_b = {"A": gpu_lane, "B": gpu_lane, "C": client.lane() if client else None}.get(mode)
    events = []
    if fault == "npu_failure":  # explicit failover on both paths
        det_b = FailoverBackend(client.detection(), gpu_det, probe_interval_s=0.5)
        lane_b = FailoverBackend(client.lane(), gpu_lane, probe_interval_s=0.5)
        mode = "B+C failover"
    with CarlaSession(cfg) as s:
        s.tick()
        foreign = [a.type_id for a in s.world.get_actors() if a.type_id.split(".")[0] in ("vehicle", "walker")]
        if foreign:
            raise RuntimeError(f"world not clean: {foreign}; run scripts/carla_reset.py")
        s.world.set_weather(getattr(carla.WeatherParameters, sc["weather"]))
        cmap = s.world.get_map()
        sps = s.spawn_points()
        sp = sps[v1.find_straight_spawn(s.world) if sp_idx is None else sp_idx]
        if sc.get("offset_m"):
            r = sp.get_right_vector()
            sp = carla.Transform(sp.location + carla.Location(x=r.x * sc["offset_m"], y=r.y * sc["offset_m"]),
                                 sp.rotation)  # fmt: skip
        ego = s.spawn_ego(sp)
        cam = s.attach_rgb(ego)
        s.attach_collision(ego, collisions)
        rq: queue.Queue = queue.Queue()
        rc = RadarConfig()
        if radar_on:
            radar = s.spawn("sensor.other.radar", carla.Transform(carla.Location(x=rc.x_m, z=rc.z_m)), attach_to=ego,
                            horizontal_fov=rc.hfov_deg, vertical_fov=rc.vfov_deg, range=rc.range_m,
                            points_per_second=rc.points_per_second, sensor_tick=0.05)  # fmt: skip
            radar.listen(rq.put)
        k_mat = geo.intrinsics(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov"])
        ego_wp = cmap.get_waypoint(sp.location)
        target, tkind, walk_dir = None, None, None
        if sc["kind"] != "none":
            target, tkind, walk_dir = v2.spawn_target(s, carla, ego_wp, sc, ego_mps)
        for _ in range(int(v1.SETTLE_S / dt)):
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            s.tick()
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        fwd = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * ego_mps, fwd.y * ego_mps, 0))
        if target is not None and sc["kind"] == "lead_brake":
            target.enable_constant_velocity(carla.Vector3D(sc["lead_speed_mps"], 0, 0))
        trigger_gap = sc.get("trigger_gap_m", 0.0) * ego_mps / (v1.DEFAULT_EGO_KMH / 3.6)
        cam_tf = cam.get_transform()
        h_cam = cam_tf.location.z - cmap.get_waypoint(cam_tf.location).transform.location.z
        half = ego.bounding_box.extent.x + ego.bounding_box.location.x
        ranger = GroundPlaneRanger(k=k_mat, image_h=cfg["camera"]["height"], cam_height_m=h_cam, cam_x_m=cfg["camera"]["x"],
                                   ego_half_len_m=half)  # fmt: skip
        geom = LaneGeometry(CameraModel(float(k_mat[0, 0]), float(k_mat[1, 1]), float(k_mat[0, 2]), float(k_mat[1, 2]),
                                        h_cam, cfg["camera"]["x"]))  # fmt: skip
        fa = FAULTS.get(fault) or sc.get("fault")
        lm_det = LatencyModel(use_measured=True, seed=7, **(fa if isinstance(fa, dict) else {}))
        if isinstance(fa, tuple) and fa[0] == "det_delay":
            det_b = FaultWindow(det_b, fa[1], fa[2], "delay", fa[3])
        if isinstance(fa, tuple) and fa[0] == "lane_fail":
            lane_b = FaultWindow(lane_b, fa[1], fa[2], "fail")
        if mode == "D":
            both = SimulatedAsyncPipeline(client.both(), LatencyModel(use_measured=True, seed=7))
            det_pipe = lane_pipe = None
        else:
            det_pipe = SimulatedAsyncPipeline(det_b, lm_det)
            lane_pipe = SimulatedAsyncPipeline(lane_b, LatencyModel(use_measured=True, seed=8))
        sched = RateScheduler(mode=sched_mode, lane_every=2)
        gate, lane_gate = ResultGate(0.3), ResultGate(0.3)
        mot = MultiObjectTracker(ranger, v2.CLASS_GROUPS)
        sm, lka, arb = AebStateMachine(), LaneKeeping(), Arbiter()
        lane_est, lane_cap, lane_backend, lane_fid, yaw_hist = None, None, "n/a", None, {}
        last_det_cap, triggered, n_cam, crashed = -1e9, False, 0, False
        out_aeb = None
        for step in range(int(sc["duration_s"] / dt)):
            fid = s.tick()
            t = step * dt
            img = s.get_image(cam, fid)
            tf = ego.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            yaw_rate = math.radians(ego.get_angular_velocity().z)
            v = ego.get_velocity().length()
            if fault == "npu_failure" and not crashed and t >= FAULTS["npu_failure"][1]:
                client.crash()
                crashed = True
            if step % 3 == 0:
                n_cam += 1
                yaw_hist[fid] = yaw
                if len(yaw_hist) > YAW_HIST_MAX:
                    yaw_hist.pop(next(iter(yaw_hist)))
                bgr = image_to_bgr(img)
                hz = out_aeb.ttc_s if out_aeb is not None and out_aeb.state in ("WATCH", "WARNING", "BRAKE") else None
                d_run, l_run = sched.plan(n_cam, hz, lane_est.confidence if lane_est else None,
                                          lane_est.offset_m if lane_est else None)  # fmt: skip
                fr = Frame(fid, t, bgr)
                if mode == "D":
                    if d_run or l_run:
                        op = "infer_both" if (d_run and l_run) else ("infer_det" if d_run else "infer_lane")
                        both.backend.method = op
                        both.on_frame(t, fr)
                else:
                    if d_run:
                        det_pipe.on_frame(t, fr)
                    if l_run:
                        lane_pipe.on_frame(t, Frame(fid, t, bgr))
            # newest results
            if mode == "D":
                dres, lres = split_result(both.poll(t))
            else:
                dres, lres = det_pipe.poll(t), lane_pipe.poll(t)
            res, why = gate.accept(dres, t)
            if res is not None:
                mot.update(res.capture_t, res.dets)
                last_det_cap = res.capture_t
            if radar_on:
                pts = None
                while not rq.empty():
                    pts = radar_points(rq.get_nowait())
                if pts is not None:
                    mot.update_radar(t, pts, half, rc)
            lr, lwhy = lane_gate.accept(lres, t) if lres is not None else (None, "none")
            if lr is not None and lr.mask is not None:
                est = geom.estimate(lr.mask)
                if est.valid:
                    lane_est, lane_cap, lane_backend, lane_fid = est, lr.capture_t, lr.backend, lr.frame_id
            # scene state
            path = None
            if lane_est is not None and lane_cap is not None and t - lane_cap <= 0.5:
                a, b, _c = lane_est.center
                path = (a, b)
            elif v > 1.0:
                path = (yaw_rate / v / 2.0, 0.0)
            views = [path_relative(view_of(tr, t, yaw_rate), path) for tr in mot.confirmed()]
            scene = SceneState(t, v, yaw, yaw_rate, lane_est, lane_cap, yaw_hist.get(lane_fid), views)
            scene.detection.age_s = t - last_det_cap
            scene.detection.backend = res.backend if res is not None else scene.detection.backend
            perception_ok = scene.detection.age_s <= 0.5
            out_aeb = sm.step(t, scene.tracks, v, perception_ok)
            cap_yaw = yaw_hist.get(lane_fid) if lane_fid is not None else None  # odometry at the lane frame's capture
            lk = lka.step(t, lane_est, lane_cap, v, yaw, cap_yaw if cap_yaw is not None else yaw)
            throttle = float(np.clip(0.35 + 0.6 * (ego_mps - v), 0.0, 0.8))
            cmd = arb.combine(t, out_aeb.brake, lk, throttle)
            ego.apply_control(carla.VehicleControl(throttle=cmd.throttle, steer=cmd.steer, brake=cmd.brake))
            # ground truth (evaluation only)
            off, he, lw = lane_gt(cmap, ego)
            ts = None
            if target is not None:
                ts = v1.target_state(ego, target, tkind if tkind != "cyclist" else "vehicle", False)
                if sc["kind"] == "lead_brake" and t >= sc["brake_at_s"] and not triggered:
                    target.disable_constant_velocity()
                    triggered = True
                if sc["kind"] == "lead_brake" and triggered:
                    target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if sc["kind"] == "ped_cross" and (triggered or ts.gap_m <= trigger_gap):
                    triggered = True
                    target.apply_control(carla.WalkerControl(direction=walk_dir, speed=sc["walk_mps"]))
                if sc["kind"] == "bike_cross" and not triggered and ts.gap_m <= trigger_gap:
                    triggered = True
                    target.enable_constant_velocity(carla.Vector3D(sc["ride_mps"], 0, 0))
            rows.append({
                "t": round(t, 3), "v": v, "gt_offset": off, "gt_heading": he, "lane_width": lw,
                "est_offset": lane_est.offset_m if lane_est else None, "est_heading": lane_est.heading_rad if lane_est else None,
                "lane_age": (t - lane_cap) if lane_cap is not None else None, "lane_backend": lane_backend,
                "lka": lk.reason, "steer": cmd.steer, "mode": cmd.mode, "aeb": out_aeb.state, "brake": cmd.brake,
                "ttc": out_aeb.ttc_s, "det_age": scene.detection.age_s, "det_backend": scene.detection.backend,
                "gt_gap": ts.gap_m if ts else None, "gt_closing": ts.closing_mps if ts else None,
                "collisions": len(collisions), "cam": step % 3 == 0, "gate": why, "lane_gate": lwhy,
            })  # fmt: skip
            if collisions and t - next(r["t"] for r in rows if r["collisions"]) > 0.5:
                break
        for _ in range(3):
            s.tick()
        if isinstance(det_b, FailoverBackend):
            events = det_b.events + lane_b.events
    out_logs.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{mode.replace('+', '_').replace(' ', '_')}__{int(kmh)}kmh__{sched_mode}{'__radar' if radar_on else ''}{'__' + fault if fault else ''}"
    with (out_logs / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    meta = {"scenario": name, "kind": sc["kind"], "route": sc["route"], "map": town, "mode": mode, "ego_kmh": kmh,
            "radar": radar_on, "scheduler": sched_mode, "fault": fault or (sc.get("fault") or [None])[0],
            "weather": sc["weather"], "sched_counts": sched.counts, "failover_events": [
                {k: (str(v) if k == "reason" else v) for k, v in e.items()} for e in events],
            "lane_geometry_stats": geom.stats, "det_gate": {k: v for k, v in vars(gate.stats).items() if k != "ages_s"},
            "lane_gate": {k: v for k, v in vars(lane_gate.stats).items() if k != "ages_s"},
            "det_age_ms": _stats(gate.stats.ages_s), "lane_age_ms": _stats(lane_gate.stats.ages_s)}  # fmt: skip
    return meta | summarize(rows, sc, collisions)


def split_result(r):
    """Mode D pipeline result -> (detection result or None, lane result or None)."""
    if r is None:
        return None, None
    if hasattr(r, "det") and hasattr(r, "lane"):
        return r.det, r.lane
    if isinstance(r, LaneResult):
        return None, r
    return r, None


def _stats(v):
    if not v:
        return None
    a = np.asarray(v) * 1000
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def summarize(rows, sc, collisions) -> dict:
    off = np.array([r["gt_offset"] for r in rows])
    he = np.array([r["gt_heading"] for r in rows])
    steer = np.array([r["steer"] for r in rows])
    lw = np.array([r["lane_width"] for r in rows])
    dep = np.abs(off) > (lw / 2 - 0.95)  # a wheel crosses the lane marking (ego half width ~0.95 m)
    departures = int(np.sum(dep[1:] & ~dep[:-1]) + (1 if dep[0] else 0))
    est = [(r["est_offset"], r["gt_offset"], r["est_heading"], r["gt_heading"]) for r in rows
           if r["est_offset"] is not None and r["lane_age"] is not None and r["lane_age"] < 0.2]  # fmt: skip
    contact = next((i for i, r in enumerate(rows) if r["collisions"]), None)
    gaps = [r["gt_gap"] for r in rows[: contact if contact is not None else len(rows)] if r["gt_gap"] is not None]
    stop_i = next((i for i, r in enumerate(rows) if r["brake"] > 0 and r["v"] < 0.05), None)
    if stop_i is not None:
        gaps = gaps[: stop_i + 1]
    passed = next((i for i, g in enumerate(gaps) if g < 0), None)
    if passed is not None and contact is None:
        gaps = gaps[:passed]
    onsets = sum(1 for a, b in zip(rows, rows[1:], strict=False) if a["brake"] == 0 and b["brake"] > 0)
    first = next((r for r in rows if r["brake"] > 0), None)
    hazard = sc["kind"] in v2.HAZARD_KINDS
    return {
        "collision": bool(collisions), "collision_with": collisions[0]["other"] if collisions else None,
        "lat_rmse_m": float(np.sqrt(np.mean(off**2))), "lat_max_m": float(np.max(np.abs(off))),
        "heading_mae_deg": float(np.degrees(np.mean(np.abs(he)))), "lane_departures": departures,
        "steer_rate_rms": float(np.sqrt(np.mean((np.diff(steer) / 0.01) ** 2))) if len(steer) > 1 else None,
        "lane_est_offset_mae_m": float(np.mean([abs(a - b) for a, b, _, _ in est])) if est else None,
        "lane_est_heading_mae_deg": float(np.degrees(np.mean([abs(c - d) for _, _, c, d in est]))) if est else None,
        "lka_unavailable_frac": float(np.mean([r["lka"] != "tracking" for r in rows])),
        "min_gap_m": min(gaps) if gaps and hazard else None, "first_brake_t": first["t"] if first else None,
        "gt_ttc_at_first_brake_s": (first["gt_gap"] / first["gt_closing"]) if first and first["gt_gap"] and
        first["gt_closing"] and first["gt_closing"] > 0.05 else None,
        "brake_onsets": onsets, "unneeded_brake": bool(onsets) and not hazard,
        "stopped": stop_i is not None, "ticks": len(rows), "distance_m": float(sum(r["v"] for r in rows) * 0.01),
    }  # fmt: skip


def _key(r):
    return (r["scenario"], r["mode"], float(r["ego_kmh"]), r["radar"], r["scheduler"], r.get("fault_arg") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--modes", nargs="*", default=["A", "D"])
    ap.add_argument("--speeds-kmh", nargs="*", type=float, default=[30.0])
    ap.add_argument("--radar", nargs="*", default=["off"])
    ap.add_argument("--sched", nargs="*", default=["fixed"])
    ap.add_argument("--faults", nargs="*", default=[""])
    ap.add_argument("--lane-weights", default="models/lane_unet_s3c32.pt")
    ap.add_argument("--out", default="reports/evidence/v3_scenarios.json")
    a = ap.parse_args()
    setup_logging("v3_run_scenarios")
    cfg = load_config()
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    lane_model = load_lane_model(str(resolve(a.lane_weights)), "cuda:0")
    need_npu = any(md in ("B", "C", "D") for md in a.modes) or "npu_failure" in a.faults
    client = None
    git = ["git", "-C", str(resolve(".")), "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(resolve(".")), "status", "--porcelain", "src", "scripts", "remote"],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    commit += "+dirty" if dirty else ""  # uncommitted code must be visible in the evidence
    out = resolve(a.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"runs": []}
    done = {_key(r) for r in results["runs"]}
    order = sorted(a.scenarios, key=lambda n: ROUTES[SCENARIOS[n]["route"]][0])  # group by map
    try:
        for name in order:
            for mode in a.modes:
                for kmh in a.speeds_kmh:
                    for radar in a.radar:
                        for sched in a.sched:
                            for fault in a.faults:
                                key = (name, mode if fault != "npu_failure" else "B+C failover", float(kmh), radar == "on",
                                       sched, fault)  # fmt: skip
                                if key in done:
                                    continue
                                if (mode in ("B", "C", "D") or fault == "npu_failure") and client is None and need_npu:
                                    client = RngdMultiModelClient(timeout_s=1.0, startup_timeout_s=900,
                                                                  log_path=resolve("outputs/v3/npu_server_v3.log"))  # fmt: skip
                                if fault == "npu_failure" and client is not None and client.proc.poll() is not None:
                                    client = RngdMultiModelClient(timeout_s=1.0, startup_timeout_s=900)
                                t0 = time.perf_counter()
                                r = run(cfg, name, SCENARIOS[name], mode, kmh, det, lane_model, client, radar == "on",
                                        sched, fault, resolve("outputs/v3/scenario_logs"))  # fmt: skip
                                if fault == "npu_failure":
                                    client.close()
                                    client = None
                                r |= {"fault_arg": fault, "wall_s": time.perf_counter() - t0, "code_commit": commit,
                                      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "npu_server": client.ready if client else None}  # fmt: skip
                                results["runs"].append(r)
                                tmp = out.with_suffix(".tmp")
                                tmp.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
                                tmp.replace(out)
                                keys = ("scenario", "mode", "ego_kmh", "radar", "scheduler", "fault", "collision",
                                        "lat_rmse_m", "lat_max_m", "lane_departures", "min_gap_m", "first_brake_t",
                                        "unneeded_brake", "lka_unavailable_frac")  # fmt: skip
                                print(json.dumps({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in keys}),
                                      round(r["wall_s"], 1), flush=True)  # fmt: skip
    finally:
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
