"""Phase 6 / Gate F: closed-loop CARLA scenarios with perception-gated AEB.

Pipeline per tick: RGB frame -> YOLO26n (GPU, in-loop) -> association with ground-truth targets
(projected GT centre inside a same-class detection box) -> TTC/gap from CARLA ground truth ->
AEB decision -> applied after ``delay_ticks`` (latency model) -> CARLA vehicle control.

Latency conditions come from measured values (see configs/latency_conditions.json, written by
scripts/make_latency_conditions.py). NPU conditions are *latency injection*: detections still come
from the in-loop GPU model, the measured NPU pipeline latency only sets the actuation delay.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb import AebController, AebParams, Decision, DelayLine, TargetState
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.simulation import geometry as geo
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr

SCENARIOS: dict[str, dict] = {
    "straight_road": {"kind": "none", "weather": "ClearNoon", "duration_s": 10.0},
    "lead_vehicle_braking": {"kind": "lead_brake", "weather": "ClearNoon", "duration_s": 12.0,
                             "gap_m": 25.0, "lead_speed_mps": 5.0, "brake_at_s": 3.0},
    "pedestrian_crossing": {"kind": "ped_cross", "weather": "ClearNoon", "duration_s": 10.0,
                            "ahead_m": 30.0, "side_m": 4.5, "trigger_gap_m": 24.0, "walk_mps": 1.6},
    "static_obstacle": {"kind": "static_vehicle", "weather": "ClearNoon", "duration_s": 10.0, "gap_m": 40.0},
    "low_light": {"kind": "lead_brake", "weather": "ClearNight", "duration_s": 12.0,
                  "gap_m": 25.0, "lead_speed_mps": 5.0, "brake_at_s": 3.0},
    "rain": {"kind": "ped_cross", "weather": "HardRainNoon", "duration_s": 10.0,
             "ahead_m": 30.0, "side_m": 4.5, "trigger_gap_m": 24.0, "walk_mps": 1.6},
}
EGO_TARGET_MPS = 30 / 3.6


def find_straight_spawn(world, min_len_m: float = 90.0) -> int:
    m = world.get_map()
    for i, sp in enumerate(m.get_spawn_points()):
        wp = m.get_waypoint(sp.location)
        yaw0, ok, d = wp.transform.rotation.yaw, True, 0.0
        while d < min_len_m:
            nxt = wp.next(5.0)
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            wp, d = nxt[0], d + 5.0
            if abs((wp.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 2.0:
                ok = False
                break
        if ok:
            return i
    raise RuntimeError("no straight spawn found")


def lane_follow(ego, cmap, carla):
    """Lateral: pursue a waypoint 8 m ahead; longitudinal: P speed control to 30 km/h."""
    tf = ego.get_transform()
    wp = cmap.get_waypoint(tf.location).next(8.0)[0]
    lon, lat = geo.to_ego_frame(wp.transform.location.x, wp.transform.location.y,
                                tf.location.x, tf.location.y, tf.rotation.yaw)
    steer = float(np.clip(math.atan2(lat, lon) * 1.2, -0.5, 0.5))
    v = ego.get_velocity().length()
    throttle = float(np.clip(0.35 + 0.6 * (EGO_TARGET_MPS - v), 0.0, 0.8))
    return steer, throttle


def target_state(ego, actor, kind: str, perceived: bool) -> TargetState:
    et, at = ego.get_transform(), actor.get_transform()
    lon, lat = geo.to_ego_frame(at.location.x, at.location.y, et.location.x, et.location.y, et.rotation.yaw)
    ext = actor.bounding_box.extent
    half_t = geo.half_length_along(ext.x, ext.y, at.rotation.yaw - et.rotation.yaw)
    gap = geo.longitudinal_gap(lon, ego.bounding_box.extent.x, half_t)
    ev, av = ego.get_velocity(), actor.get_velocity()
    closing = geo.closing_speed((ev.x, ev.y), (av.x, av.y), et.rotation.yaw)
    return TargetState(actor.id, kind, gap, lat, closing, perceived)


def run(cfg: dict, name: str, sc: dict, condition: str, delay_ticks: int | None, detector: GpuDetector,
        log_dir: Path, aeb_version: str = "v2") -> dict:
    import carla

    groups = {"vehicle": set(cfg["classes"]["vehicle"]), "pedestrian": set(cfg["classes"]["pedestrian"])}
    dt = cfg["carla"]["fixed_delta_seconds"]
    collisions: list = []
    rows: list[dict] = []
    with CarlaSession(cfg) as s:
        s.world.set_weather(getattr(carla.WeatherParameters, sc["weather"]))
        cmap = s.world.get_map()
        sp = s.spawn_points()[find_straight_spawn(s.world)]
        ego = s.spawn_ego(sp)
        cam = s.attach_rgb(ego)
        s.attach_collision(ego, collisions)
        k_mat = geo.intrinsics(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov"])
        ego_wp = cmap.get_waypoint(sp.location)
        target, tkind = None, None
        if sc["kind"] in ("lead_brake", "static_vehicle"):
            wp = ego_wp.next(sc["gap_m"] + 4.5)[0]
            tf = wp.transform
            tf.location.z += 0.3
            target, tkind = s.spawn("vehicle.audi.a2", tf), "vehicle"
        elif sc["kind"] == "ped_cross":
            wp = ego_wp.next(sc["ahead_m"])[0]
            right = wp.transform.get_right_vector()
            for side in (sc["side_m"], sc["side_m"] - 0.5, sc["side_m"] - 1.0):
                loc = wp.transform.location + carla.Location(x=right.x * side, y=right.y * side, z=1.0)
                try:
                    target = s.spawn("walker.pedestrian.0001", carla.Transform(loc, wp.transform.rotation))
                    break
                except RuntimeError:
                    continue
            if target is None:
                raise RuntimeError("could not place pedestrian")
            tkind = "pedestrian"
            walk_dir = carla.Vector3D(-right.x, -right.y, 0.0)
        s.tick()
        fwd = sp.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * EGO_TARGET_MPS, fwd.y * EGO_TARGET_MPS, 0))
        if target is not None and sc["kind"] == "lead_brake":
            target.enable_constant_velocity(carla.Vector3D(sc["lead_speed_mps"], 0, 0))
        aeb = (AebController(AebParams(hysteresis=aeb_version == "v2")) if condition != "aeb_off" else None)
        delay = DelayLine(delay_ticks or 0)
        triggered = False
        n_ticks = int(sc["duration_s"] / dt)
        for step in range(n_ticks):
            fid = s.tick()
            t_sim = step * dt
            img = s.get_image(cam, fid)
            bgr = image_to_bgr(img)
            dets, times, _ = detector(bgr)
            perceived, ts = False, None
            if target is not None:
                bb = target.bounding_box
                c = target.get_transform().transform(bb.location)
                uv = geo.project(np.array([c.x, c.y, c.z]), np.array(cam.get_transform().get_inverse_matrix()),
                                 k_mat)
                perceived = uv is not None and geo.point_in_any_box(uv, dets, groups[tkind])
                ts = target_state(ego, target, tkind, perceived)
                if sc["kind"] == "lead_brake" and t_sim >= sc["brake_at_s"] and not triggered:
                    target.disable_constant_velocity()
                    triggered = True
                if sc["kind"] == "lead_brake" and triggered:
                    target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if sc["kind"] == "ped_cross" and (triggered or ts.gap_m <= sc["trigger_gap_m"]):
                    triggered = True
                    target.apply_control(carla.WalkerControl(direction=walk_dir, speed=sc["walk_mps"]))
            v = ego.get_velocity().length()
            decision = aeb.decide([ts] if ts else [], v) if aeb else Decision(0.0, "aeb_off")
            applied = delay.push(decision)
            steer, throttle = lane_follow(ego, cmap, carla)
            brake = applied.brake
            if brake > 0:
                throttle = 0.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            rows.append({
                "step": step, "t_sim": round(t_sim, 3), "frame": fid, "ego_mps": v, "n_dets": len(dets),
                "infer_e2e_ms": times.e2e * 1000, "perceived": perceived,
                "gap_m": ts.gap_m if ts else None, "lat_m": ts.lateral_m if ts else None,
                "closing_mps": ts.closing_mps if ts else None,
                "decision": decision.reason, "decision_brake": decision.brake, "applied_brake": brake,
                "throttle": throttle, "steer": steer, "collisions": len(collisions),
            })
            if v < 0.05 and step * dt > 2.0 and brake >= 1.0:
                break  # ego stopped by AEB; outcome is decided
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}__{condition}__{aeb_version}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # separation is only meaningful while the target is ahead and before the first contact
    end = next((i for i, r in enumerate(rows) if r["collisions"]), len(rows) - 1)
    gaps = [r["gap_m"] for r in rows[: end + 1] if r["gap_m"] is not None and r["gap_m"] > -5.0]
    first = lambda key, pred: next((r["t_sim"] for r in rows if pred(r[key])), None)  # noqa: E731
    lat_ms = [r["infer_e2e_ms"] for r in rows]
    return {
        "scenario": name, "condition": condition, "aeb_version": aeb_version, "delay_ticks": delay_ticks, "dt_s": dt,
        "delay_ms": None if delay_ticks is None else delay_ticks * dt * 1000,
        "weather": sc["weather"], "ticks": len(rows),
        "collision": bool(collisions), "collision_with": collisions[0]["other"] if collisions else None,
        "min_gap_m": min(gaps) if gaps else None,
        "first_perceived_t": first("perceived", bool),
        "first_decision_brake_t": first("decision_brake", lambda b: b > 0),
        "first_applied_brake_t": first("applied_brake", lambda b: b > 0),
        "final_speed_mps": rows[-1]["ego_mps"] if rows else None,
        "inloop_gpu_e2e_ms_p50": float(np.percentile(lat_ms, 50)) if lat_ms else None,
        "inloop_gpu_e2e_ms_p95": float(np.percentile(lat_ms, 95)) if lat_ms else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--conditions", nargs="*", default=None, help="names from latency_conditions.json")
    ap.add_argument("--latency-config", default="configs/latency_conditions.json")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", default="reports/evidence/scenarios_v2.json")
    ap.add_argument("--aeb-version", choices=["v1", "v2"], default="v2")
    a = ap.parse_args()
    cfg = load_config(overrides={"carla": {"fixed_delta_seconds": a.dt}})
    lat_cfg = json.loads(resolve(a.latency_config).read_text(encoding="utf-8"))
    conds = {c["name"]: c for c in lat_cfg["conditions"]}
    names = a.conditions or list(conds)
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    for _ in range(10):
        det(np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8))
    out = resolve(a.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"runs": []}
    done = {(r["scenario"], r["condition"]) for r in results["runs"]}
    for sname in a.scenarios:
        for cname in names:
            if (sname, cname) in done:
                print("skip (already recorded)", sname, cname)
                continue
            c = conds[cname]
            delay_ticks = None if c["delay_ms"] is None else math.ceil(round(c["delay_ms"] / 1000 / a.dt, 6))
            t0 = time.perf_counter()
            r = run(cfg, sname, SCENARIOS[sname], cname, delay_ticks, det, resolve("outputs/scenario_logs"),
                  a.aeb_version)
            r["source"] = c.get("source")
            r["wall_s"] = time.perf_counter() - t0
            results["runs"].append(r)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
            tmp.replace(out)  # atomic checkpoint after every run (resumable)
            print(json.dumps({k: r[k] for k in ("scenario", "condition", "delay_ms", "collision", "min_gap_m",
                                                "first_perceived_t", "first_applied_brake_t")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
