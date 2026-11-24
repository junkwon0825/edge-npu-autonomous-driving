"""Phase 6 / Gate F: closed-loop CARLA scenarios with AEB.

Pipeline per tick: RGB frame -> YOLO26n (GPU, in-loop) -> AEB target -> decision -> applied after
``delay_ticks`` (latency model) -> CARLA vehicle control. Three AEB input modes (``--modes``):

* ``gt_oracle`` - target gap/TTC from CARLA ground truth, perception ignored (upper bound).
* ``gt_gated``  - ground-truth gap/TTC, braking only while the detector confirms the target
                  (projected GT centre inside a same-class box). This is *GT-assisted*, not
                  perception-driven. Used for scenarios_v1/v2.
* ``mono``      - perception-driven: gap and lateral offset from the detection box with a
                  flat-ground camera model, closing speed from the range history
                  (``npu_ad.perception.mono_range``). CARLA ground truth is only logged for
                  evaluation (range error, TTC at brake onset), never fed to the controller.

Latency conditions come from measured values (see configs/latency_conditions.json, written by
scripts/make_latency_conditions.py). NPU conditions are *latency injection*: detections still come
from the in-loop GPU model, the measured NPU pipeline latency only sets the actuation delay.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb import AebController, AebParams, Decision, DelayLine, TargetState
from npu_ad.logutil import setup_logging
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.mono_range import CipoTracker, GroundPlaneRanger
from npu_ad.perception.yolo_raw import letterbox
from npu_ad.simulation import geometry as geo
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr

log = logging.getLogger("run_scenarios")

SCENARIOS: dict[str, dict] = {
    "straight_road": {"kind": "none", "weather": "ClearNoon", "duration_s": 10.0},
    "lead_vehicle_braking": {
        "kind": "lead_brake",
        "weather": "ClearNoon",
        "duration_s": 12.0,
        "gap_m": 25.0,
        "lead_speed_mps": 5.0,
        "brake_at_s": 3.0,
    },
    "pedestrian_crossing": {
        "kind": "ped_cross",
        "weather": "ClearNoon",
        "duration_s": 10.0,
        "ahead_m": 30.0,
        "side_m": 4.5,
        "trigger_gap_m": 24.0,
        "walk_mps": 1.6,
    },
    "static_obstacle": {"kind": "static_vehicle", "weather": "ClearNoon", "duration_s": 10.0, "gap_m": 40.0},
    "low_light": {
        "kind": "lead_brake",
        "weather": "ClearNight",
        "duration_s": 12.0,
        "gap_m": 25.0,
        "lead_speed_mps": 5.0,
        "brake_at_s": 3.0,
    },
    "rain": {
        "kind": "ped_cross",
        "weather": "HardRainNoon",
        "duration_s": 10.0,
        "ahead_m": 30.0,
        "side_m": 4.5,
        "trigger_gap_m": 24.0,
        "walk_mps": 1.6,
    },
    # false-positive probes: a target is present but never enters the ego corridor
    "adjacent_vehicle": {
        "kind": "side_vehicle",
        "weather": "ClearNoon",
        "duration_s": 8.0,
        "gap_m": 35.0,
        "side_m": 3.5,
    },
    "pedestrian_roadside": {
        "kind": "ped_static",
        "weather": "ClearNoon",
        "duration_s": 8.0,
        "ahead_m": 30.0,
        "side_m": 4.5,
    },
}
HAZARD_KINDS = {"lead_brake", "ped_cross", "static_vehicle"}
DEFAULT_EGO_KMH = 30.0
SETTLE_S = 1.0
MODES = ("gt_oracle", "gt_gated", "mono")


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


def lane_follow(ego, cmap, target_mps: float):
    """Lateral: pursue a waypoint 8 m ahead; longitudinal: P speed control to ``target_mps``."""
    tf = ego.get_transform()
    wp = cmap.get_waypoint(tf.location).next(8.0)[0]
    lon, lat = geo.to_ego_frame(
        wp.transform.location.x, wp.transform.location.y, tf.location.x, tf.location.y, tf.rotation.yaw
    )
    steer = float(np.clip(math.atan2(lat, lon) * 1.2, -0.5, 0.5))
    v = ego.get_velocity().length()
    throttle = float(np.clip(0.35 + 0.6 * (target_mps - v), 0.0, 0.8))
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


def _onsets(rows: list[dict]) -> list[dict]:
    """Ticks where the applied brake goes from 0 to >0, with ground truth at that moment."""
    out, prev = [], 0.0
    for r in rows:
        if r["applied_brake"] > 0 and prev == 0:
            out.append({"t_sim": r["t_sim"], "gt_gap_m": r["gap_m"], "gt_lat_m": r["lat_m"], "gt_ttc_s": r["gt_ttc_s"]})
        prev = r["applied_brake"]
    return out


def run(
    cfg: dict,
    name: str,
    sc: dict,
    condition: str,
    delay_ticks: int | None,
    detector: GpuDetector,
    log_dir: Path,
    aeb_version: str = "v2",
    mode: str = "gt_gated",
    ego_kmh: float = DEFAULT_EGO_KMH,
    tag: str = "",
    dump_every: int = 0,
    on_tick=None,
) -> dict:
    """Run one scenario. ``on_tick(info)`` is an optional read-only observer (demo overlay); it
    receives the per-tick perception/decision state and must not influence the control loop."""
    import carla

    groups = {"vehicle": set(cfg["classes"]["vehicle"]), "pedestrian": set(cfg["classes"]["pedestrian"])}
    dt = cfg["carla"]["fixed_delta_seconds"]
    ego_mps = ego_kmh / 3.6
    params = AebParams(hysteresis=aeb_version == "v2")
    collisions: list = []
    rows: list[dict] = []
    with CarlaSession(cfg) as s:
        s.world.set_weather(getattr(carla.WeatherParameters, sc["weather"]))
        cmap = s.world.get_map()
        sp = s.spawn_points()[find_straight_spawn(s.world)]
        map_name = cmap.name
        spawn_pose = [
            round(sp.location.x, 3),
            round(sp.location.y, 3),
            round(sp.location.z, 3),
            round(sp.rotation.yaw, 3),
        ]
        ego = s.spawn_ego(sp)
        cam = s.attach_rgb(ego)
        s.attach_collision(ego, collisions)
        k_mat = geo.intrinsics(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov"])
        ego_wp = cmap.get_waypoint(sp.location)
        target, tkind, walk_dir = None, None, None
        if sc["kind"] in ("lead_brake", "static_vehicle", "side_vehicle"):
            wp = ego_wp.next(sc["gap_m"] + 4.5)[0]
            tf = wp.transform
            if sc["kind"] == "side_vehicle":
                right = tf.get_right_vector()
                tf = carla.Transform(
                    tf.location + carla.Location(x=right.x * sc["side_m"], y=right.y * sc["side_m"]), tf.rotation
                )
            tf.location.z += 0.3
            target, tkind = s.spawn("vehicle.audi.a2", tf), "vehicle"
        elif sc["kind"] in ("ped_cross", "ped_static"):
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
        # settle phase: spawn points are elevated; let ego and target drop and the suspension come to
        # rest before the scenario clock starts (otherwise pitch oscillation corrupts camera geometry)
        for _ in range(int(SETTLE_S / dt)):
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            s.tick()
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        fwd = sp.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * ego_mps, fwd.y * ego_mps, 0))
        if target is not None and sc["kind"] == "lead_brake":
            target.enable_constant_velocity(carla.Vector3D(sc["lead_speed_mps"], 0, 0))
        # pedestrian trigger scales with ego speed so the hazard appears at the same time-to-arrival
        trigger_gap = sc.get("trigger_gap_m", 0.0) * ego_mps / (DEFAULT_EGO_KMH / 3.6)
        # camera mounting calibration, measured once at rest: height above the road, offset to the bumper
        cam_tf = cam.get_transform()
        ranger = GroundPlaneRanger(
            k=k_mat,
            image_h=cfg["camera"]["height"],
            cam_height_m=cam_tf.location.z - cmap.get_waypoint(cam_tf.location).transform.location.z,
            cam_x_m=cfg["camera"]["x"],
            ego_half_len_m=ego.bounding_box.extent.x + ego.bounding_box.location.x,
        )
        tracker = CipoTracker(ranger, groups, corridor_half_width_m=params.corridor_half_width_m)
        aeb = AebController(params) if condition != "aeb_off" else None
        delay = DelayLine(delay_ticks or 0)
        triggered = False
        n_ticks = int(sc["duration_s"] / dt)
        dump_frames: list[np.ndarray] = []
        dump_steps: list[int] = []
        w, h = cfg["camera"]["width"], cfg["camera"]["height"]
        for step in range(n_ticks):
            fid = s.tick()
            t_sim = step * dt
            img = s.get_image(cam, fid)
            bgr = image_to_bgr(img)
            dets, times, _ = detector(bgr)
            if dump_every and step % dump_every == 0:  # exact detector inputs for offline NPU replay
                dump_frames.append(letterbox(bgr, cfg["model"]["imgsz"])[0])
                dump_steps.append(step)
            perceived, ts, uv = False, None, None
            if target is not None:
                bb = target.bounding_box
                c = target.get_transform().transform(bb.location)
                uv = geo.project(np.array([c.x, c.y, c.z]), np.array(cam.get_transform().get_inverse_matrix()), k_mat)
                perceived = uv is not None and geo.point_in_any_box(uv, dets, groups[tkind])
                ts = target_state(ego, target, tkind, perceived)
                if sc["kind"] == "lead_brake" and t_sim >= sc["brake_at_s"] and not triggered:
                    target.disable_constant_velocity()
                    triggered = True
                if sc["kind"] == "lead_brake" and triggered:
                    target.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                if sc["kind"] == "ped_cross" and (triggered or ts.gap_m <= trigger_gap):
                    triggered = True
                    target.apply_control(carla.WalkerControl(direction=walk_dir, speed=sc["walk_mps"]))
            cipo, rng = tracker.update(t_sim, dets)
            if mode == "mono":
                aeb_targets = [cipo] if cipo else []
            elif mode == "gt_oracle":
                aeb_targets = (
                    [TargetState(ts.actor_id, ts.kind, ts.gap_m, ts.lateral_m, ts.closing_mps, True)] if ts else []
                )
            else:
                aeb_targets = [ts] if ts else []
            v = ego.get_velocity().length()
            decision = aeb.decide(aeb_targets, v) if aeb else Decision(0.0, "aeb_off")
            applied = delay.push(decision)
            steer, throttle = lane_follow(ego, cmap, ego_mps)
            brake = applied.brake
            if brake > 0:
                throttle = 0.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            if on_tick is not None:
                on_tick(
                    {
                        "step": step,
                        "t_sim": t_sim,
                        "bgr": bgr,
                        "dets": dets,
                        "times": times,
                        "cipo": cipo,
                        "range": rng,
                        "decision": decision,
                        "applied_brake": brake,
                        "ego_mps": v,
                        "gt": ts,
                        "collisions": len(collisions),
                    }
                )
            # mono range error is evaluated only when the CIPO box is the GT target (GT centre inside it)
            mono_matches_gt = bool(
                rng is not None
                and uv is not None
                and geo.point_in_any_box(uv, np.array([[*rng.box, rng.conf, rng.cls]]), groups[tkind])
            )
            rows.append(
                {
                    "step": step,
                    "t_sim": round(t_sim, 3),
                    "frame": fid,
                    "ego_mps": v,
                    "n_dets": len(dets),
                    "infer_e2e_ms": times.e2e * 1000,
                    "perceived": perceived,
                    "gap_m": ts.gap_m if ts else None,
                    "lat_m": ts.lateral_m if ts else None,
                    "closing_mps": ts.closing_mps if ts else None,
                    "gt_ttc_s": (ts.gap_m / ts.closing_mps) if ts and ts.closing_mps > 0.05 and ts.gap_m > 0 else None,
                    "in_image": bool(uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h),
                    "mono_gap_m": cipo.gap_m if cipo else None,
                    "mono_lat_m": cipo.lateral_m if cipo else None,
                    "mono_closing_mps": cipo.closing_mps if cipo else None,
                    "mono_track": cipo.actor_id if cipo else None,
                    "mono_truncated": rng.truncated if rng else None,
                    "mono_matches_gt": mono_matches_gt,
                    "decision": decision.reason,
                    "decision_brake": decision.brake,
                    "applied_brake": brake,
                    "throttle": throttle,
                    "steer": steer,
                    "collisions": len(collisions),
                }
            )
            if v < 0.05 and step * dt > 2.0 and brake >= 1.0:
                break  # ego stopped by AEB; outcome is decided
        for _ in range(3):  # collision callbacks are asynchronous: let late events land before teardown
            s.tick()
    log_dir.mkdir(parents=True, exist_ok=True)
    if dump_frames:
        dump_dir = resolve("outputs/replay")
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{name}__{condition}__{aeb_version}__{mode}__{int(ego_kmh)}kmh" + (f"__{tag}" if tag else "")
        np.savez_compressed(
            dump_dir / f"{stem}.npz",
            frames=np.stack(dump_frames),
            steps=np.array(dump_steps),
            cam_height_m=ranger.cam_height_m,
            ego_half_len_m=ranger.ego_half_len_m,
            dt=dt,
        )
    suffix = f"__{mode}__{int(ego_kmh)}kmh" + (f"__{tag}" if tag else "")
    with (log_dir / f"{name}__{condition}__{aeb_version}{suffix}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "scenario": name,
        "kind": sc["kind"],
        "hazard": sc["kind"] in HAZARD_KINDS,
        "condition": condition,
        "mode": mode,
        "ego_kmh": ego_kmh,
        "tag": tag,
        "aeb_version": aeb_version,
        "delay_ticks": delay_ticks,
        "dt_s": dt,
        "delay_ms": None if delay_ticks is None else delay_ticks * dt * 1000,
        "weather": sc["weather"],
        "camera_height_m": ranger.cam_height_m,
        "world": {
            "map": map_name,
            "spawn_xyz_yaw": spawn_pose,
            "seed": cfg["carla"]["seed"],
            "fixed_delta_s": dt,
            "settle_s": SETTLE_S,
            "scenario_ticks_max": int(sc["duration_s"] / dt),
            "initial_speed_mps": ego_mps,
        },
    }
    return meta | summarize(rows, collisions, dt, meta["hazard"], params.corridor_half_width_m)


def summarize(rows: list[dict], collisions: list, dt: float, hazard: bool, corridor_m: float) -> dict:
    # separation is only meaningful while the target is ahead and before the first contact
    end = next((i for i, r in enumerate(rows) if r["collisions"]), len(rows) - 1)
    pre = rows[: end + 1]
    gaps = [r["gap_m"] for r in pre if r["gap_m"] is not None and r["gap_m"] > -5.0]
    first = lambda key, pred: next((r["t_sim"] for r in rows if pred(r[key])), None)  # noqa: E731
    lat_ms = [r["infer_e2e_ms"] for r in rows]
    onsets = _onsets(rows)
    i_brake = next((i for i, r in enumerate(rows) if r["applied_brake"] > 0), None)
    brake_dist = sum(r["ego_mps"] * dt for r in rows[i_brake:]) if i_brake is not None else None
    # detection misses: GT target in the corridor, ahead within 50 m and inside the image, but not detected
    vis = [
        r
        for r in pre
        if r["in_image"] and r["gap_m"] is not None and 0 < r["gap_m"] < 50 and abs(r["lat_m"]) <= corridor_m
    ]
    errs = [r["mono_gap_m"] - r["gap_m"] for r in pre if r["mono_matches_gt"] and r["mono_gap_m"] is not None]
    return {
        "ticks": len(rows),
        "collision": bool(collisions),
        "collision_with": collisions[0]["other"] if collisions else None,
        "min_gap_m": min(gaps) if gaps else None,
        "first_perceived_t": first("perceived", bool),
        "first_decision_brake_t": first("decision_brake", lambda b: b > 0),
        "first_applied_brake_t": first("applied_brake", lambda b: b > 0),
        "gt_ttc_at_first_brake_s": onsets[0]["gt_ttc_s"] if onsets else None,
        "gt_gap_at_first_brake_m": onsets[0]["gt_gap_m"] if onsets else None,
        "braking_distance_m": brake_dist,
        "brake_onsets": len(onsets),
        "false_positive_brake_events": None if hazard else len(onsets),
        "in_path_visible_ticks": len(vis),
        "in_path_missed_ratio": (sum(not r["perceived"] for r in vis) / len(vis)) if vis else None,
        "mono_range_err": {
            "n": len(errs),
            "mean_m": float(np.mean(errs)) if errs else None,
            "mae_m": float(np.mean(np.abs(errs))) if errs else None,
            "p95_abs_m": float(np.percentile(np.abs(errs), 95)) if errs else None,
        },
        "final_speed_mps": rows[-1]["ego_mps"] if rows else None,
        "inloop_gpu_e2e_ms_p50": float(np.percentile(lat_ms, 50)) if lat_ms else None,
        "inloop_gpu_e2e_ms_p95": float(np.percentile(lat_ms, 95)) if lat_ms else None,
    }


def _run_key(r: dict) -> tuple:
    return (r["scenario"], r["condition"], r.get("mode", "gt_gated"), float(r.get("ego_kmh", 30.0)), r.get("tag", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--conditions", nargs="*", default=None, help="names from latency_conditions.json")
    ap.add_argument("--latency-config", default="configs/latency_conditions.json")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", default="reports/evidence/scenarios_v3.json")
    ap.add_argument("--aeb-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--modes", nargs="*", choices=MODES, default=["gt_gated"])
    ap.add_argument("--speeds-kmh", nargs="*", type=float, default=[DEFAULT_EGO_KMH])
    ap.add_argument("--tag", default="", help="distinguishes repeated runs (determinism check)")
    ap.add_argument("--dump-every", type=int, default=0, help="save every N-th letterboxed frame (replay)")
    a = ap.parse_args()
    setup_logging("run_scenarios")
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
    done = {_run_key(r) for r in results["runs"]}
    for sname in a.scenarios:
        for mode in a.modes:
            for kmh in a.speeds_kmh:
                for cname in names:
                    if cname == "aeb_off" and mode != a.modes[0]:
                        continue  # AEB off does not depend on the AEB input mode
                    if (sname, cname, mode, float(kmh), a.tag) in done:
                        print("skip (already recorded)", sname, cname, mode, kmh, a.tag)
                        continue
                    c = conds[cname]
                    delay_ticks = None if c["delay_ms"] is None else math.ceil(round(c["delay_ms"] / 1000 / a.dt, 6))
                    t0 = time.perf_counter()
                    r = run(
                        cfg,
                        sname,
                        SCENARIOS[sname],
                        cname,
                        delay_ticks,
                        det,
                        resolve("outputs/scenario_logs"),
                        a.aeb_version,
                        mode,
                        kmh,
                        a.tag,
                        a.dump_every,
                    )
                    r["source"] = c.get("source")
                    r["wall_s"] = time.perf_counter() - t0
                    results["runs"].append(r)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    tmp.replace(out)  # atomic checkpoint after every run (resumable)
                    keys = ("scenario", "mode", "ego_kmh", "condition", "collision", "min_gap_m")
                    keys += ("first_applied_brake_t", "gt_ttc_at_first_brake_s", "brake_onsets")
                    print(json.dumps({k: r[k] for k in keys}), flush=True)
                    log.info("run finished", extra={k: r[k] for k in keys + ("delay_ms", "wall_s", "tag")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
