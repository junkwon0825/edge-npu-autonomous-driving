"""V2 closed-loop CARLA matrix: async frame-age-aware perception, tracking, filtered range rate, V2 AEB.

Per 10 ms tick (physics and control at 100 Hz, as in V1):
  camera frame every ``--cam-every`` ticks (33 Hz) -> ``SimulatedAsyncPipeline`` (latest-frame queue,
  one worker, seeded latency model in simulation time) -> results released at capture + latency ->
  ``ResultGate`` (stale results rejected) -> controller -> CARLA control.

Controllers (``--controllers``), all fed by the same pipeline output:
  ``v2``      MultiObjectTracker + per-track RangeKalman / lateral Kalman + AebStateMachine
  ``v1``      V1 perception-driven stack (CIPO tracker, 0.3 s LSQ rate, latched AebController v2)
  ``aeb_off`` no AEB (baseline)
Ground truth is logged for evaluation only. Perception is the local GPU detector; the latency model
says *when* each result becomes available (measured GPU / RNGD distributions or fixed stress values).

Results: ``reports/evidence/v2_scenarios.json`` (resumable, atomic), per-tick logs in
``outputs/v2/scenario_logs``. V1 evidence is never written.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import run_scenarios as v1

from npu_ad.config import load_config, resolve
from npu_ad.control.aeb import AebController, AebParams, Decision
from npu_ad.control.aeb_v2 import AebStateMachine, view_of
from npu_ad.logutil import setup_logging
from npu_ad.perception.gpu_runner import GpuDetector
from npu_ad.perception.mono_range import CipoTracker, GroundPlaneRanger
from npu_ad.perception.tracker import MultiObjectTracker
from npu_ad.runtime.backends import GpuInferenceBackend
from npu_ad.runtime.pipeline import Frame, LatencyModel, ResultGate, SimulatedAsyncPipeline
from npu_ad.simulation import geometry as geo
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr

log = logging.getLogger("v2_run_scenarios")

SCENARIOS = {k: v1.SCENARIOS[k] for k in (
    "static_obstacle", "lead_vehicle_braking", "pedestrian_crossing", "rain", "low_light", "adjacent_vehicle",
    "straight_road",
)} | {  # fmt: skip
    # the pedestrian crosses the whole lane and leaves it ~2.5 s before the ego arrives (no brake needed)
    "pedestrian_clearing": {"kind": "ped_cross", "weather": "ClearNoon", "duration_s": 12.0, "ahead_m": 50.0,
                            "side_m": 4.5, "trigger_gap_m": 48.0, "walk_mps": 1.6, "scale_ahead": True},
    # cyclist crossing from the right at 2.5 m/s, triggered at 2.4 s time-to-arrival
    "cyclist_crossing": {"kind": "bike_cross", "weather": "ClearNoon", "duration_s": 10.0, "ahead_m": 30.0,
                         "side_m": 4.5, "trigger_gap_m": 20.0, "ride_mps": 2.5},
    # false-positive probe: pedestrian standing at the kerb, 2.6 m right of the lane centre
    "pedestrian_curbside": {"kind": "ped_static", "weather": "ClearNoon", "duration_s": 8.0, "ahead_m": 30.0,
                            "side_m": 2.6},
}  # fmt: skip
HAZARD_KINDS = {"lead_brake", "ped_cross", "static_vehicle", "bike_cross"}
NO_BRAKE_NEEDED = {"pedestrian_clearing", "adjacent_vehicle", "straight_road", "pedestrian_curbside"}
CLASS_GROUPS = {"vehicle": {2, 3, 5, 7}, "pedestrian": {0}, "cyclist": {1}}


def latency_models(seed: int) -> dict[str, LatencyModel]:
    m = {f"fixed_{ms}ms": LatencyModel(base_s=ms / 1000, seed=seed) for ms in (0, 20, 50, 100, 150, 200)}
    m |= {
        "jitter_50pm40ms": LatencyModel(base_s=0.05, jitter_s=0.04, seed=seed),
        "burst_25ms_250ms": LatencyModel(
            base_s=0.025, burst_period_s=1.0, burst_len_s=0.3, burst_extra_s=0.25, seed=seed
        ),  # fmt: skip
        "drop30_25ms": LatencyModel(base_s=0.025, drop_p=0.3, seed=seed),
        "stale20_25ms": LatencyModel(base_s=0.025, stale_p=0.2, stale_extra_s=0.4, seed=seed),
    }
    lat = json.loads(resolve("reports/evidence/v2_latency_samples.json").read_text(encoding="utf-8"))
    for name, samples in lat["samples_s"].items():
        m[name] = LatencyModel(samples=samples, seed=seed)
    return m


def spawn_target(s, carla, ego_wp, sc: dict, ego_mps: float):
    kind = sc["kind"]
    if kind in ("lead_brake", "static_vehicle", "side_vehicle"):
        tf = ego_wp.next(sc["gap_m"] + 4.5)[0].transform
        if kind == "side_vehicle":
            r = tf.get_right_vector()
            tf = carla.Transform(tf.location + carla.Location(x=r.x * sc["side_m"], y=r.y * sc["side_m"]), tf.rotation)
        tf.location.z += 0.3
        return s.spawn("vehicle.audi.a2", tf), "vehicle", None
    ahead = sc["ahead_m"]
    if sc.get("scale_ahead"):
        ahead = min(85.0, ahead * ego_mps / (v1.DEFAULT_EGO_KMH / 3.6))
    wp = ego_wp.next(ahead)[0]
    right = wp.transform.get_right_vector()
    if kind == "bike_cross":
        loc = wp.transform.location + carla.Location(x=right.x * sc["side_m"], y=right.y * sc["side_m"], z=0.4)
        rot = carla.Rotation(yaw=wp.transform.rotation.yaw - 90.0)
        return s.spawn("vehicle.diamondback.century", carla.Transform(loc, rot)), "cyclist", None
    for side in (sc["side_m"], sc["side_m"] - 0.5, sc["side_m"] - 1.0):
        loc = wp.transform.location + carla.Location(x=right.x * side, y=right.y * side, z=1.0)
        try:
            ped = s.spawn("walker.pedestrian.0001", carla.Transform(loc, wp.transform.rotation))
            return ped, "pedestrian", carla.Vector3D(-right.x, -right.y, 0.0)
        except RuntimeError:
            continue
    raise RuntimeError("could not place pedestrian")


def run(cfg, name: str, sc: dict, controller: str, cond: str, model: LatencyModel, det: GpuDetector,
        ego_kmh: float, cam_every: int, stale_s: float, perception_timeout_s: float,
        log_dir: Path) -> dict:  # fmt: skip
    import carla

    dt = cfg["carla"]["fixed_delta_seconds"]
    ego_mps = ego_kmh / 3.6
    collisions: list = []
    rows: list[dict] = []
    trans: list = []
    with CarlaSession(cfg) as s:
        s.tick()  # refresh the actor snapshot (synchronous mode)
        foreign = [a.type_id for a in s.world.get_actors() if a.type_id.split(".")[0] in ("vehicle", "walker")]
        if foreign:  # integrity guard: leftovers of a killed client would act as unplanned obstacles
            raise RuntimeError(f"world not clean before the run: {foreign}; run scripts/carla_reset.py")
        s.world.set_weather(getattr(carla.WeatherParameters, sc["weather"]))
        cmap = s.world.get_map()
        sp = s.spawn_points()[v1.find_straight_spawn(s.world)]
        ego = s.spawn_ego(sp)
        cam = s.attach_rgb(ego)
        s.attach_collision(ego, collisions)
        k_mat = geo.intrinsics(cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fov"])
        ego_wp = cmap.get_waypoint(sp.location)
        target, tkind, walk_dir = (None, None, None)
        if sc["kind"] != "none":
            target, tkind, walk_dir = spawn_target(s, carla, ego_wp, sc, ego_mps)
        for _ in range(int(v1.SETTLE_S / dt)):
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            s.tick()
        ego.apply_control(carla.VehicleControl(hand_brake=False))
        fwd = sp.get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * ego_mps, fwd.y * ego_mps, 0))
        if target is not None and sc["kind"] == "lead_brake":
            target.enable_constant_velocity(carla.Vector3D(sc["lead_speed_mps"], 0, 0))
        trigger_gap = sc.get("trigger_gap_m", 0.0) * ego_mps / (v1.DEFAULT_EGO_KMH / 3.6)
        cam_tf = cam.get_transform()
        ranger = GroundPlaneRanger(
            k=k_mat, image_h=cfg["camera"]["height"],
            cam_height_m=cam_tf.location.z - cmap.get_waypoint(cam_tf.location).transform.location.z,
            cam_x_m=cfg["camera"]["x"], ego_half_len_m=ego.bounding_box.extent.x + ego.bounding_box.location.x,
        )  # fmt: skip
        pipe = SimulatedAsyncPipeline(GpuInferenceBackend(det), model, maxsize=1)
        gate = ResultGate(stale_s)
        mot = MultiObjectTracker(ranger, CLASS_GROUPS)
        sm = AebStateMachine()
        cipo_tr = CipoTracker(ranger, {k: CLASS_GROUPS[k] for k in ("vehicle", "pedestrian")})
        aeb1 = AebController(AebParams(hysteresis=True))
        cipo = None
        last_ok_capture = -1e9
        triggered = False
        stopped_since = None
        n_ticks = int(sc["duration_s"] / dt)
        for step in range(n_ticks):
            fid = s.tick()
            t = step * dt
            img = s.get_image(cam, fid)
            if step % cam_every == 0:
                pipe.on_frame(t, Frame(fid, t, image_to_bgr(img)))
            res, why = gate.accept(pipe.poll(t), t)
            dets = res.dets if res is not None else None
            if res is not None:
                last_ok_capture = res.capture_t
            perception_ok = (t - last_ok_capture) <= perception_timeout_s
            # ground truth (evaluation only)
            ts, uv, gt_in_img = None, None, False
            if target is not None:
                bb = target.bounding_box
                c = target.get_transform().transform(bb.location)
                uv = geo.project(np.array([c.x, c.y, c.z]), np.array(cam.get_transform().get_inverse_matrix()), k_mat)
                gt_in_img = (
                    uv is not None and 0 <= uv[0] < cfg["camera"]["width"] and 0 <= uv[1] < cfg["camera"]["height"]
                )
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
            v = ego.get_velocity().length()
            row = {"step": step, "t": round(t, 3), "ego_mps": v, "gate": why, "perception_ok": perception_ok}
            if res is not None:
                row |= {"res_id": res.frame_id, "res_age_s": t - res.capture_t, "n_dets": len(dets)}
            brake, state = 0.0, "OFF"
            crit = None
            if controller == "v2":
                if dets is not None:
                    mot.update(res.capture_t, dets)
                views = [view_of(tr, t) for tr in mot.confirmed()]
                out = sm.step(t, views, v, perception_ok)
                brake, state = out.brake, out.state
                crit = next((tr for tr in mot.confirmed() if tr.track_id == out.target_id), None)
                row |= {"state": state, "reason": out.reason, "ttc": out.ttc_s, "path": out.path,
                        "range": out.range_m, "closing": out.closing_mps, "n_tracks": len(views)}  # fmt: skip
                if crit is not None:
                    row |= {"crit_id": crit.track_id, "crit_lat": crit.lat.predict(t),
                            "crit_vlat": None if crit.lat.x is None else float(crit.lat.x[1])}  # fmt: skip
            elif controller == "v1":
                if dets is not None:
                    cipo, _ = cipo_tr.update(res.capture_t, dets)
                d = aeb1.decide([cipo] if cipo else [], v)
                brake, state = d.brake, d.reason
                row |= {"state": state, "range": cipo.gap_m if cipo else None,
                        "closing": cipo.closing_mps if cipo else None}  # fmt: skip
            else:
                d = Decision(0.0, "aeb_off")
            steer, throttle = v1.lane_follow(ego, cmap, ego_mps)
            if brake > 0:
                throttle = 0.0
            ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            # tracking evaluation: which confirmed track covers the projected GT centre
            gt_track = None
            if controller == "v2" and gt_in_img:
                for tr in mot.confirmed():
                    b = tr.box
                    if b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3]:
                        gt_track = tr.track_id
                        e = view_of(tr, t)
                        row |= {"gt_track_range": e.range_m, "gt_track_rate": e.rate_mps,
                                "gt_track_lat": e.lateral_m, "gt_track_vlat": e.lat_rate_mps}  # fmt: skip
                        break
            row |= {
                "brake": brake, "throttle": throttle, "collisions": len(collisions), "gt_in_img": gt_in_img,
                "gt_track": gt_track, "gt_gap": ts.gap_m if ts else None, "gt_closing": ts.closing_mps if ts else None,
                "gt_lat": ts.lateral_m if ts else None, "cam": step % cam_every == 0,
            }  # fmt: skip
            rows.append(row)
            # stop conditions: contact (+0.5 s) or standstill for 4 s
            if collisions and t - next(r["t"] for r in rows if r["collisions"]) > 0.5:
                break
            if v < 0.05 and t > 2.0:
                stopped_since = t if stopped_since is None else stopped_since
                if t - stopped_since > 4.0:
                    break
            else:
                stopped_since = None
        for _ in range(3):
            s.tick()
        trans = list(sm.transitions)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{controller}__{cond}__{int(ego_kmh)}kmh"
    with (log_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_np) + "\n")
    meta = {
        "scenario": name,
        "kind": sc["kind"],
        "hazard": sc["kind"] in HAZARD_KINDS,
        "brake_needed": name not in NO_BRAKE_NEEDED,
        "controller": controller,
        "condition": cond,
        "ego_kmh": ego_kmh,
        "weather": sc["weather"],
        "cam_every": cam_every,
        "camera_hz": 1 / (dt * cam_every),
        "stale_reject_s": stale_s,
        "perception_timeout_s": perception_timeout_s,
        "latency_model": {k: v for k, v in vars(model).items() if not k.startswith("_") and k != "samples"}
        | {"n_samples": len(model.samples or [])},
        "world": {
            "map": cfg["carla"].get("map") or "Town10HD_Opt",
            "spawn_xyz_yaw": [
                round(sp.location.x, 3),
                round(sp.location.y, 3),
                round(sp.location.z, 3),
                round(sp.rotation.yaw, 3),
            ],
            "seed": cfg["carla"]["seed"],
            "fixed_delta_s": dt,
            "settle_s": v1.SETTLE_S,
            "initial_speed_mps": ego_mps,
        },  # fmt: skip
        "model": "yolo26n sha256:9b09cc8b (GPU FP32 detections; latency model sets availability time)",
        "transitions": trans[:50],
        "pipeline": {
            "frames": sum(1 for r in rows if r["cam"]),
            "jobs": len(pipe.records),
            "dropped_frames": pipe.n_dropped_frames,
            "gate": vars(gate.stats) | {"ages_s": None},
            "age_mean_s": float(np.mean(gate.stats.ages_s)) if gate.stats.ages_s else None,
            "age_p95_s": float(np.percentile(gate.stats.ages_s, 95)) if gate.stats.ages_s else None,
        },  # fmt: skip
    }
    return meta | summarize(rows, collisions, dt, name)


def summarize(rows, collisions, dt, name) -> dict:
    contact = next((i for i, r in enumerate(rows) if r["collisions"]), None)
    onsets = [i for i in range(1, len(rows)) if rows[i]["brake"] > 0 and rows[i - 1]["brake"] == 0]
    if rows and rows[0]["brake"] > 0:
        onsets = [0, *onsets]
    first = onsets[0] if onsets else None
    stop_i = next((i for i in range(first or 0, len(rows)) if first is not None and rows[i]["ego_mps"] < 0.05), None)
    # minimum gap of the approach phase only: up to contact, the first stop, or the moment the target is
    # passed (gap < 0 without contact, e.g. after a V2 release) - later gaps are behind the ego
    end = len(rows)
    for lim in (contact, stop_i + 1 if stop_i is not None else None):
        if lim is not None:
            end = min(end, lim)
    passed = next((i for i, r in enumerate(rows[:end]) if r["gt_gap"] is not None and r["gt_gap"] < 0), None)
    if passed is not None and contact is None:
        end = passed
    gaps = [r["gt_gap"] for r in rows[:end] if r["gt_gap"] is not None]
    resumed = stop_i is not None and any(r["ego_mps"] > 2.0 for r in rows[stop_i:])
    ttc = None
    if first is not None and rows[first]["gt_closing"] and rows[first]["gt_closing"] > 0.05:
        ttc = rows[first]["gt_gap"] / rows[first]["gt_closing"]
    # tracking quality on camera ticks with the GT target in the image (v2 only)
    cam_rows = [r for r in rows if r["cam"] and r["gt_in_img"] and r.get("state") is not None]
    ids = [r["gt_track"] for r in cam_rows if r["gt_track"] is not None]
    gaps_run, longest = 0, 0
    for r in cam_rows:
        gaps_run = gaps_run + 1 if r["gt_track"] is None else 0
        longest = max(longest, gaps_run)
    rate_err = [abs(r["gt_track_rate"] + r["gt_closing"]) for r in rows
                if r.get("gt_track_rate") is not None and r["gt_closing"] is not None]  # fmt: skip
    rng_err = [abs(r["gt_track_range"] - r["gt_gap"]) for r in rows
               if r.get("gt_track_range") is not None and r["gt_gap"] is not None and r["gt_gap"] > 0]  # fmt: skip
    return {
        "collision": bool(collisions),
        "collision_with": collisions[0]["other"] if collisions else None,
        "collision_t": rows[contact]["t"] if contact is not None else None,
        "min_gap_m": min(gaps) if gaps else None,
        "first_brake_t": rows[first]["t"] if first is not None else None,
        "gt_gap_at_first_brake_m": rows[first]["gt_gap"] if first is not None else None,
        "gt_ttc_at_first_brake_s": ttc,
        "brake_onsets": len(onsets),
        "stopped": stop_i is not None,
        "stop_t": rows[stop_i]["t"] if stop_i is not None else None,
        "resumed_after_stop": resumed,
        "braking_distance_m": sum(r["ego_mps"] * dt for r in rows[first:stop_i])
        if first is not None and stop_i
        else None,
        "final_speed_mps": rows[-1]["ego_mps"] if rows else None,
        "ticks": len(rows),
        "unneeded_brake": bool(onsets) and name in NO_BRAKE_NEEDED,
        "degraded_ticks": sum(1 for r in rows if not r["perception_ok"]),
        "tracking": {
            "cam_frames_gt_visible": len(cam_rows),
            "covered_frac": len(ids) / len(cam_rows) if cam_rows else None,
            "distinct_ids": len(set(ids)),
            "longest_gap_frames": longest,
            "range_mae_m": float(np.mean(rng_err)) if rng_err else None,
            "rate_mae_mps": float(np.mean(rate_err)) if rate_err else None,
            "rate_p95_abs_mps": float(np.percentile(rate_err, 95)) if rate_err else None,
        },
    }


def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


def _key(r):
    return (r["scenario"], r["controller"], r["condition"], float(r["ego_kmh"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--controllers", nargs="*", default=["v2", "v1", "aeb_off"])
    ap.add_argument("--conditions", nargs="*", default=["gpu_local"])
    ap.add_argument("--speeds-kmh", nargs="*", type=float, default=[30.0])
    ap.add_argument("--cam-every", type=int, default=3)
    ap.add_argument("--stale-s", type=float, default=0.3, help="reject results older than this at arrival")
    ap.add_argument("--perception-timeout-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/evidence/v2_scenarios.json")
    a = ap.parse_args()
    setup_logging("v2_run_scenarios")
    cfg = load_config(overrides={"carla": {"fixed_delta_seconds": 0.01}})
    models = latency_models(a.seed)
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    for _ in range(10):
        det(np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8))
    import subprocess

    git = ["git", "-C", str(resolve(".")), "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(resolve(".")), "status", "--porcelain", "src", "scripts"],
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    commit += "+dirty" if dirty else ""
    out = resolve(a.out)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"runs": []}
    done = {_key(r) for r in results["runs"]}
    for sname in a.scenarios:
        for kmh in a.speeds_kmh:
            for ctrl in a.controllers:
                for cond in a.conditions:
                    if ctrl == "aeb_off" and cond != a.conditions[0]:
                        continue
                    key = (sname, ctrl, cond, float(kmh))
                    if key in done:
                        continue
                    t0 = time.perf_counter()
                    r = run(cfg, sname, SCENARIOS[sname], ctrl, cond, models[cond], det, kmh, a.cam_every,
                            a.stale_s, a.perception_timeout_s, resolve("outputs/v2/scenario_logs"))  # fmt: skip
                    r["wall_s"] = time.perf_counter() - t0
                    r["experiment_id"] = f"v2__{sname}__{ctrl}__{cond}__{int(kmh)}"
                    r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    r["code_commit"] = commit
                    results["runs"].append(r)
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(results, indent=2, default=_np), encoding="utf-8")
                    tmp.replace(out)
                    keys = ("scenario", "controller", "condition", "ego_kmh", "collision", "min_gap_m",
                            "first_brake_t", "brake_onsets", "stopped", "resumed_after_stop",
                            "unneeded_brake")  # fmt: skip
                    print(json.dumps({k: r[k] for k in keys}), round(r["wall_s"], 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
