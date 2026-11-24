"""V1 portfolio demo: one closed-loop CARLA run with a perception-driven AEB and an on-screen overlay.

Loop (identical code path to the v3 matrix, ``run_scenarios.run``):
CARLA RGB camera -> YOLO26n on the local GPU -> monocular range + CIPO tracker -> AEB v2 ->
measured GPU p95 latency as actuation delay -> CARLA vehicle control.

The NPU is NOT in this loop (see README "Known limitations"). The overlay is drawn by a read-only
observer and does not influence control. Ground truth is shown in a separate, labelled line for
evaluation only; the controller never sees it in ``mono`` mode.

Prerequisite: a CARLA 0.9.16 server started separately, e.g.
    C:\\CARLA_0.9.16\\CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2000

Usage:
    python scripts/run_v1_demo.py                       # pedestrian_crossing, mono, gpu_p95, 30 km/h
    python scripts/run_v1_demo.py --scenario lead_vehicle_braking
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_scenarios as rs  # noqa: E402

from npu_ad.config import load_config, resolve  # noqa: E402
from npu_ad.logutil import setup_logging  # noqa: E402
from npu_ad.perception.gpu_runner import GpuDetector  # noqa: E402

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, AMBER, RED, GREY = (255, 255, 255), (80, 200, 80), (0, 190, 255), (60, 60, 230), (170, 170, 170)


def check_carla(cfg: dict) -> str:
    import carla

    c = carla.Client(cfg["carla"]["host"], cfg["carla"]["port"])
    c.set_timeout(10.0)
    try:
        return f"server {c.get_server_version()}, map {c.get_world().get_map().name}"
    except RuntimeError as e:
        raise SystemExit(
            f"CARLA not reachable at {cfg['carla']['host']}:{cfg['carla']['port']} ({e}).\n"
            "Start it first: C:\\CARLA_0.9.16\\CarlaUE4.exe -RenderOffScreen -carla-rpc-port=2000"
        ) from e


def _fmt(v: float | None, spec: str, unit: str) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:{spec}} {unit}"


class Overlay:
    """Draws detections and the AEB state; writes an MP4 and keeps key frames for screenshots."""

    def __init__(self, out_dir: Path, dt: float, every: int, mode: str, condition: str, delay_ms: float):
        self.dir, self.every, self.mode, self.cond, self.delay_ms = out_dir, every, mode, condition, delay_ms
        self.fps_video = 1.0 / (dt * every)
        self.writer = None
        self.last_wall: float | None = None
        self.loop_fps: float | None = None
        self.shots: dict[str, np.ndarray] = {}
        self.first_track: np.ndarray | None = None

    def __call__(self, i: dict) -> None:
        now = time.perf_counter()
        if self.last_wall is not None:
            inst = 1.0 / max(now - self.last_wall, 1e-6)
            self.loop_fps = inst if self.loop_fps is None else 0.9 * self.loop_fps + 0.1 * inst
        self.last_wall = now
        img = i["bgr"].copy()
        rng, cipo, dec = i["range"], i["cipo"], i["decision"]
        for d in i["dets"]:
            x1, y1, x2, y2, conf, cls = d[:6]
            is_cipo = rng is not None and np.allclose(d[:4], rng.box)
            col = RED if is_cipo and i["applied_brake"] > 0 else (AMBER if is_cipo else GREEN)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            label = f"{CLASS_NAMES.get(int(cls), int(cls))} {conf:.2f}" + (" CIPO" if is_cipo else "")
            cv2.putText(img, label, (int(x1), max(12, int(y1) - 4)), FONT, 0.45, col, 1, cv2.LINE_AA)
        gap = cipo.gap_m if cipo else None
        clos = cipo.closing_mps if cipo else None
        ttc = gap / clos if gap is not None and clos is not None and clos > 0.05 else None
        brake = i["applied_brake"]
        aeb_txt = f"{dec.reason} (cmd {dec.brake:.2f}, applied {brake:.2f})"
        lines = [
            (f"Loop FPS (wall): {_fmt(self.loop_fps, '.1f', 'Hz')}   sim t: {i['t_sim']:.2f} s", WHITE),
            (
                f"Inference GPU eager: model {i['times'].model * 1e3:.2f} ms, e2e {i['times'].e2e * 1e3:.2f} ms"
                " (GPU shared with CARLA)",
                WHITE,
            ),
            (f"Control delay: {self.delay_ms:.1f} ms ({self.cond})   AEB input: {self.mode}", WHITE),
            (
                "Target: "
                + (f"{CLASS_NAMES.get(rng.cls, rng.cls)}  conf {rng.conf:.2f}" if rng is not None else "none in path"),
                WHITE,
            ),
            (
                f"Range (mono): {_fmt(gap, '.1f', 'm')}   Closing: {_fmt(clos, '.1f', 'm/s')}"
                f"   TTC: {_fmt(ttc, '.2f', 's')}",
                WHITE,
            ),
            (f"AEB: {aeb_txt}", RED if brake > 0 else (AMBER if dec.brake > 0 else GREEN)),
            (f"Vehicle speed: {i['ego_mps'] * 3.6:.1f} km/h   collisions: {i['collisions']}", WHITE),
        ]
        gt = i["gt"]
        if gt is not None:
            lines.append((f"GT (evaluation only, not used by controller): gap {gt.gap_m:.1f} m", GREY))
        pad, lh = 8, 18
        cv2.rectangle(img, (0, 0), (600, pad + lh * len(lines) + 4), (20, 20, 20), -1)
        for k, (t, col) in enumerate(lines):
            cv2.putText(img, t, (pad, pad + 12 + lh * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if brake > 0:
            w, h = img.shape[1], img.shape[0]
            cv2.rectangle(img, (w - 170, h - 46), (w - 8, h - 8), RED, -1)
            cv2.putText(img, "AEB BRAKE", (w - 160, h - 20), FONT, 0.7, WHITE, 2)
        # key frames: highest-confidence CIPO detection before braking, first applied brake, final frame
        if cipo is not None and dec.brake == 0 and self.first_track is None:
            self.first_track = img  # first frame with an in-path track and a closing-speed estimate
        if brake > 0 and "aeb_active" not in self.shots:
            self.shots["aeb_active"] = img
        self.shots["final"] = img
        if i["step"] % self.every == 0:
            if self.writer is None:
                h, w = img.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps_video, (w, h)
                )
            self.writer.write(img)

    def close(self) -> list[str]:
        if self.writer is not None:
            self.writer.release()
        if self.first_track is not None:
            self.shots["detection"] = self.first_track
        saved = []
        for k, im in self.shots.items():
            p = self.dir / f"{k}.png"
            cv2.imwrite(str(p), im)
            saved.append(p.name)
        return sorted(saved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="pedestrian_crossing", choices=list(rs.SCENARIOS))
    ap.add_argument("--mode", default="mono", choices=rs.MODES, help="mono = perception-driven (default)")
    ap.add_argument("--condition", default="gpu_p95", help="latency condition from configs/latency_conditions.json")
    ap.add_argument("--speed-kmh", type=float, default=rs.DEFAULT_EGO_KMH)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out-dir", default="outputs/v1/demo")
    ap.add_argument("--video-every", type=int, default=2, help="write every N-th tick (2 -> real-time at dt 0.01)")
    a = ap.parse_args()
    setup_logging("run_v1_demo")
    cfg = load_config(overrides={"carla": {"fixed_delta_seconds": a.dt}})
    print("1/4 CARLA:", check_carla(cfg), flush=True)
    lat_cfg = json.loads(resolve("configs/latency_conditions.json").read_text("utf-8"))
    conds = {c["name"]: c for c in lat_cfg["conditions"]}
    c = conds[a.condition]
    delay_ticks = None if c["delay_ms"] is None else math.ceil(round(c["delay_ms"] / 1000 / a.dt, 6))
    m = cfg["model"]
    det = GpuDetector(str(resolve(m["weights"])), m["imgsz"], m["conf"], m["iou"], m["max_det"])
    for _ in range(10):
        det(np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8))
    print(f"2/4 YOLO26n on GPU ready (conf {m['conf']}, NMS IoU {m['iou']})", flush=True)
    out = resolve(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ov = Overlay(out, a.dt, a.video_every, a.mode, a.condition, c["delay_ms"] or 0.0)
    t0 = time.perf_counter()
    print(f"3/4 running {a.scenario} / {a.mode} / {a.condition} / {a.speed_kmh:.0f} km/h", flush=True)
    try:
        r = rs.run(
            cfg,
            a.scenario,
            rs.SCENARIOS[a.scenario],
            a.condition,
            delay_ticks,
            det,
            out / "logs",
            "v2",
            a.mode,
            a.speed_kmh,
            "v1_demo",
            0,
            on_tick=ov,
        )
    finally:
        shots = ov.close()
    r["wall_s"] = time.perf_counter() - t0
    r["artifacts"] = {"video": "demo.mp4", "screenshots": shots, "per_tick_log": "logs/"}
    r["note"] = "closed loop on local GPU; NPU not in the loop; GT used for evaluation only"
    (out / "summary.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    keys = ("collision", "min_gap_m", "first_applied_brake_t", "gt_ttc_at_first_brake_s", "final_speed_mps")
    print("4/4 done:", json.dumps({k: r.get(k) for k in keys}), f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
