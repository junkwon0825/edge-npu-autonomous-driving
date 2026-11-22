"""Phase 1 / Gate B: repeatable CARLA capture smoke test.

Per run: connect -> spawn ego (autopilot) -> attach RGB camera -> tick N frames -> save frames ->
verify count, dimensions and frame-id continuity -> tear down -> verify no orphaned actors.
Also usable to build an image set (``--weather``, ``--spawn-index``) for calibration/evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from npu_ad.config import load_config, resolve
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr


def run_once(cfg: dict, frames: int, save_every: int, out_dir: Path, weather: str | None,
             spawn_index: int) -> dict:
    import carla

    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"saved": [], "frame_ids": [], "shape_ok": True, "errors": []}
    t0 = time.perf_counter()
    with CarlaSession(cfg) as s:
        if weather:
            s.world.set_weather(getattr(carla.WeatherParameters, weather))
        ego = s.spawn_ego(s.spawn_points()[spawn_index % len(s.spawn_points())])
        cam = s.attach_rgb(ego)
        ego.set_autopilot(True, s.tm.get_port())
        rec["actor_ids"] = s.actor_ids
        for _ in range(5):  # sensor warm-up ticks
            s.get_image(cam, s.tick())
        for i in range(frames):
            fid = s.tick()
            img = s.get_image(cam, fid)
            bgr = image_to_bgr(img)
            rec["frame_ids"].append(fid)
            if bgr.shape != (cfg["camera"]["height"], cfg["camera"]["width"], 3):
                rec["shape_ok"] = False
            if i % save_every == 0:
                p = out_dir / f"{fid:08d}.png"
                if not cv2.imwrite(str(p), bgr):
                    rec["errors"].append(f"imwrite failed {p}")
                rec["saved"].append(str(p))
        world = s.world
        ours = set(s.actor_ids)
    # verify teardown with a fresh world handle
    alive = {a.id for a in world.get_actors()} & ours
    rec["orphans"] = sorted(alive)
    reread_ok = all(cv2.imread(p) is not None for p in rec["saved"])
    ids = rec["frame_ids"]
    rec.update({
        "frames": len(ids),
        "contiguous": all(b - a == 1 for a, b in zip(ids, ids[1:], strict=False)),
        "saved_count": len(rec["saved"]),
        "reread_ok": reread_ok,
        "elapsed_s": time.perf_counter() - t0,
    })
    rec["pass"] = bool(len(ids) == frames and rec["shape_ok"] and rec["contiguous"] and reread_ok
                       and not rec["orphans"] and not rec["errors"])
    rec["frame_ids"] = [ids[0], ids[-1]] if ids else []
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default="outputs/capture")
    ap.add_argument("--weather", default=None, help="carla.WeatherParameters preset name")
    ap.add_argument("--spawn-index", type=int, default=None)
    ap.add_argument("--report", default="reports/evidence/capture_smoke.json")
    a = ap.parse_args()
    cfg = load_config()
    runs = []
    for r in range(a.runs):
        idx = a.spawn_index if a.spawn_index is not None else cfg["ego"]["spawn_index"] + 7 * r
        rec = run_once(cfg, a.frames, a.save_every, resolve(a.out) / f"run{r:02d}", a.weather, idx)
        rec["run"], rec["spawn_index"] = r, idx
        print(json.dumps({k: rec[k] for k in ("run", "frames", "saved_count", "contiguous",
                                                "orphans", "pass", "elapsed_s")}))
        runs.append({k: v for k, v in rec.items() if k != "saved"})
    summary = {"runs": runs, "all_pass": all(r["pass"] for r in runs), "config": cfg["camera"],
               "fixed_delta_seconds": cfg["carla"]["fixed_delta_seconds"], "weather": a.weather}
    out = resolve(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL_PASS" if summary["all_pass"] else "FAILED")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
