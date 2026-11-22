"""Collect a fixed, seeded CARLA image set shared by GPU and NPU evaluation and calibration.

Frames go to outputs/dataset/<weather>/<frame>.png with outputs/dataset/manifest.json.
The split (``calib`` vs ``eval``) is assigned deterministically by index and stored in the manifest,
so calibration frames are never used to report accuracy.
"""

from __future__ import annotations

import argparse
import json

import cv2

from npu_ad.config import load_config, resolve
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr
from npu_ad.simulation.traffic import spawn_traffic

WEATHERS = ["ClearNoon", "WetCloudySunset", "HardRainNoon", "ClearNight"]


def main() -> int:
    import carla

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--out", default="outputs/dataset")
    a = ap.parse_args()
    cfg = load_config()
    out = resolve(a.out)
    manifest = {"weathers": WEATHERS, "seed": cfg["carla"]["seed"], "ticks": a.ticks, "every": a.every,
                "map": None, "frames": []}
    for wi, weather in enumerate(WEATHERS):
        seed = cfg["carla"]["seed"] + wi
        with CarlaSession(cfg) as s:
            manifest["map"] = s.world.get_map().name
            s.world.set_weather(getattr(carla.WeatherParameters, weather))
            ego_idx = (cfg["ego"]["spawn_index"] + 11 * wi) % len(s.spawn_points())
            ego = s.spawn_ego(s.spawn_points()[ego_idx])
            cam = s.attach_rgb(ego)
            counts = spawn_traffic(s, a.vehicles, a.walkers, seed, exclude_index=ego_idx)
            ego.set_autopilot(True, s.tm.get_port())
            (out / weather).mkdir(parents=True, exist_ok=True)
            for t in range(a.ticks):
                fid = s.tick()
                img = s.get_image(cam, fid)
                if t >= 40 and t % a.every == 0:  # skip spawn transients
                    p = out / weather / f"{fid:08d}.png"
                    assert cv2.imwrite(str(p), image_to_bgr(img)), p
                    manifest["frames"].append({"path": str(p.relative_to(resolve("."))).replace("\\", "/"),
                                               "weather": weather, "frame": fid})
            print(weather, counts, "frames so far:", len(manifest["frames"]))
    for i, f in enumerate(manifest["frames"]):
        f["split"] = "calib" if i % 4 == 0 else "eval"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("total", len(manifest["frames"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
