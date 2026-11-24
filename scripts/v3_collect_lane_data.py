"""V3: collect a lane-marking segmentation dataset in CARLA (RGB + pixel-aligned RoadLine mask).

An RGB camera and a semantic-segmentation camera share the driving camera's pose and intrinsics
(800x450, FOV 90, x 1.5 m, z 1.7 m). The ego is driven by the traffic manager. Every ``--every``-th tick
one pair is stored, resized to the model input (512x256): ``rgb/<id>.jpg`` and ``mask/<id>.png``
(255 = RoadLine). Maps, weathers and the seed are recorded in ``manifest.json``.

  python scripts/v3_collect_lane_data.py --maps Town10HD_Opt Town04 Town03 --per-map 900
"""

from __future__ import annotations

import argparse
import json
import random
import time

import cv2
import numpy as np

from npu_ad.config import load_config, resolve
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr

WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "ClearNight", "WetCloudySunset"]


def roadline_tag(carla) -> int:
    for name in ("RoadLines", "RoadLine"):
        if hasattr(carla.CityObjectLabel, name):
            return int(getattr(carla.CityObjectLabel, name))
    raise RuntimeError("RoadLine label not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="*", default=["Town10HD_Opt", "Town04", "Town03"])
    ap.add_argument("--per-map", type=int, default=900)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default="outputs/v3/lane_dataset")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    import carla

    out = resolve(a.out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    tag = roadline_tag(carla)
    rng = random.Random(a.seed)
    manifest: dict = {"roadline_tag": tag, "size": [512, 256], "every": a.every, "seed": a.seed, "samples": []}
    for town in a.maps:
        cfg = load_config(overrides={"carla": {"fixed_delta_seconds": 0.05, "map": town}})
        with CarlaSession(cfg) as s:
            sps = s.spawn_points()
            ego = None
            for sp in rng.sample(sps, min(20, len(sps))):
                try:
                    ego = s.spawn_ego(sp)
                    break
                except RuntimeError:
                    continue
            cam = s.attach_rgb(ego)
            cam_cfg = cfg["camera"]
            seg = s.spawn("sensor.camera.semantic_segmentation", cam.get_transform().__class__(
                carla.Location(x=cam_cfg["x"], z=cam_cfg["z"])), attach_to=ego, image_size_x=cam_cfg["width"],
                image_size_y=cam_cfg["height"], fov=cam_cfg["fov"], sensor_tick=0.0)  # fmt: skip
            import queue

            seg_q: queue.Queue = queue.Queue()
            seg.listen(seg_q.put)
            ego.set_autopilot(True, cfg["carla"]["tm_port"])
            s.tm.ignore_lights_percentage(ego, 100)
            s.tm.random_left_lanechange_percentage(ego, 10)
            s.tm.random_right_lanechange_percentage(ego, 10)
            n, step, t0 = 0, 0, time.perf_counter()
            weather = WEATHERS[0]
            while n < a.per_map and step < a.per_map * a.every * 3:
                if step % 400 == 0:
                    weather = rng.choice(WEATHERS)
                    s.world.set_weather(getattr(carla.WeatherParameters, weather))
                fid = s.tick()
                img = s.get_image(cam, fid)
                while True:
                    sg = seg_q.get(timeout=5.0)
                    if sg.frame >= fid:
                        break
                step += 1
                if step % a.every or sg.frame != fid or ego.get_velocity().length() < 1.0:
                    continue
                bgr = image_to_bgr(img)
                labels = np.frombuffer(sg.raw_data, np.uint8).reshape(sg.height, sg.width, 4)[:, :, 2]
                mask = (labels == tag).astype(np.uint8) * 255
                sid = f"{town}_{n:05d}"
                cv2.imwrite(str(out / "rgb" / f"{sid}.jpg"), cv2.resize(bgr, (512, 256), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
                cv2.imwrite(
                    str(out / "mask" / f"{sid}.png"), cv2.resize(mask, (512, 256), interpolation=cv2.INTER_NEAREST)
                )
                manifest["samples"].append({"id": sid, "map": town, "weather": weather, "frame": fid,
                                            "roadline_px_frac": float(mask.mean() / 255)})  # fmt: skip
                n += 1
            seg.stop()
            print(town, n, "samples", round(time.perf_counter() - t0, 1), "s", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("total", len(manifest["samples"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
