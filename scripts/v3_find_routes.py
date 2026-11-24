"""Find V3 lane-keeping start points by road curvature (map waypoints; geometry only, no driving).

For every spawn point, walk the lane centre 90 m ahead in 2 m steps (no junction allowed) and compute the
curvature |dyaw/ds|. Classes by the maximum curvature over the first 80 m:
  straight  < 0.002 1/m          mild  0.005-0.012 1/m (R ~ 80-200 m)       strong  0.015-0.045 1/m (R ~ 22-65 m)
Writes configs/v3_routes.json (map, spawn index, class, curvature profile summary).
"""

from __future__ import annotations

import json

import carla
import numpy as np

from npu_ad.config import resolve

CLASSES = {"straight": (0.0, 0.002), "mild": (0.005, 0.012), "strong": (0.015, 0.045)}


def profile(cmap, sp, length=90.0, step=2.0):
    wp = cmap.get_waypoint(sp.location)
    yaws, s = [wp.transform.rotation.yaw], 0.0
    while s < length:
        nxt = wp.next(step)
        if not nxt or nxt[0].is_junction:
            return None
        wp = nxt[0]
        yaws.append(wp.transform.rotation.yaw)
        s += step
    y = np.unwrap(np.radians(yaws))
    return np.abs(np.diff(y)) / step


def main() -> int:
    c = carla.Client("localhost", 2000)
    c.set_timeout(60)
    routes = []
    for town in ("Town04", "Town03", "Town10HD_Opt"):
        w = c.load_world(town) if not c.get_world().get_map().name.endswith(town) else c.get_world()
        cmap = w.get_map()
        found = {k: [] for k in CLASSES}
        for i, sp in enumerate(cmap.get_spawn_points()):
            k = profile(cmap, sp)
            if k is None:
                continue
            kmax = float(k[:40].max())
            for name, (lo, hi) in CLASSES.items():
                if lo <= kmax < hi and (name != "straight" or float(k.max()) < hi):
                    found[name].append({"map": town, "spawn_index": i, "class": name, "kappa_max": kmax,
                                        "kappa_mean": float(k[:40].mean()),
                                        "radius_min_m": (1 / kmax) if kmax > 0 else None})  # fmt: skip
        for name, lst in found.items():
            lst.sort(key=lambda r: -r["kappa_mean"] if name != "straight" else r["kappa_max"])
            routes += lst[:3]
            print(town, name, len(lst), [round(r["kappa_max"], 4) for r in lst[:3]], flush=True)
    out = resolve("configs/v3_routes.json")
    out.write_text(json.dumps({"classes": CLASSES, "routes": routes}, indent=2), encoding="utf-8")
    print(len(routes), "routes ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
