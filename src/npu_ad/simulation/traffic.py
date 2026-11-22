"""Seeded background traffic (NPC vehicles on autopilot, AI-controlled walkers)."""

from __future__ import annotations

import random
from typing import Any

from npu_ad.simulation.carla_env import CarlaSession


def spawn_traffic(s: CarlaSession, n_vehicles: int, n_walkers: int, seed: int,
                  exclude_index: int | None = None) -> dict[str, int]:
    rng = random.Random(seed)
    lib = s.world.get_blueprint_library()
    vbps = sorted((bp.id for bp in lib.filter("vehicle.*")
                   if bp.has_attribute("number_of_wheels")
                   and int(bp.get_attribute("number_of_wheels")) == 4), key=str)
    points = list(enumerate(s.spawn_points()))
    rng.shuffle(points)
    nv = 0
    for idx, tf in points:
        if nv >= n_vehicles:
            break
        if idx == exclude_index:
            continue
        try:
            v = s.spawn(rng.choice(vbps), tf)
        except RuntimeError:
            continue
        v.set_autopilot(True, s.tm.get_port())
        nv += 1
    s.world.set_pedestrians_seed(seed)
    wbps = sorted(bp.id for bp in lib.filter("walker.pedestrian.*"))
    nw = 0
    import carla

    for _ in range(n_walkers * 3):
        if nw >= n_walkers:
            break
        loc = s.world.get_random_location_from_navigation()
        if loc is None:
            continue
        try:
            w = s.spawn(rng.choice(wbps), carla.Transform(loc + carla.Location(z=1.0)))
        except RuntimeError:
            continue
        ctrl: Any = s.spawn("controller.ai.walker", carla.Transform(), attach_to=w)
        s.tick()
        ctrl.start()
        ctrl.go_to_location(s.world.get_random_location_from_navigation())
        ctrl.set_max_speed(1.4)
        nw += 1
    return {"vehicles": nv, "walkers": nw}
