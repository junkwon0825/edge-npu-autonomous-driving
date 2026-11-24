"""Remove leftover dynamic actors (vehicles, walkers, sensors) and restore asynchronous mode.

Needed after a client was force-killed (CarlaSession could not clean up). In synchronous mode the
client's actor snapshot is only refreshed by a tick, so the world is ticked once before listing
(an earlier cleanup without the tick listed 0 actors and left 12 behind). Refuses to run while a
live CarlaSession holds the server lock.
"""

from __future__ import annotations

import collections
import glob

import carla

from npu_ad.config import resolve


def main() -> int:
    if glob.glob(str(resolve("outputs/locks/carla_*.lock"))):
        raise SystemExit("a CarlaSession holds the server lock; stop it first (or remove a stale lock)")
    c = carla.Client("localhost", 2000)
    c.set_timeout(20)
    w = c.get_world()
    if w.get_settings().synchronous_mode:
        w.tick()
    dyn = [a for a in w.get_actors() if a.type_id.split(".")[0] in ("vehicle", "walker", "sensor", "controller")]
    print("leftover:", dict(collections.Counter(a.type_id for a in dyn)))
    c.apply_batch_sync([carla.command.DestroyActor(a.id) for a in dyn])
    s = w.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = False, None
    w.apply_settings(s)
    print("destroyed", len(dyn), "synchronous_mode", w.get_settings().synchronous_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
