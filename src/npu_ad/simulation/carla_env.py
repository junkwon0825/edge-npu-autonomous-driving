"""Synchronous CARLA session with tracked actor lifecycle.

Design notes (see reports/carla_baseline.md):
* Synchronous mode + fixed_delta_seconds makes physics steps deterministic per tick; the Traffic
  Manager is put in synchronous mode with a fixed seed.
* Each ``tick()`` returns the world frame id; ``get_image(frame)`` only accepts the sensor image
  of that exact frame and drops stale ones. The original learning script
  ``code/09_yolo_detection.py`` called ``camera.listen`` twice, which can deliver duplicate
  images per tick and desynchronise a single queue.
* Only actors spawned by this session are destroyed, sensors first, and the original world
  settings are restored, also on exceptions.
"""

from __future__ import annotations

import logging
import queue
import random
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def image_to_bgr(image: Any) -> np.ndarray:
    """carla.Image (BGRA uint8) -> contiguous HxWx3 BGR array (copied, safe after callback)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return np.ascontiguousarray(arr[:, :, :3])


class FrameTimeout(RuntimeError):
    pass


class CarlaSession:
    def __init__(self, cfg: dict[str, Any], carla_module: Any = None):
        if carla_module is None:
            import carla as carla_module  # imported lazily so unit tests can inject a fake
        self.carla = carla_module
        self.cfg = cfg
        self.client = None
        self.world = None
        self.tm = None
        self._original_settings = None
        self._actors: list[Any] = []  # spawn order
        self._sensor_queues: dict[int, queue.Queue] = {}

    # ---- lifecycle -------------------------------------------------------------------------
    def __enter__(self) -> CarlaSession:
        c = self.cfg["carla"]
        self.client = self.carla.Client(c["host"], c["port"])
        self.client.set_timeout(c["timeout_s"])
        server, client_v = self.client.get_server_version(), self.client.get_client_version()
        if server != client_v:
            log.warning("CARLA server %s != client %s", server, client_v)
        self.world = self.client.get_world()
        if c.get("map") and not self.world.get_map().name.endswith(c["map"]):
            self.world = self.client.load_world(c["map"])
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = c["fixed_delta_seconds"]
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager(c["tm_port"])
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(c["seed"])
        random.seed(c["seed"])
        self.world.tick()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.world is None:
            return
        for actor in reversed(self._actors):
            try:
                if getattr(actor, "type_id", "").startswith("controller.ai.walker"):
                    actor.stop()
                elif hasattr(actor, "is_listening") and actor.is_listening:
                    actor.stop()
            except Exception as e:  # keep tearing down
                log.warning("stop failed for %s: %s", getattr(actor, "id", "?"), e)
        ids = [a.id for a in reversed(self._actors)]
        if ids:
            cmd = self.carla.command.DestroyActor
            for resp in self.client.apply_batch_sync([cmd(i) for i in ids], True):
                if resp.error:
                    log.warning("destroy error: %s", resp.error)
        self._actors.clear()
        self._sensor_queues.clear()
        try:
            if self.tm is not None:
                self.tm.set_synchronous_mode(False)
            if self._original_settings is not None:
                self.world.apply_settings(self._original_settings)
        finally:
            self.world = None

    @property
    def actor_ids(self) -> list[int]:
        return [a.id for a in self._actors]

    # ---- spawning --------------------------------------------------------------------------
    def spawn_points(self) -> list[Any]:
        return self.world.get_map().get_spawn_points()

    def spawn(self, blueprint_id: str, transform: Any, attach_to: Any = None, **attrs: Any) -> Any:
        bp = self.world.get_blueprint_library().find(blueprint_id)
        for k, v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
        if attach_to is None:
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                raise RuntimeError(f"spawn failed (collision at spawn point?) for {blueprint_id}")
        else:
            actor = self.world.spawn_actor(
                bp, transform, attach_to=attach_to, attachment_type=self.carla.AttachmentType.Rigid
            )
        self._actors.append(actor)
        return actor

    def spawn_ego(self, transform: Any = None) -> Any:
        e = self.cfg["ego"]
        if transform is None:
            transform = self.spawn_points()[e["spawn_index"]]
        return self.spawn(e["blueprint"], transform, role_name="hero")

    def attach_rgb(self, parent: Any) -> Any:
        cam = self.cfg["camera"]
        tf = self.carla.Transform(self.carla.Location(x=cam["x"], z=cam["z"]))
        sensor = self.spawn(
            "sensor.camera.rgb",
            tf,
            attach_to=parent,
            image_size_x=cam["width"],
            image_size_y=cam["height"],
            fov=cam["fov"],
            sensor_tick=0.0,
        )
        q: queue.Queue = queue.Queue()
        sensor.listen(q.put)  # exactly one listener per sensor
        self._sensor_queues[sensor.id] = q
        return sensor

    def attach_collision(self, parent: Any, events: list) -> Any:
        sensor = self.spawn("sensor.other.collision", self.carla.Transform(), attach_to=parent)
        sensor.listen(
            lambda ev: events.append(
                {"frame": ev.frame, "other": ev.other_actor.type_id, "impulse": ev.normal_impulse.length()}
            )
        )
        return sensor

    # ---- stepping --------------------------------------------------------------------------
    def tick(self) -> int:
        return self.world.tick()

    def get_image(self, sensor: Any, frame: int, timeout: float | None = None) -> Any:
        """Return the sensor image for ``frame``; older images are dropped."""
        timeout = self.cfg["camera"]["frame_timeout_s"] if timeout is None else timeout
        q = self._sensor_queues[sensor.id]
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FrameTimeout(f"no image for frame {frame} within {timeout}s")
            try:
                img = q.get(timeout=remaining)
            except queue.Empty as e:
                raise FrameTimeout(f"no image for frame {frame} within {timeout}s") from e
            if img.frame == frame:
                return img
            if img.frame > frame:
                raise FrameTimeout(f"expected frame {frame}, got newer {img.frame}")
            # older frame: drop and keep waiting
