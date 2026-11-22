"""Live CARLA integration test (skipped automatically when no server is running)."""

import pytest

from npu_ad.config import load_config
from npu_ad.simulation.carla_env import CarlaSession, image_to_bgr


@pytest.mark.carla
def test_spawn_capture_teardown_twice():
    cfg = load_config()
    for _ in range(2):
        with CarlaSession(cfg) as s:
            ego = s.spawn_ego()
            cam = s.attach_rgb(ego)
            ids = set(s.actor_ids)
            shapes = {image_to_bgr(s.get_image(cam, s.tick())).shape for _ in range(10)}
            world = s.world
        assert shapes == {(cfg["camera"]["height"], cfg["camera"]["width"], 3)}
        assert not ({a.id for a in world.get_actors()} & ids)
