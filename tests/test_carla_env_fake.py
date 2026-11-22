"""CarlaSession lifecycle against an in-memory fake of the carla module (no server needed)."""

from types import SimpleNamespace

import pytest

from npu_ad.config import load_config
from npu_ad.simulation.carla_env import CarlaSession, FrameTimeout


class FakeActor:
    _next = 100

    def __init__(self, type_id):
        FakeActor._next += 1
        self.id = FakeActor._next
        self.type_id = type_id
        self.is_listening = False
        self.cb = None

    def listen(self, cb):
        self.is_listening, self.cb = True, cb

    def stop(self):
        self.is_listening = False


class FakeBP:
    def __init__(self, i):
        self.id, self.attrs = i, {}

    def has_attribute(self, k):
        return True

    def set_attribute(self, k, v):
        self.attrs[k] = v


class FakeWorld:
    def __init__(self):
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        self.applied = []
        self.frame = 0
        self.actors = {}

    def get_settings(self):
        return SimpleNamespace(**vars(self.settings))

    def apply_settings(self, s):
        self.applied.append(s)
        self.settings = s

    def get_map(self):
        return SimpleNamespace(name="Carla/Maps/Town10HD_Opt", get_spawn_points=lambda: ["sp0", "sp1"])

    def tick(self):
        self.frame += 1
        for a in self.actors.values():
            if a.cb and a.type_id.startswith("sensor.camera"):
                a.cb(SimpleNamespace(frame=self.frame))
        return self.frame

    def get_blueprint_library(self):
        return SimpleNamespace(find=FakeBP)

    def _spawn(self, bp):
        a = FakeActor(bp.id)
        self.actors[a.id] = a
        return a

    def try_spawn_actor(self, bp, tf):
        return self._spawn(bp)

    def spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
        return self._spawn(bp)


class FakeClient:
    def __init__(self, world):
        self.world, self.destroyed = world, []

    def set_timeout(self, t):
        pass

    def get_server_version(self):
        return "0.9.16"

    def get_client_version(self):
        return "0.9.16"

    def get_world(self):
        return self.world

    def get_trafficmanager(self, port):
        return SimpleNamespace(
            set_synchronous_mode=lambda v: None, set_random_device_seed=lambda s: None, get_port=lambda: port
        )

    def apply_batch_sync(self, cmds, do_tick):
        out = []
        for c in cmds:
            self.destroyed.append(c)
            self.world.actors.pop(c, None)
            out.append(SimpleNamespace(error=""))
        return out


def make_fake():
    world = FakeWorld()
    client = FakeClient(world)
    mod = SimpleNamespace(
        Client=lambda h, p: client,
        Transform=lambda *a, **k: "tf",
        Location=lambda **k: "loc",
        AttachmentType=SimpleNamespace(Rigid="rigid"),
        command=SimpleNamespace(DestroyActor=lambda i: i),
    )
    return mod, world, client


def test_session_sets_sync_and_restores_and_destroys_sensors_first():
    mod, world, client = make_fake()
    cfg = load_config()
    with CarlaSession(cfg, carla_module=mod) as s:
        assert world.settings.synchronous_mode is True
        assert world.settings.fixed_delta_seconds == cfg["carla"]["fixed_delta_seconds"]
        ego = s.spawn_ego()
        cam = s.attach_rgb(ego)
        img = s.get_image(cam, s.tick())
        assert img.frame == world.frame
    assert client.destroyed == [cam.id, ego.id]  # sensor before vehicle
    assert world.actors == {}
    assert world.settings.synchronous_mode is False  # original restored
    assert not cam.is_listening


def test_get_image_drops_stale_frames_and_times_out():
    mod, world, _ = make_fake()
    with CarlaSession(load_config(), carla_module=mod) as s:
        cam = s.attach_rgb(s.spawn_ego())
        s.tick()
        s.tick()  # two images queued; ask for the newest
        f = s.tick()
        assert s.get_image(cam, f).frame == f
        with pytest.raises(FrameTimeout):
            s.get_image(cam, f + 5, timeout=0.05)


def test_teardown_runs_on_exception():
    mod, world, client = make_fake()
    with pytest.raises(ValueError), CarlaSession(load_config(), carla_module=mod) as s:
        s.spawn_ego()
        raise ValueError("boom")
    assert world.actors == {} and world.settings.synchronous_mode is False
