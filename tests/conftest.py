import socket

import pytest


def _carla_up(host: str = "localhost", port: int = 2000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if _carla_up():
        return
    skip = pytest.mark.skip(reason="CARLA server not reachable on localhost:2000")
    for item in items:
        if "carla" in item.keywords:
            item.add_marker(skip)
