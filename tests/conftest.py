import os
import socket

import pytest

# Live CARLA tests are opt-in: a test that connects while an experiment holds the server in
# synchronous mode would tick the shared world and corrupt that experiment (happened 2026-11-23).
LIVE = os.environ.get("NPU_AD_LIVE_CARLA") == "1"


def _carla_up(host: str = "localhost", port: int = 2000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    if LIVE and _carla_up():
        return
    reason = "set NPU_AD_LIVE_CARLA=1 to run live CARLA tests" if not LIVE else "CARLA server not reachable"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "carla" in item.keywords:
            item.add_marker(skip)
