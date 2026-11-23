"""Exclusive ownership of a live CARLA server by one process.

A second synchronous-mode client ticks the same world and silently corrupts a running experiment
(observed 2026-11-23: a live pytest during the v3 matrix). Every real ``CarlaSession`` therefore
acquires ``outputs/locks/carla_<host>_<port>.lock`` before connecting. The lock is held for the
lifetime of the owning process (released at exit), so a scenario matrix keeps the server between
runs. A lock whose PID is no longer alive is stale and is taken over.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

from npu_ad import PROJECT_ROOT


class ServerBusy(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:  # pragma: no cover - psutil is in the lock file of the venv
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


_HELD: dict[Path, bool] = {}


def acquire(host: str, port: int, lock_dir: Path | None = None) -> Path:
    lock_dir = lock_dir or PROJECT_ROOT / "outputs" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"carla_{host}_{port}.lock"
    if _HELD.get(path):
        return path
    me = {"pid": os.getpid(), "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                owner = {"pid": -1}
            if owner.get("pid") == os.getpid():
                _HELD[path] = True
                return path
            if owner.get("pid", -1) > 0 and _alive(owner["pid"]):
                msg = f"CARLA {host}:{port} is owned by pid {owner['pid']} since {owner.get('since')}"
                raise ServerBusy(msg) from None
            path.unlink(missing_ok=True)  # stale
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(me, fh)
        _HELD[path] = True
        atexit.register(release, path)
        return path
    raise ServerBusy(f"could not acquire {path}")


def release(path: Path) -> None:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        if owner.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    _HELD.pop(path, None)
