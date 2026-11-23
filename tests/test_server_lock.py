import json
import os

import pytest

from npu_ad.simulation import server_lock as sl


def test_lock_is_reentrant_and_blocks_other_live_pid(tmp_path, monkeypatch):
    p = sl.acquire("h", 1, tmp_path)
    assert json.loads(p.read_text())["pid"] == os.getpid()
    assert sl.acquire("h", 1, tmp_path) == p  # same process: re-entrant
    sl.release(p)
    p.write_text(json.dumps({"pid": 4242, "since": "x"}))
    monkeypatch.setattr(sl, "_alive", lambda pid: True)
    with pytest.raises(sl.ServerBusy, match="4242"):
        sl.acquire("h", 1, tmp_path)


def test_stale_lock_is_taken_over(tmp_path, monkeypatch):
    p = tmp_path / "carla_h_2.lock"
    p.write_text(json.dumps({"pid": 4243, "since": "x"}))
    monkeypatch.setattr(sl, "_alive", lambda pid: False)
    assert json.loads(sl.acquire("h", 2, tmp_path).read_text())["pid"] == os.getpid()
    sl.release(p)
    assert not p.exists()
