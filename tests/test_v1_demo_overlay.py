"""The V1 demo overlay is a read-only observer: it must render every state without CARLA."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from npu_ad.control.aeb import Decision, TargetState  # noqa: E402
from npu_ad.perception.gpu_runner import StageTimes  # noqa: E402
from npu_ad.perception.mono_range import RangeEstimate  # noqa: E402


def _info(step: int, cmd: float, applied: float, with_target: bool) -> dict:
    box = (300.0, 150.0, 340.0, 260.0)
    rng = RangeEstimate(8.0, 0.3, 9.0, False, box, 0, 0.8) if with_target else None
    cipo = TargetState(-1, "pedestrian", 8.0, 0.3, 4.0, True) if with_target else None
    return {
        "step": step,
        "t_sim": step * 0.01,
        "bgr": np.zeros((450, 800, 3), np.uint8),
        "dets": np.array([[*box, 0.8, 0]]) if with_target else np.zeros((0, 6)),
        "times": StageTimes(0.001, 0.0005, 0.003, 0.002, 0.007),
        "cipo": cipo,
        "range": rng,
        "decision": Decision(cmd, "full" if cmd else "clear"),
        "applied_brake": applied,
        "ego_mps": 8.3,
        "gt": TargetState(1, "pedestrian", 8.1, 0.2, 8.3, True) if with_target else None,
        "collisions": 0,
    }


def test_overlay_writes_video_and_key_frames(tmp_path):
    import run_v1_demo as demo

    ov = demo.Overlay(tmp_path, dt=0.01, every=2, mode="mono", condition="gpu_p95", delay_ms=22.7)
    ov(_info(0, 0.0, 0.0, with_target=False))  # nothing in path
    ov(_info(1, 0.0, 0.0, with_target=True))  # tracked, clear
    ov(_info(2, 1.0, 0.0, with_target=True))  # commanded, not yet applied (delay)
    ov(_info(3, 1.0, 1.0, with_target=True))  # braking
    shots = ov.close()
    assert shots == ["aeb_active.png", "detection.png", "final.png"]
    assert (tmp_path / "demo.mp4").stat().st_size > 0
    for s in shots:
        assert cv2.imread(str(tmp_path / s)).shape == (450, 800, 3)


def test_fmt_marks_unavailable_values():
    import run_v1_demo as demo

    assert demo._fmt(None, ".1f", "m") == "n/a"
    assert demo._fmt(float("inf"), ".1f", "m") == "n/a"
    assert demo._fmt(2.345, ".2f", "s") == "2.35 s"
