"""Explicit scene state (V3): the only input of the control layer.

Raw model outputs (boxes, masks) never reach the controllers directly; perception modules fill a
``SceneState`` with timestamped, age-checked quantities and health information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from npu_ad.control.aeb_v2 import TrackView
from npu_ad.perception.lane_geometry import LaneEstimate


@dataclass
class SourceHealth:
    backend: str = "n/a"  # e.g. "rngd bf16", "gpu fp32"
    last_capture_t: float | None = None
    last_result_t: float | None = None
    age_s: float | None = None
    stale_rejected: int = 0
    failures: int = 0
    fallback_state: str = "n/a"


@dataclass
class SceneState:
    t: float
    ego_speed_mps: float
    ego_yaw_rad: float
    ego_yaw_rate_rps: float
    lane: LaneEstimate | None = None
    lane_capture_t: float | None = None
    lane_yaw_at_capture: float | None = None
    tracks: list[TrackView] = field(default_factory=list)
    closest_hazard_id: int | None = None
    ttc_s: float | None = None
    road_condition: str = "unknown"  # not perceived in V3 (documented limitation)
    detection: SourceHealth = field(default_factory=SourceHealth)
    lane_source: SourceHealth = field(default_factory=SourceHealth)
    radar_used: bool = False

    @property
    def perception_ok(self) -> bool:
        return self.detection.age_s is not None and self.detection.age_s <= 0.5

    def lane_age(self) -> float | None:
        return None if self.lane_capture_t is None else self.t - self.lane_capture_t
