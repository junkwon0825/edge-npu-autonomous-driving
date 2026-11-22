"""Conservative emergency-braking (AEB) research prototype. Not a production safety function.

Inputs are per-target *validated* states: longitudinal gap and closing speed come from CARLA
ground truth (a 2D box alone gives no reliable metric distance), and ``perceived`` tells whether
the camera detector currently confirms the target. Braking is only allowed on perceived targets,
so perception misses directly show up as late or absent braking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetState:
    actor_id: int
    kind: str  # "vehicle" | "pedestrian" | ...
    gap_m: float  # bumper-to-bumper longitudinal gap along ego heading
    lateral_m: float  # signed lateral offset of target centre in ego frame
    closing_mps: float  # >0 when the gap is shrinking
    perceived: bool


@dataclass(frozen=True)
class AebParams:
    corridor_half_width_m: float = 1.8  # ego half-width + margin
    ttc_full_s: float = 1.6
    ttc_partial_s: float = 2.6
    min_gap_m: float = 4.0  # full brake below this gap regardless of TTC (when closing)
    partial_brake: float = 0.5
    perception_hold_ticks: int = 3  # keep a target "perceived" briefly through 1-2 frame misses
    # v2: once partial braking starts for a target, hold it while that target is still closing in the
    # corridor, and hold the vehicle (full brake) below standstill_mps. v1 (False) released the brake as
    # soon as TTC rose above the threshold, which let cruise control re-accelerate towards a stopped
    # obstacle (stop-and-go creep observed in lead_vehicle_braking, see reports/scenario_evaluation.md).
    hysteresis: bool = True
    standstill_mps: float = 0.5


@dataclass
class Decision:
    brake: float
    reason: str
    target_id: int | None = None
    ttc_s: float | None = None
    gap_m: float | None = None


@dataclass
class AebController:
    params: AebParams = field(default_factory=AebParams)
    latched: bool = False
    _seen: dict[int, int] = field(default_factory=dict)
    _partial_target: int | None = None

    def decide(self, targets: list[TargetState], ego_speed_mps: float) -> Decision:
        p = self.params
        if self.latched:
            if ego_speed_mps < 0.05:
                return Decision(1.0, "latched_stopped")
            return Decision(1.0, "latched")
        present = {t.actor_id for t in targets}
        for aid in list(self._seen):  # targets absent this tick also lose their hold
            if aid not in present:
                self._seen[aid] -= 1
                if self._seen[aid] <= 0:
                    del self._seen[aid]
        for t in targets:
            self._seen[t.actor_id] = (
                p.perception_hold_ticks if t.perceived else max(0, self._seen.get(t.actor_id, 0) - 1)
            )
        best: Decision = Decision(0.0, "clear")
        for t in targets:
            if abs(t.lateral_m) > p.corridor_half_width_m or t.gap_m < -1.0:
                continue
            if self._seen.get(t.actor_id, 0) <= 0:
                continue  # not confirmed by perception -> no action (logged by caller)
            if t.closing_mps <= 0.05:
                continue
            ttc = max(t.gap_m, 0.0) / t.closing_mps
            if ttc < p.ttc_full_s or t.gap_m < p.min_gap_m:
                self.latched = True
                return Decision(1.0, "full", t.actor_id, ttc, t.gap_m)
            if ttc < p.ttc_partial_s and best.brake < p.partial_brake:
                best = Decision(p.partial_brake, "partial", t.actor_id, ttc, t.gap_m)
        if not p.hysteresis:
            return best
        if best.brake == 0.0 and self._partial_target is not None:
            held = next((t for t in targets if t.actor_id == self._partial_target), None)
            if (
                held is not None
                and abs(held.lateral_m) <= p.corridor_half_width_m
                and self._seen.get(held.actor_id, 0) > 0
                and held.closing_mps > 0.05
            ):
                best = Decision(p.partial_brake, "partial_hold", held.actor_id, None, held.gap_m)
            else:
                self._partial_target = None
        if best.brake > 0.0:
            self._partial_target = best.target_id
            if ego_speed_mps < p.standstill_mps:
                self.latched = True
                return Decision(1.0, "standstill_hold", best.target_id, best.ttc_s, best.gap_m)
        return best


class DelayLine:
    """Applies a decision ``delay_ticks`` after it was computed (models perception latency)."""

    def __init__(self, delay_ticks: int, initial: Decision | None = None):
        self.delay = max(0, int(delay_ticks))
        self.buf: deque[Decision] = deque()
        self.current = initial or Decision(0.0, "init")

    def push(self, d: Decision) -> Decision:
        self.buf.append(d)
        if len(self.buf) > self.delay:
            self.current = self.buf.popleft()
        return self.current
