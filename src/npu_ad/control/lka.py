"""V3 lane keeping (LKA) and longitudinal/lateral arbitration.

LKA: pure pursuit on the perceived lane centre line (``LaneEstimate``).
* look-ahead L = clip(lookahead_s * v, 6 m, 20 m)
* latency compensation: the estimate was made at capture time; the ego has since travelled d = v * age
  and turned by dpsi (vehicle odometry) -> target lateral y_t = y_c(L + d) - L * dpsi
* pure pursuit: curvature k = 2 * y_t / L^2, steering angle delta = atan(wheelbase * k)
* command = delta / max_steer_angle, saturated to +-max_steer_cmd, rate-limited to rate_limit_per_s
* lane estimate older than ``lane_timeout_s`` -> LKA unavailable (no steering from stale geometry)

Arbitration (control precedence, documented in docs/v3_architecture.md):
1. AEB braking (WARNING/BRAKE): throttle 0, AEB brake, steering = LKA output clamped to +-brake_steer_limit
   (no swerving while braking).
2. LKA unavailable: steering decays to 0 at decay_per_s, throttle 0 (coast) - degraded mode.
3. Otherwise: LKA steering + speed controller throttle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from npu_ad.perception.lane_geometry import LaneEstimate


@dataclass
class LkaParams:
    wheelbase_m: float = 2.875
    max_steer_angle_rad: float = math.radians(70.0)
    max_steer_cmd: float = 0.5
    rate_limit_per_s: float = 1.5
    lookahead_s: float = 0.8
    lookahead_min_m: float = 6.0
    lookahead_max_m: float = 20.0
    lane_timeout_s: float = 0.5
    min_confidence: float = 0.05


@dataclass
class LkaOutput:
    steer: float
    active: bool
    reason: str
    lookahead_m: float = 0.0
    target_lateral_m: float = 0.0


class LaneKeeping:
    def __init__(self, params: LkaParams | None = None):
        self.p = params or LkaParams()
        self.steer = 0.0
        self._t: float | None = None

    def step(self, t: float, lane: LaneEstimate | None, lane_capture_t: float | None, v: float,
             yaw_now: float, yaw_at_capture: float | None) -> LkaOutput:  # fmt: skip
        p = self.p
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        if lane is None or not lane.valid or lane_capture_t is None or t - lane_capture_t > p.lane_timeout_s:
            return LkaOutput(self.steer, False, "lane_unavailable")
        if lane.confidence < p.min_confidence:
            return LkaOutput(self.steer, False, "low_confidence")
        age = max(t - lane_capture_t, 0.0)
        d = v * age
        dpsi = 0.0 if yaw_at_capture is None else math.atan2(math.sin(yaw_now - yaw_at_capture),
                                                            math.cos(yaw_now - yaw_at_capture))  # fmt: skip
        look = min(max(p.lookahead_s * v, p.lookahead_min_m), p.lookahead_max_m)
        y_t = lane.lateral_at(look + d) - look * dpsi
        delta = math.atan(p.wheelbase_m * 2.0 * y_t / (look * look))
        target = max(-p.max_steer_cmd, min(p.max_steer_cmd, delta / p.max_steer_angle_rad))
        step = p.rate_limit_per_s * dt if dt > 0 else abs(target - self.steer)
        self.steer += max(-step, min(step, target - self.steer))
        return LkaOutput(self.steer, True, "tracking", look, y_t)


@dataclass
class ArbitrationParams:
    brake_steer_limit: float = 0.1
    decay_per_s: float = 0.5


@dataclass
class ControlCommand:
    throttle: float
    brake: float
    steer: float
    mode: str  # NORMAL | AEB_PRIORITY | LKA_DEGRADED | AEB_AND_LKA_DEGRADED


class Arbiter:
    def __init__(self, params: ArbitrationParams | None = None):
        self.p = params or ArbitrationParams()
        self._steer = 0.0
        self._t: float | None = None

    def combine(self, t: float, aeb_brake: float, lka: LkaOutput, speed_throttle: float) -> ControlCommand:
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        if lka.active:
            steer = lka.steer
        else:  # degraded: decay the last steering command towards 0
            d = self.p.decay_per_s * dt
            steer = self._steer - max(-d, min(d, self._steer))
        if aeb_brake > 0:
            steer = max(-self.p.brake_steer_limit, min(self.p.brake_steer_limit, steer))
            mode = "AEB_PRIORITY" if lka.active else "AEB_AND_LKA_DEGRADED"
            cmd = ControlCommand(0.0, aeb_brake, steer, mode)
        elif not lka.active:
            cmd = ControlCommand(0.0, 0.0, steer, "LKA_DEGRADED")
        else:
            cmd = ControlCommand(speed_throttle, 0.0, steer, "NORMAL")
        self._steer = cmd.steer
        return cmd
