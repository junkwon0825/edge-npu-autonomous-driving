"""V2 AEB: gated risk model, interpretable target-clear prediction and an explicit state machine.

Per tracked object (``TrackView``) the risk model computes:

1. Gating - track confirmed (``min_hits``), mean confidence >= ``min_conf``, range in
   (0, ``max_range_m``), filter valid, physically plausible closing speed
   (-``max_opening_mps`` <= closing <= ego speed + ``max_closing_over_ego_mps``; oncoming traffic is
   out of scope), plausible lateral speed (|v_lat| <= ``max_lat_speed_mps``).
2. TTC = range / closing, only when closing > ``min_closing_mps`` (else infinite).
3. Path conflict (constant-velocity, both axes):
   * the target overlaps the ego swept path while |y + v_y t| <= W, with
     W = ego half width + target half width + ``path_margin_m``  -> interval [t_in, t_out];
   * the ego occupies the target's longitudinal position during
     [TTC, TTC + (ego length + target length) / v_ego]                -> [t_arrive, t_exit];
   * ``collision_path`` if the intervals overlap (with ``pass_margin_s``);
     ``will_clear``      if the target leaves the path before the ego arrives;
     ``ego_passes``      if the ego leaves the conflict zone before the target enters;
     ``not_in_path``     if the target never overlaps the path;
     ``uncertain``       if lateral motion is not yet estimated: treated as collision_path when the
                         target is inside W now, otherwise as not_in_path (but it keeps WATCH).

States and brake command:
  CLEAR (0) -> WATCH (0, hazard tracked) -> WARNING (``partial_brake``) -> BRAKE (1.0) -> RELEASE
  (brake ramps down at ``release_rate_per_s``) -> CLEAR/WATCH.
Escalation is immediate. De-escalation from BRAKE needs ``min_brake_s`` in BRAKE and
``release_hold_s`` of continuously lower risk. At standstill BRAKE is held while a target is in the
path within ``standstill_range_m`` and released ``standstill_clear_s`` after the path is clear (no
permanent latch). Without fresh perception (stale/failed results) the last command is held
(``degraded`` flag): a brake is never released on missing data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CLEAR, WATCH, WARNING, BRAKE, RELEASE = "CLEAR", "WATCH", "WARNING", "BRAKE", "RELEASE"
HALF_WIDTH = {"pedestrian": 0.3, "cyclist": 0.4, "vehicle": 0.9}
LENGTH = {"pedestrian": 0.6, "cyclist": 1.8, "vehicle": 4.5}


@dataclass
class RiskParams:
    ego_half_width_m: float = 1.0
    ego_length_m: float = 4.8
    path_margin_m: float = 0.3
    ttc_brake_s: float = 1.6  # same thresholds as V1 so differences come from estimation/logic
    ttc_warn_s: float = 2.6
    ttc_watch_s: float = 5.0
    min_gap_m: float = 4.0
    partial_brake: float = 0.5
    min_hits: int = 3
    min_conf: float = 0.35
    max_range_m: float = 60.0
    min_closing_mps: float = 0.3
    max_closing_over_ego_mps: float = 6.0
    max_opening_mps: float = 30.0
    max_lat_speed_mps: float = 8.0
    min_lat_track_s: float = 0.25  # lateral velocity trusted after this track age
    min_lat_speed_mps: float = 0.5  # below: zero (ego-yaw noise ~0.4 m/s at 30 m; no ego-motion compensation yet)
    pass_margin_s: float = 0.3
    min_brake_s: float = 0.4
    release_hold_s: float = 0.3
    release_ttc_margin_s: float = 1.0  # TTC must exceed ttc_warn_s + margin before WARNING/BRAKE release
    release_rate_per_s: float = 2.0
    standstill_mps: float = 0.2
    standstill_range_m: float = 30.0  # hold while an in-path target is this close (8 m caused stop-and-go)
    standstill_clear_s: float = 1.0


@dataclass
class TrackView:
    track_id: int
    kind: str
    conf: float
    hits: int
    age_s: float
    range_m: float | None
    rate_mps: float | None  # d range / dt (negative = closing)
    rate_valid: bool
    lateral_m: float | None
    lat_rate_mps: float | None
    lat_valid: bool


@dataclass
class Assessment:
    track_id: int
    kind: str
    gated: str  # "ok" or reason for rejection
    path: str = "n/a"
    ttc_s: float = math.inf
    range_m: float | None = None
    closing_mps: float | None = None
    t_in: float | None = None
    t_out: float | None = None
    lateral_m: float | None = None
    path_half_width_m: float | None = None


@dataclass
class AebOutput:
    state: str
    brake: float
    reason: str
    target_id: int | None = None
    ttc_s: float | None = None
    range_m: float | None = None
    closing_mps: float | None = None
    path: str | None = None
    degraded: bool = False
    assessments: list[Assessment] = field(default_factory=list)


def path_interval(y: float, vy: float, w: float) -> tuple[float, float] | None:
    """Times t >= 0 with |y + vy t| <= w (None if never)."""
    if abs(vy) < 1e-6:
        return (0.0, math.inf) if abs(y) <= w else None
    t1, t2 = sorted(((-w - y) / vy, (w - y) / vy))
    if t2 < 0:
        return None
    return max(t1, 0.0), t2


def assess(tv: TrackView, ego_v: float, p: RiskParams) -> Assessment:
    w = p.ego_half_width_m + HALF_WIDTH.get(tv.kind, 0.9) + p.path_margin_m
    a = Assessment(tv.track_id, tv.kind, "ok", range_m=tv.range_m, lateral_m=tv.lateral_m, path_half_width_m=w)
    if tv.hits < p.min_hits:
        a.gated = "tentative"
    elif tv.conf < p.min_conf:
        a.gated = "low_conf"
    elif tv.range_m is None or not (0.0 < tv.range_m < p.max_range_m) or tv.lateral_m is None:
        a.gated = "no_range"
    elif not tv.rate_valid or tv.rate_mps is None:
        a.gated = "rate_invalid"
    else:
        closing = -tv.rate_mps
        if closing > ego_v + p.max_closing_over_ego_mps or closing < -p.max_opening_mps:
            a.gated = "implausible_closing"
        elif tv.lat_rate_mps is not None and abs(tv.lat_rate_mps) > p.max_lat_speed_mps:
            a.gated = "implausible_lateral"
    if a.gated != "ok":
        return a
    closing = -tv.rate_mps
    a.closing_mps = closing
    a.ttc_s = tv.range_m / closing if closing > p.min_closing_mps else math.inf
    lat_known = tv.lat_valid and tv.age_s >= p.min_lat_track_s and tv.lat_rate_mps is not None
    if not lat_known:  # lateral motion unknown: only "inside the path now" counts
        a.path = "uncertain"
        if abs(tv.lateral_m) <= w:
            a.t_in, a.t_out = 0.0, math.inf
        return a
    vy = tv.lat_rate_mps if abs(tv.lat_rate_mps) >= p.min_lat_speed_mps else 0.0
    iv = path_interval(tv.lateral_m, vy, w)
    if iv is None:
        a.path = "not_in_path"
        return a
    a.t_in, a.t_out = iv
    if math.isinf(a.ttc_s):
        a.path = "in_path_not_closing" if iv[0] == 0.0 else "not_in_path"
        return a
    t_arrive = a.ttc_s
    t_exit = a.ttc_s + (p.ego_length_m + LENGTH.get(tv.kind, 4.5)) / max(ego_v, 0.5)
    if iv[1] < t_arrive - p.pass_margin_s:
        a.path = "will_clear"
    elif iv[0] > t_exit + p.pass_margin_s:
        a.path = "ego_passes"
    else:
        a.path = "collision_path"
    return a


class AebStateMachine:
    def __init__(self, params: RiskParams | None = None):
        self.p = params or RiskParams()
        self.state = CLEAR
        self.brake = 0.0
        self._since = 0.0  # time the current state was entered
        self._lower_since: float | None = None
        self._clear_since: float | None = None
        self._last_t: float | None = None
        self.transitions: list[tuple[float, str, str, str]] = []

    def _go(self, t: float, new: str, reason: str) -> None:
        if new != self.state:
            self.transitions.append((t, self.state, new, reason))
            self.state, self._since, self._lower_since = new, t, None

    def _desired(self, ass: list[Assessment]) -> tuple[str, Assessment | None, str]:
        p = self.p
        crit = [a for a in ass if a.gated == "ok" and a.path in ("collision_path", "uncertain")]
        # uncertain lateral motion: conservative only when the target is inside the path already
        crit = [a for a in crit if a.path == "collision_path" or a.t_in == 0.0]
        best = min(crit, key=lambda a: (a.ttc_s, a.range_m or math.inf), default=None)
        if best is not None:
            closing = best.closing_mps or 0.0
            if best.ttc_s <= p.ttc_brake_s or (best.range_m < p.min_gap_m and closing > p.min_closing_mps):
                return BRAKE, best, "ttc_brake" if best.ttc_s <= p.ttc_brake_s else "min_gap"
            if best.ttc_s <= p.ttc_warn_s:
                return WARNING, best, "ttc_warn"
            if best.ttc_s <= p.ttc_watch_s:
                return WATCH, best, "ttc_watch"
        watch = [a for a in ass if a.gated == "ok" and a.path in ("will_clear", "ego_passes", "uncertain")
                 or (a.gated == "ok" and a.path == "in_path_not_closing"
                     and (a.range_m or 99) < self.p.standstill_range_m)]  # fmt: skip
        if watch:
            return WATCH, min(watch, key=lambda a: a.range_m or math.inf), "target_near_path"
        return CLEAR, None, "clear"

    def step(self, t: float, tracks: list[TrackView], ego_v: float, perception_ok: bool) -> AebOutput:
        p = self.p
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        if not perception_ok:  # no fresh perception: hold the last command, never release on missing data
            return AebOutput(self.state, self.brake, "degraded_hold", degraded=True)
        ass = [assess(tv, ego_v, p) for tv in tracks]
        desired, best, why = self._desired(ass)
        rank = {CLEAR: 0, WATCH: 1, RELEASE: 1, WARNING: 2, BRAKE: 3}
        standstill = ego_v < p.standstill_mps
        # the AEB brought the car to a stop (in any braking state, partial braking can stop it too):
        # hold it while an in-path object is close, release only after the path stayed clear
        if standstill and (self.state in (BRAKE, WARNING) or (self.state == RELEASE and self.brake > 0)):
            blocking = [a for a in ass if a.gated in ("ok", "rate_invalid") and a.range_m is not None
                        and a.range_m < p.standstill_range_m and a.lateral_m is not None
                        and abs(a.lateral_m) <= a.path_half_width_m]  # fmt: skip
            if blocking:
                self._clear_since = None
                why = "standstill_hold"
                desired = BRAKE
            else:
                self._clear_since = t if self._clear_since is None else self._clear_since
                if t - self._clear_since < p.standstill_clear_s:
                    desired, why = BRAKE, "standstill_wait_clear"
        else:
            self._clear_since = None
        if rank[desired] >= rank.get(self.state, 0) and desired != self.state:
            if not (self.state == RELEASE and desired in (WATCH, CLEAR)):
                self._go(t, desired, why)
        elif self.state in (BRAKE, WARNING) and rank[desired] < rank[self.state]:
            # TTC hysteresis: our own braking raises TTC; release only with a clear margin or when the
            # target is no longer on a collision course (otherwise: brake -> release -> creep -> brake)
            margin_ok = (
                best is None or best.path != "collision_path" or best.ttc_s > p.ttc_warn_s + p.release_ttc_margin_s
            )
            if not margin_ok:
                self._lower_since = None
                if self.state == BRAKE and desired != BRAKE:
                    why = "brake_hold_ttc_margin"
                else:
                    why = "warn_hold_ttc_margin"
            else:
                self._lower_since = t if self._lower_since is None else self._lower_since
                dwell_ok = self.state != BRAKE or t - self._since >= p.min_brake_s
                if dwell_ok and t - self._lower_since >= p.release_hold_s:
                    self._go(t, RELEASE, "risk_dropped:" + why)
        elif self.state in (WATCH, CLEAR) and desired != self.state:
            self._go(t, desired, why)
        if self.state == BRAKE:
            self.brake = 1.0
        elif self.state == WARNING:
            self.brake = p.partial_brake
        elif self.state == RELEASE:
            self.brake = max(0.0, self.brake - p.release_rate_per_s * dt)
            if self.brake <= 0.0:
                self._go(t, desired if desired in (WATCH, CLEAR) else WATCH, "released")
        else:
            self.brake = 0.0
        tgt = best
        return AebOutput(
            self.state, self.brake, why, tgt.track_id if tgt else None, tgt.ttc_s if tgt else None,
            tgt.range_m if tgt else None, tgt.closing_mps if tgt else None, tgt.path if tgt else None,
            assessments=ass,
        )  # fmt: skip


def path_relative(tv: TrackView, path_coef: tuple[float, float] | None) -> TrackView:
    """V3: express a track relative to the predicted ego path y_p(x) = a*x^2 + b*x (ego frame).

    With a curved road, an in-lane object ahead has a large lateral offset in the ego frame; subtracting the
    predicted path (perceived lane centre line, or yaw-rate / speed) keeps the straight-corridor logic valid.
    Lateral velocity is left as the yaw-compensated ego-frame value (``view_of``): for an ego following the
    path, a static object on the path then has ~0 lateral velocity (adding a geometric path term on top
    double-counts the motion - caught by a unit test).
    """
    if path_coef is None or tv.range_m is None or tv.lateral_m is None:
        return tv
    a, b = path_coef
    x = tv.range_m
    lat = tv.lateral_m - (a * x * x + b * x)
    return TrackView(tv.track_id, tv.kind, tv.conf, tv.hits, tv.age_s, tv.range_m, tv.rate_mps, tv.rate_valid, lat,
                     tv.lat_rate_mps, tv.lat_valid)  # fmt: skip


def view_of(tr, t: float, ego_yaw_rate: float = 0.0) -> TrackView:
    """``Track`` (npu_ad.perception.tracker) -> ``TrackView`` at control time ``t``.

    ``ego_yaw_rate`` (rad/s, > 0 turning right; V3 ego-motion compensation): a static object x metres
    ahead appears to move laterally by -x * yaw_rate in the ego frame; this is added back.
    """
    r = tr.rng
    rng = None
    if r.x is not None:
        rng = float(r.x[0] + r.x[1] * max(t - (r._t or t), 0.0))  # predicted to control time
    if getattr(tr, "side_truncated", False) and tr.lat.x is not None:
        # clipped at a side border: freeze the lateral position, no lateral motion from a biased centre
        lat, vlat = float(tr.lat.x[0]), 0.0
    else:
        lat, vlat = tr.lat.predict(t), (None if tr.lat.x is None else float(tr.lat.x[1]))
        if vlat is not None and rng is not None:
            vlat += rng * ego_yaw_rate
    return TrackView(tr.track_id, tr.kind, tr.conf, tr.hits, tr.age_s, rng, r.rate, r.valid, lat, vlat,
                     tr.lat.n >= 3)  # fmt: skip
