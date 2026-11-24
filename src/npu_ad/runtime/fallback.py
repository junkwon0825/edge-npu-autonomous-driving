"""Explicit NPU -> GPU failover and NPU health monitoring (V2 runtime).

``FailoverBackend`` wraps a primary (RNGD) and a secondary (GPU) backend behind the same
``infer`` interface. States:

* ``PRIMARY``  - primary healthy, results come from the primary.
* ``FALLBACK`` - ``fail_threshold`` consecutive primary failures/timeouts; the failing frame is
                 re-run on the secondary immediately and later frames use the secondary. Every
                 ``probe_interval_s`` one frame is tried on the primary again; success restores
                 PRIMARY.
* ``DEGRADED`` - both backends fail: the result is ``ok=False`` and the controller applies its
                 safe policy (no fresh perception).

Every transition is appended to ``events`` and logged; results carry the backend that produced
them, so a device switch is never silent.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

from npu_ad.runtime.backends import InferenceResult

log = logging.getLogger("npu_ad.runtime.fallback")
PRIMARY, FALLBACK, DEGRADED = "PRIMARY", "FALLBACK", "DEGRADED"


class FailoverBackend:
    def __init__(
        self,
        primary,
        secondary,
        fail_threshold: int = 1,
        probe_interval_s: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.primary, self.secondary = primary, secondary
        self.fail_threshold, self.probe_interval_s, self.clock = fail_threshold, probe_interval_s, clock
        self.state = PRIMARY
        self.consecutive_fail = 0
        self._last_probe = -1e9
        self.events: list[dict] = []
        self.name = "failover"
        self.precision = getattr(primary, "precision", "?")

    def _transition(self, new: str, reason: str, frame_id: int) -> None:
        if new == self.state:
            return
        ev = {"t": self.clock(), "frame_id": frame_id, "from": self.state, "to": new, "reason": reason}
        self.events.append(ev)
        log.warning("backend transition", extra=ev)
        self.state = new

    def infer(self, frame_id: int, capture_t: float, bgr) -> InferenceResult:
        t0 = self.clock()
        use_primary = self.state == PRIMARY or (t0 - self._last_probe >= self.probe_interval_s)
        if use_primary:
            if self.state != PRIMARY:
                self._last_probe = t0
            res = self.primary.infer(frame_id, capture_t, bgr)
            if res.ok:
                self.consecutive_fail = 0
                self._transition(PRIMARY, "primary ok", frame_id)
                return res
            self.consecutive_fail += 1
            if self.state == PRIMARY and self.consecutive_fail < self.fail_threshold:
                return res  # below threshold: report the failure, do not switch yet
            self._transition(FALLBACK, f"primary failed: {res.error}", frame_id)
        res2 = self.secondary.infer(frame_id, capture_t, bgr)
        res2.stages = dict(res2.stages) | {"failover_total_s": self.clock() - t0}
        if not res2.ok:
            self._transition(DEGRADED, f"secondary failed: {res2.error}", frame_id)
        elif self.state == DEGRADED:
            self._transition(FALLBACK, "secondary recovered", frame_id)
        return res2

    def health(self) -> dict:
        return {
            "state": self.state,
            "consecutive_fail": self.consecutive_fail,
            "events": len(self.events),
            "primary": self.primary.health(),
            "secondary": self.secondary.health(),
        }

    def close(self) -> None:
        self.primary.close()
        self.secondary.close()


class HealthMonitor:
    """Bounded health sampling: client-side counters every call, remote device query rarely."""

    def __init__(self, backend, remote_every_s: float = 10.0, window: int = 100, clock=time.perf_counter):
        self.backend, self.remote_every_s, self.clock = backend, remote_every_s, clock
        self.lat = deque(maxlen=window)
        self.samples: list[dict] = []
        self._last_remote = -1e9

    def record(self, res: InferenceResult, queue_depth: int) -> None:
        if res.ok:
            self.lat.append(res.latency_s)
        now = self.clock()
        if now - self._last_remote < self.remote_every_s:
            return
        self._last_remote = now
        s = {"t": now, "queue_depth": queue_depth, "client": self.backend.health()}
        s["mean_latency_ms"] = 1000 * sum(self.lat) / len(self.lat) if self.lat else None
        remote = getattr(self.backend, "remote_health", None)
        if remote is not None:
            try:
                s["remote"] = remote()
            except Exception as e:  # noqa: BLE001 - health sampling must not stop the loop
                s["remote"] = {"error": repr(e)}
        self.samples.append(s)
