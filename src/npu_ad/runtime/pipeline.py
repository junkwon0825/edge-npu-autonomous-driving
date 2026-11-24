"""Asynchronous, frame-age-aware perception pipeline (V2 runtime).

Two executions of the same design:

* ``AsyncInferenceWorker`` - real threads, wall clock: camera thread -> ``LatestFrameQueue`` ->
  inference worker -> latest result -> control loop. Used for live GPU / live remote-NPU runs.
* ``SimulatedAsyncPipeline`` - the same queue semantics advanced in *simulation time* with a seeded
  latency model (constant, jitter, bursts, dropped and stale results). Used for the deterministic
  CARLA matrix: the worker can only start a new frame when the previous one has finished, so a
  slow backend also lowers the perception rate (V1's actuation-delay model ignored that).

The control side consumes results through ``ResultGate``: a result older than ``max_age_s`` at
control time is rejected as stale and counted; it is never used as current perception.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from npu_ad.runtime.backends import InferenceResult

log = logging.getLogger("npu_ad.runtime")


@dataclass
class Frame:
    frame_id: int
    capture_t: float
    bgr: np.ndarray | None
    enqueue_t: float = 0.0


class LatestFrameQueue:
    """Bounded queue that keeps the newest frames; a full queue drops its oldest frame."""

    def __init__(self, maxsize: int = 1):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._q: deque[Frame] = deque()
        self._cv = threading.Condition()
        self.n_put = self.n_dropped = 0
        self.closed = False

    def put(self, frame: Frame) -> int:
        with self._cv:
            dropped = 0
            while len(self._q) >= self.maxsize:
                self._q.popleft()
                dropped += 1
            self._q.append(frame)
            self.n_put += 1
            self.n_dropped += dropped
            self._cv.notify()
            return dropped

    def get(self, timeout: float | None = None) -> Frame | None:
        with self._cv:
            if not self._q and not self.closed:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()


@dataclass
class GateStats:
    n_new: int = 0
    n_stale: int = 0
    n_failed: int = 0
    n_none: int = 0  # control ticks without any result yet
    ages_s: list[float] = field(default_factory=list)


class ResultGate:
    """Hands the control loop only fresh, successful, not-yet-consumed results."""

    def __init__(self, max_age_s: float):
        self.max_age_s = max_age_s
        self.last_id = -1
        self.stats = GateStats()

    def accept(self, res: InferenceResult | None, control_t: float) -> tuple[InferenceResult | None, str]:
        """Returns (result or None, reason) with reason in new|repeat|stale|failed|none."""
        if res is None:
            self.stats.n_none += 1
            return None, "none"
        if res.frame_id <= self.last_id:
            return None, "repeat"
        self.last_id = res.frame_id
        age = control_t - res.capture_t
        if not res.ok:
            self.stats.n_failed += 1
            return None, "failed"
        if age > self.max_age_s:
            self.stats.n_stale += 1
            return None, "stale"
        self.stats.n_new += 1
        self.stats.ages_s.append(age)
        return res, "new"


class AsyncInferenceWorker:
    """Background thread: newest frame -> backend -> newest result (wall clock)."""

    def __init__(self, backend, queue: LatestFrameQueue, clock: Callable[[], float] = time.perf_counter):
        self.backend, self.queue, self.clock = backend, queue, clock
        self._latest: InferenceResult | None = None
        self._lock = threading.Lock()
        self.records: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="inference-worker")

    def start(self) -> AsyncInferenceWorker:
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            fr = self.queue.get(timeout=0.1)
            if fr is None:
                continue
            t_start = self.clock()
            res = self.backend.infer(fr.frame_id, fr.capture_t, fr.bgr)
            res.t_start, res.t_end = t_start, self.clock()
            with self._lock:
                self._latest = res
            self.records.append(
                {
                    "frame_id": fr.frame_id,
                    "capture_t": fr.capture_t,
                    "enqueue_t": fr.enqueue_t,
                    "t_start": t_start,
                    "t_end": res.t_end,
                    "queue_delay_s": t_start - fr.enqueue_t,
                    "latency_s": res.latency_s,
                    "ok": res.ok,
                    "backend": res.backend,
                    "error": res.error,
                    "stages": res.stages,
                }
            )

    def latest(self) -> InferenceResult | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self.queue.close()
        self._t.join(timeout=5)


@dataclass
class LatencyModel:
    """Seeded latency/failure model in seconds (simulation-time pipeline).

    ``samples`` (optional) is an empirical latency distribution (e.g. measured remote-NPU end-to-end
    times) drawn with replacement; otherwise ``base_s`` +- uniform ``jitter_s``. Every
    ``burst_period_s`` a burst of length ``burst_len_s`` adds ``burst_extra_s``. ``drop_p`` loses a
    result entirely; ``stale_p`` delays a result by ``stale_extra_s`` (beyond the age limit).
    """

    base_s: float = 0.0
    jitter_s: float = 0.0
    samples: list[float] | None = None
    burst_period_s: float = 0.0
    burst_len_s: float = 0.0
    burst_extra_s: float = 0.0
    drop_p: float = 0.0
    stale_p: float = 0.0
    stale_extra_s: float = 0.0
    seed: int = 0
    use_measured: bool = False  # V3 hardware-in-the-loop: use the real call's measured latency
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def draw(self, t_start: float) -> tuple[float | None, str]:
        """Latency for a job starting at ``t_start``; None if the result is dropped."""
        r = self._rng
        lat = r.choice(self.samples) if self.samples else self.base_s + r.uniform(-self.jitter_s, self.jitter_s)
        lat = max(lat, 0.0)
        tag = "ok"
        if self.burst_period_s and (t_start % self.burst_period_s) < self.burst_len_s:
            lat += self.burst_extra_s
            tag = "burst"
        u = r.random()
        if u < self.drop_p:
            return None, "dropped"
        if u < self.drop_p + self.stale_p:
            return lat + self.stale_extra_s, "stale_injected"
        return lat, tag


class SimulatedAsyncPipeline:
    """Latest-frame queue + one worker, advanced in simulation time.

    ``on_frame(t, frame)`` at every camera frame; ``poll(t)`` returns the newest result whose
    simulated completion time is <= t. The detector itself runs when the job *starts* (its wall
    time does not matter); the result is released at ``start + latency``.
    """

    def __init__(self, backend, model: LatencyModel, maxsize: int = 1, min_latency_s: float = 0.0):
        self.backend, self.model = backend, model
        self.queue = LatestFrameQueue(maxsize)
        self.min_latency_s = min_latency_s
        self._busy_until: float | None = None
        self._pending: list[tuple[float, InferenceResult, str]] = []  # (ready_t, result, tag)
        self._latest: InferenceResult | None = None
        self.records: list[dict] = []

    def _start(self, t: float, fr: Frame) -> None:
        res = self.backend.infer(fr.frame_id, fr.capture_t, fr.bgr)
        lat, tag = self.model.draw(t)
        if self.model.use_measured and lat is not None:
            lat, tag = res.latency_s, "measured"  # simulation time paused during the real call
        res.t_start = t
        compute = max(lat if lat is not None else self.model.base_s, self.min_latency_s)
        self._busy_until = t + compute
        rec = {"frame_id": fr.frame_id, "capture_t": fr.capture_t, "t_start": t, "queue_delay_s": t - fr.enqueue_t}
        rec |= {"latency_s": lat, "tag": tag, "ok": res.ok}
        self.records.append(rec)
        if lat is not None:
            res.t_end = t + max(lat, self.min_latency_s)
            res.latency_s = res.t_end - t
            self._pending.append((res.t_end, res, tag))

    def _advance(self, t: float) -> None:
        while True:
            if self._busy_until is not None and self._busy_until <= t:
                free_at = self._busy_until
                self._busy_until = None
                fr = self.queue.get(timeout=0)
                if fr is not None:
                    self._start(free_at, fr)
                    continue
            break
        ready = [p for p in self._pending if p[0] <= t]
        if ready:
            self._pending = [p for p in self._pending if p[0] > t]
            newest = max(ready, key=lambda p: p[1].frame_id)[1]
            if self._latest is None or newest.frame_id > self._latest.frame_id:
                self._latest = newest

    def on_frame(self, t: float, frame: Frame) -> None:
        self._advance(t)
        frame.enqueue_t = t
        if self._busy_until is None:
            self._start(t, frame)
        else:
            self.queue.put(frame)
        self._advance(t)

    def poll(self, t: float) -> InferenceResult | None:
        self._advance(t)
        return self._latest

    @property
    def n_dropped_frames(self) -> int:
        return self.queue.n_dropped
