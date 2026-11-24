"""Range / range-rate estimators for monocular measurements with irregular timing (V2).

All filters take ``update(t, z)`` with a range measurement ``z`` (metres) at time ``t`` (seconds;
intervals may vary, results arrive asynchronously) and expose ``range`` and ``rate`` (d range / dt,
negative while closing). ``valid`` turns true once the estimate is usable.

* ``FiniteDifference`` - raw (z_k - z_{k-1}) / dt; V2 baseline for the noise problem.
* ``WindowLsq``        - V1 estimator: least-squares slope over a 0.3 s window.
* ``AlphaBeta``        - fixed-gain alpha-beta tracker.
* ``RangeKalman``      - constant-velocity Kalman filter with range-dependent measurement noise
                         (flat-ground range error grows ~ Z^2 / (f * h) per pixel of box-bottom
                         error) and innovation gating (outliers rejected, not averaged in).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FiniteDifference:
    min_samples: int = 2
    range: float | None = None
    rate: float | None = None
    _t: float | None = None
    n: int = 0

    @property
    def valid(self) -> bool:
        return self.rate is not None

    def update(self, t: float, z: float) -> None:
        if self._t is not None and t > self._t and self.range is not None:
            self.rate = (z - self.range) / (t - self._t)
        self.range, self._t = z, t
        self.n += 1


@dataclass
class WindowLsq:
    window_s: float = 0.3
    min_samples: int = 5
    max_residual_m: float = 0.5
    range: float | None = None
    rate: float | None = None
    _h: deque = field(default_factory=deque)

    @property
    def valid(self) -> bool:
        return self.rate is not None

    def update(self, t: float, z: float) -> None:
        self._h.append((t, z))
        while self._h and t - self._h[0][0] > self.window_s:
            self._h.popleft()
        self.range = z
        if len(self._h) < self.min_samples:
            return
        ts, zs = np.array(self._h).T
        a, b = np.polyfit(ts - ts[-1], zs, 1)
        if np.std(zs - (a * (ts - ts[-1]) + b)) <= self.max_residual_m:
            self.rate = float(a)


@dataclass
class AlphaBeta:
    alpha: float = 0.5
    beta: float = 0.1
    max_dt: float = 0.5
    range: float | None = None
    rate: float | None = None
    _t: float | None = None
    n: int = 0
    min_samples: int = 3

    @property
    def valid(self) -> bool:
        return self.n >= self.min_samples

    def update(self, t: float, z: float) -> None:
        if self.range is None or self._t is None or t - self._t > self.max_dt:
            self.range, self.rate, self._t, self.n = z, 0.0, t, 1
            return
        dt = max(t - self._t, 1e-3)
        pred = self.range + self.rate * dt
        r = z - pred
        self.range = pred + self.alpha * r
        self.rate = self.rate + self.beta / dt * r
        self._t = t
        self.n += 1


@dataclass
class RangeKalman:
    """State x = [range, rate]; white-acceleration process noise ``sigma_a`` (m/s^2).

    Measurement std: sqrt(sigma_r0^2 + (sigma_px * z^2 / (fy * cam_h))^2). An update whose
    normalised innovation squared exceeds ``gate_nis`` is rejected (counted in ``n_rejected``);
    after ``max_rejects`` consecutive rejections the filter re-initialises on the measurement.
    """

    fy: float = 400.0
    cam_h: float = 1.7
    sigma_px: float = 0.75  # V2 filter eval (reports/evidence/v2_filter_eval.json)
    sigma_r0: float = 0.15
    sigma_a: float = 15.0
    gate_nis: float = 25.0  # 5 sigma: only gross outliers (wrong box) are rejected
    max_rejects: int = 3
    max_dt: float = 0.5
    min_samples: int = 3
    x: np.ndarray | None = None
    P: np.ndarray | None = None
    _t: float | None = None
    n: int = 0
    n_rejected: int = 0
    _consec_rej: int = 0
    last_nis: float = 0.0

    def meas_std(self, z: float) -> float:
        return math.hypot(self.sigma_r0, self.sigma_px * z * z / (self.fy * self.cam_h))

    @property
    def range(self) -> float | None:
        return None if self.x is None else float(self.x[0])

    @property
    def rate(self) -> float | None:
        return None if self.x is None else float(self.x[1])

    @property
    def rate_std(self) -> float | None:
        return None if self.P is None else float(math.sqrt(self.P[1, 1]))

    @property
    def valid(self) -> bool:
        return self.n >= self.min_samples

    def _init(self, t: float, z: float) -> None:
        s = self.meas_std(z)
        self.x = np.array([z, 0.0])
        self.P = np.diag([s * s, 15.0**2])  # unknown rate: +-15 m/s
        self._t, self.n, self._consec_rej = t, 1, 0

    def update(self, t: float, z: float) -> None:
        if self.x is None or self._t is None or t - self._t > self.max_dt:
            self._init(t, z)
            return
        dt = max(t - self._t, 1e-3)
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.sigma_a**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        x = f @ self.x
        p = f @ self.P @ f.T + q
        r = self.meas_std(z) ** 2
        s = p[0, 0] + r
        innov = z - x[0]
        self.last_nis = innov * innov / s
        self._t = t
        if self.last_nis > self.gate_nis and self.n >= self.min_samples:
            self.n_rejected += 1
            self._consec_rej += 1
            self.x, self.P = x, p  # predict only
            if self._consec_rej >= self.max_rejects:
                self._init(t, z)
            return
        k = p[:, 0] / s
        self.x = x + k * innov
        self.P = p - np.outer(k, p[0, :])
        self.n += 1
        self._consec_rej = 0


def make_filter(kind: str, **kw):
    return {"fd": FiniteDifference, "lsq": WindowLsq, "ab": AlphaBeta, "kf": RangeKalman}[kind](**kw)
