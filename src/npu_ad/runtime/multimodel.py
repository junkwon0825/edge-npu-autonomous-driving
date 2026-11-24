"""V3 multi-model runtime: detection and lane-segmentation backends, combined RNGD client, scheduler.

Interfaces (small, separate - no monolithic class):
* ``InferenceBackend`` (V2) - detection: ``infer(frame_id, capture_t, bgr) -> InferenceResult``
* ``LaneBackend``            - lane: ``infer(frame_id, capture_t, bgr) -> LaneResult``
* ``RngdMultiModelClient``   - one persistent session to ``remote/npu_server_v3.py`` (YOLO on one PE, lane
                               U-Net on another); exposes ``detection()``, ``lane()`` and ``both()`` views
                               that satisfy the interfaces above, sharing one transport.
* ``RateScheduler``          - decides per camera frame which models run (rate-decoupled / adaptive).
All results carry frame id, capture time, backend, precision, latency and ok/error.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from npu_ad.perception.lane_net import LaneUNet, preprocess
from npu_ad.runtime.backends import (
    FuriosaRemoteBackend,
    InferenceResult,
    RemoteTimeout,
    TransportBusy,
    _empty,
    _unletterbox,
)

OP_BOTH, OP_LANE, OP_CRASH = 6, 7, 8


@dataclass
class LaneResult:
    frame_id: int
    capture_t: float
    backend: str
    precision: str
    ok: bool
    mask: np.ndarray | None  # (256, 512) bool, lane-marking pixels
    latency_s: float
    t_start: float = 0.0
    t_end: float = 0.0
    error: str | None = None
    stages: dict = field(default_factory=dict)


@dataclass
class MultiResult:
    """Detection + lane results of one frame from one request (Mode D)."""

    frame_id: int
    capture_t: float
    backend: str
    precision: str
    ok: bool
    det: InferenceResult
    lane: LaneResult
    latency_s: float
    t_start: float = 0.0
    t_end: float = 0.0
    error: str | None = None


class GpuLaneBackend:
    def __init__(self, model: LaneUNet, device: str = "cuda:0", name: str = "gpu"):
        self.model, self.device, self.name, self.precision = model.eval().to(device), device, name, "fp32"

    @torch.no_grad()
    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray) -> LaneResult:
        t0 = time.perf_counter()
        try:
            x = preprocess(bgr).to(self.device)
            m = (self.model(x)[0, 0] > 0).cpu().numpy()
            return LaneResult(frame_id, capture_t, self.name, self.precision, True, m, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            return LaneResult(frame_id, capture_t, self.name, self.precision, False, None, time.perf_counter() - t0,
                              error=repr(e))  # fmt: skip


def _decode_mask(lane: dict) -> np.ndarray:
    png = base64.b64decode(lane["png_b64"])
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) > 0


class RngdMultiModelClient(FuriosaRemoteBackend):
    """Client of ``remote/npu_server_v3.py`` (same transport, writer thread and deadlines as V2)."""

    def __init__(self, models: str = "det,lane", lane_weights: str = "lane_unet_s3c32.pt", **kw):
        remote_dir = kw.pop("remote_dir", "npu_auto_driving_p2")
        env = kw.pop("env", "furiosa-npu")
        threads = kw.pop("threads", 2)
        lane_device = kw.pop("lane_device", "furiosa:1")  # "furiosa:0" = both models on one PE
        cmd = (f"cd {remote_dir} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate {env} && "
               f"exec nice -n 10 python -u npu_server_v3.py --models {models} --lane-weights {lane_weights} "
               f"--threads {threads} --lane-device {lane_device}")  # fmt: skip
        host = kw.pop("host", "furiosa-npu")
        opts = kw.pop("ssh_opts", ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-C"))
        kw.setdefault("command", ["ssh", *opts, host, f"bash -lc '{cmd}'"])
        kw.setdefault("encoding", "content_raw")
        super().__init__(dtype="bf16", **kw)

    def _request(self, op: int, frame_id: int, bgr: np.ndarray):
        t0 = time.perf_counter()
        enc, w, h, payload, lb, t_enc = self.encode(bgr)
        with self._lock:
            self._send(op, frame_id, enc, w, h, payload)
            fid, status, body, t_rx = self._wait(frame_id, self.timeout_s - (time.perf_counter() - t0))
        if status != 0:
            raise RuntimeError(body.get("error", "remote error"))
        return body, lb, t0, t_enc

    def _call(self, op: int, frame_id: int, capture_t: float, bgr: np.ndarray):
        t0 = time.perf_counter()
        try:
            body, lb, t0, t_enc = self._request(op, frame_id, bgr)
            stages = dict(body.get("t", {})) | {"encode": t_enc}
            lat = time.perf_counter() - t0
            det = lane = None
            if "dets" in body:
                d = np.asarray(body["dets"], np.float64).reshape(-1, 6)
                det = InferenceResult(frame_id, capture_t, "rngd", "bf16", True, _unletterbox(d, lb, bgr.shape[1],
                                      bgr.shape[0]), lat, stages=stages)  # fmt: skip
            if "lane" in body:
                lane = LaneResult(frame_id, capture_t, "rngd", "bf16", True, _decode_mask(body["lane"]), lat,
                                  stages=stages)  # fmt: skip
            self.n_ok += 1
            return det, lane, lat, None
        except RemoteTimeout as e:
            self.n_timeout += 1
            err = repr(e)
        except TransportBusy as e:
            self.n_busy += 1
            err = repr(e)
        except Exception as e:  # noqa: BLE001
            self.n_err += 1
            err = repr(e)
        return None, None, time.perf_counter() - t0, err

    def infer_det(self, frame_id, capture_t, bgr) -> InferenceResult:
        det, _, lat, err = self._call(1, frame_id, capture_t, bgr)
        return det or InferenceResult(frame_id, capture_t, "rngd", "bf16", False, _empty(), lat, error=err)

    def infer_lane(self, frame_id, capture_t, bgr) -> LaneResult:
        _, lane, lat, err = self._call(OP_LANE, frame_id, capture_t, bgr)
        return lane or LaneResult(frame_id, capture_t, "rngd", "bf16", False, None, lat, error=err)

    def infer_both(self, frame_id, capture_t, bgr) -> MultiResult:
        det, lane, lat, err = self._call(OP_BOTH, frame_id, capture_t, bgr)
        ok = det is not None and lane is not None
        det = det or InferenceResult(frame_id, capture_t, "rngd", "bf16", False, _empty(), lat, error=err)
        lane = lane or LaneResult(frame_id, capture_t, "rngd", "bf16", False, None, lat, error=err)
        return MultiResult(frame_id, capture_t, "rngd", "bf16", ok, det, lane, lat, error=err)

    def crash(self) -> None:
        """Fault injection: ask the server process to exit without answering."""
        with self._lock:
            self._send_blocking(OP_CRASH, 0xFFFFFFFF)

    def detection(self):
        return _View(self, "infer_det")

    def lane(self):
        return _View(self, "infer_lane")

    def both(self):
        return _View(self, "infer_both")


class _View:
    """Adapter so one client can serve as a detection, lane or combined backend."""

    def __init__(self, client: RngdMultiModelClient, method: str):
        self.client, self.method, self.name, self.precision = client, method, "rngd", "bf16"

    def infer(self, frame_id, capture_t, bgr):
        return getattr(self.client, self.method)(frame_id, capture_t, bgr)

    def health(self) -> dict:
        return self.client.health()

    def close(self) -> None:
        pass


class CoalescingDispatcher:
    """Mode D backend for one latest-frame pipeline that carries both models.

    The scheduler's per-frame decision (detection and/or lane) accumulates in ``want`` until a request
    actually starts; that request serves everything asked for since the previous one, on the newest
    frame. Choosing the operation at capture time instead aliases with the pipeline: at ~53 ms per
    request and 30 ms frames the frames that start keep the same parity, so "lane every 2nd frame" can
    starve the lane model (V3 pilot: half the lane results and repeated 0.5 s lane timeouts).
    """

    def __init__(self, view: _View):
        self.view, self.name, self.precision = view, view.name, view.precision
        self.want = {"det": False, "lane": False}
        self.ops: dict[str, int] = {"infer_both": 0, "infer_det": 0, "infer_lane": 0}

    def request(self, det: bool, lane: bool) -> bool:
        """Record what this frame needs; True if anything is pending (then offer the frame)."""
        self.want["det"] |= det
        self.want["lane"] |= lane
        return self.want["det"] or self.want["lane"]

    def infer(self, frame_id, capture_t, bgr):
        d, ln = self.want["det"], self.want["lane"]
        self.want = {"det": False, "lane": False}
        op = "infer_both" if (d and ln) or not (d or ln) else ("infer_det" if d else "infer_lane")
        self.ops[op] += 1
        self.view.method = op
        return self.view.infer(frame_id, capture_t, bgr)

    def health(self) -> dict:
        return self.view.health()

    def close(self) -> None:
        pass


@dataclass
class RateScheduler:
    """Which models run on a camera frame.

    ``mode`` fixed: detection every ``det_every`` frames, lane every ``lane_every`` frames.
    ``mode`` adaptive: detection every frame while a hazard is tracked (TTC < hazard_ttc_s), else every
    2nd frame; lane every 2nd frame normally, every frame when lane confidence is low, every 4th frame on a
    stable lane (confidence high and |offset| small).
    """

    mode: str = "fixed"
    det_every: int = 1
    lane_every: int = 2
    hazard_ttc_s: float = 3.0
    counts: dict = field(default_factory=lambda: {"det": 0, "lane": 0, "frames": 0})

    def plan(
        self, n: int, hazard_ttc: float | None, lane_conf: float | None, lane_offset: float | None
    ) -> tuple[bool, bool]:
        self.counts["frames"] += 1
        if self.mode == "fixed":
            d, ln = n % self.det_every == 0, n % self.lane_every == 0
        else:
            hazard = hazard_ttc is not None and hazard_ttc < self.hazard_ttc_s
            d = hazard or n % 2 == 0
            if lane_conf is None or lane_conf < 0.3:
                ln = True
            elif lane_conf > 0.6 and lane_offset is not None and abs(lane_offset) < 0.3 and not hazard:
                ln = n % 4 == 0
            else:
                ln = n % 2 == 0
        self.counts["det"] += d
        self.counts["lane"] += ln
        return d, ln
