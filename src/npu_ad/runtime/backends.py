"""Unified inference backends (V2 runtime).

Every backend turns one BGR camera frame into detections in image pixels and returns an
``InferenceResult`` carrying identity (backend, precision, model, input shape), timing and status.
The rest of the perception pipeline (tracker, range filter, AEB) only sees ``InferenceResult``.

* ``GpuInferenceBackend``     - local CUDA path of V1 (``GpuDetector``); baseline and fallback.
* ``FuriosaRemoteBackend``    - persistent session to ``remote/npu_server.py`` on the RNGD host.
* ``ReplayInferenceBackend``  - detections recorded earlier, looked up by frame id (open loop).
* ``MockBackend``             - deterministic latency / failure injection for tests.
"""

from __future__ import annotations

import json
import queue
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from npu_ad.perception.yolo_raw import Letterbox, letterbox

MODEL_VERSION = "yolo26n sha256:9b09cc8b"
REQ = struct.Struct("<4sBIBHHI")
RESP = struct.Struct("<4sIBI")
OP_INFER, OP_ECHO, OP_HEALTH, OP_STALL, OP_QUIT = 1, 2, 3, 4, 5
CONTROL_FID = 0xFFFFFFFF  # responses to control messages (stall) are never frame results
ENCODINGS = {"letterbox_raw": 0, "content_raw": 1, "content_jpeg": 2, "content_png": 2}


@dataclass
class InferenceResult:
    frame_id: int
    capture_t: float  # timestamp of the camera frame (pipeline clock)
    backend: str
    precision: str
    ok: bool
    dets: np.ndarray  # (N, 6) [x1, y1, x2, y2, conf, cls] in image pixels
    latency_s: float  # backend call wall time (incl. transport for remote backends)
    t_start: float = 0.0  # pipeline clock at inference start
    t_end: float = 0.0  # pipeline clock at inference end
    error: str | None = None
    stages: dict = field(default_factory=dict)  # per-stage durations (seconds) when available
    model: str = MODEL_VERSION
    input_shape: tuple = (1, 3, 640, 640)


class InferenceBackend(Protocol):
    name: str
    precision: str

    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray) -> InferenceResult: ...

    def health(self) -> dict: ...

    def close(self) -> None: ...


def _empty() -> np.ndarray:
    return np.zeros((0, 6), np.float64)


def _unletterbox(d: np.ndarray, lb: Letterbox, w: int, h: int) -> np.ndarray:
    if d.size == 0:
        return _empty()
    d = d.astype(np.float64).copy()
    d[:, [0, 2]] = ((d[:, [0, 2]] - lb.pad_x) / lb.gain).clip(0, w)
    d[:, [1, 3]] = ((d[:, [1, 3]] - lb.pad_y) / lb.gain).clip(0, h)
    return d


class GpuInferenceBackend:
    def __init__(self, detector, precision: str = "fp32", name: str = "gpu"):
        self.det, self.name, self.precision = detector, name, precision
        self.n_ok = self.n_err = 0

    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        try:
            dets, times, _ = self.det(bgr)
            self.n_ok += 1
            stages = {"pre": times.preprocess, "h2d": times.h2d, "exec": times.model, "post": times.postprocess}
            return InferenceResult(
                frame_id, capture_t, self.name, self.precision, True, dets, time.perf_counter() - t0, stages=stages
            )
        except Exception as e:  # noqa: BLE001 - surfaced as a failed result
            self.n_err += 1
            return InferenceResult(
                frame_id, capture_t, self.name, self.precision, False, _empty(), time.perf_counter() - t0, error=repr(e)
            )

    def health(self) -> dict:
        return {"backend": self.name, "n_ok": self.n_ok, "n_err": self.n_err, "available": True}

    def close(self) -> None:
        pass


class ReplayInferenceBackend:
    """Detections recorded by another backend, keyed by frame id (``{frame_id: (N,6) array}``)."""

    def __init__(self, table: dict[int, np.ndarray], name: str = "replay", precision: str = "recorded"):
        self.table, self.name, self.precision = table, name, precision

    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray | None = None) -> InferenceResult:
        if frame_id not in self.table:
            return InferenceResult(
                frame_id, capture_t, self.name, self.precision, False, _empty(), 0.0, error="missing"
            )
        return InferenceResult(frame_id, capture_t, self.name, self.precision, True, self.table[frame_id], 0.0)

    def health(self) -> dict:
        return {"backend": self.name, "frames": len(self.table), "available": True}

    def close(self) -> None:
        pass


class MockBackend:
    """Test backend: fixed detections, configurable latency and failures (``fail_ids``)."""

    def __init__(
        self, dets=None, latency_s: float = 0.0, fail_ids=(), name: str = "mock", precision: str = "none", sleep=True
    ):
        self.dets = _empty() if dets is None else np.asarray(dets, np.float64)
        self.latency_s, self.fail_ids, self.name, self.precision, self.sleep = (
            latency_s,
            set(fail_ids),
            name,
            precision,
            sleep,
        )
        self.calls: list[int] = []

    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray | None = None) -> InferenceResult:
        self.calls.append(frame_id)
        if self.sleep and self.latency_s:
            time.sleep(self.latency_s)
        ok = frame_id not in self.fail_ids
        return InferenceResult(
            frame_id,
            capture_t,
            self.name,
            self.precision,
            ok,
            self.dets if ok else _empty(),
            self.latency_s,
            error=None if ok else "injected failure",
        )

    def health(self) -> dict:
        return {"backend": self.name, "available": True}

    def close(self) -> None:
        pass


class RemoteTimeout(RuntimeError):
    pass


class TransportBusy(RuntimeError):
    pass


class FuriosaRemoteBackend:
    """Client for ``remote/npu_server.py`` over one persistent SSH session.

    ``infer`` blocks up to ``timeout_s``; a late response for an older frame id is counted as stale
    and discarded. Failures are returned as ``ok=False`` results (never silently replaced).
    """

    def __init__(
        self,
        host: str = "furiosa-npu",
        remote_dir: str = "npu_auto_driving_p2",
        env: str = "furiosa-npu",
        dtype: str = "bf16",
        imgsz: int = 640,
        encoding: str = "content_raw",
        jpeg_quality: int = 90,
        timeout_s: float = 1.0,
        startup_timeout_s: float = 600.0,
        ssh_opts: tuple[str, ...] = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20"),
        log_path: Path | None = None,
        threads: int = 0,
        prefilter: bool = True,
        command: list[str] | None = None,
    ):
        self.name, self.precision, self.imgsz = "rngd", dtype, imgsz
        self.encoding, self.jpeg_quality, self.timeout_s = encoding, jpeg_quality, timeout_s
        cmd = (
            f"cd {remote_dir} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate {env} && "
            f"exec nice -n 10 python -u npu_server.py --dtype {dtype} --imgsz {imgsz}"
            f" --threads {threads} --prefilter {int(prefilter)}"
        )
        self._err = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(
            command or ["ssh", *ssh_opts, host, f"bash -lc '{cmd}'"],  # ``command``: tests (local fake server)
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            bufsize=0,
        )
        self._lock = threading.Lock()
        self._resp: queue.Queue = queue.Queue()
        self.n_ok = self.n_err = self.n_timeout = self.n_stale = self.n_busy = 0
        # writes go through one writer thread so a blocked pipe/network (backpressure) can never block
        # infer() past its deadline; while a send is still in progress new requests are refused
        self._wq: queue.Queue = queue.Queue()
        self._sending = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self.ready = self._wait_ready(startup_timeout_s)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.input_shape = (1, 3, imgsz, imgsz)

    def _wait_ready(self, timeout_s: float) -> dict:
        box: dict = {}

        def rd():
            box["line"] = self.proc.stdout.readline()

        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(timeout_s)
        line = box.get("line", b"")
        if not line.startswith(b"READY "):
            self.close()
            raise RuntimeError(f"NPU server did not start: {line[:200]!r}")
        return json.loads(line[6:])

    def _read_loop(self) -> None:
        r = self.proc.stdout
        try:
            while True:
                hdr = r.read(RESP.size)
                while hdr is not None and 0 < len(hdr) < RESP.size:
                    more = r.read(RESP.size - len(hdr))
                    if not more:
                        break
                    hdr += more
                if not hdr or len(hdr) < RESP.size:
                    break
                magic, fid, status, n = RESP.unpack(hdr)
                body = b""
                while len(body) < n:
                    chunk = r.read(n - len(body))
                    if not chunk:
                        break
                    body += chunk
                self._resp.put((fid, status, json.loads(body), time.perf_counter()))
        finally:
            self._resp.put(None)

    def _write_loop(self) -> None:
        while True:
            buf = self._wq.get()
            if buf is None:
                return
            try:
                self.proc.stdin.write(buf)
                self.proc.stdin.flush()
            except OSError:
                return
            finally:
                self._sending.clear()

    def _send(self, op: int, fid: int, enc: int = 0, w: int = 0, h: int = 0, payload: bytes = b"") -> float:
        """Queue one request for the writer thread; raises ``TransportBusy`` if a send is pending."""
        t = time.perf_counter()
        if self._sending.is_set():
            raise TransportBusy("previous request still being sent (backpressure)")
        self._sending.set()
        self._wq.put(REQ.pack(b"NPQ1", op, fid, enc, w, h, len(payload)) + payload)
        return time.perf_counter() - t

    def _send_blocking(self, op: int, fid: int, payload: bytes = b"", timeout_s: float = 30.0) -> None:
        deadline = time.perf_counter() + timeout_s
        while self._sending.is_set() and time.perf_counter() < deadline:
            time.sleep(0.001)
        self._send(op, fid, payload=payload)

    def _wait(self, fid: int, timeout_s: float):
        deadline = time.perf_counter() + timeout_s
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                raise RemoteTimeout(f"no response for frame {fid} within {timeout_s:.3f} s")
            try:
                item = self._resp.get(timeout=left)
            except queue.Empty as e:
                raise RemoteTimeout(f"no response for frame {fid} within {timeout_s:.3f} s") from e
            if item is None:
                raise RuntimeError("NPU server connection closed")
            if item[0] == fid:
                return item
            if item[0] == CONTROL_FID:
                continue
            self.n_stale += 1  # late answer to an earlier (timed-out) request

    def encode(self, bgr: np.ndarray) -> tuple[int, int, int, bytes, Letterbox, float]:
        t0 = time.perf_counter()
        lb_img, lb = letterbox(bgr, self.imgsz)
        enc = ENCODINGS[self.encoding]
        if enc == 0:
            payload, w, h = lb_img.tobytes(), self.imgsz, self.imgsz
        else:
            y0, x0 = int(lb.pad_y), int(lb.pad_x)
            content = lb_img[y0 : self.imgsz - y0, x0 : self.imgsz - x0]
            h, w = content.shape[:2]
            if enc == 1:
                payload = np.ascontiguousarray(content).tobytes()
            else:
                if self.encoding == "content_png":  # lossless, fast compression level
                    ok, buf = cv2.imencode(".png", content, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                else:
                    ok, buf = cv2.imencode(".jpg", content, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                payload = buf.tobytes()
        return enc, w, h, payload, lb, time.perf_counter() - t0

    def infer(self, frame_id: int, capture_t: float, bgr: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        stages: dict = {}
        try:
            enc, w, h, payload, lb, t_enc = self.encode(bgr)
            stages["encode"] = t_enc
            stages["payload_bytes"] = len(payload)
            with self._lock:
                stages["send"] = self._send(OP_INFER, frame_id, enc, w, h, payload)
                fid, status, body, t_rx = self._wait(frame_id, self.timeout_s - (time.perf_counter() - t0))
            if status != 0:
                raise RuntimeError(body.get("error", "remote error"))
            t1 = time.perf_counter()
            d = np.asarray(body["dets"], np.float64).reshape(-1, 6)
            dets = _unletterbox(d, lb, bgr.shape[1], bgr.shape[0])
            stages.update(body["t"])
            stages["rtt"] = t_rx - t0 - t_enc
            stages["network_est"] = stages["rtt"] - body["t"]["server_total"] - stages["send"]
            stages["client_post"] = time.perf_counter() - t1
            self.n_ok += 1
            return InferenceResult(
                frame_id, capture_t, self.name, self.precision, True, dets, time.perf_counter() - t0, stages=stages,
                input_shape=self.input_shape,
            )  # fmt: skip
        except RemoteTimeout as e:
            self.n_timeout += 1
            err = repr(e)
        except TransportBusy as e:
            self.n_busy += 1
            err = repr(e)
        except Exception as e:  # noqa: BLE001
            self.n_err += 1
            err = repr(e)
        return InferenceResult(
            frame_id, capture_t, self.name, self.precision, False, _empty(), time.perf_counter() - t0, error=err,
            stages=stages, input_shape=self.input_shape,
        )  # fmt: skip

    def echo(self, n_bytes: int, fid: int = 0) -> float:
        """Transport-only round trip (no inference)."""
        t0 = time.perf_counter()
        with self._lock:
            self._send_blocking(OP_ECHO, fid, payload=b"\0" * n_bytes)
            self._wait(fid, 30.0)
        return time.perf_counter() - t0

    def stall(self, seconds: float) -> None:
        """Fault injection: the server sleeps; requests sent meanwhile time out."""
        with self._lock:
            self._send_blocking(OP_STALL, CONTROL_FID, payload=str(seconds).encode())

    def health(self) -> dict:
        h = {
            "backend": self.name,
            "precision": self.precision,
            "available": self.proc.poll() is None,
            "n_ok": self.n_ok,
            "n_err": self.n_err,
            "n_timeout": self.n_timeout,
            "n_stale": self.n_stale,
            "n_busy": self.n_busy,
        }
        return h

    def remote_health(self, fid: int = 0, timeout_s: float = 15.0) -> dict:
        with self._lock:
            self._send_blocking(OP_HEALTH, fid)
            return self._wait(fid, timeout_s)[2]

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                with self._lock:
                    self._send_blocking(OP_QUIT, 0, timeout_s=2.0)
                self._wq.put(None)
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        if self._err not in (None, subprocess.DEVNULL):
            self._err.close()
