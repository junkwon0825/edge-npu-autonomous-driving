"""Persistent YOLO26n inference server for the Furiosa RNGD (V2 runtime, runs on the NPU host).

Started once per session through the existing SSH alias; requests arrive on stdin and responses
leave on stdout (binary, length-prefixed), so there is no per-frame process or connection setup.
The compiled module comes from the SDK compile cache (same ``_build`` as ``npu_job.py``), so no
recompilation happens when the shape/dtype was compiled before.

Request : b"NPQ1" | op u8 | frame_id u32 | enc u8 | w u16 | h u16 | n u32 | payload (n bytes)
Response: b"NPR1" | frame_id u32 | status u8 | n u32 | JSON (n bytes)

ops : 1 INFER (payload = frame), 2 ECHO (payload discarded, transport only), 3 HEALTH
      (furiosa-smi sample), 4 STALL (payload = ascii seconds; fault injection), 5 QUIT
enc : 0 letterboxed BGR uint8 (size x size x 3)
      1 BGR uint8 content region w x h (letterbox padding removed; re-padded here, lossless)
      2 JPEG of the content region (lossy)
Detections are returned in letterbox coordinates [x1, y1, x2, y2, conf, cls]; the client maps them
back with its own letterbox parameters. Every response carries per-stage durations (seconds).
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

REQ = struct.Struct("<4sBIBHHI")
RESP = struct.Struct("<4sIBI")
OP_INFER, OP_ECHO, OP_HEALTH, OP_STALL, OP_QUIT = 1, 2, 3, 4, 5


def _read_exact(r, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = r.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def _smi() -> dict:
    try:
        out = subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "npu0" in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                return {"device": cells[1], "firmware": cells[2], "temp": cells[3], "power": cells[4]}
        return {"raw": out[-300:]}
    except Exception as e:  # noqa: BLE001 - health must never crash the server
        return {"error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "w8"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="furiosa:0")
    ap.add_argument("--weights", default=str(HERE / "yolo26n.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    ap.add_argument("--prefilter", type=int, default=1, help="decode only anchors above conf (same detections)")
    a = ap.parse_args()
    r, w = sys.stdin.buffer, sys.stdout.buffer
    log = sys.stderr

    import cv2
    import furiosa.torch as ft
    import numpy as np
    import torch
    from furiosa.torch import CompileModule
    from npu_job import _build

    from npu_ad.perception.yolo_raw import RawDecoder, load_raw_model, to_tensor

    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    raw, x0 = _build(a.weights, a.imgsz, a.dtype)
    _, head = load_raw_model(a.weights)
    decoder = RawDecoder(head, a.imgsz)
    compiled = CompileModule.from_module(raw, (x0,))
    load_s = time.perf_counter() - t0
    dev = torch.device(a.device)
    compiled = compiled.to(dev)
    stream = ft.current_stream(dev)
    with torch.no_grad():  # warm-up
        for _ in range(5):
            compiled(x0.to(dev))
        stream.synchronize()
    ver = ft.version() if callable(getattr(ft, "version", None)) else str(getattr(ft, "version", "?"))
    ready = {
        "ready": True,
        "dtype": a.dtype,
        "imgsz": a.imgsz,
        "device": a.device,
        "sdk": ver,
        "load_s": load_s,
        "input_dtype": str(x0.dtype),
        "input_shape": list(x0.shape),
        "threads": torch.get_num_threads(),
        "prefilter": bool(a.prefilter),
    }
    print(json.dumps(ready), file=log, flush=True)
    w.write(b"READY " + json.dumps(ready).encode() + b"\n")
    w.flush()

    def respond(fid: int, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        w.write(RESP.pack(b"NPR1", fid, status, len(body)) + body)
        w.flush()

    n_ok = n_err = 0
    while True:
        try:
            magic, op, fid, enc, fw, fh, n = REQ.unpack(_read_exact(r, REQ.size))
            t_hdr = time.perf_counter()
            payload = _read_exact(r, n) if n else b""
        except EOFError:
            return 0
        t_rx = time.perf_counter()
        if magic != b"NPQ1":
            respond(fid, 1, {"error": "bad magic"})
            return 2
        if op == OP_QUIT:
            respond(fid, 0, {"bye": True, "n_ok": n_ok, "n_err": n_err})
            return 0
        if op == OP_ECHO:
            respond(fid, 0, {"rx_s": t_rx - t_hdr})
            continue
        if op == OP_HEALTH:
            respond(
                fid, 0, {"smi": _smi(), "n_ok": n_ok, "n_err": n_err, "loadavg": list(__import__("os").getloadavg())}
            )
            continue
        if op == OP_STALL:
            time.sleep(float(payload.decode() or 0))
            respond(fid, 0, {"stalled_s": float(payload.decode() or 0)})
            continue
        try:
            s = a.imgsz
            if enc == 0:
                lb = np.frombuffer(payload, np.uint8).reshape(s, s, 3)
            else:
                if enc == 1:
                    img = np.frombuffer(payload, np.uint8).reshape(fh, fw, 3)
                elif enc == 2:
                    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                else:
                    raise ValueError(f"unknown encoding {enc}")
                if img is None or img.shape[0] > s or img.shape[1] > s:
                    raise ValueError(f"bad frame shape {None if img is None else img.shape}")
                pad_y, pad_x = s - img.shape[0], s - img.shape[1]
                top, left = pad_y // 2, pad_x // 2  # same split as yolo_raw.letterbox for even pads
                lb = cv2.copyMakeBorder(
                    img, top, pad_y - top, left, pad_x - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            t_dec = time.perf_counter()
            x = to_tensor(lb).to(x0.dtype)
            t_pre = time.perf_counter()
            xd = x.to(dev)
            stream.synchronize()
            t_h2d = time.perf_counter()
            with torch.no_grad():
                boxes, scores = compiled(xd)
            stream.synchronize()
            t_exec = time.perf_counter()
            boxes, scores = boxes.cpu().float(), scores.cpu().float()
            t_d2h = time.perf_counter()
            det = decoder(boxes, scores, a.conf, a.iou, a.max_det, prefilter=bool(a.prefilter))[0]
            dets = [[round(float(v), 3) for v in d[:4]] + [round(float(d[4]), 4), int(d[5])] for d in det]
            t_post = time.perf_counter()
            n_ok += 1
            respond(
                fid,
                0,
                {
                    "dets": dets,
                    "t": {
                        "rx": t_rx - t_hdr,
                        "decode": t_dec - t_rx,
                        "pre": t_pre - t_dec,
                        "h2d": t_h2d - t_pre,
                        "exec": t_exec - t_h2d,
                        "d2h": t_d2h - t_exec,
                        "post": t_post - t_d2h,
                        "server_total": time.perf_counter() - t_rx,
                    },
                },
            )
        except Exception as e:  # noqa: BLE001 - report the failure to the client, keep serving
            n_err += 1
            respond(fid, 1, {"error": repr(e)[:500]})


if __name__ == "__main__":
    raise SystemExit(main())
