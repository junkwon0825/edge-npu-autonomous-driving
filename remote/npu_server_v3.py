# ruff: noqa: E501  (long result/row dict literals kept on one line)
"""V3 multi-model RNGD inference server: YOLO26n detection + lane U-Net, one model per PE (NPU host).

Same transport as the V2 server (persistent SSH session, binary length-prefixed stdin/stdout protocol,
``npu_ad.runtime.backends`` client). One uploaded frame feeds both models; each model has its own
``ModelRunner`` (compiled module on its own furiosa device + worker thread), so detection and lane
segmentation execute concurrently on separate PEs.

Request : b"NPQ1" | op u8 | frame_id u32 | enc u8 | w u16 | h u16 | n u32 | payload
Response: b"NPR1" | frame_id u32 | status u8 | n u32 | JSON
ops: 1 INFER (detection), 2 ECHO, 3 HEALTH, 4 STALL (seconds), 5 QUIT, 6 INFER_BOTH, 7 INFER_LANE,
     8 CRASH (fault injection: the process exits immediately without answering)
The lane result is a binary mask (logit > 0) at 256x512, PNG-encoded, base64 in the JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE)]

REQ = struct.Struct("<4sBIBHHI")
RESP = struct.Struct("<4sIBI")
OP_INFER, OP_ECHO, OP_HEALTH, OP_STALL, OP_QUIT, OP_BOTH, OP_LANE, OP_CRASH = 1, 2, 3, 4, 5, 6, 7, 8


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
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                return {"device": c[1], "firmware": c[2], "temp": c[3], "power": c[4]}
        return {"raw": out[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


class ModelRunner:
    """One compiled model on one furiosa device with its own worker thread (FIFO of size 1)."""

    def __init__(self, name: str, module, example, device: str):
        import furiosa.torch as ft
        import torch
        from furiosa.torch import CompileModule

        t0 = time.perf_counter()
        self.name, self.device = name, torch.device(device)
        self.compiled = CompileModule.from_module(module, (example,)).to(self.device)
        self.stream = ft.current_stream(self.device)
        self.dtype = example.dtype
        with torch.no_grad():
            for _ in range(3):
                self.compiled(example.to(self.device))
        self.stream.synchronize()
        self.load_s = time.perf_counter() - t0
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self.n_ok = self.n_err = 0
        threading.Thread(target=self._loop, daemon=True, name=f"runner-{name}").start()

    def _loop(self) -> None:
        import torch

        while True:
            x, box = self._jobs.get()
            t0 = time.perf_counter()
            try:
                xd = x.to(self.dtype).to(self.device)
                self.stream.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    out = self.compiled(xd)
                self.stream.synchronize()
                t2 = time.perf_counter()
                out = [o.cpu().float() for o in out] if isinstance(out, (tuple, list)) else out.cpu().float()
                box["out"], box["t"] = out, {"h2d": t1 - t0, "exec": t2 - t1, "d2h": time.perf_counter() - t2}
                self.n_ok += 1
            except Exception as e:  # noqa: BLE001 - reported to the client
                box["error"] = repr(e)[:400]
                self.n_err += 1
            box["done"].set()

    def submit(self, x) -> dict:
        box = {"done": threading.Event()}
        self._jobs.put((x, box))
        return box


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "yolo26n.pt"))
    ap.add_argument("--lane-weights", default=str(HERE / "lane_unet_s3c32.pt"))
    ap.add_argument("--det-device", default="furiosa:0")
    ap.add_argument("--lane-device", default="furiosa:1")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--models", default="det,lane")
    a = ap.parse_args()
    r, w = sys.stdin.buffer, sys.stdout.buffer

    import cv2
    import furiosa.torch as ft
    import numpy as np
    import torch
    from npu_job import _build

    from npu_ad.perception.lane_net import LANE_IN_H, LANE_IN_W, LaneUNet, export_for_npu
    from npu_ad.perception.yolo_raw import RawDecoder, load_raw_model, to_tensor

    torch.set_num_threads(a.threads)
    models = set(a.models.split(","))
    runners: dict[str, ModelRunner] = {}
    if "det" in models:
        raw, x0 = _build(a.weights, a.imgsz, "bf16")
        _, head = load_raw_model(a.weights)
        decoder = RawDecoder(head, a.imgsz)
        runners["det"] = ModelRunner("det", raw, x0, a.det_device)
    if "lane" in models:
        ck = torch.load(a.lane_weights, map_location="cpu")
        lane = LaneUNet(**ck.get("arch", {}))
        lane.load_state_dict(ck["state_dict"])
        lm = export_for_npu(lane).to(torch.bfloat16)
        runners["lane"] = ModelRunner("lane", lm, torch.zeros(1, 3, LANE_IN_H, LANE_IN_W, dtype=torch.bfloat16),
                                      a.lane_device)  # fmt: skip
    ver = ft.version() if callable(getattr(ft, "version", None)) else "?"
    ready = {"ready": True, "server": "v3", "sdk": ver, "dtype": "bf16", "threads": torch.get_num_threads(),
             "models": {k: {"device": str(v.device), "load_s": round(v.load_s, 2)} for k, v in runners.items()},
             "imgsz": a.imgsz, "lane_input": [LANE_IN_H, LANE_IN_W]}  # fmt: skip
    print(json.dumps(ready), file=sys.stderr, flush=True)
    w.write(b"READY " + json.dumps(ready).encode() + b"\n")
    w.flush()

    def respond(fid: int, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        w.write(RESP.pack(b"NPR1", fid, status, len(body)) + body)
        w.flush()

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
            respond(fid, 0, {"bye": True})
            return 0
        if op == OP_CRASH:
            os._exit(3)  # fault injection: worker crash without an answer
        if op == OP_ECHO:
            respond(fid, 0, {"rx_s": t_rx - t_hdr})
            continue
        if op == OP_HEALTH:
            respond(fid, 0, {"smi": _smi(), "loadavg": list(os.getloadavg()),
                             "runners": {k: {"ok": v.n_ok, "err": v.n_err} for k, v in runners.items()}})  # fmt: skip
            continue
        if op == OP_STALL:
            time.sleep(float(payload.decode() or 0))
            respond(fid, 0, {"stalled": True})
            continue
        try:
            want = {OP_INFER: ["det"], OP_LANE: ["lane"], OP_BOTH: ["det", "lane"]}[op]
            missing = [m for m in want if m not in runners]
            if missing:
                raise ValueError(f"model not loaded: {missing}")
            s = a.imgsz
            if enc == 1:
                if len(payload) != fw * fh * 3:
                    raise ValueError(f"payload {len(payload)} B does not match {fw}x{fh}x3")
                img = np.frombuffer(payload, np.uint8).reshape(fh, fw, 3)
            elif enc == 2:
                img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            else:
                raise ValueError(f"encoding {enc} not supported by the v3 server")
            if img is None or img.shape[0] > s or img.shape[1] > s:
                raise ValueError(f"bad frame shape {None if img is None else img.shape}")
            t_dec = time.perf_counter()
            boxes = {}
            if "det" in want:
                py, px = s - img.shape[0], s - img.shape[1]
                lb = cv2.copyMakeBorder(img, py // 2, py - py // 2, px // 2, px - px // 2, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))  # fmt: skip
                boxes["det"] = runners["det"].submit(to_tensor(lb))
            if "lane" in want:
                li = cv2.resize(img, (LANE_IN_W, LANE_IN_H), interpolation=cv2.INTER_AREA)
                xl = torch.from_numpy(np.ascontiguousarray(li[:, :, ::-1].transpose(2, 0, 1))).float().div_(255)[None]
                boxes["lane"] = runners["lane"].submit(xl)
            t_sub = time.perf_counter()
            for b in boxes.values():
                if not b["done"].wait(10.0):
                    raise TimeoutError("runner did not finish within 10 s")
                if "error" in b:
                    raise RuntimeError(b["error"])
            t_run = time.perf_counter()
            out: dict = {}
            if "det" in boxes:
                bx, sc = boxes["det"]["out"]
                det = decoder(bx, sc, a.conf, a.iou, 300, prefilter=True)[0]
                out["dets"] = [[round(float(v), 3) for v in d[:4]] + [round(float(d[4]), 4), int(d[5])] for d in det]
            if "lane" in boxes:
                m = (boxes["lane"]["out"][0, 0].numpy() > 0).astype(np.uint8) * 255
                ok, png = cv2.imencode(".png", m)
                out["lane"] = {"png_b64": base64.b64encode(png.tobytes()).decode(), "shape": [LANE_IN_H, LANE_IN_W],
                               "pos_frac": float(m.mean() / 255)}  # fmt: skip
            out["t"] = {"rx": t_rx - t_hdr, "decode": t_dec - t_rx, "submit": t_sub - t_dec, "parallel_run": t_run - t_sub,
                        "post": time.perf_counter() - t_run, "server_total": time.perf_counter() - t_rx}  # fmt: skip
            for k, b in boxes.items():
                out["t"] |= {f"{k}_{kk}": v for kk, v in b["t"].items()}
            respond(fid, 0, out)
        except Exception as e:  # noqa: BLE001
            respond(fid, 1, {"error": repr(e)[:500]})


if __name__ == "__main__":
    raise SystemExit(main())
