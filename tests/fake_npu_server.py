"""Protocol-compatible stand-in for remote/npu_server.py (tests only): one fixed detection."""

import json
import struct
import sys
import time

REQ = struct.Struct("<4sBIBHHI")
RESP = struct.Struct("<4sIBI")
r, w = sys.stdin.buffer, sys.stdout.buffer
w.write(b'READY {"ready": true, "dtype": "fake"}\n')
w.flush()
while True:
    hdr = r.read(REQ.size)
    if len(hdr) < REQ.size:
        break
    magic, op, fid, enc, fw, fh, n = REQ.unpack(hdr)
    payload = r.read(n) if n else b""
    if op == 5:
        body = b"{}"
    elif op == 4:
        time.sleep(float(payload.decode()))
        body = b"{}"
    elif op in (1, 6, 7):
        out = {"t": {"rx": 0, "server_total": 0.001}}
        if op in (1, 6):
            out["dets"] = [[100, 200, 140, 300, 0.9, 0]]
        if op in (6, 7):  # 256x512 lane mask with one marking column, PNG base64 (V3 server format)
            import base64

            import cv2
            import numpy as np

            m = np.zeros((256, 512), np.uint8)
            m[128:, 200:203] = 255
            png = base64.b64encode(cv2.imencode(".png", m)[1].tobytes()).decode()
            out["lane"] = {"png_b64": png, "shape": [256, 512]}
        body = json.dumps(out).encode()
    elif op == 8:
        break
    else:
        body = b'{"ok": true}'
    w.write(RESP.pack(b"NPR1", fid, 0, len(body)) + body)
    w.flush()
    if op == 5:
        break
