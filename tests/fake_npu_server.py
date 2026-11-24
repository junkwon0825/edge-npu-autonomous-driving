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
    elif op == 1:
        body = json.dumps({"dets": [[100, 200, 140, 300, 0.9, 0]], "t": {"rx": 0, "server_total": 0.001}}).encode()
    else:
        body = b'{"ok": true}'
    w.write(RESP.pack(b"NPR1", fid, 0, len(body)) + body)
    w.flush()
    if op == 5:
        break
