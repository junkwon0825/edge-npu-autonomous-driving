"""Measure the desktop <-> RNGD-host link as an NPU-in-the-loop transport would see it.

One persistent SSH session runs a tiny echo server on the NPU host; for each request we send a
payload and wait for a 1-line ack. Payload sizes: a 1-byte ping, a letterboxed 640x640x3 uint8
frame (1,228,800 B, what the NPU needs per inference) and a JPEG-sized frame (~60 kB). This is
round-trip transport time only (no inference). Output: reports/evidence/npu_link.json.

  python scripts/measure_link.py [--host furiosa-npu] [--n 30]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time

from npu_ad.config import resolve
from npu_ad.evaluation.metrics import latency_stats

SERVER = (
    "import sys\n"
    "r, w = sys.stdin.buffer, sys.stdout.buffer\n"
    "while True:\n"
    "    h = r.readline()\n"
    "    if not h:\n"
    "        break\n"
    "    n = int(h)\n"
    "    got = 0\n"
    "    while got < n:\n"
    "        got += len(r.read(min(1 << 20, n - got)))\n"
    "    w.write(b'ok\\n')\n"
    "    w.flush()\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="furiosa-npu")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    code = base64.b64encode(SERVER.encode()).decode()  # survives the remote shell's quoting
    p = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", a.host, f"python3 -u -c \"import base64;exec(base64.b64decode('{code}'))\""],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    sizes = {"ping_1B": 1, "jpeg_like_60kB": 60_000, "letterboxed_frame_1.23MB": 640 * 640 * 3}
    res: dict = {"host_alias": a.host, "transport": "single persistent ssh session, payload up + 3-byte ack"}
    try:
        for name, size in sizes.items():
            buf = os.urandom(size)
            times = []
            for i in range(a.n + 3):
                t0 = time.perf_counter()
                p.stdin.write(f"{size}\n".encode() + buf)
                p.stdin.flush()
                ack = p.stdout.readline()
                if ack != b"ok\n":
                    raise RuntimeError(f"bad ack {ack!r}")
                if i >= 3:  # first requests warm the session
                    times.append(time.perf_counter() - t0)
            st = latency_stats(times)
            st["throughput_MBps"] = size / (sum(times) / len(times)) / 1e6
            res[name] = st
            print(name, round(st["mean_ms"], 2), "ms mean,", round(st["p95_ms"], 2), "ms p95")
    finally:
        p.stdin.close()
        p.wait(timeout=30)
    out = resolve("reports/evidence/npu_link.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
