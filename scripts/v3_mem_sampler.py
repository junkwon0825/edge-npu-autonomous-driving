"""Sample system memory and the main memory consumers every N seconds into a CSV (until killed or --duration)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_run_scenarios import memory_snapshot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v3/memory.csv")
    ap.add_argument("--every-s", type=float, default=15.0)
    ap.add_argument("--duration-s", type=float, default=4 * 3600)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + a.duration_s
    header = not out.exists()
    with out.open("a", encoding="utf-8") as fh:
        while time.time() < t_end:
            s = memory_snapshot()
            if header:
                fh.write("time," + ",".join(s) + "\n")
                header = False
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S") + "," + ",".join(str(v) for v in s.values()) + "\n")
            fh.flush()
            time.sleep(a.every_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
