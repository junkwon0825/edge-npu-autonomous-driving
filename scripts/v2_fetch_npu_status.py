"""Copy RNGD job status files into reports/evidence/v2_npu_<name>.json with host details redacted."""

from __future__ import annotations

import json
import subprocess
import sys

from npu_ad.config import resolve

REDACT = {"host", "hostname", "user", "home"}


def scrub(o):
    if isinstance(o, dict):
        return {k: ("<redacted>" if k in REDACT else scrub(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    if isinstance(o, str):
        return o.replace("/root/", "~/")  # remote home directory in compiler cache paths
    return o


def main(names: list[str]) -> int:
    for n in names:
        r = subprocess.run([sys.executable, str(resolve("scripts/npu_remote.py")), "status", n],
                           capture_output=True, text=True, timeout=120)  # fmt: skip
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            print(n, "no status")
            continue
        out = resolve(f"reports/evidence/v2_npu_{n}.json")
        out.write_text(json.dumps(scrub(data), indent=2), encoding="utf-8")
        print(n, "->", out.name, data.get("state") or data.get("finished"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
