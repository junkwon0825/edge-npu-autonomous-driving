"""Local helper for the Furiosa RNGD host, reached through an existing ~/.ssh/config alias.

No host names, users or keys are stored here; pass the alias (default ``furiosa-npu``).
All remote files live in one dedicated directory (default ``~/npu_auto_driving_p2``);
nothing outside it is written or deleted.

  python scripts/npu_remote.py sync
  python scripts/npu_remote.py start compile           # background (nohup, nice)
  python scripts/npu_remote.py status compile
  python scripts/npu_remote.py fetch npu_outputs.pt outputs/npu/
  python scripts/npu_remote.py put outputs/npu/inputs.pt
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=yes"]
SYNC_FILES = [
    ("remote/npu_job.py", "npu_job.py"),
    ("remote/int8_probe.py", "int8_probe.py"),
    ("src/npu_ad/__init__.py", "src/npu_ad/__init__.py"),
    ("src/npu_ad/perception/__init__.py", "src/npu_ad/perception/__init__.py"),
    ("src/npu_ad/perception/yolo_raw.py", "src/npu_ad/perception/yolo_raw.py"),
]


def ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, host, command], capture_output=True, text=True, timeout=timeout)


def scp(src: str, dst: str, timeout: int = 600) -> None:
    subprocess.run(["scp", "-q", *SSH_OPTS, src, dst], check=True, timeout=timeout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sync", "start", "status", "fetch", "put", "log"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--host", default="furiosa-npu")
    ap.add_argument("--dir", default="npu_auto_driving_p2")
    ap.add_argument("--env", default="furiosa-npu", help="conda env on the NPU host")
    a = ap.parse_args()
    d = a.dir

    if a.action == "sync":
        r = ssh(a.host, f"mkdir -p {d}/src/npu_ad/perception && test -f {d}/yolo26n.pt || cp -n yolo26n.pt {d}/")
        if r.returncode:
            print(r.stderr, file=sys.stderr)
            return r.returncode
        for local, remote in SYNC_FILES:
            scp(str(ROOT / local), f"{a.host}:{d}/{remote}")
        print(ssh(a.host, f"cd {d} && sha256sum yolo26n.pt npu_job.py src/npu_ad/perception/yolo_raw.py").stdout)
        return 0
    if a.action == "start":
        cmd, extra = a.args[0], " ".join(shlex.quote(x) for x in a.args[1:])
        script = "int8_probe.py" if cmd == "int8_probe" else f"npu_job.py {cmd}"
        # One NPU job at a time: refuse to start while a previous job of this project still runs
        # (two jobs would overwrite each other's status/output files and share the device).
        # Only the nohup command is backgrounded (an `a && b && cmd &` list would background a
        # subshell that keeps ssh's stdout open and blocks this call).
        remote = (
            f"cd {d} || exit 1; "
            "if ps -u \"$(id -u)\" -o args= | grep -qE '^python[0-9.]* (npu_job|int8_probe)[.]py'; "
            "then echo BUSY; exit 3; fi; "
            f"source ~/miniconda3/etc/profile.d/conda.sh; conda activate {a.env} || exit 1; "
            f"nohup nice -n 10 python {script} {extra} > log_{cmd}.txt 2>&1 < /dev/null & echo started $!"
        )
        r = ssh(a.host, f"bash -lc {shlex.quote(remote)}")
        print(r.stdout, r.stderr)
        return r.returncode
    if a.action == "status":
        r = ssh(a.host, f"cat {d}/status_{a.args[0]}.json 2>/dev/null || echo NO_STATUS")
        print(r.stdout)
        return r.returncode
    if a.action == "log":
        n = a.args[1] if len(a.args) > 1 else "40"
        print(ssh(a.host, f"tail -n {int(n)} {d}/log_{a.args[0]}.txt").stdout)
        return 0
    if a.action == "fetch":
        dst = Path(a.args[1] if len(a.args) > 1 else "outputs/npu")
        dst.mkdir(parents=True, exist_ok=True)
        scp(f"{a.host}:{d}/{a.args[0]}", str(dst))
        return 0
    if a.action == "put":
        scp(a.args[0], f"{a.host}:{d}/")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
