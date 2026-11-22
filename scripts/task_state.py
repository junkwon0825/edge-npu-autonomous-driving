"""Durable task state for unattended runs: reports/agent_runs/{task_state.json,failures.jsonl,latest_status.md}.

  python scripts/task_state.py show
  python scripts/task_state.py set C1 PASS --artifact reports/evidence/x.json --note "..."
  python scripts/task_state.py fail N2 --signature "UnsupportedOpError:select_scatter" --note "..."

Writes are atomic (temp file + replace). A task with status PASS is skipped on resume; a failure
signature seen 3 times marks the task BLOCKED (retry cap).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "reports" / "agent_runs"
STATE = DIR / "task_state.json"
FAIL = DIR / "failures.jsonl"
STATUS_MD = DIR / "latest_status.md"
MAX_RETRIES = 3


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    state["updated"] = _now()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE)
    render(state)


def render(state: dict) -> None:
    lines = [f"# Latest status ({state['updated']})", "", "| Task | Phase | Owner | Status | Retries | Next |",
             "|---|---|---|---|---|---|"]
    for tid, t in state["tasks"].items():
        lines.append(f"| {tid} {t['title']} | {t['phase']} | {t['owner']} | **{t['status']}** | "
                     f"{t.get('retries', 0)} | {t.get('next', '')} |")
    lines += ["", "Status values: PASS / PARTIAL / BLOCKED / BLOCKED_APPROVAL / NOT_RUN / RUNNING / TODO.", ""]
    tmp = STATUS_MD.with_suffix(".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, STATUS_MD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["show", "set", "fail"])
    ap.add_argument("task", nargs="?")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--note")
    ap.add_argument("--next")
    ap.add_argument("--signature")
    a = ap.parse_args()
    state = load()
    if a.action == "show":
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0
    t = state["tasks"][a.task]
    if a.action == "set":
        t["status"] = a.status
        t.setdefault("artifacts", []).extend(x for x in a.artifact if x not in t.get("artifacts", []))
        if a.note:
            t["note"] = a.note
        if a.next is not None:
            t["next"] = a.next
        t["last_checkpoint"] = _now()
    else:
        t["retries"] = t.get("retries", 0) + 1
        t["error_signature"] = a.signature
        with FAIL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": _now(), "task": a.task, "signature": a.signature, "note": a.note,
                                 "retry": t["retries"]}, ensure_ascii=False) + "\n")
        same = sum(1 for line in FAIL.read_text(encoding="utf-8").splitlines()
                   if json.loads(line).get("task") == a.task and json.loads(line).get("signature") == a.signature)
        t["status"] = "BLOCKED" if same >= MAX_RETRIES else "FAILED_RETRYABLE"
    save(state)
    print(a.task, t["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
