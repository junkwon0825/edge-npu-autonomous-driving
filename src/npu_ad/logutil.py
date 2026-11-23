"""Structured (JSON-lines) logging for long-running experiment scripts.

``setup_logging("run_scenarios")`` sends human-readable lines to stderr and one JSON object per
record to ``outputs/logs/<name>.jsonl`` (timestamp, level, logger, message, plus any ``extra=``
fields), so an unattended run can be audited and resumed from its log.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from npu_ad import PROJECT_ROOT

_STD = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out |= {k: v for k, v in vars(record).items() if k not in _STD}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> Path:
    log_dir = log_dir or PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}.jsonl"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_npu_ad", False):
            root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for h in (fh, sh):
        h._npu_ad = True  # type: ignore[attr-defined]
        root.addHandler(h)
    return path
