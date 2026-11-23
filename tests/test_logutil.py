import json
import logging

from npu_ad.logutil import setup_logging


def test_json_lines_with_extra_fields(tmp_path):
    path = setup_logging("t", log_dir=tmp_path)
    logging.getLogger("x").info("run done", extra={"scenario": "s1", "collision": False})
    for h in logging.getLogger().handlers:
        h.flush()
    rec = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["msg"] == "run done" and rec["scenario"] == "s1" and rec["collision"] is False
    assert rec["level"] == "INFO" and rec["logger"] == "x"
