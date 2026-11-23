"""Parameter/buffer bytes of the deployed raw module per precision variant (fp32, bf16, w8).

Counts every tensor in ``state_dict()`` of the module exactly as it is handed to the compiler
(``remote/npu_job._build``). Output: reports/evidence/model_size.json.
"""

from __future__ import annotations

import json
import sys

from npu_ad.config import load_config, resolve

sys.path.insert(0, str(resolve("remote")))
import npu_job  # noqa: E402


def main() -> int:
    w = str(resolve(load_config()["model"]["weights"]))
    out = {"weights_file_bytes": resolve(load_config()["model"]["weights"]).stat().st_size}
    for dt in ("fp32", "bf16", "w8"):
        m, _ = npu_job._build(w, 640, dt)
        sd = m.state_dict()
        by_dtype: dict[str, int] = {}
        for t in sd.values():
            by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + t.numel() * t.element_size()
        out[dt] = {"total_bytes": sum(by_dtype.values()), "by_dtype": by_dtype, "n_tensors": len(sd)}
        print(dt, out[dt]["total_bytes"], by_dtype)
    resolve("reports/evidence/model_size.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
