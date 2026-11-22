"""Build configs/latency_conditions.json from measured benchmark files only.

GPU: p95 end-to-end latency of the clean GPU benchmark.
NPU: p95 of (host preprocess from the GPU run) + H2D + compute + D2H (measured on the RNGD host)
     + host postprocess (from the GPU run, identical code). Added only if the NPU benchmark exists.
"""

from __future__ import annotations

import json

import numpy as np

from npu_ad.config import resolve


def main() -> int:
    conds = [
        {"name": "aeb_off", "delay_ms": None, "source": "baseline without AEB"},
        {"name": "ideal_0ms", "delay_ms": 0.0, "source": "idealised, no perception latency"},
    ]
    gpu_p = resolve("reports/evidence/gpu_benchmark_fp32_clean.json")
    gpu = json.loads(gpu_p.read_text(encoding="utf-8"))
    conds.append(
        {"name": "gpu_p95", "delay_ms": gpu["latency"]["e2e"]["p95_ms"], "source": f"measured: {gpu_p.name} e2e p95"}
    )
    npu_p = resolve("reports/evidence/npu_bench.json")
    if npu_p.exists():
        npu = json.loads(npu_p.read_text(encoding="utf-8"))
        t = npu["result"]["timings_s"]
        dev = np.array(t["h2d"]) + np.array(t["compute"]) + np.array(t["d2h"])
        host = gpu["latency"]["preprocess"]["p95_ms"] + gpu["latency"]["postprocess"]["p95_ms"]
        conds.append(
            {
                "name": "npu_p95",
                "delay_ms": float(np.percentile(dev * 1000, 95)) + host,
                "source": f"measured: {npu_p.name} device-side p95 + host pre/post p95 from GPU run "
                "(excludes network transfer to the remote NPU host)",
            }
        )
    conds += [
        {"name": "stress_100ms", "delay_ms": 100.0, "source": "sensitivity sweep"},
        {"name": "stress_200ms", "delay_ms": 200.0, "source": "sensitivity sweep"},
    ]
    out = resolve("configs/latency_conditions.json")
    out.write_text(json.dumps({"conditions": conds}, indent=2), encoding="utf-8")
    print(json.dumps(conds, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
