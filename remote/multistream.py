"""Bounded multi-stream / multi-PE scaling probe for YOLO26n BF16 on the RNGD (V2, NPU host).

``furiosa.torch.get_available_devices()`` lists furiosa:0..7 (one RNGD; device configuration
``Fusion(num_pe_per_cluster=1)``), i.e. each device is a single PE. The compiled module (from the
SDK cache) is moved to K devices and driven by K Python threads, each synchronising its own stream
after every inference (no multi-submission per stream: queuing 8 executions before one sync
failed with PXI-122 in V1). Configurations, each run for ``--secs`` seconds:

  pe1_t1     1 thread on furiosa:0 (baseline)
  pe1_t2     2 threads sharing furiosa:0 (same PE)
  pe2_t2     2 threads on furiosa:0,1
  pe4_t4     4 threads on furiosa:0..3
  pe8_t8     8 threads on furiosa:0..7 (only if every earlier configuration passed)

After every configuration a single inference on furiosa:0 is compared with the CPU reference
(health check); the probe stops at the first failure. Output: status_multistream.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE)]
CONFIGS = [
    ("pe1_t1", [0]),
    ("pe1_t2", [0, 0]),
    ("pe2_t2", [0, 1]),
    ("pe4_t4", [0, 1, 2, 3]),
    ("pe8_t8", list(range(8))),
]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def main() -> int:
    import furiosa.torch as ft
    import torch
    from furiosa.torch import CompileModule
    from npu_job import _build

    secs = float(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 8.0
    torch.set_num_threads(2)
    out = HERE / "status_multistream.json"
    st: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sdk": ft.version(), "secs": secs, "configs": {}}
    st["devices"] = ft.get_available_devices()
    st["device_configuration"] = repr(ft.get_device_configuration())
    st["loadavg_start"] = os.getloadavg()
    st["smi_start"] = subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True).stdout[-400:]
    raw, x0 = _build(str(HERE / "yolo26n.pt"), 640, "bf16")
    with torch.no_grad():
        ref_b, ref_s = raw(x0)
    mods: dict[int, object] = {}

    def module_on(k: int):
        # one CompileModule instance per device (cache hit): moving an instance that already lives on
        # furiosa:0 to furiosa:1 fails ("peer D2D copy requires ... the same mesh"; first attempt)
        if k not in mods:
            t = time.perf_counter()
            mods[k] = CompileModule.from_module(raw, (x0,)).to(torch.device(f"furiosa:{k}"))
            st.setdefault("module_load_s", {})[k] = time.perf_counter() - t
        return mods[k]

    def health() -> dict:
        dev = torch.device("furiosa:0")
        with torch.no_grad():
            b, s = module_on(0)(x0.to(dev))
        ft.current_stream(dev).synchronize()
        d = max(
            (b.cpu().float() - ref_b.float()).abs().max().item(), (s.cpu().float() - ref_s.float()).abs().max().item()
        )
        return {"ok": d < 1.0, "max_abs_vs_cpu": d}

    first = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else CONFIGS[0][0]
    names = [c[0] for c in CONFIGS]
    if "--from" in sys.argv:
        out = HERE / "status_multistream2.json"
    for name, devs in CONFIGS[names.index(first) :]:
        entry: dict = {"devices": devs}
        try:
            for k in set(devs):
                module_on(k)
            lat: list[list[float]] = [[] for _ in devs]
            errs: list[str] = []
            stop = time.perf_counter() + secs

            def worker(i: int, k: int, lat=lat, errs=errs, stop=stop) -> None:
                dev = torch.device(f"furiosa:{k}")
                m, xd = module_on(k), x0.to(dev)
                stream = ft.current_stream(dev)
                try:
                    with torch.no_grad():
                        while time.perf_counter() < stop:
                            t = time.perf_counter()
                            m(xd)
                            stream.synchronize()
                            lat[i].append(time.perf_counter() - t)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{type(e).__name__}: {str(e)[:300]}")

            th = [threading.Thread(target=worker, args=(i, k)) for i, k in enumerate(devs)]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join(timeout=secs + 60)
            wall = time.perf_counter() - t0
            allv = [v for li in lat for v in li]
            entry |= {
                "n": len(allv),
                "wall_s": wall,
                "throughput_fps": len(allv) / wall,
                "per_thread_n": [len(li) for li in lat],
                "lat_ms_p50": 1000 * pct(allv, 0.5) if allv else None,
                "lat_ms_p95": 1000 * pct(allv, 0.95) if allv else None,
                "lat_ms_p99": 1000 * pct(allv, 0.99) if allv else None,
                "errors": errs,
                "loadavg": os.getloadavg(),
                "smi": subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True).stdout[-200:],
            }
            entry["health_after"] = health()
            entry["state"] = "PASS" if not errs and entry["health_after"]["ok"] else "FAIL"
        except Exception as e:  # noqa: BLE001
            entry |= {"state": "FAIL", "error": f"{type(e).__name__}: {str(e)[:400]}"}
        st["configs"][name] = entry
        out.with_suffix(".tmp").write_text(json.dumps(st, indent=2, default=str))
        os.replace(out.with_suffix(".tmp"), out)
        print(name, entry.get("state"), entry.get("throughput_fps"), flush=True)
        if entry["state"] != "PASS":
            st["stopped_after"] = name
            break
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.with_suffix(".tmp").write_text(json.dumps(st, indent=2, default=str))
    os.replace(out.with_suffix(".tmp"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
