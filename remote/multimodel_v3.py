# ruff: noqa: E501  (long result-dict literals kept on one line)
"""V3: multi-model scheduling on one RNGD (runs on the NPU host; device-resident inputs, no network).

Models: YOLO26n BF16 (detection, 1x3x640x640) and the ported lane U-Net (3x3 stride-2 downsampling,
ConvTranspose -> conv+PixelShuffle, base 32, 1x3x256x512, BF16, seeded weights - timing only).
Frames arrive at ``fps`` (camera clock); every request records arrival, start and end, so result age
(end - frame arrival), queue delay, drops and throughput are measured per model.

Strategies (each run for ``--secs`` seconds per load level):
  A_sequential      one worker, one PE: for every frame run YOLO then lane (both every frame)
  B_shared_pe       two workers (YOLO, lane), both on furiosa:0, each keeps only its newest frame
  B_separate_pe     two workers, YOLO on furiosa:0, lane on furiosa:1
  C_rate_decoupled  one worker, one PE: YOLO every frame, lane every 2nd frame
  D_priority        one worker, one PE, two latest-frame slots; first half "cruise" (YOLO every 2nd
                    frame, lane every 4th), second half "hazard" (YOLO every frame with priority, lane
                    every 2nd frame, served only when no YOLO frame waits)
Load levels: camera 30 Hz (realistic single camera) and 400 Hz (stress, ~ multi-camera aggregate).
Output: status_multimodel_v3.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "src"), HERE]


def pct(v, q):
    v = sorted(v)
    return round(1000 * v[min(len(v) - 1, int(q * len(v)))], 3) if v else None


class Slot:
    """Latest-frame slot (size 1): a newer frame replaces a waiting one (counted as dropped)."""

    def __init__(self):
        self.item, self.dropped, self.cv = None, 0, threading.Condition()

    def put(self, x):
        with self.cv:
            if self.item is not None:
                self.dropped += 1
            self.item = x
            self.cv.notify_all()

    def take(self):
        with self.cv:
            x, self.item = self.item, None
            return x


def main() -> int:
    import furiosa.torch as ft
    import lane_port_v3 as lp
    import torch
    from furiosa.torch import CompileModule
    from npu_job import _build

    secs = float(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 6.0
    torch.set_num_threads(2)
    st: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sdk": ft.version(), "secs": secs,
                "loadavg_start": os.getloadavg(), "runs": {}}  # fmt: skip
    yraw, yx = _build(os.path.join(HERE, "yolo26n.pt"), 640, "bf16")
    lane, _, lx = lp.build("unet_s3_256x512_c32")
    lane, lx = lane.to(torch.bfloat16), lx.to(torch.bfloat16)
    mods: dict = {}

    def mod(name: str, k: int):
        if (name, k) not in mods:
            t0 = time.perf_counter()
            src, x = (yraw, yx) if name == "yolo" else (lane, lx)
            m = CompileModule.from_module(src, (x,)).to(torch.device(f"furiosa:{k}"))
            mods[(name, k)] = (m, x.to(torch.device(f"furiosa:{k}")))
            st.setdefault("load_s", {})[f"{name}@{k}"] = round(time.perf_counter() - t0, 2)
        return mods[(name, k)]

    def call(name: str, k: int):
        m, x = mod(name, k)
        with torch.no_grad():
            m(x)
        ft.current_stream(torch.device(f"furiosa:{k}")).synchronize()

    # isolated single-model latency
    iso = {}
    for name in ("yolo", "lane"):
        for _ in range(10):
            call(name, 0)
        ts = []
        for _ in range(200):
            t = time.perf_counter()
            call(name, 0)
            ts.append(time.perf_counter() - t)
        iso[name] = {"p50_ms": pct(ts, 0.5), "p95_ms": pct(ts, 0.95), "p99_ms": pct(ts, 0.99)}
    st["isolated"] = iso
    for k in (1,):  # warm the second PE
        call("yolo", k)
        call("lane", k)

    def run(strategy: str, fps: float) -> dict:
        rec = defaultdict(list)  # model -> [(arrival, start, end)]
        stop = threading.Event()
        period = 1.0 / fps
        t0 = time.perf_counter() + 0.05
        slots = {"yolo": Slot(), "lane": Slot()}
        hazard_at = t0 + secs / 2

        def due(model: str, n: int, t_arr: float) -> bool:
            if strategy in ("A_sequential", "B_shared_pe", "B_separate_pe"):
                return True
            if strategy == "C_rate_decoupled":
                return model == "yolo" or n % 2 == 0
            hazard = t_arr >= hazard_at
            if model == "yolo":
                return hazard or n % 2 == 0
            return n % (2 if hazard else 4) == 0

        def camera():
            n = 0
            while not stop.is_set():
                t_arr = t0 + n * period
                dt = t_arr - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                if time.perf_counter() > t0 + secs:
                    break
                for model in ("yolo", "lane"):
                    if due(model, n, t_arr):
                        slots[model].put((n, t_arr))
                n += 1
            stop.set()

        def worker(models: list[str], pe_of: dict, prio: bool):
            while not (stop.is_set() and all(slots[m].item is None for m in models)):
                order = models if prio else sorted(models, key=lambda m: (slots[m].item or (0, 9e18))[1])
                got = None
                for m in order:
                    it = slots[m].take()
                    if it is not None:
                        got = (m, it)
                        break
                if got is None:
                    time.sleep(0.0002)
                    continue
                m, (n, t_arr) = got
                s = time.perf_counter()
                call(m, pe_of[m])
                rec[m].append((t_arr, s, time.perf_counter()))
                if strategy == "A_sequential" and m == "yolo":
                    it = slots["lane"].take()  # sequential pair: lane of the same frame right after YOLO
                    if it is not None:
                        s2 = time.perf_counter()
                        call("lane", pe_of["lane"])
                        rec["lane"].append((it[1], s2, time.perf_counter()))

        if strategy == "B_shared_pe":
            ws = [threading.Thread(target=worker, args=(["yolo"], {"yolo": 0}, True)),
                  threading.Thread(target=worker, args=(["lane"], {"lane": 0}, True))]  # fmt: skip
        elif strategy == "B_separate_pe":
            ws = [threading.Thread(target=worker, args=(["yolo"], {"yolo": 0}, True)),
                  threading.Thread(target=worker, args=(["lane"], {"lane": 1}, True))]  # fmt: skip
        else:
            prio = strategy == "D_priority" or strategy == "A_sequential"
            ws = [threading.Thread(target=worker, args=(["yolo", "lane"], {"yolo": 0, "lane": 0}, prio))]
        cam = threading.Thread(target=camera)
        for w in ws:
            w.start()
        cam.start()
        cam.join()
        for w in ws:
            w.join(timeout=10)
        out = {}
        for m in ("yolo", "lane"):
            r = rec[m]
            ages = [e - a for a, s, e in r]
            qd = [s - a for a, s, e in r]
            lat = [e - s for a, s, e in r]
            e = {"served": len(r), "dropped": slots[m].dropped, "rate_hz": round(len(r) / secs, 1),
                 "age_p50_ms": pct(ages, 0.5), "age_p95_ms": pct(ages, 0.95), "age_p99_ms": pct(ages, 0.99),
                 "queue_p95_ms": pct(qd, 0.95), "lat_p50_ms": pct(lat, 0.5), "lat_p95_ms": pct(lat, 0.95)}  # fmt: skip
            if strategy == "D_priority":
                hz = [e2 - a for a, s, e2 in r if a >= hazard_at]
                cr = [e2 - a for a, s, e2 in r if a < hazard_at]
                e |= {"age_p95_ms_cruise": pct(cr, 0.95), "age_p95_ms_hazard": pct(hz, 0.95),
                      "rate_hz_cruise": round(len(cr) / (secs / 2), 1), "rate_hz_hazard": round(len(hz) / (secs / 2), 1)}  # fmt: skip
            out[m] = e
        busy = sum(e - s for m in rec for a, s, e in rec[m])
        out["device_busy_frac_sum"] = round(busy / secs, 3)  # summed over PEs used
        out["loadavg"] = os.getloadavg()
        return out

    for fps in (30.0, 400.0):
        for strat in ("A_sequential", "B_shared_pe", "B_separate_pe", "C_rate_decoupled", "D_priority"):
            key = f"{strat}@{int(fps)}Hz"
            try:
                st["runs"][key] = run(strat, fps)
            except Exception as e:  # noqa: BLE001
                st["runs"][key] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            r = st["runs"][key]
            print(key, json.dumps({m: {k: r[m][k] for k in ("rate_hz", "dropped", "age_p95_ms")} for m in ("yolo", "lane")}
                                   if "yolo" in r else r), flush=True)  # fmt: skip
            with open("status_multimodel_v3.json.tmp", "w") as fh:
                json.dump(st, fh, indent=2)
            os.replace("status_multimodel_v3.json.tmp", "status_multimodel_v3.json")
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open("status_multimodel_v3.json.tmp", "w") as fh:
        json.dump(st, fh, indent=2)
    os.replace("status_multimodel_v3.json.tmp", "status_multimodel_v3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
