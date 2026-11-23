"""Runs ON the Furiosa RNGD host (copied by scripts/npu_remote.py). Never run locally.

Subcommands (each writes <workdir>/status_<cmd>.json; stdout/stderr go to a log file):
  dryrun   operator compatibility probe of the raw YOLO module (furiosa.torch.dryrun)
  compile  AOT compile via furiosa.torch.CompileModule.from_module (SDK compile cache reused)
  bench    run preprocessed input tensors through the compiled module on furiosa:<n>,
           timing H2D, compute (stream-synchronized) and D2H separately; saves raw outputs
           so the host can apply the same decode/NMS as the GPU path.
  perf     cold-start latency, sequential vs. pipelined (no per-frame sync) throughput, and
           board power / device memory sampled from ``furiosa-smi`` while idle and under load.

``--dtype``: fp32 | bf16 (precision reduction) | w8 (weight-only INT8: per-output-channel
symmetric INT8 weights stored in the graph and dequantised to BF16 before each conv; activations
and compute stay BF16; no calibration data needed).

Only documented public names found in the installed SDK are used: CompileModule.from_module,
CompileModule.to(device), furiosa.torch.current_stream().synchronize(), dryrun.dryrun.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


def _status(workdir: Path, cmd: str, payload: dict) -> None:
    tmp = workdir / f"status_{cmd}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(workdir / f"status_{cmd}.json")


def _env() -> dict:
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__, "host": platform.node()}
    try:
        import furiosa.torch as ft

        info["furiosa_torch"] = ft.version() if callable(ft.version) else str(ft.version)
        info["devices"] = ft.get_available_devices()
    except Exception as e:  # recorded, not hidden
        info["furiosa_import_error"] = repr(e)
    return info


def _w8_quantize(module) -> int:
    """Replace every nn.Conv2d by an INT8-weight conv (per-output-channel absmax scale)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class W8Conv2d(nn.Module):
        def __init__(self, conv: nn.Conv2d):
            super().__init__()
            w = conv.weight.detach().float()
            scale = (w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0).clamp(min=1e-12)
            self.register_buffer("w_q", torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8))
            self.register_buffer("scale", scale.to(torch.bfloat16))
            b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
            self.register_buffer("bias", b.to(torch.bfloat16))
            self.stride, self.padding = conv.stride, conv.padding
            self.dilation, self.groups = conv.dilation, conv.groups

        def forward(self, x):
            w = self.w_q.to(torch.bfloat16) * self.scale
            return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, W8Conv2d(child))
            n += 1
        else:
            n += _w8_quantize(child)
    return n


def _build(weights: str, imgsz: int, dtype: str = "fp32"):
    import torch

    from npu_ad.perception.yolo_raw import load_raw_model

    raw, _ = load_raw_model(weights)
    x = torch.zeros(1, 3, imgsz, imgsz)
    if dtype == "bf16":  # precision reduction experiment, not INT8 quantization
        raw, x = raw.to(torch.bfloat16), x.to(torch.bfloat16)
    elif dtype == "w8":  # weight-only INT8, BF16 compute
        raw = raw.to(torch.bfloat16)
        raw.n_w8_convs = _w8_quantize(raw)
        x = x.to(torch.bfloat16)
    return raw, x


def _smi_sample() -> dict:
    """One furiosa-smi reading for npu0: board power (W), temperature, device memory (GiB)."""
    import re
    import subprocess

    out = {}
    info = subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True, timeout=10).stdout
    m = re.search(r"\|\s*rngd\s*\|\s*npu0\s*\|.*?\|\s*([\d.]+)\s*\S*C\s*\|\s*([\d.]+)\s*W", info)
    if m:
        out["temp_c"], out["power_w"] = float(m.group(1)), float(m.group(2))
    st = subprocess.run(["furiosa-smi", "status"], capture_output=True, text=True, timeout=10).stdout
    m = re.search(r"npu0\s*\|\s*\w+\s*\|\s*([\d.]+)/([\d.]+) GiB", st)
    if m:
        out["mem_used_gib"], out["mem_total_gib"] = float(m.group(1)), float(m.group(2))
    return out


class SmiSampler:
    def __init__(self, period_s: float = 0.5):
        import threading

        self.period, self.samples, self._halt = period_s, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._halt.is_set():
            try:
                self.samples.append({"t": time.time(), **_smi_sample()})
            except Exception as e:  # recorded, sampling continues
                self.samples.append({"t": time.time(), "error": repr(e)})
            self._halt.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._t.join(timeout=15)

    def summary(self) -> dict:
        p = [s["power_w"] for s in self.samples if "power_w" in s]
        mem = [s["mem_used_gib"] for s in self.samples if "mem_used_gib" in s]
        return {
            "n": len(self.samples),
            "power_w_mean": sum(p) / len(p) if p else None,
            "power_w_max": max(p) if p else None,
            "mem_used_gib_max": max(mem) if mem else None,
            "errors": sum("error" in s for s in self.samples),
        }


def cmd_dryrun(a) -> dict:
    import furiosa.torch as ft
    import torch

    raw, x = _build(a.weights, a.imgsz)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with torch.no_grad(), contextlib.redirect_stdout(buf):
        res = ft.dryrun.dryrun(raw, x, verbose=True, trace_only=a.trace_only)
        with contextlib.suppress(Exception):
            ft.dryrun.print_op_statistics(res)
        with contextlib.suppress(Exception):
            ft.dryrun.print_compile_statistics(res)
    (a.workdir / "dryrun_report.txt").write_text(buf.getvalue(), encoding="utf-8")
    stats = getattr(res, "statistics", None)
    return {"elapsed_s": time.perf_counter() - t0, "statistics": repr(stats), "trace_only": a.trace_only}


def cmd_compile(a) -> dict:
    import torch
    from furiosa.torch import CompileModule

    raw, x = _build(a.weights, a.imgsz, a.dtype)
    with torch.no_grad():
        cpu_boxes, cpu_scores = raw(x)
    t0 = time.perf_counter()
    compiled = CompileModule.from_module(raw, (x,))
    return {
        "compile_s": time.perf_counter() - t0,
        "compiled_type": type(compiled).__name__,
        "cpu_out_shapes": [list(cpu_boxes.shape), list(cpu_scores.shape)],
    }


def cmd_bench(a) -> dict:
    import furiosa.torch as ft
    import torch
    from furiosa.torch import CompileModule

    raw, x0 = _build(a.weights, a.imgsz, a.dtype)
    from npu_ad.perception.yolo_raw import to_tensor

    # {"names": [...], "lb_bgr_uint8": (K,H,W,3)} letterboxed on the host with the GPU-path code;
    # to_tensor is the same function the GPU path uses.
    data = torch.load(a.inputs)
    xs = torch.cat([to_tensor(img.numpy()) for img in data["lb_bgr_uint8"]]).to(x0.dtype)
    t0 = time.perf_counter()
    compiled = CompileModule.from_module(raw, (x0,))
    load_s = time.perf_counter() - t0
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    compiled = compiled.to(dev)
    to_device_s = time.perf_counter() - t0
    stream = ft.current_stream(dev)

    def run_one(x):
        t_a = time.perf_counter()
        xd = x.unsqueeze(0).to(dev)
        stream.synchronize()
        t_b = time.perf_counter()
        with torch.no_grad():
            out = compiled(xd)
        stream.synchronize()
        t_c = time.perf_counter()
        boxes, scores = (o.cpu().float() for o in out)
        t_d = time.perf_counter()
        return boxes, scores, (t_b - t_a, t_c - t_b, t_d - t_c)

    for i in range(a.warmup):
        run_one(xs[i % len(xs)])
    timings, out_boxes, out_scores = [], [], []
    for _ in range(a.repeats):
        for i in range(len(xs)):
            b, s, t = run_one(xs[i])
            timings.append(t)
            if len(out_boxes) < len(xs):
                out_boxes.append(b)
                out_scores.append(s)
    torch.save(
        {"names": data["names"], "boxes": torch.cat(out_boxes), "scores": torch.cat(out_scores)},
        a.workdir / f"npu_outputs_{a.dtype}{a.tag}.pt",
    )
    # Reference: same raw module on the host CPU of the NPU server (numerical check of the NPU)
    with torch.no_grad():
        cpu_b, cpu_s = zip(*(raw(x.unsqueeze(0)) for x in xs), strict=True)
    cb, cs = torch.cat(cpu_b).float(), torch.cat(cpu_s).float()
    nb, ns = torch.cat(out_boxes), torch.cat(out_scores)
    return {
        "compile_or_cache_load_s": load_s,
        "to_device_s": to_device_s,
        "device": a.device,
        "n_inputs": len(xs),
        "warmup": a.warmup,
        "repeats": a.repeats,
        "timings_s": {
            "h2d": [t[0] for t in timings],
            "compute": [t[1] for t in timings],
            "d2h": [t[2] for t in timings],
        },
        "npu_vs_cpu": {
            "boxes_max_abs": (nb - cb).abs().max().item(),
            "boxes_mean_abs": (nb - cb).abs().mean().item(),
            "scores_max_abs": (ns - cs).abs().max().item(),
            "scores_mean_abs": (ns - cs).abs().mean().item(),
            "out_dtype": str(nb.dtype),
        },
    }


def cmd_perf(a) -> dict:
    import furiosa.torch as ft
    import torch
    from furiosa.torch import CompileModule

    raw, x0 = _build(a.weights, a.imgsz, a.dtype)
    from npu_ad.perception.yolo_raw import to_tensor

    data = torch.load(a.inputs)
    xs = torch.cat([to_tensor(img.numpy()) for img in data["lb_bgr_uint8"]]).to(x0.dtype)
    import os

    res: dict = {
        "dtype": a.dtype,
        "n_w8_convs": getattr(raw, "n_w8_convs", None),
        "host_loadavg_start": os.getloadavg(),
    }
    with SmiSampler() as idle:
        time.sleep(a.idle_s)
    res["idle"] = idle.summary()
    t0 = time.perf_counter()
    compiled = CompileModule.from_module(raw, (x0,))
    res["compile_or_cache_load_s"] = time.perf_counter() - t0
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    compiled = compiled.to(dev)
    res["to_device_s"] = time.perf_counter() - t0
    stream = ft.current_stream(dev)
    xds = [x.unsqueeze(0).to(dev) for x in xs]  # inputs resident on device: compute-only loops
    stream.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        compiled(xds[0])
        stream.synchronize()
        res["first_inference_ms"] = (time.perf_counter() - t0) * 1000
        seq = []
        for i in range(a.n_seq):
            t0 = time.perf_counter()
            compiled(xds[i % len(xds)])
            stream.synchronize()
            seq.append(time.perf_counter() - t0)
        seq.sort()
        res["sequential"] = {
            "n": len(seq),
            "mean_ms": sum(seq) / len(seq) * 1000,
            "p50_ms": seq[len(seq) // 2] * 1000,
            "p99_ms": seq[max(0, int(len(seq) * 0.99) - 1)] * 1000,
            "fps": len(seq) / sum(seq),
        }
        _status(a.workdir, f"perf_partial{'' if a.dtype == 'fp32' else '_' + a.dtype}", res)
        # Sustained loop for power. burst > 1 enqueues several executions before one synchronize;
        # with furiosa-torch 2026.4.0 that failed on 2026-11-23 ("failed to synchronize stream ...
        # Connection timed out", then "PXI-122 IPC: Duplicate submission" in the next processes),
        # so the default is burst = 1 (synchronize after every frame).
        with SmiSampler() as load:
            t_end = time.perf_counter() + a.load_s
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() < t_end:
                outs = [compiled(xds[(n + k) % len(xds)]) for k in range(a.burst)]
                stream.synchronize()
                n += a.burst
            elapsed = time.perf_counter() - t0
            del outs
        res["sustained"] = {"burst": a.burst, "frames": n, "seconds": elapsed, "fps": n / elapsed}
        res["under_load"] = load.summary()
    res["host_loadavg_end"] = os.getloadavg()
    res["host_cpus"] = os.cpu_count()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dryrun", "compile", "bench", "perf"])
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workdir", type=Path, default=HERE)
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--inputs", default="inputs.pt")
    ap.add_argument("--device", default="furiosa:0")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", choices=["fp32", "bf16", "w8"], default="fp32")
    ap.add_argument("--n-seq", type=int, default=500)
    ap.add_argument("--burst", type=int, default=1)
    ap.add_argument("--load-s", type=float, default=20.0)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--tag", default="", help="suffix for status/output files, e.g. _replay")
    a = ap.parse_args()
    payload = {"cmd": a.cmd, "state": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    tag = (a.cmd if a.dtype == "fp32" else f"{a.cmd}_{a.dtype}") + a.tag
    _status(a.workdir, tag, payload)
    try:
        payload["env"] = _env()
        cmds = {"dryrun": cmd_dryrun, "compile": cmd_compile, "bench": cmd_bench, "perf": cmd_perf}
        payload["result"] = cmds[a.cmd](a)
        payload["state"] = "PASS"
    except Exception as e:
        payload["state"] = "FAIL"
        payload["error_type"] = type(e).__name__
        payload["error"] = str(e)[:4000]
        payload["traceback"] = traceback.format_exc()[-6000:]
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload["dtype"] = a.dtype
    _status(a.workdir, tag, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "traceback"}, indent=2, default=str))
    return 0 if payload["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
