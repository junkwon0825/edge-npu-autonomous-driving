"""Runs ON the Furiosa RNGD host (copied by scripts/npu_remote.py). Never run locally.

Subcommands (each writes <workdir>/status_<cmd>.json; stdout/stderr go to a log file):
  dryrun   operator compatibility probe of the raw YOLO module (furiosa.torch.dryrun)
  compile  AOT compile via furiosa.torch.CompileModule.from_module (SDK compile cache reused)
  bench    run preprocessed input tensors through the compiled module on furiosa:<n>,
           timing H2D, compute (stream-synchronized) and D2H separately; saves raw outputs
           so the host can apply the same decode/NMS as the GPU path.

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

        info["furiosa_torch"] = ft.version if isinstance(ft.version, str) else str(ft.version)
        info["devices"] = ft.get_available_devices()
    except Exception as e:  # recorded, not hidden
        info["furiosa_import_error"] = repr(e)
    return info


def _build(weights: str, imgsz: int):
    import torch

    from npu_ad.perception.yolo_raw import load_raw_model

    raw, _ = load_raw_model(weights)
    return raw, torch.zeros(1, 3, imgsz, imgsz)


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

    raw, x = _build(a.weights, a.imgsz)
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

    raw, x0 = _build(a.weights, a.imgsz)
    from npu_ad.perception.yolo_raw import to_tensor

    # {"names": [...], "lb_bgr_uint8": (K,H,W,3)} letterboxed on the host with the GPU-path code;
    # to_tensor is the same function the GPU path uses.
    data = torch.load(a.inputs)
    xs = torch.cat([to_tensor(img.numpy()) for img in data["lb_bgr_uint8"]])
    t0 = time.perf_counter()
    compiled = CompileModule.from_module(raw, (x0,))
    load_s = time.perf_counter() - t0
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    compiled = compiled.to(dev)
    to_device_s = time.perf_counter() - t0
    stream = ft.current_stream()

    def run_one(x):
        t_a = time.perf_counter()
        xd = x.unsqueeze(0).to(dev)
        stream.synchronize()
        t_b = time.perf_counter()
        with torch.no_grad():
            out = compiled(xd)
        stream.synchronize()
        t_c = time.perf_counter()
        boxes, scores = (o.cpu() for o in out)
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
        a.workdir / "npu_outputs.pt",
    )
    # Reference: same raw module on the host CPU of the NPU server (numerical check of the NPU)
    with torch.no_grad():
        cpu_b, cpu_s = zip(*(raw(x.unsqueeze(0)) for x in xs), strict=True)
    cb, cs = torch.cat(cpu_b), torch.cat(cpu_s)
    nb, ns = torch.cat(out_boxes), torch.cat(out_scores)
    return {
        "compile_or_cache_load_s": load_s,
        "to_device_s": to_device_s,
        "device": a.device,
        "n_inputs": len(xs),
        "warmup": a.warmup,
        "repeats": a.repeats,
        "timings_s": {"h2d": [t[0] for t in timings], "compute": [t[1] for t in timings],
                      "d2h": [t[2] for t in timings]},
        "npu_vs_cpu": {
            "boxes_max_abs": (nb - cb).abs().max().item(),
            "boxes_mean_abs": (nb - cb).abs().mean().item(),
            "scores_max_abs": (ns - cs).abs().max().item(),
            "scores_mean_abs": (ns - cs).abs().mean().item(),
            "out_dtype": str(nb.dtype),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dryrun", "compile", "bench"])
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workdir", type=Path, default=HERE)
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--inputs", default="inputs.pt")
    ap.add_argument("--device", default="furiosa:0")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    payload = {"cmd": a.cmd, "state": "RUNNING", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _status(a.workdir, a.cmd, payload)
    try:
        payload["env"] = _env()
        payload["result"] = {"dryrun": cmd_dryrun, "compile": cmd_compile, "bench": cmd_bench}[a.cmd](a)
        payload["state"] = "PASS"
    except Exception as e:
        payload["state"] = "FAIL"
        payload["error_type"] = type(e).__name__
        payload["error"] = str(e)[:4000]
        payload["traceback"] = traceback.format_exc()[-6000:]
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _status(a.workdir, a.cmd, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "traceback"}, indent=2, default=str))
    return 0 if payload["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
