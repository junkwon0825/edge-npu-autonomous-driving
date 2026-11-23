"""Bounded INT8 / FP8 capability probe for the installed furiosa-torch (runs on the RNGD host).

The SDK exposes no vision quantization API (no quantize/calibrate entry point, empty
``get_amp_supported_dtype()``), but its typecast table maps torch.int8 -> tcl.i8 and
torch.float8_e4m3fn -> tcl.f8_e4. This script asks the compiler directly, on a *tiny* conv block
(so each compile is short), which low-precision formulations it can lower and run:

  fp32_control      Conv-BN-SiLU block in FP32 (must pass, otherwise the probe is invalid)
  w8_dequant_bf16   INT8 per-channel weights stored in the graph, dequantised to BF16 before conv
                    (weight-only quantization; compute stays BF16)
  int8_conv         activations and weights as int8 tensors fed to conv2d (true INT8 compute)
  int_mm_1x1        1x1 conv as torch._int_mm(int8, int8) -> int32 (INT8 GEMM formulation)
  fp8_conv          FP8 e4m3 activations/weights cast inside the graph, conv in FP8

Each case runs in its own subprocess with a timeout; the error type and first lines of the
message are recorded. Results go to status_int8_probe.json (atomic write).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

CASES = ["fp32_control", "w8_dequant_bf16", "int8_conv", "int_mm_1x1", "fp8_conv"]
# second round (``--only``): accumulator semantics of the int8 conv and an int32-accumulating variant
EXTRA_CASES = ["int32_conv"]
TIMEOUT_S = 900


def build(case: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    conv = nn.Conv2d(16, 32, 3, padding=1, bias=True).eval()
    x = torch.randn(1, 16, 80, 80)

    if case == "fp32_control":

        class M(nn.Module):
            def forward(self, x):
                return F.silu(conv(x))

        return M().eval(), x, lambda m, x: m(x)

    w = conv.weight.detach()
    scale = w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0
    w_q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)

    if case == "w8_dequant_bf16":

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w_q", w_q)
                self.register_buffer("scale", scale.to(torch.bfloat16))
                self.register_buffer("b", conv.bias.detach().to(torch.bfloat16))

            def forward(self, x):
                w = self.w_q.to(torch.bfloat16) * self.scale
                return F.silu(F.conv2d(x, w, self.b, padding=1))

        return M().eval(), x.to(torch.bfloat16), None

    x_scale = x.abs().max() / 127.0
    x_q = torch.clamp(torch.round(x / x_scale), -127, 127).to(torch.int8)

    if case == "int8_conv":

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w_q", w_q)

            def forward(self, xq):
                return F.conv2d(xq, self.w_q, None, padding=1)

        return M().eval(), x_q, None

    if case == "int32_conv":

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w32", w_q.to(torch.int32))

            def forward(self, xq):
                return F.conv2d(xq.to(torch.int32), self.w32, None, padding=1)

        return M().eval(), x_q, None

    if case == "int_mm_1x1":
        w1 = w_q[:, :, 1, 1].contiguous()  # (32,16)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w1t", w1.t().contiguous())

            def forward(self, xq):
                a = xq.permute(0, 2, 3, 1).reshape(-1, 16)  # (6400,16)
                return torch._int_mm(a, self.w1t)  # int32 (6400,32)

        return M().eval(), x_q, None

    if case == "fp8_conv":

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w8", w.to(torch.float8_e4m3fn))

            def forward(self, x):
                return F.conv2d(x.to(torch.float8_e4m3fn), self.w8, None, padding=1)

        return M().eval(), x, None
    raise ValueError(case)


def run_case(case: str) -> dict:
    import furiosa.torch as ft
    import torch

    m, x, _ = build(case)
    t0 = time.perf_counter()
    compiled = ft.CompileModule.from_module(m, (x,))
    t_compile = time.perf_counter() - t0
    dev = torch.device("furiosa:0")
    runnable = compiled.to(dev)
    xd = x.to(dev)
    out = runnable(xd)
    out = out[0] if isinstance(out, (tuple, list)) else out
    ft.current_stream(dev).synchronize()
    out_c = out.cpu()
    res = {"compile_s": t_compile, "out_dtype": str(out_c.dtype), "out_shape": list(out_c.shape)}
    try:  # numerical check against the host where the CPU kernel exists
        with torch.no_grad():
            ref = m(x)
        res["max_abs_vs_cpu"] = (out_c.float() - ref.float()).abs().max().item()
    except Exception as e:  # e.g. conv2d has no CPU int8 kernel
        res["cpu_reference"] = f"unavailable: {type(e).__name__}: {str(e)[:160]}"
    if case in ("int8_conv", "int32_conv"):
        # exact integer result (float64 is exact here) vs. what the device returned
        w_int = next(b for n, b in m.named_buffers() if n.startswith("w")).double()
        exact = torch.nn.functional.conv2d(x.double(), w_int, None, padding=1)
        o = out_c.double()
        res["accumulator"] = {
            "exact_abs_max": exact.abs().max().item(),
            "frac_equal_exact": (o == exact).double().mean().item(),
            "frac_equal_saturated_int8": (o == exact.clamp(-128, 127)).double().mean().item(),
            "frac_equal_wrapped_int8": (o == ((exact + 128) % 256 - 128)).double().mean().item(),
        }
    return res


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(run_case(sys.argv[2])), flush=True)
        return 0
    cases, out_name = CASES, "status_int8_probe.json"
    if len(sys.argv) == 3 and sys.argv[1] == "--only":
        cases, out_name = sys.argv[2].split(","), "status_int8_probe2.json"
    status = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": {}}
    try:
        import furiosa.torch as ft

        status["sdk"] = ft.version()
    except Exception as e:
        status["sdk"] = f"import failed: {e}"
    for case in cases:
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--case", case], capture_output=True, text=True, timeout=TIMEOUT_S
            )
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if p.returncode == 0 and line:
                entry = {"state": "PASS", **json.loads(line[7:])}
            else:
                err = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
                last = next((ln for ln in reversed(err) if "Error" in ln or "error" in ln), err[-1] if err else "")
                entry = {"state": "FAIL", "returncode": p.returncode, "error": last[:400], "stderr_tail": err[-6:]}
        except subprocess.TimeoutExpired:
            entry = {"state": "TIMEOUT", "timeout_s": TIMEOUT_S}
        entry["wall_s"] = round(time.perf_counter() - t0, 1)
        status["cases"][case] = entry
        with open(out_name + ".tmp", "w") as fh:
            json.dump(status, fh, indent=2)
        os.replace(out_name + ".tmp", out_name)
        print(case, entry["state"], flush=True)
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(out_name + ".tmp", "w") as fh:
        json.dump(status, fh, indent=2)
    os.replace(out_name + ".tmp", out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
