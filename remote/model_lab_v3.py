# ruff: noqa: E501  (long result-dict literals kept on one line)
"""V3 NPU engineering lab: non-driving models to broaden operator coverage (runs on the RNGD host).

Cases (seeded random weights; correctness = RNGD BF16 vs the same module on CPU FP32):
  op probes   : layernorm, gelu, bmm (batched matmul), sdpa (scaled_dot_product_attention), linear_gelu
  resnet18    : torchvision ResNet18, BN folded with torch.fx (224x224)
  mbv3_small  : torchvision MobileNetV3-Small, BN folded (224x224)
  vit_tiny    : patch-16 ViT, d=192, 4 layers, 3 heads, explicit softmax(QK^T)V attention
  vit_tiny_sdpa : same with F.scaled_dot_product_attention
Output: status_model_lab_v3.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CASES = ["layernorm", "gelu", "bmm", "sdpa", "linear_gelu", "resnet18", "mbv3_small", "vit_tiny", "vit_tiny_sdpa"]


def build(case: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)

    class Wrap(nn.Module):  # forward(x): nn.Sequential.forward(input) breaks export naming ("input_1")
        def __init__(self, *mods):
            super().__init__()
            self.body = nn.Sequential(*mods)

        def forward(self, x):
            return self.body(x)

    if case == "layernorm":
        return Wrap(nn.Linear(192, 192), nn.LayerNorm(192)).eval(), torch.randn(1, 197, 192)
    if case == "gelu":
        return Wrap(nn.Linear(192, 192), nn.GELU()).eval(), torch.randn(1, 197, 192)
    if case == "linear_gelu":
        return Wrap(nn.Linear(192, 768), nn.GELU(), nn.Linear(768, 192)).eval(), torch.randn(1, 197, 192)

    class Attn(nn.Module):
        def __init__(self, d=192, h=3, sdpa=False):
            super().__init__()
            self.h, self.dh, self.sdpa = h, d // h, sdpa
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

        def forward(self, x):
            b, n, d = x.shape
            q, k, v = self.qkv(x).reshape(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            if self.sdpa:
                o = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            else:
                o = torch.softmax(q @ k.transpose(-2, -1) * self.dh**-0.5, dim=-1) @ v
            return self.proj(o.transpose(1, 2).contiguous().reshape(b, n, d))

    if case == "bmm":

        class B(nn.Module):
            def forward(self, x):
                return x @ x.transpose(-2, -1)

        return B().eval(), torch.randn(1, 3, 197, 64)
    if case == "sdpa":
        return Attn(sdpa=True).eval(), torch.randn(1, 197, 192)
    if case.startswith("vit_tiny"):
        sdpa = case.endswith("sdpa")

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1, self.n2 = nn.LayerNorm(192), nn.LayerNorm(192)
                self.attn = Attn(sdpa=sdpa)
                self.mlp = nn.Sequential(nn.Linear(192, 768), nn.GELU(), nn.Linear(768, 192))

            def forward(self, x):
                x = x + self.attn(self.n1(x))
                return x + self.mlp(self.n2(x))

        class ViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch = nn.Conv2d(3, 192, 16, stride=16)
                self.pos = nn.Parameter(torch.randn(1, 196, 192) * 0.02)
                self.blocks = nn.Sequential(*[Block() for _ in range(4)])
                self.norm, self.head = nn.LayerNorm(192), nn.Linear(192, 1000)

            def forward(self, x):
                x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
                return self.head(self.norm(self.blocks(x)).mean(1))

        return ViT().eval(), torch.randn(1, 3, 224, 224)
    import torch.fx.experimental.optimization as opt
    import torchvision.models as tvm

    m = tvm.resnet18(weights=None) if case == "resnet18" else tvm.mobilenet_v3_small(weights=None)
    return opt.fuse(m.eval()), torch.randn(1, 3, 224, 224)


def run_case(case: str) -> dict:
    import furiosa.torch as ft
    import torch

    m, x = build(case)
    with torch.no_grad():
        ref = m(x).float()
    m, xq = m.to(torch.bfloat16), x.to(torch.bfloat16)
    res: dict = {"input_shape": list(x.shape), "params": sum(p.numel() for p in m.parameters()),
                 "modules": sorted({type(s).__name__ for s in m.modules() if not list(s.children())})}  # fmt: skip
    t0 = time.perf_counter()
    try:
        compiled = ft.CompileModule.from_module(m, (xq,))
    except Exception as e:
        msg = " ".join(str(e).split())
        i = msg.find("Caused by")
        j = msg.find("Not importable")
        k = i if i >= 0 else (j - 120 if j >= 0 else -500)
        res |= {"compile": "FAIL", "compile_s": time.perf_counter() - t0, "error": type(e).__name__,
                "error_detail": msg[max(k, 0): max(k, 0) + 500].split("Location:")[0]}  # fmt: skip
        return res
    res |= {"compile": "PASS", "compile_s": time.perf_counter() - t0}
    dev = torch.device("furiosa:0")
    stream = ft.current_stream(dev)
    try:
        run, xd = compiled.to(dev), xq.to(dev)
        with torch.no_grad():
            out = run(xd)
            out = out[0] if isinstance(out, (tuple, list)) else out
            stream.synchronize()
            ts = []
            for _ in range(50):
                t = time.perf_counter()
                run(xd)
                stream.synchronize()
                ts.append(time.perf_counter() - t)
        o = out.cpu().float()
    except Exception as e:
        return res | {"runtime": "FAIL", "error_detail": f"{type(e).__name__}: {str(e)[:400]}"}
    ts.sort()
    d = (o - ref).abs()
    res |= {"runtime": "PASS", "latency_ms_p50": 1000 * ts[25], "latency_ms_p95": 1000 * ts[47],
            "max_abs_vs_cpu": d.max().item(), "mean_abs_vs_cpu": d.mean().item(),
            "cosine_vs_cpu": torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()}  # fmt: skip
    if o.dim() == 2:
        res["top1_agreement"] = (o.argmax(1) == ref.argmax(1)).float().mean().item()
    return res


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(run_case(sys.argv[2])), flush=True)
        return 0
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    out = "status_model_lab_v3_rerun.json" if only else "status_model_lab_v3.json"
    st: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": {}}
    for case in only or CASES:
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, __file__, "--case", case], capture_output=True, text=True, timeout=1800)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
            entry = json.loads(line[7:]) if p.returncode == 0 and line else {
                "compile": "CRASH", "stderr_tail": [ln for ln in p.stderr.splitlines() if ln.strip()][-5:]}  # fmt: skip
        except subprocess.TimeoutExpired:
            entry = {"compile": "TIMEOUT"}
        entry["wall_s"] = round(time.perf_counter() - t0, 1)
        entry["loadavg"] = os.getloadavg()
        st["cases"][case] = entry
        with open(out + ".tmp", "w") as fh:
            json.dump(st, fh, indent=2)
        os.replace(out + ".tmp", out)
        print(case, entry.get("compile"), entry.get("runtime"), flush=True)
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(out + ".tmp", "w") as fh:
        json.dump(st, fh, indent=2)
    os.replace(out + ".tmp", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
