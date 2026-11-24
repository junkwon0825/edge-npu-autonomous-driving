# ruff: noqa: E501  (long result-dict literals kept on one line)
"""V3 Phase 1: continue the lane-segmentation port on RNGD (runs on the NPU host, next to port_lab.py).

V2 ended with the ported U-Net failing in ``max_pool2d`` on a 16x256x512 tensor ("no populated
tactics") although max-pool compiles at 16x40x40. This lab localises the size limit and tries model-level
workarounds. Every case: BF16 AOT compile on furiosa:0, run, compare with CPU FP32 on the same weights.

  sweep  : max_pool2d 2x2 at several C x H x W
  unet   : U-Net variants (exact ConvTranspose rewrite) - maxpool / strided conv / avgpool downsampling,
           input 256x512 and 128x256, base width 16 and 8
Output: status_lane_port_v3_<group>.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GROUPS = {
    "sweep": ["mp_16x64x128", "mp_16x128x256", "mp_8x256x512", "mp_16x256x512", "mp_32x128x256", "mp_4x256x512"],
    "unet": ["unet_mp_256x512_c16", "unet_strided_256x512_c16", "unet_avg_256x512_c16", "unet_mp_128x256_c16",
             "unet_strided_128x256_c16", "unet_strided_256x512_c8"],
    # round 2 (from round-1 errors): pooling needs >= 32 channels?  2x2/s2 conv unsupported -> 3x3/s2
    "unet2": ["mp_32x256x512", "mp_64x256x512", "mp_24x128x256", "mp_48x128x256", "unet_s3_256x512_c16",
              "unet_s3_128x256_c16", "unet_mp_256x512_c32", "unet_s3_256x512_c32"],
}  # fmt: skip


def build(case: str):
    import port_lab as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    if case.startswith("mp_"):
        c, h, w = (int(v) for v in case[3:].split("x"))

        class MP(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(c, c, 3, padding=1)

            def forward(self, x):
                return F.max_pool2d(self.conv(x), 2)

        m = MP().eval()
        return m, m, torch.randn(1, c, h, w)
    _, kind, res, cw = case.split("_")
    hh, ww = (int(v) for v in res.split("x"))
    base = int(cw[1:])
    _, ConvT2x, *_ = pl._rewrites()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))  # fmt: skip

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = (base, 2 * base, 4 * base, 8 * base)
            self.enc, self.down, prev = nn.ModuleList(), nn.ModuleList(), 3
            for i, c in enumerate(ch):
                self.enc.append(block(prev, c))
                if i < len(ch) - 1:
                    # model-level change: learned stride-2 conv instead of max-pool (not equivalent)
                    if kind == "strided":
                        self.down.append(nn.Conv2d(c, c, 2, stride=2))
                    elif kind == "s3":  # YOLO-style 3x3 stride-2 downsampling
                        self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                    else:
                        self.down.append(nn.Identity())
                prev = c
            self.up, self.dec = nn.ModuleList(), nn.ModuleList()
            for c in reversed(ch[:-1]):
                self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
                self.dec.append(block(2 * c, c))
                prev = c
            self.head = nn.Conv2d(prev, 2, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    if kind in ("strided", "s3"):
                        x = self.down[i](x)
                    elif kind == "avg":
                        x = F.avg_pool2d(x, 2)
                    else:
                        x = F.max_pool2d(x, 2)
            for u, d in zip(self.up, self.dec, strict=True):
                x = d(torch.cat([u(x), skips.pop()], 1))
            return self.head(x)

    ref = pl._fuse(UNet().eval())
    import copy

    npu = copy.deepcopy(ref)
    for i, u in enumerate(npu.up):
        npu.up[i] = ConvT2x(u)  # exact rewrite (V2)
    return npu.eval(), ref, torch.randn(1, 3, hh, ww, generator=torch.Generator().manual_seed(1))


def run_case(case: str) -> dict:
    import furiosa.torch as ft
    import torch

    m, ref_m, x = build(case)
    with torch.no_grad():
        ref = ref_m(x).float()
        rew = (m(x).float() - ref).abs().max().item()
    m, xq = m.to(torch.bfloat16), x.to(torch.bfloat16)
    res: dict = {"input_shape": list(x.shape), "params": sum(p.numel() for p in m.parameters()),
                 "rewrite_max_abs_vs_reference_cpu": rew}  # fmt: skip
    t0 = time.perf_counter()
    try:
        compiled = ft.CompileModule.from_module(m, (xq,))
    except Exception as e:
        msg = " ".join(str(e).split())
        i = msg.find("Caused by")
        res |= {"compile": "FAIL", "compile_s": time.perf_counter() - t0,
                "error_detail": (msg[i : i + 500] if i >= 0 else msg[-500:]).split("Location:")[0]}  # fmt: skip
        return res
    res |= {"compile": "PASS", "compile_s": time.perf_counter() - t0}
    dev = torch.device("furiosa:0")
    stream = ft.current_stream(dev)
    try:
        run = compiled.to(dev)
        xd = xq.to(dev)
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
        res |= {"runtime": "FAIL", "error_detail": f"{type(e).__name__}: {str(e)[:400]}"}
        return res
    ts.sort()
    d = (o - ref).abs()
    res |= {
        "runtime": "PASS",
        "latency_ms_p50": 1000 * ts[25],
        "latency_ms_p95": 1000 * ts[47],
        "max_abs_vs_cpu": d.max().item(),
        "mean_abs_vs_cpu": d.mean().item(),
        "cosine_vs_cpu": torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item(),
    }
    if o.dim() == 4 and o.shape[1] > 1:
        res["pixel_argmax_agreement"] = (o.argmax(1) == ref.argmax(1)).float().mean().item()
    return res


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(run_case(sys.argv[2])), flush=True)
        return 0
    group = sys.argv[sys.argv.index("--group") + 1]
    out = f"status_lane_port_v3_{group}.json"
    st: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "group": group, "cases": {}}
    for case in GROUPS[group]:
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
