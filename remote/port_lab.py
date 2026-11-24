"""NPU model-porting lab (V2, runs on the RNGD host).

Each case is built, AOT-compiled with ``furiosa.torch.CompileModule`` and executed on ``furiosa:0``
in its own subprocess (timeout), then compared with the same module on the host CPU. Recorded:
compile result, runtime result, output error vs CPU, latency, and the first error line on failure.

Case groups (``--group``):
  ops    single operators in a tiny BF16 block (segmentation / MobileNet operators)
  quant  activation-quantisation formulations (per-tensor / per-channel fake quant, round/clamp)
  seg    segmentation models at 1x3x256x512: small U-Net (ConvTranspose and bilinear variants),
         torchvision LR-ASPP MobileNetV3-Large (raw and with Conv+BN fused). Weights are seeded
         random (porting correctness is NPU vs CPU on the same weights, not task accuracy).
  intro  SDK introspection only (quantisation / calibration / AMP entry points)

  python port_lab.py --group ops            -> status_port_lab_ops.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

TIMEOUT_S = {"ops": 900, "quant": 900, "seg": 3600}
GROUPS = {
    "ops": [
        "control_conv_relu",
        "bn_unfused",
        "hardswish",
        "hardsigmoid",
        "se_block",
        "depthwise_conv",
        "dilated_conv",
        "maxpool",
        "upsample_nearest_x2",
        "upsample_bilinear_x2",
        "interpolate_bilinear_size",
        "conv_transpose_s2",
        "concat_channels",
        "softmax_channels",
        "argmax_channels",
    ],
    "quant": ["fake_quant_per_tensor", "fake_quant_per_channel", "round_clamp_act_q", "int8_act_cast_roundtrip"],
    "seg": ["unet_convtranspose", "unet_bilinear", "lraspp_mbv3_raw", "lraspp_mbv3_fused"],
    # exact rewrites of the unsupported operators, each compared with the ORIGINAL op on CPU
    "fix": ["fix_bilinear2x", "fix_convtranspose_pixelshuffle", "fix_dilated_dense", "pixel_shuffle", "slice_cat_pad"],
    # segmentation models with the rewrites applied (reference: original model on CPU)
    "seg2": ["unet_convtranspose_ported", "unet_bilinear_ported", "lraspp_mbv3_ported"],
}
TIMEOUT_S["fix"] = 900
TIMEOUT_S["seg2"] = 3600


def _tiny(case: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    c = 16
    x = torch.randn(1, c, 40, 40)
    conv = nn.Conv2d(c, c, 3, padding=1)

    def mk(fn, mods=None):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.mods = nn.ModuleList(mods or [conv])

            def forward(self, x):
                return fn(self.mods, x)

        return M().eval()

    table = {
        "control_conv_relu": lambda: mk(lambda m, x: F.relu(m[0](x))),
        "bn_unfused": lambda: mk(lambda m, x: m[1](m[0](x)), [conv, nn.BatchNorm2d(c).eval()]),
        "hardswish": lambda: mk(lambda m, x: F.hardswish(m[0](x))),
        "hardsigmoid": lambda: mk(lambda m, x: F.hardsigmoid(m[0](x))),
        "se_block": lambda: mk(
            lambda m, x: x * F.hardsigmoid(m[1](F.relu(m[0](F.adaptive_avg_pool2d(x, 1))))),
            [nn.Conv2d(c, 8, 1), nn.Conv2d(8, c, 1)],
        ),
        "depthwise_conv": lambda: mk(lambda m, x: m[0](x), [nn.Conv2d(c, c, 3, padding=1, groups=c)]),
        "dilated_conv": lambda: mk(lambda m, x: m[0](x), [nn.Conv2d(c, c, 3, padding=4, dilation=4)]),
        "maxpool": lambda: mk(lambda m, x: F.max_pool2d(m[0](x), 2)),
        "upsample_nearest_x2": lambda: mk(lambda m, x: F.interpolate(m[0](x), scale_factor=2, mode="nearest")),
        "upsample_bilinear_x2": lambda: mk(
            lambda m, x: F.interpolate(m[0](x), scale_factor=2, mode="bilinear", align_corners=False)
        ),
        "interpolate_bilinear_size": lambda: mk(
            lambda m, x: F.interpolate(m[0](x), size=(64, 96), mode="bilinear", align_corners=False)
        ),
        "conv_transpose_s2": lambda: mk(lambda m, x: m[0](x), [nn.ConvTranspose2d(c, c, 2, stride=2)]),
        "concat_channels": lambda: mk(lambda m, x: torch.cat([m[0](x), x], 1)),
        "softmax_channels": lambda: mk(lambda m, x: torch.softmax(m[0](x), 1)),
        "argmax_channels": lambda: mk(lambda m, x: m[0](x).argmax(1).to(torch.float32)),
        # activation quantisation formulations (scale from a fixed calibration value)
        "fake_quant_per_tensor": lambda: mk(
            lambda m, x: torch.fake_quantize_per_tensor_affine(m[0](x), 0.05, 0, -128, 127)
        ),
        "fake_quant_per_channel": lambda: mk(
            lambda m, x: torch.fake_quantize_per_channel_affine(
                m[0](x), torch.full((c,), 0.05), torch.zeros(c, dtype=torch.int32), 1, -128, 127
            )
        ),
        "round_clamp_act_q": lambda: mk(lambda m, x: torch.clamp(torch.round(m[0](x) / 0.05), -128, 127) * 0.05),
        "int8_act_cast_roundtrip": lambda: mk(
            lambda m, x: torch.clamp(torch.round(m[0](x) / 0.05), -128, 127).to(torch.int8).to(x.dtype) * 0.05
        ),
    }
    return table[case](), x


def _unet(up: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))  # fmt: skip

    class UNet(nn.Module):
        def __init__(self, ch=(16, 32, 64, 128), n_cls=2):
            super().__init__()
            self.enc = nn.ModuleList()
            prev = 3
            for c in ch:
                self.enc.append(block(prev, c))
                prev = c
            self.up = nn.ModuleList()
            self.dec = nn.ModuleList()
            for c in reversed(ch[:-1]):
                if up == "convtranspose":
                    self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
                else:
                    self.up.append(nn.Conv2d(prev, c, 1))
                self.dec.append(block(2 * c, c))
                prev = c
            self.head = nn.Conv2d(prev, n_cls, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for i, (u, d) in enumerate(zip(self.up, self.dec, strict=True)):
                if up == "convtranspose":
                    x = u(x)
                elif getattr(self, "use_bil", False):  # ported: exact NPU-friendly bilinear x2
                    x = u(self.bil[i](x))
                else:
                    x = u(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
                x = d(torch.cat([x, skips.pop()], 1))
            return self.head(x)

    torch.manual_seed(0)
    return UNet().eval()


def _fuse(m):
    """Fold every BatchNorm2d into the Conv2d directly before it inside a Sequential (eval mode).

    Unfused BatchNorm does not lower on RNGD (V1 finding); torchvision's Conv2dNormActivation and the
    U-Net blocks are Sequential(conv, bn, act), so a structural fold covers them without tracing.
    """
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for mod in list(m.modules()):
        if isinstance(mod, nn.Sequential):
            keys = list(mod._modules)
            for a, b in zip(keys, keys[1:], strict=False):
                ca, cb = mod._modules[a], mod._modules[b]
                if isinstance(ca, nn.Conv2d) and isinstance(cb, nn.BatchNorm2d):
                    mod._modules[a] = fuse_conv_bn_eval(ca, cb)
                    mod._modules[b] = nn.Identity()
    return m


def _rewrites():
    """Exact NPU-friendly replacements for operators that do not lower on RNGD (built lazily)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def edge_pad(x):  # replicate padding by 1 via slicing + concat
        x = torch.cat([x[:, :, :1], x, x[:, :, -1:]], 2)
        return torch.cat([x[:, :, :, :1], x, x[:, :, :, -1:]], 3)

    class Bilinear2x(nn.Module):
        """bilinear x2 (align_corners=False) == nearest x2 + [1,2,1]/4 separable smoothing with edge
        replication: exact."""

        def __init__(self, c):
            super().__init__()
            k1 = torch.tensor([0.25, 0.5, 0.25])
            self.dw = nn.Conv2d(c, c, 3, groups=c, bias=False)
            self.dw.weight.data[:] = torch.outer(k1, k1)
            self.dw.weight.requires_grad_(False)

        def forward(self, x):
            return self.dw(edge_pad(F.interpolate(x, scale_factor=2, mode="nearest")))

    class ConvT2x(nn.Module):
        """ConvTranspose2d(k=2, s=2) == 1x1 conv to 4*C_out + pixel shuffle: exact."""

        def __init__(self, ct):
            super().__init__()
            cin, cout = ct.weight.shape[0], ct.weight.shape[1]
            self.conv = nn.Conv2d(cin, 4 * cout, 1, bias=ct.bias is not None)
            self.conv.weight.data[:] = ct.weight.data.permute(1, 2, 3, 0).reshape(cout * 4, cin, 1, 1)
            if ct.bias is not None:
                self.conv.bias.data[:] = ct.bias.data.repeat_interleave(4)
            self.ps = nn.PixelShuffle(2)

        def forward(self, x):
            return self.ps(self.conv(x))

    def dense_from_dilated(conv):
        """Dilated conv == conv with a zero-inserted dense kernel: exact, more MACs."""
        d, k = conv.dilation[0], conv.kernel_size[0]
        new = nn.Conv2d(conv.in_channels, conv.out_channels, (k - 1) * d + 1, stride=conv.stride,
                        padding=conv.padding[0], groups=conv.groups, bias=conv.bias is not None)  # fmt: skip
        new.weight.data.zero_()
        new.weight.data[:, :, ::d, ::d] = conv.weight.data
        if conv.bias is not None:
            new.bias.data[:] = conv.bias.data
        return new

    def replace_dilated(m) -> int:
        n = 0
        for name, child in m.named_children():
            if isinstance(child, nn.Conv2d) and child.dilation[0] > 1:
                setattr(m, name, dense_from_dilated(child))
                n += 1
            else:
                n += replace_dilated(child)
        return n

    return Bilinear2x, ConvT2x, dense_from_dilated, replace_dilated, edge_pad


def _wrap(fn, mods):
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList(mods)

        def forward(self, x):
            return fn(self.mods, x)

    return M().eval()


def build_pair(case: str):
    """(npu_module, reference_module, input); reference = ORIGINAL operator/model on CPU fp32."""
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    Bilinear2x, ConvT2x, dense_from_dilated, replace_dilated, edge_pad = _rewrites()
    torch.manual_seed(0)
    c = 16
    x = torch.randn(1, c, 40, 40)
    if case == "fix_bilinear2x":
        ref = _wrap(lambda m, x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), [])
        return _wrap(lambda m, x: m[0](x), [Bilinear2x(c)]), ref, x
    if case == "fix_convtranspose_pixelshuffle":
        ct = nn.ConvTranspose2d(c, 8, 2, stride=2).eval()
        return _wrap(lambda m, x: m[0](x), [ConvT2x(ct)]), _wrap(lambda m, x: m[0](x), [ct]), x
    if case == "fix_dilated_dense":
        dc = nn.Conv2d(c, c, 3, padding=4, dilation=4).eval()
        return _wrap(lambda m, x: m[0](x), [dense_from_dilated(dc)]), _wrap(lambda m, x: m[0](x), [dc]), x
    if case == "pixel_shuffle":
        mm = _wrap(lambda m, x: F.pixel_shuffle(m[0](x), 2), [nn.Conv2d(c, c, 1)])
        return mm, mm, x
    if case == "slice_cat_pad":
        mm = _wrap(lambda m, x: edge_pad(m[0](x)), [nn.Conv2d(c, c, 1)])
        return mm, mm, x
    xs = torch.randn(1, 3, 256, 512, generator=torch.Generator().manual_seed(1))
    if case.startswith("unet"):
        up = case.split("_")[1]
        ref = _fuse(_unet(up))
        npu = copy.deepcopy(ref)
        if up == "convtranspose":
            for i, u in enumerate(npu.up):
                npu.up[i] = ConvT2x(u)
        else:
            npu.bil = nn.ModuleList([Bilinear2x(u.in_channels) for u in npu.up])
            npu.use_bil = True
        return npu.eval(), ref.eval(), xs
    if case == "lraspp_mbv3_ported":
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        torch.manual_seed(0)
        base = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2).eval()

        class Ref(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = base

            def forward(self, x):
                return self.m(x)["out"]

        ref = _fuse(Ref().eval())
        body = copy.deepcopy(ref.m)
        n_dil = replace_dilated(body.backbone)
        head = body.classifier

        class Ported(nn.Module):
            """Dense kernels for the dilated convs, exact bilinear x2 between the two feature levels;
            the final x8 bilinear becomes three exact x2 steps (NOT identical to one x8 step)."""

            def __init__(self):
                super().__init__()
                self.backbone, self.head = body.backbone, head
                self.up_mid = Bilinear2x(head.cbr[0].out_channels)
                self.up_out = nn.ModuleList([Bilinear2x(head.low_classifier.out_channels) for _ in range(3)])
                self.n_dilated_replaced = n_dil

            def forward(self, x):
                f = self.backbone(x)
                low, high = f["low"], f["high"]
                y = self.up_mid(self.head.cbr(high) * self.head.scale(high))
                y = self.head.low_classifier(low) + self.head.high_classifier(y)
                for u in self.up_out:
                    y = u(y)
                return y

        return Ported().eval(), ref, xs
    raise KeyError(case)


def build(case: str):
    import torch

    if case in GROUPS["ops"] or case in GROUPS["quant"]:
        m, x = _tiny(case)
        return m, x, "bf16"
    x = torch.randn(1, 3, 256, 512, generator=torch.Generator().manual_seed(1))
    if case.startswith("unet"):
        return _fuse(_unet(case.split("_")[1])), x, "bf16"
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    torch.manual_seed(0)
    m = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2).eval()

    class Out(torch.nn.Module):  # torchvision returns a dict; the NPU module returns the tensor
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)["out"]

    m = Out(m).eval()
    return (_fuse(m) if case.endswith("fused") else m), x, "bf16"


def run_case(case: str) -> dict:
    import furiosa.torch as ft
    import torch

    pair = case in GROUPS["fix"] or case in GROUPS["seg2"]
    rewrite_err = None
    if pair:
        m, ref_m, x = build_pair(case)
        prec = "bf16"
        with torch.no_grad():
            ref = ref_m(x).float()
            rewrite_err = (m(x).float() - ref).abs().max().item()  # the rewrite itself, CPU fp32
    else:
        m, x, prec = build(case)
        with torch.no_grad():
            ref = m(x).float()
    if prec == "bf16":
        m, xq = m.to(torch.bfloat16), x.to(torch.bfloat16)
    res: dict = {"precision": prec, "input_shape": list(x.shape), "params": sum(p.numel() for p in m.parameters())}
    if pair:
        res |= {"reference": "original operator/model on CPU fp32", "rewrite_max_abs_vs_original_cpu": rewrite_err,
                "n_dilated_replaced": getattr(m, "n_dilated_replaced", None)}  # fmt: skip
    res["modules"] = sorted({type(mod).__name__ for mod in m.modules() if not list(mod.children())})
    try:
        g = torch.fx.symbolic_trace(m).graph
        res["functions"] = sorted({getattr(n.target, "__name__", str(n.target)) for n in g.nodes
                                   if n.op in ("call_function", "call_method")})  # fmt: skip
    except Exception as e:  # noqa: BLE001 - data-dependent control flow (torchvision) is not traceable
        res["functions"] = f"fx trace failed: {type(e).__name__}"
    t0 = time.perf_counter()
    try:
        compiled = ft.CompileModule.from_module(m, (xq,))
    except Exception as e:
        msg = str(e)
        i = msg.find("compile error")
        res |= {"compile": "FAIL", "compile_s": time.perf_counter() - t0, "error": type(e).__name__,
                "error_detail": msg[i : i + 700] if i >= 0 else msg[-700:]}  # fmt: skip
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
                o2 = run(xd)
                stream.synchronize()
                ts.append(time.perf_counter() - t)
        o = out.cpu().float()
    except Exception as e:
        res |= {"runtime": "FAIL", "error": f"{type(e).__name__}: {str(e)[:400]}"}
        return res
    del o2
    ts.sort()
    d = (o - ref).abs()
    res |= {
        "runtime": "PASS",
        "out_shape": list(o.shape),
        "latency_ms_p50": 1000 * ts[len(ts) // 2],
        "latency_ms_p95": 1000 * ts[int(len(ts) * 0.95)],
        "max_abs_vs_cpu_fp32": d.max().item(),
        "mean_abs_vs_cpu_fp32": d.mean().item(),
        "ref_abs_mean": ref.abs().mean().item(),
        "cosine_vs_cpu": torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item(),
    }
    if o.dim() == 4 and o.shape[1] > 1:  # segmentation: per-pixel class agreement
        res["pixel_argmax_agreement"] = (o.argmax(1) == ref.argmax(1)).float().mean().item()
    return res


def introspect() -> dict:
    import importlib
    import pkgutil

    import furiosa.torch as ft

    found = {}
    for mod in pkgutil.walk_packages(ft.__path__, "furiosa.torch."):
        try:
            m = importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001
            found[mod.name] = f"import error: {type(e).__name__}"
            continue
        names = [n for n in dir(m) if any(k in n.lower() for k in ("quant", "calib", "amp", "int8", "fp8", "dtype"))]
        if names:
            found[mod.name] = names
    amp = getattr(ft, "get_amp_supported_dtype", None)
    return {"sdk": ft.version(), "modules_with_quant_names": found, "amp_supported": repr(amp() if amp else None)}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        print("RESULT " + json.dumps(run_case(sys.argv[2])), flush=True)
        return 0
    group = sys.argv[sys.argv.index("--group") + 1] if "--group" in sys.argv else "ops"
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    out_name = f"status_port_lab_{group}{'_rerun' if only else ''}.json"
    status: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "group": group, "cases": {}}
    if group == "intro":
        status["introspection"] = introspect()
        cases: list[str] = []
    else:
        cases = only or GROUPS[group]
    for case in cases:
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, __file__, "--case", case], capture_output=True, text=True,
                               timeout=TIMEOUT_S[group])  # fmt: skip
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if p.returncode == 0 and line:
                entry = json.loads(line[7:])
            else:
                err = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
                last = next((ln for ln in reversed(err) if "Error" in ln or "error" in ln), err[-1] if err else "")
                entry = {"compile": "CRASH", "returncode": p.returncode, "error": last[:400], "stderr_tail": err[-6:]}
        except subprocess.TimeoutExpired:
            entry = {"compile": "TIMEOUT", "timeout_s": TIMEOUT_S[group]}
        entry["wall_s"] = round(time.perf_counter() - t0, 1)
        entry["loadavg"] = os.getloadavg()
        status["cases"][case] = entry
        with open(out_name + ".tmp", "w") as fh:
            json.dump(status, fh, indent=2)
        os.replace(out_name + ".tmp", out_name)
        print(case, entry.get("compile"), entry.get("runtime"), flush=True)
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(out_name + ".tmp", "w") as fh:
        json.dump(status, fh, indent=2)
    os.replace(out_name + ".tmp", out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
