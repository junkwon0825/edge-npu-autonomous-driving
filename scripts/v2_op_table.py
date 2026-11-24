# ruff: noqa: E501  (long Markdown-table / diagram-label literals are kept on one line for readability)
"""Generate docs/npu_operator_compatibility.md from recorded RNGD compile/run evidence (no hand-typed results)."""

from __future__ import annotations

import json
import re

from npu_ad.config import resolve

SDK = "furiosa-torch 2026.4.0 (rev 5800e2ba24), firmware 2026.4.0 (V1 + V2 + V3 evidence)"
OPS = {  # case -> (operator, shape, workaround / note)
    "control_conv_relu": ("Conv2d 3x3 + ReLU", "1x16x40x40", "baseline"),
    "bn_unfused": ("BatchNorm2d (unfused, eval)", "1x16x40x40", "fold into the preceding conv before export"),
    "hardswish": ("Hardswish", "1x16x40x40", ""),
    "hardsigmoid": ("Hardsigmoid", "1x16x40x40", ""),
    "se_block": ("SE block: AdaptiveAvgPool2d(1), 1x1 convs, Hardsigmoid, broadcast Mul", "1x16x40x40", ""),
    "depthwise_conv": ("Conv2d depthwise 3x3 (groups=C)", "1x16x40x40", ""),
    "dilated_conv": ("Conv2d 3x3 dilation 4", "1x16x40x40", "dense zero-inserted 9x9 kernel (exact, fix_dilated_dense)"),
    "maxpool": ("MaxPool2d 2x2", "1x16x40x40", ""),
    "upsample_nearest_x2": ("interpolate nearest x2", "1x16x40x40", ""),
    "upsample_bilinear_x2": ("interpolate bilinear x2 (align_corners=False)", "1x16x40x40",
                             "nearest x2 + [1,2,1]/4 separable depthwise conv with edge replication (exact)"),
    "interpolate_bilinear_size": ("interpolate bilinear to fixed size 64x96", "1x16x40x40",
                                  "only x2 steps have an exact rewrite; other ratios approximate"),
    "conv_transpose_s2": ("ConvTranspose2d k2 s2", "1x16x40x40", "1x1 conv to 4*C_out + PixelShuffle(2) (exact)"),
    "concat_channels": ("cat (channel dim)", "1x16x40x40 x2", ""),
    "softmax_channels": ("softmax (channel dim)", "1x16x40x40", ""),
    "argmax_channels": ("argmax (channel dim)", "1x16x40x40", "BF16 ties flip some pixels"),
    "fake_quant_per_tensor": ("fake_quantize_per_tensor_affine", "1x16x40x40", "none in-graph; no quantization API"),
    "fake_quant_per_channel": ("fake_quantize_per_channel_affine", "1x16x40x40", "none in-graph"),
    "round_clamp_act_q": ("round / clamp activation quantization", "1x16x40x40", "round is not lowerable"),
    "int8_act_cast_roundtrip": ("round + cast to int8 and back", "1x16x40x40", "round is not lowerable"),
    "fix_bilinear2x": ("REWRITE bilinear x2 (nearest + depthwise smoothing)", "1x16x40x40", "replaces bilinear"),
    "fix_convtranspose_pixelshuffle": ("REWRITE ConvTranspose -> 1x1 conv + PixelShuffle", "1x16x40x40", ""),
    "fix_dilated_dense": ("REWRITE dilated -> dense kernel", "1x16x40x40", "81 vs 9 taps per output"),
    "pixel_shuffle": ("PixelShuffle(2)", "1x16x40x40", ""),
    "slice_cat_pad": ("slicing + cat (replicate pad)", "1x16x40x40", ""),
    "unet_convtranspose": ("U-Net (4 levels, ConvTranspose up), BN folded", "1x3x256x512", "see ported variant"),
    "unet_bilinear": ("U-Net (4 levels, bilinear up), BN folded", "1x3x256x512", "see ported variant"),
    "lraspp_mbv3_raw": ("torchvision LR-ASPP MobileNetV3-L (BN unfused)", "1x3x256x512", "fold BN"),
    "lraspp_mbv3_fused": ("LR-ASPP MobileNetV3-L, BN folded", "1x3x256x512", "dilated convs in the backbone"),
    "unet_convtranspose_ported": ("U-Net with exact ConvTranspose rewrite", "1x3x256x512", "open: tactic lowering"),
    "unet_bilinear_ported": ("U-Net with exact bilinear rewrite", "1x3x256x512", "open: tactic lowering"),
    "lraspp_mbv3_ported": ("LR-ASPP with dense dilated kernels + bilinear rewrites", "1x3x256x512",
                           "open: tensor-group offsets"),
}  # fmt: skip
V1_ROWS = [
    ("YOLO26n", "fused backbone/neck/head convs, SiLU, concat, nearest upsample, split (raw-head split)", "1x3x640x640",
     "FP32/BF16", "PASS", "PASS", "split at the raw head; decode/NMS on host", "npu_compile*.json"),
    ("YOLO26n", "Detect head decode (concat / select_scatter path)", "1x3x640x640", "FP32", "FAIL", "-",
     "moved to host CPU (shared decode code)", "reports/npu_compilation.md"),
    ("YOLO26n", "unfused BatchNorm", "1x3x640x640", "FP32", "FAIL", "-", "Conv+BN fusion", "reports/npu_compilation.md"),
    ("probe", "conv2d on int8 tensors", "1x16x80x80", "int8", "PASS", "wrong (int8 wrap-around accumulation)",
     "none: W8A8 blocked", "npu_int8_probe2.json"),
    ("probe", "conv2d with int32 accumulation", "1x16x80x80", "int8/int32", "FAIL (O117 not yet supported)", "-",
     "none", "npu_int8_probe2.json"),
    ("probe", "aten._int_mm", "1x16x80x80", "int8", "FAIL (not importable)", "-", "none", "npu_int8_probe.json"),
    ("probe", "conv2d FP8 e4m3", "1x16x80x80", "fp8", "PASS", "not evaluated", "-", "npu_int8_probe.json"),
    ("YOLO26n", "W8 weight-only: int8 weights, cast to BF16 in graph", "1x3x640x640", "int8->bf16", "PASS", "PASS",
     "no speed-up (BF16 compute)", "npu_bench_w8.json"),
    ("YOLO26s", "unmodified fused raw head (V3 model-complexity study)", "1x3x640x640", "bf16",
     "FAIL: TransposedLower on a 512x256x3x3 conv; LowLevelTailCrop size_in_bytes <= u16::MAX", "-",
     "SplitConv (see YOLO26s split row)", "v3_npu_compile_yolo26s_bf16.json"),
]  # fmt: skip



V3_FILES = ["v3_npu_lane_port_sweep.json", "v3_npu_lane_port_unet.json", "v3_npu_lane_port_unet2.json",
            "v3_npu_model_lab.json", "v3_npu_model_lab_rerun.json", "v3_npu_split_conv.json"]
V3_OPS = {  # case -> (model, operator / variant, workaround or note)
    "mp_16x64x128": ("probe", "max_pool2d 2x2 (16x64x128)", "pooling limit depends on size and channels"),
    "mp_16x128x256": ("probe", "max_pool2d 2x2 (16x128x256)", ""),
    "mp_8x256x512": ("probe", "max_pool2d 2x2 (8x256x512)", ""),
    "mp_16x256x512": ("probe", "max_pool2d 2x2 (16x256x512)", ""),
    "mp_32x128x256": ("probe", "max_pool2d 2x2 (32x128x256)", ""),
    "mp_4x256x512": ("probe", "max_pool2d 2x2 (4x256x512)", ""),
    "mp_32x256x512": ("probe", "max_pool2d 2x2 (32x256x512)", ""),
    "mp_64x256x512": ("probe", "max_pool2d 2x2 (64x256x512)", ""),
    "mp_24x128x256": ("probe", "max_pool2d 2x2 (24x128x256)", ""),
    "mp_48x128x256": ("probe", "max_pool2d 2x2 (48x128x256)", ""),
    "unet_mp_256x512_c16": ("lane U-Net", "max-pool down, ConvT rewrite, base 16", "compile cache entry of an earlier failed graph"),
    "unet_strided_256x512_c16": ("lane U-Net", "2x2 stride-2 conv down, base 16", "2x2/s2 conv not supported"),
    "unet_avg_256x512_c16": ("lane U-Net", "avg-pool down, base 16", "avg-pool has the same tactic limit"),
    "unet_mp_128x256_c16": ("lane U-Net", "max-pool down, 128x256, base 16", ""),
    "unet_strided_128x256_c16": ("lane U-Net", "2x2 stride-2 conv down, 128x256", ""),
    "unet_strided_256x512_c8": ("lane U-Net", "2x2 stride-2 conv down, base 8", ""),
    "unet_s3_256x512_c16": ("lane U-Net", "3x3 stride-2 conv down, base 16", "WORKS: YOLO-style downsampling"),
    "unet_s3_128x256_c16": ("lane U-Net", "3x3 stride-2 conv down, 128x256, base 16", "WORKS"),
    "unet_mp_256x512_c32": ("lane U-Net", "max-pool down, base 32", "max-pool limit at 256x512"),
    "unet_s3_256x512_c32": ("lane U-Net", "3x3 stride-2 conv down, base 32 (V3 architecture)", "WORKS"),
    "layernorm": ("probe", "Linear + LayerNorm (1x197x192)", "first run failed: harness naming bug, see failures.jsonl"),
    "gelu": ("probe", "Linear + GELU", ""),
    "bmm": ("probe", "batched matmul (1x3x197x64)", ""),
    "sdpa": ("probe", "scaled_dot_product_attention", "needs contiguous q/k/v (view on strided tensor fails in decomposition)"),
    "linear_gelu": ("probe", "MLP Linear-GELU-Linear", ""),
    "resnet18": ("ResNet18", "torchvision ResNet18, BN folded (224x224)", "large 3x3 conv: TransposedLower slice > 64 KiB"),
    "mbv3_small": ("MobileNetV3-S", "torchvision MobileNetV3-Small, BN folded", ""),
    "vit_tiny": ("tiny ViT", "patch-16 ViT d192 x4 (LayerNorm, GELU, softmax attention)", ""),
    "vit_tiny_sdpa": ("tiny ViT", "same with SDPA", "contiguous q/k/v"),
    "conv_256_512": ("probe", "Conv2d 3x3 256->512 (20x20)", "TransposedLower: size_in_bytes <= u16::MAX violated"),
    "conv_256_512_split": ("probe", "SplitConv: 2 x (256->256) + cat", "WORKS: exact rewrite"),
    "yolo26s_split": ("YOLO26s", "YOLO26s raw head, 1 conv split", "WORKS (unsplit YOLO26s fails)"),
    "resnet18_split": ("ResNet18", "ResNet18 with output-channel splits", "still fails: 512 input channels (needs input split)"),
}  # fmt: skip


def main() -> int:
    rows = []
    for group in ("ops", "quant", "fix", "seg", "seg2"):
        d = json.loads(resolve(f"reports/evidence/v2_npu_port_lab_{group}.json").read_text(encoding="utf-8"))
        rerun = resolve(f"reports/evidence/v2_npu_port_lab_{group}_rerun.json")
        details = json.loads(rerun.read_text(encoding="utf-8"))["cases"] if rerun.exists() else {}
        for case, e in d["cases"].items():
            op, shape, work = OPS[case]
            err = details.get(case, {}).get("error_detail") or e.get("error_detail") or e.get("error") or ""
            err = " ".join(err.split())
            for cut in ("Location:", "(cache at"):
                if cut in err:
                    err = err.split(cut)[0] if cut == "Location:" else err.split(")", 1)[-1]
            err = re.sub(r",? ?Node \{.*", "", err)  # drop Python object addresses
            err = err.replace("compile error:", "").replace("|", "/").strip()[:150]
            run = e.get("runtime") or "-"
            if run == "PASS":
                run = f"PASS (cos {e['cosine_vs_cpu']:.5f}, p50 {e['latency_ms_p50']:.3f} ms)"
            model = "seg model" if group.startswith("seg") else ("rewrite" if group == "fix" else "op probe")
            rows.append((model, op, shape, "bf16", e.get("compile", "?") + (f": {err}" if err else ""), run, work,
                         f"v2_npu_port_lab_{group}.json"))  # fmt: skip
    for fname in V3_FILES:
        d = json.loads(resolve(f"reports/evidence/{fname}").read_text(encoding="utf-8"))
        for case, e in d["cases"].items():
            if case not in V3_OPS or (fname == "v3_npu_model_lab.json" and case in ("layernorm", "gelu", "linear_gelu", "sdpa", "vit_tiny_sdpa")):
                continue  # superseded by the corrected re-run
            model, op, work = V3_OPS[case]
            err = " ".join((e.get("error_detail") or "").split())
            err = re.sub(r",? ?Node \{.*", "", err).replace("Caused by:", "").replace("|", "/").strip()[:150]
            run = e.get("runtime") or "-"
            if run == "PASS":
                run = f"PASS (cos {e['cosine_vs_cpu']:.5f}, p50 {e['latency_ms_p50']:.3f} ms)"
            shape = "x".join(str(v) for v in e.get("input_shape", [])) or "-"
            rows.append((model, op, shape, "bf16", e.get("compile", "?") + (f": {err}" if err else ""), run, work, fname))
    head = ["Model", "Operator", "Input shape", "dtype", "Compile", "Runtime (vs CPU)", "Workaround / note", "Evidence"]
    lines = [
        "# NPU operator compatibility database (Furiosa RNGD)",
        "",
        f"SDK: {SDK}. Every row is a recorded compile/run (generated by `scripts/v2_op_table.py`; do not",
        "hand-edit). Runtime correctness is cosine similarity vs the same module on the host CPU (FP32).",
        "Compile errors are the compiler's own message (cache paths and source locations stripped).",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in V1_ROWS + rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    out = resolve("docs/npu_operator_compatibility.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, len(rows) + len(V1_ROWS), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
