"""Phase 5 / Gate E: GPU vs NPU comparison on identical inputs and identical host post-processing.

Inputs:
  outputs/gpu/fp32_clean/raw_outputs.pt   raw head tensors from the CUDA run (first pass)
  outputs/npu/npu_outputs_fp32.pt              raw head tensors from furiosa:0 (fetched from the NPU host)
  reports/evidence/npu_bench.json          NPU timings (status_bench.json from the host)
Both raw sets are decoded with the same RawDecoder/NMS and compared per image (eval split only for
detection agreement; latency over all frames). The GPU FP32 output is the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from npu_ad.config import load_config, resolve
from npu_ad.evaluation.metrics import aggregate_agreement, latency_stats, match_detections
from npu_ad.perception.yolo_raw import RawDecoder, load_raw_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    a = ap.parse_args()
    sfx = "" if a.dtype == "fp32" else f"_{a.dtype}"
    cfg = load_config()
    m = cfg["model"]
    _, head = load_raw_model(str(resolve(m["weights"])))
    dec = RawDecoder(head, m["imgsz"])
    gpu = torch.load(resolve("outputs/gpu/fp32_clean/raw_outputs.pt"))
    npu = torch.load(resolve(f"outputs/npu/npu_outputs_{a.dtype}.pt"))
    assert gpu["names"] == npu["names"], "input order mismatch"
    manifest = json.loads(resolve("outputs/dataset/manifest.json").read_text(encoding="utf-8"))
    split = {f["path"]: f["split"] for f in manifest["frames"]}

    raw_diff = {
        "boxes_max_abs": (gpu["boxes"] - npu["boxes"]).abs().max().item(),
        "boxes_mean_abs": (gpu["boxes"] - npu["boxes"]).abs().mean().item(),
        "scores_logit_max_abs": (gpu["scores"] - npu["scores"]).abs().max().item(),
        "scores_logit_mean_abs": (gpu["scores"] - npu["scores"]).abs().mean().item(),
        "sigmoid_scores_max_abs": (gpu["scores"].sigmoid() - npu["scores"].sigmoid()).abs().max().item(),
    }
    per_split: dict[str, list] = {"eval": [], "calib": []}
    for i, name in enumerate(gpu["names"]):
        g = dec(gpu["boxes"][i : i + 1], gpu["scores"][i : i + 1], m["conf"], m["iou"], m["max_det"])[0].numpy()
        n = dec(npu["boxes"][i : i + 1], npu["scores"][i : i + 1], m["conf"], m["iou"], m["max_det"])[0].numpy()
        per_split[split[name]].append(match_detections(g, n, 0.5))
    bench = json.loads(resolve(f"reports/evidence/npu_bench{sfx}.json").read_text(encoding="utf-8"))
    t = bench["result"]["timings_s"]
    dev_e2e = np.array(t["h2d"]) + np.array(t["compute"]) + np.array(t["d2h"])
    gpu_b = json.loads(resolve("reports/evidence/gpu_benchmark_fp32_clean.json").read_text(encoding="utf-8"))
    weights = Path(resolve(m["weights"]))
    result = {
        "reference": "GPU FP32 (CUDA) raw-split outputs, same host decode/NMS",
        "raw_tensor_diff_npu_vs_gpu": raw_diff,
        "npu_vs_cpu_on_npu_host": bench["result"]["npu_vs_cpu"],
        "detection_agreement_eval_split": aggregate_agreement(per_split["eval"]),
        "detection_agreement_calib_split": aggregate_agreement(per_split["calib"]),
        "npu_latency": {
            "h2d": latency_stats(t["h2d"]),
            "compute": latency_stats(t["compute"]),
            "d2h": latency_stats(t["d2h"]),
            "device_side_total": latency_stats(dev_e2e.tolist()),
        },
        "gpu_latency": {k: gpu_b["latency"][k] for k in ("h2d", "model", "e2e")},
        "artifact_sizes": {"weights_pt_bytes": weights.stat().st_size},
        "npu_env": bench.get("env"),
        "npu_compile_or_cache_load_s": bench["result"]["compile_or_cache_load_s"],
    }
    out = resolve(f"reports/evidence/gpu_vs_npu{sfx}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
