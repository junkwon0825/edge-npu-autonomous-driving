"""Latency statistics and detection agreement metrics (pure numpy, unit-tested)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def latency_stats(samples_s: Sequence[float]) -> dict[str, float]:
    """Seconds in -> milliseconds out (mean, std, p50, p95, p99, min, max) plus implied FPS."""
    a = np.asarray(samples_s, dtype=np.float64) * 1000.0
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "fps_from_mean": float(1000.0 / a.mean()) if a.mean() > 0 else float("inf"),
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes: (N,4) x (M,4) -> (N,M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_detections(ref: np.ndarray, test: np.ndarray, iou_thr: float = 0.5) -> dict[str, float]:
    """Greedy same-class matching of ``test`` against ``ref`` ([x1,y1,x2,y2,conf,cls] rows).

    Returns counts plus the IoU and |conf| deltas of matched pairs. With the GPU FP32 output
    as ``ref`` this measures *agreement*, not accuracy against ground truth.
    """
    ref, test = np.asarray(ref, float).reshape(-1, 6), np.asarray(test, float).reshape(-1, 6)
    tp, ious, dconf = 0, [], []
    used = np.zeros(len(ref), bool)
    for i in np.argsort(-test[:, 4]):
        same = np.where((ref[:, 5] == test[i, 5]) & ~used)[0]
        if same.size == 0:
            continue
        iou = box_iou(test[i, :4][None], ref[same, :4])[0]
        j = int(np.argmax(iou))
        if iou[j] >= iou_thr:
            used[same[j]] = True
            tp += 1
            ious.append(float(iou[j]))
            dconf.append(abs(float(test[i, 4] - ref[same[j], 4])))
    return {"ref": len(ref), "test": len(test), "matched": tp, "ious": ious, "dconf": dconf}


def aggregate_agreement(per_image: list[dict]) -> dict[str, float]:
    ref = sum(d["ref"] for d in per_image)
    test = sum(d["test"] for d in per_image)
    tp = sum(d["matched"] for d in per_image)
    ious = [x for d in per_image for x in d["ious"]]
    dconf = [x for d in per_image for x in d["dconf"]]
    return {
        "images": len(per_image),
        "ref_dets": ref,
        "test_dets": test,
        "matched": tp,
        "recall_vs_ref": tp / ref if ref else float("nan"),
        "precision_vs_ref": tp / test if test else float("nan"),
        "mean_iou_matched": float(np.mean(ious)) if ious else float("nan"),
        "mean_abs_conf_delta": float(np.mean(dconf)) if dconf else float("nan"),
        "max_abs_conf_delta": float(np.max(dconf)) if dconf else float("nan"),
    }
