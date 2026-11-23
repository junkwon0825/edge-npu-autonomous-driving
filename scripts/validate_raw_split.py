"""Gate E (part 1): check the raw-head split reproduces ultralytics' own decoded output.

Compares, on the same input tensor, ``RawDecoder.decode(YoloRaw(x))`` with the decoded tensor of
the unmodified ultralytics DetectionModel forward, then compares the post-NMS detections.
Writes a JSON result to reports/evidence/.

``--all`` checks every frame of outputs/dataset/manifest.json against a freshly loaded, *unfused*
ultralytics model (so Conv+BN fusion is validated too, not only the head split).
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import torch

from npu_ad.config import load_config, resolve
from npu_ad.perception.yolo_raw import RawDecoder, letterbox, load_raw_model, to_tensor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/rgb/139438.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/evidence/raw_split_validation.json")
    ap.add_argument("--all", action="store_true", help="all dataset frames vs. the unfused original model")
    args = ap.parse_args()
    if args.all:
        return validate_all(args)

    cfg = load_config()
    m = cfg["model"]
    raw, head = load_raw_model(str(resolve(m["weights"])))
    decoder = RawDecoder(head, m["imgsz"])
    bgr = cv2.imread(str(resolve(args.image)))
    if bgr is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    x = to_tensor(letterbox(bgr, m["imgsz"])[0]).to(args.device)
    raw = raw.to(args.device)

    with torch.no_grad():
        ref = raw.model(x)  # unmodified ultralytics forward (eval): (decoded, raw dict)
        ref = (ref[0] if isinstance(ref, (tuple, list)) else ref).float().cpu()
        boxes, scores = raw(x)
    ours = decoder.decode(boxes, scores)
    diff = (ours - ref).abs()
    det_ours = decoder(boxes, scores, m["conf"], m["iou"], m["max_det"])[0]
    from ultralytics.utils.nms import non_max_suppression

    det_ref = non_max_suppression(ref, m["conf"], m["iou"], max_det=m["max_det"], nc=decoder.nc)[0]
    same_count = det_ours.shape[0] == det_ref.shape[0]
    det_diff = (det_ours - det_ref).abs().max().item() if same_count and det_ours.numel() else None
    result = {
        "image": args.image,
        "device": args.device,
        "raw_shapes": {"boxes": list(boxes.shape), "scores": list(scores.shape)},
        "decoded_shape": list(ours.shape),
        "decoded_max_abs_diff": diff.max().item(),
        "detections_ours": int(det_ours.shape[0]),
        "detections_ref": int(det_ref.shape[0]),
        "detections_max_abs_diff": det_diff,
        "end2end": decoder.end2end,
        "tolerance": 1e-4,
    }
    result["pass"] = bool(result["decoded_max_abs_diff"] <= 1e-4 and same_count and (det_diff or 0) <= 1e-4)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def validate_all(args) -> int:
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression

    from npu_ad.evaluation.metrics import aggregate_agreement, match_detections

    cfg = load_config()
    m = cfg["model"]
    raw, head = load_raw_model(str(resolve(m["weights"])))
    decoder = RawDecoder(head, m["imgsz"])
    orig = YOLO(str(resolve(m["weights"]))).model.eval().to(args.device)  # not fused
    raw = raw.to(args.device)
    manifest = json.loads(resolve("outputs/dataset/manifest.json").read_text(encoding="utf-8"))
    max_dec, max_det, matches, count_mismatch = 0.0, 0.0, [], 0
    for f in manifest["frames"]:
        x = to_tensor(letterbox(cv2.imread(str(resolve(f["path"]))), m["imgsz"])[0]).to(args.device)
        with torch.no_grad():
            ref = orig(x)
            ref = (ref[0] if isinstance(ref, (tuple, list)) else ref).float().cpu()
            boxes, scores = raw(x)
        ours = decoder.decode(boxes, scores)
        max_dec = max(max_dec, (ours - ref).abs().max().item())
        d_ours = decoder(boxes, scores, m["conf"], m["iou"], m["max_det"])[0]
        d_ref = non_max_suppression(ref, m["conf"], m["iou"], max_det=m["max_det"], nc=decoder.nc)[0]
        if d_ours.shape == d_ref.shape and d_ours.numel():
            max_det = max(max_det, (d_ours - d_ref).abs().max().item())
        count_mismatch += d_ours.shape[0] != d_ref.shape[0]
        matches.append(match_detections(d_ref.numpy(), d_ours.numpy()))
    agg = aggregate_agreement(matches)
    result = {
        "frames": len(manifest["frames"]),
        "device": args.device,
        "reference": "unfused ultralytics DetectionModel (YOLO(weights).model), same letterboxed input",
        "decoded_max_abs_diff": max_dec,
        "detections_max_abs_diff_when_same_count": max_det,
        "frames_with_count_mismatch": int(count_mismatch),
        "agreement": agg,
    }
    result["pass"] = bool(count_mismatch == 0 and agg["recall_vs_ref"] == 1.0 and agg["precision_vs_ref"] == 1.0)
    out = resolve(args.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
