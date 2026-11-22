"""Gate E (part 1): check the raw-head split reproduces ultralytics' own decoded output.

Compares, on the same input tensor, ``RawDecoder.decode(YoloRaw(x))`` with the decoded tensor of
the unmodified ultralytics DetectionModel forward, then compares the post-NMS detections.
Writes a JSON result to reports/evidence/.
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
