"""Device-agnostic YOLO pipeline shared by the GPU and the Furiosa NPU paths.

The model is split at the detection head:

* ``YoloRaw`` runs backbone + neck + head convolutions and returns the *raw* head tensors
  ``boxes`` (B, 4*reg_max, N) and ``scores`` (B, nc, N) logits. This is the part that runs on the
  accelerator (CUDA or ``furiosa:0``). Anchor decoding, sigmoid and NMS are excluded on purpose:
  earlier RNGD experiments failed on the head's concat/decode path (``select_scatter`` lowering
  error), while the raw-head split compiled (see reports/npu_compilation.md).
* ``RawDecoder`` reproduces ultralytics' Detect decoding on the host CPU, so both accelerators
  share byte-identical pre/post-processing. ``tests/`` and ``scripts/validate_raw_split.py``
  check it against ultralytics' own forward pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn


class YoloRaw(nn.Module):
    """Backbone + neck + head convs of an ultralytics DetectionModel, without decode/NMS."""

    def __init__(self, det_model: nn.Module):
        super().__init__()
        self.model = det_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y: list[torch.Tensor | None] = []
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if type(m).__name__ == "Detect":
                heads = m.one2one if m.end2end else m.one2many
                preds = m.forward_head(x, **heads)
                return preds["boxes"], preds["scores"]
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        raise RuntimeError("no Detect head found in model")


def load_raw_model(weights: str, fuse: bool = True) -> tuple[YoloRaw, nn.Module]:
    """Return (raw accelerator module, detect head copy) on CPU in eval mode."""
    from ultralytics import YOLO

    det_model = YOLO(weights).model.float().eval()
    if fuse:
        det_model.fuse(verbose=False)  # folds Conv+BN: unfused BN is unsupported on RNGD
    head = det_model.model[-1]
    return YoloRaw(det_model).eval(), copy.deepcopy(head).cpu().eval()


@dataclass(frozen=True)
class Letterbox:
    """Fixed-size letterbox transform parameters (needed to map boxes back)."""

    gain: float
    pad_x: float
    pad_y: float
    size: int


def letterbox(bgr: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to a fixed square (static NPU input shape)."""
    h, w = bgr.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = round(w * gain), round(h * gain)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return out, Letterbox(gain, left, top, size)


def to_tensor(letterboxed_bgr: np.ndarray) -> torch.Tensor:
    """HWC BGR uint8 -> 1x3xHxW RGB float32 in [0, 1] (ultralytics convention)."""
    rgb = np.ascontiguousarray(letterboxed_bgr[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).float().div_(255.0).unsqueeze(0)


class RawDecoder:
    """Host-side decode of raw head outputs -> detections [x1, y1, x2, y2, conf, cls]."""

    def __init__(self, head: nn.Module, imgsz: int):
        from ultralytics.utils.tal import make_anchors

        self.head = head
        self.nc = int(head.nc)
        self.reg_max = int(head.reg_max)
        self.end2end = bool(head.end2end)
        self.xywh = not self.end2end and not getattr(head, "xyxy", False)
        feats = [torch.zeros(1, 1, imgsz // int(s), imgsz // int(s)) for s in head.stride]
        anchors, strides = make_anchors(feats, head.stride, 0.5)
        self.anchors = anchors.transpose(0, 1).unsqueeze(0)  # (1, 2, N)
        self.strides = strides.transpose(0, 1)  # (1, N)

    @torch.no_grad()
    def decode(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Raw tensors -> (B, 4 + nc, N) like ultralytics ``Detect._inference``."""
        from ultralytics.utils.tal import dist2bbox

        boxes, scores = boxes.float().cpu(), scores.float().cpu()
        dist = self.head.dfl(boxes) if self.reg_max > 1 else boxes
        dbox = dist2bbox(dist, self.anchors, xywh=self.xywh, dim=1) * self.strides
        return torch.cat((dbox, scores.sigmoid()), 1)

    @torch.no_grad()
    def decode_prefiltered(self, boxes: torch.Tensor, scores: torch.Tensor, conf: float) -> torch.Tensor:
        """Decode only anchors whose best class score exceeds ``conf`` (batch 1).

        NMS discards every anchor below ``conf`` before anything else, so the detections are the same
        as with ``decode``; only the DFL/box work for the rejected anchors is skipped (V2 runtime).
        """
        from ultralytics.utils.tal import dist2bbox

        boxes, scores = boxes.float().cpu(), scores.float().cpu()
        logit = float(np.log(conf / (1.0 - conf)))
        keep = (scores[0].amax(0) > logit).nonzero().flatten()
        b, s = boxes[:, :, keep], scores[:, :, keep]
        dist = self.head.dfl(b) if self.reg_max > 1 else b
        dbox = dist2bbox(dist, self.anchors[:, :, keep], xywh=self.xywh, dim=1) * self.strides[:, keep]
        return torch.cat((dbox, s.sigmoid()), 1)

    @torch.no_grad()
    def __call__(
        self, boxes: torch.Tensor, scores: torch.Tensor, conf: float, iou: float, max_det: int, prefilter: bool = False
    ) -> list[torch.Tensor]:
        y = self.decode_prefiltered(boxes, scores, conf) if prefilter else self.decode(boxes, scores)
        if self.end2end:  # NMS-free one-to-one head: top-k selection, then threshold
            out = self.head.postprocess(y.permute(0, 2, 1))
            return [d[d[:, 4] > conf] for d in out]
        from ultralytics.utils.nms import non_max_suppression

        return non_max_suppression(y, conf_thres=conf, iou_thres=iou, max_det=max_det, nc=self.nc)


def scale_boxes(det: torch.Tensor, lb: Letterbox, orig_w: int, orig_h: int) -> np.ndarray:
    """Map letterboxed detections back to original image pixels, clipped to the image."""
    d = det.detach().cpu().numpy().astype(np.float64).copy()
    if d.size == 0:
        return d.reshape(0, 6)
    d[:, [0, 2]] = ((d[:, [0, 2]] - lb.pad_x) / lb.gain).clip(0, orig_w)
    d[:, [1, 3]] = ((d[:, [1, 3]] - lb.pad_y) / lb.gain).clip(0, orig_h)
    return d
