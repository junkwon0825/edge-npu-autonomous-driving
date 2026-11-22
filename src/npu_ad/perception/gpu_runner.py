"""CUDA execution of the raw YOLO module with stage-separated, synchronized timing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from npu_ad.perception.yolo_raw import RawDecoder, letterbox, load_raw_model, scale_boxes, to_tensor


@dataclass
class StageTimes:
    preprocess: float
    h2d: float
    model: float  # CUDA-event measured kernel time of the raw module
    postprocess: float  # D2H + decode + NMS + rescale on host
    e2e: float  # wall clock from BGR frame to detections in image pixels


class GpuDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str = "cuda:0",
        half: bool = False,
    ):
        self.raw, head = load_raw_model(weights)
        self.device = torch.device(device)
        self.half = half
        self.raw = self.raw.to(self.device)
        if half:
            self.raw = self.raw.half()
        self.decoder = RawDecoder(head, imgsz)
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self._ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, StageTimes, tuple[torch.Tensor, torch.Tensor]]:
        t0 = time.perf_counter()
        lb_img, lb = letterbox(bgr, self.imgsz)
        x = to_tensor(lb_img)
        t1 = time.perf_counter()
        x = x.to(self.device, non_blocking=False)
        if self.half:
            x = x.half()
        torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        start, end = self._ev
        start.record()
        boxes, scores = self.raw(x)
        end.record()
        torch.cuda.synchronize(self.device)
        t3 = time.perf_counter()
        boxes_c, scores_c = boxes.float().cpu(), scores.float().cpu()
        det = self.decoder(boxes_c, scores_c, self.conf, self.iou, self.max_det)[0]
        out = scale_boxes(det, lb, bgr.shape[1], bgr.shape[0])
        t4 = time.perf_counter()
        times = StageTimes(t1 - t0, t2 - t1, start.elapsed_time(end) / 1000.0, t4 - t3, t4 - t0)
        return out, times, (boxes_c, scores_c)
