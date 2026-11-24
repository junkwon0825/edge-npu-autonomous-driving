"""Lane-marking segmentation network designed for the Furiosa RNGD constraints found in V2/V3.

U-Net (4 levels, base width 32 as in the author's earlier lane U-Net, input 1x3x256x512 RGB in [0, 1],
output 1 logit channel = lane-marking probability). Deviations from a textbook U-Net, each motivated by
a recorded compiler result (``docs/v3_lane_npu_porting.md``):

* downsampling by 3x3 stride-2 Conv-BN-ReLU instead of max-pool (max-pool has no lowering tactic at
  256x512; 2x2 stride-2 conv is not supported),
* upsampling by ConvTranspose2d(k2, s2) for training, exported as 1x1 conv + PixelShuffle,
* BatchNorm only directly after a conv inside ``nn.Sequential`` so it can be folded for export.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from npu_ad.npu.rewrites import fold_bn, replace_convtranspose

LANE_IN_W, LANE_IN_H = 512, 256


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))  # fmt: skip


class LaneUNet(nn.Module):
    def __init__(self, base: int = 32, levels: int = 4):
        super().__init__()
        ch = [base * 2**i for i in range(levels)]
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        prev = 3
        for i, c in enumerate(ch):
            self.enc.append(_block(prev, c))
            if i < levels - 1:
                self.down.append(nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.BatchNorm2d(c),
                                               nn.ReLU(inplace=True)))  # fmt: skip
            prev = c
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(_block(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.down[i](x)
        for u, d in zip(self.up, self.dec, strict=True):
            x = d(torch.cat([u(x), skips.pop()], 1))
        return self.head(x)


def export_for_npu(model: LaneUNet) -> nn.Module:
    """Eval copy with BatchNorm folded and ConvTranspose rewritten (numerically equivalent on CPU)."""
    import copy

    m = copy.deepcopy(model).eval().float()
    fold_bn(m)
    replace_convtranspose(m)
    return m


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR camera frame (any size) -> 1x3x256x512 RGB float in [0, 1] (plain resize, no letterbox)."""
    img = cv2.resize(bgr, (LANE_IN_W, LANE_IN_H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float().div_(255).unsqueeze(0)


def load_lane_model(path: str, device: str = "cpu") -> LaneUNet:
    ck = torch.load(path, map_location="cpu")
    m = LaneUNet(**ck.get("arch", {}))
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(device)
