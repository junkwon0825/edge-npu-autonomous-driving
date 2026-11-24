"""Exact NPU-friendly graph rewrites for Furiosa RNGD (furiosa-torch 2026.4.0), validated in the porting lab.

* ``fold_bn`` - fold every BatchNorm2d into the Conv2d directly before it inside an ``nn.Sequential``
  (unfused BatchNorm is not importable by the RNGD compiler).
* ``ConvT2x`` - ``ConvTranspose2d(k=2, s=2)`` == 1x1 conv to 4*C_out + ``PixelShuffle(2)``
  ("transposed convolution is not implemented yet").
* ``Bilinear2x`` - bilinear x2 (align_corners=False) == nearest x2 + [1,2,1]/4 separable depthwise conv
  with edge replication ("from_table_lookup expects a 1D table").
* ``dense_from_dilated`` - dilated conv == zero-inserted dense kernel.
Each rewrite matches the original on CPU to float rounding (<= 1e-6, ``docs/v2_operator_compatibility.md``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fold_bn(m: nn.Module) -> nn.Module:
    for mod in list(m.modules()):
        if isinstance(mod, nn.Sequential):
            keys = list(mod._modules)
            for a, b in zip(keys, keys[1:], strict=False):
                ca, cb = mod._modules[a], mod._modules[b]
                if isinstance(ca, nn.Conv2d) and isinstance(cb, nn.BatchNorm2d):
                    mod._modules[a] = fuse_conv_bn_eval(ca, cb)
                    mod._modules[b] = nn.Identity()
    return m


class ConvT2x(nn.Module):
    def __init__(self, ct: nn.ConvTranspose2d):
        super().__init__()
        if ct.kernel_size != (2, 2) or ct.stride != (2, 2):
            raise ValueError("only ConvTranspose2d(k=2, s=2) has this exact rewrite")
        cin, cout = ct.weight.shape[0], ct.weight.shape[1]
        self.conv = nn.Conv2d(cin, 4 * cout, 1, bias=ct.bias is not None)
        self.conv.weight.data[:] = ct.weight.data.permute(1, 2, 3, 0).reshape(cout * 4, cin, 1, 1)
        if ct.bias is not None:
            self.conv.bias.data[:] = ct.bias.data.repeat_interleave(4)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


def edge_pad(x: torch.Tensor) -> torch.Tensor:
    x = torch.cat([x[:, :, :1], x, x[:, :, -1:]], 2)
    return torch.cat([x[:, :, :, :1], x, x[:, :, :, -1:]], 3)


class Bilinear2x(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        k1 = torch.tensor([0.25, 0.5, 0.25])
        self.dw = nn.Conv2d(c, c, 3, groups=c, bias=False)
        self.dw.weight.data[:] = torch.outer(k1, k1)
        self.dw.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(edge_pad(F.interpolate(x, scale_factor=2, mode="nearest")))


def dense_from_dilated(conv: nn.Conv2d) -> nn.Conv2d:
    d, k = conv.dilation[0], conv.kernel_size[0]
    new = nn.Conv2d(conv.in_channels, conv.out_channels, (k - 1) * d + 1, stride=conv.stride,
                    padding=conv.padding[0], groups=conv.groups, bias=conv.bias is not None)  # fmt: skip
    new.weight.data.zero_()
    new.weight.data[:, :, ::d, ::d] = conv.weight.data
    if conv.bias is not None:
        new.bias.data[:] = conv.bias.data
    return new


def replace_convtranspose(m: nn.Module) -> int:
    n = 0
    for name, child in m.named_children():
        if isinstance(child, nn.ConvTranspose2d):
            setattr(m, name, ConvT2x(child))
            n += 1
        else:
            n += replace_convtranspose(child)
    return n


class SplitConv(nn.Module):
    """Conv2d split along output channels into chunks of <= ``chunk`` (exact; outputs concatenated).

    Large 3x3 convolutions (e.g. 256->512) fail on RNGD in the TransposedLower pass with
    "LowLevelTailCrop ... size_in_bytes ... <= u16::MAX" (YOLO26s, ResNet18); splitting keeps each
    weight slice small.
    """

    def __init__(self, conv: nn.Conv2d, chunk: int = 256):
        super().__init__()
        self.parts = nn.ModuleList()
        for lo in range(0, conv.out_channels, chunk):
            hi = min(lo + chunk, conv.out_channels)
            c = nn.Conv2d(conv.in_channels, hi - lo, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                          conv.groups if conv.groups == 1 else 1, bias=conv.bias is not None)  # fmt: skip
            c.weight.data[:] = conv.weight.data[lo:hi]
            if conv.bias is not None:
                c.bias.data[:] = conv.bias.data[lo:hi]
            self.parts.append(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([c(x) for c in self.parts], 1)


def split_large_convs(m: nn.Module, max_out: int = 256, kernel: int = 3) -> int:
    """Replace every dense (groups=1) kxk Conv2d with more than ``max_out`` output channels by SplitConv."""
    n = 0
    for name, child in m.named_children():
        if (isinstance(child, nn.Conv2d) and child.groups == 1 and child.kernel_size == (kernel, kernel)
                and child.out_channels > max_out):  # fmt: skip
            setattr(m, name, SplitConv(child, max_out))
            n += 1
        else:
            n += split_large_convs(child, max_out, kernel)
    return n
