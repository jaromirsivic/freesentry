"""XTX4 backbone: stem + 4 stages producing P2 (stride 4) ... P5 (stride 32).

``block="rep"`` (CONTRACTS section 4.3): stem ``ConvAct 3x3/2``, each stage a
``RepConv3x3`` stride-2 downsample followed by ``CSPRep``; stage 4 ends with
``SPPF(k=5)``. Dense 3x3 convolutions throughout: on the Pi 5 / ncnn a dense
3x3 (Winograd) is cheaper per MAC than the depthwise/partial convolutions of
XTX3-u, and RepConv training branches are free at deploy time.

``block="dw"`` reproduces the XTX3-u topology (PConv downsample in stages 1-2,
DWConv downsample in stages 3-4, depthwise CSP bottlenecks, SPPF) as the
recipe-control architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvAct, CSPBlock, CSPRep, DWConv, PConv, RepConv3x3, SPPF, fuse_children
from .scaling import VariantSpec


def _downsample(spec: VariantSpec, *, in_channels: int, out_channels: int, early: bool) -> nn.Module:
    if spec.block == "rep":
        return RepConv3x3(in_channels=in_channels, out_channels=out_channels, stride=2, act=spec.act)
    if early:
        return PConv(in_channels=in_channels, out_channels=out_channels, stride=2, act=spec.act)
    return DWConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, act=spec.act)


def _stage(spec: VariantSpec, *, channels: int, depth: int) -> nn.Module:
    if spec.block == "rep":
        return CSPRep(in_channels=channels, out_channels=channels, num_blocks=depth, act=spec.act)
    return CSPBlock(
        in_channels=channels, out_channels=channels, num_blocks=depth, depthwise=True, act=spec.act
    )


class Backbone(nn.Module):
    """Produces ``{"P2", "P3", "P4", "P5"}`` feature maps."""

    def __init__(self, *, spec: VariantSpec) -> None:
        super().__init__()
        w0, w1, w2, w3, w4 = spec.widths
        d1, d2, d3, d4 = spec.depths
        self.out_channels: dict[str, int] = {"P2": w1, "P3": w2, "P4": w3, "P5": w4}

        self.stem = ConvAct(in_channels=3, out_channels=w0, kernel_size=3, stride=2, act=spec.act)
        self.down1 = _downsample(spec, in_channels=w0, out_channels=w1, early=True)
        self.stage1 = _stage(spec, channels=w1, depth=d1)
        self.down2 = _downsample(spec, in_channels=w1, out_channels=w2, early=True)
        self.stage2 = _stage(spec, channels=w2, depth=d2)
        self.down3 = _downsample(spec, in_channels=w2, out_channels=w3, early=False)
        self.stage3 = _stage(spec, channels=w3, depth=d3)
        self.down4 = _downsample(spec, in_channels=w3, out_channels=w4, early=False)
        self.stage4 = _stage(spec, channels=w4, depth=d4)
        self.sppf = SPPF(in_channels=w4, out_channels=w4, kernel_size=5, act=spec.act)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage1(self.down1(x))
        p3 = self.stage2(self.down2(p2))
        p4 = self.stage3(self.down3(p3))
        p5 = self.sppf(self.stage4(self.down4(p4)))
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}

    def fuse(self) -> "Backbone":
        fuse_children(self)
        return self
