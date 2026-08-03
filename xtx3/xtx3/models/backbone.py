"""XTX3 backbone: 5 stages, strides 2/4/8/16/32.

Exposes feature maps P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32).
For a 384 input these are 96/48/24/12 spatial sizes (80/40/20/10 at 320,
64/32/16/8 at 256). ``u``/``n`` use a cheap PConv/depthwise path in the early
high-resolution stages (Raspberry Pi FLOPs budget); ``m``/``l`` use full CSP
blocks. The deepest stage ends with SPPF for a large receptive field.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, CSPBlock, DWConv, PConv, SPPF
from .scaling import VariantConfig, scaled_channels, scaled_depths


def _downsample(*, in_channels: int, out_channels: int, depthwise: bool, early: bool = False) -> nn.Module:
    if depthwise and early:
        # PConv downsampling for the u/n variants' high-resolution early stages.
        return PConv(in_channels=in_channels, out_channels=out_channels, stride=2)
    if depthwise:
        return DWConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
    return ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)


class Backbone(nn.Module):
    """CSP-style backbone producing {P2, P3, P4, P5}."""

    def __init__(self, *, cfg: VariantConfig) -> None:
        super().__init__()
        channels = scaled_channels(cfg)  # (stem, s1, s2, s3, s4)
        depths = scaled_depths(cfg)  # (d1, d2, d3, d4)
        dw = cfg.depthwise

        self.out_channels: dict[str, int] = {
            "P2": channels[1],
            "P3": channels[2],
            "P4": channels[3],
            "P5": channels[4],
        }

        # Stem: stride 2. The u/n variants use a plain 3x3 stem conv followed by
        # PConv downsamples -- keeping the very first conv full is cheap (3 input
        # channels) and preserves low-level detail.
        self.stem = ConvBNAct(in_channels=3, out_channels=channels[0], kernel_size=3, stride=2)

        # Stage 1 -> P2 (stride 4). Early stage: PConv path for u/n.
        self.down1 = _downsample(in_channels=channels[0], out_channels=channels[1], depthwise=dw, early=True)
        self.stage1 = CSPBlock(
            in_channels=channels[1], out_channels=channels[1], num_blocks=depths[0], depthwise=dw
        )

        # Stage 2 -> P3 (stride 8). Early stage: PConv path for u/n.
        self.down2 = _downsample(in_channels=channels[1], out_channels=channels[2], depthwise=dw, early=True)
        self.stage2 = CSPBlock(
            in_channels=channels[2], out_channels=channels[2], num_blocks=depths[1], depthwise=dw
        )

        # Stage 3 -> P4 (stride 16)
        self.down3 = _downsample(in_channels=channels[2], out_channels=channels[3], depthwise=dw)
        self.stage3 = CSPBlock(
            in_channels=channels[3], out_channels=channels[3], num_blocks=depths[2], depthwise=dw
        )

        # Stage 4 -> P5 (stride 32) + SPPF
        self.down4 = _downsample(in_channels=channels[3], out_channels=channels[4], depthwise=dw)
        self.stage4 = CSPBlock(
            in_channels=channels[4], out_channels=channels[4], num_blocks=depths[3], depthwise=dw
        )
        self.sppf = SPPF(in_channels=channels[4], out_channels=channels[4])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage1(self.down1(x))
        p3 = self.stage2(self.down2(p2))
        p4 = self.stage3(self.down3(p3))
        p5 = self.sppf(self.stage4(self.down4(p4)))
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}

    def fuse(self) -> "Backbone":
        for module in self.children():
            if hasattr(module, "fuse"):
                module.fuse()
        return self
