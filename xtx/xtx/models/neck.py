"""PAN/FPN neck (spec section 6.3): top-down then bottom-up fusion.

Fuses {P3, P4, P5} (and optionally P2 for small-person emphasis) and emits one
feature map per pyramid level to the detection head. CSPBlock fusion is applied at
every merge point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, CSPBlock, DWConv
from .scaling import VariantConfig


def _downsample(*, channels: int, depthwise: bool) -> nn.Module:
    if depthwise:
        return DWConv(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
    return ConvBNAct(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)


class PANNeck(nn.Module):
    """Path-Aggregation neck over backbone features."""

    def __init__(
        self,
        *,
        cfg: VariantConfig,
        backbone_channels: dict[str, int],
        use_p2: bool,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.use_p2 = use_p2
        dw = cfg.depthwise
        c1 = backbone_channels["P2"]
        c2 = backbone_channels["P3"]
        c3 = backbone_channels["P4"]
        c4 = backbone_channels["P5"]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- top-down ---
        self.td_p4 = CSPBlock(
            in_channels=c4 + c3, out_channels=c3, num_blocks=num_blocks, shortcut=False, depthwise=dw
        )
        self.td_p3 = CSPBlock(
            in_channels=c3 + c2, out_channels=c2, num_blocks=num_blocks, shortcut=False, depthwise=dw
        )
        if use_p2:
            self.td_p2 = CSPBlock(
                in_channels=c2 + c1, out_channels=c1, num_blocks=num_blocks, shortcut=False, depthwise=dw
            )

        # --- bottom-up ---
        if use_p2:
            self.down_p2 = _downsample(channels=c1, depthwise=dw)
            self.bu_p3 = CSPBlock(
                in_channels=c1 + c2, out_channels=c2, num_blocks=num_blocks, shortcut=False, depthwise=dw
            )
        self.down_p3 = _downsample(channels=c2, depthwise=dw)
        self.bu_p4 = CSPBlock(
            in_channels=c2 + c3, out_channels=c3, num_blocks=num_blocks, shortcut=False, depthwise=dw
        )
        self.down_p4 = _downsample(channels=c3, depthwise=dw)
        self.bu_p5 = CSPBlock(
            in_channels=c3 + c4, out_channels=c4, num_blocks=num_blocks, shortcut=False, depthwise=dw
        )

        if use_p2:
            self.out_channels = [c1, c2, c3, c4]
            self.out_strides = [4, 8, 16, 32]
        else:
            self.out_channels = [c2, c3, c4]
            self.out_strides = [8, 16, 32]

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        p2, p3, p4, p5 = features["P2"], features["P3"], features["P4"], features["P5"]

        t4 = self.td_p4(torch.cat([self.upsample(p5), p4], dim=1))
        t3 = self.td_p3(torch.cat([self.upsample(t4), p3], dim=1))

        if self.use_p2:
            n2 = self.td_p2(torch.cat([self.upsample(t3), p2], dim=1))
            n3 = self.bu_p3(torch.cat([self.down_p2(n2), t3], dim=1))
            n4 = self.bu_p4(torch.cat([self.down_p3(n3), t4], dim=1))
            n5 = self.bu_p5(torch.cat([self.down_p4(n4), p5], dim=1))
            return [n2, n3, n4, n5]

        n3 = t3
        n4 = self.bu_p4(torch.cat([self.down_p3(n3), t4], dim=1))
        n5 = self.bu_p5(torch.cat([self.down_p4(n4), p5], dim=1))
        return [n3, n4, n5]

    def fuse(self) -> "PANNeck":
        for module in self.children():
            if hasattr(module, "fuse"):
                module.fuse()
        return self
