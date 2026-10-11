"""PAN neck (top-down then bottom-up) with three level configurations.

* ``P3P4P5`` (default): ``td_p4`` on ``cat(up2(P5), P4)``, ``td_p3`` on
  ``cat(up2(td_p4), P3)`` = N3, ``bu_p4`` on ``cat(down(N3), td_p4)`` = N4,
  ``bu_p5`` on ``cat(down(N4), P5)`` = N5.
* ``P3P5``: keeps the full top-down path to N3; the bottom-up path goes
  ``N3 -> down -> down -> cat with P5 -> bu_p5`` = N5 (no P4 head, no bu_p4).
* ``P4P5``: no ``td_p3``; N4 = ``td_p4``, N5 = ``bu_p5(cat(down(N4), P5))``.

Optional P2-assist (``spec.p2_assist``): ``SpaceToDepth(P2)`` (``4*w1`` channels
at P3 resolution) -> zero-initialised 1x1 -> ``w1 // 2`` channels. It is NOT part
of the level features; the head concatenates it to the pose-branch input at P3
only, so at initialisation the model is numerically identical to the model
without the assist.

``block="dw"`` uses the XTX3 ``CSPBlock(depthwise, shortcut=False)`` and DWConv
downsamples; ``block="rep"`` uses ``CSPRep`` and ``RepConv3x3`` stride 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CSPBlock, CSPRep, DWConv, RepConv3x3, SpaceToDepth, fuse_children
from .scaling import LEVEL_SETS, VariantSpec


def _merge(spec: VariantSpec, *, in_channels: int, out_channels: int) -> nn.Module:
    if spec.block == "rep":
        return CSPRep(in_channels=in_channels, out_channels=out_channels, num_blocks=1, act=spec.act)
    return CSPBlock(
        in_channels=in_channels, out_channels=out_channels, num_blocks=1,
        shortcut=False, depthwise=True, act=spec.act,
    )


def _down(spec: VariantSpec, *, channels: int) -> nn.Module:
    if spec.block == "rep":
        return RepConv3x3(in_channels=channels, out_channels=channels, stride=2, act=spec.act)
    return DWConv(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, act=spec.act)


class PANNeck(nn.Module):
    """Path-aggregation neck; emits one feature map per active level (ascending stride)."""

    def __init__(self, *, spec: VariantSpec, backbone_channels: dict[str, int]) -> None:
        super().__init__()
        self.levels = spec.levels
        self.out_strides: list[int] = list(LEVEL_SETS[spec.levels])
        c2 = backbone_channels["P3"]
        c3 = backbone_channels["P4"]
        c4 = backbone_channels["P5"]

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_p4 = _merge(spec, in_channels=c4 + c3, out_channels=c3)

        if spec.levels == "P3P4P5":
            self.td_p3 = _merge(spec, in_channels=c3 + c2, out_channels=c2)
            self.down_p3 = _down(spec, channels=c2)
            self.bu_p4 = _merge(spec, in_channels=c2 + c3, out_channels=c3)
            self.down_p4 = _down(spec, channels=c3)
            self.bu_p5 = _merge(spec, in_channels=c3 + c4, out_channels=c4)
            self.out_channels = [c2, c3, c4]
        elif spec.levels == "P3P5":
            self.td_p3 = _merge(spec, in_channels=c3 + c2, out_channels=c2)
            self.down_p3 = _down(spec, channels=c2)
            self.down_p4 = _down(spec, channels=c2)
            self.bu_p5 = _merge(spec, in_channels=c2 + c4, out_channels=c4)
            self.out_channels = [c2, c4]
        elif spec.levels == "P4P5":
            self.down_p4 = _down(spec, channels=c3)
            self.bu_p5 = _merge(spec, in_channels=c3 + c4, out_channels=c4)
            self.out_channels = [c3, c4]
        else:  # pragma: no cover - validated by VariantSpec
            raise ValueError(f"Unsupported levels {spec.levels!r}")

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = features["P3"], features["P4"], features["P5"]
        t4 = self.td_p4(torch.cat([self.upsample(p5), p4], dim=1))
        if self.levels == "P3P4P5":
            n3 = self.td_p3(torch.cat([self.upsample(t4), p3], dim=1))
            n4 = self.bu_p4(torch.cat([self.down_p3(n3), t4], dim=1))
            n5 = self.bu_p5(torch.cat([self.down_p4(n4), p5], dim=1))
            return [n3, n4, n5]
        if self.levels == "P3P5":
            n3 = self.td_p3(torch.cat([self.upsample(t4), p3], dim=1))
            n5 = self.bu_p5(torch.cat([self.down_p4(self.down_p3(n3)), p5], dim=1))
            return [n3, n5]
        n5 = self.bu_p5(torch.cat([self.down_p4(t4), p5], dim=1))
        return [t4, n5]

    def fuse(self) -> "PANNeck":
        fuse_children(self)
        return self


class P2Assist(nn.Module):
    """``SpaceToDepth(P2) -> 1x1 (zero-init, bias) -> w1 // 2`` at P3 resolution."""

    def __init__(self, *, p2_channels: int) -> None:
        super().__init__()
        self.out_channels = max(8, p2_channels // 2)
        self.s2d = SpaceToDepth(block_size=2)
        self.proj = nn.Conv2d(4 * p2_channels, self.out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        return self.proj(self.s2d(p2))

    def fuse(self) -> "P2Assist":
        return self
