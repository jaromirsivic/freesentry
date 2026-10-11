"""Re-usable network building blocks.

Two families live here:

* The XTX3 blocks (``ConvBNAct``, ``DWConv``, ``PConv``, ``RepConv``, ``Bottleneck``,
  ``CSPBlock``, ``SPPF``) used by the ``block="dw"`` recipe-control topology.
* The XTX4 full-convolution family (``RepConv3x3``, ``CSPRep``, ``SpaceToDepth``)
  used by ``block="rep"``: dense 3x3 convolutions trained as RepVGG-style
  multi-branch blocks and deployed as a single 3x3 conv (Winograd-friendly on
  the Raspberry Pi 5 / ncnn).

Every block exposes ``fuse()`` which folds BatchNorm into the preceding
convolution and re-parameterises RepConv branches, in place and idempotently,
so the deployment graph only contains Conv2d(+bias), activation, Concat,
Upsample(nearest), MaxPool2d and the reshape/permute of ``SpaceToDepth``.

Activations are chosen by name (``"relu"`` default for Pi deployment because
ncnn fuses it into the convolution, ``"hswish"``/``"silu"`` as ablations); the
``act=True/False`` form of the XTX3 blocks is still accepted (``True`` = SiLU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_NAMES: tuple[str, ...] = ("relu", "hswish", "silu")


def make_act(act: str | bool | None) -> nn.Module:
    """Activation factory: ``"relu" | "hswish" | "silu" | "none"`` (bool: True = SiLU)."""

    if act is True:
        act = "silu"
    if act is False or act is None:
        return nn.Identity()
    key = str(act).lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key in ("hswish", "hardswish"):
        return nn.Hardswish(inplace=True)
    if key == "silu":
        return nn.SiLU(inplace=True)
    if key in ("none", "identity", "linear"):
        return nn.Identity()
    raise ValueError(f"Unknown activation {act!r}; expected one of {ACTIVATION_NAMES}")


def autopad(kernel_size: int, padding: int | None = None) -> int:
    """Return 'same' padding for an odd kernel when ``padding`` is not given."""

    if padding is not None:
        return padding
    return kernel_size // 2


class ConvBNAct(nn.Module):
    """Conv2d (no bias) + BatchNorm2d + activation."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act: nn.Module = make_act(act)
        self._fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self) -> "ConvBNAct":
        """Fold BatchNorm into ``conv`` in place (idempotent)."""

        if self._fused:
            return self
        self.conv = _fuse_conv_bn(self.conv, self.bn)
        self.bn = nn.Identity()  # type: ignore[assignment]
        self._fused = True
        return self


# Alias used by the "rep" topology for the stem (CONTRACTS section 4.3).
ConvAct = ConvBNAct


class DWConv(nn.Module):
    """Depthwise-separable conv: depthwise KxK + pointwise 1x1 (dw topology)."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        self.dw = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pw = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

    def fuse(self) -> "DWConv":
        self.dw.fuse()
        self.pw.fuse()
        return self


class PConv(nn.Module):
    """Partial convolution (FasterNet-style).

    Convolves only the first ``in_channels // partial_ratio`` channels with a 3x3
    conv and passes the remaining channels through untouched, then mixes with a
    pointwise conv. Used by the ``dw`` (XTX3-u control) topology in stages 1-2.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        partial_ratio: int = 4,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        self.conv_channels = max(8, in_channels // partial_ratio)
        self.conv_channels = min(self.conv_channels, in_channels)
        self.pass_channels = in_channels - self.conv_channels
        self.stride = stride

        self.partial = ConvBNAct(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=3,
            stride=stride,
            act=act,
        )
        # Untouched channels still need downsampling when stride > 1; a cheap
        # max-pool preserves them without any multiply-accumulates.
        self.pool: nn.Module = (
            nn.MaxPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
        )
        self.pw = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head, tail = x.split((self.conv_channels, self.pass_channels), dim=1)
        head = self.partial(head)
        if self.pass_channels > 0:
            tail = self.pool(tail)
            out = torch.cat([head, tail], dim=1)
        else:
            out = head
        return self.pw(out)

    def fuse(self) -> "PConv":
        self.partial.fuse()
        self.pw.fuse()
        return self


class RepConv(nn.Module):
    """Re-parameterisable 3x3 block: 3x3 + 1x1 (+ BN identity) branches at train time.

    Works with any stride (the identity branch exists only for stride 1 and equal
    channel counts) and any activation. :meth:`fuse` collapses all branches into
    a single 3x3 conv with bias (RepVGG re-parameterisation).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.act: nn.Module = make_act(act)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.identity = (
            nn.BatchNorm2d(in_channels)
            if (in_channels == out_channels and stride == 1)
            else None
        )
        self.fused_conv: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_conv is not None:
            return self.act(self.fused_conv(x))
        out = self.conv3(x) + self.conv1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    @torch.no_grad()
    def reparameterize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused ``(kernel_3x3, bias)`` equivalent to all branches."""

        k3, b3 = _fuse_branch_to_kernel_bias(self.conv3, target_kernel=3)
        k1, b1 = _fuse_branch_to_kernel_bias(self.conv1, target_kernel=3)
        if self.identity is not None:
            kid, bid = _identity_branch_kernel_bias(self.identity, self.in_channels, self.groups)
        else:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)
        return k3 + k1 + kid, b3 + b1 + bid

    @torch.no_grad()
    def fuse(self) -> "RepConv":
        """Collapse branches into a single 3x3 conv in place (idempotent)."""

        if self.fused_conv is not None:
            return self
        kernel, bias = self.reparameterize()
        fused = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            self.stride,
            1,
            groups=self.groups,
            bias=True,
        )
        fused.weight.data = kernel
        fused.bias.data = bias  # type: ignore[union-attr]
        self.fused_conv = fused.to(kernel.device)
        # Drop training-only branches to free memory and simplify export.
        self.__delattr__("conv3")
        self.__delattr__("conv1")
        if self.identity is not None:
            self.__delattr__("identity")
        self.identity = None
        return self


class RepConv3x3(RepConv):
    """XTX4 name of :class:`RepConv` (dense 3x3, any stride, any activation)."""


class Bottleneck(nn.Module):
    """XTX3 residual bottleneck: 1x1 conv then RepConv (or DWConv) 3x3."""

    def __init__(
        self,
        *,
        channels: int,
        shortcut: bool = True,
        depthwise: bool = False,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(in_channels=channels, out_channels=channels, kernel_size=1, act=act)
        if depthwise:
            self.cv2: nn.Module = DWConv(in_channels=channels, out_channels=channels, kernel_size=3, act=act)
        else:
            self.cv2 = RepConv(in_channels=channels, out_channels=channels, act=act)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def fuse(self) -> "Bottleneck":
        self.cv1.fuse()
        self.cv2.fuse()  # type: ignore[union-attr]
        return self


class CSPBlock(nn.Module):
    """XTX3 C2f-style cross-stage block: split -> n bottlenecks -> concat -> 1x1 fuse."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        depthwise: bool = False,
        expansion: float = 0.5,
        act: str | bool = True,
    ) -> None:
        super().__init__()
        hidden = max(8, int(out_channels * expansion))
        self.hidden = hidden
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=2 * hidden, kernel_size=1, act=act)
        self.blocks = nn.ModuleList(
            Bottleneck(channels=hidden, shortcut=shortcut, depthwise=depthwise, act=act)
            for _ in range(num_blocks)
        )
        self.cv2 = ConvBNAct(
            in_channels=(2 + num_blocks) * hidden, out_channels=out_channels, kernel_size=1, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.hidden, self.hidden), dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

    def fuse(self) -> "CSPBlock":
        self.cv1.fuse()
        self.cv2.fuse()
        for block in self.blocks:
            block.fuse()
        return self


class RepBottleneck(nn.Module):
    """XTX4 bottleneck: ``RepConv3x3 h->h``, ``RepConv3x3 h->h``, residual."""

    def __init__(self, *, channels: int, shortcut: bool = True, act: str | bool = "relu") -> None:
        super().__init__()
        self.cv1 = RepConv3x3(in_channels=channels, out_channels=channels, act=act)
        self.cv2 = RepConv3x3(in_channels=channels, out_channels=channels, act=act)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def fuse(self) -> "RepBottleneck":
        self.cv1.fuse()
        self.cv2.fuse()
        return self


class CSPRep(nn.Module):
    """C2f-style block with dense RepConv bottlenecks (CONTRACTS section 4.3).

    1x1 ``C_in -> 2h`` (``h = C_out // 2``), split, ``n`` :class:`RepBottleneck`
    on the second half (each appended to the concat list), concat ``(2+n)*h``,
    1x1 ``-> C_out``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        act: str | bool = "relu",
    ) -> None:
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.hidden = hidden
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=2 * hidden, kernel_size=1, act=act)
        self.blocks = nn.ModuleList(
            RepBottleneck(channels=hidden, shortcut=shortcut, act=act) for _ in range(num_blocks)
        )
        self.cv2 = ConvBNAct(
            in_channels=(2 + num_blocks) * hidden, out_channels=out_channels, kernel_size=1, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.hidden, self.hidden), dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

    def fuse(self) -> "CSPRep":
        self.cv1.fuse()
        self.cv2.fuse()
        for block in self.blocks:
            block.fuse()
        return self


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (three sequential 5x5 max-pools)."""

    def __init__(
        self, *, in_channels: int, out_channels: int, kernel_size: int = 5, act: str | bool = True
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=hidden, kernel_size=1, act=act)
        self.cv2 = ConvBNAct(in_channels=hidden * 4, out_channels=out_channels, kernel_size=1, act=act)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

    def fuse(self) -> "SPPF":
        self.cv1.fuse()
        self.cv2.fuse()
        return self


class SpaceToDepth(nn.Module):
    """``(B, C, H, W) -> (B, 4C, H/2, W/2)`` by pixel-unshuffle (reshape/permute only).

    Used by the optional P2-assist: it brings the stride-4 feature map to the P3
    grid without any multiply-accumulates. Channel order is the PyTorch
    ``pixel_unshuffle`` order ``c * 4 + dy * 2 + dx``; the following 1x1 conv
    is learned, so the exact order does not matter. Parameter-free, so ``fuse``
    is a no-op.
    """

    def __init__(self, *, block_size: int = 2) -> None:
        super().__init__()
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(x, self.block_size)

    def fuse(self) -> "SpaceToDepth":
        return self


def fuse_children(module: nn.Module) -> nn.Module:
    """Call ``fuse()`` on every direct child that implements it (in place)."""

    for child in module.children():
        if hasattr(child, "fuse"):
            child.fuse()
    return module


@torch.no_grad()
def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a single Conv2d (with bias) equivalent to ``bn(conv(x))``."""

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,  # type: ignore[arg-type]
        conv.stride,  # type: ignore[arg-type]
        conv.padding,  # type: ignore[arg-type]
        dilation=conv.dilation,  # type: ignore[arg-type]
        groups=conv.groups,
        bias=True,
    )
    w_conv = conv.weight.clone()
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    fused.weight.data = w_conv * scale.reshape(-1, 1, 1, 1)
    conv_bias = (
        conv.bias
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=bn.weight.dtype, device=bn.weight.device)
    )
    fused.bias.data = bn.bias + (conv_bias - bn.running_mean) * scale  # type: ignore[union-attr]
    return fused.to(conv.weight.device)


@torch.no_grad()
def _fuse_branch_to_kernel_bias(
    branch: nn.Sequential, *, target_kernel: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a (conv, bn) branch and pad its kernel to ``target_kernel`` size."""

    conv = branch[0]
    bn = branch[1]
    fused = _fuse_conv_bn(conv, bn)
    kernel = fused.weight
    bias = fused.bias
    if kernel.shape[-1] != target_kernel:
        pad = (target_kernel - kernel.shape[-1]) // 2
        kernel = F.pad(kernel, [pad, pad, pad, pad])
    return kernel, bias  # type: ignore[return-value]


@torch.no_grad()
def _identity_branch_kernel_bias(
    bn: nn.BatchNorm2d, channels: int, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the equivalent 3x3 kernel/bias for a BN-only identity branch."""

    input_dim = channels // groups
    kernel = torch.zeros((channels, input_dim, 3, 3), dtype=bn.weight.dtype, device=bn.weight.device)
    for i in range(channels):
        kernel[i, i % input_dim, 1, 1] = 1.0
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
    fused_bias = bn.bias - bn.running_mean * scale
    return fused_kernel, fused_bias
