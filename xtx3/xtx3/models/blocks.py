"""Re-usable network building blocks (adopted from XTX2's validated design).

All blocks support a ``fuse()`` step that folds BatchNorm into the preceding
convolution and re-parameterises :class:`RepConv` into a single 3x3 convolution,
yielding a clean, edge-friendly graph for ONNX / NCNN export.

:class:`PConv` (partial convolution, FasterNet-style) convolves only a fraction
of channels and passes the rest through; it is used in the ``u``/``n`` stems and
early stages as the main Raspberry Pi FLOPs lever.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(kernel_size: int, padding: int | None = None) -> int:
    """Return 'same' padding for an odd kernel when ``padding`` is not given."""

    if padding is not None:
        return padding
    return kernel_size // 2


class ConvBNAct(nn.Module):
    """Conv2d (no bias) + BatchNorm2d + activation (SiLU by default)."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
        act: bool = True,
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
        self.act: nn.Module = nn.SiLU(inplace=True) if act else nn.Identity()
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


class DWConv(nn.Module):
    """Depthwise-separable conv: depthwise KxK + pointwise 1x1 (u/n variants)."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.dw = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pw = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

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
    pointwise conv. Dramatically cheaper than a full 3x3 at similar accuracy for
    early, high-resolution stages of the ``u``/``n`` variants.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        partial_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.conv_channels = max(8, in_channels // partial_ratio)
        # Round to something sane when in_channels is tiny.
        self.conv_channels = min(self.conv_channels, in_channels)
        self.pass_channels = in_channels - self.conv_channels
        self.stride = stride

        self.partial = ConvBNAct(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=3,
            stride=stride,
        )
        # Untouched channels still need downsampling when stride > 1; a cheap
        # max-pool preserves them without any multiply-accumulates.
        self.pool: nn.Module = (
            nn.MaxPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()
        )
        self.pw = ConvBNAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

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
    """Re-parameterisable block: 3x3 + 1x1 (+ identity) branches at train time.

    At deploy time :meth:`fuse` / :meth:`reparameterize` collapses all branches
    into a single 3x3 conv with bias (RepVGG re-parameterisation).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.act: nn.Module = nn.SiLU(inplace=True) if act else nn.Identity()

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
        identity_out = 0.0 if self.identity is None else self.identity(x)
        return self.act(self.conv3(x) + self.conv1(x) + identity_out)

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


class Bottleneck(nn.Module):
    """Residual bottleneck of two convs; RepConv (or DWConv) for the 3x3."""

    def __init__(
        self,
        *,
        channels: int,
        shortcut: bool = True,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(in_channels=channels, out_channels=channels, kernel_size=1)
        if depthwise:
            self.cv2: nn.Module = DWConv(in_channels=channels, out_channels=channels, kernel_size=3)
        else:
            self.cv2 = RepConv(in_channels=channels, out_channels=channels)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

    def fuse(self) -> "Bottleneck":
        self.cv1.fuse()
        self.cv2.fuse()  # type: ignore[union-attr]
        return self


class CSPBlock(nn.Module):
    """C2f-style cross-stage block: split -> n bottlenecks -> concat -> 1x1 fuse."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        depthwise: bool = False,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = max(8, int(out_channels * expansion))
        self.hidden = hidden
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=2 * hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            Bottleneck(channels=hidden, shortcut=shortcut, depthwise=depthwise)
            for _ in range(num_blocks)
        )
        self.cv2 = ConvBNAct(
            in_channels=(2 + num_blocks) * hidden, out_channels=out_channels, kernel_size=1
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


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (three sequential 5x5 max-pools)."""

    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=hidden, kernel_size=1)
        self.cv2 = ConvBNAct(in_channels=hidden * 4, out_channels=out_channels, kernel_size=1)
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
