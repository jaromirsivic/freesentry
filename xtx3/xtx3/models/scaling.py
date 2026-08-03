"""Depth/width scaling configuration for XTX3-u / -n / -m / -l.

Variant roles:

* ``u`` -- ultra-light Raspberry Pi 5 real-time model. Whole-frame single-pass
  inference at 256/320/384 input; PConv/depthwise blocks; the tightest head
  caps. Budget: <= 1.5M params, <= 0.75 GFLOPs at 320.
* ``n`` -- primary edge model (Raspberry Pi via NCNN), same backbone scale as
  XTX2-n but with the nine-point head (54 vs 102 keypoint channels), so it is
  strictly cheaper per pass than XTX2-n at equal backbone capacity.
* ``m`` -- balanced high-accuracy NVIDIA GPU model (RepConv blocks).
* ``l`` -- highest-accuracy NVIDIA GPU model; adds the P2 (stride 4) level for
  small people.

Width multipliers are tuned so realised param/FLOP counts land inside the
budgets declared in ``scripts/profile_flops.py`` (the budgets are the binding
constraint; multipliers are design anchors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """Static scaling parameters for a single XTX3 variant."""

    name: str
    depth_mult: float
    width_mult: float
    max_channels: int
    use_p2_default: bool
    depthwise: bool  # u/n use PConv + depthwise-separable convs (Pi budget).
    head_box_cap: int  # hidden-width cap of the box/cls head branches
    head_kpt_cap: int  # hidden-width cap of the keypoint head branch


VARIANTS: Final[dict[str, VariantConfig]] = {
    "u": VariantConfig(
        name="u", depth_mult=0.34, width_mult=0.375, max_channels=384,
        use_p2_default=False, depthwise=True, head_box_cap=48, head_kpt_cap=64,
    ),
    "n": VariantConfig(
        name="n", depth_mult=0.34, width_mult=0.56, max_channels=512,
        use_p2_default=False, depthwise=True, head_box_cap=64, head_kpt_cap=96,
    ),
    "m": VariantConfig(
        name="m", depth_mult=0.67, width_mult=0.72, max_channels=768,
        use_p2_default=False, depthwise=False, head_box_cap=80, head_kpt_cap=128,
    ),
    "l": VariantConfig(
        name="l", depth_mult=1.00, width_mult=0.78, max_channels=1024,
        use_p2_default=True, depthwise=False, head_box_cap=80, head_kpt_cap=128,
    ),
}

# Base channel widths at width_mult=1.0 for stem + stages 1..4 (strides 2,4,8,16,32).
BASE_CHANNELS: Final[tuple[int, int, int, int, int]] = (64, 128, 256, 512, 768)
# Base number of bottleneck units per CSP stage at depth_mult=1.0.
BASE_DEPTHS: Final[tuple[int, int, int, int]] = (2, 4, 4, 2)


def make_divisible(value: float, *, divisor: int = 8) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (min == divisor)."""

    return max(divisor, int(round(value / divisor) * divisor))


def get_variant(variant: str) -> VariantConfig:
    """Return the :class:`VariantConfig` for ``"u" | "n" | "m" | "l"`` (raises otherwise)."""

    key = variant.strip().lower()
    if key not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return VARIANTS[key]


def scaled_channels(cfg: VariantConfig) -> tuple[int, ...]:
    """Return per-stage output channels (stem + 4 stages) for a variant."""

    return tuple(
        make_divisible(min(base, cfg.max_channels) * cfg.width_mult)
        for base in BASE_CHANNELS
    )


def scaled_depths(cfg: VariantConfig) -> tuple[int, ...]:
    """Return per-stage bottleneck counts for a variant (min 1)."""

    return tuple(max(1, int(round(base * cfg.depth_mult))) for base in BASE_DEPTHS)
