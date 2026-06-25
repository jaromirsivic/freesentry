"""Depth/width scaling configuration for XTX-n / XTX-m / XTX-l (spec section 6.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """Static scaling parameters for a single XTX variant."""

    name: str
    depth_mult: float
    width_mult: float
    max_channels: int
    use_p2_default: bool
    depthwise: bool  # XTX-n uses depthwise-separable convs to cut FLOPs for the Pi.


# Variant configs. The width multipliers are tuned (rather than the literal table
# values 0.25/0.75/1.00) so the realised param/FLOP counts land in the section 6.6
# budgets: the documented 16x width spread cannot hit the ~9x param-target spread
# with a uniform architecture, and the spec instructs tuning to the compute budget.
VARIANTS: Final[dict[str, VariantConfig]] = {
    "n": VariantConfig(
        name="n", depth_mult=0.34, width_mult=0.56, max_channels=512,
        use_p2_default=False, depthwise=True,
    ),
    "m": VariantConfig(
        name="m", depth_mult=0.67, width_mult=0.75, max_channels=768,
        use_p2_default=False, depthwise=False,
    ),
    "l": VariantConfig(
        name="l", depth_mult=0.85, width_mult=0.80, max_channels=768,
        use_p2_default=True, depthwise=False,
    ),
}

# Base channel widths at width_mult=1.0 for stem + stages 1..4 (strides 2,4,8,16,32).
BASE_CHANNELS: Final[tuple[int, int, int, int, int]] = (64, 128, 256, 512, 768)
# Base number of bottleneck units per CSP stage at depth_mult=1.0.
BASE_DEPTHS: Final[tuple[int, int, int, int]] = (2, 4, 4, 2)


def make_divisible(value: float, *, divisor: int = 8) -> int:
    """Round ``value`` up to the nearest multiple of ``divisor`` (min == divisor)."""

    return max(divisor, int(round(value / divisor) * divisor))


def get_variant(variant: str) -> VariantConfig:
    """Return the :class:`VariantConfig` for ``"n" | "m" | "l"`` (raises otherwise)."""

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
