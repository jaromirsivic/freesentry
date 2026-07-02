"""Depth/width scaling configuration for XTX2-n / XTX2-m / XTX2-l (spec 6.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """Static scaling parameters for a single XTX2 variant."""

    name: str
    depth_mult: float
    width_mult: float
    max_channels: int
    use_p2_default: bool
    depthwise: bool  # XTX2-n uses PConv + depthwise-separable convs (Pi budget).


# Variant configs. Width multipliers are tuned (rather than the literal table
# values 0.25/0.75/1.00) so the realised param/FLOP counts land inside the
# section 6.6 budgets with this architecture: the documented multipliers are
# design anchors, the budgets are the binding constraint. Verified by
# scripts/profile_flops.py.
VARIANTS: Final[dict[str, VariantConfig]] = {
    "n": VariantConfig(
        name="n", depth_mult=0.34, width_mult=0.56, max_channels=512,
        use_p2_default=False, depthwise=True,
    ),
    "m": VariantConfig(
        name="m", depth_mult=0.67, width_mult=0.72, max_channels=768,
        use_p2_default=False, depthwise=False,
    ),
    "l": VariantConfig(
        name="l", depth_mult=1.00, width_mult=0.78, max_channels=1024,
        use_p2_default=True, depthwise=False,
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
