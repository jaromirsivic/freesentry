"""Variant presets and architecture resolution for XTX4 (CONTRACTS.md section 4.2).

Why a frozen dataclass instead of width/depth multipliers (XTX3 style): the
Raspberry Pi budget is decided per *absolute* channel width (multiples of 8 for
ncnn pack layouts, ideally 16) and the candidates A/B/C of the plan differ in
individual widths, not by a single scalar. Keeping every width explicit makes
the MAC table reproducible and the checkpoint self-describing
(``VariantSpec.to_dict()`` is stored as ``variant_spec``).

``resolve_spec(model_cfg)`` applies the config overrides (``levels``, ``act``,
``block``, ``p2_assist``, ``widths``, ``depths``, ``head_hidden``, ``pose_hidden``,
``num_keypoints``) on top of a preset and validates the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Final

LEVEL_SETS: Final[dict[str, tuple[int, ...]]] = {
    "P3P4P5": (8, 16, 32),
    "P3P5": (8, 32),
    "P4P5": (16, 32),
}
ACTIVATIONS: Final[tuple[str, ...]] = ("relu", "hswish", "silu")
BLOCKS: Final[tuple[str, ...]] = ("rep", "dw")
DEFAULT_NUM_KEYPOINTS: Final[int] = 9


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Fully resolved architecture description (immutable).

    ``widths`` = (stem, P2, P3, P4, P5) output channels, ``depths`` = bottleneck
    counts of backbone stages 1..4, ``head_hidden`` = width of the shared head
    conv, ``pose_hidden`` = width of the pose tower.
    """

    name: str
    widths: tuple[int, int, int, int, int]
    depths: tuple[int, int, int, int]
    head_hidden: int
    pose_hidden: int
    block: str = "rep"
    act: str = "relu"
    levels: str = "P3P4P5"
    p2_assist: bool = False
    default_size: int = 320
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS

    @property
    def strides(self) -> list[int]:
        return list(LEVEL_SETS[self.levels])

    def validate(self) -> "VariantSpec":
        if len(self.widths) != 5 or len(self.depths) != 4:
            raise ValueError("widths must have 5 entries and depths 4 entries")
        if any(int(w) <= 0 or int(w) % 8 for w in self.widths):
            raise ValueError(f"all widths must be positive multiples of 8, got {self.widths}")
        if any(int(d) < 1 for d in self.depths):
            raise ValueError(f"all depths must be >= 1, got {self.depths}")
        if self.head_hidden % 8 or self.pose_hidden % 8:
            raise ValueError("head_hidden and pose_hidden must be multiples of 8")
        if self.block not in BLOCKS:
            raise ValueError(f"block must be one of {BLOCKS}, got {self.block!r}")
        if self.act not in ACTIVATIONS:
            raise ValueError(f"act must be one of {ACTIVATIONS}, got {self.act!r}")
        if self.levels not in LEVEL_SETS:
            raise ValueError(f"levels must be one of {tuple(LEVEL_SETS)}, got {self.levels!r}")
        if self.p2_assist and self.levels == "P4P5":
            raise ValueError("p2_assist feeds the P3 pose branch; it requires a P3 level")
        if self.num_keypoints < 1:
            raise ValueError("num_keypoints must be >= 1")
        if self.default_size % 32:
            raise ValueError("default_size must be a multiple of 32")
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["widths"] = list(self.widths)
        d["depths"] = list(self.depths)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariantSpec":
        d = dict(data)
        d["widths"] = tuple(int(w) for w in d["widths"])
        d["depths"] = tuple(int(x) for x in d["depths"])
        return cls(**d).validate()


def _spec(name: str, widths: tuple[int, ...], depths: tuple[int, ...], head: int, pose: int,
          block: str, act: str, size: int) -> VariantSpec:
    return VariantSpec(
        name=name, widths=tuple(widths), depths=tuple(depths),  # type: ignore[arg-type]
        head_hidden=head, pose_hidden=pose, block=block, act=act, default_size=size,
    )


# CONTRACTS.md section 4.2.
PRESETS: Final[dict[str, VariantSpec]] = {
    "u_a": _spec("u_a", (16, 24, 48, 96, 128), (1, 1, 1, 1), 48, 48, "rep", "relu", 320),
    "u_b": _spec("u_b", (16, 32, 64, 128, 192), (1, 2, 2, 1), 64, 64, "rep", "relu", 320),
    "u_c": _spec("u_c", (24, 48, 96, 192, 256), (1, 2, 2, 1), 64, 96, "rep", "relu", 320),
    "n": _spec("n", (24, 48, 96, 192, 256), (1, 2, 2, 1), 96, 96, "rep", "relu", 384),
    "m": _spec("m", (32, 64, 128, 256, 384), (1, 3, 3, 1), 96, 128, "rep", "silu", 384),
    "l": _spec("l", (48, 96, 192, 384, 512), (2, 4, 4, 2), 128, 192, "rep", "silu", 512),
    "xtx3u_ctrl": _spec("xtx3u_ctrl", (24, 48, 96, 144, 144), (1, 1, 1, 1), 48, 64, "dw", "silu", 320),
}
ALIASES: Final[dict[str, str]] = {"u": "u_b"}

_OVERRIDE_KEYS: Final[tuple[str, ...]] = (
    "levels", "act", "block", "p2_assist", "widths", "depths",
    "head_hidden", "pose_hidden", "num_keypoints", "default_size",
)


def get_variant(name: str) -> VariantSpec:
    """Return the preset for ``name`` (aliases allowed, case-insensitive)."""

    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(f"Unknown variant {name!r}; expected one of {sorted(PRESETS) + sorted(ALIASES)}")
    return PRESETS[key]


def resolve_spec(model_cfg: dict[str, Any] | None) -> VariantSpec:
    """Resolve ``config["model"]`` (preset ``variant`` + overrides) into a validated spec."""

    cfg = dict(model_cfg or {})
    spec = get_variant(str(cfg.get("variant", "u_b")))
    overrides: dict[str, Any] = {}
    for key in _OVERRIDE_KEYS:
        if key not in cfg or cfg[key] is None:
            continue
        value = cfg[key]
        if key in ("widths", "depths"):
            value = tuple(int(v) for v in value)
        elif key == "p2_assist":
            value = bool(value)
        elif key in ("head_hidden", "pose_hidden", "num_keypoints", "default_size"):
            value = int(value)
        else:
            value = str(value).strip()
            value = value.upper() if key == "levels" else value.lower()
        overrides[key] = value
    unknown = set(cfg) - set(_OVERRIDE_KEYS) - {"variant"}
    if unknown:
        # Tolerate but surface typos: other owners may add keys to the model section.
        from ..utils.logging import get_logger

        get_logger(__name__).warning("Ignoring unknown model config keys: %s", sorted(unknown))
    return replace(spec, **overrides).validate()
