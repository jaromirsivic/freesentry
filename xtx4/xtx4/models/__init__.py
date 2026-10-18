"""Box-free XTX4 model: blocks, backbone, PAN neck, dual pose head, variants."""

from .head_types import HeadOutput, RawPredictions
from .organs import OrganLayer, decode_organs
from .scaling import PRESETS, VariantSpec, get_variant, resolve_spec
from .xtx4 import XTX4Model, build_model, count_macs

__all__ = [
    "XTX4Model",
    "build_model",
    "count_macs",
    "VariantSpec",
    "get_variant",
    "resolve_spec",
    "PRESETS",
    "HeadOutput",
    "RawPredictions",
    "OrganLayer",
    "decode_organs",
]
