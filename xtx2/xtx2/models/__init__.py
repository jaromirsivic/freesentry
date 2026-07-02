"""XTX2 network: blocks, backbone, PAN neck, dual NMS-free pose head."""

from .xtx2 import XTX2Model, build_model

__all__ = ["XTX2Model", "build_model"]
