"""XTX Pose - extremely fast, NMS-free human pose estimation (from scratch).

Public API:
    load_model(weights, variant=..., device=...) -> XTXModel
    detect_poses(image, ...) -> list[Pose]
    detect_poses_batch(images, ...) -> list[list[Pose]]
    Pose  (dataclass)

The public symbols are imported lazily (PEP 562) so that lightweight sub-modules
(e.g. ``xtx.utils``) can be used without importing torch / opencv eagerly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Pose", "detect_poses", "detect_poses_batch", "load_model"]

__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .api import Pose, detect_poses, detect_poses_batch, load_model


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'xtx' has no attribute {name!r}")
