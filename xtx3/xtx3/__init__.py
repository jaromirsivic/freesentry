"""XTX3 Pose: efficient head-and-torso (nine-landmark) human pose detection.

Public API::

    from xtx3 import load_model, detect_poses, detect_poses_batch, Pose

Heavy imports (torch, cv2) are deferred until first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

__all__ = ["Pose", "load_model", "detect_poses", "detect_poses_batch", "poses_to_array"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .api import Pose, detect_poses, detect_poses_batch, load_model, poses_to_array


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'xtx3' has no attribute {name!r}")
