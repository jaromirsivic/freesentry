"""XTX4 Pose: box-free nine-landmark (head + torso) multi-person pose detection.

Public API (lazy)::

    from xtx4 import load_model, detect_poses, Pose
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["Pose", "load_model", "detect_poses", "detect_poses_batch", "poses_to_array"]

if TYPE_CHECKING:  # pragma: no cover
    from .api import Pose, detect_poses, detect_poses_batch, load_model, poses_to_array


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'xtx4' has no attribute {name!r}")
