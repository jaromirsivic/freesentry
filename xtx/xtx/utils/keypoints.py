"""COCO 17-keypoint metadata shared by losses, merge matching, and drawing.

This is the single source of truth for keypoint order, left/right flip pairs,
skeleton edges, and OKS sigmas (COCO standard).
"""

from __future__ import annotations

from typing import Final

import numpy as np

NUM_KEYPOINTS: Final[int] = 17

KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Index pairs whose values must be swapped on a horizontal flip.
FLIP_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

# COCO per-keypoint sigmas (used for OKS in both loss and cross-crop merge).
OKS_SIGMAS: Final[np.ndarray] = np.array(
    [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ],
    dtype=np.float32,
)

# Skeleton edges for visualisation (1-based COCO convention converted to 0-based).
SKELETON_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)


def build_flip_index() -> np.ndarray:
    """Return a length-17 permutation array mapping each index to its flipped index."""

    idx = np.arange(NUM_KEYPOINTS, dtype=np.int64)
    for left, right in FLIP_PAIRS:
        idx[left], idx[right] = right, left
    return idx


def validate_keypoint_meta() -> None:
    """Fail fast if the metadata arrays are internally inconsistent."""

    if len(KEYPOINT_NAMES) != NUM_KEYPOINTS:
        raise ValueError(f"Expected {NUM_KEYPOINTS} keypoint names, got {len(KEYPOINT_NAMES)}")
    if OKS_SIGMAS.shape[0] != NUM_KEYPOINTS:
        raise ValueError(f"Expected {NUM_KEYPOINTS} OKS sigmas, got {OKS_SIGMAS.shape[0]}")
