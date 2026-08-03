"""XTX3 nine-landmark metadata: the single source of truth for keypoint order,
COCO selection, left/right flip pairs, skeleton edges, and OKS sigmas.

XTX3 detects only head-and-torso landmarks. The nine retained points are a
deterministic subset of COCO's 17 keypoints (indices below); left/right
semantics are preserved exactly. Arms below the shoulders and everything below
the hips are out of scope.

OKS sigma rationale: the COCO per-keypoint sigmas were derived per landmark
(from redundant human annotations), so the subset of sigmas for the retained
landmarks remains a valid per-landmark falloff. We deliberately do NOT
renormalise them: keeping the absolute values makes XTX3's OKS directly
comparable with the same nine components of a 17-point COCO evaluation
(e.g. against the XTX2 baseline). Note the mean sigma of this subset is lower
than COCO's full-body mean (no easy knee/ankle points), i.e. the nine-point
metric is slightly *stricter* per landmark than full-body OKS.
"""

from __future__ import annotations

from typing import Final

import numpy as np

NUM_KEYPOINTS: Final[int] = 9

KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)

# Indices into COCO's 17-keypoint order that produce the XTX3 order above.
COCO_KEYPOINT_INDICES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6, 11, 12)
NUM_COCO_KEYPOINTS: Final[int] = 17

# Index pairs whose values must be swapped on a horizontal flip:
# eyes, ears, shoulders, hips.
FLIP_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
)

# Per-landmark OKS sigmas: COCO's per-keypoint values for the retained points
# (see module docstring for why the subset is kept unscaled).
OKS_SIGMAS: Final[np.ndarray] = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.107, 0.107],
    dtype=np.float32,
)

# Skeleton edges for visualisation: face connections, the shoulder line, the
# shoulder-to-hip torso sides, and the hip line. Only meaningful head/torso
# connections are drawn (no limbs).
SKELETON_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),  # nose - left eye
    (0, 2),  # nose - right eye
    (1, 3),  # left eye - left ear
    (2, 4),  # right eye - right ear
    (5, 6),  # shoulder line
    (5, 7),  # left shoulder - left hip
    (6, 8),  # right shoulder - right hip
    (7, 8),  # hip line
)

# Landmark groups used by the scale-split evaluation.
FACE_INDICES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
TORSO_INDICES: Final[tuple[int, ...]] = (5, 6, 7, 8)


def select_nine(kpts17: np.ndarray) -> np.ndarray:
    """Select the nine XTX3 landmarks from a ``(..., 17, 3)`` COCO keypoint array.

    Order is preserved exactly as :data:`KEYPOINT_NAMES`; left/right semantics
    are never reinterpreted.
    """

    arr = np.asarray(kpts17)
    if arr.shape[-2] != NUM_COCO_KEYPOINTS:
        raise ValueError(
            f"Expected {NUM_COCO_KEYPOINTS} keypoints on axis -2, got shape {arr.shape}"
        )
    return arr[..., list(COCO_KEYPOINT_INDICES), :]


def build_flip_index() -> np.ndarray:
    """Return a length-9 permutation array mapping each index to its flipped index."""

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
    if len(COCO_KEYPOINT_INDICES) != NUM_KEYPOINTS:
        raise ValueError("COCO selection index count mismatch")
    for left, right in FLIP_PAIRS:
        left_name = KEYPOINT_NAMES[left]
        right_name = KEYPOINT_NAMES[right]
        if left_name.replace("left", "") != right_name.replace("right", ""):
            raise ValueError(f"Flip pair mismatch: {left_name} <-> {right_name}")
