"""Nine-landmark metadata tests: COCO selection, flip semantics, sigmas."""

from __future__ import annotations

import numpy as np
import pytest

from xtx3.utils.keypoints import (
    COCO_KEYPOINT_INDICES,
    FLIP_PAIRS,
    KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
    NUM_KEYPOINTS,
    OKS_SIGMAS,
    SKELETON_EDGES,
    build_flip_index,
    select_nine,
    validate_keypoint_meta,
)

# The full COCO order, for cross-checking the deterministic selection.
_COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_COCO_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
     0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089], dtype=np.float32,
)


class TestMetadata:
    def test_meta_is_consistent(self) -> None:
        validate_keypoint_meta()

    def test_selection_names_match_coco(self) -> None:
        assert tuple(_COCO_NAMES[i] for i in COCO_KEYPOINT_INDICES) == KEYPOINT_NAMES

    def test_sigmas_are_coco_subset(self) -> None:
        np.testing.assert_allclose(OKS_SIGMAS, _COCO_SIGMAS[list(COCO_KEYPOINT_INDICES)])

    def test_skeleton_edges_in_range(self) -> None:
        for a, b in SKELETON_EDGES:
            assert 0 <= a < NUM_KEYPOINTS
            assert 0 <= b < NUM_KEYPOINTS


class TestSelectNine:
    def test_selects_correct_rows(self) -> None:
        kpts17 = np.arange(NUM_COCO_KEYPOINTS * 3, dtype=np.float32).reshape(17, 3)
        nine = select_nine(kpts17)
        assert nine.shape == (NUM_KEYPOINTS, 3)
        for out_idx, src_idx in enumerate(COCO_KEYPOINT_INDICES):
            np.testing.assert_array_equal(nine[out_idx], kpts17[src_idx])

    def test_batched_selection(self) -> None:
        kpts = np.random.default_rng(0).random((5, 17, 3)).astype(np.float32)
        nine = select_nine(kpts)
        assert nine.shape == (5, NUM_KEYPOINTS, 3)
        np.testing.assert_array_equal(nine[:, 5], kpts[:, 5])  # left shoulder
        np.testing.assert_array_equal(nine[:, 7], kpts[:, 11])  # left hip

    def test_wrong_count_raises(self) -> None:
        with pytest.raises(ValueError):
            select_nine(np.zeros((9, 3)))


class TestFlip:
    def test_flip_pairs_swap_left_right(self) -> None:
        for left, right in FLIP_PAIRS:
            assert KEYPOINT_NAMES[left].startswith("left")
            assert KEYPOINT_NAMES[right].startswith("right")
            assert KEYPOINT_NAMES[left][4:] == KEYPOINT_NAMES[right][5:]

    def test_flip_index_is_involution(self) -> None:
        idx = build_flip_index()
        assert idx.shape == (NUM_KEYPOINTS,)
        np.testing.assert_array_equal(idx[idx], np.arange(NUM_KEYPOINTS))
        assert idx[0] == 0  # nose maps to itself
        assert idx[1] == 2 and idx[2] == 1  # eyes swap
        assert idx[7] == 8 and idx[8] == 7  # hips swap
