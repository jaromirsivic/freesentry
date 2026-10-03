"""Self-contained checks of the numpy decode (no torch / xtx3 needed)."""

from __future__ import annotations

import numpy as np
import pytest

from livetest.decode import anchor_count, build_anchors, decode_raw, infer_strides, to_frame_poses
from livetest.preprocess import prepare_views


@pytest.mark.parametrize("size,expected", [(256, 1344), (320, 2100), (384, 3024)])
def test_anchor_count_matches_ncnn_export(size: int, expected: int) -> None:
    assert anchor_count(size, (8, 16, 32)) == expected
    anchors, strides = build_anchors(size, (8, 16, 32))
    assert anchors.shape == (expected, 2) and strides.shape == (expected,)
    assert tuple(anchors[0]) == (4.0, 4.0)
    assert infer_strides(expected, size) == (8, 16, 32)


def test_infer_strides_with_p2_level() -> None:
    assert infer_strides(anchor_count(384, (4, 8, 16, 32)), 384) == (4, 8, 16, 32)
    assert infer_strides(12345, 384) is None


def test_zero_raw_outputs_decode_to_anchor_centres() -> None:
    size, k = 256, 9
    n = anchor_count(size, (8, 16, 32))
    box = np.zeros((n, 4), np.float32)
    cls = np.full((n, 1), -10.0, np.float32)
    cls[5] = 3.0  # one confident anchor
    kpt = np.zeros((n, k * 6), np.float32)
    kpt.reshape(n, k, 6)[5, :, 5] = 4.0  # "visible" logit dominates
    boxes, scores, kpts, vis = decode_raw(
        box, cls, kpt, network_size=size, strides=(8, 16, 32), num_keypoints=k,
        conf_threshold=0.5, max_detections=10,
    )
    anchors, _ = build_anchors(size, (8, 16, 32))
    ax, ay = anchors[5]
    d = np.log(2.0) * 8  # softplus(0) * stride
    assert scores.shape == (1,)
    np.testing.assert_allclose(scores[0], 1 / (1 + np.exp(-3.0)), rtol=1e-6)
    np.testing.assert_allclose(boxes[0], [ax - d, ay - d, ax + d, ay + d], rtol=1e-5)
    np.testing.assert_allclose(kpts[0, :, 0], ax)
    np.testing.assert_allclose(kpts[0, :, 1], ay)
    p = np.exp([0.0, 0.0, 4.0]) / np.exp([0.0, 0.0, 4.0]).sum()
    np.testing.assert_allclose(kpts[0, :, 2], p[1] + p[2], rtol=1e-6)
    assert (vis[0] == 2).all()


def test_max_detections_keeps_highest_scores() -> None:
    size, k = 256, 9
    n = anchor_count(size, (8, 16, 32))
    cls = np.linspace(-1, 1, n, dtype=np.float32)[:, None]
    _, scores, _, _ = decode_raw(
        np.zeros((n, 4)), cls, np.zeros((n, k * 6)), network_size=size, strides=(8, 16, 32),
        num_keypoints=k, conf_threshold=0.0, max_detections=7,
    )
    assert scores.shape == (7,)
    assert (np.diff(scores) <= 0).all()
    np.testing.assert_allclose(scores[0], 1 / (1 + np.exp(-1.0)), rtol=1e-6)


def test_whole_view_round_trip_maps_back_to_frame() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    prepared = prepare_views(frame, mode="whole", network_size=320)
    t = prepared.transforms[0]
    # Frame corners land in the letterboxed band and map back exactly.
    boxes = np.array([[0.0, 68.0, 320.0, 248.0]], np.float32)
    poses = to_frame_poses(
        boxes, np.array([0.9], np.float32), np.zeros((1, 9, 3), np.float32),
        np.zeros((1, 9), np.int64), transform=t, view=0,
    )
    np.testing.assert_allclose(poses[0].bbox, [0, 0, 1280, 720], atol=1e-3)
