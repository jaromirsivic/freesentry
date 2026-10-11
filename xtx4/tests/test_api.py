"""Public API tests on a random-weight model: shapes, coordinate mapping, ordering, OKS-NMS."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xtx4.api import (
    POSE_ARRAY_SIZE,
    Pose,
    detect_poses,
    detect_poses_batch,
    load_model,
    oks_nms,
    poses_to_array,
)
from xtx4.engine.trainer import load_config
from xtx4.utils.keypoints import NUM_KEYPOINTS


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = load_config()
    config["model"]["variant"] = "u_a"
    return load_model(None, device="cpu", fuse=True, config=config)


def _image(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def test_load_model_attaches_inference_attributes(model):
    assert model.fused
    assert model.network_size == 320  # data.eval_size of configs/u.json
    assert model.decode["max_detections"] == 100
    assert model.device_str == "cpu"


def test_detect_poses_shapes_and_bounds(model):
    image = _image(640, 360, seed=1)
    poses = detect_poses(image, model, conf_threshold=0.0, max_detections=50)
    assert 0 < len(poses) <= 50
    scores = [p.score for p in poses]
    assert scores == sorted(scores, reverse=True)
    for pose in poses:
        assert isinstance(pose, Pose)
        assert pose.keypoints.shape == (NUM_KEYPOINTS, 3)
        assert pose.keypoints_norm.shape == (NUM_KEYPOINTS, 3)
        assert pose.visibility.shape == (NUM_KEYPOINTS,) and pose.visibility.dtype == np.int64
        assert pose.sigma_px.shape == (NUM_KEYPOINTS,) and np.all(pose.sigma_px > 0)
        assert 0.0 <= pose.score <= 1.0
        assert np.all((pose.keypoints[:, 2] >= 0) & (pose.keypoints[:, 2] <= 1))
        assert set(np.unique(pose.visibility)).issubset({0, 1, 2})
        np.testing.assert_allclose(pose.keypoints_norm[:, 0], pose.keypoints[:, 0] / 640, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pose.keypoints_norm[:, 1], pose.keypoints[:, 1] / 360, rtol=1e-5, atol=1e-5)
    # Random weights keep offsets small, so decoded points stay close to the frame:
    # anchors are inside the letterboxed image region, mapped back they are inside the image
    # up to a margin of one stride at the original scale.
    kpts = np.stack([p.keypoints for p in poses])
    margin = 32 * (640 / 320) * 2
    assert kpts[..., 0].min() > -margin and kpts[..., 0].max() < 640 + margin
    assert kpts[..., 1].min() > -margin and kpts[..., 1].max() < 360 + margin
    arr = poses_to_array(poses)
    assert arr.shape == (len(poses), POSE_ARRAY_SIZE) and arr.dtype == np.float32
    assert poses_to_array([]).shape == (0, POSE_ARRAY_SIZE)


def test_detect_poses_batch_matches_single(model):
    images = [_image(320, 240, seed=2), _image(500, 700, seed=3)]
    batch = detect_poses_batch(images, model, conf_threshold=0.0, max_detections=20)
    assert len(batch) == 2
    for image, poses in zip(images, batch):
        single = detect_poses(image, model, conf_threshold=0.0, max_detections=20)
        assert len(single) == len(poses)
        for a, b in zip(single, poses):
            assert abs(a.score - b.score) < 1e-5
            np.testing.assert_allclose(a.keypoints, b.keypoints, atol=1e-3)
    assert detect_poses_batch([], model) == []


def test_conf_threshold_and_max_detections(model):
    image = _image(400, 400, seed=4)
    all_poses = detect_poses(image, model, conf_threshold=0.0, max_detections=1000)
    few = detect_poses(image, model, conf_threshold=0.0, max_detections=3)
    assert len(few) == 3
    assert [p.score for p in few] == [p.score for p in all_poses[:3]]
    high = detect_poses(image, model, conf_threshold=0.999)
    assert all(p.score >= 0.999 for p in high)


def test_network_size_override_keeps_image_coordinates(model):
    image = _image(640, 480, seed=5)
    for size in (256, 384):
        poses = detect_poses(image, model, conf_threshold=0.0, max_detections=10, network_size=size)
        assert poses
        kpts = np.stack([p.keypoints for p in poses])
        assert kpts[..., 0].max() < 640 + 64 * 640 / size and kpts[..., 1].max() < 480 + 64 * 640 / size


def test_oks_nms_removes_duplicate():
    rng = np.random.default_rng(0)
    base = np.zeros((NUM_KEYPOINTS, 3), np.float32)
    base[:, 0] = 100 + rng.uniform(-30, 30, NUM_KEYPOINTS)
    base[:, 1] = 100 + rng.uniform(-40, 40, NUM_KEYPOINTS)
    base[:, 2] = 0.9
    duplicate = base.copy()
    duplicate[:, :2] += 0.5  # sub-pixel jitter -> OKS ~ 1
    far = base.copy()
    far[:, 0] += 400  # a distinct person
    kpts = np.stack([base, duplicate, far])
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = oks_nms(kpts, scores, threshold=0.5)
    assert keep.tolist() == [0, 2]
    # The injected duplicate also disappears through the public entry point.
    assert oks_nms(kpts[[1, 0]], scores[[1, 0]], threshold=0.5).tolist() == [1]


def test_detect_poses_oks_nms_flag(model):
    image = _image(480, 320, seed=6)
    without = detect_poses(image, model, conf_threshold=0.0, max_detections=100, oks_nms=False)
    with_nms = detect_poses(image, model, conf_threshold=0.0, max_detections=100, oks_nms=True)
    assert len(with_nms) <= len(without)
    scores = [p.score for p in with_nms]
    assert scores == sorted(scores, reverse=True)
