"""Host-side decode of raw XTX3 head outputs (numpy port of ``xtx3.models.head``).

Raw tensors per anchor (``N`` anchors over all pyramid levels):

* ``box`` (N, 4): raw ltrb; distance = ``softplus(raw) * stride`` around the anchor.
* ``cls`` (N, 1): person logit; score = ``sigmoid``.
* ``kpt`` (N, K*6): per keypoint ``(dx, dy, log_sigma, vis0, vis1, vis2)``;
  ``x = anchor_x + dx * stride``, conf = ``softmax(vis)[1] + softmax(vis)[2]``,
  visibility class = ``argmax(vis)`` (0 absent, 1 occluded, 2 visible).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

KPT_CHANNELS_PER_POINT = 6
KNOWN_STRIDE_SETS: tuple[tuple[int, ...], ...] = ((8, 16, 32), (4, 8, 16, 32))


@dataclass(slots=True)
class Pose:
    """One detected person in frame pixel coordinates."""

    bbox: np.ndarray  # (4,) xyxy
    score: float
    keypoints: np.ndarray  # (K, 3): x, y, conf
    visibility: np.ndarray  # (K,) int: 0 absent, 1 occluded, 2 visible
    source_view: int  # 0 whole-frame | 1|2|3 pyramid view


def anchor_count(network_size: int, strides: tuple[int, ...] | list[int]) -> int:
    return sum((network_size // s) ** 2 for s in strides)


def infer_strides(num_anchors: int, network_size: int) -> tuple[int, ...] | None:
    """Guess the stride set from the number of anchors the network emitted."""

    for strides in KNOWN_STRIDE_SETS:
        if anchor_count(network_size, strides) == num_anchors:
            return strides
    return None


@lru_cache(maxsize=16)
def build_anchors(network_size: int, strides: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(anchors (N,2) centre px, strides (N,))`` for a square input."""

    points: list[np.ndarray] = []
    stride_list: list[np.ndarray] = []
    for stride in strides:
        cells = network_size // stride
        coords = (np.arange(cells, dtype=np.float32) + 0.5) * stride
        grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
        points.append(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1))
        stride_list.append(np.full(cells * cells, float(stride), dtype=np.float32))
    anchors = np.concatenate(points, axis=0)
    stride_arr = np.concatenate(stride_list, axis=0)
    anchors.setflags(write=False)
    stride_arr.setflags(write=False)
    return anchors, stride_arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def decode_raw(
    box: np.ndarray,
    cls: np.ndarray,
    kpt: np.ndarray,
    *,
    network_size: int,
    strides: tuple[int, ...],
    num_keypoints: int,
    conf_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw outputs to network-pixel detections above ``conf_threshold``.

    Returns ``(boxes (D,4), scores (D,), keypoints (D,K,3), visibility (D,K))``
    sorted by descending score.
    """

    anchors, stride_arr = build_anchors(network_size, tuple(strides))
    num_anchors = anchors.shape[0]
    box = np.asarray(box, dtype=np.float32).reshape(num_anchors, 4)
    logits = np.asarray(cls, dtype=np.float32).reshape(num_anchors)
    kpt = np.asarray(kpt, dtype=np.float32).reshape(num_anchors, num_keypoints, KPT_CHANNELS_PER_POINT)

    scores = _sigmoid(logits)
    keep = np.flatnonzero(scores >= conf_threshold)
    if keep.size == 0:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0, num_keypoints, 3), np.float32),
            np.zeros((0, num_keypoints), np.int64),
        )
    keep = keep[np.argsort(-scores[keep], kind="stable")][:max_detections]

    s = scores[keep]
    a = anchors[keep]
    st = stride_arr[keep][:, None]

    dist = _softplus(box[keep]) * st
    boxes = np.stack(
        [a[:, 0] - dist[:, 0], a[:, 1] - dist[:, 1], a[:, 0] + dist[:, 2], a[:, 1] + dist[:, 3]],
        axis=1,
    )

    kp = kpt[keep]
    kx = a[:, 0:1] + kp[..., 0] * st
    ky = a[:, 1:2] + kp[..., 1] * st
    vis_logits = kp[..., 3:6]
    vis_exp = np.exp(vis_logits - vis_logits.max(axis=-1, keepdims=True))
    vis_prob = vis_exp / vis_exp.sum(axis=-1, keepdims=True)
    conf = vis_prob[..., 1] + vis_prob[..., 2]
    vis_class = vis_logits.argmax(axis=-1).astype(np.int64)

    keypoints = np.stack([kx, ky, conf], axis=-1).astype(np.float32)
    return boxes.astype(np.float32), s.astype(np.float32), keypoints, vis_class


def to_frame_poses(
    boxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    *,
    transform: np.ndarray,
    view: int,
) -> list[Pose]:
    """Map network-pixel detections to frame pixels with the view affine ``T_k``."""

    if scores.shape[0] == 0:
        return []
    lin = transform[:, :2]
    offset = transform[:, 2]
    c1 = boxes[:, 0:2] @ lin.T + offset
    c2 = boxes[:, 2:4] @ lin.T + offset
    frame_boxes = np.concatenate([np.minimum(c1, c2), np.maximum(c1, c2)], axis=1)
    kpts = keypoints.copy()
    kpts[..., :2] = keypoints[..., :2] @ lin.T + offset
    return [
        Pose(
            bbox=frame_boxes[i].astype(np.float32),
            score=float(scores[i]),
            keypoints=kpts[i].astype(np.float32),
            visibility=visibility[i],
            source_view=view,
        )
        for i in range(scores.shape[0])
    ]
