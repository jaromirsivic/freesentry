"""Host-side NumPy decode of the raw XTX4 graph outputs (CONTRACTS.md sections 3 and 12).

This is the deployment twin of :func:`xtx4.models.head_types.decode_detections`
and must stay numerically identical to it (up to float32 rounding of the
transcendental functions); ``tests/test_runtime.py`` compares both on random
raw tensors. It deliberately imports **no torch**: the Raspberry Pi runtime only
ships ``numpy`` + ``opencv`` + ``ncnn``.

Raw tensors per anchor (``N`` anchors over all pyramid levels, ``K`` keypoints):

* ``person`` ``[1, N, 1]`` (ONNX) or ``(N, 1)`` (ncnn): person logit.
* ``pose`` ``[1, N, K*5]`` / ``(N, K*5)``: keypoint-major
  ``(dx, dy, log_sigma, present_logit, visible_logit)`` per keypoint.

Decode (``decode_version = 1``)::

    score        = sigmoid(person)
    x, y         = anchor + (dx, dy) * stride
    sigma_units  = softplus(log_sigma) + sigma_floor        (stride units)
    sigma_px     = sigma_units * stride
    present_prob = sigmoid(present_logit); visible_prob = sigmoid(visible_logit)
    conf         = present_prob * exp(-sigma_conf_coeff * sigma_units)
    visibility   = 0 if present_prob < present_threshold
                   else (2 if visible_prob >= visible_threshold else 1)   (defaults 0.5 / 0.5)

Why the score filter runs **before** the keypoint decode: with ``N = 2100``
anchors at 320 and typically < 10 persons, decoding only the selected rows makes
the host post-processing O(D) instead of O(N) (< 0.1 ms on the Pi).

Anchors: centres ``(i + 0.5) * stride``, levels in ascending stride order,
row-major (y outer, x inner) within a level -- identical to
``head_types.build_anchors``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Iterable, Sequence

import numpy as np

from ..utils.keypoints import OKS_SIGMAS

KPT_CHANNELS_PER_POINT: Final[int] = 5
SIGMA_FLOOR_UNITS: Final[float] = 0.05
DECODE_VERSION: Final[int] = 1

# Stride sets of the CONTRACTS 4.2 level options (P3P4P5, P3P5, P4P5).
KNOWN_STRIDE_SETS: Final[tuple[tuple[int, ...], ...]] = ((8, 16, 32), (8, 32), (16, 32))

# Visibility-class decision thresholds (decode.present_threshold / decode.visible_threshold).
PRESENT_THRESHOLD: Final[float] = 0.5
VISIBLE_THRESHOLD: Final[float] = 0.5

# OKS-NMS defaults (host safety net; the o2o head should not need it).
DEFAULT_NMS_KPT_THRESHOLD: Final[float] = 0.5
DEFAULT_NMS_MIN_SIDE_PX: Final[float] = 16.0
DEFAULT_NMS_SCALE_FACTOR: Final[float] = 1.5  # decode.oks_nms_scale_factor


@dataclass(slots=True)
class Pose:
    """One detected person.

    Attributes:
        score: person score ``sigmoid(person_logit)``.
        keypoints: ``(K, 3)`` float32 ``(x, y, conf)`` in frame pixels (network
            pixels before :func:`to_frame_poses`).
        visibility: ``(K,)`` int64: 0 absent, 1 occluded, 2 visible.
        sigma_px: ``(K,)`` float32 RLE scale in the same pixel space as ``keypoints``.
        source_view: 0 whole-frame, 1|2|3 pyramid view.
        anchor_index: index of the emitting anchor in the raw output (-1 if unknown).
    """

    score: float
    keypoints: np.ndarray
    visibility: np.ndarray
    sigma_px: np.ndarray
    source_view: int = 0
    anchor_index: int = -1
    extras: dict = field(default_factory=dict)

    def extent(self, kpt_threshold: float = 0.0) -> tuple[float, float, float, float] | None:
        """Axis-aligned extent ``(x0, y0, x1, y1)`` of the points with ``conf >= kpt_threshold``."""

        mask = self.keypoints[:, 2] >= kpt_threshold
        if not mask.any():
            return None
        xy = self.keypoints[mask, :2]
        return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


# --------------------------------------------------------------------- anchors
def anchor_count(network_size: int, strides: Iterable[int]) -> int:
    """``N = sum((S // stride)^2)`` over the active levels."""

    return int(sum((int(network_size) // int(s)) ** 2 for s in strides))


def infer_strides(num_anchors: int, network_size: int) -> tuple[int, ...] | None:
    """Guess the stride set from the anchor count (``None`` if no known set matches)."""

    for strides in KNOWN_STRIDE_SETS:
        if anchor_count(network_size, strides) == num_anchors:
            return strides
    return None


@lru_cache(maxsize=32)
def _build_anchors_cached(network_size: int, strides: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    stride_list: list[np.ndarray] = []
    for stride in strides:
        cells = network_size // stride
        coords = (np.arange(cells, dtype=np.float32) + np.float32(0.5)) * np.float32(stride)
        grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
        points.append(np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1))
        stride_list.append(np.full(cells * cells, float(stride), dtype=np.float32))
    anchors = np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float32)
    stride_arr = np.ascontiguousarray(np.concatenate(stride_list, axis=0), dtype=np.float32)
    anchors.setflags(write=False)
    stride_arr.setflags(write=False)
    return anchors, stride_arr


def build_anchors(network_size: int, strides: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return read-only ``(anchors (N, 2) centre px, strides (N,))`` float32 (cached).

    Same order as ``head_types.build_anchors``: ascending stride, row-major.
    """

    if int(network_size) % max(int(s) for s in strides):
        raise ValueError(f"network_size {network_size} is not divisible by the strides {list(strides)}")
    return _build_anchors_cached(int(network_size), tuple(int(s) for s in strides))


# ------------------------------------------------------------------ elementwise
def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32, copy=False)


def _softplus(x: np.ndarray) -> np.ndarray:
    """``torch.nn.functional.softplus`` (beta 1, threshold 20) in float32."""

    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        soft = np.log1p(np.exp(np.minimum(x, np.float32(20.0))))
    return np.where(x > np.float32(20.0), x, soft).astype(np.float32, copy=False)


# ----------------------------------------------------------------------- decode
def _as_person(person: np.ndarray, num_anchors: int) -> np.ndarray:
    arr = np.asarray(person, dtype=np.float32)
    if arr.size != num_anchors:
        raise ValueError(f"person output has {arr.size} values, expected N = {num_anchors}")
    return arr.reshape(num_anchors)


def _as_pose(pose: np.ndarray, num_anchors: int, num_keypoints: int) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32)
    expected = num_anchors * num_keypoints * KPT_CHANNELS_PER_POINT
    if arr.size != expected:
        raise ValueError(
            f"pose output has {arr.size} values, expected N x K x 5 = {num_anchors} x {num_keypoints} x 5"
        )
    return arr.reshape(num_anchors, num_keypoints, KPT_CHANNELS_PER_POINT)


def person_scores(person: np.ndarray) -> np.ndarray:
    """``sigmoid`` of the person logits, flattened to ``(N,)`` float32."""

    return _sigmoid(np.asarray(person, dtype=np.float32).reshape(-1))


def select_candidates(
    scores: np.ndarray, *, conf_threshold: float, max_detections: int
) -> tuple[np.ndarray, int]:
    """Anchor indices with ``score >= conf_threshold``, sorted by descending score.

    Ties keep ascending anchor order (stable sort). Returns ``(keep (D,), num_above)``
    where ``num_above`` counts all anchors above the threshold before the top-k cut.
    """

    keep = np.flatnonzero(scores >= np.float32(conf_threshold))
    num_above = int(keep.size)
    if num_above == 0 or max_detections <= 0:
        return np.zeros((0,), dtype=np.int64), num_above
    order = np.argsort(-scores[keep], kind="stable")[: int(max_detections)]
    return keep[order].astype(np.int64, copy=False), num_above


def decode_selected(
    pose: np.ndarray,
    keep: np.ndarray,
    *,
    network_size: int,
    strides: Sequence[int],
    num_keypoints: int,
    sigma_conf_coeff: float = 0.0,
    sigma_floor: float = SIGMA_FLOOR_UNITS,
    present_threshold: float = PRESENT_THRESHOLD,
    visible_threshold: float = VISIBLE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the keypoints of the anchors ``keep`` only.

    Returns ``(keypoints (D, K, 3) [x, y, conf], visibility (D, K) int64, sigma_px (D, K))``
    in network pixels.
    """

    anchors, stride_arr = build_anchors(network_size, strides)
    num_anchors = anchors.shape[0]
    kp_all = _as_pose(pose, num_anchors, num_keypoints)
    if keep.size == 0:
        return (
            np.zeros((0, num_keypoints, 3), np.float32),
            np.zeros((0, num_keypoints), np.int64),
            np.zeros((0, num_keypoints), np.float32),
        )
    kp = kp_all[keep]  # (D, K, 5)
    a = anchors[keep]
    st = stride_arr[keep][:, None]  # (D, 1)
    kx = a[:, 0:1] + kp[..., 0] * st
    ky = a[:, 1:2] + kp[..., 1] * st
    sigma_units = _softplus(kp[..., 2]) + np.float32(sigma_floor)
    present = _sigmoid(kp[..., 3])
    visible = _sigmoid(kp[..., 4])
    conf = present
    if sigma_conf_coeff != 0.0:
        conf = conf * np.exp(np.float32(-float(sigma_conf_coeff)) * sigma_units)
    visibility = np.where(
        present < np.float32(present_threshold), 0, np.where(visible >= np.float32(visible_threshold), 2, 1)
    ).astype(np.int64)
    keypoints = np.stack([kx, ky, conf], axis=-1).astype(np.float32, copy=False)
    sigma_px = (sigma_units * st).astype(np.float32, copy=False)
    return keypoints, visibility, sigma_px


def decode_raw(
    person: np.ndarray,
    pose: np.ndarray,
    *,
    network_size: int,
    strides: Sequence[int],
    num_keypoints: int,
    conf_threshold: float,
    max_detections: int,
    sigma_conf_coeff: float = 0.0,
    sigma_floor: float = SIGMA_FLOOR_UNITS,
    present_threshold: float = PRESENT_THRESHOLD,
    visible_threshold: float = VISIBLE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's raw outputs to network-pixel detections (no NMS).

    Returns ``(scores (D,), keypoints (D, K, 3) [x, y, conf], visibility (D, K) int64,
    sigma_px (D, K))`` sorted by descending score, ``D <= max_detections``.
    Equivalent to ``head_types.decode_detections`` for a batch of one.
    """

    anchors, _ = build_anchors(network_size, strides)
    scores_all = _sigmoid(_as_person(person, anchors.shape[0]))
    keep, _ = select_candidates(scores_all, conf_threshold=conf_threshold, max_detections=max_detections)
    keypoints, visibility, sigma_px = decode_selected(
        pose, keep, network_size=network_size, strides=strides, num_keypoints=num_keypoints,
        sigma_conf_coeff=sigma_conf_coeff, sigma_floor=sigma_floor,
        present_threshold=present_threshold, visible_threshold=visible_threshold,
    )
    return scores_all[keep].astype(np.float32, copy=False), keypoints, visibility, sigma_px


# ------------------------------------------------------------- frame mapping
def to_frame_poses(
    scores: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    sigma_px: np.ndarray,
    *,
    transform: np.ndarray,
    view: int = 0,
    anchor_indices: np.ndarray | None = None,
) -> list[Pose]:
    """Map network-pixel detections to frame pixels with the view affine ``T`` (2x3,
    network px -> frame px). ``sigma_px`` is scaled by the mean absolute axis scale
    of ``T`` (the letterbox is isotropic up to +-0.5 px rounding)."""

    if scores.shape[0] == 0:
        return []
    transform = np.asarray(transform, dtype=np.float64)
    lin = transform[:, :2]
    offset = transform[:, 2]
    kpts = keypoints.astype(np.float64, copy=True)
    kpts[..., :2] = keypoints[..., :2].astype(np.float64) @ lin.T + offset
    sigma_scale = 0.5 * (abs(lin[0, 0]) + abs(lin[1, 1]))
    sig = sigma_px.astype(np.float64) * sigma_scale
    return [
        Pose(
            score=float(scores[i]),
            keypoints=kpts[i].astype(np.float32),
            visibility=np.asarray(visibility[i], dtype=np.int64),
            sigma_px=sig[i].astype(np.float32),
            source_view=int(view),
            anchor_index=int(anchor_indices[i]) if anchor_indices is not None else -1,
        )
        for i in range(scores.shape[0])
    ]


# --------------------------------------------------------------------- OKS-NMS
def _extent_boxes(
    kpts: np.ndarray, kpt_threshold: float, scale_factor: float, min_side: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per pose: ``(area (D,), box (D, 4) xyxy)`` from the extent of its confident points.

    The extent box of the points with ``conf >= kpt_threshold`` (all points when none
    is confident) is expanded by ``scale_factor`` about its centre.
    ``area = max((f w)(f h), min_side^2)`` is the OKS scale ``s^2``; the expanded box
    (each side floored at ``min_side``) is used for the optional IoU test.
    """

    use = kpts[..., 2] >= kpt_threshold  # (D, K)
    use = use | ~use.any(axis=1, keepdims=True)
    x = kpts[..., 0].astype(np.float64)
    y = kpts[..., 1].astype(np.float64)
    x0, x1 = np.where(use, x, np.inf).min(axis=1), np.where(use, x, -np.inf).max(axis=1)
    y0, y1 = np.where(use, y, np.inf).min(axis=1), np.where(use, y, -np.inf).max(axis=1)
    f = float(scale_factor)
    w, h = f * (x1 - x0), f * (y1 - y0)
    area = np.maximum(w * h, float(min_side) ** 2)
    bw, bh = np.maximum(w, min_side), np.maximum(h, min_side)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    boxes = np.stack([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], axis=1)
    return area, boxes


def _oks_from_arrays(
    kpts: np.ndarray, sigmas: np.ndarray, kpt_threshold: float, scale_factor: float, min_side: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(oks (D, D), boxes (D, 4))`` with ``oks[i, j]`` scored against reference ``j``."""

    sig = np.asarray(sigmas, dtype=np.float64)
    if sig.shape[0] != kpts.shape[1]:
        raise ValueError(f"{sig.shape[0]} sigmas for {kpts.shape[1]} keypoints")
    areas, boxes = _extent_boxes(kpts, kpt_threshold, scale_factor, min_side)
    diff = kpts[:, None, :, :2] - kpts[None, :, :, :2]
    d2 = (diff**2).sum(-1)  # (D, D, K)
    var2 = (2.0 * sig) ** 2
    e = d2 / (2.0 * np.maximum(areas, 1e-6)[None, :, None] * var2[None, None, :])
    use = kpts[..., 2] >= kpt_threshold
    use = use | ~use.any(axis=1, keepdims=True)  # (D, K) of the reference pose
    ref = np.broadcast_to(use[None, :, :], e.shape)
    return (np.exp(-e) * ref).sum(-1) / np.maximum(ref.sum(-1), 1), boxes


def box_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU ``(D, D)`` of xyxy boxes."""

    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = np.clip(rb - lt, 0.0, None).prod(-1)
    area = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None).prod(-1)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def oks_matrix(
    poses: Sequence[Pose],
    *,
    sigmas: np.ndarray | Sequence[float] = OKS_SIGMAS,
    kpt_threshold: float = DEFAULT_NMS_KPT_THRESHOLD,
    min_side_px: float = DEFAULT_NMS_MIN_SIDE_PX,
    scale_factor: float = DEFAULT_NMS_SCALE_FACTOR,
) -> np.ndarray:
    """Pairwise OKS ``(D, D)``: ``oks[i, j]`` scores pose ``i`` against pose ``j`` as the reference.

    OKS scale: there is no box (and no GT area) at runtime, so the reference pose's
    scale is derived from the **extent box of its confident points**
    (``conf >= kpt_threshold``; all points if none), expanded by ``scale_factor``:
    ``s^2 = max((f w)(f h), min_side_px^2)``. ``f = 1.5`` approximates the
    head+torso box (the assignment fallback of CONTRACTS 6.2). The evaluation uses
    the full-person GT area, which is larger, so for the same pair the NMS OKS is
    *lower* than the metric's OKS -- a duplicate the metric counts can survive NMS.
    Raise ``scale_factor`` (or set ``iou_threshold`` in :func:`oks_nms_indices`) to
    suppress more. Only the reference's confident points contribute. Formula as
    COCO: ``mean_k exp(-d_k^2 / (2 s^2 (2 sigma_k)^2))``.
    """

    if len(poses) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    kpts = np.stack([p.keypoints for p in poses]).astype(np.float64)
    return _oks_from_arrays(kpts, np.asarray(sigmas), kpt_threshold, scale_factor, min_side_px)[0]


def oks_nms_indices(
    keypoints: np.ndarray,
    scores: np.ndarray | None,
    *,
    threshold: float = 0.5,
    sigmas: np.ndarray | Sequence[float] = OKS_SIGMAS,
    kpt_threshold: float = DEFAULT_NMS_KPT_THRESHOLD,
    min_side_px: float = DEFAULT_NMS_MIN_SIDE_PX,
    scale_factor: float = DEFAULT_NMS_SCALE_FACTOR,
    iou_threshold: float | None = None,
    max_detections: int | None = None,
) -> np.ndarray:
    """Greedy OKS-NMS on arrays ``(D, K, 3)``; returns kept indices in visiting order.

    Visiting order is descending score (ties: lower index first), or the given
    order when ``scores`` is None. A candidate ``j`` is dropped when, for an
    already kept pose ``i``, ``OKS(i, j) >= threshold`` (``j`` is the reference,
    see :func:`oks_matrix`) **or**, when ``iou_threshold`` is set, the IoU of
    their expanded extent boxes is ``>= iou_threshold`` (XTX3's merge rule was
    OKS >= 0.5 OR box IoU >= 0.5). Single implementation behind :func:`oks_nms`,
    :func:`merge_views` and ``xtx4.api.oks_nms``.
    """

    kpts = np.asarray(keypoints, dtype=np.float64)
    n = kpts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if scores is None:
        order = np.arange(n)
    else:
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    kpts = kpts[order]
    oks, boxes = _oks_from_arrays(kpts, np.asarray(sigmas), kpt_threshold, scale_factor, min_side_px)
    suppress = oks >= threshold
    if iou_threshold is not None:
        suppress |= box_iou_matrix(boxes) >= float(iou_threshold)
    kept: list[int] = []
    removed = np.zeros(n, dtype=bool)
    for i in range(n):
        if removed[i]:
            continue
        kept.append(i)
        if max_detections is not None and len(kept) >= max_detections:
            break
        removed |= suppress[i]
    return order[np.asarray(kept, dtype=np.int64)].astype(np.int64)


def oks_nms(
    poses: Sequence[Pose],
    threshold: float = 0.5,
    sigmas: np.ndarray | Sequence[float] = OKS_SIGMAS,
    *,
    kpt_threshold: float = DEFAULT_NMS_KPT_THRESHOLD,
    min_side_px: float = DEFAULT_NMS_MIN_SIDE_PX,
    scale_factor: float = DEFAULT_NMS_SCALE_FACTOR,
    iou_threshold: float | None = None,
    max_detections: int | None = None,
    presorted: bool = False,
) -> list[Pose]:
    """Greedy OKS-NMS over :class:`Pose` objects (rule: :func:`oks_nms_indices`).

    Poses are visited by descending score (or in the given order when ``presorted``,
    used for the pyramid view priority); the result is sorted by score.
    O(D^2) with ``D <= max_detections`` -- negligible on the host.
    """

    if not poses:
        return []
    keep = oks_nms_indices(
        np.stack([p.keypoints for p in poses]),
        None if presorted else np.array([p.score for p in poses], dtype=np.float64),
        threshold=threshold, sigmas=sigmas, kpt_threshold=kpt_threshold, min_side_px=min_side_px,
        scale_factor=scale_factor, iou_threshold=iou_threshold, max_detections=max_detections,
    )
    result = [poses[int(i)] for i in keep]
    result.sort(key=lambda p: -p.score)
    return result


def merge_views(
    by_view: dict[int, list[Pose]],
    *,
    threshold: float,
    sigmas: np.ndarray | Sequence[float] = OKS_SIGMAS,
    max_detections: int,
    nms: bool = False,
    kpt_threshold: float = DEFAULT_NMS_KPT_THRESHOLD,
    scale_factor: float = DEFAULT_NMS_SCALE_FACTOR,
    iou_threshold: float | None = None,
) -> list[Pose]:
    """Combine per-view detections.

    One view (whole-frame mode): score sort + top-k, optional OKS-NMS. Several views
    (pyramid): view priority 1 > 2 > 3 then score; a pose is dropped when it matches
    an accepted one (OKS ``>= threshold`` or expanded-box IoU ``>= iou_threshold``),
    always applied across views.
    """

    kw = dict(kpt_threshold=kpt_threshold, scale_factor=scale_factor, iou_threshold=iou_threshold,
              max_detections=max_detections)
    if len(by_view) <= 1:
        poses = [p for view in by_view.values() for p in view]
        if nms:
            return oks_nms(poses, threshold, sigmas, **kw)
        poses.sort(key=lambda p: -p.score)
        return poses[:max_detections]
    ordered: list[Pose] = []
    for key in sorted(by_view):
        ordered.extend(sorted(by_view[key], key=lambda p: -p.score))
    return oks_nms(ordered, threshold, sigmas, presorted=True, **kw)
