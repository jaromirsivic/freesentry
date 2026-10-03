"""Cross-view dedup for pyramid mode (port of ``xtx3.engine.merge``, threshold mode).

Priority is view 1 > view 2 > view 3: a detection from a finer view is dropped
when it matches (OKS or IoU above threshold) an already accepted one. Whole-frame
mode has a single view, so this reduces to a score sort + top-k.
"""

from __future__ import annotations

import numpy as np

from .decode import Pose

VIEW_PRIORITY = (0, 1, 2, 3)


def _box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = float(wh[0] * wh[1])
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _oks(a: Pose, b: Pose, sigmas: np.ndarray) -> float:
    area = 0.5 * (_box_area(a.bbox) + _box_area(b.bbox))
    if area <= 0:
        return 0.0
    ka, kb = a.keypoints, b.keypoints
    valid = (ka[:, 2] > 0.1) & (kb[:, 2] > 0.1)
    if not valid.any():
        return 0.0
    d2 = (ka[:, 0] - kb[:, 0]) ** 2 + (ka[:, 1] - kb[:, 1]) ** 2
    e = d2 / ((2 * sigmas) ** 2 * (area + np.spacing(1)) * 2)
    return float(np.exp(-e[valid]).mean())


def _is_match(a: Pose, b: Pose, *, oks_thr: float, iou_thr: float, sigmas: np.ndarray) -> bool:
    diag = float(np.hypot(*(a.bbox[2:] - a.bbox[:2])))
    if diag > 0 and float(np.linalg.norm(_center(a.bbox) - _center(b.bbox))) > 1.5 * diag:
        return False
    return _oks(a, b, sigmas) >= oks_thr or _iou(a.bbox, b.bbox) >= iou_thr


def merge_views(
    by_view: dict[int, list[Pose]],
    *,
    oks_thr: float,
    iou_thr: float,
    sigmas: np.ndarray,
    max_detections: int,
    dedup_single_view: bool = False,
) -> list[Pose]:
    """``dedup_single_view`` also suppresses duplicates within the whole-frame view
    (the NMS-free head should not need it; it is a live-test convenience)."""

    if len(by_view) <= 1 and not dedup_single_view:
        poses = [p for view in by_view.values() for p in view]
        poses.sort(key=lambda p: p.score, reverse=True)
        return poses[:max_detections]

    accepted: list[Pose] = []
    for view in VIEW_PRIORITY:
        candidates = sorted(by_view.get(view, []), key=lambda p: p.score, reverse=True)
        for det in candidates:
            if any(_is_match(det, kept, oks_thr=oks_thr, iou_thr=iou_thr, sigmas=sigmas) for kept in accepted):
                continue
            accepted.append(det)
    accepted.sort(key=lambda p: p.score, reverse=True)
    return accepted[:max_detections]
