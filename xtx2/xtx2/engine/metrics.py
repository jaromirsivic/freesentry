"""OKS-based keypoint evaluation: AP, AP50, AP75, AR (spec section 9.2).

A pragmatic COCO-style keypoint evaluator. For each image, detections are matched
to ground-truth people greedily by descending score across OKS thresholds
(0.50..0.95 step 0.05). Average Precision is 101-point interpolated over recall;
AP is the mean across thresholds, AP50/AP75 the single-threshold values, and AR
the mean maximum recall across thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.keypoints import OKS_SIGMAS

_OKS_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


@dataclass(slots=True)
class ImagePredictions:
    keypoints: np.ndarray  # (D, K, 3) px (x, y, conf)
    scores: np.ndarray  # (D,)


@dataclass(slots=True)
class ImageGroundTruth:
    keypoints: np.ndarray  # (G, K, 3) px (x, y, vis)
    areas: np.ndarray  # (G,)


def compute_oks(
    gt_kpts: np.ndarray, gt_area: float, dt_kpts: np.ndarray, *, sigmas: np.ndarray | None = None
) -> float:
    """Object Keypoint Similarity between one GT and one detection."""

    sigmas = OKS_SIGMAS if sigmas is None else sigmas
    visible = gt_kpts[:, 2] > 0
    if not visible.any():
        return 0.0
    dx = dt_kpts[:, 0] - gt_kpts[:, 0]
    dy = dt_kpts[:, 1] - gt_kpts[:, 1]
    d2 = dx**2 + dy**2
    var = (2 * sigmas) ** 2
    e = d2 / (var * (gt_area + np.spacing(1)) * 2)
    return float(np.exp(-e[visible]).mean())


def _oks_matrix(gt: ImageGroundTruth, dt: ImagePredictions, sigmas: np.ndarray) -> np.ndarray:
    num_gt = gt.keypoints.shape[0]
    num_dt = dt.keypoints.shape[0]
    matrix = np.zeros((num_dt, num_gt), dtype=np.float64)
    for d in range(num_dt):
        for g in range(num_gt):
            matrix[d, g] = compute_oks(
                gt.keypoints[g], float(gt.areas[g]), dt.keypoints[d], sigmas=sigmas
            )
    return matrix


def _average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.spacing(1))
    # 101-point interpolation (COCO style)
    recall_levels = np.linspace(0, 1, 101)
    interp = np.zeros_like(recall_levels)
    for i, level in enumerate(recall_levels):
        mask = recall >= level
        interp[i] = precision[mask].max() if mask.any() else 0.0
    return float(interp.mean())


def evaluate(
    predictions: list[ImagePredictions],
    ground_truths: list[ImageGroundTruth],
    *,
    sigmas: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AP, AP50, AP75, AR over a dataset."""

    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have equal length")
    sigmas = OKS_SIGMAS if sigmas is None else sigmas

    per_threshold_ap: dict[float, float] = {}
    per_threshold_recall: dict[float, float] = {}

    for thr in _OKS_THRESHOLDS:
        all_scores: list[float] = []
        all_tp: list[float] = []
        all_fp: list[float] = []
        total_gt = 0
        for dt, gt in zip(predictions, ground_truths):
            total_gt += gt.keypoints.shape[0]
            num_dt = dt.keypoints.shape[0]
            if num_dt == 0:
                continue
            order = np.argsort(-dt.scores)
            oks = _oks_matrix(gt, dt, sigmas)
            gt_matched = np.zeros(gt.keypoints.shape[0], dtype=bool)
            for d in order:
                all_scores.append(float(dt.scores[d]))
                if gt.keypoints.shape[0] == 0:
                    all_tp.append(0.0)
                    all_fp.append(1.0)
                    continue
                candidates = np.where((oks[d] >= thr) & (~gt_matched))[0]
                if candidates.size > 0:
                    best = candidates[np.argmax(oks[d, candidates])]
                    gt_matched[best] = True
                    all_tp.append(1.0)
                    all_fp.append(0.0)
                else:
                    all_tp.append(0.0)
                    all_fp.append(1.0)
        if all_scores:
            order = np.argsort(-np.array(all_scores))
            tp = np.array(all_tp)[order]
            fp = np.array(all_fp)[order]
            per_threshold_ap[float(thr)] = _average_precision(tp, fp, total_gt)
            per_threshold_recall[float(thr)] = (tp.sum() / total_gt) if total_gt else float("nan")
        else:
            per_threshold_ap[float(thr)] = 0.0 if total_gt else float("nan")
            per_threshold_recall[float(thr)] = 0.0 if total_gt else float("nan")

    valid_ap = [v for v in per_threshold_ap.values() if not np.isnan(v)]
    valid_ar = [v for v in per_threshold_recall.values() if not np.isnan(v)]
    return {
        "AP": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "AP50": per_threshold_ap.get(0.5, 0.0),
        "AP75": per_threshold_ap.get(0.75, 0.0),
        "AR": float(np.mean(valid_ar)) if valid_ar else 0.0,
    }
