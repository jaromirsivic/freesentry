"""Nine-point OKS evaluation: AP/AR plus the detailed acceptance metrics.

The OKS metric uses the task-specific nine-landmark sigmas from
``xtx3.utils.keypoints`` (COCO's per-landmark sigmas for the retained points --
see that module for the rationale; the 17-point sigma vector is never used).

Two evaluators:

* :func:`evaluate` -- COCO-style keypoint AP, AP50, AP75, AR (thresholds
  0.50..0.95 step 0.05, 101-point interpolated precision), used for per-epoch
  validation and best-checkpoint selection.
* :func:`evaluate_detailed` -- the full acceptance report: person detection
  precision/recall (greedy match at OKS >= 0.5), per-landmark localisation
  quality (mean per-landmark OKS component over matched pairs), per-landmark
  visibility classification accuracy, and recall/quality split by **face scale**
  and **torso scale** buckets computed from the ground-truth landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.keypoints import (
    FACE_INDICES,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    OKS_SIGMAS,
    TORSO_INDICES,
)

_OKS_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
_MATCH_THRESHOLD = 0.5  # operating point for the detailed report

# Scale buckets in original-image pixels (documented heuristics for the
# deployment scenes; "small" faces are near the recognition limit).
FACE_SCALE_EDGES = (16.0, 48.0)  # face spread px: small < 16 <= medium < 48 <= large
TORSO_SCALE_EDGES = (32.0, 96.0)  # shoulder-hip distance px


@dataclass(slots=True)
class ImagePredictions:
    keypoints: np.ndarray  # (D, K, 3) px (x, y, conf)
    scores: np.ndarray  # (D,)
    visibility: np.ndarray | None = None  # (D, K) int in {0,1,2}, optional


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


def per_landmark_oks_components(
    gt_kpts: np.ndarray, gt_area: float, dt_kpts: np.ndarray, *, sigmas: np.ndarray | None = None
) -> np.ndarray:
    """Per-landmark OKS terms ``exp(-d^2 / ...)`` for one matched GT/detection pair.

    Returns ``(K,)`` with NaN for landmarks whose GT visibility is 0.
    """

    sigmas = OKS_SIGMAS if sigmas is None else sigmas
    dx = dt_kpts[:, 0] - gt_kpts[:, 0]
    dy = dt_kpts[:, 1] - gt_kpts[:, 1]
    d2 = dx**2 + dy**2
    var = (2 * sigmas) ** 2
    e = d2 / (var * (gt_area + np.spacing(1)) * 2)
    components = np.exp(-e)
    components[gt_kpts[:, 2] <= 0] = np.nan
    return components


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
    """Compute AP, AP50, AP75, AR over a dataset (nine-point OKS)."""

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


def _landmark_group_scale(kpts: np.ndarray, indices: tuple[int, ...]) -> float:
    """Max pairwise distance between the visible landmarks of a group, in px.

    Returns NaN when fewer than two group landmarks are labelled visible.
    """

    pts = kpts[list(indices)]
    valid = pts[:, 2] > 0
    if valid.sum() < 2:
        return float("nan")
    xy = pts[valid, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    return float(np.sqrt((diff**2).sum(-1)).max())


def _bucket(value: float, edges: tuple[float, float]) -> str | None:
    if np.isnan(value):
        return None
    if value < edges[0]:
        return "small"
    if value < edges[1]:
        return "medium"
    return "large"


def evaluate_detailed(
    predictions: list[ImagePredictions],
    ground_truths: list[ImageGroundTruth],
    *,
    sigmas: np.ndarray | None = None,
    match_threshold: float = _MATCH_THRESHOLD,
) -> dict:
    """The full acceptance report at the OKS >= ``match_threshold`` operating point.

    Detections are greedily matched to GTs by descending score. Reported:

    * ``detection``: precision / recall / f1 / TP / FP / FN counts,
    * ``per_landmark``: mean OKS component + visibility accuracy per landmark,
    * ``face_scale`` / ``torso_scale``: GT count, recall, and mean matched OKS
      per scale bucket (buckets NaN-scale GTs are excluded).
    """

    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have equal length")
    sigmas = OKS_SIGMAS if sigmas is None else sigmas

    tp = fp = 0
    total_gt = 0
    landmark_oks_sum = np.zeros(NUM_KEYPOINTS)
    landmark_oks_count = np.zeros(NUM_KEYPOINTS)
    vis_correct = np.zeros(NUM_KEYPOINTS)
    vis_total = np.zeros(NUM_KEYPOINTS)
    buckets: dict[str, dict[str, dict[str, float]]] = {
        group: {name: {"gt": 0.0, "matched": 0.0, "oks_sum": 0.0} for name in ("small", "medium", "large")}
        for group in ("face_scale", "torso_scale")
    }

    for dt, gt in zip(predictions, ground_truths):
        num_gt = gt.keypoints.shape[0]
        total_gt += num_gt

        face_scales = [_landmark_group_scale(gt.keypoints[g], FACE_INDICES) for g in range(num_gt)]
        torso_scales = [_landmark_group_scale(gt.keypoints[g], TORSO_INDICES) for g in range(num_gt)]
        for g in range(num_gt):
            for group, scale, edges in (
                ("face_scale", face_scales[g], FACE_SCALE_EDGES),
                ("torso_scale", torso_scales[g], TORSO_SCALE_EDGES),
            ):
                name = _bucket(scale, edges)
                if name is not None:
                    buckets[group][name]["gt"] += 1

        if dt.keypoints.shape[0] == 0:
            continue
        oks = _oks_matrix(gt, dt, sigmas)
        gt_matched = np.zeros(num_gt, dtype=bool)
        order = np.argsort(-dt.scores)
        for d in order:
            candidates = (
                np.where((oks[d] >= match_threshold) & (~gt_matched))[0] if num_gt else np.array([], int)
            )
            if candidates.size == 0:
                fp += 1
                continue
            g = int(candidates[np.argmax(oks[d, candidates])])
            gt_matched[g] = True
            tp += 1
            pair_oks = float(oks[d, g])

            components = per_landmark_oks_components(
                gt.keypoints[g], float(gt.areas[g]), dt.keypoints[d], sigmas=sigmas
            )
            valid = ~np.isnan(components)
            landmark_oks_sum[valid] += components[valid]
            landmark_oks_count[valid] += 1

            if dt.visibility is not None:
                gt_vis = np.clip(gt.keypoints[g][:, 2].astype(np.int64), 0, 2)
                pred_vis = np.clip(dt.visibility[d].astype(np.int64), 0, 2)
                vis_correct += (gt_vis == pred_vis).astype(np.float64)
                vis_total += 1

            for group, scale, edges in (
                ("face_scale", face_scales[g], FACE_SCALE_EDGES),
                ("torso_scale", torso_scales[g], TORSO_SCALE_EDGES),
            ):
                name = _bucket(scale, edges)
                if name is not None:
                    buckets[group][name]["matched"] += 1
                    buckets[group][name]["oks_sum"] += pair_oks

    fn = total_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    per_landmark = {}
    for k, name in enumerate(KEYPOINT_NAMES):
        per_landmark[name] = {
            "mean_oks": float(landmark_oks_sum[k] / landmark_oks_count[k]) if landmark_oks_count[k] else float("nan"),
            "matched_points": int(landmark_oks_count[k]),
            "visibility_accuracy": float(vis_correct[k] / vis_total[k]) if vis_total[k] else float("nan"),
        }

    scale_report = {}
    for group, group_buckets in buckets.items():
        scale_report[group] = {}
        for name, stats in group_buckets.items():
            gt_count = stats["gt"]
            matched = stats["matched"]
            scale_report[group][name] = {
                "gt": int(gt_count),
                "recall": float(matched / gt_count) if gt_count else float("nan"),
                "mean_matched_oks": float(stats["oks_sum"] / matched) if matched else float("nan"),
            }

    return {
        "detection": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        },
        "per_landmark": per_landmark,
        "face_scale": scale_report["face_scale"],
        "torso_scale": scale_report["torso_scale"],
    }
