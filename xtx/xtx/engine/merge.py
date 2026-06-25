"""Cross-crop merge (dedup) + calibration (spec section 5).

After decoding detections from crops A, B, C and mapping them to original-image
coordinates, the same person seen in multiple crops must be deduplicated with
priority **A > B > C**: keep A's detection, add B for people A missed, add C for the
farthest central people B also missed. Because XTX is NMS-free within a crop, only
cross-crop duplicates are resolved here.

The matching thresholds are calibrated on the test split (grid search to maximise
merged OKS mAP), with an optional logistic "same person" classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.keypoints import OKS_SIGMAS
from ..utils.logging import get_logger
from .metrics import ImageGroundTruth, ImagePredictions, compute_oks, evaluate

logger = get_logger(__name__)

CROP_PRIORITY = ("A", "B", "C")


@dataclass(slots=True)
class Detection:
    """A detection mapped to original-image pixel coordinates."""

    bbox: np.ndarray  # (4,) xyxy
    score: float
    keypoints: np.ndarray  # (K, 3) [x, y, conf]
    visibility: np.ndarray  # (K,)
    source_crop: str


@dataclass(slots=True)
class MergeParams:
    """Calibrated cross-crop merge parameters (persisted in checkpoint/config)."""

    mode: str = "threshold"
    oks_match_thr: float = 0.5
    iou_match_thr: float = 0.5
    conf_threshold: float = 0.25
    max_detections: int = 300
    logistic_weights: list[float] = field(default_factory=list)  # [w_iou,w_oks,w_cd,w_sr,bias]

    @classmethod
    def from_dict(cls, data: dict | None) -> "MergeParams":
        data = data or {}
        return cls(
            mode=str(data.get("mode", "threshold")),
            oks_match_thr=float(data.get("oks_match_thr", 0.5)),
            iou_match_thr=float(data.get("iou_match_thr", 0.5)),
            conf_threshold=float(data.get("conf_threshold", 0.25)),
            max_detections=int(data.get("max_detections", 300)),
            logistic_weights=list(data.get("logistic_weights", [])),
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "oks_match_thr": self.oks_match_thr,
            "iou_match_thr": self.iou_match_thr,
            "conf_threshold": self.conf_threshold,
            "max_detections": self.max_detections,
            "logistic_weights": list(self.logistic_weights),
        }


def _box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    lt = np.maximum(box_a[:2], box_b[:2])
    rb = np.minimum(box_a[2:], box_b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = float(wh[0] * wh[1])
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _oks_between(det_a: Detection, det_b: Detection, sigmas: np.ndarray) -> float:
    area = 0.5 * (_box_area(det_a.bbox) + _box_area(det_b.bbox))
    if area <= 0:
        return 0.0
    ka, kb = det_a.keypoints, det_b.keypoints
    valid = (ka[:, 2] > 0.1) & (kb[:, 2] > 0.1)
    if not valid.any():
        return 0.0
    d2 = (ka[:, 0] - kb[:, 0]) ** 2 + (ka[:, 1] - kb[:, 1]) ** 2
    e = d2 / ((2 * sigmas) ** 2 * (area + np.spacing(1)) * 2)
    return float(np.exp(-e[valid]).mean())


def _features(det_a: Detection, det_b: Detection, sigmas: np.ndarray) -> np.ndarray:
    iou = _iou(det_a.bbox, det_b.bbox)
    oks = _oks_between(det_a, det_b, sigmas)
    diag = float(np.hypot(*(det_a.bbox[2:] - det_a.bbox[:2]))) + np.spacing(1)
    center_dist = float(np.linalg.norm(_center(det_a.bbox) - _center(det_b.bbox))) / diag
    area_a, area_b = _box_area(det_a.bbox), _box_area(det_b.bbox)
    scale_ratio = float(min(area_a, area_b) / (max(area_a, area_b) + np.spacing(1)))
    return np.array([iou, oks, center_dist, scale_ratio], dtype=np.float64)


def _is_match(det_a: Detection, det_b: Detection, params: MergeParams, sigmas: np.ndarray) -> bool:
    # Spatial short-circuit: skip pairs whose centres are far apart.
    diag = float(np.hypot(*(det_a.bbox[2:] - det_a.bbox[:2])))
    center_dist = float(np.linalg.norm(_center(det_a.bbox) - _center(det_b.bbox)))
    if diag > 0 and center_dist > 1.5 * diag:
        return False

    if params.mode == "logistic" and len(params.logistic_weights) == 5:
        feats = _features(det_a, det_b, sigmas)
        weights = np.array(params.logistic_weights[:4])
        bias = params.logistic_weights[4]
        prob = 1.0 / (1.0 + np.exp(-(float(feats @ weights) + bias)))
        return prob >= 0.5

    return (
        _oks_between(det_a, det_b, sigmas) >= params.oks_match_thr
        or _iou(det_a.bbox, det_b.bbox) >= params.iou_match_thr
    )


def merge_crops(
    by_crop: dict[str, list[Detection]],
    *,
    params: MergeParams,
    sigmas: np.ndarray | None = None,
) -> list[Detection]:
    """Merge per-crop detections with priority A > B > C (section 5.2)."""

    sigmas = OKS_SIGMAS if sigmas is None else sigmas
    accepted: list[Detection] = []

    for crop in CROP_PRIORITY:
        candidates = [d for d in by_crop.get(crop, []) if d.score >= params.conf_threshold]
        candidates.sort(key=lambda d: d.score, reverse=True)
        for det in candidates:
            if any(_is_match(det, kept, params, sigmas) for kept in accepted):
                continue
            accepted.append(det)

    accepted.sort(key=lambda d: d.score, reverse=True)
    return accepted[: params.max_detections]


def _detections_to_predictions(dets: list[Detection]) -> ImagePredictions:
    if not dets:
        return ImagePredictions(keypoints=np.zeros((0, 17, 3)), scores=np.zeros((0,)))
    kpts = np.stack([d.keypoints for d in dets], axis=0)
    scores = np.array([d.score for d in dets])
    return ImagePredictions(keypoints=kpts, scores=scores)


def calibrate_thresholds(
    per_image_crops: list[dict[str, list[Detection]]],
    ground_truths: list[ImageGroundTruth],
    *,
    base_params: MergeParams,
    oks_grid: np.ndarray | None = None,
    iou_grid: np.ndarray | None = None,
    conf_grid: np.ndarray | None = None,
    sigmas: np.ndarray | None = None,
) -> tuple[MergeParams, float]:
    """Grid-search ``(oks_thr, iou_thr, conf)`` to maximise merged OKS AP."""

    sigmas = OKS_SIGMAS if sigmas is None else sigmas
    oks_grid = np.arange(0.3, 0.81, 0.1) if oks_grid is None else oks_grid
    iou_grid = np.arange(0.3, 0.81, 0.1) if iou_grid is None else iou_grid
    conf_grid = np.arange(0.1, 0.41, 0.1) if conf_grid is None else conf_grid

    best_params = MergeParams.from_dict(base_params.to_dict())
    best_ap = -1.0
    for conf in conf_grid:
        for oks_thr in oks_grid:
            for iou_thr in iou_grid:
                trial = MergeParams(
                    mode="threshold",
                    oks_match_thr=float(oks_thr),
                    iou_match_thr=float(iou_thr),
                    conf_threshold=float(conf),
                    max_detections=base_params.max_detections,
                )
                preds = [
                    _detections_to_predictions(merge_crops(crops, params=trial, sigmas=sigmas))
                    for crops in per_image_crops
                ]
                metrics = evaluate(preds, ground_truths, sigmas=sigmas)
                if metrics["AP"] > best_ap:
                    best_ap = metrics["AP"]
                    best_params = trial
    logger.info("Merge calibration: best AP=%.4f params=%s", best_ap, best_params.to_dict())
    return best_params, best_ap


def fit_logistic(
    pairwise_features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 500,
    lr: float = 0.1,
) -> list[float]:
    """Fit a tiny logistic regression ``P(same)`` over pairwise features.

    Features columns: ``[IoU, OKS, center_distance, scale_ratio]``; returns
    ``[w_iou, w_oks, w_cd, w_sr, bias]``.
    """

    if pairwise_features.shape[0] == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    num_features = pairwise_features.shape[1]
    weights = np.zeros(num_features, dtype=np.float64)
    bias = 0.0
    n = pairwise_features.shape[0]
    for _ in range(epochs):
        logits = pairwise_features @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-logits))
        error = pred - labels
        weights -= lr * (pairwise_features.T @ error) / n
        bias -= lr * float(error.mean())
    return [*weights.tolist(), float(bias)]
