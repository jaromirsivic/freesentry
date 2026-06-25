"""Public library API (spec section 2): ``numpy image -> pose array``.

``detect_poses`` runs the full A/B/C cascade in a single forward pass, decodes each
crop with the NMS-free one-to-one head, maps every detection back to original-image
pixels, and merges across crops with priority A > B > C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.preprocess import CROP_KEYS, preprocess_image
from .engine.merge import Detection, MergeParams, merge_crops
from .models import XTXModel, build_model
from .utils import geometry
from .utils.keypoints import NUM_KEYPOINTS
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Pose:
    """A single detected person (spec section 2.2)."""

    bbox_xyxy: np.ndarray  # (4,) float32, original-image pixels
    score: float
    keypoints: np.ndarray  # (17, 3): [x_pixel, y_pixel, confidence]
    keypoints_norm: np.ndarray  # (17, 3): [x_norm, y_norm, confidence]
    visibility: np.ndarray  # (17,) int in {0,1,2}
    source_crop: str  # "A" | "B" | "C"

    def to_array(self) -> np.ndarray:
        """Flatten to ``(56,)`` = ``[x1,y1,x2,y2,score, 17*(x,y,conf)]`` (section 2.2)."""

        return np.concatenate(
            [self.bbox_xyxy.astype(np.float32), np.array([self.score], np.float32), self.keypoints.reshape(-1).astype(np.float32)]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox_xyxy.tolist(),
            "score": float(self.score),
            "keypoints": self.keypoints.tolist(),
            "keypoints_norm": self.keypoints_norm.tolist(),
            "visibility": self.visibility.tolist(),
            "source_crop": self.source_crop,
        }


def load_model(
    weights: str | Path | None = None,
    *,
    variant: str = "n",
    device: str = "cpu",
    fuse: bool = True,
) -> XTXModel:
    """Load a trained XTX model ready for inference.

    Uses EMA weights when present. Fuses by default (Conv+BN fold, RepConv reparam,
    one-to-many head dropped) so inference is NMS-free via the one-to-one head only.
    Merge params and config are attached as attributes for :func:`detect_poses`.
    """

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"load_model supports device 'cpu'|'cuda', got {device!r} (use export for ncnn)")

    config: dict[str, Any] = {}
    merge_params = MergeParams()
    if weights is not None:
        from .engine.checkpoint import load_checkpoint

        ckpt = load_checkpoint(Path(weights), map_location=device)
        variant = ckpt.get("variant", variant)
        config = ckpt.get("config", {}) or {}
        model = build_model(variant=variant, config=config)
        state = ckpt.get("ema_state") or ckpt.get("model_state")
        if state is None:
            raise ValueError("Checkpoint has neither ema_state nor model_state")
        model.load_state_dict(state, strict=True)
        merge_params = MergeParams.from_dict(ckpt.get("merge_params") or config.get("merge"))
    else:
        model = build_model(variant=variant, config=config)
        logger.warning("load_model called without weights; using randomly initialised model")

    model.to(device).eval()
    if fuse:
        model.fuse()
    model.xtx_config = config  # type: ignore[attr-defined]
    model.merge_params = merge_params  # type: ignore[attr-defined]
    model.device_str = device  # type: ignore[attr-defined]
    return model


def _crops_to_batch(crops: dict[str, np.ndarray], *, device: str) -> torch.Tensor:
    ordered = [crops[key] for key in CROP_KEYS]
    stacked = np.stack(ordered, axis=0).astype(np.float32) / 255.0  # (3,384,384,3) BGR
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def _detections_per_crop(
    model: XTXModel,
    image: np.ndarray,
    *,
    conf_threshold: float,
    max_detections: int,
    device: str,
) -> tuple[dict[str, list[Detection]], tuple[int, int]]:
    """Run one forward pass over A/B/C and return detections mapped to original px."""

    result = preprocess_image(image)
    batch = _crops_to_batch(result.crops, device=device)
    dets, vis = model.predict(batch, conf_threshold=conf_threshold, max_detections=max_detections)

    by_crop: dict[str, list[Detection]] = {key: [] for key in CROP_KEYS}
    for crop_idx, crop in enumerate(CROP_KEYS):
        transform = result.transforms[crop]
        det = dets[crop_idx].cpu().numpy()
        vclass = vis[crop_idx].cpu().numpy()
        if det.shape[0] == 0:
            continue
        boxes = geometry.apply_to_boxes_xyxy(transform, det[:, :4])
        kpts = det[:, 5:].reshape(det.shape[0], NUM_KEYPOINTS, 3)
        kpts_mapped = geometry.apply_to_keypoints(transform, kpts.copy())
        for i in range(det.shape[0]):
            by_crop[crop].append(
                Detection(
                    bbox=boxes[i].astype(np.float32),
                    score=float(det[i, 4]),
                    keypoints=kpts_mapped[i].astype(np.float32),
                    visibility=vclass[i].astype(np.int64),
                    source_crop=crop,
                )
            )
    return by_crop, result.orig_size


def _detection_to_pose(det: Detection, *, orig_size: tuple[int, int], return_normalized: bool) -> Pose:
    width, height = orig_size
    kpts_norm = det.keypoints.copy()
    if return_normalized:
        kpts_norm[:, 0] = det.keypoints[:, 0] / max(width, 1)
        kpts_norm[:, 1] = det.keypoints[:, 1] / max(height, 1)
    return Pose(
        bbox_xyxy=det.bbox.astype(np.float32),
        score=det.score,
        keypoints=det.keypoints.astype(np.float32),
        keypoints_norm=kpts_norm.astype(np.float32),
        visibility=det.visibility.astype(np.int64),
        source_crop=det.source_crop,
    )


def detect_poses(
    image: np.ndarray,
    *,
    model: XTXModel,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
) -> list[Pose]:
    """Detect poses in a single BGR ``uint8`` image (spec section 2.1)."""

    if model is None:
        raise ValueError("model is required; build it with load_model(...)")
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    conf = min(conf_threshold, params.conf_threshold)

    by_crop, orig_size = _detections_per_crop(
        model, image, conf_threshold=conf, max_detections=max_detections, device=device
    )
    merge_params = MergeParams.from_dict(params.to_dict())
    merge_params.conf_threshold = conf_threshold
    merge_params.max_detections = max_detections
    merged = merge_crops(by_crop, params=merge_params)
    return [
        _detection_to_pose(det, orig_size=orig_size, return_normalized=return_normalized)
        for det in merged
    ]


def detect_poses_batch(
    images: list[np.ndarray],
    *,
    model: XTXModel,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
) -> list[list[Pose]]:
    """Batch variant: efficient across images and across the 3 crops (section 2.3)."""

    if not images:
        return []
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    conf = min(conf_threshold, params.conf_threshold)

    results = [preprocess_image(img) for img in images]
    batch = torch.cat([_crops_to_batch(r.crops, device=device) for r in results], dim=0)
    dets, vis = model.predict(batch, conf_threshold=conf, max_detections=max_detections)

    outputs: list[list[Pose]] = []
    for img_idx, result in enumerate(results):
        by_crop: dict[str, list[Detection]] = {key: [] for key in CROP_KEYS}
        for c, crop in enumerate(CROP_KEYS):
            flat_idx = img_idx * len(CROP_KEYS) + c
            det = dets[flat_idx].cpu().numpy()
            vclass = vis[flat_idx].cpu().numpy()
            if det.shape[0] == 0:
                continue
            transform = result.transforms[crop]
            boxes = geometry.apply_to_boxes_xyxy(transform, det[:, :4])
            kpts = det[:, 5:].reshape(det.shape[0], NUM_KEYPOINTS, 3)
            kpts_mapped = geometry.apply_to_keypoints(transform, kpts.copy())
            for i in range(det.shape[0]):
                by_crop[crop].append(
                    Detection(
                        bbox=boxes[i].astype(np.float32),
                        score=float(det[i, 4]),
                        keypoints=kpts_mapped[i].astype(np.float32),
                        visibility=vclass[i].astype(np.int64),
                        source_crop=crop,
                    )
                )
        merge_params = MergeParams.from_dict(params.to_dict())
        merge_params.conf_threshold = conf_threshold
        merge_params.max_detections = max_detections
        merged = merge_crops(by_crop, params=merge_params)
        outputs.append(
            [
                _detection_to_pose(det, orig_size=result.orig_size, return_normalized=return_normalized)
                for det in merged
            ]
        )
    return outputs
