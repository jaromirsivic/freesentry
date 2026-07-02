"""Public library API (spec section 2): ``numpy image -> pose array``.

``detect_poses`` runs the full 3-level pyramid in a single batched forward pass
(``(3, 3, 384, 384)``), decodes each level with the NMS-free one-to-one head,
maps every detection back to original-image pixels via ``T_k``, and merges
across levels with priority level_1 > level_2 > level_3. No NMS code path
exists anywhere in this module (acceptance criterion 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.preprocess import LEVEL_KEYS, preprocess_image
from .engine.merge import Detection, MergeParams, merge_levels
from .models import XTX2Model, build_model
from .utils import geometry
from .utils.keypoints import NUM_KEYPOINTS
from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Pose:
    """A single detected person (spec section 2.2)."""

    bbox_xyxy: np.ndarray  # (4,) float32, person bbox in ORIGINAL image pixels
    score: float  # detection confidence in [0, 1]
    keypoints: np.ndarray  # (17, 3): [x_pixel, y_pixel, confidence]
    keypoints_norm: np.ndarray  # (17, 3): [x_norm, y_norm, confidence]
    visibility: np.ndarray  # (17,) int in {0,1,2} predicted visibility class
    source_level: int  # 1 | 2 | 3, pyramid level of the surviving detection

    def to_array(self) -> np.ndarray:
        """Flatten to ``(56,)`` = ``[x1,y1,x2,y2,score, 17*(x,y,conf)]`` (section 2.2)."""

        return np.concatenate(
            [
                self.bbox_xyxy.astype(np.float32),
                np.array([self.score], np.float32),
                self.keypoints.reshape(-1).astype(np.float32),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox_xyxy.tolist(),
            "score": float(self.score),
            "keypoints": self.keypoints.tolist(),
            "keypoints_norm": self.keypoints_norm.tolist(),
            "visibility": self.visibility.tolist(),
            "source_level": int(self.source_level),
        }


def poses_to_array(poses: list[Pose]) -> np.ndarray:
    """Stack poses into a single ``(N, 56)`` array (section 2.2 convenience)."""

    if not poses:
        return np.zeros((0, 56), dtype=np.float32)
    return np.stack([p.to_array() for p in poses], axis=0)


def load_model(
    weights: str | Path | None = None,
    *,
    variant: str = "n",
    device: str = "cpu",
    fuse: bool = True,
) -> XTX2Model:
    """Load a trained XTX2 model ready for inference.

    Uses EMA weights when present. Fuses by default (Conv+BN fold, RepConv
    reparam, one-to-many head dropped) so inference is NMS-free via the
    one-to-one head only. Merge params and config are attached as attributes
    for :func:`detect_poses`.
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
    model.xtx2_config = config  # type: ignore[attr-defined]
    model.merge_params = merge_params  # type: ignore[attr-defined]
    model.device_str = device  # type: ignore[attr-defined]
    return model


def _levels_to_batch(levels: dict[int, np.ndarray], *, device: str) -> torch.Tensor:
    ordered = [levels[key] for key in LEVEL_KEYS]
    stacked = np.stack(ordered, axis=0).astype(np.float32) / 255.0  # (3,384,384,3) BGR
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def _collect_level_detections(
    det: np.ndarray, vclass: np.ndarray, transform: np.ndarray, level: int
) -> list[Detection]:
    if det.shape[0] == 0:
        return []
    boxes = geometry.apply_to_boxes_xyxy(transform, det[:, :4])
    kpts = det[:, 5:].reshape(det.shape[0], NUM_KEYPOINTS, 3)
    kpts_mapped = geometry.apply_to_keypoints(transform, kpts.copy())
    return [
        Detection(
            bbox=boxes[i].astype(np.float32),
            score=float(det[i, 4]),
            keypoints=kpts_mapped[i].astype(np.float32),
            visibility=vclass[i].astype(np.int64),
            source_level=level,
        )
        for i in range(det.shape[0])
    ]


def _detections_per_level(
    model: XTX2Model,
    image: np.ndarray,
    *,
    conf_threshold: float,
    max_detections: int,
    device: str,
) -> tuple[dict[int, list[Detection]], tuple[int, int]]:
    """One batched forward over levels 1/2/3; detections mapped to original px."""

    result = preprocess_image(image)
    batch = _levels_to_batch(result.levels, device=device)
    dets, vis = model.predict(batch, conf_threshold=conf_threshold, max_detections=max_detections)

    by_level: dict[int, list[Detection]] = {key: [] for key in LEVEL_KEYS}
    for idx, level in enumerate(LEVEL_KEYS):
        by_level[level] = _collect_level_detections(
            dets[idx].cpu().numpy(), vis[idx].cpu().numpy(), result.transforms[level], level
        )
    return by_level, result.orig_size


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
        source_level=det.source_level,
    )


def detect_poses(
    image: np.ndarray,
    *,
    model: XTX2Model,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
) -> list[Pose]:
    """Detect poses in a single BGR ``uint8`` image (spec section 2.1).

    Returns a list of :class:`Pose` in original-image coordinates, deduplicated
    across levels with priority level_1 > level_2 > level_3.
    """

    if model is None:
        raise ValueError("model is required; build it with load_model(...)")
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    conf = min(conf_threshold, params.conf_threshold)

    by_level, orig_size = _detections_per_level(
        model, image, conf_threshold=conf, max_detections=max_detections, device=device
    )
    merge_params = MergeParams.from_dict(params.to_dict())
    merge_params.conf_threshold = conf_threshold
    merge_params.max_detections = max_detections
    merged = merge_levels(by_level, params=merge_params)
    return [
        _detection_to_pose(det, orig_size=orig_size, return_normalized=return_normalized)
        for det in merged
    ]


def detect_poses_batch(
    images: list[np.ndarray],
    *,
    model: XTX2Model,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
) -> list[list[Pose]]:
    """Batch variant (section 2.3): one forward over ``3 * len(images)`` levels."""

    if not images:
        return []
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    conf = min(conf_threshold, params.conf_threshold)

    results = [preprocess_image(img) for img in images]
    batch = torch.cat([_levels_to_batch(r.levels, device=device) for r in results], dim=0)
    dets, vis = model.predict(batch, conf_threshold=conf, max_detections=max_detections)

    outputs: list[list[Pose]] = []
    for img_idx, result in enumerate(results):
        by_level: dict[int, list[Detection]] = {key: [] for key in LEVEL_KEYS}
        for c, level in enumerate(LEVEL_KEYS):
            flat_idx = img_idx * len(LEVEL_KEYS) + c
            by_level[level] = _collect_level_detections(
                dets[flat_idx].cpu().numpy(),
                vis[flat_idx].cpu().numpy(),
                result.transforms[level],
                level,
            )
        merge_params = MergeParams.from_dict(params.to_dict())
        merge_params.conf_threshold = conf_threshold
        merge_params.max_detections = max_detections
        merged = merge_levels(by_level, params=merge_params)
        outputs.append(
            [
                _detection_to_pose(det, orig_size=result.orig_size, return_normalized=return_normalized)
                for det in merged
            ]
        )
    return outputs
