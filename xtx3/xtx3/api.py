"""Public library API: ``BGR numpy image -> nine-landmark pose array``.

``detect_poses`` supports two view modes:

* **whole** (XTX3-u default): one letterboxed network pass per frame.
* **pyramid**: the three centre-focused views run as a single batched forward
  pass, decoded with the NMS-free one-to-one head, mapped back to
  original-image pixels via ``T_k``, and deduplicated with priority
  view 1 > view 2 > view 3.

No NMS code path exists anywhere in this module.

Output contract (per pose):

* ``bbox_xyxy`` -- person box in original-image pixels.
* ``score`` -- person confidence in [0, 1].
* ``keypoints`` -- ``(9, 3)`` array of original-image ``x``, ``y`` and keypoint
  confidence, in the fixed order of ``xtx3.utils.keypoints.KEYPOINT_NAMES``.
  Keypoint confidence is P(occluded) + P(visible); occluded and out-of-image
  points still receive coordinate estimates but carry the visibility class.
* ``keypoints_norm`` -- same with x/y divided by image width/height.
* ``visibility`` -- ``(9,)`` predicted class per landmark: 0 absent/outside,
  1 occluded, 2 visible.
* ``source_view`` -- 0 for whole-frame; 1|2|3 for the pyramid view that
  produced the surviving detection.

The flat array is exactly 32 values: ``[x1, y1, x2, y2, person_score,
9 * (x, y, keypoint_confidence)]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.preprocess import preprocess_image, view_keys_for_mode
from .engine.merge import Detection, MergeParams, merge_views
from .models import XTX3Model, build_model
from .utils import geometry
from .utils.keypoints import NUM_KEYPOINTS
from .utils.logging import get_logger

logger = get_logger(__name__)

POSE_ARRAY_SIZE = 5 + 3 * NUM_KEYPOINTS  # 32

# (mode, network_size) used when a checkpoint/config does not specify them.
DEFAULT_INFERENCE: dict[str, tuple[str, int]] = {
    "u": ("whole", 320),
    "n": ("pyramid", 384),
    "m": ("pyramid", 384),
    "l": ("pyramid", 384),
}


@dataclass(slots=True)
class Pose:
    """A single detected person."""

    bbox_xyxy: np.ndarray  # (4,) float32, person bbox in ORIGINAL image pixels
    score: float  # detection confidence in [0, 1]
    keypoints: np.ndarray  # (9, 3): [x_pixel, y_pixel, confidence]
    keypoints_norm: np.ndarray  # (9, 3): [x_norm, y_norm, confidence]
    visibility: np.ndarray  # (9,) int in {0,1,2} predicted visibility class
    source_view: int  # 0 whole-frame | 1|2|3 pyramid view

    def to_array(self) -> np.ndarray:
        """Flatten to ``(32,)`` = ``[x1,y1,x2,y2,score, 9*(x,y,conf)]``."""

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
            "source_view": int(self.source_view),
        }


def poses_to_array(poses: list[Pose]) -> np.ndarray:
    """Stack poses into a single ``(N, 32)`` array."""

    if not poses:
        return np.zeros((0, POSE_ARRAY_SIZE), dtype=np.float32)
    return np.stack([p.to_array() for p in poses], axis=0)


def _inference_settings(config: dict[str, Any], variant: str) -> tuple[str, int]:
    default_mode, default_size = DEFAULT_INFERENCE.get(variant, ("pyramid", 384))
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    variant_cfg = inference_cfg.get(variant, {}) if isinstance(inference_cfg, dict) else {}
    mode = str(variant_cfg.get("mode", default_mode))
    size = int(variant_cfg.get("size", default_size))
    return mode, size


def load_model(
    weights: str | Path | None = None,
    *,
    variant: str = "n",
    device: str = "cpu",
    fuse: bool = True,
    mode: str | None = None,
    network_size: int | None = None,
) -> XTX3Model:
    """Load a trained XTX3 model ready for inference.

    Uses EMA weights when present. Fuses by default (Conv+BN fold, RepConv
    reparam, one-to-many head dropped) so inference is NMS-free via the
    one-to-one head only. The inference view mode and input size come from the
    checkpoint config's ``inference`` section (overridable via arguments) and
    are attached to the model together with merge params for
    :func:`detect_poses`.
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
    cfg_mode, cfg_size = _inference_settings(config, variant)
    model.xtx3_config = config  # type: ignore[attr-defined]
    model.merge_params = merge_params  # type: ignore[attr-defined]
    model.device_str = device  # type: ignore[attr-defined]
    model.view_mode = mode or cfg_mode  # type: ignore[attr-defined]
    model.network_size = int(network_size or cfg_size)  # type: ignore[attr-defined]
    return model


def _views_to_batch(views: dict[int, np.ndarray], *, keys: tuple[int, ...], device: str) -> torch.Tensor:
    ordered = [views[key] for key in keys]
    stacked = np.stack(ordered, axis=0).astype(np.float32) / 255.0  # (V,S,S,3) BGR
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def _collect_view_detections(
    det: np.ndarray, vclass: np.ndarray, transform: np.ndarray, view: int
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
            source_view=view,
        )
        for i in range(det.shape[0])
    ]


def _detections_per_view(
    model: XTX3Model,
    image: np.ndarray,
    *,
    mode: str,
    network_size: int,
    conf_threshold: float,
    max_detections: int,
    device: str,
) -> tuple[dict[int, list[Detection]], tuple[int, int]]:
    """One batched forward over all views; detections mapped to original px."""

    keys = view_keys_for_mode(mode)
    result = preprocess_image(image, mode=mode, network_size=network_size)
    batch = _views_to_batch(result.views, keys=keys, device=device)
    dets, vis = model.predict(batch, conf_threshold=conf_threshold, max_detections=max_detections)

    by_view: dict[int, list[Detection]] = {key: [] for key in keys}
    for idx, view in enumerate(keys):
        by_view[view] = _collect_view_detections(
            dets[idx].cpu().numpy(), vis[idx].cpu().numpy(), result.transforms[view], view
        )
    return by_view, result.orig_size


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
        source_view=det.source_view,
    )


def detect_poses(
    image: np.ndarray,
    *,
    model: XTX3Model,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
    mode: str | None = None,
    network_size: int | None = None,
) -> list[Pose]:
    """Detect head-and-torso poses in a single BGR ``uint8`` image.

    Returns a list of :class:`Pose` in original-image coordinates. In pyramid
    mode detections are deduplicated across views with priority
    view 1 > view 2 > view 3.
    """

    if model is None:
        raise ValueError("model is required; build it with load_model(...)")
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    view_mode = mode or getattr(model, "view_mode", "pyramid")
    size = int(network_size or getattr(model, "network_size", 384))
    conf = min(conf_threshold, params.conf_threshold)

    by_view, orig_size = _detections_per_view(
        model,
        image,
        mode=view_mode,
        network_size=size,
        conf_threshold=conf,
        max_detections=max_detections,
        device=device,
    )
    merge_params = MergeParams.from_dict(params.to_dict())
    merge_params.conf_threshold = conf_threshold
    merge_params.max_detections = max_detections
    merged = merge_views(by_view, params=merge_params)
    return [
        _detection_to_pose(det, orig_size=orig_size, return_normalized=return_normalized)
        for det in merged
    ]


def detect_poses_batch(
    images: list[np.ndarray],
    *,
    model: XTX3Model,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
    return_normalized: bool = True,
    mode: str | None = None,
    network_size: int | None = None,
) -> list[list[Pose]]:
    """Batch variant: one forward over ``num_views * len(images)`` views."""

    if not images:
        return []
    device = getattr(model, "device_str", "cpu")
    params: MergeParams = getattr(model, "merge_params", MergeParams())
    view_mode = mode or getattr(model, "view_mode", "pyramid")
    size = int(network_size or getattr(model, "network_size", 384))
    keys = view_keys_for_mode(view_mode)
    conf = min(conf_threshold, params.conf_threshold)

    results = [preprocess_image(img, mode=view_mode, network_size=size) for img in images]
    batch = torch.cat(
        [_views_to_batch(r.views, keys=keys, device=device) for r in results], dim=0
    )
    dets, vis = model.predict(batch, conf_threshold=conf, max_detections=max_detections)

    outputs: list[list[Pose]] = []
    for img_idx, result in enumerate(results):
        by_view: dict[int, list[Detection]] = {key: [] for key in keys}
        for c, view in enumerate(keys):
            flat_idx = img_idx * len(keys) + c
            by_view[view] = _collect_view_detections(
                dets[flat_idx].cpu().numpy(),
                vis[flat_idx].cpu().numpy(),
                result.transforms[view],
                view,
            )
        merge_params = MergeParams.from_dict(params.to_dict())
        merge_params.conf_threshold = conf_threshold
        merge_params.max_detections = max_detections
        merged = merge_views(by_view, params=merge_params)
        outputs.append(
            [
                _detection_to_pose(det, orig_size=result.orig_size, return_normalized=return_normalized)
                for det in merged
            ]
        )
    return outputs
