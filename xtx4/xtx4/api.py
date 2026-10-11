"""Public library API: ``BGR numpy image -> nine-landmark poses``.

One whole-frame letterboxed pass per image (``xtx4.data.preprocess``, mode
``"whole"``), decoded by the NMS-free one-to-one head, mapped back to original
image pixels with the letterbox affine. An optional OKS-NMS (numpy) exists as a
safety net for duplicate detections; it is off by default (``decode.oks_nms``).

Output contract (per :class:`Pose`):

* ``score`` -- person confidence in [0, 1].
* ``keypoints`` -- ``(9, 3)`` original-image ``x``, ``y`` and keypoint confidence
  (``present_prob * exp(-sigma_conf_coeff * sigma)``), order of
  ``xtx4.utils.keypoints.KEYPOINT_NAMES``.
* ``keypoints_norm`` -- same with x/y divided by image width/height.
* ``visibility`` -- ``(9,)`` int class: 0 absent/outside, 1 occluded, 2 visible.
* ``sigma_px`` -- ``(9,)`` RLE scale per landmark in original-image pixels.

``poses_to_array`` flattens to ``(D, 28) = [score, 9 * (x, y, conf)]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data.preprocess import preprocess_image
from .models import XTX4Model, build_model
from .models.head_types import decode_detections
from .runtime.decode import DEFAULT_NMS_SCALE_FACTOR, oks_nms_indices
from .utils import geometry
from .utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS
from .utils.logging import get_logger

logger = get_logger(__name__)

POSE_ARRAY_SIZE = 1 + 3 * NUM_KEYPOINTS  # 28
_WHOLE_VIEW = 0

DEFAULT_DECODE: dict[str, Any] = {
    "conf_threshold": 0.25,
    "max_detections": 100,
    "sigma_conf_coeff": 0.0,
    "sigma_floor": 0.05,
    "oks_nms": False,
    "oks_nms_threshold": 0.5,
    "oks_nms_iou_threshold": None,  # null = OKS only; else also suppress on expanded-box IoU >= value
    "oks_nms_scale_factor": DEFAULT_NMS_SCALE_FACTOR,  # extent-box expansion for the NMS OKS scale
    "present_threshold": 0.5,  # visibility class: 0 if present_prob < present_threshold
    "visible_threshold": 0.5,  # else 2 if visible_prob >= visible_threshold else 1
}


@dataclass(slots=True)
class Pose:
    """A single detected person in original-image pixels."""

    score: float
    keypoints: np.ndarray  # (9, 3): [x_px, y_px, confidence]
    keypoints_norm: np.ndarray  # (9, 3): [x_norm, y_norm, confidence]
    visibility: np.ndarray  # (9,) int in {0, 1, 2}
    sigma_px: np.ndarray  # (9,) float32 RLE scale in original-image pixels

    def to_array(self) -> np.ndarray:
        """Flatten to ``(28,)`` = ``[score, 9 * (x, y, conf)]``."""

        return np.concatenate(
            [np.array([self.score], np.float32), self.keypoints.reshape(-1).astype(np.float32)]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "keypoints": self.keypoints.tolist(),
            "keypoints_norm": self.keypoints_norm.tolist(),
            "visibility": self.visibility.tolist(),
            "sigma_px": self.sigma_px.tolist(),
        }


def poses_to_array(poses: Sequence[Pose]) -> np.ndarray:
    """Stack poses into a single ``(D, 28)`` float32 array."""

    if not poses:
        return np.zeros((0, POSE_ARRAY_SIZE), dtype=np.float32)
    return np.stack([p.to_array() for p in poses], axis=0)


# --------------------------------------------------------------------------- loading


def decode_params_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(DEFAULT_DECODE)
    decode_cfg = (config or {}).get("decode", {}) if isinstance(config, dict) else {}
    for key in params:
        if key in decode_cfg:
            params[key] = decode_cfg[key]
    params["conf_threshold"] = float(params["conf_threshold"])
    params["max_detections"] = int(params["max_detections"])
    params["sigma_conf_coeff"] = float(params["sigma_conf_coeff"])
    params["sigma_floor"] = float(params["sigma_floor"])
    params["oks_nms"] = bool(params["oks_nms"])
    params["oks_nms_threshold"] = float(params["oks_nms_threshold"])
    iou = params["oks_nms_iou_threshold"]
    params["oks_nms_iou_threshold"] = None if iou is None else float(iou)
    params["oks_nms_scale_factor"] = float(params["oks_nms_scale_factor"])
    params["present_threshold"] = float(params["present_threshold"])
    params["visible_threshold"] = float(params["visible_threshold"])
    return params


def attach_inference_attributes(
    model: XTX4Model, *, config: dict[str, Any] | None, device: str, network_size: int | None = None
) -> XTX4Model:
    """Attach ``xtx4_config``, ``network_size``, ``decode`` and ``device_str`` used by ``detect_poses``."""

    config = config or {}
    size = network_size
    if size is None:
        size = int(config.get("data", {}).get("eval_size", 0) or 0) or int(model.spec.default_size)
    model.xtx4_config = config  # type: ignore[attr-defined]
    model.network_size = int(size)  # type: ignore[attr-defined]
    model.decode = decode_params_from_config(config)  # type: ignore[attr-defined]
    model.device_str = device  # type: ignore[attr-defined]
    return model


def load_model(
    weights: str | Path | None = None,
    *,
    device: str = "cpu",
    fuse: bool = True,
    network_size: int | None = None,
    config: dict[str, Any] | None = None,
) -> XTX4Model:
    """Load a trained XTX4 model ready for inference.

    Rebuilds the network from the checkpoint's ``config["model"]`` (falls back to
    ``variant_spec``), loads ``ema_state`` when present else ``model_state``,
    fuses by default (Conv+BN fold, RepConv reparam, one2many head dropped) and
    attaches the inference attributes consumed by :func:`detect_poses`.
    """

    if not (device == "cpu" or device.startswith("cuda")):
        raise ValueError(f"load_model supports device 'cpu'|'cuda', got {device!r} (use export for ncnn)")

    if weights is not None:
        from .engine.checkpoint import load_checkpoint

        ckpt = load_checkpoint(Path(weights), map_location="cpu")
        config = ckpt.get("config") or config or {}
        if isinstance(config, dict) and config.get("model"):
            model = build_model(config)
        elif ckpt.get("variant_spec"):
            from .models.scaling import VariantSpec

            model = XTX4Model(spec=VariantSpec.from_dict(ckpt["variant_spec"]))
        else:
            raise ValueError("Checkpoint carries neither config['model'] nor variant_spec")
        state = ckpt.get("ema_state") or ckpt.get("model_state")
        if state is None:
            raise ValueError("Checkpoint has neither ema_state nor model_state")
        model.load_state_dict(state, strict=True)
    else:
        config = config or {}
        model = build_model(config)
        logger.warning("load_model called without weights; using randomly initialised model")

    model.to(device).eval()
    if fuse:
        model.fuse()
    return attach_inference_attributes(model, config=config, device=device, network_size=network_size)


# --------------------------------------------------------------------------- OKS-NMS


def oks_between(
    kpts_a: np.ndarray, kpts_b: np.ndarray, areas_b: np.ndarray, *, sigmas: np.ndarray = OKS_SIGMAS
) -> np.ndarray:
    """OKS matrix ``(A, B)`` between two pose sets using ``areas_b`` and points of ``b`` with conf >= 0.5."""

    if kpts_a.shape[0] == 0 or kpts_b.shape[0] == 0:
        return np.zeros((kpts_a.shape[0], kpts_b.shape[0]), dtype=np.float64)
    dx = kpts_a[:, None, :, 0] - kpts_b[None, :, :, 0]
    dy = kpts_a[:, None, :, 1] - kpts_b[None, :, :, 1]
    var = (2.0 * np.asarray(sigmas, dtype=np.float64)) ** 2
    e = (dx * dx + dy * dy) / (2.0 * np.maximum(areas_b, 1e-6)[None, :, None] * var[None, None, :])
    labeled = kpts_b[None, :, :, 2] >= 0.5
    labeled = labeled | ~labeled.any(axis=2, keepdims=True)
    return (np.exp(-e) * labeled).sum(axis=2) / np.maximum(labeled.sum(axis=2), 1)


def oks_nms(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    sigmas: np.ndarray = OKS_SIGMAS,
    iou_threshold: float | None = None,
    scale_factor: float = DEFAULT_NMS_SCALE_FACTOR,
) -> np.ndarray:
    """Greedy OKS-NMS; returns the kept indices (descending score).

    Delegates to :func:`xtx4.runtime.decode.oks_nms_indices` so the Python API, the
    evaluation and the deployed runtime share one rule: OKS scale from the
    ``scale_factor``-expanded extent box of the candidate's confident points, and
    optionally also expanded-box IoU ``>= iou_threshold``.
    """

    return oks_nms_indices(
        keypoints, scores, threshold=threshold, sigmas=sigmas, iou_threshold=iou_threshold,
        scale_factor=scale_factor,
    )


# --------------------------------------------------------------------------- inference


def _resolve(model: XTX4Model, conf_threshold, max_detections, network_size, use_nms) -> dict[str, Any]:
    decode = dict(getattr(model, "decode", None) or DEFAULT_DECODE)
    return {
        "conf_threshold": float(decode["conf_threshold"] if conf_threshold is None else conf_threshold),
        "max_detections": int(decode["max_detections"] if max_detections is None else max_detections),
        "network_size": int(network_size or getattr(model, "network_size", 0) or model.spec.default_size),
        "sigma_conf_coeff": float(decode.get("sigma_conf_coeff", 0.0)),
        "sigma_floor": float(decode.get("sigma_floor", 0.05)),
        "oks_nms": bool(decode.get("oks_nms", False) if use_nms is None else use_nms),
        "oks_nms_threshold": float(decode.get("oks_nms_threshold", 0.5)),
        "oks_nms_iou_threshold": (
            None if decode.get("oks_nms_iou_threshold") is None else float(decode["oks_nms_iou_threshold"])
        ),
        "oks_nms_scale_factor": float(decode.get("oks_nms_scale_factor", DEFAULT_NMS_SCALE_FACTOR)),
        "present_threshold": float(decode.get("present_threshold", 0.5)),
        "visible_threshold": float(decode.get("visible_threshold", 0.5)),
        "device": str(getattr(model, "device_str", None) or next(model.parameters()).device),
    }


def _views_to_batch(views: list[np.ndarray], *, device: str, channels_last: bool) -> torch.Tensor:
    stacked = np.stack(views, axis=0)  # (B, S, S, 3) uint8 BGR
    tensor = torch.from_numpy(stacked).to(device, non_blocking=True)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if channels_last:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor.contiguous()


def _raw_detections(model: XTX4Model, batch: torch.Tensor, params: dict[str, Any]):
    """``(dets, vis, sigma)`` lists via ``model.predict`` (all decode parameters passed through)."""

    kwargs = dict(
        conf_threshold=params["conf_threshold"],
        max_detections=params["max_detections"],
        sigma_conf_coeff=params["sigma_conf_coeff"],
        sigma_floor=params["sigma_floor"],
        present_threshold=params.get("present_threshold", 0.5),
        visible_threshold=params.get("visible_threshold", 0.5),
    )
    if hasattr(model, "predict"):
        return model.predict(batch, **kwargs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(batch)
    finally:
        if was_training:
            model.train()
    return decode_detections(output.one2one, **kwargs)


def _to_poses(
    det: np.ndarray,
    vis: np.ndarray,
    sigma: np.ndarray,
    *,
    transform: np.ndarray,
    orig_size: tuple[int, int],
    params: dict[str, Any],
) -> list[Pose]:
    if det.shape[0] == 0:
        return []
    width, height = orig_size
    scores = det[:, 0].astype(np.float32)
    kpts = det[:, 1:].reshape(det.shape[0], NUM_KEYPOINTS, 3)
    kpts = geometry.apply_to_keypoints(transform, kpts).astype(np.float32)
    # Letterbox is a pure scale + translation; sigma scales with the (isotropic) factor.
    scale = float(abs(transform[0, 0]))
    sigma_px = (sigma * scale).astype(np.float32)
    keep = np.arange(det.shape[0])
    if params["oks_nms"] and det.shape[0] > 1:
        keep = oks_nms(
            kpts, scores, threshold=params["oks_nms_threshold"],
            iou_threshold=params.get("oks_nms_iou_threshold"),
            scale_factor=params.get("oks_nms_scale_factor", DEFAULT_NMS_SCALE_FACTOR),
        )
    poses: list[Pose] = []
    for i in keep:
        kp = kpts[i]
        norm = kp.copy()
        norm[:, 0] /= max(width, 1)
        norm[:, 1] /= max(height, 1)
        poses.append(
            Pose(
                score=float(scores[i]),
                keypoints=kp,
                keypoints_norm=norm,
                visibility=vis[i].astype(np.int64),
                sigma_px=sigma_px[i],
            )
        )
    return poses


def detect_poses_batch(
    images: Sequence[np.ndarray],
    model: XTX4Model,
    conf_threshold: float | None = None,
    max_detections: int | None = None,
    network_size: int | None = None,
    oks_nms: bool | None = None,
) -> list[list[Pose]]:
    """Detect poses in several BGR ``uint8`` images with one batched forward pass."""

    if model is None:
        raise ValueError("model is required; build it with load_model(...)")
    if not images:
        return []
    params = _resolve(model, conf_threshold, max_detections, network_size, oks_nms)
    size = params["network_size"]
    results = [preprocess_image(img, mode="whole", network_size=size) for img in images]
    channels_last = bool(getattr(model, "channels_last", False))
    batch = _views_to_batch([r.views[_WHOLE_VIEW] for r in results], device=params["device"], channels_last=channels_last)
    dets, vis, sigma = _raw_detections(model, batch, params)
    outputs: list[list[Pose]] = []
    for idx, result in enumerate(results):
        outputs.append(
            _to_poses(
                dets[idx].cpu().numpy(),
                vis[idx].cpu().numpy(),
                sigma[idx].cpu().numpy(),
                transform=result.transforms[_WHOLE_VIEW],
                orig_size=result.orig_size,
                params=params,
            )
        )
    return outputs


def detect_poses(
    image_bgr: np.ndarray,
    model: XTX4Model,
    conf_threshold: float | None = None,
    max_detections: int | None = None,
    network_size: int | None = None,
    oks_nms: bool | None = None,
) -> list[Pose]:
    """Detect head-and-torso poses in a single BGR ``uint8`` image (original-image pixels, score-sorted)."""

    return detect_poses_batch(
        [image_bgr],
        model,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
        network_size=network_size,
        oks_nms=oks_nms,
    )[0]
