from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RectangleLike = Mapping[str, float | int] | Sequence[float | int]
_NormalizedRectangle = tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class MotionVector:
    x: float = 0.0
    y: float = 0.0


@dataclass(slots=True, frozen=True)
class CameraMotionConfig:
    downscale_factor: float = 1.0
    max_features: int = 250
    quality_level: float = 0.01
    min_distance: float = 10.0
    block_size: int = 7
    lk_window_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    lk_epsilon: float = 0.01
    min_tracked_points: int = 12
    ransac_reproj_threshold: float = 3.0
    ransac_max_iterations: int = 2000
    ransac_confidence: float = 0.99
    ransac_refine_iterations: int = 10
    min_inlier_count: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.downscale_factor <= 1.0:
            raise ValueError("downscale_factor must be in the range (0.0, 1.0].")
        if self.max_features < 1:
            raise ValueError("max_features must be at least 1.")
        if self.quality_level <= 0.0:
            raise ValueError("quality_level must be greater than 0.")
        if self.min_distance < 0.0:
            raise ValueError("min_distance must not be negative.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if len(self.lk_window_size) != 2 or min(self.lk_window_size) < 1:
            raise ValueError("lk_window_size must contain two positive integers.")
        if self.lk_max_level < 0:
            raise ValueError("lk_max_level must not be negative.")
        if self.lk_max_iterations < 1:
            raise ValueError("lk_max_iterations must be at least 1.")
        if self.lk_epsilon <= 0.0:
            raise ValueError("lk_epsilon must be greater than 0.")
        if self.min_tracked_points < 3:
            raise ValueError("min_tracked_points must be at least 3.")
        if self.ransac_reproj_threshold <= 0.0:
            raise ValueError("ransac_reproj_threshold must be greater than 0.")
        if self.ransac_max_iterations < 1:
            raise ValueError("ransac_max_iterations must be at least 1.")
        if not 0.0 < self.ransac_confidence < 1.0:
            raise ValueError("ransac_confidence must be in the range (0.0, 1.0).")
        if self.ransac_refine_iterations < 0:
            raise ValueError("ransac_refine_iterations must not be negative.")
        if self.min_inlier_count < 3:
            raise ValueError("min_inlier_count must be at least 3.")


DEFAULT_CAMERA_MOTION_CONFIG = CameraMotionConfig()
RASPBERRY_PI_CAMERA_MOTION_CONFIG = CameraMotionConfig(
    downscale_factor=0.5,
    max_features=180,
    quality_level=0.02,
    min_distance=8.0,
    block_size=5,
    lk_window_size=(15, 15),
    lk_max_level=2,
    lk_max_iterations=20,
    lk_epsilon=0.03,
    min_tracked_points=10,
    ransac_reproj_threshold=2.5,
    ransac_max_iterations=1500,
    ransac_refine_iterations=8,
    min_inlier_count=6,
)


@dataclass(slots=True, frozen=True)
class CameraMovementResult:
    valid: bool
    pixel: MotionVector = field(default_factory=MotionVector)
    normalized: MotionVector = field(default_factory=MotionVector)
    rotation_degrees: float = 0.0
    feature_count: int = 0
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    confidence: float = 0.0
    downscale_factor: float = 1.0
    reason: str = ""


@dataclass(slots=True)
class _PreparedFrame:
    gray: np.ndarray
    allowed_mask: np.ndarray | None
    width: int
    height: int
    scale_x: float
    scale_y: float


def get_camera_movement(
    *,
    frames: Sequence[Any],
    masked_rectangles: Sequence[Sequence[RectangleLike] | None] | None = None,
    config: CameraMotionConfig | None = None,
) -> CameraMovementResult:
    """
    Estimate dominant image motion between two consecutive frames.

    Positive X means the image content moved right. Positive Y means the image
    content moved down between the first and second frame.
    """
    effective_config = config or DEFAULT_CAMERA_MOTION_CONFIG
    previous_image, current_image = _normalize_frame_pair(frames)
    previous_rectangles, current_rectangles = _normalize_masked_rectangles(masked_rectangles)
    previous_prepared = _prepare_frame(previous_image, previous_rectangles, config=effective_config)
    current_prepared = _prepare_frame(current_image, current_rectangles, config=effective_config)
    return _estimate_movement_from_prepared_frames(
        previous_prepared,
        current_prepared,
        config=effective_config,
    )


def get_camera_movement_from_files(
    *,
    previous_frame_filename: str | Path,
    current_frame_filename: str | Path,
    masked_rectangles: Sequence[Sequence[RectangleLike] | None] | None = None,
    config: CameraMotionConfig | None = None,
) -> CameraMovementResult:
    """
    Load two image files (JPEG, PNG, WebP, TIFF, ...) with OpenCV and run get_camera_movement.

    Paths must be readable by cv2.imread; missing or unsupported files raise ValueError.
    """
    prev_path = str(Path(previous_frame_filename))
    curr_path = str(Path(current_frame_filename))
    previous_image = cv2.imread(prev_path)
    if previous_image is None:
        raise ValueError(f"Could not load image: {prev_path}")
    current_image = cv2.imread(curr_path)
    if current_image is None:
        raise ValueError(f"Could not load image: {curr_path}")
    return get_camera_movement(
        frames=[previous_image, current_image],
        masked_rectangles=masked_rectangles,
        config=config,
    )


class CameraMotionEstimator:
    """Stateful estimator that reuses the previously prepared frame."""

    def __init__(self, *, config: CameraMotionConfig | None = None):
        self.config = config or DEFAULT_CAMERA_MOTION_CONFIG
        self._previous_prepared: _PreparedFrame | None = None

    def reset(self) -> None:
        self._previous_prepared = None

    def update(
        self,
        frame: Any,
        masked_rectangles: Sequence[RectangleLike] | None = None,
    ) -> CameraMovementResult:
        current_prepared = _prepare_frame(
            _extract_image(frame),
            _normalize_rectangle_collection(masked_rectangles),
            config=self.config,
        )
        if self._previous_prepared is None:
            self._previous_prepared = current_prepared
            return _make_result(
                valid=False,
                width=current_prepared.width,
                height=current_prepared.height,
                downscale_factor=self.config.downscale_factor,
                reason="warmup",
            )
        if (
            self._previous_prepared.width != current_prepared.width
            or self._previous_prepared.height != current_prepared.height
        ):
            self._previous_prepared = current_prepared
            return _make_result(
                valid=False,
                width=current_prepared.width,
                height=current_prepared.height,
                downscale_factor=self.config.downscale_factor,
                reason="frame_size_changed",
            )
        result = _estimate_movement_from_prepared_frames(
            self._previous_prepared,
            current_prepared,
            config=self.config,
        )
        self._previous_prepared = current_prepared
        return result


class RaspberryPiCameraMotionEstimator(CameraMotionEstimator):
    """Pi-oriented estimator with conservative defaults for throughput."""

    def __init__(self, *, config: CameraMotionConfig | None = None):
        super().__init__(config=config or RASPBERRY_PI_CAMERA_MOTION_CONFIG)


def _normalize_frame_pair(frames: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frames, np.ndarray):
        raise ValueError("frames must contain exactly two frame objects or image arrays.")
    normalized_frames = tuple(frames)
    if len(normalized_frames) != 2:
        raise ValueError("frames must contain exactly two frame objects or image arrays.")
    return _extract_image(normalized_frames[0]), _extract_image(normalized_frames[1])


def _normalize_masked_rectangles(
    masked_rectangles: Sequence[Sequence[RectangleLike] | None] | None,
) -> tuple[tuple[_NormalizedRectangle, ...], tuple[_NormalizedRectangle, ...]]:
    if masked_rectangles is None:
        return (), ()
    rectangles_per_frame = tuple(masked_rectangles)
    if len(rectangles_per_frame) != 2:
        raise ValueError("masked_rectangles must contain one rectangle collection per frame.")
    return (
        _normalize_rectangle_collection(rectangles_per_frame[0]),
        _normalize_rectangle_collection(rectangles_per_frame[1]),
    )


def _normalize_rectangle_collection(
    rectangles: Sequence[RectangleLike] | None,
) -> tuple[_NormalizedRectangle, ...]:
    if rectangles is None:
        return ()
    return tuple(_coerce_rectangle(rectangle) for rectangle in rectangles)


def _coerce_rectangle(rectangle: RectangleLike) -> _NormalizedRectangle:
    if isinstance(rectangle, Mapping):
        if {"x", "y", "width", "height"} <= rectangle.keys():
            x = rectangle["x"]
            y = rectangle["y"]
            width = rectangle["width"]
            height = rectangle["height"]
        elif {"left", "top", "width", "height"} <= rectangle.keys():
            x = rectangle["left"]
            y = rectangle["top"]
            width = rectangle["width"]
            height = rectangle["height"]
        elif {"x1", "y1", "x2", "y2"} <= rectangle.keys():
            x = rectangle["x1"]
            y = rectangle["y1"]
            width = float(rectangle["x2"]) - float(rectangle["x1"])
            height = float(rectangle["y2"]) - float(rectangle["y1"])
        else:
            raise ValueError(
                "Rectangle mappings must provide x/y/width/height, left/top/width/height, "
                "or x1/y1/x2/y2 keys."
            )
        return float(x), float(y), float(width), float(height)
    if isinstance(rectangle, Sequence) and not isinstance(rectangle, (str, bytes)) and len(rectangle) == 4:
        x, y, width, height = rectangle
        return float(x), float(y), float(width), float(height)
    raise ValueError("Rectangles must be mappings or four-item sequences.")


def _extract_image(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        image = frame
    else:
        image = getattr(frame, "image", None)
    if not isinstance(image, np.ndarray):
        raise TypeError("Each frame must be a numpy array or expose a numpy image attribute.")
    if image.ndim not in (2, 3):
        raise ValueError("Frame images must be grayscale or multi-channel numpy arrays.")
    return np.ascontiguousarray(image)


def _prepare_frame(
    image: np.ndarray,
    rectangles: Sequence[_NormalizedRectangle],
    *,
    config: CameraMotionConfig,
) -> _PreparedFrame:
    gray = _to_grayscale(image)
    original_height, original_width = gray.shape[:2]

    if config.downscale_factor < 1.0:
        target_width = max(1, int(round(original_width * config.downscale_factor)))
        target_height = max(1, int(round(original_height * config.downscale_factor)))
        if target_width != original_width or target_height != original_height:
            gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

    scale_x = gray.shape[1] / original_width
    scale_y = gray.shape[0] / original_height
    allowed_mask = _build_allowed_mask(
        gray_shape=gray.shape,
        rectangles=rectangles,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return _PreparedFrame(
        gray=np.ascontiguousarray(gray),
        allowed_mask=allowed_mask,
        width=original_width,
        height=original_height,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image shape for grayscale conversion.")
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    return gray


def _build_allowed_mask(
    *,
    gray_shape: tuple[int, ...],
    rectangles: Sequence[_NormalizedRectangle],
    scale_x: float,
    scale_y: float,
) -> np.ndarray | None:
    if not rectangles:
        return None

    height, width = gray_shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x, y, rect_width, rect_height in rectangles:
        if rect_width <= 0.0 or rect_height <= 0.0:
            continue
        x0 = max(0, min(width, int(math.floor(x * scale_x))))
        y0 = max(0, min(height, int(math.floor(y * scale_y))))
        x1 = max(0, min(width, int(math.ceil((x + rect_width) * scale_x))))
        y1 = max(0, min(height, int(math.ceil((y + rect_height) * scale_y))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask


def _estimate_movement_from_prepared_frames(
    previous_frame: _PreparedFrame,
    current_frame: _PreparedFrame,
    *,
    config: CameraMotionConfig,
) -> CameraMovementResult:
    if previous_frame.width != current_frame.width or previous_frame.height != current_frame.height:
        return _make_result(
            valid=False,
            width=current_frame.width,
            height=current_frame.height,
            downscale_factor=config.downscale_factor,
            reason="frame_size_mismatch",
        )

    feature_points = cv2.goodFeaturesToTrack(
        previous_frame.gray,
        maxCorners=config.max_features,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=previous_frame.allowed_mask,
        blockSize=config.block_size,
    )
    feature_count = 0 if feature_points is None else int(len(feature_points))
    if feature_count < config.min_tracked_points:
        return _make_result(
            valid=False,
            width=previous_frame.width,
            height=previous_frame.height,
            downscale_factor=config.downscale_factor,
            feature_count=feature_count,
            reason="insufficient_features",
        )

    tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_frame.gray,
        current_frame.gray,
        feature_points,
        None,
        winSize=config.lk_window_size,
        maxLevel=config.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.lk_max_iterations,
            config.lk_epsilon,
        ),
    )
    if tracked_points is None or status is None:
        return _make_result(
            valid=False,
            width=previous_frame.width,
            height=previous_frame.height,
            downscale_factor=config.downscale_factor,
            feature_count=feature_count,
            reason="optical_flow_failed",
        )

    previous_points = feature_points.reshape(-1, 2)
    current_points = tracked_points.reshape(-1, 2)
    status_mask = status.reshape(-1) == 1
    finite_mask = np.isfinite(previous_points).all(axis=1) & np.isfinite(current_points).all(axis=1)
    bounds_mask = (
        (current_points[:, 0] >= 0.0)
        & (current_points[:, 0] < current_frame.gray.shape[1])
        & (current_points[:, 1] >= 0.0)
        & (current_points[:, 1] < current_frame.gray.shape[0])
    )
    keep_mask = status_mask & finite_mask & bounds_mask
    previous_points = previous_points[keep_mask]
    current_points = current_points[keep_mask]

    if current_frame.allowed_mask is not None and len(current_points) > 0:
        allowed_points = _points_inside_allowed_mask(current_points, current_frame.allowed_mask)
        previous_points = previous_points[allowed_points]
        current_points = current_points[allowed_points]

    tracked_count = int(len(previous_points))
    if tracked_count < config.min_tracked_points:
        return _make_result(
            valid=False,
            width=previous_frame.width,
            height=previous_frame.height,
            downscale_factor=config.downscale_factor,
            feature_count=feature_count,
            tracked_count=tracked_count,
            reason="insufficient_tracked_points",
        )

    matrix, inliers = cv2.estimateAffinePartial2D(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=config.ransac_max_iterations,
        confidence=config.ransac_confidence,
        refineIters=config.ransac_refine_iterations,
    )
    if matrix is None or inliers is None:
        return _make_result(
            valid=False,
            width=previous_frame.width,
            height=previous_frame.height,
            downscale_factor=config.downscale_factor,
            feature_count=feature_count,
            tracked_count=tracked_count,
            reason="affine_estimation_failed",
        )

    inlier_count = int(inliers.reshape(-1).sum())
    if inlier_count < config.min_inlier_count:
        return _make_result(
            valid=False,
            width=previous_frame.width,
            height=previous_frame.height,
            downscale_factor=config.downscale_factor,
            feature_count=feature_count,
            tracked_count=tracked_count,
            inlier_count=inlier_count,
            reason="insufficient_inliers",
        )

    pixel_x = float(matrix[0, 2] / previous_frame.scale_x)
    pixel_y = float(matrix[1, 2] / previous_frame.scale_y)
    inlier_ratio = inlier_count / tracked_count
    confidence = _estimate_confidence(
        config=config,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
    )
    rotation_degrees = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    return _make_result(
        valid=True,
        width=previous_frame.width,
        height=previous_frame.height,
        downscale_factor=config.downscale_factor,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        rotation_degrees=rotation_degrees,
        feature_count=feature_count,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        confidence=confidence,
    )


def _points_inside_allowed_mask(points: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    x_indices = np.clip(np.rint(points[:, 0]).astype(int), 0, allowed_mask.shape[1] - 1)
    y_indices = np.clip(np.rint(points[:, 1]).astype(int), 0, allowed_mask.shape[0] - 1)
    return allowed_mask[y_indices, x_indices] > 0


def _estimate_confidence(
    *,
    config: CameraMotionConfig,
    tracked_count: int,
    inlier_count: int,
    inlier_ratio: float,
) -> float:
    track_score = min(1.0, tracked_count / max(config.min_tracked_points * 3, 1))
    inlier_score = min(1.0, inlier_count / max(config.min_inlier_count * 3, 1))
    confidence = 0.5 * inlier_ratio + 0.3 * inlier_score + 0.2 * track_score
    return float(max(0.0, min(1.0, confidence)))


def _make_result(
    *,
    valid: bool,
    width: int,
    height: int,
    downscale_factor: float,
    pixel_x: float = 0.0,
    pixel_y: float = 0.0,
    rotation_degrees: float = 0.0,
    feature_count: int = 0,
    tracked_count: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    confidence: float = 0.0,
    reason: str = "",
) -> CameraMovementResult:
    normalized_x = pixel_x / width if width > 0 else 0.0
    normalized_y = pixel_y / height if height > 0 else 0.0
    return CameraMovementResult(
        valid=valid,
        pixel=MotionVector(x=float(pixel_x), y=float(pixel_y)),
        normalized=MotionVector(x=float(normalized_x), y=float(normalized_y)),
        rotation_degrees=float(rotation_degrees),
        feature_count=feature_count,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        confidence=float(confidence),
        downscale_factor=float(downscale_factor),
        reason=reason,
    )


__all__ = [
    "CameraMotionConfig",
    "CameraMotionEstimator",
    "CameraMovementResult",
    "DEFAULT_CAMERA_MOTION_CONFIG",
    "MotionVector",
    "RASPBERRY_PI_CAMERA_MOTION_CONFIG",
    "RaspberryPiCameraMotionEstimator",
    "get_camera_movement",
    "get_camera_movement_from_files",
]
