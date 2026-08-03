"""Affine geometry helpers for mapping detections between coordinate spaces.

Every view (whole-frame or pyramid level) carries a 2x3 float64 affine matrix
``T`` mapping **network pixel -> original-image pixel**. These helpers build,
compose, invert, and apply such matrices to points / boxes / keypoints so
detections map back with one matrix multiply.

All matrices are ``float64`` numpy arrays of shape ``(2, 3)`` representing::

    [x_out, y_out]^T = M @ [x_in, y_in, 1]^T

XTX3 transforms are pure scale + translation (no rotation), but the helpers
accept any 2x3 affine so they remain reusable.
"""

from __future__ import annotations

import numpy as np

AffineMatrix = np.ndarray  # shape (2, 3), float64


def make_affine(*, scale_x: float, scale_y: float, tx: float, ty: float) -> AffineMatrix:
    """Build a scale + translation 2x3 affine matrix."""

    return np.array(
        [[scale_x, 0.0, tx], [0.0, scale_y, ty]],
        dtype=np.float64,
    )


def identity_affine() -> AffineMatrix:
    """Return the identity 2x3 affine."""

    return make_affine(scale_x=1.0, scale_y=1.0, tx=0.0, ty=0.0)


def _to_3x3(matrix: AffineMatrix) -> np.ndarray:
    if matrix.shape != (2, 3):
        raise ValueError(f"Affine matrix must be (2, 3), got {matrix.shape}")
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = matrix
    return full


def compose(outer: AffineMatrix, inner: AffineMatrix) -> AffineMatrix:
    """Return the affine equivalent to applying ``inner`` then ``outer``.

    ``compose(A, B)`` maps ``x -> A(B(x))``.
    """

    return (_to_3x3(outer) @ _to_3x3(inner))[:2, :]


def invert_affine(matrix: AffineMatrix) -> AffineMatrix:
    """Return the inverse 2x3 affine (raises if singular)."""

    full = _to_3x3(matrix)
    det = full[0, 0] * full[1, 1] - full[0, 1] * full[1, 0]
    if abs(det) < 1e-12:
        raise ValueError("Affine matrix is singular and cannot be inverted")
    return np.linalg.inv(full)[:2, :]


def apply_to_points(matrix: AffineMatrix, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an ``(N, 2)`` array of points; returns ``(N, 2)``."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {pts.shape}")
    if pts.shape[0] == 0:
        return pts.astype(np.float64)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return homogeneous @ matrix.T


def apply_to_keypoints(matrix: AffineMatrix, keypoints: np.ndarray) -> np.ndarray:
    """Apply affine to ``(N, K, 3)`` keypoints (x, y, conf); conf untouched."""

    kpts = np.asarray(keypoints, dtype=np.float64)
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError(f"keypoints must be (N, K, 3), got {kpts.shape}")
    if kpts.shape[0] == 0:
        return kpts
    out = kpts.copy()
    n, k = kpts.shape[0], kpts.shape[1]
    flat_xy = kpts[:, :, :2].reshape(-1, 2)
    mapped = apply_to_points(matrix, flat_xy).reshape(n, k, 2)
    out[:, :, :2] = mapped
    return out


def apply_to_boxes_xyxy(matrix: AffineMatrix, boxes: np.ndarray) -> np.ndarray:
    """Apply an axis-aligned affine to ``(N, 4)`` xyxy boxes.

    For scale + translation (no rotation) we map both corners and re-order so the
    result stays a valid ``x1<=x2, y1<=y2`` box even under axis flips.
    """

    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must be (N, 4), got {arr.shape}")
    if arr.shape[0] == 0:
        return arr
    corners_a = apply_to_points(matrix, arr[:, [0, 1]])
    corners_b = apply_to_points(matrix, arr[:, [2, 3]])
    x1 = np.minimum(corners_a[:, 0], corners_b[:, 0])
    y1 = np.minimum(corners_a[:, 1], corners_b[:, 1])
    x2 = np.maximum(corners_a[:, 0], corners_b[:, 0])
    y2 = np.maximum(corners_a[:, 1], corners_b[:, 1])
    return np.stack([x1, y1, x2, y2], axis=1)
