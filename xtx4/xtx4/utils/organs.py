"""Organ circles (brain, chest) derived from the nine XTX4 landmarks: the single source of truth.

The formulas are ported from FreeSentry ``server/cameraai.py`` (``_get_brain_position`` and the chest
part of ``_get_chest_and_abdomen_position``); see ``extradetection.md`` section 1 for the derivation.
They need no training: every organ is a fixed linear combination of landmarks plus distances, maxima
and minima. The same constants drive the NumPy reference here (host fallback, tests) and the
parameter-free ``xtx4.models.organs.OrganLayer`` appended to the exported graph.

Brain -- the server's ``calculate_point`` chain expands to an affine combination (weights sum to 1)::

    brain_full = -0.75 N + 0.5 LEy + 0.5 REy + 0.375 LEa + 0.375 REa
    r_full     = 0.59 * max(|B - N|, |B - LEa|, |B - REa|)
    conf_full  = 0.2 * min5 + 0.8 * mean5                        (nose, eyes, ears)

    ear branch, used when both ears have conf >= 0.01 but not all five face points do:
    brain_ear  = 0.5 LEa + 0.5 REa,  r_ear = 0.59 * |B - LEa|,  conf_ear = mean(c_LEa, c_REa)

Chest -- ``0.8 * mid(shoulders) + 0.2 * mid(hips)``::

    chest      = 0.4 LS + 0.4 RS + 0.1 LH + 0.1 RH
    r_chest    = max(0.21 * (|LS-RS| + |LH-RH|), 0.1 * (|LS-LH| + |RS-RH|))
    conf_chest = (2 c_LS + 2 c_RS + c_LH + c_RH + min4) / 7      (server: / 5, i.e. up to 1.4)

Deliberate deviation from the server: the chest confidence is a true weighted mean (divided by 7, the
sum of the weights), so it stays in ``[0, 1]``; ``conf_xtx4 = conf_server * 5 / 7``. Both confidences
are finally clamped to ``[0, 1]`` (FP16 rounding guard).

Because both centres are affine combinations and the radii are distances, the computation is
translation-invariant and scales linearly: it can run on anchor-relative stride units (the raw
``(dx, dy)`` of the pose output) and be decoded exactly like a keypoint afterwards.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .keypoints import NUM_KEYPOINTS

ORGAN_NAMES: Final[tuple[str, ...]] = ("brain", "chest")
NUM_ORGANS: Final[int] = len(ORGAN_NAMES)
ORGAN_CHANNELS: Final[tuple[str, ...]] = ("dx", "dy", "r_units", "conf")
ORGAN_CHANNELS_PER_ORGAN: Final[int] = len(ORGAN_CHANNELS)
ORGAN_VERSION: Final[int] = 1
ORGAN_SOURCE: Final[str] = "server/cameraai.py (_get_brain_position, _get_chest_and_abdomen_position)"

# Landmark indices (KEYPOINT_NAMES order).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 7, 8
FACE_POINTS: Final[tuple[int, ...]] = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
TORSO_POINTS: Final[tuple[int, ...]] = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# Centre weights over the nine landmarks (each row sums to 1).
BRAIN_FULL_WEIGHTS: Final[np.ndarray] = np.array([-0.75, 0.5, 0.5, 0.375, 0.375, 0, 0, 0, 0], np.float64)
BRAIN_EAR_WEIGHTS: Final[np.ndarray] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0], np.float64)
CHEST_WEIGHTS: Final[np.ndarray] = np.array([0, 0, 0, 0, 0, 0.4, 0.4, 0.1, 0.1], np.float64)

BRAIN_RADIUS_FACTOR: Final[float] = 0.59
BRAIN_EAR_THRESHOLD: Final[float] = 0.01  # server default confidence_threshold (decision O1)
BRAIN_CONF_MIN_WEIGHT: Final[float] = 0.2  # min + 0.8 * (avg - min) = 0.2 min + 0.8 avg
CHEST_RADIUS_LR: Final[float] = 0.21  # * (shoulder width + hip width)
CHEST_RADIUS_TB: Final[float] = 0.1  # * (left side + right side)
CHEST_CONF_WEIGHTS: Final[tuple[float, ...]] = (2.0, 2.0, 1.0, 1.0, 1.0)  # LS, RS, LH, RH, min4
CHEST_CONF_DIVISOR: Final[float] = 7.0  # sum of CHEST_CONF_WEIGHTS (server: 5, decision O2)

# Branch-free gate g(c) = clip((c - threshold) * gain + 1, 0, 1): exactly 1 for c >= threshold (as the
# server's ``>=``), 0 below threshold - 1/gain, linear in between. The gain is FP16-safe (< 65504).
GATE_GAIN: Final[float] = 1e4


def organ_constants() -> dict:
    """All constants as plain JSON values (recorded in the export manifest for auditing)."""

    return {
        "brain_full_weights": BRAIN_FULL_WEIGHTS.tolist(),
        "brain_ear_weights": BRAIN_EAR_WEIGHTS.tolist(),
        "chest_weights": CHEST_WEIGHTS.tolist(),
        "brain_radius_factor": BRAIN_RADIUS_FACTOR,
        "brain_ear_threshold": BRAIN_EAR_THRESHOLD,
        "brain_conf_min_weight": BRAIN_CONF_MIN_WEIGHT,
        "chest_radius_lr": CHEST_RADIUS_LR,
        "chest_radius_tb": CHEST_RADIUS_TB,
        "chest_conf_weights": list(CHEST_CONF_WEIGHTS),
        "chest_conf_divisor": CHEST_CONF_DIVISOR,
        "gate_gain": GATE_GAIN,
    }


def gate_ramp_rows(conf: np.ndarray, *, margin: float = 1e-5) -> np.ndarray:
    """``(...,)`` bool: rows with a face confidence inside the brain gate's ramp (plus ``margin``).

    There the gate's slope (``GATE_GAIN``) turns float32 rounding of a confidence (~1e-7) into a
    ~1e-3 blend between the full and the ear branch, so parity checks between two float32 backends
    exclude these rows (they are a ~1e-4-wide band just below the 0.01 threshold).
    """

    face = np.asarray(conf, dtype=np.float64)[..., list(FACE_POINTS)]
    lo = BRAIN_EAR_THRESHOLD - 1.0 / GATE_GAIN - margin
    return ((face >= lo) & (face <= BRAIN_EAR_THRESHOLD + margin)).any(axis=-1)


def _gate(conf: np.ndarray, gate_gain: float | None) -> np.ndarray:
    if gate_gain is None:
        return (conf >= BRAIN_EAR_THRESHOLD).astype(np.float64)
    return np.clip((conf - BRAIN_EAR_THRESHOLD) * gate_gain + 1.0, 0.0, 1.0)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def compute_organs(xy: np.ndarray, conf: np.ndarray, *, gate_gain: float | None = GATE_GAIN) -> np.ndarray:
    """Brain and chest circles from the nine landmarks.

    Args:
        xy: ``(..., 9, 2)`` landmark positions in any linear unit (frame px, network px or
            anchor-relative stride units -- the result is in the same unit).
        conf: ``(..., 9)`` landmark confidences in ``[0, 1]`` (``sigmoid(present_logit)``).
        gate_gain: gain of the branch-free brain gate (the graph's value by default, so host and
            graph agree exactly); ``None`` uses the server's hard ``>=`` comparison.

    Returns:
        ``(..., 2, 4)`` float64, rows ``brain, chest``, columns ``(x, y, radius, conf)``.
    """

    xy = np.asarray(xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    if xy.shape[-2:] != (NUM_KEYPOINTS, 2) or conf.shape != xy.shape[:-1]:
        raise ValueError(f"Expected xy (..., {NUM_KEYPOINTS}, 2) and conf (..., {NUM_KEYPOINTS}), "
                         f"got {xy.shape} and {conf.shape}")

    # ---- brain
    nose, l_ear, r_ear = xy[..., NOSE, :], xy[..., LEFT_EAR, :], xy[..., RIGHT_EAR, :]
    b_full = np.einsum("k,...kd->...d", BRAIN_FULL_WEIGHTS, xy)
    r_full = BRAIN_RADIUS_FACTOR * np.maximum(np.maximum(_dist(b_full, nose), _dist(b_full, l_ear)),
                                              _dist(b_full, r_ear))
    face = conf[..., list(FACE_POINTS)]
    c_full = BRAIN_CONF_MIN_WEIGHT * face.min(axis=-1) + (1.0 - BRAIN_CONF_MIN_WEIGHT) * face.mean(axis=-1)

    b_ear = np.einsum("k,...kd->...d", BRAIN_EAR_WEIGHTS, xy)
    r_ear = BRAIN_RADIUS_FACTOR * _dist(b_ear, l_ear)
    c_ear = 0.5 * (conf[..., LEFT_EAR] + conf[..., RIGHT_EAR])

    g = _gate(face, gate_gain)
    ears_ok = g[..., 3] * g[..., 4]
    m = ears_ok * (1.0 - g.prod(axis=-1))  # both ears ok AND NOT all five ok
    brain_xy = m[..., None] * b_ear + (1.0 - m[..., None]) * b_full
    brain_r = m * r_ear + (1.0 - m) * r_full
    brain_c = np.clip(m * c_ear + (1.0 - m) * c_full, 0.0, 1.0)

    # ---- chest
    ls, rs = xy[..., LEFT_SHOULDER, :], xy[..., RIGHT_SHOULDER, :]
    lh, rh = xy[..., LEFT_HIP, :], xy[..., RIGHT_HIP, :]
    chest_xy = np.einsum("k,...kd->...d", CHEST_WEIGHTS, xy)
    chest_r = np.maximum(CHEST_RADIUS_LR * (_dist(ls, rs) + _dist(lh, rh)),
                         CHEST_RADIUS_TB * (_dist(ls, lh) + _dist(rs, rh)))
    torso = conf[..., list(TORSO_POINTS)]
    w = CHEST_CONF_WEIGHTS
    chest_c = (w[0] * torso[..., 0] + w[1] * torso[..., 1] + w[2] * torso[..., 2] + w[3] * torso[..., 3]
               + w[4] * torso.min(axis=-1)) / CHEST_CONF_DIVISOR
    chest_c = np.clip(chest_c, 0.0, 1.0)

    brain = np.concatenate([brain_xy, brain_r[..., None], brain_c[..., None]], axis=-1)
    chest = np.concatenate([chest_xy, chest_r[..., None], chest_c[..., None]], axis=-1)
    return np.stack([brain, chest], axis=-2)


def organs_to_dict(organs: np.ndarray) -> dict[str, dict]:
    """One person's ``(2, 4)`` organs in the FreeSentry server shape
    ``{"brain": {"point": (x, y), "radius": r, "confidence": c}, "chest": {...}}``."""

    organs = np.asarray(organs, dtype=np.float64)
    return {
        name: {
            "point": (float(organs[i, 0]), float(organs[i, 1])),
            "radius": float(organs[i, 2]),
            "confidence": float(organs[i, 3]),
        }
        for i, name in enumerate(ORGAN_NAMES)
    }
