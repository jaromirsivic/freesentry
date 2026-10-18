"""Organ reference (``xtx4.utils.organs``) against the FreeSentry server formulas.

The server functions below are copied verbatim from ``server/cameraai.py`` (no import of the server
package): ``calculate_point``, ``calculate_distance``, ``_get_brain_position`` and
``_get_chest_and_abdomen_position``. The only intended difference is the chest confidence
(divided by 7 instead of 5, decision O2 in ``extradetection.md``).
"""

from __future__ import annotations

import numpy as np
import pytest

from xtx4.utils.keypoints import FLIP_PAIRS, KEYPOINT_NAMES, build_flip_index
from xtx4.utils.organs import (
    BRAIN_EAR_THRESHOLD,
    BRAIN_EAR_WEIGHTS,
    BRAIN_FULL_WEIGHTS,
    CHEST_WEIGHTS,
    compute_organs,
    organs_to_dict,
)


# ----------------------------------------------------------------- server code (verbatim copy)
def calculate_point(point_start: tuple[float, float],
                    point_end: tuple[float, float],
                    fraction: float) -> tuple[float, float]:
    """
    Calculate a point between two points.
    """
    return (point_start[0] + fraction * (point_end[0] - point_start[0]),
            point_start[1] + fraction * (point_end[1] - point_start[1]))


def calculate_distance(point_start: tuple[float, float],
                       point_end: tuple[float, float]) -> float:
    """
    Calculate the distance between two points.
    """
    return np.sqrt((point_start[0] - point_end[0])**2 + (point_start[1] - point_end[1])**2)


def _get_brain_position(*, keypoints: list[dict], confidence_threshold: float = 0.01):
    # Calculate the position of the brain
    for pose in keypoints:
        # Set all points in pose
        left_ear_point = pose["left_ear"]["point"]
        right_ear_point = pose["right_ear"]["point"]
        left_eye_point = pose["left_eye"]["point"]
        right_eye_point = pose["right_eye"]["point"]
        nose_point = pose["nose"]["point"]
        # Compute overall confidence
        left_ear_confidence = pose["left_ear"]["confidence"]
        right_ear_confidence = pose["right_ear"]["confidence"]
        left_eye_confidence = pose["left_eye"]["confidence"]
        right_eye_confidence = pose["right_eye"]["confidence"]
        nose_confidence = pose["nose"]["confidence"]
        confidence_of_all_points_is_ok = left_ear_confidence >= confidence_threshold and \
            right_ear_confidence >= confidence_threshold and \
            left_eye_confidence >= confidence_threshold and \
            right_eye_confidence >= confidence_threshold and \
            nose_confidence >= confidence_threshold
        # Calculate the position of the brain from the left ear and right ear
        if (not confidence_of_all_points_is_ok and \
            left_ear_confidence >= confidence_threshold and \
            right_ear_confidence >= confidence_threshold):
            # Calculate center
            brain = calculate_point(left_ear_point, right_ear_point, 0.5)
            # Calculate radius
            d = calculate_distance(brain, left_ear_point)
            radius = d * 0.59
            radius = int(radius)
            # Calculate confidence
            avg_confidence = (left_ear_confidence + right_ear_confidence) / 2
            pose["brain"] = {
                "point": brain,
                "confidence": avg_confidence,
                "radius": radius
            }
        else:
            # Calculate center
            nose_left_ear = calculate_point(nose_point, left_ear_point, 2)
            nose_right_ear = calculate_point(nose_point, right_ear_point, 2)
            nose_left_ear2 = calculate_point(nose_point, left_ear_point, 0.5)
            nose_right_ear2 = calculate_point(nose_point, right_ear_point, 0.5)
            nose_left_eye = calculate_point(nose_left_ear2, left_eye_point, 2)
            nose_right_eye = calculate_point(nose_right_ear2, right_eye_point, 2)
            nose_left_ear_eye = calculate_point(nose_left_ear, nose_left_eye, 0.5)
            nose_right_ear_eye = calculate_point(nose_right_ear, nose_right_eye, 0.5)
            brain = calculate_point(nose_left_ear_eye, nose_right_ear_eye, 0.5)
            # Calculate radius
            distance_brain_nose = calculate_distance(brain, nose_point)
            distance_brain_left_ear = calculate_distance(brain, left_ear_point)
            distance_brain_right_ear = calculate_distance(brain, right_ear_point)
            d = distance_brain_nose if distance_brain_nose > distance_brain_left_ear else distance_brain_left_ear
            d = d if d > distance_brain_right_ear else distance_brain_right_ear
            radius = d * 0.59
            radius = int(radius)
            avg_confidence = (
                nose_confidence +
                left_ear_confidence +
                right_ear_confidence +
                left_eye_confidence +
                right_eye_confidence) / 5
            min_confidence = min(
                nose_confidence,
                left_ear_confidence,
                right_ear_confidence,
                left_eye_confidence,
                right_eye_confidence
            )
            confidence = min_confidence + (avg_confidence - min_confidence) * 0.8
            pose["brain"] = {
                "point": brain,
                "confidence": confidence,
                "radius": radius
            }


def _get_chest_and_abdomen_position(*, keypoints: list[dict], confidence_threshold: float = 0.01):
    # Calculate the position of the chest and abdomen
    for pose in keypoints:
        left_shoulder_right_shoulder = calculate_point(pose["left_shoulder"]["point"], pose["right_shoulder"]["point"], 0.5)
        left_hip_right_hip = calculate_point(pose["left_hip"]["point"], pose["right_hip"]["point"], 0.5)
        left_shoulder_left_hip = calculate_point(pose["left_shoulder"]["point"], pose["left_hip"]["point"], 0.4)
        right_shoulder_right_hip = calculate_point(pose["right_shoulder"]["point"], pose["right_hip"]["point"], 0.4)
        chest = calculate_point(left_shoulder_right_shoulder, left_hip_right_hip, 0.2)
        heart = calculate_point(left_shoulder_left_hip, chest, 0.5)
        abdomen = calculate_point(left_shoulder_right_shoulder, left_hip_right_hip, 0.7)
        liver = calculate_point(left_shoulder_left_hip, right_shoulder_right_hip, 0.7)
        d1 = calculate_distance(pose["left_shoulder"]["point"], pose["right_shoulder"]["point"])
        d2 = calculate_distance(pose["left_hip"]["point"], pose["right_hip"]["point"])
        d3 = calculate_distance(pose["left_shoulder"]["point"], pose["left_hip"]["point"])
        d4 = calculate_distance(pose["right_shoulder"]["point"], pose["right_hip"]["point"])
        davg_left_right = (d1 + d2) * 0.21
        davg_top_bottom = (d3 + d4) * 0.1
        davg = davg_left_right if davg_left_right > davg_top_bottom else davg_top_bottom
        dmin_left_right = d1 if d1 < d2 else d2
        dmin_top_bottom = d3 if d3 < d4 else d4
        dmin_top_bottom = dmin_top_bottom * 0.25
        dmin = dmin_left_right if dmin_left_right > dmin_top_bottom else dmin_top_bottom
        # compute the minimum confidence
        left_shoulder_confidence = pose["left_shoulder"]["confidence"]
        right_shoulder_confidence = pose["right_shoulder"]["confidence"]
        left_hip_confidence = pose["left_hip"]["confidence"]
        right_hip_confidence = pose["right_hip"]["confidence"]
        left_shoulder_confidence_doubled = left_shoulder_confidence * 2
        right_shoulder_confidence_doubled = right_shoulder_confidence * 2
        left_hip_confidence_doubled = left_hip_confidence * 2
        right_hip_confidence_doubled = right_hip_confidence * 2
        min_confidence = min(left_shoulder_confidence, right_shoulder_confidence, left_hip_confidence, right_hip_confidence)
        #avg_confidence_top = (left_shoulder_confidence + right_shoulder_confidence) / 2
        #avg_confidence_bottom = (left_hip_confidence + right_hip_confidence) / 2
        pose["chest"] = {
            "point": chest,
            "confidence": (left_shoulder_confidence_doubled + right_shoulder_confidence_doubled +
                          left_hip_confidence + right_hip_confidence + min_confidence) / 5,
            "radius": int(davg)
        }
        pose["heart"] = {
            "point": heart,
            "confidence": (left_shoulder_confidence_doubled + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence + min_confidence) / 5,
            "radius": int(davg * 0.5)
        }
        pose["abdomen"] = {
            "point": abdomen,
            "confidence": (left_shoulder_confidence + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence_doubled + min_confidence) / 5,
            "radius": int(dmin * 0.5)
        }
        pose["liver"] = {
            "point": liver,
            "confidence": (left_shoulder_confidence + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence_doubled + min_confidence) / 5,
            "radius": int(davg * 0.65)
        }


# ------------------------------------------------------------------------------------ helpers
def _server_organs(xy: np.ndarray, conf: np.ndarray) -> dict:
    """Run the verbatim server code on one person (integer pixel points, as the server rounds)."""

    pose = {name: {"point": (int(xy[k, 0]), int(xy[k, 1])), "confidence": float(conf[k]), "radius": 0}
            for k, name in enumerate(KEYPOINT_NAMES)}
    _get_brain_position(keypoints=[pose])
    _get_chest_and_abdomen_position(keypoints=[pose])
    return pose


def _random_people(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Plausible-ish integer head/torso layouts plus confidences covering both brain branches."""

    centre = rng.uniform(100, 1100, size=(n, 1, 2))
    xy = np.rint(centre + rng.normal(0, 60, size=(n, 9, 2)))
    conf = rng.uniform(0, 1, size=(n, 9))
    # a third of the people get a sub-threshold nose/eye (ear branch), some a sub-threshold ear
    low = rng.uniform(0, BRAIN_EAR_THRESHOLD * 0.9, size=n)
    conf[: n // 3, rng.integers(0, 3)] = low[: n // 3]
    conf[n // 3: n // 2, 3 + rng.integers(0, 2)] = low[n // 3: n // 2]
    # keep clear of the soft gate's 1e-4 ramp just below the threshold (covered by test_brain_branches)
    conf[(conf >= BRAIN_EAR_THRESHOLD - 2e-4) & (conf < BRAIN_EAR_THRESHOLD)] = BRAIN_EAR_THRESHOLD
    return xy, conf


# -------------------------------------------------------------------------------------- tests
def test_weights_are_affine() -> None:
    for weights in (BRAIN_FULL_WEIGHTS, BRAIN_EAR_WEIGHTS, CHEST_WEIGHTS):
        assert weights.shape == (9,)
        assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("gate_gain", [None, 1e4])
def test_matches_server_formulas(gate_gain: float | None) -> None:
    rng = np.random.default_rng(0)
    xy, conf = _random_people(rng, 1000)
    organs = compute_organs(xy, conf, gate_gain=gate_gain)
    assert organs.shape == (1000, 2, 4)
    for i in range(xy.shape[0]):
        ref = _server_organs(xy[i], conf[i])
        brain, chest = organs[i]
        np.testing.assert_allclose(brain[:2], ref["brain"]["point"], atol=1e-6)
        assert ref["brain"]["radius"] <= brain[2] < ref["brain"]["radius"] + 1  # server truncates to int
        assert brain[3] == pytest.approx(ref["brain"]["confidence"], abs=1e-9)
        np.testing.assert_allclose(chest[:2], ref["chest"]["point"], atol=1e-6)
        assert ref["chest"]["radius"] <= chest[2] < ref["chest"]["radius"] + 1
        # decision O2: chest confidence is divided by 7 instead of 5
        assert chest[3] == pytest.approx(ref["chest"]["confidence"] * 5.0 / 7.0, abs=1e-9)


def _face_case(face_conf: list[float]) -> np.ndarray:
    xy = np.array([[50, 60], [40, 50], [60, 50], [25, 55], [75, 55],
                   [20, 100], [80, 100], [30, 200], [70, 200]], dtype=np.float64)
    conf = np.full(9, 0.9)
    conf[:5] = face_conf
    return compute_organs(xy, conf, gate_gain=None), compute_organs(xy, conf), xy


@pytest.mark.parametrize(
    ("face_conf", "ear_branch"),
    [
        ([0.9, 0.9, 0.9, 0.9, 0.9], False),  # all five ok -> full
        ([0.9, 0.9, 0.9, 0.005, 0.9], False),  # an ear below threshold -> full
        ([0.005, 0.9, 0.9, 0.9, 0.9], True),  # nose below, ears ok -> ear branch
        ([0.9, 0.9, 0.0, 0.9, 0.9], True),  # an eye below, ears ok -> ear branch
        ([0.9, 0.9, 0.9, BRAIN_EAR_THRESHOLD, BRAIN_EAR_THRESHOLD], False),  # exactly on threshold = ok
        ([BRAIN_EAR_THRESHOLD - 1e-3, 0.9, 0.9, 0.9, 0.9], True),  # just below the gate's ramp
    ],
)
def test_brain_branches(face_conf: list[float], ear_branch: bool) -> None:
    hard, soft, xy = _face_case(face_conf)
    expected = 0.5 * (xy[3] + xy[4]) if ear_branch else BRAIN_FULL_WEIGHTS @ xy
    np.testing.assert_allclose(hard[0, :2], expected)
    np.testing.assert_allclose(soft, hard)  # the soft gate equals the hard one away from the ramp


def test_translation_and_scale_equivariance() -> None:
    rng = np.random.default_rng(1)
    xy, conf = _random_people(rng, 50)
    base = compute_organs(xy, conf)
    moved = compute_organs(xy * 2.5 + np.array([13.0, -7.0]), conf)
    np.testing.assert_allclose(moved[..., :2], base[..., :2] * 2.5 + np.array([13.0, -7.0]), atol=1e-9)
    np.testing.assert_allclose(moved[..., 2], base[..., 2] * 2.5, atol=1e-9)
    np.testing.assert_allclose(moved[..., 3], base[..., 3])


def test_left_right_symmetry() -> None:
    rng = np.random.default_rng(2)
    xy, conf = _random_people(rng, 50)
    flip = build_flip_index()
    assert len(FLIP_PAIRS) == 4
    np.testing.assert_allclose(compute_organs(xy[:, flip], conf[:, flip]), compute_organs(xy, conf), atol=1e-9)


@pytest.mark.parametrize("value", [0.0, 1.0, 1.0 + 5e-4])
def test_confidence_range(value: float) -> None:
    xy = np.random.default_rng(3).normal(0, 50, size=(9, 2))
    organs = compute_organs(xy, np.full(9, value))
    assert np.all(organs[:, 3] >= 0.0) and np.all(organs[:, 3] <= 1.0)
    rng = np.random.default_rng(4)
    many = compute_organs(rng.normal(0, 50, size=(500, 9, 2)), rng.uniform(0, 1, size=(500, 9)))
    assert many[..., 3].min() >= 0.0 and many[..., 3].max() <= 1.0


def test_organs_to_dict() -> None:
    out = organs_to_dict(np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.25]]))
    assert out["brain"] == {"point": (1.0, 2.0), "radius": 3.0, "confidence": 0.5}
    assert out["chest"]["radius"] == 6.0 and out["chest"]["confidence"] == 0.25


def test_gate_ramp_rows() -> None:
    from xtx4.utils.organs import GATE_GAIN, gate_ramp_rows

    conf = np.full((4, 9), 0.9)
    conf[1, 0] = BRAIN_EAR_THRESHOLD - 0.5 / GATE_GAIN  # inside the ramp
    conf[2, 4] = BRAIN_EAR_THRESHOLD  # on the threshold
    conf[3, 7] = BRAIN_EAR_THRESHOLD  # a torso point never matters
    assert gate_ramp_rows(conf).tolist() == [False, True, True, False]
