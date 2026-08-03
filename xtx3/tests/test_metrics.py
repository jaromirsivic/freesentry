"""Metric tests: OKS, AP/AR on constructed cases, and the detailed report
(detection P/R, per-landmark OKS, visibility accuracy, scale buckets)."""

from __future__ import annotations

import numpy as np

from xtx3.engine.metrics import (
    ImageGroundTruth,
    ImagePredictions,
    compute_oks,
    evaluate,
    evaluate_detailed,
    per_landmark_oks_components,
)
from xtx3.utils.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


def _person(x: float = 100.0, y: float = 100.0, spread: float = 80.0) -> np.ndarray:
    kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
    kpts[:, 0] = np.linspace(x, x + spread, NUM_KEYPOINTS)
    kpts[:, 1] = np.linspace(y, y + spread, NUM_KEYPOINTS)
    kpts[:, 2] = 2
    return kpts


def _gt(*people: np.ndarray, area: float = 10000.0) -> ImageGroundTruth:
    if not people:
        return ImageGroundTruth(
            keypoints=np.zeros((0, NUM_KEYPOINTS, 3)), areas=np.zeros((0,))
        )
    return ImageGroundTruth(
        keypoints=np.stack(people), areas=np.full(len(people), area)
    )


def _pred(*people: np.ndarray, scores: list[float] | None = None) -> ImagePredictions:
    if not people:
        return ImagePredictions(keypoints=np.zeros((0, NUM_KEYPOINTS, 3)), scores=np.zeros((0,)))
    kpts = np.stack(people).copy()
    kpts[:, :, 2] = 0.9  # prediction channel 2 is confidence, not GT visibility
    return ImagePredictions(
        keypoints=kpts,
        scores=np.array(scores if scores is not None else [0.9] * len(people)),
    )


class TestOKS:
    def test_perfect_match_is_one(self) -> None:
        person = _person()
        assert compute_oks(person, 10000.0, person) == 1.0

    def test_distance_decreases_oks(self) -> None:
        person = _person()
        shifted = person.copy()
        shifted[:, 0] += 20.0
        assert compute_oks(person, 10000.0, shifted) < compute_oks(person, 10000.0, person)

    def test_unlabelled_points_ignored(self) -> None:
        person = _person()
        person[3:, 2] = 0  # only first 3 labelled
        detection = person.copy()
        detection[3:, 0] += 500.0  # wildly wrong on unlabelled points
        assert compute_oks(person, 10000.0, detection) == 1.0

    def test_per_landmark_components(self) -> None:
        person = _person()
        person[0, 2] = 0  # nose unlabelled
        components = per_landmark_oks_components(person, 10000.0, person)
        assert components.shape == (NUM_KEYPOINTS,)
        assert np.isnan(components[0])
        np.testing.assert_allclose(components[1:], 1.0)


class TestEvaluate:
    def test_perfect_predictions(self) -> None:
        person = _person()
        metrics = evaluate([_pred(person)], [_gt(person)])
        assert metrics["AP"] == 1.0
        assert metrics["AP50"] == 1.0
        assert metrics["AR"] == 1.0

    def test_missed_person_halves_recall(self) -> None:
        person_a, person_b = _person(), _person(x=600.0)
        metrics = evaluate([_pred(person_a)], [_gt(person_a, person_b)])
        assert metrics["AR"] == 0.5

    def test_false_positive_lowers_precision(self) -> None:
        person = _person()
        ghost = _person(x=900.0)
        metrics = evaluate(
            [_pred(person, ghost, scores=[0.9, 0.95])], [_gt(person)]
        )
        assert metrics["AP"] < 1.0
        assert metrics["AR"] == 1.0

    def test_empty_everything(self) -> None:
        metrics = evaluate([_pred()], [_gt()])
        assert metrics["AP"] == 0.0


class TestDetailedReport:
    def test_report_structure_and_counts(self) -> None:
        person_a, person_b = _person(), _person(x=600.0)
        ghost = _person(x=1200.0)
        report = evaluate_detailed(
            [_pred(person_a, ghost, scores=[0.9, 0.8])], [_gt(person_a, person_b)]
        )
        det = report["detection"]
        assert det["tp"] == 1 and det["fp"] == 1 and det["fn"] == 1
        assert det["precision"] == 0.5
        assert det["recall"] == 0.5
        assert set(report["per_landmark"].keys()) == set(KEYPOINT_NAMES)
        assert report["per_landmark"]["nose"]["mean_oks"] == 1.0
        for group in ("face_scale", "torso_scale"):
            assert set(report[group].keys()) == {"small", "medium", "large"}

    def test_visibility_accuracy(self) -> None:
        person = _person()
        pred = _pred(person)
        pred.visibility = np.full((1, NUM_KEYPOINTS), 2, dtype=np.int64)
        pred.visibility[0, 0] = 1  # nose misclassified occluded vs visible
        report = evaluate_detailed([pred], [_gt(person)])
        assert report["per_landmark"]["nose"]["visibility_accuracy"] == 0.0
        assert report["per_landmark"]["left_eye"]["visibility_accuracy"] == 1.0
