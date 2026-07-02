"""Cross-level merge tests: priority level_1 > level_2 > level_3 (spec section 5)."""

from __future__ import annotations

import numpy as np

from xtx2.engine.merge import Detection, MergeParams, merge_levels


def _detection(
    *,
    x: float = 100.0,
    y: float = 100.0,
    size: float = 50.0,
    score: float = 0.9,
    level: int = 1,
    kpt_offset: float = 0.0,
) -> Detection:
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = np.linspace(x, x + size, 17) + kpt_offset
    kpts[:, 1] = np.linspace(y, y + size, 17) + kpt_offset
    kpts[:, 2] = 0.9
    return Detection(
        bbox=np.array([x, y, x + size, y + size], dtype=np.float32),
        score=score,
        keypoints=kpts,
        visibility=np.full(17, 2, dtype=np.int64),
        source_level=level,
    )


class TestPriority:
    def test_level1_wins_over_duplicates(self) -> None:
        """The same person seen by all 3 levels keeps only the level_1 detection."""

        by_level = {
            1: [_detection(level=1, score=0.7)],
            2: [_detection(level=2, score=0.95, kpt_offset=1.0)],
            3: [_detection(level=3, score=0.99, kpt_offset=-1.0)],
        }
        merged = merge_levels(by_level, params=MergeParams())
        assert len(merged) == 1
        assert merged[0].source_level == 1

    def test_level2_wins_over_level3(self) -> None:
        by_level = {
            1: [],
            2: [_detection(level=2, score=0.6)],
            3: [_detection(level=3, score=0.9, kpt_offset=1.0)],
        }
        merged = merge_levels(by_level, params=MergeParams())
        assert len(merged) == 1
        assert merged[0].source_level == 2

    def test_unique_people_all_kept(self) -> None:
        by_level = {
            1: [_detection(x=0, y=0, level=1)],
            2: [_detection(x=300, y=300, level=2)],
            3: [_detection(x=600, y=600, level=3)],
        }
        merged = merge_levels(by_level, params=MergeParams())
        assert len(merged) == 3
        assert sorted(d.source_level for d in merged) == [1, 2, 3]

    def test_no_person_twice(self) -> None:
        """Two people; one duplicated across all levels, one only in level_3."""

        person_a = dict(x=100.0, y=100.0, size=60.0)
        person_b = dict(x=500.0, y=500.0, size=20.0)
        by_level = {
            1: [_detection(**person_a, level=1, score=0.8)],
            2: [_detection(**person_a, level=2, score=0.85, kpt_offset=0.5)],
            3: [
                _detection(**person_a, level=3, score=0.9, kpt_offset=-0.5),
                _detection(**person_b, level=3, score=0.7),
            ],
        }
        merged = merge_levels(by_level, params=MergeParams())
        assert len(merged) == 2
        levels = sorted(d.source_level for d in merged)
        assert levels == [1, 3]


class TestThresholds:
    def test_conf_threshold_filters(self) -> None:
        by_level = {
            1: [_detection(level=1, score=0.1)],
            2: [],
            3: [],
        }
        merged = merge_levels(by_level, params=MergeParams(conf_threshold=0.25))
        assert merged == []

    def test_iou_match_only(self) -> None:
        """Overlapping boxes with disagreeing keypoints still match by IoU."""

        det_a = _detection(level=1)
        det_b = _detection(level=2, kpt_offset=30.0)  # keypoints far apart -> low OKS
        merged = merge_levels(
            {1: [det_a], 2: [det_b], 3: []},
            params=MergeParams(oks_match_thr=0.99, iou_match_thr=0.5),
        )
        assert len(merged) == 1
        assert merged[0].source_level == 1

    def test_oks_match_only(self) -> None:
        """Same keypoints but shifted boxes still match by OKS."""

        det_a = _detection(level=1, size=50.0)
        det_b = _detection(level=2, x=125.0, size=50.0)  # box IoU < 0.5
        det_b.keypoints = det_a.keypoints.copy()  # identical keypoints -> OKS = 1
        merged = merge_levels(
            {1: [det_a], 2: [det_b], 3: []},
            params=MergeParams(oks_match_thr=0.5, iou_match_thr=0.99),
        )
        assert len(merged) == 1

    def test_max_detections_cap(self) -> None:
        by_level = {
            1: [
                _detection(x=1000.0 * i, y=1000.0 * i, score=0.5 + 0.001 * i, level=1)
                for i in range(10)
            ],
            2: [],
            3: [],
        }
        merged = merge_levels(by_level, params=MergeParams(max_detections=5))
        assert len(merged) == 5
        # Highest scores kept, sorted descending.
        scores = [d.score for d in merged]
        assert scores == sorted(scores, reverse=True)

    def test_logistic_mode(self) -> None:
        """Logistic mode with a strongly positive IoU weight merges duplicates."""

        params = MergeParams(mode="logistic", logistic_weights=[20.0, 0.0, 0.0, 0.0, 0.0, -5.0])
        by_level = {
            1: [_detection(level=1)],
            2: [_detection(level=2)],
            3: [],
        }
        merged = merge_levels(by_level, params=params)
        assert len(merged) == 1

    def test_empty_input(self) -> None:
        assert merge_levels({1: [], 2: [], 3: []}, params=MergeParams()) == []
