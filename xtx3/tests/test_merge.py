"""Cross-view merge tests: priority view 1 > 2 > 3 for the pyramid, and the
whole-frame degenerate case (view 0, confidence filter only)."""

from __future__ import annotations

import numpy as np

from xtx3.engine.merge import Detection, MergeParams, merge_views
from xtx3.utils.keypoints import NUM_KEYPOINTS


def _detection(
    *,
    x: float = 100.0,
    y: float = 100.0,
    size: float = 50.0,
    score: float = 0.9,
    view: int = 1,
    kpt_offset: float = 0.0,
) -> Detection:
    kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    kpts[:, 0] = np.linspace(x, x + size, NUM_KEYPOINTS) + kpt_offset
    kpts[:, 1] = np.linspace(y, y + size, NUM_KEYPOINTS) + kpt_offset
    kpts[:, 2] = 0.9
    return Detection(
        bbox=np.array([x, y, x + size, y + size], dtype=np.float32),
        score=score,
        keypoints=kpts,
        visibility=np.full(NUM_KEYPOINTS, 2, dtype=np.int64),
        source_view=view,
    )


class TestPriority:
    def test_view1_wins_over_duplicates(self) -> None:
        """The same person seen by all 3 pyramid views keeps only view 1."""

        by_view = {
            1: [_detection(view=1, score=0.7)],
            2: [_detection(view=2, score=0.95, kpt_offset=1.0)],
            3: [_detection(view=3, score=0.99, kpt_offset=-1.0)],
        }
        merged = merge_views(by_view, params=MergeParams())
        assert len(merged) == 1
        assert merged[0].source_view == 1

    def test_view2_wins_over_view3(self) -> None:
        by_view = {
            1: [],
            2: [_detection(view=2, score=0.6)],
            3: [_detection(view=3, score=0.9, kpt_offset=1.0)],
        }
        merged = merge_views(by_view, params=MergeParams())
        assert len(merged) == 1
        assert merged[0].source_view == 2

    def test_unique_people_all_kept(self) -> None:
        by_view = {
            1: [_detection(x=0, y=0, view=1)],
            2: [_detection(x=300, y=300, view=2)],
            3: [_detection(x=600, y=600, view=3)],
        }
        merged = merge_views(by_view, params=MergeParams())
        assert len(merged) == 3
        assert sorted(d.source_view for d in merged) == [1, 2, 3]

    def test_no_person_twice(self) -> None:
        """Two people; one duplicated across all views, one only in view 3."""

        person_a = dict(x=100.0, y=100.0, size=60.0)
        person_b = dict(x=500.0, y=500.0, size=20.0)
        by_view = {
            1: [_detection(**person_a, view=1, score=0.8)],
            2: [_detection(**person_a, view=2, score=0.85, kpt_offset=0.5)],
            3: [
                _detection(**person_a, view=3, score=0.9, kpt_offset=-0.5),
                _detection(**person_b, view=3, score=0.7),
            ],
        }
        merged = merge_views(by_view, params=MergeParams())
        assert len(merged) == 2
        assert sorted(d.source_view for d in merged) == [1, 3]


class TestWholeFrame:
    def test_single_view_passthrough(self) -> None:
        """Whole-frame mode (view 0): merge is a confidence filter + sort."""

        by_view = {
            0: [
                _detection(x=0, y=0, view=0, score=0.9),
                _detection(x=300, y=300, view=0, score=0.5),
                _detection(x=600, y=600, view=0, score=0.1),  # below threshold
            ]
        }
        merged = merge_views(by_view, params=MergeParams(conf_threshold=0.25))
        assert len(merged) == 2
        assert [d.score for d in merged] == [0.9, 0.5]
        assert all(d.source_view == 0 for d in merged)


class TestThresholds:
    def test_conf_threshold_filters(self) -> None:
        by_view = {1: [_detection(view=1, score=0.1)], 2: [], 3: []}
        merged = merge_views(by_view, params=MergeParams(conf_threshold=0.25))
        assert merged == []

    def test_iou_match_only(self) -> None:
        """Overlapping boxes with disagreeing keypoints still match by IoU."""

        det_a = _detection(view=1)
        det_b = _detection(view=2, kpt_offset=30.0)  # keypoints far apart -> low OKS
        merged = merge_views(
            {1: [det_a], 2: [det_b], 3: []},
            params=MergeParams(oks_match_thr=0.99, iou_match_thr=0.5),
        )
        assert len(merged) == 1
        assert merged[0].source_view == 1

    def test_oks_match_only(self) -> None:
        """Same keypoints but shifted boxes still match by OKS."""

        det_a = _detection(view=1, size=50.0)
        det_b = _detection(view=2, x=125.0, size=50.0)  # box IoU < 0.5
        det_b.keypoints = det_a.keypoints.copy()  # identical keypoints -> OKS = 1
        merged = merge_views(
            {1: [det_a], 2: [det_b], 3: []},
            params=MergeParams(oks_match_thr=0.5, iou_match_thr=0.99),
        )
        assert len(merged) == 1

    def test_max_detections_cap(self) -> None:
        by_view = {
            1: [
                _detection(x=1000.0 * i, y=1000.0 * i, score=0.5 + 0.001 * i, view=1)
                for i in range(10)
            ],
            2: [],
            3: [],
        }
        merged = merge_views(by_view, params=MergeParams(max_detections=5))
        assert len(merged) == 5
        scores = [d.score for d in merged]
        assert scores == sorted(scores, reverse=True)

    def test_logistic_mode(self) -> None:
        """Logistic mode with a strongly positive IoU weight merges duplicates."""

        params = MergeParams(mode="logistic", logistic_weights=[20.0, 0.0, 0.0, 0.0, 0.0, -5.0])
        by_view = {1: [_detection(view=1)], 2: [_detection(view=2)], 3: []}
        merged = merge_views(by_view, params=params)
        assert len(merged) == 1

    def test_empty_input(self) -> None:
        assert merge_views({0: []}, params=MergeParams()) == []
        assert merge_views({1: [], 2: [], 3: []}, params=MergeParams()) == []
