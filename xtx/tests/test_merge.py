"""Cross-crop merge priority A > B > C (spec section 5.2, acceptance 4)."""

from __future__ import annotations

import numpy as np

from xtx.engine.merge import Detection, MergeParams, merge_crops


def _det(box: tuple[float, float, float, float], score: float, crop: str) -> Detection:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = cx
    kpts[:, 1] = cy
    kpts[:, 2] = 1.0
    return Detection(
        bbox=np.array(box, dtype=np.float32),
        score=score,
        keypoints=kpts,
        visibility=np.full((17,), 2, dtype=np.int64),
        source_crop=crop,
    )


def test_priority_keeps_a_over_b() -> None:
    p1_a = _det((100, 100, 200, 300), 0.9, "A")
    p1_b = _det((102, 101, 201, 301), 0.95, "B")  # same person, higher score but B
    by_crop = {"A": [p1_a], "B": [p1_b], "C": []}
    merged = merge_crops(by_crop, params=MergeParams(iou_match_thr=0.5, oks_match_thr=0.5, conf_threshold=0.1))
    assert len(merged) == 1
    assert merged[0].source_crop == "A"  # A wins despite B's higher score


def test_b_added_when_a_missed() -> None:
    p1_a = _det((100, 100, 200, 300), 0.9, "A")
    p2_b = _det((600, 600, 700, 800), 0.8, "B")  # different person, far away
    by_crop = {"A": [p1_a], "B": [p2_b], "C": []}
    merged = merge_crops(by_crop, params=MergeParams(conf_threshold=0.1))
    crops = sorted(d.source_crop for d in merged)
    assert crops == ["A", "B"]


def test_c_only_when_a_and_b_missed() -> None:
    p1_a = _det((100, 100, 200, 300), 0.9, "A")
    p2_b = _det((600, 600, 700, 800), 0.8, "B")
    p2_c = _det((602, 601, 701, 801), 0.85, "C")  # duplicate of B's person
    p3_c = _det((900, 900, 980, 1050), 0.7, "C")  # unique central far person
    by_crop = {"A": [p1_a], "B": [p2_b], "C": [p2_c, p3_c]}
    merged = merge_crops(by_crop, params=MergeParams(conf_threshold=0.1))
    crops = sorted(d.source_crop for d in merged)
    assert crops == ["A", "B", "C"]  # C duplicate dropped, unique C kept


def test_conf_threshold_filters() -> None:
    low = _det((100, 100, 200, 300), 0.05, "A")
    by_crop = {"A": [low], "B": [], "C": []}
    merged = merge_crops(by_crop, params=MergeParams(conf_threshold=0.25))
    assert merged == []


def test_max_detections_cap() -> None:
    dets = [_det((10 + i * 50, 10, 40 + i * 50, 80), 0.5 + i * 0.001, "A") for i in range(10)]
    by_crop = {"A": dets, "B": [], "C": []}
    merged = merge_crops(by_crop, params=MergeParams(conf_threshold=0.1, max_detections=5))
    assert len(merged) == 5
