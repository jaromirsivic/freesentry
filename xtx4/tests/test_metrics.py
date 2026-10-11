"""Tests for the XTX4 evaluator (CONTRACTS §9, §13)."""

from __future__ import annotations

import numpy as np
import pytest

from xtx4.engine.metrics import (
    ImageGroundTruth,
    ImagePredictions,
    coco_eval,
    evaluate,
    evaluate_detailed,
    oks_matrix,
)
from xtx4.utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS

K = NUM_KEYPOINTS


def _person(cx: float, cy: float, scale: float = 40.0, v: float = 2.0) -> np.ndarray:
    """A plausible 9-point head/torso pose centred at (cx, cy)."""

    offsets = np.array(
        [[0, -1.0], [-0.2, -1.1], [0.2, -1.1], [-0.4, -1.0], [0.4, -1.0],
         [-0.8, 0.0], [0.8, 0.0], [-0.6, 1.5], [0.6, 1.5]]
    )
    kp = np.zeros((K, 3))
    kp[:, 0] = cx + offsets[:, 0] * scale
    kp[:, 1] = cy + offsets[:, 1] * scale
    kp[:, 2] = v
    return kp


def _gt(persons: list[np.ndarray], *, crowd: list[bool] | None = None, area: float = 6400.0, **kw) -> ImageGroundTruth:
    kps = np.stack(persons) if persons else np.zeros((0, K, 3))
    return ImageGroundTruth(
        keypoints=kps,
        areas=np.full(len(persons), area),
        iscrowd=None if crowd is None else np.asarray(crowd),
        **kw,
    )


def _pred(persons: list[np.ndarray], scores: list[float]) -> ImagePredictions:
    if not persons:
        return ImagePredictions(keypoints=np.zeros((0, K, 3)), scores=np.zeros(0))
    kps = np.stack(persons).copy()
    kps[:, :, 2] = 1.0
    return ImagePredictions(keypoints=kps, scores=np.asarray(scores, dtype=np.float64))


def test_perfect_single_detection() -> None:
    gt = _person(100, 100)
    res = evaluate([_pred([gt], [0.9])], [_gt([gt])])
    assert res["AP"] == pytest.approx(1.0)
    assert res["AP50"] == pytest.approx(1.0)
    assert res["AP75"] == pytest.approx(1.0)
    assert res["AR"] == pytest.approx(1.0)
    assert sorted(res["per_threshold"]) == list(range(50, 100, 5))
    assert all(isinstance(k, int) for k in res["per_threshold"])


def test_duplicate_detection_is_one_tp_one_fp() -> None:
    gt = _person(100, 100)
    dup = gt.copy()
    dup[:, :2] += 0.5
    preds = [_pred([gt, dup], [0.9, 0.8])]
    gts = [_gt([gt])]
    det = evaluate_detailed(preds, gts, conf_threshold=0.25)
    assert det["detection"]["tp"] == 1
    assert det["detection"]["fp"] == 1
    assert det["detection"]["fp_duplicate"] == 1
    assert det["duplicates_per_image"] == pytest.approx(1.0)
    res = evaluate(preds, gts)
    # The FP comes after the TP in score order, so AP stays 1 (precision at recall 1 is 1).
    assert res["AP"] == pytest.approx(1.0)
    # Below conf_threshold the duplicate disappears.
    det_hi = evaluate_detailed(preds, gts, conf_threshold=0.85)
    assert det_hi["detection"]["fp"] == 0 and det_hi["duplicates_per_image"] == 0.0


def test_empty_image_is_ignored() -> None:
    gt = _person(100, 100)
    base = evaluate([_pred([gt], [0.9])], [_gt([gt])])
    with_empty = evaluate([_pred([gt], [0.9]), _pred([], [])], [_gt([gt]), _gt([])])
    assert with_empty["AP"] == pytest.approx(base["AP"])
    assert with_empty["AR"] == pytest.approx(base["AR"])
    assert with_empty["num_gt"] == 1
    det = evaluate_detailed([_pred([], [])], [_gt([])])
    assert det["detection"]["tp"] == 0 and det["detection"]["fp"] == 0


def test_gt_without_detections_has_zero_recall() -> None:
    res = evaluate([_pred([], [])], [_gt([_person(50, 50)])])
    assert res["AR"] == 0.0 and res["AP"] == 0.0
    det = evaluate_detailed([_pred([], [])], [_gt([_person(50, 50)])])
    assert det["detection"]["recall"] == 0.0
    assert det["detection"]["fn"] == 1


def test_crowd_gt_is_ignore_region() -> None:
    person = _person(100, 100)
    crowd = _person(300, 300)
    preds = [_pred([person, crowd, crowd + np.array([0.3, 0.3, 0])], [0.9, 0.8, 0.7])]
    gts = [_gt([person, crowd], crowd=[False, True])]
    res = evaluate(preds, gts)
    assert res["num_gt"] == 1
    assert res["AP"] == pytest.approx(1.0)  # crowd matches are neither TP nor FP
    det = evaluate_detailed(preds, gts)
    assert det["detection"]["tp"] == 1
    assert det["detection"]["fp"] == 0
    assert det["detection"]["ignored_dets"] == 2


def test_overlapping_persons_matched_to_distinct_detections() -> None:
    a = _person(100, 100)
    b = _person(112, 100)  # heavy overlap
    preds = [_pred([b, a], [0.9, 0.8])]
    gts = [_gt([a, b])]
    res = evaluate(preds, gts)
    assert res["AP"] == pytest.approx(1.0)
    det = evaluate_detailed(preds, gts)
    assert det["detection"]["tp"] == 2 and det["detection"]["fp"] == 0


def test_pck_counts_missed_person_as_zero() -> None:
    a = _person(100, 100)
    b = _person(400, 100)
    det = evaluate_detailed([_pred([a], [0.9])], [_gt([a, b])])
    assert det["pck"]["face@0.1"] == pytest.approx(0.5)
    assert det["pck"]["torso@0.1"] == pytest.approx(0.5)
    assert det["detection"]["recall"] == pytest.approx(0.5)


def test_detailed_buckets_and_visibility() -> None:
    a = _person(100, 100, scale=40)
    b = _person(5, 60, scale=4)  # tiny face, at the left edge
    pred = _pred([a, b], [0.9, 0.8])
    pred.visibility = np.full((2, K), 2, dtype=np.int64)
    gt = _gt([a, b], image_size=(640, 480), net_scale=0.5)
    det = evaluate_detailed([pred], [gt])
    assert det["visibility"]["accuracy"] == pytest.approx(1.0)
    assert det["face_spread_net"][">=32"]["gt"] + det["face_spread_net"]["16-32"]["gt"] == 1
    assert det["face_spread_net"]["<8"]["gt"] == 1
    assert det["edge"]["edge"]["gt"] == 1 and det["edge"]["interior"]["gt"] == 1
    assert det["crowd"]["2-4"]["images"] == 1


def _naive_oks(dt: np.ndarray, gt: np.ndarray, areas: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    out = np.zeros((dt.shape[0], gt.shape[0]))
    var = (2 * sigmas.astype(np.float64)) ** 2
    for d in range(dt.shape[0]):
        for g in range(gt.shape[0]):
            vis = gt[g, :, 2] > 0
            if not vis.any():
                continue
            d2 = (dt[d, :, 0] - gt[g, :, 0]) ** 2 + (dt[d, :, 1] - gt[g, :, 1]) ** 2
            e = d2 / var / (areas[g] + np.spacing(1)) / 2
            out[d, g] = np.exp(-e[vis]).mean()
    return out


def test_vectorised_oks_matches_naive_loop() -> None:
    rng = np.random.default_rng(0)
    dt = rng.uniform(0, 100, (13, K, 3))
    gt = rng.uniform(0, 100, (7, K, 3))
    gt[:, :, 2] = rng.integers(-1, 3, (7, K))
    gt[0, :, 2] = 0  # no labeled points
    areas = rng.uniform(100, 5000, 7)
    fast = oks_matrix(dt, gt, areas, sigmas=OKS_SIGMAS)
    slow = _naive_oks(dt, gt, areas, OKS_SIGMAS)
    np.testing.assert_allclose(fast, slow, atol=1e-9, rtol=0)


def _random_set(seed: int, num_images: int = 60):
    rng = np.random.default_rng(seed)
    preds, gts = [], []
    for _ in range(num_images):
        n = int(rng.integers(0, 5))
        persons = [_person(*rng.uniform(60, 500, 2), scale=float(rng.uniform(8, 60))) for _ in range(n)]
        for p in persons:
            p[:, 2] = rng.choice([0, 1, 2], K, p=[0.2, 0.2, 0.6])
        crowd = [bool(rng.random() < 0.08) for _ in range(n)]
        areas = np.array([(np.ptp(p[:, 0]) * 1.5 + 4) * (np.ptp(p[:, 1]) * 1.5 + 4) for p in persons]) if n else np.zeros(0)
        gts.append(ImageGroundTruth(
            keypoints=np.stack(persons) if n else np.zeros((0, K, 3)), areas=areas, iscrowd=np.asarray(crowd, bool),
        ))
        dets, scores = [], []
        for p in persons:
            for _ in range(int(rng.integers(0, 3))):
                noise = rng.normal(0, rng.uniform(0.5, 6), (K, 2))
                q = p.copy()
                q[:, :2] += noise
                dets.append(q)
                scores.append(float(rng.uniform(0.05, 1)))
        for _ in range(int(rng.integers(0, 4))):
            dets.append(_person(*rng.uniform(0, 600, 2), scale=float(rng.uniform(8, 60))))
            scores.append(float(rng.uniform(0, 0.6)))
        preds.append(_pred(dets, scores))
    return preds, gts


def test_pycocotools_cross_check() -> None:
    pytest.importorskip("pycocotools")
    preds, gts = _random_set(1)
    ours = evaluate(preds, gts, max_det=20)
    ref = coco_eval(preds, gts, list(range(1, len(gts) + 1)), max_det=20)
    for key in ("AP", "AP50", "AP75", "AR"):
        assert abs(ours[key] - ref[key]) < 0.005, (key, ours[key], ref[key])


# ------------------------------------------------------------ sweep / operating points


def test_sweep_monotone_and_consistent_with_detailed() -> None:
    from xtx4.engine.metrics import confidence_sweep

    preds, gts = _random_set(3, 80)
    ths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = confidence_sweep(preds, gts, thresholds=ths)
    assert [r["threshold"] for r in rows] == ths
    recalls = [r["recall"] for r in rows]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))
    fps = [r["fp_per_image"] for r in rows]
    assert all(a >= b for a, b in zip(fps, fps[1:]))
    for row in rows:
        det = evaluate_detailed(preds, gts, conf_threshold=row["threshold"])
        assert row["tp"] == det["detection"]["tp"]
        assert row["fp"] == det["detection"]["fp"]
        assert row["fp_duplicate"] == det["detection"]["fp_duplicate"]
        assert row["fp_localisation"] == det["detection"]["fp_localisation"]
        assert row["fp_background"] == det["detection"]["fp_background"]
        assert row["duplicates_per_image"] == pytest.approx(det["duplicates_per_image"])
        assert row["f1"] == pytest.approx(det["detection"]["f1"])


def test_f1_max_selection() -> None:
    from xtx4.engine.metrics import confidence_sweep, select_operating_points

    a, b = _person(100, 100), _person(300, 100)
    far = _person(500, 400)
    # TP 0.9, background FP 0.8, TP 0.4, background FP 0.3
    preds = [_pred([a, far, b, far + np.array([40.0, 0, 0])], [0.9, 0.8, 0.4, 0.3])]
    gts = [_gt([a, b])]
    rows = confidence_sweep(preds, gts, thresholds=[0.2, 0.35, 0.5, 0.85])
    f1 = {r["threshold"]: r["f1"] for r in rows}
    assert f1[0.85] == pytest.approx(2 * 1.0 * 0.5 / 1.5)  # P=1, R=0.5
    assert f1[0.35] == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1))  # P=2/3, R=1 -> 0.8
    points = select_operating_points(rows)
    assert points["f1_max"]["threshold"] == 0.35
    assert points["f1_max"]["f1"] == pytest.approx(max(f1.values()))
    assert points["fp_match"] is None


def test_fp_match_selection() -> None:
    from xtx4.engine.metrics import select_operating_points

    rows = [
        {"threshold": 0.1, "f1": 0.5, "fp_per_image": 2.0},
        {"threshold": 0.2, "f1": 0.6, "fp_per_image": 1.0},
        {"threshold": 0.3, "f1": 0.6, "fp_per_image": 0.5},
        {"threshold": 0.4, "f1": 0.4, "fp_per_image": 0.1},
    ]
    points = select_operating_points(rows, ref_fp_per_image=0.6)
    assert points["fp_match"]["threshold"] == 0.3
    assert points["fp_match"]["ref_fp_per_image"] == 0.6
    assert points["f1_max"]["threshold"] == 0.2  # tie on F1 -> lower threshold
    tie = select_operating_points(rows, ref_fp_per_image=0.75)  # |1.0-0.75| == |0.5-0.75|
    assert tie["fp_match"]["threshold"] == 0.3  # tie -> higher threshold


def test_visibility_calibration_recovers_thresholds() -> None:
    from xtx4.engine.metrics import calibrate_visibility

    rng = np.random.default_rng(7)
    preds, gts = [], []
    for _ in range(40):
        p = _person(*rng.uniform(80, 400, 2))
        v = rng.choice([0, 1, 2], K)
        p[:, 2] = v
        present = np.where(v > 0, rng.uniform(0.75, 1.0, K), rng.uniform(0.0, 0.65, K))
        visible = np.where(v == 2, rng.uniform(0.35, 1.0, K), rng.uniform(0.0, 0.25, K))
        pred = _pred([p], [0.9])
        pred.present_prob = present[None]
        pred.visible_prob = visible[None]
        if (v > 0).any():
            preds.append(pred)
            gts.append(_gt([p]))
    res = calibrate_visibility(preds, gts, conf_threshold=0.25)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["present_threshold"] == pytest.approx(0.7)
    assert res["visible_threshold"] == pytest.approx(0.3)
    assert res["default@0.5"]["macro_f1"] < 1.0
    assert res["points"] == sum(int((g.keypoints[0, :, 2] >= 0).sum()) for g in gts)
    # Without probabilities there is nothing to calibrate.
    plain = [_pred([g.keypoints[0]], [0.9]) for g in gts]
    assert calibrate_visibility(plain, gts) is None


# ------------------------------------------------------------ ncnn code path of scripts/evaluate.py


def test_ncnn_eval_path_matches_onnxruntime() -> None:
    """``--model-kind ncnn`` (scripts/evaluate.py NcnnRunner) vs onnxruntime on the random-weight
    ``bench/graphs/u_b_320`` export: shapes, scores, frame-px keypoints and probabilities."""

    import sys
    from pathlib import Path

    pytest.importorskip("ncnn")
    ort = pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parents[1]
    graphs = root / "bench" / "graphs"
    param, bin_path, onnx_path = (graphs / f"u_b_320{ext}" for ext in (".ncnn.param", ".ncnn.bin", ".onnx"))
    if not (param.is_file() and bin_path.is_file() and onnx_path.is_file()):
        pytest.skip("bench/graphs/u_b_320 export not present")
    sys.path.insert(0, str(root / "scripts"))
    try:
        import evaluate as ev
    finally:
        sys.path.remove(str(root / "scripts"))
    from xtx4.runtime import decode as rdec
    from xtx4.runtime.preprocess import prepare_views

    runner = ev.NcnnRunner(size=320, param=param, bin_path=bin_path, num_threads=2)
    num_anchors = runner.runner.num_anchors
    assert num_anchors == (320 // 8) ** 2 + (320 // 16) ** 2 + (320 // 32) ** 2
    image = np.random.default_rng(0).uniform(0, 255, (360, 500, 3)).astype(np.uint8)
    pred = runner.predict_one(image, conf=0.0, max_det=num_anchors)
    keep = runner.last_keep
    assert pred.keypoints.shape == (num_anchors, K, 3)
    assert pred.present_prob.shape == pred.visible_prob.shape == pred.visibility.shape == (num_anchors, K)
    assert np.all(np.diff(pred.scores) <= 0)

    prepared = prepare_views(image, network_size=320, mode="whole")
    (key, view), = prepared.views.items()
    blob = view.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    person, pose = sess.run(None, {sess.get_inputs()[0].name: blob})[:2]
    assert person.shape == (1, num_anchors, 1) and pose.shape == (1, num_anchors, K * 5)
    nc_person, nc_pose = runner.runner.forward(view)
    assert float(np.abs(nc_person.reshape(-1) - person.reshape(-1)).max()) < 0.05
    assert float(np.abs(nc_pose.reshape(-1) - pose.reshape(-1)).max()) < 0.05

    scores = rdec.person_scores(person)
    kp, vis, sig = rdec.decode_selected(pose, keep, network_size=320, strides=runner.runner.strides, num_keypoints=K)
    ref = rdec.to_frame_poses(scores[keep], kp, vis, sig, transform=prepared.transforms[key], anchor_indices=keep)
    ref_xy = np.stack([p.keypoints for p in ref])[..., :2]
    raw = pose.reshape(num_anchors, K, 5)[keep]
    np.testing.assert_allclose(pred.scores, scores[keep], atol=1e-3)
    np.testing.assert_allclose(pred.keypoints[..., :2], ref_xy, atol=0.05)
    np.testing.assert_allclose(pred.present_prob, 1 / (1 + np.exp(-raw[..., 3])), atol=1e-3)
    np.testing.assert_allclose(pred.visible_prob, 1 / (1 + np.exp(-raw[..., 4])), atol=1e-3)
    # The public runner path (NMS off) returns the same detections.
    api = runner.api_reference(image, conf=0.0, max_det=num_anchors)
    np.testing.assert_allclose(api.keypoints, pred.keypoints, atol=1e-5)
