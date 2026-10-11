"""Nine-point OKS evaluation for XTX4 (CONTRACTS §9).

Why this module exists (and does not reuse XTX3's evaluator):

* XTX3 keyed its per-threshold AP dict by ``np.arange(0.5, 1.0, 0.05)`` floats, so
  ``0.75`` was never found and AP75 was always reported as 0 (defect D1). Here the
  thresholds are the integers ``range(50, 100, 5)``; AP75 is ``per_threshold[75]``.
* XTX3 built OKS matrices with a double Python loop; here they are one numpy
  broadcast ``(D, G, K)``.
* XTX3 had no notion of crowd/ignore GTs. Here matching follows ``COCOeval``
  exactly: detections are visited by descending score (stable), each takes the
  unmatched regular GT with the highest OKS ``>= t``; if there is none it may take a
  crowd GT (which can absorb any number of detections) and is then *ignored* -
  neither TP nor FP. GTs with no labeled landmark are ignore regions as well (COCO
  sets ``ignore = num_keypoints == 0``); their OKS is 0 so they never absorb anything.
* Precision is made monotone and sampled at 101 recall points exactly like
  ``COCOeval.accumulate``, so :func:`evaluate` with ``max_det=20`` reproduces
  pycocotools (:func:`coco_eval`) up to floating point noise.

:func:`evaluate_detailed` reports the operating-point diagnostics the XTX4 plan asks
for (§9 of the plan): detection P/R/F1 at a confidence threshold, duplicates,
per-landmark OKS/NME, visibility confusion, face/torso PCK and scale/crowd/edge
buckets. Quantities named ``*_net`` are in network pixels of the whole-frame
letterbox (``ImageGroundTruth.net_scale`` = network px per source px); when the
scale is unknown, source pixels are used (scale 1).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..utils.keypoints import FACE_INDICES, KEYPOINT_NAMES, NUM_KEYPOINTS, OKS_SIGMAS, TORSO_INDICES

OKS_THRESHOLDS: tuple[int, ...] = tuple(range(50, 100, 5))  # percent, integer keys
RECALL_LEVELS: np.ndarray = np.linspace(0.0, 1.0, 101)
_EPS = np.spacing(1)  # same epsilon as COCOeval.computeOks / accumulate

FACE_SPREAD_EDGES_NET: tuple[float, ...] = (8.0, 16.0, 32.0)
FACE_SPREAD_LABELS: tuple[str, ...] = ("<8", "8-16", "16-32", ">=32")
TORSO_SPREAD_EDGES_NET: tuple[float, ...] = (16.0, 32.0, 64.0)
TORSO_SPREAD_LABELS: tuple[str, ...] = ("<16", "16-32", "32-64", ">=64")
CROWD_LABELS: tuple[str, ...] = ("0", "1", "2-4", "5+")
EDGE_MARGIN_NET = 16.0
PCK_ALPHA = 0.1
PCK_PX_NET = 2.0
_LOCALISATION_FP_OKS = 0.1  # FP with max OKS in [0.1, match) = poorly localised person


@dataclass(slots=True)
class ImagePredictions:
    """Detections of one image in source pixels."""

    keypoints: np.ndarray  # (D, K, 3) [x, y, conf]
    scores: np.ndarray  # (D,)
    visibility: np.ndarray | None = None  # (D, K) int {0,1,2}
    # Optional per-landmark probabilities for visibility calibration (§2 semantics):
    present_prob: np.ndarray | None = None  # (D, K) P(labeled/present)
    visible_prob: np.ndarray | None = None  # (D, K) P(visible | present)


@dataclass(slots=True)
class ImageGroundTruth:
    """Ground truth of one image in source pixels.

    ``v = -1`` (unknown landmark) is treated as unlabeled (0) for OKS and skipped in
    the visibility confusion. ``image_size`` (w, h) enables the edge bucket and
    ``net_scale`` (network px per source px) expresses spreads/PCK@2px in network
    pixels; both are optional extensions of the §9 contract.
    """

    keypoints: np.ndarray  # (G, K, 3) [x, y, v]
    areas: np.ndarray  # (G,)
    iscrowd: np.ndarray | None = None  # (G,) bool; None = no crowd
    image_size: tuple[int, int] | None = None  # (width, height) source px
    net_scale: float | None = None


# --------------------------------------------------------------------------- OKS


def oks_matrix(
    dt_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    gt_areas: np.ndarray,
    *,
    sigmas: np.ndarray = OKS_SIGMAS,
) -> np.ndarray:
    """COCO OKS between every detection and every GT, ``(D, G)`` float64.

    ``e = d^2 / ((2 sigma)^2 * (area + eps) * 2)``, averaged over GT landmarks with
    ``v > 0``; a GT without labeled landmarks has OKS 0 to everything.
    """

    dt = np.asarray(dt_keypoints, dtype=np.float64)
    gt = np.asarray(gt_keypoints, dtype=np.float64)
    num_dt, num_gt = dt.shape[0], gt.shape[0]
    if num_dt == 0 or num_gt == 0:
        return np.zeros((num_dt, num_gt), dtype=np.float64)
    var2 = (2.0 * np.asarray(sigmas, dtype=np.float64)) ** 2 * 2.0  # (K,)
    labeled = gt[:, :, 2] > 0  # (G, K)
    d2 = (dt[:, None, :, 0] - gt[None, :, :, 0]) ** 2 + (dt[:, None, :, 1] - gt[None, :, :, 1]) ** 2
    denom = var2[None, None, :] * (np.asarray(gt_areas, dtype=np.float64)[None, :, None] + _EPS)
    comp = np.exp(-d2 / denom) * labeled[None]
    count = labeled.sum(axis=1)  # (G,)
    return comp.sum(axis=2) / np.maximum(count, 1)[None, :]


def oks_components(
    dt_keypoints: np.ndarray, gt_keypoints: np.ndarray, gt_areas: np.ndarray, *, sigmas: np.ndarray = OKS_SIGMAS
) -> np.ndarray:
    """Per-landmark OKS terms for aligned pairs ``(P, K)``; NaN where the GT is unlabeled."""

    dt = np.asarray(dt_keypoints, dtype=np.float64)
    gt = np.asarray(gt_keypoints, dtype=np.float64)
    var2 = (2.0 * np.asarray(sigmas, dtype=np.float64)) ** 2 * 2.0
    d2 = (dt[..., 0] - gt[..., 0]) ** 2 + (dt[..., 1] - gt[..., 1]) ** 2
    comp = np.exp(-d2 / (var2[None, :] * (np.asarray(gt_areas, dtype=np.float64)[:, None] + _EPS)))
    comp[gt[..., 2] <= 0] = np.nan
    return comp


# ---------------------------------------------------------------------- matching


def _gt_fields(gt: ImageGroundTruth) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kpts = np.asarray(gt.keypoints, dtype=np.float64).reshape(-1, NUM_KEYPOINTS, 3)
    areas = np.asarray(gt.areas, dtype=np.float64).reshape(-1)
    crowd = (
        np.zeros(kpts.shape[0], dtype=bool)
        if gt.iscrowd is None
        else np.asarray(gt.iscrowd, dtype=bool).reshape(-1)
    )
    has_kpts = (kpts[:, :, 2] > 0).any(axis=1)
    regular = ~crowd & has_kpts  # counted in recall
    return kpts, areas, crowd, regular


def _det_order(pred: ImagePredictions, *, max_det: int, score_threshold: float) -> np.ndarray:
    """Indices of kept detections: score >= threshold, stable descending score, top ``max_det``."""

    scores = np.asarray(pred.scores, dtype=np.float64).reshape(-1)
    keep = np.nonzero(scores >= score_threshold)[0]
    return keep[np.argsort(-scores[keep], kind="mergesort")][:max_det]


def _sorted_dets(
    pred: ImagePredictions, *, max_det: int, score_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    kpts = np.asarray(pred.keypoints, dtype=np.float64).reshape(-1, NUM_KEYPOINTS, 3)
    scores = np.asarray(pred.scores, dtype=np.float64).reshape(-1)
    vis = None if pred.visibility is None else np.asarray(pred.visibility).reshape(-1, NUM_KEYPOINTS)
    order = _det_order(pred, max_det=max_det, score_threshold=score_threshold)
    return kpts[order], scores[order], (None if vis is None else vis[order])


def _match(
    oks: np.ndarray, regular: np.ndarray, crowd: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """COCO greedy matching for all thresholds at once.

    ``oks`` (D, G) with detections already in descending score order. Returns
    ``dt_gt (T, D)`` matched GT index or -1 and ``dt_ignored (T, D)`` (matched to a
    crowd GT). The loop is over detections only and skips detections that cannot
    match anything at the lowest threshold.
    """

    num_t = thresholds.shape[0]
    num_dt, num_gt = oks.shape
    dt_gt = np.full((num_t, num_dt), -1, dtype=np.int64)
    dt_ignored = np.zeros((num_t, num_dt), dtype=bool)
    if num_dt == 0 or num_gt == 0:
        return dt_gt, dt_ignored
    reachable = (regular | crowd)[None, :] & (oks >= thresholds.min())
    candidates = np.nonzero(reachable.any(axis=1))[0]
    if candidates.size == 0:
        return dt_gt, dt_ignored
    taken = np.zeros((num_t, num_gt), dtype=bool)
    t_idx = np.arange(num_t)
    for d in candidates:
        row = oks[d]
        above = row[None, :] >= thresholds[:, None]  # (T, G)
        free = above & regular[None, :] & ~taken
        score_reg = np.where(free, row[None, :], -1.0)
        best = score_reg.argmax(axis=1)
        hit = free[t_idx, best]
        if hit.any():
            dt_gt[hit, d] = best[hit]
            taken[t_idx[hit], best[hit]] = True
        miss = ~hit
        if miss.any() and crowd.any():
            crowd_ok = above & crowd[None, :]
            score_cr = np.where(crowd_ok, row[None, :], -1.0)
            best_cr = score_cr.argmax(axis=1)
            hit_cr = miss & crowd_ok[t_idx, best_cr]
            dt_gt[hit_cr, d] = best_cr[hit_cr]
            dt_ignored[hit_cr, d] = True
    return dt_gt, dt_ignored


def _accumulate(scores: np.ndarray, tps: np.ndarray, fps: np.ndarray, num_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """COCOeval.accumulate for one category/area/maxDet: returns (AP per T, recall per T)."""

    num_t = tps.shape[0]
    if num_gt == 0:
        return np.full(num_t, np.nan), np.full(num_t, np.nan)
    order = np.argsort(-scores, kind="mergesort")
    tp_cum = np.cumsum(tps[:, order], axis=1, dtype=np.float64)
    fp_cum = np.cumsum(fps[:, order], axis=1, dtype=np.float64)
    ap = np.zeros(num_t)
    recall = np.zeros(num_t)
    nd = order.shape[0]
    for t in range(num_t):
        if nd == 0:
            continue
        rc = tp_cum[t] / num_gt
        pr = tp_cum[t] / (fp_cum[t] + tp_cum[t] + _EPS)
        recall[t] = rc[-1]
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, RECALL_LEVELS, side="left")
        q = np.where(inds < nd, pr[np.minimum(inds, nd - 1)], 0.0)
        ap[t] = q.mean()
    return ap, recall


# ---------------------------------------------------------------------- evaluate


def evaluate(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    *,
    sigmas: np.ndarray = OKS_SIGMAS,
    max_det: int = 100,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """COCO-style keypoint AP/AR over the dataset (integer thresholds 50..95)."""

    if len(preds) != len(gts):
        raise ValueError(f"preds ({len(preds)}) and gts ({len(gts)}) must have equal length")
    thresholds = np.asarray(OKS_THRESHOLDS, dtype=np.float64) / 100.0
    all_scores: list[np.ndarray] = []
    all_tp: list[np.ndarray] = []
    all_fp: list[np.ndarray] = []
    num_gt = num_det = num_ignored_gt = 0
    for pred, gt in zip(preds, gts):
        g_kpts, g_area, crowd, regular = _gt_fields(gt)
        num_gt += int(regular.sum())
        num_ignored_gt += int((~regular).sum())
        d_kpts, d_scores, _ = _sorted_dets(pred, max_det=max_det, score_threshold=score_threshold)
        if d_scores.size == 0:
            continue
        num_det += d_scores.size
        oks = oks_matrix(d_kpts, g_kpts, g_area, sigmas=sigmas)
        dt_gt, dt_ign = _match(oks, regular, crowd, thresholds)
        # COCO drops ignored detections from both TP and FP; keep the score columns
        # aligned by marking them as neither.
        all_scores.append(d_scores)
        all_tp.append((dt_gt >= 0) & ~dt_ign)
        all_fp.append((dt_gt < 0) & ~dt_ign)
    if all_scores:
        scores = np.concatenate(all_scores)
        tps = np.concatenate(all_tp, axis=1)
        fps = np.concatenate(all_fp, axis=1)
        # A detection ignored at threshold t is neither TP nor FP at t but keeps its
        # column, exactly like COCO's per-threshold dtIg (it only repeats a PR point).
    else:
        scores = np.zeros(0)
        tps = fps = np.zeros((thresholds.size, 0), dtype=bool)
    ap, recall = _accumulate(scores, tps, fps, num_gt)
    per_threshold = {t: float(v) for t, v in zip(OKS_THRESHOLDS, ap)}
    per_threshold_recall = {t: float(v) for t, v in zip(OKS_THRESHOLDS, recall)}
    return {
        "AP": float(np.mean(ap)) if num_gt else float("nan"),
        "AP50": per_threshold[50],
        "AP75": per_threshold[75],
        "AR": float(np.mean(recall)) if num_gt else float("nan"),
        "per_threshold": per_threshold,
        "per_threshold_recall": per_threshold_recall,
        "num_gt": int(num_gt),
        "num_det": int(num_det),
        "num_ignored_gt": int(num_ignored_gt),
        "num_images": len(gts),
        "max_det": int(max_det),
        "score_threshold": float(score_threshold),
    }


# -------------------------------------------------------------- detailed report


def _group_spread(kpts: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Max pairwise distance between labeled landmarks of a group, (G,); NaN if < 2 labeled."""

    pts = kpts[:, list(indices), :]
    labeled = pts[:, :, 2] > 0
    diff = pts[:, :, None, :2] - pts[:, None, :, :2]
    dist = np.sqrt((diff**2).sum(-1))
    pair_ok = labeled[:, :, None] & labeled[:, None, :]
    spread = np.where(pair_ok, dist, -1.0).max(axis=(1, 2)) if pts.shape[0] else np.zeros(0)
    spread = np.asarray(spread, dtype=np.float64)
    spread[labeled.sum(axis=1) < 2] = np.nan
    return spread


def _bucket_index(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Bucket index per value; -1 for NaN."""

    idx = np.searchsorted(np.asarray(edges), values, side="right")
    return np.where(np.isnan(values), -1, idx)


class _Acc:
    """Sum/count accumulator keyed by name."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, float]] = {}

    def add(self, key: str, **fields: float) -> None:
        slot = self.data.setdefault(key, {})
        for name, value in fields.items():
            slot[name] = slot.get(name, 0.0) + float(value)


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _macro_f1(confusion: np.ndarray) -> tuple[float, list[float]]:
    f1s = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    per_class = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c]
        den = 2 * tp + (confusion[:, c].sum() - tp) + (confusion[c, :].sum() - tp)
        per_class.append(float(2 * tp / den) if den else float("nan"))
    return (float(np.mean(f1s)) if f1s else float("nan")), per_class


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }


def evaluate_detailed(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    *,
    match_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    sigmas: np.ndarray = OKS_SIGMAS,
    max_det: int = 100,
) -> dict[str, Any]:
    """Operating-point diagnostics at ``score >= conf_threshold`` and ``OKS >= match_threshold``.

    Matching is the same COCO greedy matching as :func:`evaluate` at one threshold.
    FPs are split into ``duplicate`` (OKS >= match to a regular GT that an earlier,
    higher-scored detection already took), ``localisation`` (best OKS in
    [0.1, match)) and ``background``. ``duplicates_per_image`` = duplicate FPs /
    images. PCK is point-level over labeled group landmarks of every regular GT whose
    group spread is defined (>= 2 labeled points); points of missed persons count 0.
    """

    if len(preds) != len(gts):
        raise ValueError(f"preds ({len(preds)}) and gts ({len(gts)}) must have equal length")
    thr = np.asarray([match_threshold], dtype=np.float64)
    face_idx = np.asarray(FACE_INDICES)
    torso_idx = np.asarray(TORSO_INDICES)

    tp = fp = fp_dup = fp_loc = fp_bg = ignored = 0
    num_gt = 0
    images_with_dup = 0
    oks_sum = np.zeros(NUM_KEYPOINTS)
    oks_cnt = np.zeros(NUM_KEYPOINTS)
    nme_values: list[list[float]] = [[] for _ in range(NUM_KEYPOINTS)]
    face_nme: list[float] = []
    matched_oks: list[float] = []
    confusion = np.zeros((NUM_KEYPOINTS, 3, 3), dtype=np.int64)
    pck = _Acc()
    face_buckets = _Acc()
    torso_buckets = _Acc()
    crowd_buckets = _Acc()
    edge_buckets = _Acc()
    has_vis = False
    has_edge = False

    for pred, gt in zip(preds, gts):
        g_kpts, g_area, crowd, regular = _gt_fields(gt)
        d_kpts, d_scores, d_vis = _sorted_dets(pred, max_det=max_det, score_threshold=conf_threshold)
        scale = float(gt.net_scale) if gt.net_scale else 1.0
        n_reg = int(regular.sum())
        num_gt += n_reg

        oks = oks_matrix(d_kpts, g_kpts, g_area, sigmas=sigmas)
        dt_gt, dt_ign = _match(oks, regular, crowd, thr)
        dt_gt, dt_ign = dt_gt[0], dt_ign[0]
        is_tp = (dt_gt >= 0) & ~dt_ign
        is_fp = dt_gt < 0
        reg_oks = oks[:, regular] if n_reg else np.zeros((d_scores.size, 0))
        best_reg = reg_oks.max(axis=1) if reg_oks.shape[1] else np.zeros(d_scores.size)
        dup = is_fp & (best_reg >= match_threshold)
        loc = is_fp & ~dup & (best_reg >= _LOCALISATION_FP_OKS)
        n_tp, n_fp, n_dup = int(is_tp.sum()), int(is_fp.sum()), int(dup.sum())
        tp += n_tp
        fp += n_fp
        fp_dup += n_dup
        fp_loc += int(loc.sum())
        fp_bg += n_fp - n_dup - int(loc.sum())
        ignored += int(dt_ign.sum())
        images_with_dup += int(n_dup > 0)
        crowd_key = "0" if n_reg == 0 else "1" if n_reg == 1 else "2-4" if n_reg <= 4 else "5+"
        crowd_buckets.add(crowd_key, images=1, gt=n_reg, tp=n_tp, fp=n_fp, dup=n_dup)

        if g_kpts.shape[0] == 0:
            continue
        # GT -> detection for matched regular GTs
        gt_det = np.full(g_kpts.shape[0], -1, dtype=np.int64)
        tp_dets = np.nonzero(is_tp)[0]
        gt_det[dt_gt[tp_dets]] = tp_dets
        face_spread = _group_spread(g_kpts, FACE_INDICES) * scale
        torso_spread = _group_spread(g_kpts, TORSO_INDICES) * scale
        labeled = g_kpts[:, :, 2] > 0
        err = np.full((g_kpts.shape[0], NUM_KEYPOINTS), np.inf)
        matched = gt_det >= 0
        if matched.any():
            mg = np.nonzero(matched)[0]
            md = gt_det[mg]
            err[mg] = np.hypot(d_kpts[md, :, 0] - g_kpts[mg, :, 0], d_kpts[md, :, 1] - g_kpts[mg, :, 1])
            comp = oks_components(d_kpts[md], g_kpts[mg], g_area[mg], sigmas=sigmas)
            valid = ~np.isnan(comp)
            oks_sum += np.where(valid, comp, 0.0).sum(axis=0)
            oks_cnt += valid.sum(axis=0)
            norm = np.sqrt(np.maximum(g_area[mg], _EPS))
            for row, g in enumerate(mg):
                for k in np.nonzero(labeled[g])[0]:
                    nme_values[k].append(float(err[g, k] / norm[row]))
                fl = labeled[g, face_idx]
                if fl.sum() >= 1 and not np.isnan(face_spread[g]) and face_spread[g] > 0:
                    face_nme.append(float(err[g, face_idx][fl].mean() * scale / face_spread[g]))
            matched_oks.extend(oks[md, mg].tolist())
            if d_vis is not None:
                has_vis = True
                gv = np.rint(g_kpts[mg, :, 2]).astype(np.int64)
                pv = np.clip(np.asarray(d_vis[md], dtype=np.int64), 0, 2)
                known = gv >= 0
                for k in range(NUM_KEYPOINTS):
                    sel = known[:, k]
                    np.add.at(confusion[k], (np.clip(gv[sel, k], 0, 2), pv[sel, k]), 1)

        err_net = err * scale
        width_height = gt.image_size
        if width_height is not None:
            has_edge = True
            w, h = width_height
            margin = EDGE_MARGIN_NET / scale
            x, y = g_kpts[:, :, 0], g_kpts[:, :, 1]
            near = labeled & ((x < margin) | (y < margin) | (x > w - margin) | (y > h - margin))
            edge = near.any(axis=1)
        else:
            edge = None

        face_bi = _bucket_index(face_spread, FACE_SPREAD_EDGES_NET)
        torso_bi = _bucket_index(torso_spread, TORSO_SPREAD_EDGES_NET)
        for g in np.nonzero(regular)[0]:
            m = float(matched[g])
            moks = float(oks[gt_det[g], g]) if matched[g] else 0.0
            f_lab = labeled[g, face_idx]
            t_lab = labeled[g, torso_idx]
            f_ok = t_ok = f_px = 0.0
            if not np.isnan(face_spread[g]):
                f_ok = float((err_net[g, face_idx][f_lab] <= PCK_ALPHA * face_spread[g]).sum())
                f_px = float((err_net[g, face_idx][f_lab] <= PCK_PX_NET).sum())
                pck.add("face", correct=f_ok, total=f_lab.sum(), persons=1, correct_2px=f_px)
            if not np.isnan(torso_spread[g]):
                t_ok = float((err_net[g, torso_idx][t_lab] <= PCK_ALPHA * torso_spread[g]).sum())
                pck.add("torso", correct=t_ok, total=t_lab.sum(), persons=1)
            fkey = "none" if face_bi[g] < 0 else FACE_SPREAD_LABELS[face_bi[g]]
            face_buckets.add(
                fkey, gt=1, matched=m, oks_sum=moks,
                pck_correct=f_ok, pck_total=(f_lab.sum() if fkey != "none" else 0), pck2_correct=f_px,
            )
            tkey = "none" if torso_bi[g] < 0 else TORSO_SPREAD_LABELS[torso_bi[g]]
            torso_buckets.add(
                tkey, gt=1, matched=m, oks_sum=moks,
                pck_correct=t_ok, pck_total=(t_lab.sum() if tkey != "none" else 0),
            )
            if edge is not None:
                ekey = "edge" if edge[g] else "interior"
                edge_buckets.add(
                    ekey, gt=1, matched=m, oks_sum=moks,
                    face_pck_correct=f_ok, face_pck_total=(f_lab.sum() if not np.isnan(face_spread[g]) else 0),
                )

    num_images = len(gts)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, num_gt)
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    per_landmark = {}
    for k, name in enumerate(KEYPOINT_NAMES):
        q = _quantiles(nme_values[k])
        entry: dict[str, Any] = {
            "mean_oks": _ratio(oks_sum[k], oks_cnt[k]),
            "matched_points": int(oks_cnt[k]),
            "nme_mean": q["mean"],
            "nme_median": q["median"],
            "nme_p90": q["p90"],
        }
        if has_vis:
            total = confusion[k].sum()
            entry["visibility_accuracy"] = _ratio(np.trace(confusion[k]), total)
        per_landmark[name] = entry

    visibility: dict[str, Any] | None = None
    if has_vis:
        total_conf = confusion.sum(axis=0)
        macro, per_class = _macro_f1(total_conf)
        visibility = {
            "classes": ["absent(0)", "occluded(1)", "visible(2)"],
            "confusion_rows_gt_cols_pred": total_conf.tolist(),
            "accuracy": _ratio(np.trace(total_conf), total_conf.sum()),
            "macro_f1": macro,
            "f1_per_class": per_class,
            "per_landmark_macro_f1": {name: _macro_f1(confusion[k])[0] for k, name in enumerate(KEYPOINT_NAMES)},
            "per_landmark_confusion": {name: confusion[k].tolist() for k, name in enumerate(KEYPOINT_NAMES)},
        }

    def _scale_buckets(acc: _Acc, labels: tuple[str, ...], pck2: bool) -> dict[str, Any]:
        out = {}
        for label in (*labels, "none"):
            s = acc.data.get(label, {})
            entry = {
                "gt": int(s.get("gt", 0)),
                "recall": _ratio(s.get("matched", 0), s.get("gt", 0)),
                "mean_matched_oks": _ratio(s.get("oks_sum", 0), s.get("matched", 0)),
                "pck@0.1": _ratio(s.get("pck_correct", 0), s.get("pck_total", 0)),
            }
            if pck2:
                entry["pck@2px"] = _ratio(s.get("pck2_correct", 0), s.get("pck_total", 0))
            out[label] = entry
        return out

    crowd_out = {}
    for label in CROWD_LABELS:
        s = crowd_buckets.data.get(label, {})
        crowd_out[label] = {
            "images": int(s.get("images", 0)),
            "gt": int(s.get("gt", 0)),
            "tp": int(s.get("tp", 0)),
            "fp": int(s.get("fp", 0)),
            "duplicates": int(s.get("dup", 0)),
            "precision": _ratio(s.get("tp", 0), s.get("tp", 0) + s.get("fp", 0)),
            "recall": _ratio(s.get("tp", 0), s.get("gt", 0)),
        }

    edge_out = None
    if has_edge:
        edge_out = {}
        for label in ("edge", "interior"):
            s = edge_buckets.data.get(label, {})
            edge_out[label] = {
                "gt": int(s.get("gt", 0)),
                "recall": _ratio(s.get("matched", 0), s.get("gt", 0)),
                "mean_matched_oks": _ratio(s.get("oks_sum", 0), s.get("matched", 0)),
                "face_pck@0.1": _ratio(s.get("face_pck_correct", 0), s.get("face_pck_total", 0)),
            }

    face = pck.data.get("face", {})
    torso = pck.data.get("torso", {})
    return {
        "conf_threshold": float(conf_threshold),
        "match_threshold": float(match_threshold),
        "num_images": num_images,
        "detection": {
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(num_gt - tp),
            "ignored_dets": int(ignored),
            "num_gt": int(num_gt),
            "fp_duplicate": int(fp_dup),
            "fp_localisation": int(fp_loc),
            "fp_background": int(fp_bg),
        },
        "duplicates_per_image": _ratio(fp_dup, num_images),
        "images_with_duplicates": int(images_with_dup),
        "mean_matched_oks": float(np.mean(matched_oks)) if matched_oks else float("nan"),
        "per_landmark": per_landmark,
        "face_nme_by_spread": _quantiles(face_nme),
        "visibility": visibility,
        "pck": {
            "face@0.1": _ratio(face.get("correct", 0), face.get("total", 0)),
            "face@2px_net": _ratio(face.get("correct_2px", 0), face.get("total", 0)),
            "torso@0.1": _ratio(torso.get("correct", 0), torso.get("total", 0)),
            "face_points": int(face.get("total", 0)),
            "torso_points": int(torso.get("total", 0)),
        },
        "face_spread_net": _scale_buckets(face_buckets, FACE_SPREAD_LABELS, pck2=True),
        "torso_spread_net": _scale_buckets(torso_buckets, TORSO_SPREAD_LABELS, pck2=False),
        "crowd": crowd_out,
        "edge": edge_out,
    }


# ---------------------------------------------------------------- pycocotools


def coco_eval(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    image_ids: Sequence[int] | None = None,
    *,
    max_det: int = 20,
    sigmas: np.ndarray = OKS_SIGMAS,
    score_threshold: float = 0.0,
) -> dict[str, Any]:
    """Cross-check with pycocotools ``COCOeval(iouType='keypoints')`` using the 9 sigmas.

    GT/DT structures are built in memory. GTs without labeled landmarks are left
    out (COCO would mark them ignore and match them by box; :func:`evaluate` gives
    them OKS 0), so both evaluators see the same problem.
    """

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if image_ids is None:
        image_ids = list(range(1, len(gts) + 1))
    image_ids = [int(i) for i in image_ids]
    gt_anns: list[dict[str, Any]] = []
    dt_anns: list[dict[str, Any]] = []
    for img_id, pred, gt in zip(image_ids, preds, gts):
        g_kpts, g_area, crowd, _ = _gt_fields(gt)
        for g in range(g_kpts.shape[0]):
            kp = g_kpts[g].copy()
            kp[:, 2] = np.maximum(kp[:, 2], 0)
            lab = kp[:, 2] > 0
            if not lab.any():
                continue
            x0, y0 = kp[lab, 0].min(), kp[lab, 1].min()
            x1, y1 = kp[lab, 0].max(), kp[lab, 1].max()
            gt_anns.append({
                "id": len(gt_anns) + 1,
                "image_id": img_id,
                "category_id": 1,
                "keypoints": kp.reshape(-1).tolist(),
                "num_keypoints": int(lab.sum()),
                "area": float(g_area[g]),
                "iscrowd": int(crowd[g]),
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            })
        d_kpts, d_scores, _ = _sorted_dets(pred, max_det=10**9, score_threshold=score_threshold)
        for d in range(d_scores.size):
            dt_anns.append({
                "image_id": img_id,
                "category_id": 1,
                "keypoints": d_kpts[d].reshape(-1).tolist(),
                "score": float(d_scores[d]),
            })
    nan_result = {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "AR": float("nan")}
    if not gt_anns or not dt_anns:
        return {**nan_result, "num_gt": len(gt_anns), "num_det": len(dt_anns), "max_det": max_det}
    dataset = {
        "images": [{"id": i} for i in image_ids],
        "annotations": gt_anns,
        "categories": [{"id": 1, "name": "person", "keypoints": list(KEYPOINT_NAMES), "skeleton": []}],
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt_anns)
        ev = COCOeval(coco_gt, coco_dt, "keypoints")
        ev.params.kpt_oks_sigmas = np.asarray(sigmas, dtype=np.float64)
        ev.params.imgIds = image_ids
        ev.params.maxDets = [int(max_det)]
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval["precision"][:, :, 0, 0, -1]  # (T, R) area=all, maxDet=last
    recall = ev.eval["recall"][:, 0, 0, -1]  # (T,)

    def _mean(arr: np.ndarray) -> float:
        valid = arr[arr > -1]
        return float(valid.mean()) if valid.size else float("nan")

    return {
        "AP": _mean(precision),
        "AP50": _mean(precision[0]),
        "AP75": _mean(precision[5]),
        "AR": _mean(recall),
        "num_gt": len(gt_anns),
        "num_det": len(dt_anns),
        "max_det": int(max_det),
    }


# ------------------------------------------------- confidence sweep / operating points

DEFAULT_SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05..0.95
_KIND_TP, _KIND_DUP, _KIND_LOC, _KIND_BG, _KIND_IGN = 0, 1, 2, 3, 4


def _classify_detections(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    *,
    match_threshold: float,
    min_score: float,
    max_det: int,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score and outcome (TP / duplicate / localisation / background / ignored) of every detection.

    Greedy matching visits detections by descending score, so the outcome of a detection
    depends only on higher-scored detections of its image: matching once at the lowest
    threshold gives the exact matching at every higher confidence threshold (the top
    ``max_det`` survivors of a higher threshold are the survivors of the top ``max_det``).
    """

    thr = np.asarray([match_threshold], dtype=np.float64)
    scores_all: list[np.ndarray] = []
    kinds_all: list[np.ndarray] = []
    num_gt = 0
    for pred, gt in zip(preds, gts):
        g_kpts, g_area, crowd, regular = _gt_fields(gt)
        num_gt += int(regular.sum())
        d_kpts, d_scores, _ = _sorted_dets(pred, max_det=max_det, score_threshold=min_score)
        if d_scores.size == 0:
            continue
        oks = oks_matrix(d_kpts, g_kpts, g_area, sigmas=sigmas)
        dt_gt, dt_ign = _match(oks, regular, crowd, thr)
        dt_gt, dt_ign = dt_gt[0], dt_ign[0]
        best_reg = oks[:, regular].max(axis=1) if regular.any() else np.zeros(d_scores.size)
        kind = np.full(d_scores.size, _KIND_BG, dtype=np.int8)
        kind[(dt_gt < 0) & (best_reg >= _LOCALISATION_FP_OKS)] = _KIND_LOC
        kind[(dt_gt < 0) & (best_reg >= match_threshold)] = _KIND_DUP
        kind[(dt_gt >= 0) & ~dt_ign] = _KIND_TP
        kind[dt_ign] = _KIND_IGN
        scores_all.append(d_scores)
        kinds_all.append(kind)
    if not scores_all:
        return np.zeros(0), np.zeros(0, np.int8), num_gt
    return np.concatenate(scores_all), np.concatenate(kinds_all), num_gt


def confidence_sweep(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    *,
    thresholds: Sequence[float] = DEFAULT_SWEEP_THRESHOLDS,
    match_threshold: float = 0.5,
    max_det: int = 100,
    sigmas: np.ndarray = OKS_SIGMAS,
) -> list[dict[str, Any]]:
    """Detection P/R/F1 and FP composition per confidence threshold (OKS >= ``match_threshold``).

    Each row equals ``evaluate_detailed(..., conf_threshold=t)['detection']`` plus per-image
    rates, but all rows come from one matching pass. Rows are sorted by threshold.
    """

    ths = sorted({float(t) for t in thresholds})
    if not ths:
        return []
    scores, kinds, num_gt = _classify_detections(
        preds, gts, match_threshold=match_threshold, min_score=ths[0], max_det=max_det, sigmas=sigmas
    )
    num_images = max(len(gts), 1)
    rows = []
    for t in ths:
        counts = np.bincount(kinds[scores >= t].astype(np.int64), minlength=5)
        tp, dup, loc, bg, ign = (int(c) for c in counts[:5])
        fp = dup + loc + bg
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, num_gt)
        if np.isnan(recall):
            f1 = float("nan")
        elif np.isnan(precision) or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": int(num_gt - tp),
            "ignored_dets": ign,
            "fp_duplicate": dup,
            "fp_localisation": loc,
            "fp_background": bg,
            "fp_per_image": fp / num_images,
            "duplicates_per_image": dup / num_images,
            "fp_localisation_per_image": loc / num_images,
            "fp_background_per_image": bg / num_images,
            "dets_per_image": (tp + fp + ign) / num_images,
        })
    return rows


def select_operating_points(
    sweep: Sequence[dict[str, Any]], *, ref_fp_per_image: float | None = None
) -> dict[str, dict[str, Any] | None]:
    """Pick operating points from a :func:`confidence_sweep`.

    ``f1_max``: highest F1 (ties -> lowest threshold). ``fp_match``: FP/img closest to
    ``ref_fp_per_image`` (ties -> highest threshold).
    """

    out: dict[str, dict[str, Any] | None] = {"f1_max": None, "fp_match": None}
    valid = [r for r in sweep if r.get("f1") is not None and not np.isnan(r["f1"])]
    if valid:
        out["f1_max"] = dict(max(valid, key=lambda r: (r["f1"], -r["threshold"])))
    if ref_fp_per_image is not None and sweep:
        best = min(sweep, key=lambda r: (abs(r["fp_per_image"] - ref_fp_per_image), -r["threshold"]))
        out["fp_match"] = {**best, "ref_fp_per_image": float(ref_fp_per_image)}
    return out


# ------------------------------------------------------- visibility calibration

DEFAULT_VIS_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 10))  # 0.1..0.9


def _vis_classes(present: np.ndarray, visible: np.ndarray, present_thr: float, visible_thr: float) -> np.ndarray:
    """§2 decision rule with adjustable thresholds: 0 if present < p_thr, else 2 if visible >= v_thr else 1."""

    return np.where(present < present_thr, 0, np.where(visible >= visible_thr, 2, 1))


def _confusion3(gt_cls: np.ndarray, pred_cls: np.ndarray) -> np.ndarray:
    return np.bincount(gt_cls * 3 + pred_cls, minlength=9).reshape(3, 3)


def calibrate_visibility(
    preds: Sequence[ImagePredictions],
    gts: Sequence[ImageGroundTruth],
    *,
    conf_threshold: float = 0.25,
    match_threshold: float = 0.5,
    present_grid: Sequence[float] = DEFAULT_VIS_GRID,
    visible_grid: Sequence[float] = DEFAULT_VIS_GRID,
    max_det: int = 100,
    sigmas: np.ndarray = OKS_SIGMAS,
) -> dict[str, Any] | None:
    """Grid-search the (present, visible) thresholds that maximise visibility macro-F1.

    Uses the landmarks of detections matched (OKS >= ``match_threshold``, score >=
    ``conf_threshold``) to regular GTs, with GT ``v`` in {0, 1, 2} (``v = -1`` skipped).
    Requires ``present_prob`` and ``visible_prob`` on every non-empty prediction and returns
    ``None`` otherwise. Ties prefer thresholds closest to (0.5, 0.5).
    """

    if not preds or any(
        p.present_prob is None or p.visible_prob is None for p in preds if np.asarray(p.scores).size
    ):
        return None
    thr = np.asarray([match_threshold], dtype=np.float64)
    gt_v: list[np.ndarray] = []
    pres: list[np.ndarray] = []
    visp: list[np.ndarray] = []
    for pred, gt in zip(preds, gts):
        g_kpts, g_area, crowd, regular = _gt_fields(gt)
        order = _det_order(pred, max_det=max_det, score_threshold=conf_threshold)
        if order.size == 0 or not regular.any():
            continue
        d_kpts = np.asarray(pred.keypoints, dtype=np.float64).reshape(-1, NUM_KEYPOINTS, 3)[order]
        oks = oks_matrix(d_kpts, g_kpts, g_area, sigmas=sigmas)
        dt_gt, dt_ign = _match(oks, regular, crowd, thr)
        tp = np.nonzero((dt_gt[0] >= 0) & ~dt_ign[0])[0]
        if tp.size == 0:
            continue
        gv = np.rint(g_kpts[dt_gt[0][tp], :, 2]).astype(np.int64)
        pp = np.asarray(pred.present_prob, dtype=np.float64).reshape(-1, NUM_KEYPOINTS)[order[tp]]
        vp = np.asarray(pred.visible_prob, dtype=np.float64).reshape(-1, NUM_KEYPOINTS)[order[tp]]
        known = gv >= 0
        gt_v.append(np.clip(gv[known], 0, 2))
        pres.append(pp[known])
        visp.append(vp[known])
    if not gt_v:
        return {"conf_threshold": float(conf_threshold), "points": 0, "present_threshold": None,
                "visible_threshold": None, "macro_f1": float("nan")}
    g = np.concatenate(gt_v)
    pp = np.concatenate(pres)
    vp = np.concatenate(visp)

    def _score(pt: float, vt: float) -> tuple[float, np.ndarray, list[float]]:
        conf = _confusion3(g, _vis_classes(pp, vp, pt, vt))
        macro, per_class = _macro_f1(conf)
        return macro, conf, per_class

    grid = np.full((len(present_grid), len(visible_grid)), np.nan)
    best_key: tuple[float, float] | None = None
    best: tuple | None = None
    for i, pt in enumerate(present_grid):
        for j, vt in enumerate(visible_grid):
            macro, conf, per_class = _score(float(pt), float(vt))
            grid[i, j] = macro
            key = (round(macro, 12), -abs(float(pt) - 0.5) - abs(float(vt) - 0.5))
            if best_key is None or key > best_key:
                best_key, best = key, (float(pt), float(vt), macro, conf, per_class)
    d_macro, d_conf, d_per = _score(0.5, 0.5)
    pt, vt, macro, conf, per_class = best  # type: ignore[misc]
    return {
        "conf_threshold": float(conf_threshold),
        "points": int(g.size),
        "present_threshold": pt,
        "visible_threshold": vt,
        "macro_f1": float(macro),
        "f1_per_class": per_class,
        "accuracy": _ratio(np.trace(conf), conf.sum()),
        "confusion_rows_gt_cols_pred": conf.tolist(),
        "default@0.5": {
            "macro_f1": float(d_macro),
            "f1_per_class": d_per,
            "accuracy": _ratio(np.trace(d_conf), d_conf.sum()),
            "confusion_rows_gt_cols_pred": d_conf.tolist(),
        },
        "present_grid": [float(x) for x in present_grid],
        "visible_grid": [float(x) for x in visible_grid],
        "macro_f1_grid_rows_present_cols_visible": grid.tolist(),
    }
