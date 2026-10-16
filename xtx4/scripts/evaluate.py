"""Evaluate an XTX3 or XTX4 checkpoint with the XTX4 evaluator (plan §9, CONTRACTS §9).

Why a separate script: the B0 baseline has to be re-measured with the fixed metric
(integer OKS thresholds, crowd-aware COCO matching, score threshold 0.001, full test
split) and every later XTX4 run must be measured the same way. The script

* loads the ground truth with XTX3's scanner (``xtx3.data.annotations.scan_split`` +
  ``denormalise_sample``; GT ``area`` = box area exactly as XTX3 used it, reported as
  ``area_source: "bbox"``) - the XTX4 data loader is not a dependency yet;
* runs the model through its public API (``xtx3.api`` replicated batch-wise with a
  parity check against ``detect_poses``; ``xtx4.api`` imported lazily);
* writes a JSON report (AP9_project = maxDet 100 at the score threshold, AP9_coco20 =
  maxDet 20, detailed reports at conf 0.25 and 0.05, optional pycocotools cross-check)
  and upserts a row per run into a markdown summary table.

For XTX3 in ``whole`` mode it also evaluates the *raw* one-to-one detections before
``merge_views`` - XTX3's merge step performs a greedy OKS/IoU suppression even for a
single view, so the raw numbers show what the "NMS-free" head really produces.

Predictions can be cached (``--save-preds``) and re-evaluated without the model
(``--load-preds``), e.g. to run the pycocotools cross-check in another interpreter.

Usage::

    python scripts/evaluate.py --model-kind xtx3 --weights ../xtx3/runs/xtx3-u/best.pt \
        --dataset D:/xtxtraining --split test --sizes 320 --view-mode whole --device cuda \
        --out reports/B0_xtx3u_whole_320.json --coco-check 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
XTX3_ROOT = REPO_ROOT.parent / "xtx3"  # sibling checkout of the XTX3 baseline
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xtx4.engine.metrics import (  # noqa: E402
    DEFAULT_SWEEP_THRESHOLDS,
    ImageGroundTruth,
    ImagePredictions,
    calibrate_visibility,
    coco_eval,
    confidence_sweep,
    evaluate,
    evaluate_detailed,
    select_operating_points,
)
from xtx4.utils.keypoints import NUM_KEYPOINTS  # noqa: E402
from xtx4.utils.logging import get_logger  # noqa: E402

logger = get_logger("xtx4.scripts.evaluate")

K = NUM_KEYPOINTS
_SUMMARY_BEGIN = "<!-- runs:begin -->"
_SUMMARY_END = "<!-- runs:end -->"


# ------------------------------------------------------------------- ground truth


@dataclass(slots=True)
class GTSet:
    image_paths: list[str]
    sizes: list[tuple[int, int]]  # (w, h)
    gts: list[ImageGroundTruth]


def _ensure_xtx3_path() -> None:
    if str(XTX3_ROOT) not in sys.path:
        sys.path.insert(0, str(XTX3_ROOT))


def load_ground_truth(dataset: Path, split: str, max_images: int | None) -> GTSet:
    """XTX3 scan + denormalisation; area = person box area in source px (XTX3 legacy)."""

    _ensure_xtx3_path()
    from xtx3.data.annotations import scan_split  # type: ignore[import-not-found]
    from xtx3.data.dataset import denormalise_sample  # type: ignore[import-not-found]

    samples = scan_split(dataset / split)
    if max_images:
        samples = samples[:max_images]
    paths, sizes, gts = [], [], []
    for sample in samples:
        width, height = int(sample.width), int(sample.height)
        if width <= 0 or height <= 0:
            import cv2

            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
        boxes, kpts = denormalise_sample(sample, width=width, height=height)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        crowd = np.array([bool(a.iscrowd) for a in sample.annotations], dtype=bool)
        paths.append(str(sample.image_path))
        sizes.append((width, height))
        gts.append(ImageGroundTruth(keypoints=kpts.astype(np.float64), areas=areas.astype(np.float64),
                                    iscrowd=crowd, image_size=(width, height)))
    return GTSet(paths, sizes, gts)


# --------------------------------------------------------------------- inference


def _read_images(paths: list[str], workers: int) -> Iterator[tuple[int, np.ndarray]]:
    import cv2

    def _load(item: tuple[int, str]) -> tuple[int, np.ndarray]:
        idx, path = item
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image {path}")
        return idx, image

    # Bounded prefetch: Executor.map would submit every image at once and decode the
    # whole split into RAM ahead of the GPU.
    from collections import deque

    depth = max(2, 4 * workers)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque = deque()
        items = iter(enumerate(paths))
        for item in items:
            pending.append(pool.submit(_load, item))
            if len(pending) >= depth:
                break
        while pending:
            result = pending.popleft().result()
            nxt = next(items, None)
            if nxt is not None:
                pending.append(pool.submit(_load, nxt))
            yield result


def _batched(it: Iterator, size: int) -> Iterator[list]:
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _empty_pred() -> ImagePredictions:
    return ImagePredictions(keypoints=np.zeros((0, K, 3)), scores=np.zeros(0), visibility=np.zeros((0, K), np.int64))


def _dets_to_pred(dets: list) -> ImagePredictions:
    if not dets:
        return _empty_pred()
    return ImagePredictions(
        keypoints=np.stack([np.asarray(d.keypoints, np.float32) for d in dets]),
        scores=np.array([float(d.score) for d in dets]),
        visibility=np.stack([np.asarray(d.visibility, np.int64) for d in dets]),
    )


class XTX3Runner:
    """``xtx3.api.detect_poses`` replicated batch-wise, keeping the pre-merge detections."""

    def __init__(self, weights: Path, *, device: str, view_mode: str, size: int) -> None:
        _ensure_xtx3_path()
        import torch
        import xtx3.api as xapi  # type: ignore[import-not-found]

        self.torch = torch
        self.xapi = xapi
        self.model = xapi.load_model(weights, device=device, fuse=True, mode=view_mode, network_size=size)
        self.device = device
        self.view_mode = view_mode
        self.size = size
        self.keys = xapi.view_keys_for_mode(view_mode)

    def run_batch(self, images: list[np.ndarray], *, conf: float, max_det: int) -> tuple[list, list]:
        xapi = self.xapi
        params = self.model.merge_params
        pre_conf = min(conf, params.conf_threshold)
        results = [xapi.preprocess_image(img, mode=self.view_mode, network_size=self.size) for img in images]
        batch = self.torch.cat([xapi._views_to_batch(r.views, keys=self.keys, device=self.device) for r in results])
        with self.torch.inference_mode():
            dets, vis = self.model.predict(batch, conf_threshold=pre_conf, max_detections=max_det)
        merged_out, raw_out = [], []
        for i, result in enumerate(results):
            by_view = {}
            for c, view in enumerate(self.keys):
                j = i * len(self.keys) + c
                by_view[view] = xapi._collect_view_detections(
                    dets[j].cpu().numpy(), vis[j].cpu().numpy(), result.transforms[view], view
                )
            merge_params = xapi.MergeParams.from_dict(params.to_dict())
            merge_params.conf_threshold = conf
            merge_params.max_detections = max_det
            merged_out.append(_dets_to_pred(xapi.merge_views(by_view, params=merge_params)))
            if len(self.keys) == 1:
                raw = [d for d in by_view[self.keys[0]] if d.score >= conf]
                raw.sort(key=lambda d: d.score, reverse=True)
                raw_out.append(_dets_to_pred(raw[:max_det]))
        return merged_out, raw_out

    def api_reference(self, image: np.ndarray, *, conf: float, max_det: int) -> ImagePredictions:
        with self.torch.inference_mode():
            poses = self.xapi.detect_poses(image, model=self.model, conf_threshold=conf, max_detections=max_det)
        return _poses_to_pred(poses)


def _poses_to_pred(poses: list) -> ImagePredictions:
    if not poses:
        return _empty_pred()
    vis = [getattr(p, "visibility", None) for p in poses]
    return ImagePredictions(
        keypoints=np.stack([np.asarray(p.keypoints, np.float64) for p in poses]),
        scores=np.array([float(p.score) for p in poses]),
        visibility=None if any(v is None for v in vis) else np.stack([np.asarray(v, np.int64) for v in vis]),
    )


class XTX4Runner:
    """``xtx4.api.detect_poses_batch`` replicated with one forward pass that also keeps the
    per-landmark ``present_prob`` / ``visible_prob`` (the public ``Pose`` exposes only the
    visibility class and ``conf``). Parity with ``detect_poses`` is checked on a few images.
    """

    def __init__(self, weights: Path, *, device: str, view_mode: str, size: int) -> None:
        try:
            import xtx4.api as api  # type: ignore[import-not-found]
            from xtx4.models.head_types import decode_detections, decode_keypoints
        except ImportError as exc:  # pragma: no cover - depends on wave-2 progress
            raise SystemExit(f"--model-kind xtx4 needs xtx4/api.py (not available: {exc})") from exc
        import torch

        if view_mode != "whole":
            raise SystemExit("XTX4 supports only --view-mode whole")
        self.torch = torch
        self.api = api
        self.decode_detections = decode_detections
        self.decode_keypoints = decode_keypoints
        self.device = device
        self.size = size
        self.model = api.load_model(weights, device=device, fuse=True, network_size=size)

    def run_batch(self, images: list[np.ndarray], *, conf: float, max_det: int) -> tuple[list, list]:
        api, torch = self.api, self.torch
        params = api._resolve(self.model, conf, max_det, self.size, False)  # NMS only via --oks-nms
        results = [api.preprocess_image(img, mode="whole", network_size=params["network_size"]) for img in images]
        view = api._WHOLE_VIEW
        batch = api._views_to_batch([r.views[view] for r in results], device=params["device"],
                                    channels_last=bool(getattr(self.model, "channels_last", False)))
        model = self.model
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                raw = model(batch).one2one
                dets, vis, sigma = self.decode_detections(
                    raw, conf_threshold=params["conf_threshold"], max_detections=params["max_detections"],
                    sigma_conf_coeff=params["sigma_conf_coeff"], sigma_floor=params["sigma_floor"],
                )
                dec = self.decode_keypoints(raw.kpt, raw.anchors, raw.strides,
                                            sigma_conf_coeff=params["sigma_conf_coeff"],
                                            sigma_floor=params["sigma_floor"])
                scores = torch.sigmoid(raw.person).squeeze(-1)
        finally:
            if was_training:
                model.train()
        out = []
        for b, result in enumerate(results):
            # same selection rule as head_types.decode_detections
            keep = torch.nonzero(scores[b] >= params["conf_threshold"]).flatten()
            order = torch.argsort(scores[b][keep], descending=True, stable=True)[: params["max_detections"]]
            keep = keep[order]
            det = dets[b].float().cpu().numpy()
            present = dec["present_prob"][b][keep].float().cpu().numpy()
            visible = dec["visible_prob"][b][keep].float().cpu().numpy()
            if det.shape[0] == 0:
                out.append(_empty_pred())
                continue
            kpts = det[:, 1:].reshape(det.shape[0], K, 3)
            kpts = self.api.geometry.apply_to_keypoints(result.transforms[view], kpts).astype(np.float64)
            idx = np.arange(det.shape[0])
            if params["oks_nms"] and det.shape[0] > 1:
                idx = api.oks_nms(kpts.astype(np.float32), det[:, 0], threshold=params["oks_nms_threshold"])
            out.append(ImagePredictions(
                keypoints=kpts[idx],
                scores=det[idx, 0].astype(np.float64),
                visibility=vis[b].cpu().numpy()[idx].astype(np.int64),
                present_prob=present[idx].astype(np.float64),
                visible_prob=visible[idx].astype(np.float64),
            ))
        return out, []

    def api_reference(self, image: np.ndarray, *, conf: float, max_det: int) -> ImagePredictions:
        with self.torch.inference_mode():
            poses = self.api.detect_poses(image, self.model, conf_threshold=conf, max_detections=max_det,
                                          network_size=self.size, oks_nms=False)
        return _poses_to_pred(poses)


class NcnnRunner:
    """Exported ncnn graph through the torch-free runtime (``xtx4.runtime``).

    Replicates ``NcnnPoseRunner.detect`` (whole mode, runner NMS **off**) step by step so the
    per-landmark ``present_prob`` / ``visible_prob`` can be read from the raw pose output; parity
    with ``detect(..., oks_nms=False)`` is checked on a few images. Host NMS, when wanted, is
    applied once by ``--oks-nms`` exactly as for the other model kinds.
    """

    def __init__(self, *, size: int, manifest: Path | None = None, param: Path | None = None,
                 bin_path: Path | None = None, num_threads: int = 4, fp16: bool | None = None,
                 view_mode: str = "whole") -> None:
        from xtx4.runtime import decode as rdec
        from xtx4.runtime.ncnn_runner import NcnnPoseRunner
        from xtx4.runtime.preprocess import prepare_views

        if view_mode != "whole":
            raise SystemExit("--model-kind ncnn supports only --view-mode whole")
        kwargs: dict[str, Any] = {"num_threads": num_threads}
        if fp16 is not None:
            kwargs["fp16"] = fp16
        if manifest is not None:
            self.runner = NcnnPoseRunner.from_manifest(manifest, size, **kwargs)
        elif param is not None and bin_path is not None:
            self.runner = NcnnPoseRunner(param, bin_path, size=size, **kwargs)
        else:
            raise SystemExit("--model-kind ncnn needs --manifest or --param + --bin")
        self.rdec = rdec
        self.prepare_views = prepare_views
        self.device = "cpu"
        self.torch = None

    def predict_one(self, image: np.ndarray, *, conf: float, max_det: int) -> ImagePredictions:
        r, rdec = self.runner, self.rdec
        prepared = self.prepare_views(image, network_size=r.size, mode="whole")
        (key, view), = prepared.views.items()
        person, pose = r.forward(view)
        scores = rdec.person_scores(person)
        keep, _ = rdec.select_candidates(scores, conf_threshold=conf, max_detections=max_det)
        self.last_keep = keep  # anchor indices of the returned detections (tests / diagnostics)
        if keep.size == 0:
            return _empty_pred()
        keypoints, visibility, sigma_px = rdec.decode_selected(
            pose, keep, network_size=r.size, strides=r.strides, num_keypoints=r.num_keypoints,
            sigma_conf_coeff=r.sigma_conf_coeff, sigma_floor=r.sigma_floor,
            present_threshold=r.present_threshold, visible_threshold=r.visible_threshold,
        )
        poses = rdec.to_frame_poses(scores[keep], keypoints, visibility, sigma_px,
                                    transform=prepared.transforms[key], view=key, anchor_indices=keep)
        raw = np.asarray(pose, dtype=np.float32).reshape(-1, r.num_keypoints, 5)[keep]
        present = 1.0 / (1.0 + np.exp(-raw[..., 3].astype(np.float64)))
        visible = 1.0 / (1.0 + np.exp(-raw[..., 4].astype(np.float64)))
        return ImagePredictions(
            keypoints=np.stack([np.asarray(q.keypoints, np.float64) for q in poses]),
            scores=np.array([q.score for q in poses], dtype=np.float64),
            visibility=np.stack([np.asarray(q.visibility, np.int64) for q in poses]),
            present_prob=present,
            visible_prob=visible,
        )

    def run_batch(self, images: list[np.ndarray], *, conf: float, max_det: int) -> tuple[list, list]:
        return [self.predict_one(img, conf=conf, max_det=max_det) for img in images], []

    def api_reference(self, image: np.ndarray, *, conf: float, max_det: int) -> ImagePredictions:
        result = self.runner.detect(image, conf_threshold=conf, max_detections=max_det, oks_nms=False,
                                    keep_views=False)
        return _poses_to_pred(result.poses)

    def describe(self) -> dict[str, Any]:
        r = self.runner
        return {"param": str(r.param_path), "bin": str(r.bin_path), "fp16": bool(r.fp16),
                "num_threads": int(r.num_threads), "strides": list(r.strides), "describe": r.describe()}


def _parity(a: ImagePredictions, b: ImagePredictions) -> dict[str, Any]:
    same_count = a.scores.size == b.scores.size
    out: dict[str, Any] = {"count_a": int(a.scores.size), "count_b": int(b.scores.size), "same_count": bool(same_count)}
    if same_count and a.scores.size:
        out["max_abs_score"] = float(np.abs(a.scores - b.scores).max())
        out["max_abs_xy"] = float(np.abs(a.keypoints[:, :, :2] - b.keypoints[:, :, :2]).max())
        if a.visibility is not None and b.visibility is not None:
            out["visibility_equal_frac"] = float((np.asarray(a.visibility) == np.asarray(b.visibility)).mean())
        if a.present_prob is not None:
            # with sigma_conf_coeff = 0 the public keypoint conf equals present_prob
            out["max_abs_present_vs_api_conf"] = float(np.abs(a.present_prob - b.keypoints[:, :, 2]).max())
    return out


def apply_oks_nms(preds: list[ImagePredictions | None], threshold: float) -> tuple[list[ImagePredictions], dict]:
    """Host-side greedy OKS-NMS per image with ``xtx4.runtime.decode.oks_nms`` (deployment semantics:
    reference = candidate's kept competitor, extent-box scale of confident points)."""

    from xtx4.runtime.decode import Pose as RuntimePose
    from xtx4.runtime.decode import oks_nms

    out: list[ImagePredictions] = []
    before = after = 0
    for pred in preds:
        n = pred.scores.size
        before += n
        if n <= 1:
            out.append(pred)
            after += n
            continue
        poses = [
            RuntimePose(
                score=float(pred.scores[i]),
                keypoints=np.asarray(pred.keypoints[i], np.float32),
                visibility=np.zeros(K, np.int64) if pred.visibility is None else np.asarray(pred.visibility[i]),
                sigma_px=np.zeros(K, np.float32),
                anchor_index=i,
            )
            for i in range(n)
        ]
        keep = np.asarray([q.anchor_index for q in oks_nms(poses, threshold=threshold)], dtype=np.int64)
        after += keep.size

        def _sel(arr: np.ndarray | None) -> np.ndarray | None:
            return None if arr is None else np.asarray(arr)[keep]

        out.append(ImagePredictions(
            keypoints=pred.keypoints[keep],
            scores=pred.scores[keep],
            visibility=_sel(pred.visibility),
            present_prob=_sel(pred.present_prob),
            visible_prob=_sel(pred.visible_prob),
        ))
    return out, {"threshold": float(threshold), "dets_before": int(before), "dets_after": int(after)}


def run_inference(
    runner, gtset: GTSet, *, conf: float, max_det: int, batch_size: int, workers: int, parity_images: int = 4
) -> dict[str, Any]:
    num = len(gtset.image_paths)
    preds: list[ImagePredictions | None] = [None] * num
    raw: list[ImagePredictions | None] = [None] * num
    size_mismatch = 0
    parity: list[dict[str, Any]] = []
    cuda = getattr(runner, "device", "") == "cuda"
    # warm-up (cudnn autotune, lazy init) outside the timed region
    warm = _read_images(gtset.image_paths[:1], 1)
    for _, image in warm:
        runner.run_batch([image], conf=conf, max_det=max_det)
    start = time.perf_counter()
    last_log = start
    done = 0
    for chunk in _batched(_read_images(gtset.image_paths, workers), batch_size):
        idxs = [i for i, _ in chunk]
        images = [img for _, img in chunk]
        for i, img in zip(idxs, images):
            if (img.shape[1], img.shape[0]) != gtset.sizes[i]:
                size_mismatch += 1
        merged, raw_batch = runner.run_batch(images, conf=conf, max_det=max_det)
        for j, i in enumerate(idxs):
            preds[i] = merged[j]
            if raw_batch:
                raw[i] = raw_batch[j]
        done += len(idxs)
        now = time.perf_counter()
        if now - last_log > 60:
            rate = done / (now - start)
            logger.info("  %d/%d images, %.1f img/s, ETA %.1f min", done, num, rate, (num - done) / rate / 60)
            last_log = now
    if cuda:
        runner.torch.cuda.synchronize()
    wall = time.perf_counter() - start
    # parity of the batched replica against the public single-image API
    for i, image in _read_images(gtset.image_paths[:parity_images], 1):
        ref = runner.api_reference(image, conf=conf, max_det=max_det)
        parity.append(_parity(preds[i], ref))
    return {
        "preds": preds,
        "raw": raw if raw[0] is not None else None,
        "wall_s": wall,
        "images_per_s": num / wall if wall > 0 else float("nan"),
        "size_mismatch": size_mismatch,
        "api_parity": parity,
    }


# ------------------------------------------------------------------ caching


def _pack(preds: list[ImagePredictions]) -> dict[str, np.ndarray]:
    counts = np.array([p.scores.size for p in preds], np.int64)
    kp = np.concatenate([p.keypoints for p in preds]).astype(np.float32) if counts.sum() else np.zeros((0, K, 3), np.float32)
    sc = np.concatenate([p.scores for p in preds]).astype(np.float32) if counts.sum() else np.zeros(0, np.float32)
    has_vis = all(p.visibility is not None for p in preds)
    vis = (
        np.concatenate([np.asarray(p.visibility, np.int8).reshape(-1, K) for p in preds])
        if has_vis and counts.sum()
        else np.zeros((0, K), np.int8)
    )
    packed = {"counts": counts, "kpts": kp, "scores": sc, "vis": vis, "has_vis": np.array(has_vis)}
    for name in ("present_prob", "visible_prob"):
        if counts.sum() and all(getattr(p, name) is not None or p.scores.size == 0 for p in preds):
            packed[name] = np.concatenate(
                [np.asarray(getattr(p, name), np.float32).reshape(-1, K) for p in preds if p.scores.size]
            )
    return packed


def _unpack(data: dict[str, np.ndarray]) -> list[ImagePredictions]:
    out, offset = [], 0
    has_vis = bool(data["has_vis"])
    for c in data["counts"].tolist():
        sl = slice(offset, offset + c)
        out.append(ImagePredictions(
            keypoints=data["kpts"][sl].astype(np.float64),
            scores=data["scores"][sl].astype(np.float64),
            visibility=data["vis"][sl].astype(np.int64) if has_vis else None,
            present_prob=data["present_prob"][sl].astype(np.float64) if "present_prob" in data else None,
            visible_prob=data["visible_prob"][sl].astype(np.float64) if "visible_prob" in data else None,
        ))
        offset += c
    return out


def save_preds(path: Path, result: dict[str, Any], meta: dict[str, Any], limit: int | None = None) -> None:
    """Cache predictions (optionally only the first ``limit`` images, e.g. for the coco check)."""

    n = len(result["preds"]) if not limit else min(limit, len(result["preds"]))
    arrays = {f"merged_{k}": v for k, v in _pack(result["preds"][:n]).items()}
    if result.get("raw") is not None and not limit:
        arrays.update({f"raw_{k}": v for k, v in _pack(result["raw"][:n]).items()})
    meta = {**meta, "num_images": n}
    meta = {**meta, **{k: result[k] for k in ("wall_s", "images_per_s", "size_mismatch", "api_parity")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    logger.info("Saved predictions to %s", path)


def load_preds(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))

    def _sub(prefix: str) -> dict[str, np.ndarray] | None:
        keys = [k for k in data.files if k.startswith(prefix)]
        return {k[len(prefix):]: data[k] for k in keys} if keys else None

    merged = _unpack(_sub("merged_"))
    raw_d = _sub("raw_")
    result = {
        "preds": merged,
        "raw": _unpack(raw_d) if raw_d else None,
        **{k: meta.get(k) for k in ("wall_s", "images_per_s", "size_mismatch", "api_parity")},
    }
    return result, meta


# --------------------------------------------------------------------- report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(obj: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def metrics_block(preds: list[ImagePredictions], gts: list[ImageGroundTruth], args, *, coco_n: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    block: dict[str, Any] = {
        "AP9_project": evaluate(preds, gts, max_det=args.max_det, score_threshold=args.score_threshold),
        "AP9_coco20": evaluate(preds, gts, max_det=20, score_threshold=args.score_threshold),
        "detailed@0.25": evaluate_detailed(preds, gts, conf_threshold=0.25, max_det=args.max_det),
        "detailed@0.05": evaluate_detailed(preds, gts, conf_threshold=0.05, max_det=args.max_det),
    }
    thresholds = sorted({*DEFAULT_SWEEP_THRESHOLDS, *(args.thresholds or [])})
    sweep = confidence_sweep(preds, gts, thresholds=thresholds, max_det=args.max_det)
    ref_fp, ref_source = resolve_ref_fp(args)
    points = select_operating_points(sweep, ref_fp_per_image=ref_fp)
    block["sweep"] = {
        "match_threshold": 0.5,
        "ref_fp_per_image": ref_fp,
        "ref_source": ref_source,
        "operating_points": {
            name: (None if row is None else {k: row[k] for k in (
                "threshold", "precision", "recall", "f1", "fp_per_image", "duplicates_per_image")})
            for name, row in points.items()
        },
        "rows": sweep,
    }
    for name, row in points.items():
        if row is not None:
            block[f"detailed@{name}"] = evaluate_detailed(
                preds, gts, conf_threshold=row["threshold"], max_det=args.max_det
            )
    vis_conf = points["f1_max"]["threshold"] if points.get("f1_max") else 0.25
    block["visibility_calibrated"] = calibrate_visibility(preds, gts, conf_threshold=vis_conf, max_det=args.max_det)
    if coco_n:
        n = min(coco_n, len(gts))
        try:
            ref = coco_eval(preds[:n], gts[:n], list(range(1, n + 1)), max_det=20, score_threshold=args.score_threshold)
            ours = evaluate(preds[:n], gts[:n], max_det=20, score_threshold=args.score_threshold)
            block["coco_check"] = {
                "images": n,
                "pycocotools": ref,
                "ours_maxdet20": {k: ours[k] for k in ("AP", "AP50", "AP75", "AR", "num_gt", "num_det")},
                "delta": {k: ours[k] - ref[k] for k in ("AP", "AP50", "AP75", "AR")},
            }
        except ImportError:
            block["coco_check"] = {"skipped": "pycocotools not installed in this interpreter"}
            logger.warning("pycocotools missing: coco cross-check skipped (re-run with --load-preds elsewhere)")
    block["metric_seconds"] = time.perf_counter() - t0
    return block


_B0_REF_REPORT = REPO_ROOT / "reports" / "B0_xtx3u_whole_320.json"
_B0_REF_FALLBACK = 9631 / 11320  # B0 xtx3-u@320 FP/img at conf 0.25 (full test split)


def resolve_ref_fp(args: argparse.Namespace) -> tuple[float, str]:
    """Reference FP/img for the ``fp_match`` operating point (CLI value or B0 xtx3-u@320 @0.25)."""

    if getattr(args, "ref_fp_per_image", None) is not None:
        return float(args.ref_fp_per_image), "--ref-fp-per-image"
    try:
        report = json.loads(_B0_REF_REPORT.read_text(encoding="utf-8"))
        run = next(r for r in report["runs"] if r["size"] == 320)
        det = run["api"]["detailed@0.25"]
        return det["detection"]["fp"] / det["num_images"], f"{_B0_REF_REPORT.name} api detailed@0.25"
    except (OSError, KeyError, StopIteration, ValueError, ZeroDivisionError):
        return _B0_REF_FALLBACK, "built-in B0 xtx3-u@320 (9631/11320)"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _summary_row(run: dict[str, Any], variant: str) -> tuple[str, str]:
    m = run[variant]
    ap = m["AP9_project"]
    d = m["detailed@0.25"]
    fb = d["face_spread_net"]
    coco = m.get("coco_check", {})
    delta = coco.get("delta", {}).get("AP") if isinstance(coco, dict) else None
    key = f"{run['name']}|{run['view_mode']}|{run['size']}|{variant}"
    cells = [
        run["name"], run["view_mode"], str(run["size"]), variant, str(run["num_images"]), str(ap["num_gt"]),
        _fmt(ap["AP"]), _fmt(ap["AP50"]), _fmt(ap["AP75"]), _fmt(ap["AR"]), _fmt(m["AP9_coco20"]["AP"]),
        _fmt(d["detection"]["precision"]), _fmt(d["detection"]["recall"]), _fmt(d["duplicates_per_image"], 3),
        _fmt(d["pck"]["face@0.1"]), _fmt(d["pck"]["face@2px_net"]),
        " / ".join(_fmt(fb[b]["recall"], 3) for b in ("<8", "8-16", "16-32", ">=32")),
        _fmt(run.get("images_per_s"), 1), "-" if delta is None else f"{delta:+.4f}",
    ]
    return key, "| " + " | ".join(cells) + " |"


_SUMMARY_HEADER = (
    "| run | view | S | preds | images | GT | AP | AP50 | AP75 | AR | AP coco20 | P@0.25 | R@0.25 | dup/img@0.25 "
    "| facePCK@0.1 | facePCK@2px | face recall@0.25 <8/8-16/16-32/>=32 | img/s | coco ΔAP |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def upsert_summary(path: Path, rows: list[tuple[str, str]]) -> None:
    """Replace/insert rows (keyed by run|size|preds) inside the managed table block."""

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _SUMMARY_BEGIN not in text:
        text = (text.rstrip() + "\n\n" if text.strip() else "# B0 baseline – XTX3 re-measured with the XTX4 evaluator\n\n")
        text += f"{_SUMMARY_BEGIN}\n{_SUMMARY_HEADER}\n{_SUMMARY_END}\n"
    head, rest = text.split(_SUMMARY_BEGIN, 1)
    block, tail = rest.split(_SUMMARY_END, 1)
    existing: dict[str, str] = {}
    order: list[str] = []
    for line in block.strip().splitlines()[2:]:
        if line.startswith("| ") and "<!--key:" in line:
            key = line.split("<!--key:", 1)[1].split("-->", 1)[0]
            existing[key] = line
            order.append(key)
    for key, row in rows:
        if key not in existing:
            order.append(key)
        existing[key] = f"{row} <!--key:{key}-->"
    body = "\n".join(existing[k] for k in order)
    text = f"{head}{_SUMMARY_BEGIN}\n{_SUMMARY_HEADER}\n{body}\n{_SUMMARY_END}{tail}"
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-kind", choices=("xtx3", "xtx4", "ncnn"), required=True)
    p.add_argument("--weights", type=Path, default=None, help="checkpoint (xtx3/xtx4)")
    p.add_argument("--manifest", type=Path, default=None, help="ncnn: export manifest.json (or its directory)")
    p.add_argument("--param", type=Path, default=None, help="ncnn: .ncnn.param (with --bin, instead of --manifest)")
    p.add_argument("--bin", type=Path, default=None, help="ncnn: .ncnn.bin")
    p.add_argument("--ncnn-threads", type=int, default=4)
    p.add_argument("--ncnn-fp32", action="store_true", help="ncnn: disable fp16 storage/arithmetic")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", choices=("test", "train"), default="test")
    p.add_argument("--sizes", type=int, nargs="+", default=[320])
    p.add_argument("--view-mode", choices=("whole", "pyramid"), default="whole")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--score-threshold", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coco-check", type=int, default=0, help="pycocotools cross-check on the first N images")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=6, help="image decoding threads")
    p.add_argument("--name", default=None, help="run label (default: weights parent dir name)")
    p.add_argument("--save-preds", type=Path, default=None, help="npz cache; '{size}' is substituted")
    p.add_argument("--load-preds", type=Path, default=None, help="evaluate a cached npz ('{size}' substituted)")
    p.add_argument("--thresholds", type=float, nargs="*", default=None,
                   help="extra confidence thresholds for the sweep (0.05..0.95 step 0.05 always included)")
    p.add_argument("--ref-fp-per-image", type=float, default=None,
                   help="FP/img target of the fp_match operating point (default: B0 xtx3-u@320 @0.25 = 0.851)")
    p.add_argument("--oks-nms", type=float, default=None, metavar="THR",
                   help="apply host greedy OKS-NMS (xtx4.runtime.decode.oks_nms) at THR to every image's "
                        "detections before the metrics (default: off)")
    p.add_argument("--save-preds-limit", type=int, default=None, help="cache only the first N images")
    p.add_argument("--attach-to", type=Path, default=None,
                   help="with --load-preds/--coco-check: add the coco_check block to this existing report "
                        "(matching size) instead of writing a new one")
    p.add_argument("--summary-md", type=Path, default=REPO_ROOT / "reports" / "B0_summary.md")
    p.add_argument("--no-summary", action="store_true")
    return p.parse_args(argv)


def attach_coco_check(args: argparse.Namespace, gtset: GTSet) -> int:
    """Compute only the pycocotools cross-check from a (small) cache and merge it into a report."""

    target = args.attach_to if args.attach_to.is_absolute() else REPO_ROOT / args.attach_to
    report = json.loads(target.read_text(encoding="utf-8"))
    rows = []
    for size in args.sizes:
        result, _ = load_preds(Path(str(args.load_preds).replace("{size}", str(size))))
        n = min(args.coco_check, len(result["preds"]), len(gtset.gts))
        block = metrics_block(result["preds"][:n], gtset.gts[:n], args, coco_n=n)["coco_check"]
        for run in report["runs"]:
            if run["size"] == size:
                run["api"]["coco_check"] = _clean(block)
                rows.append(_summary_row(run, "api"))
        logger.info("S=%d coco check on %d images: delta %s", size, n, block.get("delta"))
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_summary:
        summary = args.summary_md if args.summary_md.is_absolute() else REPO_ROOT / args.summary_md
        upsert_summary(summary, rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if args.model_kind == "ncnn":
        if args.manifest is None and (args.param is None or args.bin is None):
            raise SystemExit("--model-kind ncnn needs --manifest or --param + --bin")
        src = args.manifest if args.manifest is not None else args.param
        src_dir = src if src.is_dir() else src.parent
        default_name = f"ncnn:{src_dir.parent.name}/{src_dir.name}"
    else:
        if args.weights is None:
            raise SystemExit(f"--model-kind {args.model_kind} needs --weights")
        default_name = f"{args.model_kind}:{args.weights.parent.name}"
    name = args.name or default_name
    t_gt = time.perf_counter()
    gtset = load_ground_truth(args.dataset, args.split, args.max_images)
    logger.info("Ground truth: %d images, %d persons (%.1fs)", len(gtset.gts),
                sum(g.keypoints.shape[0] for g in gtset.gts), time.perf_counter() - t_gt)
    if args.attach_to:
        if not (args.load_preds and args.coco_check):
            raise SystemExit("--attach-to needs --load-preds and --coco-check")
        return attach_coco_check(args, gtset)

    report: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_kind": args.model_kind,
        "weights": None if args.weights is None else str(args.weights),
        "weights_sha256": _sha256(args.weights) if args.weights is not None and args.weights.exists() else None,
        "ncnn_manifest": None if args.manifest is None else str(args.manifest),
        "dataset": str(args.dataset),
        "split": args.split,
        "view_mode": args.view_mode,
        "max_images": args.max_images,
        "num_images": len(gtset.gts),
        "num_persons": int(sum(g.keypoints.shape[0] for g in gtset.gts)),
        "score_threshold": args.score_threshold,
        "max_det": args.max_det,
        "device": args.device,
        "batch_size": args.batch_size,
        "area_source": "bbox",
        "gt_loader": "xtx3.data.annotations.scan_split + denormalise_sample",
        "image_order": "scan_split order (same as XTX3 trainer; first N when --max-images)",
        "net_scale": "min(1, S / max(w, h)) of the whole-frame letterbox",
        "metric_notes": {
            "sweep": "detection P/R/F1 + FP split per conf threshold from one greedy matching (OKS>=0.5)",
            "f1_max": "sweep threshold with max F1 (ties -> lower threshold)",
            "fp_match": "sweep threshold whose FP/img is closest to ref_fp_per_image (ties -> higher threshold)",
            "visibility_calibrated": "(present, visible) thresholds 0.1..0.9 maximising visibility macro-F1 on "
                                     "pairs matched at conf=f1_max threshold; null without probabilities",
            "AP9_project": "OKS 9 sigmas, thresholds 50:5:95, 101-pt, maxDet=max_det, score>=score_threshold",
            "AP9_coco20": "same with maxDet 20 (pycocotools default for keypoints)",
            "gt_without_labeled_points": "ignored (COCO num_keypoints==0 rule), counted in num_ignored_gt",
            "duplicates": "FP with OKS>=0.5 to a GT already matched by a higher-scored detection",
        },
        "runs": [],
    }
    summary_rows: list[tuple[str, str]] = []
    for size in args.sizes:
        for gt, (w, h) in zip(gtset.gts, gtset.sizes):
            gt.net_scale = min(1.0, size / float(max(w, h)))
        if args.load_preds:
            path = Path(str(args.load_preds).replace("{size}", str(size)))
            result, meta = load_preds(path)
            if len(result["preds"]) != len(gtset.gts):
                raise SystemExit(f"{path}: {len(result['preds'])} cached images vs {len(gtset.gts)} GT images")
            logger.info("Loaded cached predictions %s (S=%d)", path, size)
        else:
            if args.model_kind == "ncnn":
                runner = NcnnRunner(size=size, manifest=args.manifest, param=args.param, bin_path=args.bin,
                                    num_threads=args.ncnn_threads, fp16=False if args.ncnn_fp32 else None,
                                    view_mode=args.view_mode)
                ncnn_info = runner.describe()
                ncnn_info["bin_sha256"] = _sha256(Path(ncnn_info["bin"]))
            else:
                runner_cls = XTX3Runner if args.model_kind == "xtx3" else XTX4Runner
                runner = runner_cls(args.weights, device=args.device, view_mode=args.view_mode, size=size)
                ncnn_info = None
            logger.info("Inference %s S=%d mode=%s on %d images", name, size, args.view_mode, len(gtset.gts))
            result = run_inference(runner, gtset, conf=args.score_threshold, max_det=args.max_det,
                                   batch_size=args.batch_size, workers=args.workers)
            logger.info("Inference done: %.1fs, %.1f img/s", result["wall_s"], result["images_per_s"])
            if args.save_preds:
                path = Path(str(args.save_preds).replace("{size}", str(size)))
                save_preds(path, result, {"name": name, "size": size, "view_mode": args.view_mode,
                                          "weights": str(args.weights)}, limit=args.save_preds_limit)
            del runner
        nms_stats = None
        if args.oks_nms is not None:
            result["preds"], nms_stats = apply_oks_nms(result["preds"], args.oks_nms)
            if result.get("raw") is not None:
                result["raw"], _ = apply_oks_nms(result["raw"], args.oks_nms)
            logger.info("Host OKS-NMS @%.2f: %d -> %d detections", args.oks_nms,
                        nms_stats["dets_before"], nms_stats["dets_after"])
        run: dict[str, Any] = {
            "name": name,
            "size": size,
            "view_mode": args.view_mode,
            "num_images": len(gtset.gts),
            "wall_s": result["wall_s"],
            "images_per_s": result["images_per_s"],
            "image_size_mismatch": result["size_mismatch"],
            "api_parity": result["api_parity"],
            "oks_nms_threshold": args.oks_nms,
            "ncnn": ncnn_info if not args.load_preds else None,
            "oks_nms": nms_stats,
            "api": metrics_block(result["preds"], gtset.gts, args, coco_n=args.coco_check),
        }
        if result.get("raw") is not None:
            run["raw_o2o"] = metrics_block(result["raw"], gtset.gts, args, coco_n=0)
        report["runs"].append(run)
        ap = run["api"]["AP9_project"]
        logger.info("S=%d  AP=%.4f AP50=%.4f AP75=%.4f AR=%.4f", size, ap["AP"], ap["AP50"], ap["AP75"], ap["AR"])
        summary_rows.append(_summary_row(_clean(run), "api"))
        if "raw_o2o" in run:
            summary_rows.append(_summary_row(_clean(run), "raw_o2o"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    logger.info("Report written to %s", out_path)
    if not args.no_summary:
        summary = args.summary_md if args.summary_md.is_absolute() else REPO_ROOT / args.summary_md
        upsert_summary(summary, summary_rows)
        logger.info("Summary updated: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
