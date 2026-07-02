"""COCO-style annotation scanning, parsing, and validation (spec section 8).

Recursively scans every ``*.json`` under a split folder, aggregates ``images`` and
``annotations`` across files (image ids are namespaced per JSON file so ids never
collide across files), groups annotations by ``image_id`` and resolves image paths
relative to each JSON's directory. Bounding boxes and keypoints are stored
**normalised** (``[0, 1]``) exactly as provided; denormalisation happens in the
dataset target transform. Malformed records are warned about and skipped, never
crashing the scan (section 8.2 / 16).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.keypoints import NUM_KEYPOINTS
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PersonAnnotation:
    """A single person: normalised bbox (xywh) + normalised keypoints (K, 3)."""

    bbox_xywh_norm: np.ndarray  # (4,) float32, normalised [x, y, w, h]
    keypoints_norm: np.ndarray  # (K, 3) float32, [x_norm, y_norm, vis in {0,1,2}]
    iscrowd: int = 0


@dataclass(slots=True)
class ImageSample:
    """An image plus all of its person annotations."""

    image_path: Path
    width: int
    height: int
    annotations: list[PersonAnnotation] = field(default_factory=list)


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_keypoints(raw: object, *, num_keypoints: int) -> np.ndarray | None:
    if not isinstance(raw, (list, tuple)):
        return None
    expected = num_keypoints * 3
    if len(raw) != expected:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(num_keypoints, 3)
    except (ValueError, TypeError):
        return None
    # Clamp normalised coordinates into range, sanitise visibility to {0,1,2}.
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    arr[:, 2] = np.clip(np.round(arr[:, 2]), 0, 2)
    return arr


def _parse_bbox(raw: object) -> np.ndarray | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(arr).all():
        return None
    return np.clip(arr, 0.0, 1.0)


def _parse_single_json(json_path: Path, *, num_keypoints: int) -> dict[Path, ImageSample]:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Skipping JSON with unexpected top-level type: %s", json_path)
        return {}

    images_raw = data.get("images", [])
    annotations_raw = data.get("annotations", [])
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list):
        logger.warning("Skipping JSON missing images/annotations lists: %s", json_path)
        return {}

    # image ids are local to this JSON file (namespaced), avoiding collisions
    # between files that reuse the same numeric ids (section 8.2).
    id_to_sample: dict[int, ImageSample] = {}
    for image in images_raw:
        if not isinstance(image, dict):
            continue
        image_id = _coerce_int(image.get("id"), default=-1)
        file_name = image.get("file_name")
        if image_id < 0 or not isinstance(file_name, str):
            logger.warning("Skipping image record without valid id/file_name in %s", json_path)
            continue
        image_path = (json_path.parent / file_name).resolve()
        if not image_path.exists():
            logger.warning("Image file missing, skipping: %s", image_path)
            continue
        id_to_sample[image_id] = ImageSample(
            image_path=image_path,
            width=_coerce_int(image.get("width"), default=0),
            height=_coerce_int(image.get("height"), default=0),
        )

    for ann in annotations_raw:
        if not isinstance(ann, dict):
            continue
        image_id = _coerce_int(ann.get("image_id"), default=-1)
        sample = id_to_sample.get(image_id)
        if sample is None:
            logger.warning("Annotation references unknown image_id=%s in %s", image_id, json_path)
            continue
        bbox = _parse_bbox(ann.get("bbox"))
        keypoints = _parse_keypoints(ann.get("keypoints"), num_keypoints=num_keypoints)
        if bbox is None or keypoints is None:
            logger.warning("Skipping malformed annotation (image_id=%s) in %s", image_id, json_path)
            continue
        sample.annotations.append(
            PersonAnnotation(
                bbox_xywh_norm=bbox,
                keypoints_norm=keypoints,
                iscrowd=_coerce_int(ann.get("iscrowd"), default=0),
            )
        )

    return {sample.image_path: sample for sample in id_to_sample.values()}


def scan_split(split_dir: Path, *, num_keypoints: int = NUM_KEYPOINTS) -> list[ImageSample]:
    """Recursively scan ``split_dir`` for ``*.json`` and return aggregated samples.

    Supports both one-JSON-per-image and one JSON covering many images. Images
    appearing in multiple JSON files are merged (annotations concatenated).
    """

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    merged: dict[Path, ImageSample] = {}
    json_files = sorted(split_dir.rglob("*.json"))
    if not json_files:
        logger.warning("No JSON annotation files found under %s", split_dir)

    for json_path in json_files:
        for path, sample in _parse_single_json(json_path, num_keypoints=num_keypoints).items():
            if path in merged:
                merged[path].annotations.extend(sample.annotations)
            else:
                merged[path] = sample

    samples = list(merged.values())
    logger.info(
        "Scanned %d JSON files -> %d images, %d people",
        len(json_files),
        len(samples),
        sum(len(s.annotations) for s in samples),
    )
    return samples


def dataset_statistics(samples: list[ImageSample], *, num_keypoints: int = NUM_KEYPOINTS) -> dict:
    """Compute summary stats: #images, #people, per-keypoint visibility histogram."""

    num_people = 0
    vis_hist = np.zeros((num_keypoints, 3), dtype=np.int64)
    for sample in samples:
        for ann in sample.annotations:
            num_people += 1
            vis = np.clip(ann.keypoints_norm[:, 2].astype(np.int64), 0, 2)
            for k in range(num_keypoints):
                vis_hist[k, vis[k]] += 1
    return {
        "num_images": len(samples),
        "num_people": num_people,
        "visibility_histogram": vis_hist,
    }
