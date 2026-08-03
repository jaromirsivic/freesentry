"""COCO-style annotation scanning, parsing, validation, and 17 -> 9 selection.

Recursively scans every ``*.json`` under a split folder, aggregates ``images``
and ``annotations`` across files (image ids are namespaced per JSON file so ids
never collide across files), groups annotations by ``image_id`` and resolves
image paths relative to each JSON's directory.

Two source formats are accepted per annotation, auto-detected:

* **normalised** (the local XTX layout): bbox ``[x, y, w, h]`` and keypoint
  coordinates in ``[0, 1]``;
* **absolute** (official COCO ``person_keypoints_*``): bbox and keypoint
  coordinates in pixels -- detected when any coordinate exceeds 1.5 -- and
  normalised here using the image width/height.

Keypoints may come as COCO's 17 points (the nine XTX3 landmarks are selected
deterministically via :data:`~xtx3.utils.keypoints.COCO_KEYPOINT_INDICES`,
preserving left/right semantics) or already as 9 points. Point order is never
silently reinterpreted: any other count is reported and skipped.

Malformed records are warned about, counted, and skipped -- the scan never
crashes, and a summary of skipped records is logged at the end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.keypoints import NUM_COCO_KEYPOINTS, NUM_KEYPOINTS, select_nine
from ..utils.logging import get_logger

logger = get_logger(__name__)

_ABSOLUTE_THRESHOLD = 1.5  # normalised values are <= 1.0; anything above is pixels


@dataclass(slots=True)
class PersonAnnotation:
    """A single person: normalised bbox (xywh) + normalised keypoints (9, 3)."""

    bbox_xywh_norm: np.ndarray  # (4,) float32, normalised [x, y, w, h]
    keypoints_norm: np.ndarray  # (9, 3) float32, [x_norm, y_norm, vis in {0,1,2}]
    iscrowd: int = 0


@dataclass(slots=True)
class ImageSample:
    """An image plus all of its person annotations."""

    image_path: Path
    width: int
    height: int
    annotations: list[PersonAnnotation] = field(default_factory=list)


@dataclass(slots=True)
class ScanReport:
    """Counters describing what the scan skipped and why."""

    json_files: int = 0
    unreadable_json: int = 0
    missing_images: int = 0
    bad_image_records: int = 0
    bad_annotations: int = 0
    non_person: int = 0

    def log(self) -> None:
        if self.unreadable_json or self.missing_images or self.bad_image_records or self.bad_annotations:
            logger.warning(
                "Scan skipped records: unreadable_json=%d missing_images=%d "
                "bad_image_records=%d bad_annotations=%d (non-person filtered: %d)",
                self.unreadable_json, self.missing_images,
                self.bad_image_records, self.bad_annotations, self.non_person,
            )


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_keypoints(raw: object, *, width: int, height: int) -> np.ndarray | None:
    """Parse keypoints into a normalised ``(9, 3)`` array, or None when invalid.

    Accepts 17-point COCO input (selects the nine retained landmarks) or
    nine-point input. Absolute pixel coordinates are normalised by the image
    size (which must then be known).
    """

    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) == NUM_COCO_KEYPOINTS * 3:
        source_points = NUM_COCO_KEYPOINTS
    elif len(raw) == NUM_KEYPOINTS * 3:
        source_points = NUM_KEYPOINTS
    else:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(source_points, 3)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(arr).all():
        return None
    if source_points == NUM_COCO_KEYPOINTS:
        arr = select_nine(arr)

    if float(np.abs(arr[:, :2]).max(initial=0.0)) > _ABSOLUTE_THRESHOLD:
        # Absolute pixel coordinates (official COCO): normalise by image size.
        if width <= 0 or height <= 0:
            return None
        arr = arr.copy()
        arr[:, 0] /= float(width)
        arr[:, 1] /= float(height)

    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
    arr[:, 2] = np.clip(np.round(arr[:, 2]), 0, 2)
    return arr


def _parse_bbox(raw: object, *, width: int, height: int) -> np.ndarray | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(arr).all():
        return None
    if float(np.abs(arr).max(initial=0.0)) > _ABSOLUTE_THRESHOLD:
        # Absolute [x, y, w, h] in pixels (official COCO): normalise.
        if width <= 0 or height <= 0:
            return None
        arr = arr / np.array([width, height, width, height], dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _parse_single_json(json_path: Path, report: ScanReport) -> dict[Path, ImageSample]:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
        report.unreadable_json += 1
        return {}

    if not isinstance(data, dict):
        logger.warning("Skipping JSON with unexpected top-level type: %s", json_path)
        report.unreadable_json += 1
        return {}

    images_raw = data.get("images", [])
    annotations_raw = data.get("annotations", [])
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list):
        logger.warning("Skipping JSON missing images/annotations lists: %s", json_path)
        report.unreadable_json += 1
        return {}

    # image ids are local to this JSON file (namespaced), avoiding collisions
    # between files that reuse the same numeric ids.
    id_to_sample: dict[int, ImageSample] = {}
    for image in images_raw:
        if not isinstance(image, dict):
            report.bad_image_records += 1
            continue
        image_id = _coerce_int(image.get("id"), default=-1)
        file_name = image.get("file_name")
        if image_id < 0 or not isinstance(file_name, str):
            logger.warning("Skipping image record without valid id/file_name in %s", json_path)
            report.bad_image_records += 1
            continue
        image_path = (json_path.parent / file_name).resolve()
        if not image_path.exists():
            logger.warning("Image file missing, skipping: %s", image_path)
            report.missing_images += 1
            continue
        id_to_sample[image_id] = ImageSample(
            image_path=image_path,
            width=_coerce_int(image.get("width"), default=0),
            height=_coerce_int(image.get("height"), default=0),
        )

    for ann in annotations_raw:
        if not isinstance(ann, dict):
            report.bad_annotations += 1
            continue
        category_id = ann.get("category_id")
        if category_id is not None and _coerce_int(category_id, default=1) != 1:
            report.non_person += 1
            continue
        image_id = _coerce_int(ann.get("image_id"), default=-1)
        sample = id_to_sample.get(image_id)
        if sample is None:
            logger.warning("Annotation references unknown image_id=%s in %s", image_id, json_path)
            report.bad_annotations += 1
            continue
        bbox = _parse_bbox(ann.get("bbox"), width=sample.width, height=sample.height)
        keypoints = _parse_keypoints(ann.get("keypoints"), width=sample.width, height=sample.height)
        if bbox is None or keypoints is None:
            logger.warning("Skipping malformed annotation (image_id=%s) in %s", image_id, json_path)
            report.bad_annotations += 1
            continue
        sample.annotations.append(
            PersonAnnotation(
                bbox_xywh_norm=bbox,
                keypoints_norm=keypoints,
                iscrowd=_coerce_int(ann.get("iscrowd"), default=0),
            )
        )

    return {sample.image_path: sample for sample in id_to_sample.values()}


def scan_split(split_dir: Path) -> list[ImageSample]:
    """Recursively scan ``split_dir`` for ``*.json`` and return aggregated samples.

    Supports both one-JSON-per-image and one JSON covering many images. Images
    appearing in multiple JSON files are merged (annotations concatenated).
    """

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    report = ScanReport()
    merged: dict[Path, ImageSample] = {}
    json_files = sorted(split_dir.rglob("*.json"))
    report.json_files = len(json_files)
    if not json_files:
        logger.warning("No JSON annotation files found under %s", split_dir)

    for json_path in json_files:
        for path, sample in _parse_single_json(json_path, report).items():
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
    report.log()
    return samples


def dataset_statistics(samples: list[ImageSample]) -> dict:
    """Compute summary stats: #images, #people, per-keypoint visibility histogram."""

    num_people = 0
    vis_hist = np.zeros((NUM_KEYPOINTS, 3), dtype=np.int64)
    for sample in samples:
        for ann in sample.annotations:
            num_people += 1
            vis = np.clip(ann.keypoints_norm[:, 2].astype(np.int64), 0, 2)
            for k in range(NUM_KEYPOINTS):
                vis_hist[k, vis[k]] += 1
    return {
        "num_images": len(samples),
        "num_people": num_people,
        "visibility_histogram": vis_hist,
    }
