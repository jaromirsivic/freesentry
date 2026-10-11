"""Dataset scanning and conversion of heterogeneous pose annotations to the nine XTX landmarks.

Every supported source format is converted once, at scan time, into the same
in-memory representation (:class:`ImageSample` / :class:`PersonAnnotation`)
with **normalised** coordinates (``x / width``, ``y / height``) so that image
caching (re-encoding at a smaller size) never invalidates annotations.

Why a ``keypoint_map``: partial-label datasets (CrowdPose, AI Challenger, MPII)
annotate shoulders and hips but no face points. Those landmarks are not
"absent" -- the dataset simply never labels them -- so they get ``v = -1``
(*unknown*, CONTRACTS §2) and are ignored by every loss term, instead of being
trained as "not present" (XTX3 defect D5).

Supported formats (``DatasetSpec.format``):

* ``coco17``  -- official COCO ``person_keypoints_*`` (absolute px) and the XTX
  normalised layout (``D:/xtxtraining``); 17 points, nine selected via
  :func:`xtx4.utils.keypoints.select_nine`. A 27-value keypoint list is
  accepted as an already-selected nine-point record.
* ``xtx9``    -- already nine points in XTX order.
* ``coco_wholebody`` -- 133 points (or COCO-WholeBody's separate 17-point
  ``keypoints`` field); body indices 0-16 are COCO order.
* ``crowdpose`` -- 14 points (shoulders/hips mapped, face ``v = -1``).
* ``aic``     -- AI Challenger, 14 points (same idea; raw AIC visibility
  ``1 visible / 2 occluded / 3 unlabeled`` can be remapped via
  ``DatasetSpec.visibility_map``).
* ``mpii``    -- 16 points, either COCO-style JSON or the mmpose list layout
  (``[{"image", "joints", "joints_vis"}, ...]``); labeled MPII joints become
  ``v = 2`` (MPII does not separate occlusion in the mmpose release).

Absolute vs normalised coordinates are detected **per JSON file** (any bbox or
labeled keypoint coordinate above 1.5 => absolute pixels) and the decision is
logged, so a mixed directory never silently mis-scales one file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable

import numpy as np

from ..utils.keypoints import COCO_KEYPOINT_INDICES, NUM_KEYPOINTS
from ..utils.logging import get_logger

logger = get_logger(__name__)

_ABSOLUTE_THRESHOLD: Final[float] = 1.5  # normalised values are <= 1.0 (+ rounding slack)

FORMATS: Final[tuple[str, ...]] = ("coco17", "xtx9", "coco_wholebody", "crowdpose", "aic", "mpii")

# Source keypoint count per format.
SOURCE_POINTS: Final[dict[str, int]] = {
    "coco17": 17,
    "xtx9": 9,
    "coco_wholebody": 133,
    "crowdpose": 14,
    "aic": 14,
    "mpii": 16,
}

# For each of the nine XTX landmarks (nose, l_eye, r_eye, l_ear, r_ear, l_sho,
# r_sho, l_hip, r_hip) the source index, or None when the dataset never labels it.
DEFAULT_KEYPOINT_MAPS: Final[dict[str, tuple[int | None, ...]]] = {
    "coco17": tuple(COCO_KEYPOINT_INDICES),
    "xtx9": tuple(range(NUM_KEYPOINTS)),
    "coco_wholebody": tuple(COCO_KEYPOINT_INDICES),
    # CrowdPose: 0 l_sho, 1 r_sho, 2 l_elb, 3 r_elb, 4 l_wri, 5 r_wri, 6 l_hip, 7 r_hip, ...
    "crowdpose": (None, None, None, None, None, 0, 1, 6, 7),
    # AIC: 0 r_sho, 1 r_elb, 2 r_wri, 3 l_sho, 4 l_elb, 5 l_wri, 6 r_hip, 7 r_knee, 8 r_ank, 9 l_hip, ...
    "aic": (None, None, None, None, None, 3, 0, 9, 6),
    # MPII: 2 r_hip, 3 l_hip, 12 r_sho, 13 l_sho
    "mpii": (None, None, None, None, None, 13, 12, 3, 2),
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """How to read one dataset.

    Attributes:
        name: dataset name recorded in samples / manifests (e.g. ``"xtx"``).
        format: one of :data:`FORMATS`.
        keypoint_map: length-9 tuple, source index per XTX landmark or ``None``
            (=> ``v = -1`` for every person). ``None`` uses the format default.
        visibility_map: optional remap of raw visibility values (applied before
            clipping to ``{0, 1, 2}``), e.g. raw AIC ``{1: 2, 2: 1, 3: 0}``.
    """

    name: str = "xtx"
    format: str = "coco17"
    keypoint_map: tuple[int | None, ...] | None = None
    visibility_map: dict[int, int] | None = None

    def resolved_map(self) -> tuple[int | None, ...]:
        if self.format not in FORMATS:
            raise ValueError(f"Unknown dataset format {self.format!r}; expected one of {FORMATS}")
        mapping = self.keypoint_map if self.keypoint_map is not None else DEFAULT_KEYPOINT_MAPS[self.format]
        mapping = tuple(None if m is None else int(m) for m in mapping)
        if len(mapping) != NUM_KEYPOINTS:
            raise ValueError(f"keypoint_map must have {NUM_KEYPOINTS} entries, got {len(mapping)}")
        n_src = SOURCE_POINTS[self.format]
        for m in mapping:
            if m is not None and not 0 <= m < n_src:
                raise ValueError(f"keypoint_map index {m} out of range for {self.format} ({n_src} points)")
        return mapping

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSpec":
        vis_map = data.get("visibility_map")
        kmap = data.get("keypoint_map")
        return cls(
            name=str(data.get("name", "xtx")),
            format=str(data.get("format", "coco17")),
            keypoint_map=None if kmap is None else tuple(None if m is None else int(m) for m in kmap),
            visibility_map=None if vis_map is None else {int(k): int(v) for k, v in vis_map.items()},
        )


DEFAULT_SPEC: Final[DatasetSpec] = DatasetSpec("xtx", "coco17", None)


@dataclass(slots=True)
class PersonAnnotation:
    """One person in normalised image coordinates.

    Attributes:
        keypoints_norm: ``(9, 3)`` float32 ``[x/W, y/H, v]`` with ``v ∈ {-1, 0, 1, 2}``.
        bbox_xywh_norm: ``(4,)`` float32 normalised ``[x, y, w, h]`` or ``None``.
        area_norm: segmentation area / (W·H) when the source provides it, else ``None``.
        iscrowd: 1 for crowd ignore regions (keypoints then all ``v = 0``).
    """

    keypoints_norm: np.ndarray
    bbox_xywh_norm: np.ndarray | None = None
    area_norm: float | None = None
    iscrowd: int = 0

    @property
    def num_labeled(self) -> int:
        return int((self.keypoints_norm[:, 2] > 0).sum())


@dataclass(slots=True)
class ImageSample:
    """An image and its person annotations (``dataset`` = source dataset name)."""

    image_path: Path
    width: int
    height: int
    annotations: list[PersonAnnotation] = field(default_factory=list)
    dataset: str = "xtx"


@dataclass(slots=True)
class ScanReport:
    """Counters describing what the scan skipped and why."""

    json_files: int = 0
    unreadable_json: int = 0
    missing_images: int = 0
    bad_image_records: int = 0
    bad_annotations: int = 0
    non_person: int = 0
    unlabeled_skipped: int = 0
    crowd_kept: int = 0

    def log(self, name: str) -> None:
        logger.info(
            "[%s] scan report: json=%d unreadable=%d missing_images=%d bad_images=%d "
            "bad_annotations=%d non_person=%d unlabeled_skipped=%d crowd_regions=%d",
            name, self.json_files, self.unreadable_json, self.missing_images, self.bad_image_records,
            self.bad_annotations, self.non_person, self.unlabeled_skipped, self.crowd_kept,
        )


# --------------------------------------------------------------------------- helpers


def read_image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` reading only the header when Pillow is available."""

    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 - fall back to a full decode
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return 0, 0
        return int(img.shape[1]), int(img.shape[0])


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _source_array(raw: object, fmt: str) -> np.ndarray | None:
    """Raw keypoint list -> ``(P, 3)`` float array in source order (or None)."""

    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return None
    flat = raw
    if isinstance(raw[0], (list, tuple)):  # [[x, y, v], ...]
        flat = [c for p in raw for c in p]
    n = len(flat)
    if n % 3 != 0:
        return None
    points = n // 3
    expected = SOURCE_POINTS[fmt]
    allowed = {expected}
    if fmt == "coco17":
        allowed.add(NUM_KEYPOINTS)  # already selected nine-point record (XTX3 compatibility)
    if fmt == "coco_wholebody":
        allowed.add(17)  # separate COCO-WholeBody body field
    if points not in allowed:
        return None
    try:
        arr = np.asarray(flat, dtype=np.float64).reshape(points, 3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _map_to_nine(
    src: np.ndarray, fmt: str, mapping: tuple[int | None, ...], vis_map: dict[int, int] | None
) -> np.ndarray:
    """Select / map source points to ``(9, 3)``; unmapped landmarks get ``v = -1``."""

    out = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
    if fmt == "coco17" and src.shape[0] == NUM_KEYPOINTS:
        mapping = tuple(range(NUM_KEYPOINTS))
    vis = src[:, 2].copy()
    if vis_map:
        vis = np.array([vis_map.get(int(round(v)), v) for v in vis], dtype=np.float64)
    vis = np.clip(np.round(vis), 0, 2)
    for k, m in enumerate(mapping):
        if m is None:
            out[k, 2] = -1.0
        else:
            out[k, :2] = src[m, :2]
            out[k, 2] = vis[m]
    return out


@dataclass(slots=True)
class _RawPerson:
    kpts: np.ndarray  # (9,3) source units
    bbox: np.ndarray | None  # (4,) xywh source units
    area: float | None
    iscrowd: int


def _finalise(
    person: _RawPerson, *, width: int, height: int, absolute: bool, report: ScanReport,
    unlabeled_as_ignore: bool = False,
) -> PersonAnnotation | None:
    kpts = person.kpts.copy()
    bbox = None if person.bbox is None else person.bbox.astype(np.float64).copy()
    area = person.area
    if absolute:
        if width <= 0 or height <= 0:
            report.bad_annotations += 1
            return None
        kpts[:, 0] /= float(width)
        kpts[:, 1] /= float(height)
        if bbox is not None:
            bbox /= np.array([width, height, width, height], dtype=np.float64)
        if area is not None:
            area = float(area) / float(width * height)
    labeled = kpts[:, 2] > 0
    if unlabeled_as_ignore and not labeled.any() and not person.iscrowd and bbox is not None:
        person = _RawPerson(kpts=person.kpts, bbox=person.bbox, area=person.area, iscrowd=1)
    # Points marked unlabeled carry no coordinates; zero them for determinism.
    kpts[~labeled & (kpts[:, 2] == 0), :2] = 0.0
    kpts[labeled, 0] = np.clip(kpts[labeled, 0], 0.0, 1.0)
    kpts[labeled, 1] = np.clip(kpts[labeled, 1], 0.0, 1.0)
    if person.iscrowd:
        kpts[:, 2] = np.where(kpts[:, 2] < 0, -1.0, 0.0)
        kpts[:, :2] = 0.0
        if bbox is None:
            report.bad_annotations += 1
            return None
        report.crowd_kept += 1
    elif not labeled.any():
        report.unlabeled_skipped += 1
        return None
    return PersonAnnotation(
        keypoints_norm=kpts.astype(np.float32),
        bbox_xywh_norm=None if bbox is None else bbox.astype(np.float32),
        area_norm=None if area is None else float(area),
        iscrowd=int(person.iscrowd),
    )


def _parse_bbox(raw: object) -> np.ndarray | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all() or arr[2] < 0 or arr[3] < 0:
        return None
    return arr


def _detect_absolute(persons: Iterable[_RawPerson]) -> bool:
    peak = 0.0
    for p in persons:
        lab = p.kpts[:, 2] > 0
        if lab.any():
            peak = max(peak, float(np.abs(p.kpts[lab, :2]).max()))
        if p.bbox is not None:
            peak = max(peak, float(np.abs(p.bbox).max()))
        if peak > _ABSOLUTE_THRESHOLD:
            return True
    return False


# --------------------------------------------------------------------------- parsers


def _parse_coco_json(
    json_path: Path, data: dict, spec: DatasetSpec, report: ScanReport, check_files: bool,
    unlabeled_as_ignore: bool = False,
) -> list[ImageSample]:
    images_raw = data.get("images", [])
    annotations_raw = data.get("annotations", [])
    if not isinstance(images_raw, list) or not isinstance(annotations_raw, list):
        logger.warning("Skipping JSON missing images/annotations lists: %s", json_path)
        report.unreadable_json += 1
        return []
    fmt = spec.format
    mapping = spec.resolved_map()

    id_to_sample: dict[int, ImageSample] = {}
    for image in images_raw:
        if not isinstance(image, dict):
            report.bad_image_records += 1
            continue
        image_id = _coerce_int(image.get("id"), default=-1)
        file_name = image.get("file_name")
        if image_id < 0 or not isinstance(file_name, str):
            report.bad_image_records += 1
            continue
        image_path = (json_path.parent / file_name).resolve()
        if check_files and not image_path.exists():
            report.missing_images += 1
            continue
        id_to_sample[image_id] = ImageSample(
            image_path=image_path,
            width=_coerce_int(image.get("width"), default=0),
            height=_coerce_int(image.get("height"), default=0),
            dataset=spec.name,
        )

    raw_by_image: dict[int, list[_RawPerson]] = {}
    for ann in annotations_raw:
        if not isinstance(ann, dict):
            report.bad_annotations += 1
            continue
        category_id = ann.get("category_id")
        if category_id is not None and _coerce_int(category_id, default=1) != 1:
            report.non_person += 1
            continue
        image_id = _coerce_int(ann.get("image_id"), default=-1)
        if image_id not in id_to_sample:
            report.bad_annotations += 1
            continue
        iscrowd = _coerce_int(ann.get("iscrowd"), default=0)
        src = _source_array(ann.get("keypoints"), fmt)
        if src is None:
            if iscrowd:
                src = np.zeros((SOURCE_POINTS[fmt], 3), dtype=np.float64)
            else:
                report.bad_annotations += 1
                continue
        kpts = _map_to_nine(src, fmt, mapping, spec.visibility_map)
        area_raw = ann.get("area")
        area = float(area_raw) if isinstance(area_raw, (int, float)) and area_raw > 0 else None
        raw_by_image.setdefault(image_id, []).append(
            _RawPerson(kpts=kpts, bbox=_parse_bbox(ann.get("bbox")), area=area, iscrowd=1 if iscrowd else 0)
        )

    all_persons = [p for plist in raw_by_image.values() for p in plist]
    absolute = _detect_absolute(all_persons)
    logger.info(
        "[%s] %s: %d images, %d annotations -> coordinates detected as %s",
        spec.name, json_path.name, len(id_to_sample), len(all_persons),
        "ABSOLUTE pixels" if absolute else "NORMALISED [0,1]",
    )
    for image_id, plist in raw_by_image.items():
        sample = id_to_sample[image_id]
        if absolute and (sample.width <= 0 or sample.height <= 0):
            sample.width, sample.height = read_image_size(sample.image_path)
        for person in plist:
            ann = _finalise(person, width=sample.width, height=sample.height, absolute=absolute, report=report,
                            unlabeled_as_ignore=unlabeled_as_ignore)
            if ann is not None:
                sample.annotations.append(ann)
    return list(id_to_sample.values())


def _parse_mmpose_mpii(
    json_path: Path, data: list, spec: DatasetSpec, report: ScanReport, check_files: bool
) -> list[ImageSample]:
    """mmpose MPII layout: list of ``{"image", "joints" (16,2), "joints_vis" (16,)}``."""

    mapping = spec.resolved_map()
    by_image: dict[Path, ImageSample] = {}
    image_dir = json_path.parent / "images"
    for rec in data:
        if not isinstance(rec, dict) or not isinstance(rec.get("image"), str):
            report.bad_annotations += 1
            continue
        candidate = image_dir / rec["image"]
        image_path = (candidate if candidate.exists() else json_path.parent / rec["image"]).resolve()
        if image_path not in by_image:
            if check_files and not image_path.exists():
                report.missing_images += 1
                continue
            w, h = read_image_size(image_path) if image_path.exists() else (0, 0)
            by_image[image_path] = ImageSample(image_path=image_path, width=w, height=h, dataset=spec.name)
        sample = by_image[image_path]
        try:
            joints = np.asarray(rec.get("joints"), dtype=np.float64).reshape(16, 2)
            vis = np.asarray(rec.get("joints_vis"), dtype=np.float64).reshape(16)
        except (TypeError, ValueError):
            report.bad_annotations += 1
            continue
        src = np.concatenate([joints, np.where(vis > 0, 2.0, 0.0)[:, None]], axis=1)
        kpts = _map_to_nine(src, "mpii", mapping, spec.visibility_map)
        ann = _finalise(
            _RawPerson(kpts=kpts, bbox=None, area=None, iscrowd=0),
            width=sample.width, height=sample.height, absolute=True, report=report,
        )
        if ann is not None:
            sample.annotations.append(ann)
    logger.info("[%s] %s: mmpose MPII list layout -> coordinates ABSOLUTE pixels", spec.name, json_path.name)
    return list(by_image.values())


def _parse_json_file(
    json_path: Path, spec: DatasetSpec, report: ScanReport, check_files: bool, unlabeled_as_ignore: bool
) -> list[ImageSample]:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable JSON %s: %s", json_path, exc)
        report.unreadable_json += 1
        return []
    if isinstance(data, list) and spec.format == "mpii":
        return _parse_mmpose_mpii(json_path, data, spec, report, check_files)
    if not isinstance(data, dict) or "images" not in data:
        logger.warning("Skipping JSON without a COCO 'images' list: %s", json_path)
        report.unreadable_json += 1
        return []
    return _parse_coco_json(json_path, data, spec, report, check_files, unlabeled_as_ignore)


def scan_split(
    split_dir: Path | str,
    spec: DatasetSpec = DEFAULT_SPEC,
    *,
    check_files: bool = True,
    unlabeled_as_ignore: bool = False,
) -> list[ImageSample]:
    """Recursively scan ``split_dir`` for ``*.json`` and return samples sorted by path.

    Images appearing in several JSON files are merged (annotations concatenated
    in file order). Images without any kept person are retained (negatives).
    ``check_files`` verifies every referenced image exists (one ``stat`` each).
    ``unlabeled_as_ignore`` keeps persons with no labeled XTX landmark but a bbox
    as crowd-style ignore regions instead of dropping them (default: dropped).
    """

    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    spec.resolved_map()  # validate early

    report = ScanReport()
    merged: dict[Path, ImageSample] = {}
    json_files = sorted(split_dir.rglob("*.json"))
    report.json_files = len(json_files)
    if not json_files:
        logger.warning("No JSON annotation files found under %s", split_dir)
    for json_path in json_files:
        if json_path.name.startswith("_") or json_path.name.endswith(".sha1cache.json"):
            continue  # sidecar / manifest files written by our tools
        for sample in _parse_json_file(json_path, spec, report, check_files, unlabeled_as_ignore):
            if sample.image_path in merged:
                merged[sample.image_path].annotations.extend(sample.annotations)
            else:
                merged[sample.image_path] = sample

    samples = sorted(merged.values(), key=lambda s: str(s.image_path))
    logger.info(
        "[%s] scanned %s: %d JSON files -> %d images, %d persons",
        spec.name, split_dir, len(json_files), len(samples), sum(len(s.annotations) for s in samples),
    )
    report.log(spec.name)
    return samples


def dataset_statistics(samples: list[ImageSample]) -> dict:
    """Summary: images, persons, per-landmark histogram over v ∈ {-1, 0, 1, 2}."""

    vis_hist = np.zeros((NUM_KEYPOINTS, 4), dtype=np.int64)
    persons = crowd = 0
    for sample in samples:
        for ann in sample.annotations:
            if ann.iscrowd:
                crowd += 1
                continue
            persons += 1
            v = ann.keypoints_norm[:, 2].astype(np.int64) + 1
            vis_hist[np.arange(NUM_KEYPOINTS), v] += 1
    return {"num_images": len(samples), "num_persons": persons, "num_crowd": crowd,
            "visibility_histogram": vis_hist}
