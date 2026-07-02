"""Annotation scanning/parsing/validation tests (spec section 8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xtx2.data.annotations import dataset_statistics, scan_split


def _keypoints(vis: int = 2) -> list[float]:
    values: list[float] = []
    for k in range(17):
        values.extend([0.1 + 0.02 * k, 0.2 + 0.02 * k, float(vis)])
    return values


def _write_image(path: Path) -> None:
    import cv2

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    split = tmp_path / "train"
    split.mkdir()
    return split


class TestScan:
    def test_single_json_many_images(self, split_dir: Path) -> None:
        _write_image(split_dir / "a.jpg")
        _write_image(split_dir / "b.jpg")
        _write_json(
            split_dir / "all.json",
            {
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 64, "height": 48},
                    {"id": 2, "file_name": "b.jpg", "width": 64, "height": 48},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1,
                     "bbox": [0.1, 0.1, 0.3, 0.5], "keypoints": _keypoints()},
                    {"id": 2, "image_id": 2, "category_id": 1,
                     "bbox": [0.2, 0.2, 0.3, 0.5], "keypoints": _keypoints(vis=1)},
                ],
            },
        )
        samples = scan_split(split_dir)
        assert len(samples) == 2
        assert sum(len(s.annotations) for s in samples) == 2

    def test_one_json_per_image(self, split_dir: Path) -> None:
        for name in ("a", "b"):
            _write_image(split_dir / f"{name}.jpg")
            _write_json(
                split_dir / f"{name}.json",
                {
                    "images": [{"id": 1, "file_name": f"{name}.jpg", "width": 64, "height": 48}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "bbox": [0.1, 0.1, 0.2, 0.2],
                         "keypoints": _keypoints()}
                    ],
                },
            )
        samples = scan_split(split_dir)
        assert len(samples) == 2
        # Same numeric image id 1 in both files must not collide.
        assert all(len(s.annotations) == 1 for s in samples)

    def test_recursive_scan(self, split_dir: Path) -> None:
        nested = split_dir / "sub" / "deeper"
        nested.mkdir(parents=True)
        _write_image(nested / "a.jpg")
        _write_json(
            nested / "a.json",
            {
                "images": [{"id": 5, "file_name": "a.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 5, "bbox": [0, 0, 1, 1], "keypoints": _keypoints()}
                ],
            },
        )
        samples = scan_split(split_dir)
        assert len(samples) == 1

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_split(tmp_path / "does-not-exist")


class TestValidation:
    def test_missing_image_skipped(self, split_dir: Path) -> None:
        _write_json(
            split_dir / "x.json",
            {
                "images": [{"id": 1, "file_name": "missing.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _keypoints()}
                ],
            },
        )
        assert scan_split(split_dir) == []

    def test_dangling_image_id_skipped(self, split_dir: Path) -> None:
        _write_image(split_dir / "a.jpg")
        _write_json(
            split_dir / "a.json",
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 999, "bbox": [0, 0, 1, 1], "keypoints": _keypoints()}
                ],
            },
        )
        samples = scan_split(split_dir)
        assert len(samples) == 1
        assert samples[0].annotations == []

    def test_wrong_keypoint_count_skipped(self, split_dir: Path) -> None:
        _write_image(split_dir / "a.jpg")
        _write_json(
            split_dir / "a.json",
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1],
                     "keypoints": [0.5, 0.5, 2]},  # only 1 keypoint, expected 17
                ],
            },
        )
        samples = scan_split(split_dir)
        assert samples[0].annotations == []

    def test_out_of_range_values_clamped(self, split_dir: Path) -> None:
        _write_image(split_dir / "a.jpg")
        kpts = _keypoints()
        kpts[0] = 1.7  # x out of range
        kpts[2] = 9.0  # visibility out of range
        _write_json(
            split_dir / "a.json",
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [-0.5, 0, 2.0, 1], "keypoints": kpts}
                ],
            },
        )
        samples = scan_split(split_dir)
        ann = samples[0].annotations[0]
        assert 0.0 <= ann.keypoints_norm[:, 0].max() <= 1.0
        assert ann.keypoints_norm[:, 2].max() <= 2
        assert 0.0 <= ann.bbox_xywh_norm.min() and ann.bbox_xywh_norm.max() <= 1.0

    def test_corrupt_json_skipped(self, split_dir: Path) -> None:
        (split_dir / "bad.json").write_text("{not json", encoding="utf-8")
        assert scan_split(split_dir) == []


class TestStatistics:
    def test_visibility_histogram(self, split_dir: Path) -> None:
        _write_image(split_dir / "a.jpg")
        _write_json(
            split_dir / "a.json",
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _keypoints(vis=2)},
                    {"id": 2, "image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _keypoints(vis=0)},
                ],
            },
        )
        stats = dataset_statistics(scan_split(split_dir))
        assert stats["num_images"] == 1
        assert stats["num_people"] == 2
        hist = stats["visibility_histogram"]
        assert hist[:, 2].sum() == 17  # one fully visible person
        assert hist[:, 0].sum() == 17  # one fully absent person
