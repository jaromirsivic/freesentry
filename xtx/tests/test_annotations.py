"""COCO-style JSON parsing/validation (spec section 8)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xtx.data.annotations import dataset_statistics, scan_split


def _write_image(path: Path, width: int = 1280, height: int = 720) -> None:
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def _keypoints() -> list[float]:
    out: list[float] = []
    for _ in range(17):
        out += [0.5, 0.5, 2]
    return out


def test_scan_split_basic(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    _write_image(split / "img0.jpg")
    (split / "img0.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img0.jpg", "width": 1280, "height": 720}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0.1, 0.1, 0.3, 0.4], "keypoints": _keypoints()}
                ],
            }
        )
    )
    samples = scan_split(split)
    assert len(samples) == 1
    assert len(samples[0].annotations) == 1
    stats = dataset_statistics(samples)
    assert stats["num_images"] == 1
    assert stats["num_people"] == 1
    assert stats["visibility_histogram"].shape == (17, 3)


def test_missing_image_skipped(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    (split / "ann.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "does_not_exist.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _keypoints()}],
            }
        )
    )
    assert scan_split(split) == []


def test_malformed_annotation_skipped(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    _write_image(split / "img0.jpg")
    (split / "img0.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img0.jpg", "width": 1280, "height": 720}],
                "annotations": [
                    {"id": 1, "image_id": 1, "bbox": [0.1, 0.1, 0.3, 0.4], "keypoints": [0.5, 0.5]},  # too short
                    {"id": 2, "image_id": 1, "bbox": [0.1, 0.1, 0.3, 0.4], "keypoints": _keypoints()},
                ],
            }
        )
    )
    samples = scan_split(split)
    assert len(samples) == 1
    assert len(samples[0].annotations) == 1  # malformed one dropped


def test_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_split(tmp_path / "nope")
