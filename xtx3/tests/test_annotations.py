"""Annotation scanning tests: 17->9 selection, absolute-COCO auto-detection,
nine-point passthrough, and malformed-record resilience."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from xtx3.data.annotations import dataset_statistics, scan_split
from xtx3.utils.keypoints import NUM_KEYPOINTS


def _write_image(path: Path, width: int = 64, height: int = 48) -> None:
    image = np.random.default_rng(0).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _kpts17_normalised() -> list[float]:
    """17 normalised keypoints with distinct x per index (x = idx/20)."""

    values: list[float] = []
    for idx in range(17):
        values.extend([idx / 20.0, 0.5, 2])
    return values


class TestSeventeenToNine:
    def test_selection_from_17(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img.jpg", "width": 64, "height": 48}],
                    "annotations": [
                        {"image_id": 1, "bbox": [0.1, 0.1, 0.5, 0.7], "keypoints": _kpts17_normalised()}
                    ],
                }
            ),
            encoding="utf-8",
        )
        samples = scan_split(tmp_path)
        assert len(samples) == 1
        ann = samples[0].annotations[0]
        assert ann.keypoints_norm.shape == (NUM_KEYPOINTS, 3)
        # COCO indices 0..6 then 11, 12 (x = idx/20 in the source array).
        expected_x = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12]) / 20.0
        np.testing.assert_allclose(ann.keypoints_norm[:, 0], expected_x, atol=1e-6)

    def test_nine_point_passthrough(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        kpts9 = []
        for idx in range(9):
            kpts9.extend([idx / 10.0, 0.25, 1])
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img.jpg", "width": 64, "height": 48}],
                    "annotations": [{"image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": kpts9}],
                }
            ),
            encoding="utf-8",
        )
        samples = scan_split(tmp_path)
        ann = samples[0].annotations[0]
        np.testing.assert_allclose(ann.keypoints_norm[:, 0], np.arange(9) / 10.0, atol=1e-6)
        assert (ann.keypoints_norm[:, 2] == 1).all()


class TestAbsoluteCoco:
    def test_absolute_pixels_normalised(self, tmp_path: Path) -> None:
        """Official-COCO style: absolute pixel bbox/keypoints get normalised."""

        _write_image(tmp_path / "img.jpg", width=200, height=100)
        kpts = []
        for idx in range(17):
            kpts.extend([idx * 10.0, 50.0, 2])  # pixels
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 7, "file_name": "img.jpg", "width": 200, "height": 100}],
                    "annotations": [
                        {
                            "image_id": 7,
                            "category_id": 1,
                            "bbox": [20.0, 10.0, 100.0, 80.0],
                            "keypoints": kpts,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        samples = scan_split(tmp_path)
        ann = samples[0].annotations[0]
        np.testing.assert_allclose(ann.bbox_xywh_norm, [0.1, 0.1, 0.5, 0.8], atol=1e-6)
        np.testing.assert_allclose(ann.keypoints_norm[:, 1], 0.5, atol=1e-6)
        # x of left hip (COCO idx 11): 110 px / 200 = 0.55
        assert ann.keypoints_norm[7, 0] == pytest.approx(0.55, abs=1e-6)

    def test_non_person_category_filtered(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img.jpg", "width": 64, "height": 48}],
                    "annotations": [
                        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1],
                         "keypoints": _kpts17_normalised()}
                    ],
                }
            ),
            encoding="utf-8",
        )
        samples = scan_split(tmp_path)
        assert len(samples[0].annotations) == 0


class TestRobustness:
    def test_malformed_records_skipped(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "img.jpg", "width": 64, "height": 48},
                        {"id": 2, "file_name": "missing.jpg"},
                        {"file_name": "no_id.jpg"},
                    ],
                    "annotations": [
                        {"image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _kpts17_normalised()},
                        {"image_id": 1, "bbox": [0, 0, 1], "keypoints": _kpts17_normalised()},
                        {"image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": [1, 2, 3]},
                        {"image_id": 99, "bbox": [0, 0, 1, 1], "keypoints": _kpts17_normalised()},
                        "not a dict",
                    ],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "broken.json").write_text("{ not valid json", encoding="utf-8")
        samples = scan_split(tmp_path)
        assert len(samples) == 1
        assert len(samples[0].annotations) == 1

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            scan_split(tmp_path / "nope")

    def test_statistics(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "img.jpg")
        (tmp_path / "annotations.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img.jpg", "width": 64, "height": 48}],
                    "annotations": [
                        {"image_id": 1, "bbox": [0, 0, 1, 1], "keypoints": _kpts17_normalised()}
                    ],
                }
            ),
            encoding="utf-8",
        )
        stats = dataset_statistics(scan_split(tmp_path))
        assert stats["num_images"] == 1
        assert stats["num_people"] == 1
        assert stats["visibility_histogram"].shape == (NUM_KEYPOINTS, 3)
        assert stats["visibility_histogram"][:, 2].sum() == NUM_KEYPOINTS
