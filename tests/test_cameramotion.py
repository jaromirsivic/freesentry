import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from server import cameramotion


def _make_feature_rich_frame(*, width: int = 320, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(12345)
    image = rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8)
    for row, y in enumerate(range(20, height - 20, 24)):
        for col, x in enumerate(range(20, width - 20, 24)):
            intensity = int(60 + ((row * 23 + col * 31) % 170))
            color = (intensity, intensity, intensity)
            inverse = 255 - intensity
            cv2.rectangle(image, (x - 6, y - 6), (x + 6, y + 6), color, -1)
            cv2.rectangle(image, (x - 8, y - 8), (x + 8, y + 8), (inverse, inverse, inverse), 1)
            cv2.line(image, (x - 10, y), (x + 10, y), (255, 255, 255), 1)
            cv2.line(image, (x, y - 10), (x, y + 10), (255, 255, 255), 1)
    return image


def _translate_image(image: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _draw_moving_object(image: np.ndarray, rectangle: tuple[int, int, int, int]) -> None:
    x, y, width, height = rectangle
    step = 12
    for row_index, top in enumerate(range(y, y + height, step)):
        for col_index, left in enumerate(range(x, x + width, step)):
            value = 255 if (row_index + col_index) % 2 == 0 else 35
            bottom = min(top + step - 1, y + height - 1)
            right = min(left + step - 1, x + width - 1)
            cv2.rectangle(image, (left, top), (right, bottom), (value, value, value), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)
    cv2.line(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
    cv2.line(image, (x + width, y), (x, y + height), (0, 255, 0), 2)


class CameraMotionTests(unittest.TestCase):
    def test_get_camera_movement_recovers_known_translation(self):
        previous_frame = _make_feature_rich_frame()
        dx = 14
        dy = -9
        current_frame = _translate_image(previous_frame, dx=dx, dy=dy)

        result = cameramotion.get_camera_movement(
            frames=[previous_frame, current_frame],
            masked_rectangles=[[], []],
        )

        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.pixel.x, dx, delta=1.5)
        self.assertAlmostEqual(result.pixel.y, dy, delta=1.5)
        self.assertAlmostEqual(result.normalized.x, dx / previous_frame.shape[1], delta=0.01)
        self.assertAlmostEqual(result.normalized.y, dy / previous_frame.shape[0], delta=0.01)
        self.assertGreater(result.inlier_count, 0)
        self.assertGreater(result.confidence, 0.5)

    def test_get_camera_movement_ignores_masked_moving_object(self):
        background = _make_feature_rich_frame()
        dx = 11
        dy = 7
        previous_frame = background.copy()
        current_frame = _translate_image(background, dx=dx, dy=dy)
        previous_object = (120, 50, 96, 96)
        current_object = (30, 120, 96, 96)
        _draw_moving_object(previous_frame, previous_object)
        _draw_moving_object(current_frame, current_object)

        result = cameramotion.get_camera_movement(
            frames=[previous_frame, current_frame],
            masked_rectangles=[[previous_object], [current_object]],
            config=cameramotion.CameraMotionConfig(max_features=350),
        )

        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.pixel.x, dx, delta=2.0)
        self.assertAlmostEqual(result.pixel.y, dy, delta=2.0)
        self.assertGreaterEqual(result.tracked_count, result.inlier_count)

    def test_get_camera_movement_rejects_featureless_frames(self):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)

        result = cameramotion.get_camera_movement(
            frames=[blank, blank],
            masked_rectangles=[[], []],
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "insufficient_features")
        self.assertEqual(result.feature_count, 0)
        self.assertEqual(result.pixel.x, 0.0)
        self.assertEqual(result.pixel.y, 0.0)

    def test_raspberry_pi_estimator_reuses_previous_frame_state(self):
        estimator = cameramotion.RaspberryPiCameraMotionEstimator()
        previous_frame = _make_feature_rich_frame()
        current_frame = _translate_image(previous_frame, dx=8, dy=5)

        warmup = estimator.update(previous_frame, [])
        result = estimator.update(current_frame, [])

        self.assertFalse(warmup.valid)
        self.assertEqual(warmup.reason, "warmup")
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.pixel.x, 8, delta=2.0)
        self.assertAlmostEqual(result.pixel.y, 5, delta=2.0)
        self.assertEqual(result.downscale_factor, 0.5)

    def test_get_camera_movement_from_files_matches_in_memory_pipeline(self):
        previous_frame = _make_feature_rich_frame()
        dx = 10
        dy = -6
        current_frame = _translate_image(previous_frame, dx=dx, dy=dy)
        with tempfile.TemporaryDirectory() as tmp:
            prev_path = Path(tmp) / 'a.png'
            curr_path = Path(tmp) / 'b.png'
            cv2.imwrite(str(prev_path), previous_frame)
            cv2.imwrite(str(curr_path), current_frame)
            from_disk = cameramotion.get_camera_movement_from_files(prev_path, curr_path)
            from_arrays = cameramotion.get_camera_movement(
                frames=[previous_frame, current_frame],
                masked_rectangles=[[], []],
            )
        self.assertTrue(from_disk.valid, from_disk.reason)
        self.assertTrue(from_arrays.valid, from_arrays.reason)
        self.assertAlmostEqual(from_disk.pixel.x, from_arrays.pixel.x, delta=0.5)
        self.assertAlmostEqual(from_disk.pixel.y, from_arrays.pixel.y, delta=0.5)

    def test_get_camera_movement_from_files_missing_file_raises(self):
        with self.assertRaises(ValueError) as ctx:
            cameramotion.get_camera_movement_from_files('missing_prev.png', 'missing_curr.png')
        self.assertIn('Could not load image', str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
