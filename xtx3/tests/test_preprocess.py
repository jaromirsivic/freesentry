"""Preprocessing tests: whole-frame letterbox geometry, the pyramid views,
round-trip mapping within <= 0.5 px, and one-shot rendering agreeing with the
recorded affines and the full-canvas reference."""

from __future__ import annotations

import numpy as np
import pytest

from xtx3.data.preprocess import (
    PYRAMID_VIEW_KEYS,
    WHOLE_VIEW_KEY,
    build_canvas,
    build_views,
    canvas_size_for_mode,
    coerce_bgr,
    compute_placement,
    compute_transforms,
    preprocess_image,
    view_keys_for_mode,
)
from xtx3.utils import geometry


def _synthetic_image(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(width * 1000 + height)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


class TestWholeFrame:
    @pytest.mark.parametrize("size", [256, 320, 384])
    def test_letterbox_fits_and_centres(self, size: int) -> None:
        placement = compute_placement(width=1920, height=1080, canvas_size=size)
        assert placement.resized_w <= size
        assert placement.resized_h <= size
        assert max(placement.resized_w, placement.resized_h) == size
        assert abs(placement.paste_x - (size - placement.resized_w) // 2) < 4
        assert abs(placement.paste_y - (size - placement.resized_h) // 2) < 4

    def test_small_image_not_upscaled(self) -> None:
        placement = compute_placement(width=200, height=150, canvas_size=320)
        assert placement.resized_w == 200
        assert placement.resized_h == 150
        assert placement.scale_x == 1.0

    @pytest.mark.parametrize("size", [256, 320, 384])
    @pytest.mark.parametrize("dims", [(1920, 1080), (320, 240)])
    def test_round_trip(self, size: int, dims: tuple[int, int]) -> None:
        width, height = dims
        transforms = compute_transforms(width=width, height=height, mode="whole", network_size=size)
        rng = np.random.default_rng(0)
        points = rng.uniform([0, 0], [width, height], size=(64, 2))
        forward = geometry.invert_affine(transforms[WHOLE_VIEW_KEY])
        net = geometry.apply_to_points(forward, points)
        back = geometry.apply_to_points(transforms[WHOLE_VIEW_KEY], net)
        assert np.abs(back - points).max() <= 0.5
        # All points land inside the network frame (whole image is visible).
        assert net.min() >= -0.5
        assert net.max() <= size + 0.5

    def test_result_shape(self) -> None:
        result = preprocess_image(_synthetic_image(1920, 1080), mode="whole", network_size=320)
        assert set(result.views.keys()) == {WHOLE_VIEW_KEY}
        assert result.views[WHOLE_VIEW_KEY].shape == (320, 320, 3)
        assert result.views[WHOLE_VIEW_KEY].dtype == np.uint8

    def test_rendering_agrees_with_affine(self) -> None:
        """A bright marker placed in the original must land where T^-1 says."""

        for size in (256, 320, 384):
            for width, height in [(1920, 1080), (300, 200)]:
                image = np.zeros((height, width, 3), dtype=np.uint8)
                mx, my = width // 2 + 15, height // 2 - 10
                image[max(0, my - 4) : my + 4, max(0, mx - 4) : mx + 4] = 255

                result = preprocess_image(image, mode="whole", network_size=size)
                inv = geometry.invert_affine(result.transforms[WHOLE_VIEW_KEY])
                expected = geometry.apply_to_points(inv, np.array([[mx, my]], float))[0]
                gray = result.views[WHOLE_VIEW_KEY].sum(axis=2).astype(np.float64)
                assert gray.max() > 0
                ys, xs = np.nonzero(gray > gray.max() * 0.5)
                cx = float((xs * gray[ys, xs]).sum() / gray[ys, xs].sum())
                cy = float((ys * gray[ys, xs]).sum() / gray[ys, xs].sum())
                err = np.hypot(cx - expected[0], cy - expected[1])
                assert err <= 1.5, f"whole@{size} {width}x{height}: marker error {err:.2f}px"


class TestPyramid:
    def test_matches_xtx2_geometry_at_384(self) -> None:
        """At S=384 the pyramid must reproduce the XTX2 worked example exactly."""

        transforms = compute_transforms(width=1920, height=1080, mode="pyramid", network_size=384)
        pt = geometry.apply_to_points(transforms[1], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx(4 * 100 / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((4 * 100 - 336) / 0.8, abs=1e-6)
        pt = geometry.apply_to_points(transforms[2], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx((2 * 100 + 384) / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((2 * 100 + 384 - 336) / 0.8, abs=1e-6)
        pt = geometry.apply_to_points(transforms[3], np.array([[100.0, 100.0]]))[0]
        assert pt[0] == pytest.approx((100 + 576) / 0.8, abs=1e-6)
        assert pt[1] == pytest.approx((100 + 576 - 336) / 0.8, abs=1e-6)

    @pytest.mark.parametrize("dims", [(1920, 1080), (320, 240)])
    def test_round_trip(self, dims: tuple[int, int]) -> None:
        width, height = dims
        transforms = compute_transforms(width=width, height=height, mode="pyramid", network_size=384)
        rng = np.random.default_rng(0)
        points = rng.uniform([0, 0], [width, height], size=(64, 2))
        for view in PYRAMID_VIEW_KEYS:
            forward = geometry.invert_affine(transforms[view])
            net = geometry.apply_to_points(forward, points)
            back = geometry.apply_to_points(transforms[view], net)
            assert np.abs(back - points).max() <= 0.5

    def test_small_branch_matches_canvas_reference(self) -> None:
        """With aligned offsets and s = 1, one-shot views match the full-canvas
        reference bit-for-bit (view 3) and near-exactly (views 1, 2)."""

        import cv2

        image = _synthetic_image(320, 240)
        placement = compute_placement(width=320, height=240, canvas_size=1536)
        views = build_views(image, placement, mode="pyramid", network_size=384)
        canvas = build_canvas(image, placement)

        ref1 = cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_AREA)
        ref2 = cv2.resize(canvas[384:1152, 384:1152], (384, 384), interpolation=cv2.INTER_AREA)
        ref3 = canvas[576:960, 576:960]
        assert np.array_equal(views[3], ref3)
        assert np.abs(views[1].astype(int) - ref1.astype(int)).max() <= 1
        assert np.abs(views[2].astype(int) - ref2.astype(int)).max() <= 1

    def test_shapes(self) -> None:
        result = preprocess_image(_synthetic_image(1920, 1080), mode="pyramid", network_size=384)
        assert set(result.views.keys()) == {1, 2, 3}
        for view in PYRAMID_VIEW_KEYS:
            assert result.views[view].shape == (384, 384, 3)


class TestModeHelpers:
    def test_view_keys(self) -> None:
        assert view_keys_for_mode("whole") == (0,)
        assert view_keys_for_mode("pyramid") == (1, 2, 3)
        with pytest.raises(ValueError):
            view_keys_for_mode("bogus")

    def test_canvas_sizes(self) -> None:
        assert canvas_size_for_mode("whole", 320) == 320
        assert canvas_size_for_mode("pyramid", 384) == 1536


class TestCoerce:
    def test_grayscale_promoted(self) -> None:
        gray = np.zeros((10, 12), dtype=np.uint8)
        assert coerce_bgr(gray).shape == (10, 12, 3)

    def test_bgra_alpha_dropped(self) -> None:
        bgra = np.zeros((10, 12, 4), dtype=np.uint8)
        assert coerce_bgr(bgra).shape == (10, 12, 3)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            coerce_bgr(np.zeros((10, 12, 2), dtype=np.uint8))
