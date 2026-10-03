"""Bit-exact parity of the letterbox geometry with ``xtx3.data.preprocess``.

Needs the training package (run from the xtx3 venv); skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from livetest.preprocess import prepare_views

ref = pytest.importorskip("xtx3.data.preprocess", reason="xtx3 training package not importable")

FRAME_SIZES = [(640, 480), (1280, 720), (1920, 1080), (300, 200), (480, 640), (2000, 333), (37, 1500)]


@pytest.mark.parametrize("width,height", FRAME_SIZES)
@pytest.mark.parametrize("mode,size", [("whole", 256), ("whole", 320), ("whole", 384), ("pyramid", 384)])
def test_views_and_transforms_match_training(width: int, height: int, mode: str, size: int) -> None:
    rng = np.random.default_rng(width * 7 + height)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    expected = ref.preprocess_image(frame, mode=mode, network_size=size)
    actual = prepare_views(frame, mode=mode, network_size=size)

    assert actual.orig_size == expected.orig_size
    assert set(actual.views) == set(expected.views)
    for key in expected.views:
        np.testing.assert_array_equal(actual.views[key], expected.views[key], err_msg=f"view {key}")
        np.testing.assert_allclose(actual.transforms[key], expected.transforms[key], rtol=0, atol=1e-9)
