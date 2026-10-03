"""Parity of the numpy decode with the torch reference ``decode_raw_outputs``.

Needs torch + the xtx3 training package (run from the xtx3 venv); skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from livetest.decode import anchor_count, decode_raw

pytest.importorskip("torch", reason="torch not installed")
ref = pytest.importorskip("xtx3.export.ncnn_export", reason="xtx3 training package not importable")


@pytest.mark.parametrize("size,strides", [(256, (8, 16, 32)), (320, (8, 16, 32)), (384, (4, 8, 16, 32))])
def test_decode_matches_torch_reference(size: int, strides: tuple[int, ...]) -> None:
    k = 9
    n = anchor_count(size, strides)
    rng = np.random.default_rng(size)
    box = rng.normal(0, 2, (1, n, 4)).astype(np.float32)
    cls = rng.normal(-2, 2, (1, n, 1)).astype(np.float32)
    kpt = rng.normal(0, 2, (1, n, k * 6)).astype(np.float32)

    det_ref, vis_ref = ref.decode_raw_outputs(
        box, cls, kpt, strides=list(strides), num_keypoints=k, input_size=size,
        conf_threshold=0.3, max_detections=50,
    )
    boxes, scores, kpts, vis = decode_raw(
        box[0], cls[0], kpt[0], network_size=size, strides=strides, num_keypoints=k,
        conf_threshold=0.3, max_detections=50,
    )
    det = np.concatenate([boxes, scores[:, None], kpts.reshape(len(scores), -1)], axis=1)

    # Reference keeps anchor order; ours is score-sorted -> compare sorted by score.
    order_ref = np.argsort(-det_ref[:, 4], kind="stable")
    assert det.shape == det_ref.shape
    np.testing.assert_allclose(det, det_ref[order_ref], rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(vis, vis_ref[order_ref])
