"""NCNN runner vs ONNX Runtime on the exported xtx3-u models.

Checks the raw tensors (blob mapping, BGR order, /255 normalisation) and the
final decoded detections. Needs ``ncnn`` + ``onnxruntime``; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import EXPORT_DIR
from livetest.config import load_config
from livetest.decode import decode_raw
from livetest.preprocess import prepare_views
from livetest.runner import NcnnPoseRunner

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("ncnn", reason="ncnn not installed")


@pytest.fixture(scope="module")
def network():
    return load_config().network("xtx3-u")


def _frame() -> np.ndarray:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    # Add smooth structure so the network produces non-trivial activations.
    yy, xx = np.mgrid[0:720, 0:1280]
    frame[..., 0] = ((np.sin(xx / 40.0) + 1) * 120).astype(np.uint8)
    return frame


@pytest.mark.parametrize("size", [256, 320, 384])
def test_ncnn_matches_onnx(network, size: int) -> None:
    onnx_path = EXPORT_DIR / f"xtx3_u_{size}.onnx"
    if not onnx_path.is_file() or not network.size_available(size):
        pytest.skip(f"export files for size {size} missing")

    runner = NcnnPoseRunner(network, size, num_threads=2, use_vulkan=False)
    try:
        prepared = prepare_views(_frame(), mode="whole", network_size=size)
        view = prepared.views[0]
        box_n, cls_n, kpt_n = runner._forward(view)  # noqa: SLF001 - raw tensors under test

        blob = (view.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # BGR, NCHW
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        box_o, cls_o, kpt_o = session.run(["box", "cls", "kpt"], {"in0": blob})

        # Negative control: the same frame fed as RGB (a typical wiring bug).
        box_w, cls_w, kpt_w = session.run(["box", "cls", "kpt"], {"in0": np.ascontiguousarray(blob[:, ::-1])})

        # pnnx stores NCNN weights as fp16, so activations drift slightly (a few
        # up to ~0.3). A wiring error (wrong blob, RGB instead of BGR, missing
        # /255) shifts everything, so NCNN must sit far closer to the correct
        # ONNX output than the RGB control does.
        pairs = (("box", box_n, box_o, box_w), ("cls", cls_n, cls_o, cls_w), ("kpt", kpt_n, kpt_o, kpt_w))
        for name, a, good, wrong in pairs:
            diff = float(np.median(np.abs(a.reshape(good.shape) - good)))
            control = float(np.median(np.abs(wrong - good)))
            assert diff < 0.02, f"{name}: median |diff| {diff:.4f}"
            assert diff < control / 4, f"{name}: median |diff| {diff:.4f} vs RGB control {control:.4f}"
            assert float(np.abs(a.reshape(good.shape) - good).max()) < 0.5, f"{name}: max |diff| too large"

        kwargs = dict(network_size=size, strides=runner.strides, num_keypoints=9,
                      conf_threshold=0.0, max_detections=10_000)
        _, s_n, _, _ = decode_raw(box_n, cls_n, kpt_n, **kwargs)
        _, s_o, _, _ = decode_raw(box_o[0], cls_o[0], kpt_o[0], **kwargs)
        # Top-scoring anchors agree (sets, since near-ties may swap order).
        assert abs(float(s_n[0]) - float(s_o[0])) < 0.02
        np.testing.assert_allclose(np.sort(s_n)[-20:], np.sort(s_o)[-20:], atol=0.02)
    finally:
        runner.close()
