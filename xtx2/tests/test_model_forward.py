"""Model forward tests: shapes for n/m/l, fuse(), export-readiness (criterion 4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xtx2.models import build_model
from xtx2.models.head import KPT_CHANNELS_PER_POINT

INPUT = torch.zeros(1, 3, 384, 384)


def _expected_anchors(use_p2: bool) -> int:
    total = 48 * 48 + 24 * 24 + 12 * 12
    if use_p2:
        total += 96 * 96
    return total


@pytest.mark.parametrize("variant,use_p2", [("n", False), ("m", False), ("l", True)])
class TestForward:
    def test_train_shapes(self, variant: str, use_p2: bool) -> None:
        model = build_model(variant=variant)
        assert model.use_p2 == use_p2
        model.train()
        output = model(INPUT)
        n = _expected_anchors(use_p2)
        assert output.one2one.box.shape == (1, n, 4)
        assert output.one2one.cls.shape == (1, n, 1)
        assert output.one2one.kpt.shape == (1, n, 17, KPT_CHANNELS_PER_POINT)
        assert output.one2one.anchors.shape == (n, 2)
        assert output.one2one.strides.shape == (n,)
        # Dual head: one-to-many populated during training.
        assert output.one2many is not None
        assert output.one2many.box.shape == (1, n, 4)

    def test_eval_drops_one2many_output(self, variant: str, use_p2: bool) -> None:
        model = build_model(variant=variant).eval()
        with torch.no_grad():
            output = model(INPUT)
        assert output.one2many is None


class TestFuse:
    def test_fuse_preserves_output(self) -> None:
        model = build_model(variant="n").eval()
        x = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            before = model(x).one2one
            model.fuse()
            after = model(x).one2one
        assert model.head.one2many is None  # NMS-free deployment head only
        assert torch.allclose(before.box, after.box, atol=1e-3)
        assert torch.allclose(before.cls, after.cls, atol=1e-3)
        assert torch.allclose(before.kpt, after.kpt, atol=1e-3)

    def test_fuse_idempotent(self) -> None:
        model = build_model(variant="n").eval()
        model.fuse()
        model.fuse()  # must not raise

    def test_predict_after_fuse(self) -> None:
        model = build_model(variant="n").eval()
        model.fuse()
        dets, vis = model.predict(INPUT, conf_threshold=0.0, max_detections=10)
        assert len(dets) == 1
        assert dets[0].shape[1] == 56  # section 6.5 layout
        assert dets[0].shape[0] <= 10
        assert vis[0].shape[1] == 17


class TestBatchedLevels:
    def test_three_level_batch(self) -> None:
        """The 3-level pyramid runs as one (3, 3, 384, 384) forward."""

        model = build_model(variant="n").eval()
        model.fuse()
        batch = torch.rand(3, 3, 384, 384)
        dets, vis = model.predict(batch, conf_threshold=0.0, max_detections=5)
        assert len(dets) == 3
        assert len(vis) == 3


class TestExportReadiness:
    def test_export_model_raw_outputs(self) -> None:
        from xtx2.export.ncnn_export import ExportModel

        model = build_model(variant="n").eval()
        export_model = ExportModel(model).eval()
        with torch.no_grad():
            box, cls, kpt = export_model(INPUT)
        n = _expected_anchors(False)
        assert box.shape == (1, n, 4)
        assert cls.shape == (1, n, 1)
        assert kpt.shape == (1, n, 17 * KPT_CHANNELS_PER_POINT)

    def test_onnx_export(self, tmp_path) -> None:
        onnx = pytest.importorskip("onnx")
        from xtx2.export.ncnn_export import export_onnx

        model = build_model(variant="n").eval()
        path = export_onnx(model, tmp_path / "xtx2_n.onnx", simplify=False)
        assert path.exists()
        onnx.checker.check_model(onnx.load(str(path)))

    def test_decode_raw_outputs(self) -> None:
        from xtx2.export.ncnn_export import decode_raw_outputs

        n = _expected_anchors(False)
        rng = np.random.default_rng(0)
        det, vis = decode_raw_outputs(
            rng.normal(size=(1, n, 4)).astype(np.float32),
            rng.normal(size=(1, n, 1)).astype(np.float32),
            rng.normal(size=(1, n, 17 * KPT_CHANNELS_PER_POINT)).astype(np.float32),
            num_keypoints=17,
            strides=[8, 16, 32],
            conf_threshold=0.0,
            max_detections=25,
        )
        assert det.shape == (25, 56)
        assert vis.shape == (25, 17)


class TestNoNMS:
    def test_no_nms_in_api(self) -> None:
        """No NMS code path exists in the inference module (criterion 4)."""

        import inspect

        import xtx2.api as api
        import xtx2.models.head as head

        for module in (api, head):
            source = inspect.getsource(module).lower()
            assert "torchvision.ops.nms" not in source
            assert "batched_nms" not in source
