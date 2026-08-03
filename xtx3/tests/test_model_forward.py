"""Model tests: forward shapes for all variants, multi-size inputs for u,
fuse parity, NMS-free decode, and the 32-value detection layout."""

from __future__ import annotations

import pytest
import torch

from xtx3.models import build_model
from xtx3.models.head import decode_detections_full
from xtx3.models.scaling import VARIANTS
from xtx3.utils.keypoints import NUM_KEYPOINTS


def _anchor_count(size: int, strides: list[int]) -> int:
    return sum((size // s) ** 2 for s in strides)


class TestForward:
    @pytest.mark.parametrize("variant", ["u", "n", "m", "l"])
    def test_output_shapes(self, variant: str) -> None:
        model = build_model(variant=variant).eval()
        size = 384
        n = _anchor_count(size, model.strides)
        with torch.no_grad():
            out = model(torch.zeros(1, 3, size, size))
        assert out.one2one.box.shape == (1, n, 4)
        assert out.one2one.cls.shape == (1, n, 1)
        assert out.one2one.kpt.shape == (1, n, NUM_KEYPOINTS, 6)
        assert out.one2one.anchors.shape == (n, 2)
        assert out.one2many is None  # eval mode: one-to-one only

    def test_dual_head_in_training(self) -> None:
        model = build_model(variant="u").train()
        out = model(torch.zeros(1, 3, 320, 320))
        assert out.one2many is not None
        assert out.one2many.box.shape == out.one2one.box.shape

    @pytest.mark.parametrize("size", [256, 320, 384])
    def test_u_multi_size(self, size: int) -> None:
        """The u variant must accept all three deployment input sizes."""

        model = build_model(variant="u").eval()
        n = _anchor_count(size, model.strides)
        with torch.no_grad():
            out = model(torch.zeros(1, 3, size, size))
        assert out.one2one.box.shape == (1, n, 4)
        # Anchors span the actual input size.
        assert float(out.one2one.anchors.max()) <= size

    def test_l_uses_p2(self) -> None:
        model = build_model(variant="l")
        assert model.strides == [4, 8, 16, 32]

    def test_u_n_use_three_levels(self) -> None:
        for variant in ("u", "n"):
            assert build_model(variant=variant).strides == [8, 16, 32]

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model(variant="x")


class TestScaling:
    def test_u_is_smallest(self) -> None:
        params = {v: build_model(variant=v).num_parameters() for v in VARIANTS}
        assert params["u"] < params["n"] < params["m"] < params["l"]

    def test_u_budget(self) -> None:
        model = build_model(variant="u")
        assert model.num_parameters() <= 1.6e6


class TestFuse:
    @pytest.mark.parametrize("variant", ["u", "n"])
    def test_fuse_preserves_outputs(self, variant: str) -> None:
        torch.manual_seed(0)
        model = build_model(variant=variant).eval()
        x = torch.rand(1, 3, 320, 320)
        with torch.no_grad():
            before = model(x).one2one
            model.fuse()
            after = model(x).one2one
        assert model.head.one2many is None
        assert torch.allclose(before.box, after.box, atol=1e-4)
        assert torch.allclose(before.cls, after.cls, atol=1e-4)
        assert torch.allclose(before.kpt, after.kpt, atol=1e-4)


class TestDecode:
    def test_detection_layout_is_32_values(self) -> None:
        model = build_model(variant="u").eval()
        with torch.no_grad():
            out = model(torch.rand(2, 3, 256, 256))
        dets, vis = decode_detections_full(
            out.one2one, num_keypoints=NUM_KEYPOINTS, conf_threshold=0.0, max_detections=50
        )
        assert len(dets) == 2
        for det, v in zip(dets, vis):
            assert det.shape[1] == 5 + 3 * NUM_KEYPOINTS  # 32
            assert det.shape[0] <= 50
            assert v.shape == (det.shape[0], NUM_KEYPOINTS)
            assert v.dtype == torch.int64

    def test_high_threshold_yields_empty(self) -> None:
        model = build_model(variant="u").eval()
        with torch.no_grad():
            dets, vis = model.predict(torch.rand(1, 3, 256, 256), conf_threshold=0.9999)
        assert dets[0].shape == (0, 32)
        assert vis[0].shape == (0, NUM_KEYPOINTS)
