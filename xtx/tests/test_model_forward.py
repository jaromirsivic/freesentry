"""Model forward shapes, fuse(), and export-readiness (spec acceptance 2, 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from xtx.models import build_model
from xtx.models.head import KPT_CHANNELS_PER_POINT


@pytest.mark.parametrize("variant", ["n", "m", "l"])
def test_forward_shapes_and_dual_head(variant: str) -> None:
    model = build_model(variant=variant).train()
    x = torch.randn(2, 3, 384, 384)
    output = model(x)
    o2o = output.one2one
    num_anchors = o2o.box.shape[1]
    assert o2o.box.shape == (2, num_anchors, 4)
    assert o2o.cls.shape == (2, num_anchors, 1)
    assert o2o.kpt.shape == (2, num_anchors, 17, KPT_CHANNELS_PER_POINT)
    assert o2o.anchors.shape == (num_anchors, 2)
    assert output.one2many is not None  # dual head active during training


@pytest.mark.parametrize("variant", ["n", "m", "l"])
def test_decode_max_detections(variant: str) -> None:
    model = build_model(variant=variant)
    dets, vis = model.predict(torch.randn(1, 3, 384, 384), conf_threshold=0.0, max_detections=300)
    assert dets[0].shape == (300, 56)
    assert vis[0].shape == (300, 17)


def test_fuse_is_nms_free_and_drops_one2many() -> None:
    model = build_model(variant="n").eval()
    model.fuse()
    assert model.head.one2many is None  # auxiliary head removed -> NMS-free deploy
    # forward still works after fuse and is idempotent
    out = model(torch.randn(1, 3, 384, 384))
    assert out.one2many is None
    model.fuse()  # idempotent
    assert out.one2one.box.shape[-1] == 4


def test_export_model_returns_raw_tensors() -> None:
    from xtx.export.ncnn_export import ExportModel

    model = build_model(variant="n")
    export_model = ExportModel(model).eval()
    box, cls, kpt = export_model(torch.zeros(1, 3, 384, 384))
    assert box.shape[0] == 1 and box.shape[-1] == 4
    assert cls.shape[-1] == 1
    assert kpt.shape[-1] == 17 * KPT_CHANNELS_PER_POINT


def test_params_within_budget() -> None:
    budgets = {"n": (2.5, 3.0), "m": (18.0, 22.0), "l": (24.0, 28.0)}
    for variant, (lo, hi) in budgets.items():
        params_m = build_model(variant=variant).num_parameters() / 1e6
        assert lo <= params_m <= hi, f"XTX-{variant} params {params_m:.2f}M outside {lo}-{hi}M"
