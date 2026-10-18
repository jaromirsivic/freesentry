"""Model tests: shapes for every preset / level set / size, fuse parity, init,
P2-assist zero-init equivalence, deploy-graph op allowlist (CONTRACTS section 13)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from xtx4.models import PRESETS, XTX4Model, build_model, get_variant, resolve_spec
from xtx4.models.blocks import ConvBNAct, RepConv3x3, SpaceToDepth, make_act
from xtx4.models.head import PERSON_BIAS_PRIOR, POSE_BIAS_PER_POINT
from xtx4.models.head_types import KPT_CHANNELS_PER_POINT, build_anchors
from xtx4.models.scaling import LEVEL_SETS, VariantSpec

SIZES = (256, 320, 384)
LEVELS = tuple(LEVEL_SETS)
# Expected anchor counts at 320 per CONTRACTS section 3 / plan section 5.2.
EXPECTED_N_320 = {"P3P4P5": 2100, "P3P5": 1700, "P4P5": 500}

torch.set_num_threads(min(8, torch.get_num_threads()))


def _model(variant: str, **overrides) -> XTX4Model:
    torch.manual_seed(0)
    return build_model({"model": {"variant": variant, **overrides}})


def _expected_n(size: int, levels: str) -> int:
    return sum((size // s) ** 2 for s in LEVEL_SETS[levels])


# ----------------------------------------------------------------------------- scaling


def test_presets_match_contract_table() -> None:
    assert set(PRESETS) == {"u_a", "u_b", "u_c", "n", "m", "l", "xtx3u_ctrl"}
    ub = get_variant("u_b")
    assert ub.widths == (16, 32, 64, 128, 192) and ub.depths == (1, 2, 2, 1)
    assert ub.head_hidden == 64 and ub.pose_hidden == 64 and ub.block == "rep" and ub.act == "relu"
    assert get_variant("u") is ub  # alias
    assert get_variant("xtx3u_ctrl").block == "dw" and get_variant("xtx3u_ctrl").act == "silu"
    assert get_variant("l").default_size == 512 and get_variant("m").act == "silu"
    for spec in PRESETS.values():
        assert all(w % 8 == 0 for w in spec.widths)


def test_resolve_spec_overrides_and_validation() -> None:
    spec = resolve_spec({"variant": "u", "levels": "p3p5", "act": "HSWISH", "p2_assist": 1,
                         "widths": [16, 32, 64, 128, 192], "head_hidden": 32})
    assert spec.name == "u_b" and spec.levels == "P3P5" and spec.act == "hswish"
    assert spec.p2_assist is True and spec.head_hidden == 32 and spec.strides == [8, 32]
    assert VariantSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        resolve_spec({"variant": "nope"})
    with pytest.raises(ValueError):
        resolve_spec({"variant": "u_b", "widths": [16, 30, 64, 128, 192]})
    with pytest.raises(ValueError):
        resolve_spec({"variant": "u_b", "levels": "P2P3"})
    with pytest.raises(ValueError):
        resolve_spec({"variant": "u_b", "levels": "P4P5", "p2_assist": True})
    with pytest.raises(ValueError):
        resolve_spec({"variant": "u_b", "act": "gelu"})


# ----------------------------------------------------------------------------- blocks


def test_activation_factory() -> None:
    assert isinstance(make_act("relu"), nn.ReLU)
    assert isinstance(make_act("hswish"), nn.Hardswish)
    assert isinstance(make_act("silu"), nn.SiLU)
    assert isinstance(make_act(True), nn.SiLU)
    assert isinstance(make_act(False), nn.Identity)
    with pytest.raises(ValueError):
        make_act("gelu")


@pytest.mark.parametrize("stride,act", [(1, "relu"), (2, "hswish"), (2, "silu"), (1, "silu")])
def test_repconv3x3_fuse_parity(stride: int, act: str) -> None:
    torch.manual_seed(1)
    block = RepConv3x3(in_channels=16, out_channels=16 if stride == 1 else 24, stride=stride, act=act)
    # Randomise BN statistics so the fold is non-trivial.
    for m in block.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    block.eval()
    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        ref = block(x)
        block.fuse()
        out = block(x)
        block.fuse()
        out2 = block(x)
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 1e-5
    assert torch.equal(out, out2)
    assert isinstance(block.fused_conv, nn.Conv2d) and not hasattr(block, "conv3")


def test_space_to_depth_shape_and_content() -> None:
    x = torch.arange(1 * 3 * 8 * 6, dtype=torch.float32).reshape(1, 3, 8, 6)
    y = SpaceToDepth()(x)
    assert y.shape == (1, 12, 4, 3)
    # Every input value appears exactly once.
    assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)


# ----------------------------------------------------------------------------- shapes


@pytest.mark.parametrize("variant", sorted(PRESETS))
@pytest.mark.parametrize("levels", LEVELS)
def test_forward_shapes_all_presets_levels_sizes(variant: str, levels: str) -> None:
    model = _model(variant, levels=levels)
    assert model.strides == list(LEVEL_SETS[levels])
    assert model.spec.levels == levels and model.num_keypoints == 9
    k5 = 9 * KPT_CHANNELS_PER_POINT
    for size in SIZES:
        model.train()
        with torch.no_grad():
            out = model(torch.rand(1, 3, size, size))
        n = _expected_n(size, levels)
        if size == 320:
            assert n == EXPECTED_N_320[levels]
        for raw in (out.one2one, out.one2many):
            assert raw is not None
            assert raw.person.shape == (1, n, 1)
            assert raw.kpt.shape == (1, n, 9, KPT_CHANNELS_PER_POINT)
            assert raw.anchors.shape == (n, 2) and raw.strides.shape == (n,)
            assert torch.isfinite(raw.person).all() and torch.isfinite(raw.kpt).all()
        # Anchor layout = head_types.build_anchors in ascending-stride order.
        anchors, strides = build_anchors(
            feature_sizes=[(size // s, size // s) for s in model.strides],
            strides=model.strides, device=torch.device("cpu"), dtype=torch.float32,
        )
        assert torch.equal(out.one2one.anchors, anchors) and torch.equal(out.one2one.strides, strides)
        assert out.one2one.kpt.reshape(1, n, k5).shape == (1, n, k5)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 256, 256))
    assert out.one2many is None, "one2many must only be produced in training mode"


def test_batch_dimension_and_gradients_reach_both_heads() -> None:
    model = _model("u_a").train()
    out = model(torch.rand(2, 3, 256, 256))
    assert out.one2one.person.shape[0] == 2 and out.one2many is not None
    loss = out.one2one.person.mean() + out.one2one.kpt.mean() + out.one2many.person.mean() + out.one2many.kpt.mean()
    loss.backward()
    for head in (model.head.one2one, model.head.one2many):
        grads = [p.grad for p in head.parameters()]
        assert all(g is not None for g in grads)
    assert all(p.grad is not None for p in model.backbone.parameters())


# ----------------------------------------------------------------------------- init


def test_head_initialisation() -> None:
    model = _model("u_b", p2_assist=True)
    for head in (model.head.one2one, model.head.one2many):
        for level in head.levels:
            assert torch.allclose(level.person.bias, torch.full_like(level.person.bias, PERSON_BIAS_PRIOR))
            expected = torch.tensor(POSE_BIAS_PER_POINT).repeat(9)
            assert torch.allclose(level.pose_out.bias, expected)
            assert level.pose_out.weight.std().item() < 0.02
    assert model.p2_assist is not None
    assert model.p2_assist.proj.weight.abs().sum().item() == 0.0
    assert model.p2_assist.proj.bias.abs().sum().item() == 0.0
    # The person prior keeps early scores low on a random input.
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 256, 256))
    assert torch.sigmoid(out.one2one.person).mean().item() < 0.1
    sigma_units = torch.nn.functional.softplus(out.one2one.kpt[..., 2]) + 0.05
    assert 0.8 < sigma_units.mean().item() < 1.3


# ----------------------------------------------------------------------------- fuse


def _randomise_bn(model: nn.Module) -> None:
    gen = torch.Generator().manual_seed(123)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.rand(m.num_features, generator=gen) - 0.5)
            m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
            m.bias.data.copy_(torch.rand(m.num_features, generator=gen) - 0.5)


@pytest.mark.parametrize(
    "variant,overrides",
    [
        ("u_b", {}),
        ("u_b", {"p2_assist": True}),
        ("u_a", {"levels": "P3P5", "act": "hswish"}),
        ("u_a", {"levels": "P4P5", "act": "silu"}),
        ("xtx3u_ctrl", {}),
    ],
)
def test_fuse_parity_idempotent_and_param_counts(variant: str, overrides: dict) -> None:
    model = _model(variant, **overrides)
    _randomise_bn(model)
    model.eval()
    train_params = model.num_parameters()
    deploy_params = model.num_parameters(deploy=True)
    assert deploy_params < train_params
    assert model.num_parameters() == train_params  # profiling must not touch the model
    assert not model.fused

    x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        ref = model(x).one2one
        model.fuse()
        assert model.fused and model.head.one2many is None
        out = model(x).one2one
        model.fuse()  # idempotent
        out2 = model(x).one2one
    assert (out.person - ref.person).abs().max().item() < 1e-4
    assert (out.kpt - ref.kpt).abs().max().item() < 1e-4
    assert torch.equal(out.person, out2.person) and torch.equal(out.kpt, out2.kpt)
    assert model.num_parameters() == deploy_params
    assert model.num_parameters(deploy=True) == deploy_params
    # Training-mode forward of a fused model never yields one2many.
    model.train()
    with torch.no_grad():
        assert model(x).one2many is None


def test_fused_graph_op_allowlist() -> None:
    from xtx4.models.organs import OrganLayer

    # OrganLayer: the parameter-free brain/chest layer (constants only, see test_export allowlists)
    allowed_leaves = (nn.Conv2d, nn.ReLU, nn.Hardswish, nn.SiLU, nn.Identity, nn.MaxPool2d, nn.Upsample, SpaceToDepth,
                      OrganLayer)
    for variant, overrides in (("u_b", {"p2_assist": True}), ("xtx3u_ctrl", {}), ("u_a", {"act": "hswish"})):
        model = _model(variant, **overrides).fuse()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                assert isinstance(module, allowed_leaves), f"{name}: {type(module).__name__} not allowed after fuse"
        assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        for m in model.modules():
            if isinstance(m, RepConv3x3):
                assert m.fused_conv is not None and not hasattr(m, "conv1")
            if isinstance(m, ConvBNAct):
                assert isinstance(m.bn, nn.Identity)


# ----------------------------------------------------------------------------- p2 assist


def _copy_shared_weights(dst: XTX4Model, src: XTX4Model) -> None:
    """Copy src (with assist) into dst (without); slice the extra pose-conv input channels."""

    src_state = src.state_dict()
    new_state = {}
    for key, value in dst.state_dict().items():
        if key not in src_state:
            # The baseline's P3 pose RepConv (64->64) has a BN identity branch that the
            # 80->64 assist version structurally cannot have: switch it off (gamma = beta = 0).
            assert ".pose_conv.identity." in key, key
            new_state[key] = torch.zeros_like(value) if key.endswith(("weight", "bias")) else value
            continue
        s = src_state[key]
        if s.shape == value.shape:
            new_state[key] = s
        else:
            assert s.dim() == 4 and s.shape[0] == value.shape[0] and s.shape[1] > value.shape[1], key
            new_state[key] = s[:, : value.shape[1]]
    dst.load_state_dict(new_state)


def test_p2_assist_zero_init_matches_baseline() -> None:
    with_assist = _model("u_b", p2_assist=True)
    without = _model("u_b", p2_assist=False)
    _randomise_bn(with_assist)
    _copy_shared_weights(without, with_assist)
    with_assist.eval()
    without.eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = with_assist(x).one2one
        b = without(x).one2one
        assert torch.allclose(a.person, b.person, atol=1e-6)
        assert torch.allclose(a.kpt, b.kpt, atol=1e-6)
        # Non-zero assist weights change the pose branch but never the person branch.
        with_assist.p2_assist.proj.weight.normal_(0, 0.5)
        c = with_assist(x).one2one
        assert torch.allclose(c.person, a.person)
        assert not torch.allclose(c.kpt, a.kpt)
    # The assist is only wired into the P3 (level 0) pose conv.
    levels = with_assist.head.one2one.levels
    assert levels[0].pose_conv.in_channels == with_assist.spec.head_hidden + with_assist.p2_assist.out_channels
    assert all(lv.pose_conv.in_channels == with_assist.spec.head_hidden for lv in levels[1:])


# ----------------------------------------------------------------------------- predict / budget


def test_predict_returns_decoded_lists() -> None:
    model = _model("u_a").eval()
    dets, vis, sigma = model.predict(torch.rand(2, 3, 256, 256), conf_threshold=0.0, max_detections=5)
    assert len(dets) == len(vis) == len(sigma) == 2
    for d, v, s in zip(dets, vis, sigma):
        assert d.shape == (5, 1 + 3 * 9) and v.shape == (5, 9) and s.shape == (5, 9)
        assert v.dtype == torch.int64
        assert torch.all(d[:-1, 0] >= d[1:, 0])  # sorted by score
    dets, _, _ = model.predict(torch.rand(1, 3, 256, 256), conf_threshold=1.01)
    assert dets[0].shape == (0, 28)


def test_u_b_budget_near_plan() -> None:
    model = _model("u_b")
    gmac = model.mac_count(320)
    assert 0.65 <= gmac <= 0.9, f"u_b @320 = {gmac:.3f} GMAC, plan estimates 0.765"
    assert model.mac_count(256) < gmac < model.mac_count(384)
    assert 1.5e6 <= model.num_parameters(deploy=True) <= 2.3e6
    assert 0.3 <= _model("u_a").mac_count(320) <= 0.5


# ----------------------------------------------------------------------------- organ layer


def test_organ_layer_matches_numpy_reference() -> None:
    import numpy as np

    from xtx4.models.organs import OrganLayer
    from xtx4.utils.organs import compute_organs

    torch.manual_seed(0)
    pose = torch.randn(2, 300, 9 * KPT_CHANNELS_PER_POINT) * 3.0
    out = OrganLayer()(pose)
    assert out.shape == (2, 300, 8)
    p = pose.reshape(2, 300, 9, KPT_CHANNELS_PER_POINT).double().numpy()
    ref = compute_organs(p[..., 0:2], 1.0 / (1.0 + np.exp(-p[..., 3])))
    np.testing.assert_allclose(out.reshape(2, 300, 2, 4).double().numpy(), ref, atol=2e-5, rtol=1e-5)


def test_organ_layer_keeps_checkpoint_and_budget() -> None:
    model = _model("u_a")
    assert model.organs is not None
    assert not any(k.startswith("organs.") for k in model.state_dict())
    assert sum(p.numel() for p in model.organs.parameters()) == 0
    fresh = _model("u_a")
    fresh.load_state_dict(model.state_dict(), strict=True)


def test_predict_with_organs() -> None:
    model = _model("u_a").eval()
    x = torch.rand(2, 3, 256, 256)
    dets, vis, sigma, organs = model.predict(x, conf_threshold=0.0, max_detections=5, with_organs=True)
    plain = model.predict(x, conf_threshold=0.0, max_detections=5)
    assert len(organs) == 2
    for d, d0, o in zip(dets, plain[0], organs):
        assert torch.equal(d, d0)
        assert o.shape == (5, 2, 4)
        assert torch.all((o[..., 3] >= 0) & (o[..., 3] <= 1)) and torch.all(o[..., 2] >= 0)
    # the brain/chest centres agree with the decoded keypoints (affine combination of the same points)
    from xtx4.utils.organs import compute_organs

    kp = dets[0][:, 1:].reshape(-1, 9, 3).double().numpy()
    ref = compute_organs(kp[..., :2], kp[..., 2])
    import numpy as np

    np.testing.assert_allclose(organs[0].double().numpy(), ref, atol=1e-3, rtol=1e-4)
