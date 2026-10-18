"""Export tests: ONNX parity via onnxruntime, static shapes, op allowlist, and
(when pnnx is available) ncnn conversion with blob-name discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from xtx4.export.onnx_ncnn import (
    INPUT_NAME,
    OUTPUT_NAMES,
    ExportModel,
    convert_onnx_to_ncnn,
    export_onnx,
    find_pnnx,
    onnx_op_types,
    parse_ncnn_param,
    verify_onnx,
)
from xtx4.models import build_model

ONNX_ALLOWED_OPS = {
    "Conv", "Relu", "HardSwish", "Sigmoid", "Mul", "Add", "Concat", "Split", "Resize", "MaxPool",
    "Reshape", "Transpose", "Constant", "SpaceToDepth", "Slice",
}
# the parameter-free organ layer (extradetection.md, section 3.4)
ONNX_ORGAN_OPS = {"Gather", "MatMul", "Sqrt", "Max", "ReduceMin", "ReduceMean", "ReduceProd", "Sub", "Clip"}
NCNN_ALLOWED_LAYERS = {
    "Input", "Convolution", "ConvolutionDepthWise", "ReLU", "HardSwish", "Swish", "Sigmoid", "BinaryOp",
    "Concat", "Split", "Slice", "Interp", "Pooling", "Reshape", "Permute", "Reorg", "Crop",
}
NCNN_ORGAN_LAYERS = {"Gemm", "MatMul", "MemoryData", "UnaryOp", "Reduction", "Clip", "Eltwise", "Squeeze"}


def _model(variant: str, **overrides):
    torch.manual_seed(0)
    model = build_model({"model": {"variant": variant, **overrides}})
    # Randomised BN statistics make the BN fold observable in the exported weights.
    gen = torch.Generator().manual_seed(5)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.copy_(torch.rand(m.num_features, generator=gen) - 0.5)
            m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
    return model.eval()


def test_export_model_outputs_contract_shapes() -> None:
    model = _model("u_a")
    export_model = ExportModel(model).eval()
    assert model.fused
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        person, pose, organs = export_model(x)
        person2, pose2 = ExportModel(model, with_organs=False).eval()(x)
    n = sum((256 // s) ** 2 for s in model.strides)
    assert person.shape == (1, n, 1) and pose.shape == (1, n, 45) and organs.shape == (1, n, 8)
    # the organ output is purely additive: person and pose are bit-identical without it
    assert torch.equal(person, person2) and torch.equal(pose, pose2)
    assert torch.equal(organs, model.organs(pose))
    # Keypoint-major flattening: pose[:, :, k*5:(k+1)*5] == kpt[:, :, k, :].
    with torch.no_grad():
        raw = model(torch.zeros(1, 3, 256, 256)).one2one
        _, pose0, _ = export_model(torch.zeros(1, 3, 256, 256))
    assert torch.allclose(pose0.reshape(1, n, 9, 5), raw.kpt)


@pytest.mark.parametrize("variant,overrides,size", [("u_a", {}, 256), ("u_a", {"p2_assist": True, "act": "hswish"}, 320)])
def test_export_onnx_parity_and_allowlist(tmp_path: Path, variant: str, overrides: dict, size: int) -> None:
    import onnx
    import onnxruntime as ort

    model = _model(variant, **overrides)
    path = export_onnx(model, tmp_path / f"{variant}.onnx", input_size=size)
    assert path.exists()
    diff = verify_onnx(model, path, input_size=size)
    assert diff < 1e-3

    graph = onnx.load(str(path)).graph
    assert [i.name for i in graph.input] == [INPUT_NAME]
    assert [o.name for o in graph.output] == list(OUTPUT_NAMES)
    n = sum((size // s) ** 2 for s in model.strides)
    dims = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in graph.output]
    assert dims == [[1, n, 1], [1, n, 45], [1, n, 8]], "static shapes required"
    in_dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    assert in_dims == [1, 3, size, size]
    ops = onnx_op_types(path)
    allowed = ONNX_ALLOWED_OPS | ONNX_ORGAN_OPS
    assert set(ops) <= allowed, f"unexpected ONNX ops: {set(ops) - allowed}"

    # Batch-independent sanity: a constant input gives finite outputs through onnxruntime.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = session.run(None, {INPUT_NAME: np.full((1, 3, size, size), 0.5, np.float32)})
    assert all(np.isfinite(o).all() for o in outs)


def test_parse_ncnn_param(tmp_path: Path) -> None:
    text = "\n".join([
        "7767517",
        "4 5",
        "Input                    in0                      0 1 in0",
        "Convolution              conv_0                   1 1 in0 1 0=8 1=3",
        "Split                    splitncnn_0              1 2 1 2 3",
        "Convolution              conv_1                   1 1 2 out0 0=1 1=1",
        "Convolution              conv_2                   1 1 3 out1 0=45 1=1",
        "",
    ])
    p = tmp_path / "t.ncnn.param"
    p.write_text(text, encoding="utf-8")
    inputs, outputs, types = parse_ncnn_param(p)
    assert inputs == ["in0"] and outputs == ["out0", "out1"]
    assert types == {"Input": 1, "Convolution": 3, "Split": 1}


@pytest.mark.skipif(find_pnnx() is None, reason="pnnx executable not available")
def test_pnnx_conversion_names_and_layers(tmp_path: Path) -> None:
    model = _model("u_a", p2_assist=True)
    size = 256
    onnx_path = export_onnx(model, tmp_path / "u_a.onnx", input_size=size)
    conv = convert_onnx_to_ncnn(onnx_path, tmp_path / "u_a_256", fp16=True)
    assert conv.param_path.name == "u_a_256.ncnn.param" and conv.bin_path.exists()
    assert conv.input_names == ["in0"]
    assert conv.output_names == ["out0", "out1", "out2"]
    allowed = NCNN_ALLOWED_LAYERS | NCNN_ORGAN_LAYERS
    assert set(conv.layer_types) <= allowed, set(conv.layer_types) - allowed
    assert conv.layer_types.get("Reorg", 0) == 1  # SpaceToDepth of the P2 assist
    ncnn = pytest.importorskip("ncnn")
    assert ncnn is not None
    from xtx4.export.onnx_ncnn import verify_ncnn

    parity = verify_ncnn(model, conv, input_size=size)
    assert parity["person_max_abs"] < 5e-2 and parity["pose_max_abs"] < 5e-2  # fp16 weights
    assert parity["organs_max_abs"] < 5e-2


@pytest.mark.skipif(find_pnnx() is None, reason="pnnx executable not available")
def test_export_without_organs_keeps_two_outputs(tmp_path: Path) -> None:
    model = _model("u_a")
    onnx_path = export_onnx(model, tmp_path / "plain.onnx", input_size=256, with_organs=False)
    assert verify_onnx(model, onnx_path, input_size=256) < 1e-3
    assert set(onnx_op_types(onnx_path)) <= ONNX_ALLOWED_OPS
    conv = convert_onnx_to_ncnn(onnx_path, tmp_path / "plain_256", fp16=False)
    assert conv.output_names == ["out0", "out1"]
    assert set(conv.layer_types) <= NCNN_ALLOWED_LAYERS


def _organ_inputs(n: int = 600) -> np.ndarray:
    """Raw pose rows that exercise both brain branches (present logits far below / above the gate)."""

    rng = np.random.default_rng(7)
    pose = rng.normal(0.0, 4.0, size=(n, 9, 5)).astype(np.float32)
    pose[..., 3] = rng.choice([-9.0, -6.0, -2.0, 0.0, 3.0, 8.0], size=(n, 9))  # sigmoid(-6) < 0.01
    # (N, K*5): ncnn drops the batch axis of the full graph, so the layer sees exactly this rank there
    return pose.reshape(n, 45)


@pytest.mark.skipif(find_pnnx() is None, reason="pnnx executable not available")
@pytest.mark.parametrize("fp16", [False, True])
def test_organ_layer_ncnn_parity(tmp_path: Path, fp16: bool) -> None:
    ncnn = pytest.importorskip("ncnn")
    from xtx4.models.organs import OrganLayer

    layer = OrganLayer().eval()
    pose = _organ_inputs()
    onnx_path = tmp_path / "organs.onnx"
    with torch.no_grad():
        torch.onnx.export(layer, (torch.from_numpy(pose),), str(onnx_path), input_names=["in0"],
                          output_names=["organs"], opset_version=17, dynamo=False)
        expected = layer(torch.from_numpy(pose)).numpy()
    ok = 1.0 / (1.0 + np.exp(-pose.reshape(-1, 9, 5)[..., 3])) >= 0.01
    ear_branch = ok[:, 3] & ok[:, 4] & ~ok[:, :5].all(axis=1)
    assert ear_branch.any() and (~ear_branch).any()  # both brain branches are exercised
    conv = convert_onnx_to_ncnn(onnx_path, tmp_path / "organs", fp16=fp16)
    assert set(conv.layer_types) <= NCNN_ALLOWED_LAYERS | NCNN_ORGAN_LAYERS, conv.layer_types
    net = ncnn.Net()
    net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = net.opt.use_fp16_packed = fp16
    net.load_param(str(conv.param_path))
    net.load_model(str(conv.bin_path))
    ex = net.create_extractor()
    ex.input(conv.input_names[0], ncnn.Mat(np.ascontiguousarray(pose)))
    ret, mat = ex.extract(conv.output_names[0])
    assert ret == 0
    got = np.array(mat, dtype=np.float32).reshape(expected.shape)
    tol = 2e-2 if fp16 else 1e-4
    np.testing.assert_allclose(got, expected, atol=tol, rtol=tol)
