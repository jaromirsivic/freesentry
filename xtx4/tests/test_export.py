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
NCNN_ALLOWED_LAYERS = {
    "Input", "Convolution", "ConvolutionDepthWise", "ReLU", "HardSwish", "Swish", "Sigmoid", "BinaryOp",
    "Concat", "Split", "Slice", "Interp", "Pooling", "Reshape", "Permute", "Reorg", "Crop",
}


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
    with torch.no_grad():
        person, pose = export_model(torch.rand(1, 3, 256, 256))
    n = sum((256 // s) ** 2 for s in model.strides)
    assert person.shape == (1, n, 1) and pose.shape == (1, n, 45)
    # Keypoint-major flattening: pose[:, :, k*5:(k+1)*5] == kpt[:, :, k, :].
    with torch.no_grad():
        raw = model(torch.zeros(1, 3, 256, 256)).one2one
        _, pose0 = export_model(torch.zeros(1, 3, 256, 256))
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
    assert dims == [[1, n, 1], [1, n, 45]], "static shapes required"
    in_dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    assert in_dims == [1, 3, size, size]
    ops = onnx_op_types(path)
    assert set(ops) <= ONNX_ALLOWED_OPS, f"unexpected ONNX ops: {set(ops) - ONNX_ALLOWED_OPS}"

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
    assert conv.output_names == ["out0", "out1"]
    assert set(conv.layer_types) <= NCNN_ALLOWED_LAYERS, set(conv.layer_types) - NCNN_ALLOWED_LAYERS
    assert conv.layer_types.get("Reorg", 0) == 1  # SpaceToDepth of the P2 assist
    ncnn = pytest.importorskip("ncnn")
    assert ncnn is not None
    from xtx4.export.onnx_ncnn import verify_ncnn

    parity = verify_ncnn(model, conv, input_size=size)
    assert parity["person_max_abs"] < 5e-2 and parity["pose_max_abs"] < 5e-2  # fp16 weights
