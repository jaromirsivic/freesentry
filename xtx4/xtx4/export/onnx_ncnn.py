"""ONNX + ncnn export of the fused one2one XTX4 graph (CONTRACTS.md section 12).

Pipeline (no ultralytics):

1. ``model.fuse()`` (Conv+BN fold, RepConv reparameterise, drop one2many).
2. :class:`ExportModel` returns the **raw** one2one tensors ``person [1, N, 1]``
   and ``pose [1, N, K*5]`` (keypoint-major: all five values of keypoint 0, then
   keypoint 1, ...) plus, by default, ``organs [1, N, 8]`` from the parameter-free
   :class:`~xtx4.models.organs.OrganLayer` (brain, chest: ``dx, dy, r_units, conf``).
   Decoding happens on the host (``xtx4.runtime.decode``), so the graph stays
   static and ncnn-friendly: Conv / activation / Concat / Interp / Pooling /
   Reshape / Permute, plus MatMul / BinaryOp / UnaryOp / Reduction / Clip for the
   organ layer.
3. ONNX export at a fixed ``[1, 3, S, S]`` with the TorchScript exporter
   (``dynamo=False``), then ``onnxsim`` and ``onnx.checker``.
4. :func:`verify_onnx`: onnxruntime vs PyTorch on a deterministic input
   (max abs diff, tolerance 1e-3 on raw outputs).
5. :func:`convert_onnx_to_ncnn`: ``pnnx`` (``fp16=1`` for deployment, ``fp16=0``
   for the FP32 reference) -> ``<prefix>.ncnn.param`` / ``.bin``. pnnx renames
   blobs: the input is ``in0`` and the outputs are ``out0`` (person), ``out1``
   (pose) and ``out2`` (organs). The real names are parsed from the ``.ncnn.param`` file and returned
   in :class:`NcnnConversion` so the deployment manifest can record them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

from ..models.head_types import KPT_CHANNELS_PER_POINT, flatten_level_outputs
from ..models.xtx4 import XTX4Model
from ..utils.organs import gate_ramp_rows
from ..utils.logging import get_logger

logger = get_logger(__name__)

INPUT_NAME = "in0"
OUTPUT_NAMES: tuple[str, ...] = ("person", "pose", "organs")
POSE_OUTPUT_NAMES: tuple[str, str] = ("person", "pose")  # graphs exported with ``with_organs=False``
VERIFY_TOLERANCE = 1e-3  # max abs difference between PyTorch and ONNX Runtime (raw outputs)
DEFAULT_OPSET = 17


def output_names(with_organs: bool) -> tuple[str, ...]:
    return OUTPUT_NAMES if with_organs else POSE_OUTPUT_NAMES


class ExportModel(nn.Module):
    """ONNX-friendly wrapper around a fused model: ``(person [1,N,1], pose [1,N,K*5])`` and, with
    ``with_organs`` (default), ``organs [1,N,8]`` -- the organ layer is the last layer of the graph."""

    def __init__(self, model: XTX4Model, *, with_organs: bool = True) -> None:
        super().__init__()
        if not model.fused:
            model.fuse()
        if with_organs and model.organs is None:
            raise ValueError("with_organs needs the nine XTX4 landmarks (model.organs is None)")
        self.model = model
        self.num_keypoints = model.num_keypoints
        self.with_organs = with_organs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        pyramid, assist = self.model.forward_features(x)
        persons, poses = self.model.head.one2one(pyramid, assist)
        person, kpt, _ = flatten_level_outputs(persons, poses, self.num_keypoints)
        pose = kpt.reshape(kpt.shape[0], kpt.shape[1], self.num_keypoints * KPT_CHANNELS_PER_POINT)
        if not self.with_organs:
            return person, pose
        return person, pose, self.model.organs(pose)


def deterministic_input(input_size: int, *, seed: int = 0) -> np.ndarray:
    """Uniform ``[0, 1)`` float32 image batch ``(1, 3, S, S)`` with a fixed seed."""

    rng = np.random.default_rng(seed)
    return rng.random((1, 3, input_size, input_size), dtype=np.float32)


def export_onnx(
    model: XTX4Model,
    path: Path | str,
    *,
    input_size: int,
    opset: int = DEFAULT_OPSET,
    simplify: bool = True,
    check: bool = True,
    with_organs: bool = True,
) -> Path:
    """Export the fused one2one graph to ONNX (static shape), simplify and check it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if input_size % 32:
        raise ValueError(f"input_size must be a multiple of 32, got {input_size}")
    export_model = ExportModel(model, with_organs=with_organs).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=list(output_names(with_organs)),
            opset_version=opset,
            dynamic_axes=None,
            dynamo=False,
        )
    logger.info("Exported ONNX -> %s (opset %d, input %d)", path, opset, input_size)

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_onnx = onnx.load(str(path))
            simplified, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(path))
                logger.info("Simplified ONNX with onnxsim")
            else:
                logger.warning("onnxsim reported failure; keeping unsimplified ONNX")
        except ImportError:
            logger.warning("onnx/onnxsim not installed; skipping simplification")
        except Exception as exc:  # noqa: BLE001 - simplification is best-effort
            logger.warning("ONNX simplification failed (%s); keeping unsimplified ONNX", exc)
    if check:
        import onnx

        onnx.checker.check_model(str(path))
    return path


def onnx_op_types(path: Path | str) -> Counter[str]:
    """Histogram of ONNX node op types (for the deploy-graph allowlist check)."""

    import onnx

    graph = onnx.load(str(path)).graph
    return Counter(node.op_type for node in graph.node)


def verify_onnx(
    model: XTX4Model,
    path: Path | str,
    *,
    input_size: int,
    tolerance: float = VERIFY_TOLERANCE,
    seed: int = 0,
) -> float:
    """Max abs diff between onnxruntime and PyTorch on a deterministic input (raises > tolerance).

    All outputs are compared. For ``organs`` the anchors whose face confidence lies in the brain
    gate's ramp are left out (:func:`xtx4.utils.organs.gate_ramp_rows`: the gate's slope turns
    float32 rounding into a ~1e-3 branch blend there).
    """

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping export verification")
        return float("nan")

    inputs = deterministic_input(input_size, seed=seed)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with_organs = "organs" in {o.name for o in session.get_outputs()}
    names = output_names(with_organs)
    export_model = ExportModel(model, with_organs=with_organs).eval()
    with torch.no_grad():
        torch_outputs = [t.cpu().numpy() for t in export_model(torch.from_numpy(inputs))]

    ort_outputs = session.run(list(names), {INPUT_NAME: inputs})
    for name, a, b in zip(names, torch_outputs, ort_outputs):
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"ONNX output {name!r} shape {b.shape} != PyTorch {a.shape}")
    diffs = [np.abs(a - b) for a, b in zip(torch_outputs, ort_outputs)]
    skipped = 0
    if with_organs:
        pose = torch_outputs[1].reshape(-1, model.num_keypoints, KPT_CHANNELS_PER_POINT)
        ramp = gate_ramp_rows(1.0 / (1.0 + np.exp(-pose[..., 3].astype(np.float64))))
        skipped = int(ramp.sum())
        diffs[2] = diffs[2].reshape(ramp.shape[0], -1)[~ramp]
    max_diff = max(float(d.max()) if d.size else 0.0 for d in diffs)
    if skipped:
        logger.info("ONNX organs check: %d anchor(s) in the brain gate ramp left out", skipped)
    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX verification failed for {path}: max |diff| = {max_diff:.2e} > tolerance {tolerance:.2e}"
        )
    logger.info("Verified ONNX vs PyTorch: max |diff| = %.2e (tolerance %.0e)", max_diff, tolerance)
    return max_diff


@dataclass(slots=True)
class NcnnConversion:
    """Result of :func:`convert_onnx_to_ncnn`.

    ``output_names`` are the blob names pnnx produced, in graph order (person
    first, pose second, organs third for XTX4 graphs: normally ``["out0", "out1", "out2"]``).
    """

    param_path: Path
    bin_path: Path
    input_names: list[str]
    output_names: list[str]
    layer_types: dict[str, int] = field(default_factory=dict)
    fp16: bool = True
    converter: str = "pnnx"


def parse_ncnn_param(param_path: Path | str) -> tuple[list[str], list[str], dict[str, int]]:
    """Return ``(input_blobs, output_blobs, layer_type_histogram)`` of an ncnn ``.param``.

    Output blobs are those produced by a layer and never consumed by another.
    """

    lines = Path(param_path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError(f"{param_path} is not an ncnn param file (magic 7767517 missing)")
    produced: list[str] = []
    consumed: set[str] = set()
    inputs: list[str] = []
    types: Counter[str] = Counter()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        layer_type, _name, n_in, n_out = parts[0], parts[1], int(parts[2]), int(parts[3])
        blobs_in = parts[4 : 4 + n_in]
        blobs_out = parts[4 + n_in : 4 + n_in + n_out]
        types[layer_type] += 1
        if layer_type == "Input":
            inputs.extend(blobs_out)
        consumed.update(blobs_in)
        produced.extend(blobs_out)
    outputs = [b for b in produced if b not in consumed and b not in inputs]
    return inputs, outputs, dict(types)


def find_pnnx() -> str | None:
    """Locate the ``pnnx`` executable (PATH, then next to the running interpreter)."""

    return shutil.which("pnnx") or shutil.which("pnnx", path=str(Path(sys.executable).parent))


def convert_onnx_to_ncnn(
    onnx_path: Path | str,
    prefix: Path | str,
    *,
    converter: str = "pnnx",
    fp16: bool = True,
    timeout_s: float = 600.0,
    keep_intermediates: bool = False,
) -> NcnnConversion:
    """Convert ONNX -> ``<prefix>.ncnn.param`` / ``<prefix>.ncnn.bin`` via pnnx.

    pnnx also writes its own ``<prefix>.pnnx.{param,bin,onnx}``, ``<prefix>_pnnx.py``,
    ``<prefix>_ncnn.py`` and ``<onnx stem>.pnnxsim.onnx``; they are deleted after a
    successful conversion unless ``keep_intermediates`` is set (they are several
    times the size of the deployment files).
    """

    onnx_path = Path(onnx_path)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    param_path = prefix.parent / f"{prefix.name}.ncnn.param"
    bin_path = prefix.parent / f"{prefix.name}.ncnn.bin"

    if converter == "pnnx":
        tool = find_pnnx()
        if tool is None:
            raise RuntimeError("pnnx not found on PATH or next to the interpreter (pip install pnnx)")
        intermediates = {
            "ncnnpy": prefix.parent / f"{prefix.name}_ncnn.py",
            "pnnxparam": prefix.parent / f"{prefix.name}.pnnx.param",
            "pnnxbin": prefix.parent / f"{prefix.name}.pnnx.bin",
            "pnnxpy": prefix.parent / f"{prefix.name}_pnnx.py",
            "pnnxonnx": prefix.parent / f"{prefix.name}.pnnx.onnx",
        }
        cmd = [
            tool,
            str(onnx_path),
            f"ncnnparam={param_path}",
            f"ncnnbin={bin_path}",
            *(f"{key}={value}" for key, value in intermediates.items()),
            f"fp16={1 if fp16 else 0}",
        ]
        logger.info("Running pnnx: %s", " ".join(cmd))
        # pnnx prints non-ASCII bytes on Windows consoles; never let decoding fail the export.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pnnx failed (exit {proc.returncode}) for {onnx_path}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        if not keep_intermediates:
            for extra in list(intermediates.values()) + [onnx_path.with_name(onnx_path.stem + ".pnnxsim.onnx")]:
                try:
                    extra.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Could not remove pnnx intermediate %s: %s", extra, exc)
    elif converter == "onnx2ncnn":
        tool = shutil.which("onnx2ncnn")
        if tool is None:
            raise RuntimeError("onnx2ncnn not found on PATH. Install ncnn build tools.")
        subprocess.run([tool, str(onnx_path), str(param_path), str(bin_path)], check=True, timeout=timeout_s)
    else:
        raise ValueError(f"Unknown converter {converter!r}; expected 'pnnx' or 'onnx2ncnn'")

    if not param_path.exists() or not bin_path.exists():
        raise RuntimeError(f"Conversion did not produce expected files: {param_path}, {bin_path}")
    inputs, outputs, types = parse_ncnn_param(param_path)
    unconverted = sorted(t for t in types if "." in t)  # pnnx.* / F.* fallbacks ncnn cannot execute
    if unconverted:
        raise RuntimeError(f"pnnx left operators ncnn cannot run in {param_path.name}: {unconverted}")
    logger.info(
        "Converted to ncnn -> %s (+ .bin, fp16=%d): inputs=%s outputs=%s layers=%s",
        param_path, int(fp16), inputs, outputs, dict(sorted(types.items())),
    )
    return NcnnConversion(
        param_path=param_path, bin_path=bin_path, input_names=inputs, output_names=outputs,
        layer_types=types, fp16=fp16, converter=converter,
    )


def verify_ncnn(
    model: XTX4Model,
    conversion: NcnnConversion,
    *,
    input_size: int,
    seed: int = 0,
    num_threads: int = 4,
) -> dict[str, float]:
    """Compare the ncnn graph with PyTorch on the deterministic input (needs the ``ncnn`` package).

    Returns ``{"person_max_abs", "pose_max_abs", "pose_xy_max_abs"}`` (raw units;
    xy in stride units) and, for a graph with the organ output, ``organs_max_abs``,
    ``organs_xyr_max_abs`` (stride units) and ``organs_conf_max_abs``. Raises ImportError
    when ncnn is not installed.
    """

    import ncnn

    inputs = deterministic_input(input_size, seed=seed)
    with_organs = len(conversion.output_names) >= 3
    export_model = ExportModel(model, with_organs=with_organs).eval()
    with torch.no_grad():
        torch_outs = [t.cpu().numpy() for t in export_model(torch.from_numpy(inputs))]
    person_t, pose_t = torch_outs[0], torch_outs[1]

    net = ncnn.Net()
    net.opt.num_threads = num_threads
    net.opt.use_fp16_storage = conversion.fp16
    net.opt.use_fp16_arithmetic = conversion.fp16
    net.opt.use_fp16_packed = conversion.fp16
    net.load_param(str(conversion.param_path))
    net.load_model(str(conversion.bin_path))
    ex = net.create_extractor()
    ex.input(conversion.input_names[0], ncnn.Mat(np.ascontiguousarray(inputs[0])))
    outs = []
    for name in conversion.output_names:
        ret, mat = ex.extract(name)
        if ret != 0:
            raise RuntimeError(f"ncnn extract({name!r}) failed with code {ret}")
        outs.append(np.array(mat, dtype=np.float32))
    if outs[0].size != person_t.size or outs[1].size != pose_t.size:
        raise RuntimeError(
            f"ncnn output sizes {[o.shape for o in outs]} do not match PyTorch {[person_t.shape, pose_t.shape]}"
        )
    person_n = outs[0].reshape(person_t.shape)
    pose_n = outs[1].reshape(pose_t.shape)
    num_keypoints = model.num_keypoints
    pose_t4 = pose_t.reshape(1, -1, num_keypoints, KPT_CHANNELS_PER_POINT)
    pose_n4 = pose_n.reshape(1, -1, num_keypoints, KPT_CHANNELS_PER_POINT)
    result = {
        "person_max_abs": float(np.abs(person_t - person_n).max()),
        "pose_max_abs": float(np.abs(pose_t - pose_n).max()),
        "pose_xy_max_abs": float(np.abs(pose_t4[..., :2] - pose_n4[..., :2]).max()),
    }
    if with_organs:
        organs_t = torch_outs[2]
        if outs[2].size != organs_t.size:
            raise RuntimeError(f"ncnn organs output size {outs[2].shape} does not match PyTorch {organs_t.shape}")
        organs_t4 = organs_t.reshape(1, -1, 2, 4)
        organs_n4 = outs[2].reshape(organs_t4.shape)
        result.update({
            "organs_max_abs": float(np.abs(organs_t4 - organs_n4).max()),
            "organs_xyr_max_abs": float(np.abs(organs_t4[..., :3] - organs_n4[..., :3]).max()),
            "organs_conf_max_abs": float(np.abs(organs_t4[..., 3] - organs_n4[..., 3]).max()),
        })
    return result
