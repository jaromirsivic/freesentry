"""ONNX + NCNN export for every XTX3 variant.

Pipeline (no ultralytics):

1. ``model.fuse()`` (Conv+BN fold, RepConv reparameterise, drop one-to-many head).
2. Export to ONNX at a fixed input ``(1, 3, S, S)``, opset >= 17. We export the
   **raw one-to-one head tensors** (box, cls, kpt) and decode in host code,
   keeping the ONNX/NCNN graph simple and edge-friendly.
3. Simplify with ``onnxsim``.
4. Verify the ONNX graph against PyTorch on deterministic random inputs
   (max-abs tolerance, ``onnxruntime``).
5. Convert ONNX -> NCNN via ``pnnx`` (preferred) or ``onnx2ncnn``, producing
   ``xtx3_{variant}_{size}.ncnn.param`` + ``.bin``.

Every variant is exported to both formats. The ``u`` variant is exported at all
of its supported input sizes (256/320/384 by default); the others at 384. The
trainer calls :func:`export_checkpoint` automatically after training.

Decode contract for the raw outputs (host side):
    * ``box`` (1, N, 4): raw ltrb; decode ``softplus(box) * stride`` around the
      anchor centres of the stride 8/16/32 (and 4 when P2) grids of the export
      input size.
    * ``cls`` (1, N, 1): person logit; score = sigmoid.
    * ``kpt`` (1, N, K*6): per keypoint ``(dx, dy, log_sigma, vis0, vis1, vis2)``;
      x = anchor_x + dx * stride, conf = softmax(vis)[1] + softmax(vis)[2].
    * Detections are in network pixels; map to original pixels with the view's
      affine ``T_k`` from ``xtx3.data.preprocess``.

Raspberry Pi 5 runtime dependencies (converter runs on the desktop):
    * desktop:  ``pip install onnx onnxsim onnxruntime pnnx``
    * Pi 5:     ``pip install ncnn numpy opencv-python`` (piwheels provides
      aarch64 builds; ncnn uses 4 threads by default on the Pi's Cortex-A76).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..data.preprocess import DEFAULT_NETWORK_SIZE, preprocess_image, view_keys_for_mode
from ..models import XTX3Model
from ..models.head import (
    KPT_CHANNELS_PER_POINT,
    build_anchors,
    decode_boxes,
    decode_keypoints,
    flatten_level_outputs,
)
from ..utils.keypoints import NUM_KEYPOINTS
from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_EXPORT_SIZES: dict[str, list[int]] = {
    "u": [256, 320, 384],
    "n": [384],
    "m": [384],
    "l": [384],
}

VERIFY_TOLERANCE = 1e-3  # max abs difference between PyTorch and ONNX Runtime


class ExportModel(nn.Module):
    """ONNX-friendly wrapper: returns raw one-to-one head tensors (box, cls, kpt)."""

    def __init__(self, model: XTX3Model) -> None:
        super().__init__()
        if not model._fused:  # noqa: SLF001 - intentional: export requires a fused graph
            model.fuse()
        self.model = model
        self.num_keypoints = model.num_keypoints

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.model.backbone(x)
        pyramid = self.model.neck(features)
        boxes, clss, kpts = self.model.head.one2one(pyramid)
        box, cls, kpt, _ = flatten_level_outputs(boxes, clss, kpts, self.num_keypoints)
        # Flatten kpt to (B, N, K*6) for a simple ONNX output.
        kpt = kpt.reshape(kpt.shape[0], kpt.shape[1], self.num_keypoints * KPT_CHANNELS_PER_POINT)
        return box, cls, kpt


def export_onnx(
    model: XTX3Model,
    output_path: Path,
    *,
    input_size: int = DEFAULT_NETWORK_SIZE,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """Export the fused one-to-one head to ONNX and (optionally) simplify."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = ExportModel(model).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)

    # Use the stable TorchScript-based exporter (dynamo=False) to keep a static,
    # NCNN/pnnx-friendly graph.
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        input_names=["in0"],
        output_names=["box", "cls", "kpt"],
        opset_version=opset,
        dynamic_axes=None,
        dynamo=False,
    )
    logger.info("Exported ONNX -> %s (opset %d, input %d)", output_path, opset, input_size)

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_onnx = onnx.load(str(output_path))
            simplified, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(output_path))
                logger.info("Simplified ONNX with onnxsim")
            else:
                logger.warning("onnxsim reported failure; keeping unsimplified ONNX")
        except ImportError:
            logger.warning("onnx/onnxsim not installed; skipping simplification")
        except Exception as exc:  # noqa: BLE001 - simplification is best-effort
            logger.warning("ONNX simplification failed (%s); keeping unsimplified ONNX", exc)
    return output_path


def verify_onnx(
    model: XTX3Model,
    onnx_path: Path,
    *,
    input_size: int,
    tolerance: float = VERIFY_TOLERANCE,
    seed: int = 0,
) -> float:
    """Compare ONNX Runtime outputs against PyTorch on a deterministic input.

    Returns the maximum absolute difference across the three output tensors and
    raises if it exceeds ``tolerance``.
    """

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping export verification")
        return float("nan")

    rng = np.random.default_rng(seed)
    inputs = rng.random((1, 3, input_size, input_size), dtype=np.float32)

    export_model = ExportModel(model).eval()
    with torch.no_grad():
        torch_outputs = [t.numpy() for t in export_model(torch.from_numpy(inputs))]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(["box", "cls", "kpt"], {"in0": inputs})

    max_diff = max(
        float(np.abs(a - b).max()) for a, b in zip(torch_outputs, ort_outputs)
    )
    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX verification failed for {onnx_path}: max |diff| = {max_diff:.2e} "
            f"> tolerance {tolerance:.2e}"
        )
    logger.info("Verified ONNX vs PyTorch: max |diff| = %.2e (tolerance %.0e)", max_diff, tolerance)
    return max_diff


def convert_onnx_to_ncnn(
    onnx_path: Path, out_prefix: Path, *, converter: str = "pnnx"
) -> tuple[Path, Path]:
    """Convert ONNX -> NCNN ``*.ncnn.param``/``*.ncnn.bin`` via pnnx or onnx2ncnn."""

    onnx_path = Path(onnx_path)
    out_prefix = Path(out_prefix)
    param_path = out_prefix.with_suffix(".ncnn.param")
    bin_path = out_prefix.with_suffix(".ncnn.bin")

    if converter == "pnnx":
        # Also look next to the running interpreter (venv Scripts/bin), which is
        # where "pip install pnnx" places the executable even when the venv is
        # not activated.
        tool = shutil.which("pnnx") or shutil.which(
            "pnnx", path=str(Path(sys.executable).parent)
        )
        if tool is None:
            raise RuntimeError(
                "pnnx not found on PATH. Install it (pip install pnnx) or use converter='onnx2ncnn'."
            )
        subprocess.run(
            [tool, str(onnx_path), f"ncnnparam={param_path}", f"ncnnbin={bin_path}"], check=True
        )
    elif converter == "onnx2ncnn":
        tool = shutil.which("onnx2ncnn")
        if tool is None:
            raise RuntimeError("onnx2ncnn not found on PATH. Install ncnn build tools.")
        subprocess.run([tool, str(onnx_path), str(param_path), str(bin_path)], check=True)
    else:
        raise ValueError(f"Unknown converter {converter!r}; expected 'pnnx' or 'onnx2ncnn'")

    if not param_path.exists() or not bin_path.exists():
        raise RuntimeError(f"Conversion did not produce expected files: {param_path}, {bin_path}")
    logger.info("Converted to NCNN -> %s + %s", param_path, bin_path)
    return param_path, bin_path


def export_sizes_for_variant(config: dict[str, Any], variant: str) -> list[int]:
    """Return the export input sizes for a variant from config (with defaults)."""

    export_cfg = config.get("export", {}) if isinstance(config, dict) else {}
    sizes_cfg = export_cfg.get("sizes", {})
    if isinstance(sizes_cfg, dict) and variant in sizes_cfg:
        return [int(s) for s in sizes_cfg[variant]]
    return list(DEFAULT_EXPORT_SIZES.get(variant, [DEFAULT_NETWORK_SIZE]))


def export_checkpoint(
    *,
    weights: Path,
    out_dir: Path,
    config: dict[str, Any] | None = None,
    variant: str | None = None,
    sizes: list[int] | None = None,
    converter: str | None = None,
    opset: int | None = None,
    skip_ncnn: bool = False,
) -> list[Path]:
    """Export a trained checkpoint to ONNX (+ NCNN) at every configured size.

    Produces ``xtx3_{variant}_{size}.onnx`` and ``xtx3_{variant}_{size}.ncnn.*``
    under ``out_dir`` and verifies each ONNX graph against PyTorch. Returns the
    list of produced ONNX paths.
    """

    from ..api import load_model

    model = load_model(weights, variant=variant or "n", device="cpu", fuse=True)
    variant = model.variant
    config = config or getattr(model, "xtx3_config", {}) or {}
    export_cfg = config.get("export", {}) if isinstance(config, dict) else {}
    sizes = sizes or export_sizes_for_variant(config, variant)
    converter = converter or str(export_cfg.get("converter", "pnnx"))
    opset = int(opset or export_cfg.get("onnx_opset", 17))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for size in sizes:
        prefix = out_dir / f"xtx3_{variant}_{size}"
        onnx_path = export_onnx(model, prefix.with_suffix(".onnx"), input_size=size, opset=opset)
        verify_onnx(model, onnx_path, input_size=size)
        produced.append(onnx_path)
        if not skip_ncnn:
            try:
                convert_onnx_to_ncnn(onnx_path, prefix, converter=converter)
            except RuntimeError as exc:
                logger.warning(
                    "NCNN conversion skipped for %s: %s (the ONNX artifact is ready; "
                    "install pnnx or ncnn tools and re-run to produce NCNN files)",
                    onnx_path, exc,
                )
    logger.info("Export complete: %d size(s) for XTX3-%s -> %s", len(sizes), variant, out_dir)
    return produced


def decode_raw_outputs(
    box: np.ndarray,
    cls: np.ndarray,
    kpt: np.ndarray,
    *,
    strides: list[int],
    num_keypoints: int = NUM_KEYPOINTS,
    input_size: int = DEFAULT_NETWORK_SIZE,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side decode of raw NCNN/ONNX outputs into ``(det (D,32), vis (D,K))``."""

    box_t = torch.from_numpy(np.ascontiguousarray(box)).float()
    cls_t = torch.from_numpy(np.ascontiguousarray(cls)).float()
    kpt_t = torch.from_numpy(np.ascontiguousarray(kpt)).float().reshape(
        box_t.shape[0], box_t.shape[1], num_keypoints, KPT_CHANNELS_PER_POINT
    )
    feature_sizes = [(input_size // s, input_size // s) for s in strides]
    anchors, stride_t = build_anchors(
        feature_sizes=feature_sizes, strides=strides, device=box_t.device, dtype=box_t.dtype
    )
    scores = torch.sigmoid(cls_t).squeeze(-1)[0]
    boxes = decode_boxes(box_t, anchors, stride_t)[0]
    kxy, kconf, kvis = decode_keypoints(kpt_t, anchors, stride_t)
    kxy, kconf, kvis = kxy[0], kconf[0], kvis[0]

    keep = scores >= conf_threshold
    if keep.sum() == 0:
        return np.zeros((0, 5 + 3 * num_keypoints), np.float32), np.zeros((0, num_keypoints), np.int64)
    s = scores[keep]
    bx = boxes[keep]
    kp = kxy[keep]
    kc = kconf[keep]
    kv = kvis[keep]
    if s.numel() > max_detections:
        idx = torch.topk(s, max_detections).indices
        s, bx, kp, kc, kv = s[idx], bx[idx], kp[idx], kc[idx], kv[idx]
    kpt_flat = torch.cat([kp, kc.unsqueeze(-1)], dim=-1).reshape(s.shape[0], -1)
    det = torch.cat([bx, s.unsqueeze(-1), kpt_flat], dim=-1)
    return det.numpy().astype(np.float32), kv.numpy().astype(np.int64)


class NCNNPoseRunner:
    """Raw-NCNN inference (input blob ``in0``) for whole-frame or pyramid mode."""

    def __init__(
        self,
        param_path: Path,
        bin_path: Path,
        *,
        strides: list[int],
        input_size: int = DEFAULT_NETWORK_SIZE,
        mode: str = "whole",
        num_keypoints: int = NUM_KEYPOINTS,
    ) -> None:
        try:
            import ncnn  # noqa: PLC0415 - optional, edge-only dependency
        except ImportError as exc:  # pragma: no cover - depends on platform install
            raise RuntimeError("The 'ncnn' python package is required for NCNNPoseRunner.") from exc
        self._ncnn = ncnn
        self.net = ncnn.Net()
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))
        self.strides = strides
        self.input_size = input_size
        self.mode = mode
        self.num_keypoints = num_keypoints

    def _run_single(self, view_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ncnn = self._ncnn
        mat = ncnn.Mat.from_pixels(
            np.ascontiguousarray(view_bgr),
            ncnn.Mat.PixelType.PIXEL_BGR,
            view_bgr.shape[1],
            view_bgr.shape[0],
        )
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        extractor = self.net.create_extractor()
        extractor.input("in0", mat)
        outputs = []
        for name in ("box", "cls", "kpt"):
            _, out = extractor.extract(name)
            outputs.append(np.array(out)[None, ...])
        return outputs[0], outputs[1], outputs[2]

    def detect(self, image: np.ndarray, *, conf_threshold: float = 0.25, max_detections: int = 300):
        """Run the configured pipeline; per-view raw decodes for host merge."""

        result = preprocess_image(image, mode=self.mode, network_size=self.input_size)
        per_view = {}
        for view in view_keys_for_mode(self.mode):
            box, cls, kpt = self._run_single(result.views[view])
            det, vis = decode_raw_outputs(
                box, cls, kpt,
                num_keypoints=self.num_keypoints,
                strides=self.strides,
                input_size=self.input_size,
                conf_threshold=conf_threshold,
                max_detections=max_detections,
            )
            per_view[view] = (det, vis, result.transforms[view])
        return per_view, result.orig_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an XTX3 checkpoint to ONNX + NCNN.")
    parser.add_argument("--weights", required=True, help="Trained XTX3 checkpoint (.pt)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <weights dir>/export)")
    parser.add_argument("--variant", choices=["u", "n", "m", "l"], default=None,
                        help="Only needed when the checkpoint lacks a variant field")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Input sizes to export (default: per-variant config)")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--converter", choices=["pnnx", "onnx2ncnn"], default=None)
    parser.add_argument("--no-ncnn", action="store_true", help="Export ONNX only (skip NCNN conversion)")
    return parser.parse_args()


def main() -> None:
    from ..utils.logging import configure_logging

    configure_logging()
    args = parse_args()
    weights = Path(args.weights)
    out_dir = Path(args.out_dir) if args.out_dir else weights.parent / "export"
    export_checkpoint(
        weights=weights,
        out_dir=out_dir,
        variant=args.variant,
        sizes=args.sizes,
        converter=args.converter,
        opset=args.opset,
        skip_ncnn=args.no_ncnn,
    )


if __name__ == "__main__":
    main()
