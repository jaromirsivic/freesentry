"""NCNN export for XTX2-n (spec section 10.2).

Pipeline (no ultralytics):

1. ``model.fuse()`` (Conv+BN fold, RepConv reparameterise, drop one-to-many head).
2. Export to ONNX at fixed input ``(1, 3, 384, 384)``, opset >= 17. We export the
   **raw one-to-one head tensors** (box, cls, kpt) and decode in host code,
   keeping the ONNX/NCNN graph simple and edge-friendly.
3. Simplify with ``onnxsim``.
4. Convert ONNX -> NCNN via ``pnnx`` (preferred) or ``onnx2ncnn``, producing
   ``xtx2_n.ncnn.param`` + ``xtx2_n.ncnn.bin``.
5. A raw-NCNN inference path (:class:`NCNNPoseRunner`) loads ``*.param``/``*.bin``,
   runs the 3 levels (CHW float32 / 255), decodes, and merges in host code.

Decode contract for the raw outputs (host side):
    * ``box`` (1, N, 4): raw ltrb; decode ``softplus(box) * stride`` around the
      anchor centres of the 48/24/12 (and 96 when P2) grids.
    * ``cls`` (1, N, 1): person logit; score = sigmoid.
    * ``kpt`` (1, N, K*6): per keypoint ``(dx, dy, log_sigma, vis0, vis1, vis2)``;
      x = anchor_x + dx * stride, conf = softmax(vis)[1] + softmax(vis)[2].
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.preprocess import LEVEL_KEYS, NETWORK_SIZE, preprocess_image
from ..models import XTX2Model
from ..models.head import (
    KPT_CHANNELS_PER_POINT,
    build_anchors,
    decode_boxes,
    decode_keypoints,
    flatten_level_outputs,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExportModel(nn.Module):
    """ONNX-friendly wrapper: returns raw one-to-one head tensors (box, cls, kpt)."""

    def __init__(self, model: XTX2Model) -> None:
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
    model: XTX2Model,
    output_path: Path,
    *,
    input_size: int = NETWORK_SIZE,
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
    logger.info("Exported ONNX -> %s (opset %d)", output_path, opset)

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


def convert_onnx_to_ncnn(
    onnx_path: Path, out_prefix: Path, *, converter: str = "pnnx"
) -> tuple[Path, Path]:
    """Convert ONNX -> NCNN ``*.ncnn.param``/``*.ncnn.bin`` via pnnx or onnx2ncnn."""

    onnx_path = Path(onnx_path)
    out_prefix = Path(out_prefix)
    param_path = out_prefix.with_suffix(".ncnn.param")
    bin_path = out_prefix.with_suffix(".ncnn.bin")

    if converter == "pnnx":
        tool = shutil.which("pnnx")
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


def decode_raw_outputs(
    box: np.ndarray,
    cls: np.ndarray,
    kpt: np.ndarray,
    *,
    num_keypoints: int,
    strides: list[int],
    input_size: int = NETWORK_SIZE,
    conf_threshold: float = 0.25,
    max_detections: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side decode of raw NCNN/ONNX outputs into ``(det (D,56), vis (D,K))``."""

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
    """Raw-NCNN inference for the 3-level XTX2-n pipeline (input blob ``in0``)."""

    def __init__(
        self, param_path: Path, bin_path: Path, *, strides: list[int], num_keypoints: int = 17
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
        self.num_keypoints = num_keypoints

    def _run_single(self, level_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ncnn = self._ncnn
        mat = ncnn.Mat.from_pixels(
            np.ascontiguousarray(level_bgr),
            ncnn.Mat.PixelType.PIXEL_BGR,
            level_bgr.shape[1],
            level_bgr.shape[0],
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
        """Run the full 3-level pipeline; per-level raw decodes for host merge."""

        result = preprocess_image(image)
        per_level = {}
        for level in LEVEL_KEYS:
            box, cls, kpt = self._run_single(result.levels[level])
            det, vis = decode_raw_outputs(
                box, cls, kpt,
                num_keypoints=self.num_keypoints,
                strides=self.strides,
                conf_threshold=conf_threshold,
                max_detections=max_detections,
            )
            per_level[level] = (det, vis, result.transforms[level])
        return per_level, result.orig_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XTX2-n to NCNN.")
    parser.add_argument("--weights", required=True, help="Trained XTX2-n checkpoint")
    parser.add_argument("--out", default="xtx2_n", help="Output prefix for ONNX/NCNN files")
    parser.add_argument("--variant", choices=["n", "m", "l"], default="n")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--converter", choices=["pnnx", "onnx2ncnn"], default="pnnx")
    parser.add_argument("--no-ncnn", action="store_true", help="Export ONNX only (skip NCNN conversion)")
    return parser.parse_args()


def main() -> None:
    from ..api import load_model
    from ..utils.logging import configure_logging

    configure_logging()
    args = parse_args()
    model = load_model(args.weights, variant=args.variant, device="cpu", fuse=True)
    onnx_path = export_onnx(model, Path(f"{args.out}.onnx"), opset=args.opset)
    if not args.no_ncnn:
        convert_onnx_to_ncnn(onnx_path, Path(args.out), converter=args.converter)


if __name__ == "__main__":
    main()
