"""ONNX Runtime pose runner: the FP32 reference next to the deployed ncnn graph.

Same letterbox, decode and OKS-NMS path as :class:`~xtx4.runtime.ncnn_runner.NcnnPoseRunner`
(both derive from :class:`~xtx4.runtime.ncnn_runner.PoseRunnerBase`); only the graph
execution differs. Used by the live test to compare the FP16 ncnn deployment with
the exported FP32 ONNX graph on the same frames. Torch-free; ``onnxruntime`` is
imported lazily (CPU execution provider).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.logging import get_logger
from .decode import anchor_count
from .ncnn_runner import FROM_MANIFEST, PoseRunnerBase, RunnerError

logger = get_logger(__name__)


def onnx_file(manifest: Mapping[str, Any], size: int, base_dir: Path) -> Path:
    """Path of the ONNX graph for ``size`` listed in a deployment manifest."""

    files = manifest.get("outputs", {}).get("onnx", {}).get("files", {})
    entry = files.get(str(int(size)))
    if not entry:
        raise RunnerError(f"manifest has no ONNX graph for size {size} (sizes {sorted(files)})")
    return Path(base_dir) / entry["path"]


class OnnxPoseRunner(PoseRunnerBase):
    """One exported XTX4 ONNX graph at one input size (FP32, ONNX Runtime CPU)."""

    backend = "onnxruntime"

    def __init__(
        self,
        onnx_path: Path | str,
        *,
        size: int,
        strides: Sequence[int] | None = None,
        manifest: Mapping[str, Any] | Path | str | None = None,
        num_threads: int = 4,
        output_names: Sequence[str] = ("person", "pose"),
        num_keypoints: int | None = None,
        mode: str = "whole",
        sigma_conf_coeff: float | None = None,
        sigma_floor: float | None = None,
        sigmas: Sequence[float] | None = None,
        present_threshold: float | None = None,
        visible_threshold: float | None = None,
        oks_nms_threshold: float | None = None,
        oks_nms_iou_threshold: float | None | str = FROM_MANIFEST,
        oks_nms_scale_factor: float | None = None,
    ) -> None:
        import onnxruntime as ort

        self.session = None
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise RunnerError(f"Model file not found: {self.onnx_path}")
        strides = self._init_decode(
            size=size, mode=mode, manifest=manifest, strides=strides, num_keypoints=num_keypoints,
            sigma_conf_coeff=sigma_conf_coeff, sigma_floor=sigma_floor, sigmas=sigmas,
            present_threshold=present_threshold, visible_threshold=visible_threshold,
            oks_nms_threshold=oks_nms_threshold, oks_nms_iou_threshold=oks_nms_iou_threshold,
            oks_nms_scale_factor=oks_nms_scale_factor,
        )
        if self.manifest is not None:
            names = self.manifest.get("outputs", {}).get("onnx", {}).get("names")
            if names:
                output_names = (names["person"], names["pose"])

        self.num_threads = max(1, int(num_threads))
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(self.onnx_path), options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own exception types
            raise RunnerError(f"ONNX Runtime failed to load {self.onnx_path}: {exc}") from exc

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RunnerError(f"{self.onnx_path.name}: expected one input, got {[i.name for i in inputs]}")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) == 4 and all(isinstance(d, int) for d in shape[2:]) and tuple(shape[2:]) != (self.size, self.size):
            raise RunnerError(f"{self.onnx_path.name} expects {shape[2]}x{shape[3]} input, not size {self.size}")
        outputs = [o.name for o in self.session.get_outputs()]
        wanted = [str(n) for n in output_names]
        if all(n in outputs for n in wanted):
            self.output_names = (wanted[0], wanted[1])
        elif len(outputs) == 2:
            logger.warning("Outputs %s not in %s; using graph outputs %s (person, pose)", wanted,
                           self.onnx_path.name, outputs)
            self.output_names = (outputs[0], outputs[1])
        else:
            raise RunnerError(f"Outputs {wanted} not found; graph outputs: {outputs}")

        self.strides = self._validate_shapes(strides)
        self.num_anchors = anchor_count(self.size, self.strides)
        logger.info("Loaded %s @%d (strides %s, N=%d, ONNX Runtime CPU x%d, fp32)", self.onnx_path.name, self.size,
                    list(self.strides), self.num_anchors, self.num_threads)

    @classmethod
    def from_manifest(cls, manifest_path: Path | str, size: int, **kwargs: Any) -> "OnnxPoseRunner":
        """Build a runner from ``<export>/manifest.json`` (graph resolved next to it)."""

        from ..export.manifest import load_manifest

        path = Path(manifest_path)
        if path.is_dir():
            path = path / "manifest.json"
        manifest = load_manifest(path)
        return cls(onnx_file(manifest, size, path.parent), size=size, manifest=manifest, **kwargs)

    # ------------------------------------------------------------ inference
    def forward_blob(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw outputs for an already normalised ``(1, 3, S, S)`` / ``(3, S, S)`` float32 tensor."""

        if self.session is None:
            raise RunnerError("runner is closed")
        arr = np.asarray(blob, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None]
        person, pose = self.session.run(list(self.output_names), {self.input_name: np.ascontiguousarray(arr)})
        return (np.asarray(person, dtype=np.float32).reshape(-1, 1),
                np.asarray(pose, dtype=np.float32).reshape(person.size, -1))

    def forward(self, view_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw outputs ``(person (N, 1), pose (N, K*5))`` for one letterboxed uint8 BGR view."""

        view = np.asarray(view_bgr, dtype=np.uint8)
        blob = view.transpose(2, 0, 1).astype(np.float32) * np.float32(1.0 / 255.0)
        return self.forward_blob(blob)

    # ------------------------------------------------------------- lifetime
    def close(self) -> None:
        self.session = None

    def describe(self) -> str:
        return f"{self.onnx_path.name} @{self.size} {self.mode} ONNX Runtime CPU x{self.num_threads} fp32"
