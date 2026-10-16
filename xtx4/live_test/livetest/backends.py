"""``NetworkConfig`` + ``Settings`` -> a pose runner from ``xtx4.runtime`` (decode lives there).

Two backends share the letterbox / decode / OKS-NMS code (``PoseRunnerBase``):

* ``ncnn`` -- the deployment path (Raspberry Pi): FP16 or FP32 weights depending on
  the export, FP16 arithmetic switchable, CPU threads or Vulkan GPU;
* ``onnxruntime`` -- the exported FP32 ONNX graph on the CPU, a reference to
  compare the ncnn output with on the same frames.
"""

from __future__ import annotations

import importlib.util
import logging

from xtx4.runtime import DetectionResult, NcnnPoseRunner, PoseRunnerBase, RunnerError, shutdown_gpu
from xtx4.runtime.ncnn_runner import vulkan_device_count

from .config import ConfigError, NetworkConfig
from .settings import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "DetectionResult", "PoseRunnerBase", "RunnerError", "apply_decode_attrs", "backend_installed", "create_runner",
    "shutdown_gpu", "vulkan_devices",
]

BACKEND_LABELS = {"ncnn": "ncnn (deployment)", "onnxruntime": "ONNX Runtime FP32 (reference)"}
_vulkan_cache: list[str] | None = None


def backend_installed(backend: str) -> bool:
    module = {"ncnn": "ncnn", "onnxruntime": "onnxruntime"}.get(backend)
    return module is not None and importlib.util.find_spec(module) is not None


def vulkan_devices() -> list[str]:
    """Names of the Vulkan devices ncnn can use (cached; empty without Vulkan)."""

    global _vulkan_cache
    if _vulkan_cache is None:
        names: list[str] = []
        try:
            count = vulkan_device_count()
            import ncnn

            for index in range(count):
                try:
                    names.append(str(ncnn.get_gpu_info(index).device_name()))
                except Exception:  # noqa: BLE001 - older ncnn bindings
                    names.append(f"GPU {index}")
        except Exception as exc:  # noqa: BLE001 - ncnn without Vulkan
            logger.debug("Vulkan unavailable: %s", exc)
        _vulkan_cache = names
    return list(_vulkan_cache)


def create_runner(network: NetworkConfig, settings: Settings) -> PoseRunnerBase:
    """Load ``network`` at ``settings.size`` with ``settings.backend`` and the decode attributes."""

    size = settings.size
    if size not in network.sizes:
        raise RunnerError(f"Size {size} is not configured for network '{network.id}' (sizes {network.sizes})")
    if network.manifest and network.manifest_data is None:
        raise RunnerError(f"Network '{network.id}': {network.manifest_error or 'manifest not loaded'}")
    if not backend_installed(settings.backend):
        raise RunnerError(f"Backend {settings.backend!r} is not installed (pip install {settings.backend})")
    common = dict(
        size=size,
        strides=network.strides,
        manifest=network.manifest_data,
        num_threads=settings.num_threads,
        num_keypoints=network.skeleton.num_keypoints,
        sigmas=network.skeleton.oks_sigmas,
        **settings.decode_attrs(),
    )
    if settings.backend == "onnxruntime":
        from xtx4.runtime import OnnxPoseRunner

        try:
            onnx = network.onnx_path(size)
        except ConfigError as exc:
            raise RunnerError(str(exc)) from exc
        if not onnx.is_file():
            raise RunnerError(f"Model file not found: {onnx}")
        return OnnxPoseRunner(onnx, **common)

    param, bin_path = network.param_path(size), network.bin_path(size)
    for path in (param, bin_path):
        if not path.is_file():
            raise RunnerError(f"Model file not found: {path}")
    return NcnnPoseRunner(
        param,
        bin_path,
        use_vulkan=settings.use_vulkan,
        fp16=settings.fp16,
        input_blob=network.input_blob,
        output_blobs=network.output_blobs,
        **common,
    )


def apply_decode_attrs(runner: PoseRunnerBase, settings: Settings) -> None:
    """Set the decode attributes that need no graph reload (call while no detection runs)."""

    for key, value in settings.decode_attrs().items():
        setattr(runner, key, value)
