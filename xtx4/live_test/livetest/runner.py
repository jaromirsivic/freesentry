"""Thin adapter: ``NetworkConfig`` -> ``xtx4.runtime.NcnnPoseRunner`` (decode lives in xtx4.runtime)."""

from __future__ import annotations

from xtx4.runtime import DetectionResult, NcnnPoseRunner, RunnerError, shutdown_gpu

from .config import NetworkConfig

__all__ = ["DetectionResult", "NcnnPoseRunner", "RunnerError", "create_runner", "shutdown_gpu"]


def create_runner(
    network: NetworkConfig,
    size: int,
    *,
    num_threads: int = 4,
    use_vulkan: bool = False,
    fp16: bool = True,
    decode_overrides: dict | None = None,
) -> NcnnPoseRunner:
    """Runner for ``network`` at ``size``; ``decode_overrides`` (``InferenceConfig.decode_overrides()``)
    win over the manifest ``decode`` block (visibility thresholds, OKS-NMS threshold / IoU / scale)."""
    if size not in network.sizes:
        raise RunnerError(f"Size {size} is not configured for network '{network.id}' (sizes {network.sizes})")
    if network.manifest and network.manifest_data is None:
        raise RunnerError(f"Network '{network.id}': {network.manifest_error or 'manifest not loaded'}")
    param, bin_path = network.param_path(size), network.bin_path(size)
    for path in (param, bin_path):
        if not path.is_file():
            raise RunnerError(f"Model file not found: {path}")
    return NcnnPoseRunner(
        param,
        bin_path,
        size=size,
        strides=network.strides,
        manifest=network.manifest_data,
        num_threads=num_threads,
        use_vulkan=use_vulkan,
        fp16=fp16,
        input_blob=network.input_blob,
        output_blobs=network.output_blobs,
        num_keypoints=network.skeleton.num_keypoints,
        mode=network.mode,
        sigmas=network.skeleton.oks_sigmas,
        **(decode_overrides or {}),
    )
