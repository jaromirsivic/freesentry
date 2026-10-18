"""Torch-free runtime: NumPy decode, letterbox, ncnn runner, capture/inference pipeline.

Nothing under ``xtx4.runtime`` may import torch (checked by
``tests/test_runtime.py`` in a clean subprocess); the Raspberry Pi deployment
needs only ``numpy``, ``opencv-python`` and ``ncnn``. ``ncnn`` itself is imported
lazily by :class:`NcnnPoseRunner` (and ``onnxruntime`` by the FP32 reference
:class:`OnnxPoseRunner`), so decode/preprocess work without them.
"""

from __future__ import annotations

from .decode import (
    DECODE_VERSION,
    DEFAULT_NMS_SCALE_FACTOR,
    KPT_CHANNELS_PER_POINT,
    PRESENT_THRESHOLD,
    SIGMA_FLOOR_UNITS,
    VISIBLE_THRESHOLD,
    Pose,
    anchor_count,
    box_iou_matrix,
    build_anchors,
    decode_organs_selected,
    decode_raw,
    decode_selected,
    infer_strides,
    merge_views,
    oks_matrix,
    oks_nms,
    oks_nms_indices,
    organs_from_pose_selected,
    person_scores,
    select_candidates,
    to_frame_poses,
)
from .ncnn_runner import DetectionResult, NcnnPoseRunner, PoseRunnerBase, RunnerError, Timings, shutdown_gpu
from .onnx_runner import OnnxPoseRunner
from .pipeline import InferencePipeline, LatestFrameQueue, PipelineResult, PipelineStats, run_headless
from .preprocess import PreparedViews, letterbox, prepare_views, to_blob

__all__ = [
    "DECODE_VERSION",
    "DEFAULT_NMS_SCALE_FACTOR",
    "KPT_CHANNELS_PER_POINT",
    "PRESENT_THRESHOLD",
    "SIGMA_FLOOR_UNITS",
    "VISIBLE_THRESHOLD",
    "Pose",
    "anchor_count",
    "box_iou_matrix",
    "build_anchors",
    "decode_organs_selected",
    "decode_raw",
    "decode_selected",
    "infer_strides",
    "merge_views",
    "oks_matrix",
    "oks_nms",
    "oks_nms_indices",
    "organs_from_pose_selected",
    "person_scores",
    "select_candidates",
    "to_frame_poses",
    "DetectionResult",
    "NcnnPoseRunner",
    "OnnxPoseRunner",
    "PoseRunnerBase",
    "RunnerError",
    "Timings",
    "shutdown_gpu",
    "InferencePipeline",
    "LatestFrameQueue",
    "PipelineResult",
    "PipelineStats",
    "run_headless",
    "PreparedViews",
    "letterbox",
    "prepare_views",
    "to_blob",
]
