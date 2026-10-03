"""NCNN pose runner: frame -> views -> NCNN -> decode -> merge -> poses."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import NetworkConfig
from .decode import Pose, anchor_count, decode_raw, infer_strides, to_frame_poses
from .merge import merge_views
from .preprocess import PreparedViews, prepare_views

logger = logging.getLogger(__name__)

# Blob names produced by the ONNX exporter; pnnx renames them to out0..out2.
_FALLBACK_OUTPUT_BLOBS = {"box": "box", "cls": "cls", "kpt": "kpt"}
_NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]

_gpu_instance_used = False


class RunnerError(RuntimeError):
    pass


def vulkan_device_count() -> int:
    """Number of Vulkan devices ncnn can use (0 when unavailable)."""

    global _gpu_instance_used
    import ncnn

    try:
        count = int(ncnn.get_gpu_count())
    except Exception:  # noqa: BLE001 - ncnn builds without Vulkan raise here
        return 0
    _gpu_instance_used = True
    return count


def shutdown_gpu() -> None:
    """Release the ncnn Vulkan instance (must run after every Net is cleared).

    Skipping this makes the Windows process exit with a non-zero code.
    """

    global _gpu_instance_used
    if not _gpu_instance_used:
        return
    import ncnn

    try:
        ncnn.destroy_gpu_instance()
    finally:
        _gpu_instance_used = False


@dataclass(slots=True)
class Timings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(slots=True)
class DetectionResult:
    poses: list[Pose]
    timings: Timings
    prepared: PreparedViews
    raw_candidates: int = 0
    extras: dict[str, float] = field(default_factory=dict)


class NcnnPoseRunner:
    """Loads one network at one input size and runs it on BGR frames."""

    def __init__(
        self,
        network: NetworkConfig,
        size: int,
        *,
        num_threads: int = 4,
        use_vulkan: bool = False,
    ) -> None:
        import ncnn

        if size not in network.sizes:
            raise RunnerError(f"Size {size} is not configured for network '{network.id}'")
        param_path = network.param_path(size)
        bin_path = network.bin_path(size)
        for path in (param_path, bin_path):
            if not path.is_file():
                raise RunnerError(f"Model file not found: {path}")

        self.network = network
        self.size = size
        self.mode = network.mode
        self.num_keypoints = network.skeleton.num_keypoints
        self.sigmas = np.asarray(network.skeleton.oks_sigmas, dtype=np.float64)
        self._ncnn = ncnn

        self.vulkan_active = bool(use_vulkan and vulkan_device_count() > 0)
        if use_vulkan and not self.vulkan_active:
            logger.warning("Vulkan requested but no Vulkan device is available; using CPU")

        self.net = ncnn.Net()
        self.net.opt.num_threads = max(1, int(num_threads))
        self.net.opt.use_vulkan_compute = self.vulkan_active
        if self.net.load_param(str(param_path)) != 0:
            raise RunnerError(f"ncnn failed to load param file {param_path}")
        if self.net.load_model(str(bin_path)) != 0:
            raise RunnerError(f"ncnn failed to load bin file {bin_path}")

        self.input_blob = network.input_blob
        self.output_blobs = self._resolve_output_blobs(network.output_blobs)
        self.strides = self._validate_strides(tuple(network.strides))
        logger.info(
            "Loaded %s @%d (%s, strides %s, %s)", network.id, size, self.mode,
            list(self.strides), "Vulkan" if self.vulkan_active else f"CPU x{self.net.opt.num_threads}",
        )

    # ------------------------------------------------------------------ setup
    def _resolve_output_blobs(self, configured: dict[str, str]) -> dict[str, str]:
        try:
            names = set(self.net.output_names())
        except Exception:  # noqa: BLE001 - very old pyncnn without output_names()
            return dict(configured)
        if all(configured.get(k) in names for k in ("box", "cls", "kpt")):
            return dict(configured)
        if all(v in names for v in _FALLBACK_OUTPUT_BLOBS.values()):
            logger.info("Configured output blobs not found; using box/cls/kpt")
            return dict(_FALLBACK_OUTPUT_BLOBS)
        raise RunnerError(
            f"Output blobs {configured} not found in {self.network.id}; network outputs: {sorted(names)}"
        )

    def _validate_strides(self, strides: tuple[int, ...]) -> tuple[int, ...]:
        """Run a dummy pass and check the anchor count matches the strides."""

        dummy = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        box, _, kpt = self._forward(dummy)
        num_anchors = int(box.size // 4)
        if kpt.size != num_anchors * self.num_keypoints * 6:
            raise RunnerError(
                f"Keypoint output has {kpt.size} values, expected {num_anchors} anchors x "
                f"{self.num_keypoints} keypoints x 6; check the skeleton preset"
            )
        if anchor_count(self.size, strides) == num_anchors:
            return strides
        guessed = infer_strides(num_anchors, self.size)
        if guessed is None:
            raise RunnerError(
                f"Network emits {num_anchors} anchors at size {self.size}, which matches neither the "
                f"configured strides {list(strides)} nor any known stride set"
            )
        logger.warning(
            "Configured strides %s do not match %d anchors; using %s", list(strides), num_anchors, list(guessed)
        )
        return guessed

    # -------------------------------------------------------------- inference
    def _forward(self, view_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ncnn = self._ncnn
        mat = ncnn.Mat.from_pixels(view_bgr, ncnn.Mat.PixelType.PIXEL_BGR, view_bgr.shape[1], view_bgr.shape[0])
        mat.substract_mean_normalize([], _NORM)
        with self.net.create_extractor() as extractor:
            extractor.input(self.input_blob, mat)
            outputs = []
            for key in ("box", "cls", "kpt"):
                ret, out = extractor.extract(self.output_blobs[key])
                if ret != 0:
                    raise RunnerError(f"ncnn extract of blob '{self.output_blobs[key]}' failed ({ret})")
                outputs.append(np.array(out, dtype=np.float32, copy=True))
        return outputs[0], outputs[1], outputs[2]

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        conf_threshold: float,
        max_detections: int,
        oks_match_thr: float = 0.5,
        iou_match_thr: float = 0.5,
        dedup: bool = False,
    ) -> DetectionResult:
        timings = Timings()
        t0 = time.perf_counter()
        prepared = prepare_views(frame_bgr, mode=self.mode, network_size=self.size)
        t1 = time.perf_counter()
        timings.preprocess_ms = (t1 - t0) * 1000.0

        by_view: dict[int, list[Pose]] = {}
        raw_candidates = 0
        infer_s = 0.0
        post_s = 0.0
        for key, view in prepared.views.items():
            ti = time.perf_counter()
            box, cls, kpt = self._forward(view)
            tp = time.perf_counter()
            boxes, scores, kpts, vis = decode_raw(
                box, cls, kpt,
                network_size=self.size,
                strides=self.strides,
                num_keypoints=self.num_keypoints,
                conf_threshold=conf_threshold,
                max_detections=max_detections,
            )
            raw_candidates += int(scores.shape[0])
            by_view[key] = to_frame_poses(boxes, scores, kpts, vis, transform=prepared.transforms[key], view=key)
            te = time.perf_counter()
            infer_s += tp - ti
            post_s += te - tp

        tm = time.perf_counter()
        poses = merge_views(
            by_view,
            oks_thr=oks_match_thr,
            iou_thr=iou_match_thr,
            sigmas=self.sigmas,
            max_detections=max_detections,
            dedup_single_view=dedup,
        )
        post_s += time.perf_counter() - tm
        timings.inference_ms = infer_s * 1000.0
        timings.postprocess_ms = post_s * 1000.0
        return DetectionResult(poses=poses, timings=timings, prepared=prepared, raw_candidates=raw_candidates)

    def close(self) -> None:
        net = getattr(self, "net", None)
        if net is not None:
            net.clear()
            self.net = None
