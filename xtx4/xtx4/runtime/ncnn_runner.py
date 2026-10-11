"""ncnn pose runner: BGR frame -> letterbox -> ncnn ``extract`` -> NumPy decode -> poses.

Torch-free (``ncnn`` + ``numpy`` + ``cv2``). Defaults follow the Raspberry Pi 5
deployment of ``implementation_plan_fable.md`` section 10.2: CPU, 4 threads,
FP16 storage/arithmetic, packing layout, ``lightmode`` on, Vulkan off.

Construction validates the graph against the decode contract:

1. the configured input/output blob names exist in the ``.param`` file (if the
   configured output names are missing but the graph has exactly two outputs,
   they are used in graph order -- person first -- with a warning);
2. a dummy pass checks the anchor count ``N`` and ``pose = N x K x 5``;
   strides are inferred from ``N`` when not given (or when the given ones do
   not match, with a warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS
from ..utils.logging import get_logger
from .decode import (
    DEFAULT_NMS_SCALE_FACTOR,
    KPT_CHANNELS_PER_POINT,
    PRESENT_THRESHOLD,
    SIGMA_FLOOR_UNITS,
    VISIBLE_THRESHOLD,
    Pose,
    anchor_count,
    decode_selected,
    infer_strides,
    merge_views,
    person_scores,
    select_candidates,
    to_frame_poses,
)
from .preprocess import PreparedViews, prepare_views

logger = get_logger(__name__)

_NORM = [1 / 255.0, 1 / 255.0, 1 / 255.0]
FROM_MANIFEST = "manifest"  # sentinel: take the value from the manifest ``decode`` block (else the default)
_gpu_instance_used = False


class RunnerError(RuntimeError):
    """Invalid graph / files / blob names."""


def parse_param_blobs(param_path: Path | str) -> tuple[list[str], list[str]]:
    """``(input_blobs, output_blobs)`` of an ncnn ``.param`` (outputs = produced, never consumed)."""

    lines = Path(param_path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "7767517":
        raise RunnerError(f"{param_path} is not an ncnn param file (magic 7767517 missing)")
    produced: list[str] = []
    consumed: set[str] = set()
    inputs: list[str] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        n_in, n_out = int(parts[2]), int(parts[3])
        blobs_in = parts[4 : 4 + n_in]
        blobs_out = parts[4 + n_in : 4 + n_in + n_out]
        if parts[0] == "Input":
            inputs.extend(blobs_out)
        consumed.update(blobs_in)
        produced.extend(blobs_out)
    return inputs, [b for b in produced if b not in consumed and b not in inputs]


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
    """Release the ncnn Vulkan instance after every Net is cleared (clean exit on Windows)."""

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
    """``poses`` in frame pixels; ``raw_candidates`` = anchors with ``score >= conf_threshold``
    (before top-k / NMS), summed over views."""

    poses: list[Pose]
    timings: Timings
    raw_candidates: int = 0
    prepared: PreparedViews | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class NcnnPoseRunner:
    """One exported XTX4 graph at one input size."""

    def __init__(
        self,
        param: Path | str,
        bin: Path | str,  # noqa: A002 - mirrors the ncnn file pair naming
        *,
        size: int,
        strides: Sequence[int] | None = None,
        manifest: Mapping[str, Any] | Path | str | None = None,
        num_threads: int = 4,
        use_vulkan: bool = False,
        fp16: bool = True,
        input_blob: str = "in0",
        output_blobs: Sequence[str] = ("out0", "out1"),
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
        import ncnn

        self._ncnn = ncnn
        self.param_path = Path(param)
        self.bin_path = Path(bin)
        for path in (self.param_path, self.bin_path):
            if not path.is_file():
                raise RunnerError(f"Model file not found: {path}")
        self.size = int(size)
        if self.size % 32:
            raise RunnerError(f"Input size must be a multiple of 32, got {self.size}")
        self.mode = mode

        if isinstance(manifest, (str, Path)):
            from ..export.manifest import load_manifest

            manifest = load_manifest(manifest)
        self.manifest: dict[str, Any] | None = dict(manifest) if manifest is not None else None
        decode_cfg: dict[str, Any] = dict(self.manifest.get("decode", {})) if self.manifest else {}
        if self.manifest is not None:
            if strides is None:
                strides = self.manifest.get("strides")
            if num_keypoints is None:
                num_keypoints = int(self.manifest.get("num_keypoints", NUM_KEYPOINTS))
            ncnn_out = self.manifest.get("outputs", {}).get("ncnn", {})
            blobs = ncnn_out.get("blobs")
            if blobs:
                output_blobs = (blobs["person"], blobs["pose"])
            input_blob = ncnn_out.get("input", input_blob)
            if sigmas is None:
                sigmas = self.manifest.get("keypoints", {}).get("sigmas")
        self.num_keypoints = int(num_keypoints or NUM_KEYPOINTS)
        self.sigma_conf_coeff = float(
            sigma_conf_coeff if sigma_conf_coeff is not None else decode_cfg.get("sigma_conf_coeff", 0.0)
        )
        self.sigma_floor = float(sigma_floor if sigma_floor is not None else decode_cfg.get("sigma_floor", SIGMA_FLOOR_UNITS))
        self.sigmas = np.asarray(sigmas if sigmas is not None else OKS_SIGMAS, dtype=np.float64)
        if self.sigmas.shape[0] != self.num_keypoints:
            self.sigmas = np.full(self.num_keypoints, 0.05)
        self.default_conf_threshold = float(decode_cfg.get("conf_threshold", 0.25))
        self.default_max_detections = int(decode_cfg.get("max_detections", 100))

        def pick(value: Any, key: str, default: Any) -> Any:
            return decode_cfg.get(key, default) if value is None else value

        self.present_threshold = float(pick(present_threshold, "present_threshold", PRESENT_THRESHOLD))
        self.visible_threshold = float(pick(visible_threshold, "visible_threshold", VISIBLE_THRESHOLD))
        self.oks_nms_threshold = float(pick(oks_nms_threshold, "oks_nms_threshold", 0.5))
        self.oks_nms_scale_factor = float(pick(oks_nms_scale_factor, "oks_nms_scale_factor", DEFAULT_NMS_SCALE_FACTOR))
        iou = decode_cfg.get("oks_nms_iou_threshold") if oks_nms_iou_threshold == FROM_MANIFEST else oks_nms_iou_threshold
        self.oks_nms_iou_threshold: float | None = None if iou is None else float(iou)

        self.vulkan_active = bool(use_vulkan and vulkan_device_count() > 0)
        if use_vulkan and not self.vulkan_active:
            logger.warning("Vulkan requested but no Vulkan device is available; using CPU")
        self.fp16 = bool(fp16)
        self.num_threads = max(1, int(num_threads))

        self.net = ncnn.Net()
        opt = self.net.opt
        opt.num_threads = self.num_threads
        opt.use_vulkan_compute = self.vulkan_active
        opt.use_fp16_packed = self.fp16
        opt.use_fp16_storage = self.fp16
        opt.use_fp16_arithmetic = self.fp16
        opt.use_packing_layout = True
        opt.lightmode = True
        if self.net.load_param(str(self.param_path)) != 0:
            raise RunnerError(f"ncnn failed to load param file {self.param_path}")
        if self.net.load_model(str(self.bin_path)) != 0:
            raise RunnerError(f"ncnn failed to load bin file {self.bin_path}")

        self.input_blob, self.output_blobs = self._resolve_blobs(input_blob, tuple(output_blobs))
        self.strides = self._validate_shapes(None if strides is None else tuple(int(s) for s in strides))
        self.num_anchors = anchor_count(self.size, self.strides)
        logger.info(
            "Loaded %s @%d (strides %s, N=%d, %s, fp16=%s)", self.param_path.name, self.size, list(self.strides),
            self.num_anchors, "Vulkan" if self.vulkan_active else f"CPU x{self.num_threads}", self.fp16,
        )

    # ------------------------------------------------------------ factories
    @classmethod
    def from_manifest(cls, manifest_path: Path | str, size: int, **kwargs: Any) -> "NcnnPoseRunner":
        """Build a runner from ``<export>/manifest.json`` (graph files resolved next to it)."""

        from ..export.manifest import load_manifest, ncnn_files

        path = Path(manifest_path)
        if path.is_dir():
            path = path / "manifest.json"
        manifest = load_manifest(path)
        param, bin_path = ncnn_files(manifest, size, path.parent)
        kwargs.setdefault("fp16", bool(manifest.get("outputs", {}).get("ncnn", {}).get("fp16_weights", True)))
        return cls(param, bin_path, size=size, manifest=manifest, **kwargs)

    # ---------------------------------------------------------------- setup
    def _resolve_blobs(self, input_blob: str, output_blobs: tuple[str, ...]) -> tuple[str, tuple[str, str]]:
        inputs, outputs = parse_param_blobs(self.param_path)
        if input_blob not in inputs:
            if len(inputs) == 1:
                logger.warning("Input blob %r not in %s; using %r", input_blob, self.param_path.name, inputs[0])
                input_blob = inputs[0]
            else:
                raise RunnerError(f"Input blob {input_blob!r} not found; graph inputs: {inputs}")
        if len(output_blobs) != 2:
            raise RunnerError(f"Expected two output blobs (person, pose), got {output_blobs}")
        if all(b in outputs for b in output_blobs):
            return input_blob, (output_blobs[0], output_blobs[1])
        if len(outputs) == 2:
            logger.warning(
                "Output blobs %s not in %s; using graph outputs %s (person, pose)",
                list(output_blobs), self.param_path.name, outputs,
            )
            return input_blob, (outputs[0], outputs[1])
        raise RunnerError(f"Output blobs {list(output_blobs)} not found; graph outputs: {outputs}")

    def _validate_shapes(self, strides: tuple[int, ...] | None) -> tuple[int, ...]:
        person, pose = self.forward(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        num_anchors = int(person.size)
        k5 = self.num_keypoints * KPT_CHANNELS_PER_POINT
        if pose.size != num_anchors * k5:
            raise RunnerError(
                f"Pose output has {pose.size} values, expected N x K x 5 = {num_anchors} x {self.num_keypoints} x 5 "
                f"(are the output blobs swapped or K wrong?)"
            )
        if strides is not None and anchor_count(self.size, strides) == num_anchors:
            return strides
        guessed = infer_strides(num_anchors, self.size)
        if guessed is None:
            raise RunnerError(
                f"Graph emits N = {num_anchors} anchors at size {self.size}, which matches neither the "
                f"configured strides {list(strides) if strides else None} nor a known stride set"
            )
        if strides is None:
            logger.warning("Strides not given; inferred %s from N = %d", list(guessed), num_anchors)
        else:
            logger.warning("Strides %s do not match N = %d; using %s", list(strides), num_anchors, list(guessed))
        return guessed

    # ------------------------------------------------------------ inference
    def _run(self, mat: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.net is None:
            raise RunnerError("runner is closed")
        with self.net.create_extractor() as extractor:
            extractor.input(self.input_blob, mat)
            outputs = []
            for name in self.output_blobs:
                ret, out = extractor.extract(name)
                if ret != 0:
                    raise RunnerError(f"ncnn extract of blob {name!r} failed ({ret})")
                outputs.append(np.array(out, dtype=np.float32, copy=True))
        return outputs[0], outputs[1]

    def forward(self, view_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw outputs ``(person (N, 1), pose (N, K*5))`` for one letterboxed uint8 BGR view."""

        ncnn = self._ncnn
        view = np.ascontiguousarray(view_bgr, dtype=np.uint8)
        mat = ncnn.Mat.from_pixels(view, ncnn.Mat.PixelType.PIXEL_BGR, view.shape[1], view.shape[0])
        mat.substract_mean_normalize([], _NORM)
        return self._run(mat)

    def forward_blob(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw outputs for an already normalised ``(1, 3, S, S)`` / ``(3, S, S)`` float32 tensor."""

        arr = np.asarray(blob, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        return self._run(self._ncnn.Mat(np.ascontiguousarray(arr)))

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float | None = None,
        max_detections: int | None = None,
        oks_nms: bool = False,
        *,
        oks_nms_threshold: float | None = None,
        nms_kpt_threshold: float = 0.5,
        keep_views: bool = True,
        oks_nms_iou_threshold: float | None | str = FROM_MANIFEST,
        oks_nms_scale_factor: float | None = None,
    ) -> DetectionResult:
        """Run the full frame -> poses path; poses are in frame pixels, sorted by score.

        Decode/NMS parameters left at ``None`` (or ``FROM_MANIFEST``) use the runner's
        values (constructor argument, else manifest ``decode`` block, else defaults).
        """

        conf = self.default_conf_threshold if conf_threshold is None else float(conf_threshold)
        max_det = self.default_max_detections if max_detections is None else int(max_detections)
        timings = Timings()
        t0 = time.perf_counter()
        prepared = prepare_views(frame_bgr, network_size=self.size, mode=self.mode)
        timings.preprocess_ms = (time.perf_counter() - t0) * 1000.0

        by_view: dict[int, list[Pose]] = {}
        raw_candidates = 0
        infer_s = post_s = 0.0
        for key, view in prepared.views.items():
            ti = time.perf_counter()
            person, pose = self.forward(view)
            tp = time.perf_counter()
            scores = person_scores(person)
            keep, num_above = select_candidates(scores, conf_threshold=conf, max_detections=max_det)
            keypoints, visibility, sigma_px = decode_selected(
                pose, keep, network_size=self.size, strides=self.strides, num_keypoints=self.num_keypoints,
                sigma_conf_coeff=self.sigma_conf_coeff, sigma_floor=self.sigma_floor,
                present_threshold=self.present_threshold, visible_threshold=self.visible_threshold,
            )
            raw_candidates += num_above
            by_view[key] = to_frame_poses(
                scores[keep], keypoints, visibility, sigma_px, transform=prepared.transforms[key], view=key,
                anchor_indices=keep,
            )
            infer_s += tp - ti
            post_s += time.perf_counter() - tp
        tm = time.perf_counter()
        poses = merge_views(
            by_view,
            threshold=self.oks_nms_threshold if oks_nms_threshold is None else float(oks_nms_threshold),
            sigmas=self.sigmas, max_detections=max_det, nms=oks_nms, kpt_threshold=nms_kpt_threshold,
            scale_factor=self.oks_nms_scale_factor if oks_nms_scale_factor is None else float(oks_nms_scale_factor),
            iou_threshold=(
                self.oks_nms_iou_threshold if oks_nms_iou_threshold == FROM_MANIFEST else oks_nms_iou_threshold
            ),
        )
        post_s += time.perf_counter() - tm
        timings.inference_ms = infer_s * 1000.0
        timings.postprocess_ms = post_s * 1000.0
        return DetectionResult(
            poses=poses, timings=timings, raw_candidates=raw_candidates,
            prepared=prepared if keep_views else None,
        )

    # ------------------------------------------------------------- lifetime
    def close(self) -> None:
        net = getattr(self, "net", None)
        if net is not None:
            net.clear()
            self.net = None

    def __enter__(self) -> "NcnnPoseRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def describe(self) -> str:
        backend = "Vulkan" if self.vulkan_active else f"CPU x{self.num_threads}"
        return f"{self.param_path.name} @{self.size} {self.mode} {backend} fp16={'on' if self.fp16 else 'off'}"
