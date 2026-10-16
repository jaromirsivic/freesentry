"""Source + graph + threaded pipeline behind the test app (no GUI code here).

:meth:`LiveEngine.apply` takes a complete :class:`~livetest.settings.Settings` and
does only what the change needs:

* ``RUNNER_KEYS`` (network, size, backend, threads, FP16, Vulkan) -> load a new
  graph and swap it in while no detection runs; on failure the old one stays;
* ``DECODE_ATTRS`` (view mode, visibility thresholds, sigma decode) -> set on the
  running graph;
* ``SOURCE_KEYS`` -> close the source and the pipeline, open the new source;
* anything else (detection threshold, NMS, mirror, display) is read per frame.

Pausing blocks the capture side (a paused video file does not run on), ``step``
lets exactly one frame through, and :meth:`redetect` re-runs the current graph on
a frozen frame so parameter changes are visible while paused.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from xtx4.runtime import InferencePipeline, PipelineResult

from .backends import DetectionResult, PoseRunnerBase, apply_decode_attrs, create_runner
from .camera import CameraSource, FrameReader, StillImageReader, default_backend, source_from_uri
from .config import AppConfig, NetworkConfig
from .settings import DECODE_ATTRS, RUNNER_KEYS, SOURCE_KEYS, Settings

logger = logging.getLogger(__name__)


def source_for(settings: Settings) -> CameraSource:
    if settings.source_kind == "camera":
        name = settings.camera_name or f"Camera {settings.camera_index}"
        return CameraSource(name=name, index=settings.camera_index, backend=default_backend(settings.windows_backend))
    if not settings.source_path.strip():
        raise RuntimeError(f"No {'image' if settings.source_kind == 'image' else 'video file / URL'} selected")
    return source_from_uri(settings.source_path)


def open_reader(settings: Settings):
    """``FrameReader`` for a camera / video file / stream, ``StillImageReader`` for an image."""

    if settings.source_kind == "image":
        path = settings.source_path.strip().strip('"')
        if not path:
            raise RuntimeError("No image selected")
        return StillImageReader(path)
    return FrameReader(source_for(settings), width=settings.width, height=settings.height, fps=settings.camera_fps,
                       mjpg=settings.mjpg)


class LiveEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings: Settings | None = None
        self.network: NetworkConfig | None = None
        self.runner: PoseRunnerBase | None = None
        self.reader = None
        self.pipeline: InferencePipeline | None = None
        self.source_error: str | None = None
        self._runner_lock = threading.Lock()  # guards runner swaps when no pipeline is running
        self._pause = threading.Condition()
        self._paused = False
        self._step_budget = 0
        self._closing = False

    # ------------------------------------------------------------------ apply
    def apply(self, new: Settings, *, reopen_source: bool | None = None) -> list[str]:
        """Apply ``new``; returns short messages. Raises ``RunnerError`` when the graph
        cannot be loaded (nothing changed then); a source failure only sets
        :attr:`source_error` (the app keeps running without a source)."""

        old = self.settings
        messages: list[str] = []
        if self.runner is None or new.differs(old, RUNNER_KEYS):
            network = self.config.network(new.network_id)
            runner = create_runner(network, new)  # may raise: keep the old graph
            with self.detector_lock():
                previous, self.runner, self.network = self.runner, runner, network
                if previous is not None:
                    previous.close()
            messages.append(f"Loaded {runner.describe()}")
            self.reset_stats()
        elif new.differs(old, DECODE_ATTRS):
            with self.detector_lock():
                apply_decode_attrs(self.runner, new)
            if old is not None and new.mode != old.mode:
                messages.append(f"View mode -> {new.mode}")
                self.reset_stats()
        self.settings = new
        if reopen_source or (reopen_source is None and (self.reader is None or new.differs(old, SOURCE_KEYS))):
            self.open_source()
            messages.append(f"Source: {self.source_error}" if self.source_error else f"Source: {self.describe_source()}")
        return messages

    def detector_lock(self) -> threading.Lock:
        return self.pipeline.detector_lock() if self.pipeline is not None else self._runner_lock

    # ----------------------------------------------------------------- source
    def open_source(self) -> None:
        self.close_source()
        settings = self.settings
        assert settings is not None
        self.source_error = None
        try:
            self.reader = open_reader(settings)
        except (RuntimeError, OSError) as exc:
            self.source_error = str(exc)
            logger.warning("%s", exc)
            return
        # A still image is served as fast as asked: pace it like a camera.
        pace = float(settings.camera_fps) if settings.source_kind == "image" else 0.0
        self.pipeline = InferencePipeline(self._read, self._detect, frame_transform=self._transform,
                                          max_capture_fps=pace)
        self.pipeline.start()

    def close_source(self) -> None:
        pipeline, reader = self.pipeline, self.reader
        if reader is not None:
            reader.close()  # unblocks a pending camera read
        with self._pause:
            self._closing = True
            self._pause.notify_all()
        if pipeline is not None:
            pipeline.stop()
        with self._pause:
            self._closing = False
        self.pipeline = self.reader = None

    def describe_source(self) -> str:
        if self.reader is None:
            return "none"
        w, h = self.reader.frame_size
        return f"{self.reader.source.describe()} {w}x{h}"

    # ------------------------------------------------------------- per frame
    def _read(self) -> np.ndarray | None:
        with self._pause:
            while self._paused and self._step_budget <= 0 and not self._closing:
                self._pause.wait(0.1)
            if self._closing:
                return None
            if self._paused:
                self._step_budget -= 1
        return self.reader.read() if self.reader is not None else None

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        settings = self.settings
        return cv2.flip(frame, 1) if settings is not None and settings.mirror else frame

    def _detect(self, frame: np.ndarray) -> DetectionResult:
        """Called by the inference thread with the detector lock held."""

        return self.runner.detect(frame, **self.settings.detect_kwargs())

    def redetect(self, frame: np.ndarray) -> DetectionResult:
        """Run the current graph and settings on ``frame`` (GUI thread, e.g. while paused)."""

        with self.detector_lock():
            return self._detect(frame)

    # ---------------------------------------------------------------- control
    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        with self._pause:
            self._paused = paused
            self._step_budget = 0
            self._pause.notify_all()

    def step(self) -> None:
        """While paused: let the next frame through the pipeline."""

        with self._pause:
            self._step_budget += 1
            self._pause.notify_all()

    def latest(self) -> PipelineResult | None:
        return self.pipeline.latest() if self.pipeline is not None else None

    @property
    def stats(self):
        return self.pipeline.stats if self.pipeline is not None else None

    def reset_stats(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stats.reset()

    @property
    def error(self) -> BaseException | None:
        return self.pipeline.error if self.pipeline is not None else None

    def close(self) -> None:
        self.close_source()
        with self._runner_lock:
            if self.runner is not None:
                self.runner.close()
                self.runner = None

