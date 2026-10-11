"""Live window on top of ``xtx4.runtime.InferencePipeline``.

Threads: capture (camera reader -> depth-1 latest-frame queue), inference
(letterbox -> ncnn -> NumPy decode), and this main thread for drawing and
``imshow`` (OpenCV GUI must stay on one thread). The displayed frame is always
the frame the shown poses were computed on, so the overlay never lags the image.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from xtx4.runtime import InferencePipeline, PipelineResult
from xtx4.runtime.preprocess import apply_affine_points, invert_affine

from .config import AppConfig, NetworkConfig
from .draw import DrawOptions, draw_debug_views, draw_panel, draw_poses
from .runner import DetectionResult, NcnnPoseRunner, RunnerError, create_runner

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "screenshots"

HELP_LINES = [
    "q / Esc   quit",
    "space     pause / resume",
    "+ / -     detection threshold",
    "[ / ]     keypoint threshold",
    "1..9      input size (configured sizes)",
    "n         OKS-NMS on/off",
    "c         person scores",
    "l         keypoint labels",
    "i         info panel (HUD)",
    "m         mirror",
    "d         debug: network input view",
    "f         fullscreen",
    "s         screenshot",
    "h         this help",
]


class RateMeter:
    """Events per second over a sliding ~1 s window."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._count = 0
        self.value = 0.0

    def tick(self) -> None:
        self._count += 1
        elapsed = time.perf_counter() - self._start
        if elapsed >= 1.0:
            self.value = self._count / elapsed
            self._count = 0
            self._start = time.perf_counter()


class MovingAverage:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def add(self, x: float) -> None:
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x

    def text(self, fmt: str = "{:.1f}") -> str:
        return fmt.format(self.value) if self.value is not None else "-"


@dataclass(slots=True)
class LiveState:
    conf_threshold: float
    kpt_threshold: float
    mirror: bool
    show_scores: bool
    show_labels: bool
    show_hud: bool
    oks_nms: bool = False
    show_help: bool = False
    show_debug: bool = False
    paused: bool = False
    fullscreen: bool = False
    message: str = ""
    message_until: float = 0.0

    def flash(self, text: str, seconds: float = 2.0) -> None:
        self.message = text
        self.message_until = time.perf_counter() + seconds
        logger.info("%s", text)


class LiveApp:
    def __init__(
        self,
        config: AppConfig,
        network: NetworkConfig,
        size: int,
        reader,
        *,
        conf_threshold: float,
        mirror: bool,
        use_vulkan: bool,
    ) -> None:
        self.config = config
        self.network = network
        self.reader = reader
        self.use_vulkan = use_vulkan
        disp = config.display
        self.state = LiveState(
            conf_threshold=conf_threshold,
            kpt_threshold=config.inference.kpt_threshold,
            mirror=mirror,
            show_scores=disp.show_scores,
            show_labels=disp.show_labels,
            show_hud=disp.show_hud,
            oks_nms=config.inference.oks_nms,
        )
        self.runner: NcnnPoseRunner = self._load_runner(size)
        self.pipeline = InferencePipeline(reader.read, self._detect, frame_transform=self._transform)
        self.window = disp.window_title
        self.display_rate = RateMeter()
        self.infer_rate = RateMeter()
        self._reset_meters()

    # ---------------------------------------------------------------- runner
    def _reset_meters(self) -> None:
        self.pre_ms = MovingAverage()
        self.infer_ms = MovingAverage()
        self.post_ms = MovingAverage()
        self.age_ms = MovingAverage()

    def _load_runner(self, size: int) -> NcnnPoseRunner:
        inf = self.config.inference
        return create_runner(self.network, size, num_threads=inf.num_threads, use_vulkan=self.use_vulkan,
                             fp16=inf.fp16, decode_overrides=inf.decode_overrides())

    def _switch_size(self, size: int) -> None:
        if size == self.runner.size:
            return
        if not self.network.size_available(size):
            self.state.flash(f"Size {size}: model files missing")
            return
        try:
            new_runner = self._load_runner(size)
        except RunnerError as exc:
            self.state.flash(str(exc), 4.0)
            return
        with self.pipeline.detector_lock():  # waits for a running detection
            old, self.runner = self.runner, new_runner
            old.close()
        self._reset_meters()
        self.state.flash(f"Input size -> {size}")

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        return cv2.flip(frame, 1) if self.state.mirror else frame

    def _detect(self, frame: np.ndarray) -> DetectionResult:
        st = self.state
        inf = self.config.inference
        return self.runner.detect(
            frame,
            conf_threshold=st.conf_threshold,
            max_detections=inf.max_detections,
            oks_nms=st.oks_nms,
        )

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        fw, fh = self.reader.frame_size
        cv2.resizeWindow(self.window, min(fw, 1600), min(fh, 900))
        self.pipeline.start()
        seq = 0
        dirty = True
        current: PipelineResult | None = None
        try:
            while True:
                item = self.pipeline.wait_result(after_seq=seq, timeout=0.05)
                if item is not None and item.seq != seq:
                    seq = item.seq
                    current = item
                    self._record(item)
                    dirty = True
                if current is not None and (dirty or time.perf_counter() < self.state.message_until):
                    canvas = current.frame.copy()
                    self._render(canvas, current)
                    cv2.imshow(self.window, canvas)
                    self.display_rate.tick()
                key = cv2.waitKey(1 if not self.state.paused else 30) & 0xFF
                dirty = key != 255
                if not self._handle_key(key, current):
                    break
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if self.pipeline.error is not None:
                    raise RuntimeError("live pipeline failed") from self.pipeline.error
                if not self.pipeline.running:
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            with self.pipeline.detector_lock():
                self.runner.close()
            logger.info("Pipeline statistics:\n%s", self.pipeline.stats.report())

    def _record(self, item: PipelineResult) -> None:
        t = item.result.timings
        self.infer_rate.tick()
        self.pre_ms.add(t.preprocess_ms)
        self.infer_ms.add(t.inference_ms)
        self.post_ms.add(t.postprocess_ms)
        self.age_ms.add(item.frame_age_ms)

    # ---------------------------------------------------------------- render
    def _render(self, canvas: np.ndarray, item: PipelineResult | None) -> None:
        result: DetectionResult | None = item.result if item is not None else None
        if result is not None:
            draw_poses(
                canvas,
                result.poses,
                self.network.skeleton,
                DrawOptions(
                    kpt_threshold=self.state.kpt_threshold,
                    show_scores=self.state.show_scores,
                    show_labels=self.state.show_labels,
                    point_radius=self.config.display.point_radius,
                    line_thickness=self.config.display.line_thickness,
                ),
            )
            if self.state.show_debug and result.prepared is not None:
                net_points: dict[int, list[np.ndarray]] = {}
                for pose in result.poses:
                    inv = invert_affine(result.prepared.transforms[pose.source_view])
                    shown = pose.keypoints[:, 2] >= self.state.kpt_threshold
                    net_points.setdefault(pose.source_view, []).append(
                        apply_affine_points(inv, pose.keypoints[shown, :2].astype(np.float64))
                    )
                draw_debug_views(canvas, result.prepared.views, poses_net=net_points)

        if self.state.show_hud:
            draw_panel(canvas, self._hud_lines(result), corner="tl")
        if self.state.show_help:
            draw_panel(canvas, HELP_LINES, corner="tr")
        if self.state.message and time.perf_counter() < self.state.message_until:
            draw_panel(canvas, [self.state.message], corner="bl")

    def _hud_lines(self, result: DetectionResult | None) -> list[str]:
        r = self.runner
        fw, fh = self.reader.frame_size
        stats = self.pipeline.stats
        people = len(result.poses) if result is not None else 0
        candidates = result.raw_candidates if result is not None else 0
        lines = [
            f"{self.network.id}  @{r.size}  {'Vulkan' if r.vulkan_active else f'CPU x{r.num_threads}'}"
            f"  fp16={'on' if r.fp16 else 'off'}",
            f"source: {self.reader.source.describe()}  {fw}x{fh}",
            f"camera {self.reader.fps:4.1f} fps | infer {self.infer_rate.value:4.1f} fps | "
            f"display {self.display_rate.value:4.1f} fps | dropped {stats.dropped}",
            f"pre {self.pre_ms.text()} | net {self.infer_ms.text()} | post {self.post_ms.text()} ms"
            f" | frame age {self.age_ms.text()} ms",
            f"people {people} (cand {candidates}) | conf >= {self.state.conf_threshold:.2f} | "
            f"kpt >= {self.state.kpt_threshold:.2f} | OKS-NMS {'on' if self.state.oks_nms else 'off'}",
        ]
        if self.state.paused:
            lines.append("PAUSED")
        if getattr(self.reader, "status", "ok") != "ok":
            lines.append(f"camera: {self.reader.status}")
        lines.append("h = help")
        return lines

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: int, item: PipelineResult | None) -> bool:
        if key == 255:
            return True
        st = self.state
        ch = chr(key) if key < 128 else ""
        if key == 27 or ch == "q":
            return False
        if ch == " ":
            st.paused = not st.paused
            self.pipeline.set_paused(st.paused)
        elif ch in "+=":
            st.conf_threshold = min(0.95, round(st.conf_threshold + 0.05, 2))
            st.flash(f"Detection threshold {st.conf_threshold:.2f}")
        elif ch in "-_":
            st.conf_threshold = max(0.05, round(st.conf_threshold - 0.05, 2))
            st.flash(f"Detection threshold {st.conf_threshold:.2f}")
        elif ch == "]":
            st.kpt_threshold = min(0.95, round(st.kpt_threshold + 0.05, 2))
            st.flash(f"Keypoint threshold {st.kpt_threshold:.2f}")
        elif ch == "[":
            st.kpt_threshold = max(0.05, round(st.kpt_threshold - 0.05, 2))
            st.flash(f"Keypoint threshold {st.kpt_threshold:.2f}")
        elif ch.isdigit() and ch != "0":
            idx = int(ch) - 1
            if idx < len(self.network.sizes):
                self._switch_size(self.network.sizes[idx])
        elif ch == "c":
            st.show_scores = not st.show_scores
        elif ch == "l":
            st.show_labels = not st.show_labels
        elif ch == "i":
            st.show_hud = not st.show_hud
        elif ch == "m":
            st.mirror = not st.mirror
            st.flash(f"Mirror {'on' if st.mirror else 'off'}")
        elif ch == "d":
            st.show_debug = not st.show_debug
        elif ch == "n":
            st.oks_nms = not st.oks_nms
            st.flash(f"OKS-NMS {'on' if st.oks_nms else 'off (raw one-to-one output)'}")
        elif ch == "h":
            st.show_help = not st.show_help
        elif ch == "f":
            st.fullscreen = not st.fullscreen
            cv2.setWindowProperty(
                self.window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if st.fullscreen else cv2.WINDOW_NORMAL,
            )
        elif ch == "s" and item is not None:
            self._screenshot(item)
        return True

    def _screenshot(self, item: PipelineResult) -> None:
        canvas = item.frame.copy()
        self._render(canvas, item)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = SCREENSHOT_DIR / f"{self.network.id}_{self.runner.size}_{stamp}.png"
        cv2.imwrite(str(path), canvas)
        self.state.flash(f"Saved {path.name}")
