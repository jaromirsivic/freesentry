"""Live window: grab frame -> detect poses -> draw -> show, with hotkeys."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import FrameGrabber
from .config import AppConfig, NetworkConfig
from .draw import DrawOptions, draw_debug_views, draw_panel, draw_poses
from .runner import DetectionResult, NcnnPoseRunner, RunnerError

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "screenshots"

HELP_LINES = [
    "q / Esc   quit",
    "space     pause / resume",
    "+ / -     detection threshold",
    "[ / ]     keypoint threshold",
    "1..9      input size (configured sizes)",
    "b         bounding boxes",
    "l         keypoint labels",
    "i         info panel (HUD)",
    "m         mirror",
    "d         debug: network input view(s)",
    "n         duplicate suppression on/off",
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


@dataclass(slots=True)
class LiveState:
    conf_threshold: float
    kpt_threshold: float
    mirror: bool
    show_bbox: bool
    show_labels: bool
    show_hud: bool
    dedup: bool = False
    show_help: bool = False
    show_debug: bool = False
    paused: bool = False
    fullscreen: bool = False
    message: str = ""
    message_until: float = 0.0
    extra: dict = field(default_factory=dict)

    def flash(self, text: str, seconds: float = 2.0) -> None:
        self.message = text
        self.message_until = time.perf_counter() + seconds
        logger.info("%s", text)


class AsyncDetector:
    """Runs detection in a worker thread; the display shows the latest result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.busy = threading.Lock()  # held while a runner is executing
        self._pending: tuple[np.ndarray, dict] | None = None
        self._result: DetectionResult | None = None
        self._event = threading.Event()
        self._stop = False
        self.runner: NcnnPoseRunner | None = None
        self._thread = threading.Thread(target=self._run, name="AsyncDetector", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, kwargs: dict) -> None:
        with self._lock:
            self._pending = (frame, kwargs)
        self._event.set()

    def result(self) -> DetectionResult | None:
        with self._lock:
            return self._result

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(0.2)
            self._event.clear()
            with self._lock:
                job, self._pending = self._pending, None
                runner = self.runner
            if job is None or runner is None:
                continue
            try:
                with self.busy:
                    if runner.net is None:  # closed by a size switch meanwhile
                        continue
                    res = runner.detect(job[0], **job[1])
            except Exception:  # noqa: BLE001 - keep the worker alive
                logger.exception("Async detection failed")
                continue
            with self._lock:
                self._result = res

    def set_runner(self, runner: NcnnPoseRunner) -> None:
        with self._lock:
            self.runner = runner
            self._result = None

    def close(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=2.0)


class LiveApp:
    def __init__(
        self,
        config: AppConfig,
        network: NetworkConfig,
        size: int,
        grabber: FrameGrabber,
        *,
        conf_threshold: float,
        mirror: bool,
        use_vulkan: bool,
    ) -> None:
        self.config = config
        self.network = network
        self.grabber = grabber
        self.use_vulkan = use_vulkan
        disp = config.display
        self.state = LiveState(
            conf_threshold=conf_threshold,
            kpt_threshold=config.inference.kpt_threshold,
            mirror=mirror,
            show_bbox=disp.show_bbox,
            show_labels=disp.show_labels,
            show_hud=disp.show_hud,
            dedup=config.inference.dedup,
        )
        self.runner = self._load_runner(size)
        self.async_detector = AsyncDetector() if config.inference.async_inference else None
        if self.async_detector:
            self.async_detector.set_runner(self.runner)
        self.window = disp.window_title
        self.display_rate = RateMeter()
        self.infer_rate = RateMeter()
        self.infer_ms = MovingAverage()
        self.total_ms = MovingAverage()

    # ---------------------------------------------------------------- runner
    def _load_runner(self, size: int) -> NcnnPoseRunner:
        return NcnnPoseRunner(
            self.network,
            size,
            num_threads=self.config.inference.num_threads,
            use_vulkan=self.use_vulkan,
        )

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
        old = self.runner
        self.runner = new_runner
        if self.async_detector:
            self.async_detector.set_runner(new_runner)
            with self.async_detector.busy:
                old.close()
        else:
            old.close()
        self.infer_ms = MovingAverage()
        self.total_ms = MovingAverage()
        self.state.flash(f"Input size -> {size}")

    def _detect_kwargs(self) -> dict:
        inf = self.config.inference
        return dict(
            conf_threshold=self.state.conf_threshold,
            max_detections=inf.max_detections,
            oks_match_thr=inf.oks_match_thr,
            iou_match_thr=inf.iou_match_thr,
            dedup=self.state.dedup,
        )

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        fw, fh = self.grabber.frame_size
        cv2.resizeWindow(self.window, min(fw, 1600), min(fh, 900))

        seq = 0
        frame: np.ndarray | None = None
        result: DetectionResult | None = None
        try:
            while True:
                if not self.state.paused or frame is None:
                    seq, raw = self.grabber.latest(after_seq=seq, timeout=0.5)
                    if raw is not None:
                        frame = cv2.flip(raw, 1) if self.state.mirror else raw.copy()
                        result = self._process(frame) or result

                if frame is not None:
                    canvas = frame.copy()
                    self._render(canvas, result)
                    cv2.imshow(self.window, canvas)
                    self.display_rate.tick()

                key = cv2.waitKey(1 if not self.state.paused else 30) & 0xFF
                if not self._handle_key(key, frame, result):
                    break
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            if self.async_detector:
                self.async_detector.close()
                with self.async_detector.busy:
                    self.runner.close()
            else:
                self.runner.close()

    def _process(self, frame: np.ndarray) -> DetectionResult | None:
        if self.async_detector:
            self.async_detector.submit(frame, self._detect_kwargs())
            res = self.async_detector.result()
            if res is not None and res is not self.state.extra.get("last_async"):
                self.state.extra["last_async"] = res
                self._record(res)
            return res
        res = self.runner.detect(frame, **self._detect_kwargs())
        self._record(res)
        return res

    def _record(self, res: DetectionResult) -> None:
        self.infer_rate.tick()
        self.infer_ms.add(res.timings.inference_ms)
        self.total_ms.add(res.timings.total_ms)

    # ---------------------------------------------------------------- render
    def _render(self, canvas: np.ndarray, result: DetectionResult | None) -> None:
        if result is not None:
            draw_poses(
                canvas,
                result.poses,
                self.network.skeleton,
                DrawOptions(
                    kpt_threshold=self.state.kpt_threshold,
                    show_bbox=self.state.show_bbox,
                    show_labels=self.state.show_labels,
                    point_radius=self.config.display.point_radius,
                    line_thickness=self.config.display.line_thickness,
                ),
            )
            if self.state.show_debug:
                draw_debug_views(canvas, result.prepared.views)

        if self.state.show_hud:
            draw_panel(canvas, self._hud_lines(result), corner="tl")
        if self.state.show_help:
            draw_panel(canvas, HELP_LINES, corner="tr")
        if self.state.message and time.perf_counter() < self.state.message_until:
            draw_panel(canvas, [self.state.message], corner="bl")

    def _hud_lines(self, result: DetectionResult | None) -> list[str]:
        r = self.runner
        fw, fh = self.grabber.frame_size
        backend = "Vulkan" if r.vulkan_active else f"CPU x{self.config.inference.num_threads}"
        infer_ms = f"{self.infer_ms.value:.1f}" if self.infer_ms.value is not None else "-"
        total_ms = f"{self.total_ms.value:.1f}" if self.total_ms.value is not None else "-"
        people = len(result.poses) if result is not None else 0
        lines = [
            f"{self.network.id}  @{r.size}  mode={r.mode}  {backend}",
            f"source: {self.grabber.source.describe()}  {fw}x{fh}",
            f"camera {self.grabber.fps:4.1f} fps | infer {self.infer_rate.value:4.1f} fps | "
            f"display {self.display_rate.value:4.1f} fps",
            f"network {infer_ms} ms | pipeline {total_ms} ms",
            f"people {people} | conf >= {self.state.conf_threshold:.2f} | kpt >= {self.state.kpt_threshold:.2f}"
            + (" | dedup" if self.state.dedup else ""),
        ]
        if self.state.paused:
            lines.append("PAUSED")
        if self.grabber.status != "ok":
            lines.append(f"camera: {self.grabber.status}")
        lines.append("h = help")
        return lines

    # ------------------------------------------------------------------ keys
    def _handle_key(self, key: int, frame: np.ndarray | None, result: DetectionResult | None) -> bool:
        if key == 255:
            return True
        st = self.state
        ch = chr(key) if key < 128 else ""
        if key == 27 or ch == "q":
            return False
        if ch == " ":
            st.paused = not st.paused
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
        elif ch == "b":
            st.show_bbox = not st.show_bbox
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
            st.dedup = not st.dedup
            st.flash(f"Duplicate suppression {'on' if st.dedup else 'off (raw network output)'}")
        elif ch == "h":
            st.show_help = not st.show_help
        elif ch == "f":
            st.fullscreen = not st.fullscreen
            cv2.setWindowProperty(
                self.window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if st.fullscreen else cv2.WINDOW_NORMAL,
            )
        elif ch == "s" and frame is not None:
            self._screenshot(frame, result)
        return True

    def _screenshot(self, frame: np.ndarray, result: DetectionResult | None) -> None:
        canvas = frame.copy()
        self._render(canvas, result)
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = SCREENSHOT_DIR / f"{self.network.id}_{self.runner.size}_{stamp}.png"
        cv2.imwrite(str(path), canvas)
        self.state.flash(f"Saved {path.name}")
