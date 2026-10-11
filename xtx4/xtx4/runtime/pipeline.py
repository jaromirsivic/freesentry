"""Threaded capture -> inference (-> display) pipeline with latest-frame semantics.

Why threads: on the Pi the camera read + MJPEG decode (5-8 ms), the network
(~15-25 ms) and drawing/``imshow`` (a few ms) would add up if run serially.
Running capture and inference in their own threads makes the throughput equal
to the slowest stage instead of the sum (``implementation_plan_fable.md`` 4).

Latency rule: the hand-over between capture and inference is a **depth-1
latest-frame queue** -- a new frame overwrites a frame inference has not taken
yet (counted as *dropped*). Inference therefore always works on the freshest
frame and the end-to-end latency stays bounded; the *frame age* (capture
timestamp -> result ready) is recorded per processed frame.

Display is left to the caller's main thread (OpenCV GUI calls must stay on
one thread): it polls :meth:`InferencePipeline.wait_result`. Headless benchmark
mode simply waits for N processed frames and prints :class:`PipelineStats`.
Torch-free; depends only on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

ReadFn = Callable[[], "np.ndarray | None"]
DetectFn = Callable[[np.ndarray], Any]  # returns an object with ``.timings`` (DetectionResult)


@dataclass(slots=True)
class FramePacket:
    seq: int
    frame: np.ndarray
    t_capture: float  # time.perf_counter() when the frame was read


@dataclass(slots=True)
class PipelineResult:
    seq: int
    frame: np.ndarray
    result: Any  # DetectionResult
    t_capture: float
    t_done: float

    @property
    def frame_age_ms(self) -> float:
        return (self.t_done - self.t_capture) * 1000.0


class LatestFrameQueue:
    """Depth-1 queue: ``put`` replaces a frame nobody has taken yet (counted as dropped)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        self._closed = False
        self.dropped = 0

    def put(self, packet: FramePacket) -> bool:
        """Store ``packet``; returns True when an unconsumed frame was overwritten."""

        with self._cond:
            dropped = self._packet is not None
            if dropped:
                self.dropped += 1
            self._packet = packet
            self._cond.notify_all()
            return dropped

    def get(self, timeout: float | None = None) -> FramePacket | None:
        """Take the latest frame (waits up to ``timeout``); None on timeout or close."""

        with self._cond:
            if self._packet is None and not self._closed:
                self._cond.wait(timeout)
            packet, self._packet = self._packet, None
            return packet

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


@dataclass
class PipelineStats:
    """Thread-safe counters and per-frame timing samples (ms)."""

    captured: int = 0
    processed: int = 0
    dropped: int = 0
    preprocess_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    postprocess_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    frame_age_ms: list[float] = field(default_factory=list)
    queue_wait_ms: list[float] = field(default_factory=list)
    detections: list[int] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    max_samples: int = 100_000
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> None:
        with self._lock:
            self.captured = self.processed = self.dropped = 0
            for samples in self._series().values():
                samples.clear()
            self.detections.clear()
            self.started = time.perf_counter()

    def _series(self) -> dict[str, list[float]]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
            "frame_age_ms": self.frame_age_ms,
            "queue_wait_ms": self.queue_wait_ms,
        }

    def on_capture(self, dropped: bool) -> None:
        with self._lock:
            self.captured += 1
            self.dropped += int(dropped)

    def on_result(self, result: Any, *, frame_age_ms: float, queue_wait_ms: float) -> None:
        timings = getattr(result, "timings", None)
        with self._lock:
            self.processed += 1
            if len(self.total_ms) >= self.max_samples:
                return
            if timings is not None:
                self.preprocess_ms.append(float(timings.preprocess_ms))
                self.inference_ms.append(float(timings.inference_ms))
                self.postprocess_ms.append(float(timings.postprocess_ms))
                self.total_ms.append(float(timings.total_ms))
            self.frame_age_ms.append(float(frame_age_ms))
            self.queue_wait_ms.append(float(queue_wait_ms))
            poses = getattr(result, "poses", None)
            if poses is not None:
                self.detections.append(len(poses))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1e-9, time.perf_counter() - self.started)
            out: dict[str, Any] = {
                "elapsed_s": elapsed,
                "captured": self.captured,
                "processed": self.processed,
                "dropped": self.dropped,
                "drop_ratio": self.dropped / self.captured if self.captured else 0.0,
                "capture_fps": self.captured / elapsed,
                "inference_fps": self.processed / elapsed,
                "mean_detections": float(np.mean(self.detections)) if self.detections else 0.0,
            }
            for name, values in self._series().items():
                out[name] = _percentiles(list(values))
        return out

    def report(self) -> str:
        s = self.summary()
        lines = [
            f"frames: captured {s['captured']}, processed {s['processed']}, dropped {s['dropped']} "
            f"({100.0 * s['drop_ratio']:.1f} %) in {s['elapsed_s']:.1f} s",
            f"rate:   capture {s['capture_fps']:.1f} fps, inference {s['inference_fps']:.1f} fps, "
            f"mean persons {s['mean_detections']:.2f}",
        ]
        for name in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "queue_wait_ms", "frame_age_ms"):
            p = s[name]
            lines.append(
                f"{name[:-3]:>12s}: p50 {p['p50']:7.2f} ms | p95 {p['p95']:7.2f} ms | "
                f"mean {p['mean']:7.2f} ms | max {p['max']:7.2f} ms"
            )
        return "\n".join(lines)


class InferencePipeline:
    """Capture thread + inference thread around a depth-1 latest-frame queue.

    Args:
        read_frame: blocking reader returning the next BGR frame, or ``None`` at
            the end of the source (stops the pipeline). Exceptions stop it too and
            are kept in :attr:`error`.
        detect: ``frame -> DetectionResult``; swappable at runtime with
            :meth:`set_detector` (e.g. input-size switch in the live test).
        frame_transform: optional per-frame op in the capture thread (e.g. mirror).
        max_frames: stop after this many processed frames (headless benchmark).
        max_capture_fps: pace the capture loop (for looping a still image /
            file source); 0 = as fast as the source delivers.
    """

    def __init__(
        self,
        read_frame: ReadFn,
        detect: DetectFn,
        *,
        frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        max_frames: int | None = None,
        max_capture_fps: float = 0.0,
        on_result: Callable[[PipelineResult], None] | None = None,
    ) -> None:
        self._read = read_frame
        self._detect = detect
        self._detect_lock = threading.Lock()
        self.frame_transform = frame_transform
        self.max_frames = max_frames
        self.on_result = on_result
        self._period = 1.0 / max_capture_fps if max_capture_fps and max_capture_fps > 0 else 0.0
        self.queue = LatestFrameQueue()
        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._paused = threading.Event()
        self._result_cond = threading.Condition()
        self._result: PipelineResult | None = None
        self.error: BaseException | None = None
        self._threads: list[threading.Thread] = []
        self._seq = 0

    # ----------------------------------------------------------------- control
    def start(self) -> "InferencePipeline":
        if self._threads:
            return self
        self.stats.reset()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="xtx4-capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="xtx4-inference", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self.queue.close()
        with self._result_cond:
            self._result_cond.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the pipeline finished (max_frames reached / source ended / stop)."""

        return self._done.wait(timeout)

    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._done.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def set_detector(self, detect: DetectFn) -> None:
        """Swap the detector; waits until a running detection finished."""

        with self._detect_lock:
            self._detect = detect

    def detector_lock(self) -> threading.Lock:
        """Held while a detection runs (close an old runner under it)."""

        return self._detect_lock

    # ------------------------------------------------------------------ results
    def latest(self) -> PipelineResult | None:
        with self._result_cond:
            return self._result

    def wait_result(self, after_seq: int = 0, timeout: float = 0.5) -> PipelineResult | None:
        """Newest result with ``seq > after_seq`` (waits up to ``timeout``), else the current one."""

        with self._result_cond:
            if (self._result is None or self._result.seq <= after_seq) and not self._done.is_set():
                self._result_cond.wait(timeout)
            return self._result

    # ------------------------------------------------------------------ threads
    def _capture_loop(self) -> None:
        next_due = time.perf_counter()
        try:
            while not self._stop.is_set():
                if self._period:
                    delay = next_due - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    next_due = max(next_due + self._period, time.perf_counter() - self._period)
                frame = self._read()
                if frame is None:
                    logger.info("Frame source ended")
                    break
                t_capture = time.perf_counter()
                if self.frame_transform is not None:
                    frame = self.frame_transform(frame)
                self._seq += 1
                dropped = self.queue.put(FramePacket(seq=self._seq, frame=frame, t_capture=t_capture))
                self.stats.on_capture(dropped)
        except BaseException as exc:  # noqa: BLE001 - surface to the caller
            self.error = exc
            logger.exception("Capture thread failed")
        finally:
            self.queue.close()

    def _inference_loop(self) -> None:
        try:
            while not self._stop.is_set():
                packet = self.queue.get(timeout=0.2)
                if packet is None:
                    if self.queue.closed:
                        break
                    continue
                if self._paused.is_set():
                    continue
                t_start = time.perf_counter()
                with self._detect_lock:
                    result = self._detect(packet.frame)
                t_done = time.perf_counter()
                item = PipelineResult(
                    seq=packet.seq, frame=packet.frame, result=result, t_capture=packet.t_capture, t_done=t_done
                )
                self.stats.on_result(
                    result, frame_age_ms=item.frame_age_ms, queue_wait_ms=(t_start - packet.t_capture) * 1000.0
                )
                with self._result_cond:
                    self._result = item
                    self._result_cond.notify_all()
                if self.on_result is not None:
                    self.on_result(item)
                if self.max_frames is not None and self.stats.processed >= self.max_frames:
                    break
        except BaseException as exc:  # noqa: BLE001 - surface to the caller
            self.error = exc
            logger.exception("Inference thread failed")
        finally:
            self._stop.set()
            self.queue.close()
            self._done.set()
            with self._result_cond:
                self._result_cond.notify_all()


def run_headless(
    read_frame: ReadFn,
    detect: DetectFn,
    *,
    frames: int,
    warmup: int = 10,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    max_capture_fps: float = 0.0,
    timeout_s: float | None = None,
) -> PipelineStats:
    """Benchmark the threaded pipeline: ``warmup`` frames are excluded from the stats."""

    if warmup > 0:
        warm = InferencePipeline(read_frame, detect, frame_transform=frame_transform, max_frames=warmup,
                                 max_capture_fps=max_capture_fps)
        warm.start()
        warm.wait(timeout_s)
        warm.stop()
        if warm.error is not None:
            raise RuntimeError("pipeline warm-up failed") from warm.error
    pipe = InferencePipeline(read_frame, detect, frame_transform=frame_transform, max_frames=frames,
                             max_capture_fps=max_capture_fps)
    pipe.start()
    finished = pipe.wait(timeout_s)
    pipe.stop()
    if pipe.error is not None:
        raise RuntimeError("pipeline failed") from pipe.error
    if not finished:
        logger.warning("Headless run timed out after %s s (%d frames processed)", timeout_s, pipe.stats.processed)
    return pipe.stats
