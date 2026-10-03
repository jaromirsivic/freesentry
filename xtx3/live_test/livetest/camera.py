"""Cross-platform camera discovery and a threaded "latest frame" grabber.

Backends: Windows MSMF (or DirectShow), Linux V4L2, macOS AVFoundation. Video
files and stream URLs (RTSP/HTTP) are supported as sources too; files loop.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SYSTEM = platform.system()


@dataclass(slots=True)
class CameraSource:
    """Either a local camera (``index`` + ``backend``) or a file / URL ``uri``."""

    name: str
    index: int | None = None
    backend: int = cv2.CAP_ANY
    uri: str | None = None

    @property
    def is_camera(self) -> bool:
        return self.uri is None

    @property
    def is_file(self) -> bool:
        return self.uri is not None and Path(self.uri).is_file()

    def describe(self) -> str:
        if self.is_camera:
            return f"[{self.index}] {self.name}"
        return self.name

    def key(self) -> str:
        """Stable identifier used to remember the last selection."""

        return f"cam:{self.index}:{self.name}" if self.is_camera else f"uri:{self.uri}"


def default_backend(windows_backend: str = "msmf") -> int:
    if SYSTEM == "Windows":
        return cv2.CAP_DSHOW if windows_backend.lower() == "dshow" else cv2.CAP_MSMF
    if SYSTEM == "Linux":
        return cv2.CAP_V4L2
    if SYSTEM == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _probe_cameras(backend: int, max_index: int) -> list[CameraSource]:
    """Fallback discovery: try to open indices ``0..max_index`` and grab a frame."""

    found: list[CameraSource] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(CameraSource(name=f"Camera {index}", index=index, backend=backend))
        finally:
            cap.release()
    return found


def list_cameras(*, windows_backend: str = "msmf", max_probe_index: int = 8) -> list[CameraSource]:
    """Enumerate cameras with names when possible, else by probing indices."""

    backend = default_backend(windows_backend)
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        infos = enumerate_cameras(backend)
        cameras = [CameraSource(name=str(info.name), index=int(info.index), backend=backend) for info in infos]
        if cameras:
            return cameras
    except Exception as exc:  # noqa: BLE001 - optional dependency / unsupported backend
        logger.info("Named camera enumeration unavailable (%s); probing indices", exc)
    return _probe_cameras(backend, max_probe_index)


def source_from_uri(uri: str) -> CameraSource:
    uri = uri.strip().strip('"')
    if uri.isdigit():
        return CameraSource(name=f"Camera {uri}", index=int(uri), backend=default_backend())
    return CameraSource(name=Path(uri).name if Path(uri).is_file() else uri, uri=uri)


def _permission_hint() -> str:
    if SYSTEM == "Darwin":
        return ("macOS: allow camera access for your terminal app in System Settings -> "
                "Privacy & Security -> Camera, then restart the terminal.")
    if SYSTEM == "Linux":
        return "Linux: make sure your user is in the 'video' group (sudo usermod -aG video $USER)."
    return "Windows: check Settings -> Privacy & security -> Camera and that no other app holds the camera."


def open_capture(source: CameraSource, *, width: int, height: int, fps: int, mjpg: bool) -> cv2.VideoCapture:
    """Open a capture and request resolution / FPS (the driver may pick others)."""

    if source.is_camera:
        cap = cv2.VideoCapture(source.index, source.backend)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera {source.describe()}. {_permission_hint()}")
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source.uri)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video source {source.uri}")
    return cap


class FrameGrabber:
    """Background thread that keeps only the most recent frame."""

    def __init__(self, source: CameraSource, *, width: int, height: int, fps: int, mjpg: bool) -> None:
        self.source = source
        self._open_args = dict(width=width, height=height, fps=fps, mjpg=mjpg)
        self._cap = open_capture(source, **self._open_args)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(f"Source {source.describe()} opened but returned no frame. {_permission_hint()}")

        self.frame_size = (frame.shape[1], frame.shape[0])
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) if source.is_file else 0.0
        self._file_period = 1.0 / file_fps if file_fps and file_fps > 0 else 0.0

        self._cond = threading.Condition()
        self._frame: np.ndarray | None = frame
        self._seq = 1
        self._stop = threading.Event()
        self._fps = 0.0
        self.status = "ok"
        self._thread = threading.Thread(target=self._run, name="FrameGrabber", daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:
        return self._fps

    def _reopen(self) -> bool:
        self._cap.release()
        try:
            self._cap = open_capture(self.source, **self._open_args)
            return True
        except RuntimeError as exc:
            logger.warning("%s", exc)
            return False

    def _run(self) -> None:
        failures = 0
        window_start = time.perf_counter()
        window_frames = 0
        next_due = time.perf_counter()
        while not self._stop.is_set():
            if self._file_period:
                delay = next_due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_due = max(next_due + self._file_period, time.perf_counter() - self._file_period)

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.source.is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    continue
                failures += 1
                if failures >= 30:
                    self.status = "reconnecting"
                    if not self._reopen():
                        self._stop.wait(1.0)
                    failures = 0
                else:
                    self._stop.wait(0.01)
                continue

            failures = 0
            self.status = "ok"
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

            window_frames += 1
            elapsed = time.perf_counter() - window_start
            if elapsed >= 1.0:
                self._fps = window_frames / elapsed
                window_frames = 0
                window_start = time.perf_counter()

    def latest(self, after_seq: int = 0, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        """Return ``(seq, frame)``; waits up to ``timeout`` for a frame newer than ``after_seq``."""

        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait(timeout)
            return self._seq, self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
