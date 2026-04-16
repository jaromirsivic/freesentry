"""
Camera proxy — lightweight main-process thread that owns a worker process.

The heavy work (capture, flip, crop, mask, YOLO inference, draw_pose) runs
in a separate OS process (``CameraWorkerProcess``) with its own GIL.
This thread reads finished frames from shared memory, runs the AI engagement
state machine, and exposes the same public API as the old Camera class.
"""

from __future__ import annotations

import time
import threading
from copy import deepcopy
from enum import Enum

import cv2
import numpy as np

from .common import Frame, EPSILON_DELAY
from .settingscontroller import get_settings_sync
from .cameradevice import create_camera_device
from .cameraframetransport import FrameTransport
from .cameraworker import CameraWorkerProcess


class CameraType(Enum):
    CV2 = "cv2"
    RPI = "rpi"
    DUMMY = "dummy"
    UNKNOWN = "unknown"


_WORKER_STOP_TIMEOUT = 4.0
_MAX_RESTART_ATTEMPTS = 5
_RESTART_BACKOFF_BASE = 1.0
_RESTART_BACKOFF_MAX = 30.0
_HEARTBEAT_INTERVAL = 5.0


class Camera(threading.Thread):
    """Proxy that owns a worker process and exposes frames to the rest of
    the application.  Public API is identical to the previous Camera class.
    """

    TIMEOUT_SECONDS = 5
    STOP_TIMEOUT_SECONDS = 4.0

    def __init__(
        self,
        *,
        index: int,
        camera_index: int,
        camera_code: str | None = None,
        camera_name: str | None = None,
        camera_type: str = "dummy",
        settings: dict | None = None,
        master_controller: "MasterController" = None,  # pyright: ignore[reportUndefinedVariable]
    ) -> None:
        super().__init__()
        self.daemon = True

        self._master_controller = master_controller
        self._index = index
        self._camera_index = camera_index
        self._camera_code = camera_code
        self._camera_type_str = camera_type
        self._camera_type = CameraType(camera_type) if camera_type in [e.value for e in CameraType] else CameraType.UNKNOWN
        self._camera_name = camera_name or f"{index}: Loading, please wait a minute..."

        self._state_lock = threading.RLock()
        self._active = False

        # Frame storage (read by MJPEG consumers)
        self._frame: Frame = self._create_blank_frame()
        self._frame_masked: Frame = self._create_blank_frame()
        self._frame_masked_ai: Frame = self._create_blank_frame()
        self._lock_frame = threading.Lock()
        self._lock_frame_masked = threading.Lock()
        self._lock_frame_masked_ai = threading.Lock()

        self._last_access_time_raw_frame: float = 0
        self._last_access_time_masked_frame: float = 0
        self._last_access_time_masked_ai_frame: float = 0

        self._stream_id = 0

        # --- Probe hardware for capabilities / resolutions ---
        probe_device = create_camera_device(
            camera_type=camera_type,
            camera_index=camera_index,
            camera_name=self._camera_name,
            settings=settings or {},
        )
        self.supported_resolutions = probe_device.get_supported_resolutions()
        self._capabilities = probe_device.get_capabilities()

        self._default_settings = probe_device.get_properties()
        self._default_settings["auto_focus"] = True
        self._default_settings["auto_white_balance_temperature"] = True
        self._default_settings["auto_exposure"] = True
        self._default_settings["index"] = index
        self._default_settings["camera_index"] = camera_index
        self._default_settings["camera_type"] = camera_type
        self._default_settings["name"] = self._camera_name
        self._default_settings["supported_resolutions"] = self.supported_resolutions
        probe_device.close()

        # Settings
        self._settings: dict | None = None
        self._settings_version = 0
        self._applied_settings_version = -1
        self._settings_modified = False

        camera_settings = deepcopy(self._default_settings)
        if settings is not None:
            camera_settings.update(settings)
        self.settings = camera_settings

        # Worker / transport state
        self._transport: FrameTransport | None = None
        self._worker: CameraWorkerProcess | None = None
        self._worker_started = False
        self._restart_count = 0
        self._last_restart_time = 0.0

        # Global settings change tracking
        self._last_sent_global_settings: object = None

        # Shared memory sequence tracking
        self._last_seq_raw = 0
        self._last_seq_masked = 0
        self._last_seq_ai = 0

        self._last_heartbeat_send = 0.0

    # ------------------------------------------------------------------
    # Blank frame helper
    # ------------------------------------------------------------------
    @staticmethod
    def _create_blank_frame() -> Frame:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "Loading, please wait a minute..."
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(image, text, ((640 - ts[0]) // 2, (480 + ts[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Frame(valid=False, image=image, time=0)

    # ------------------------------------------------------------------
    # Public properties (unchanged API)
    # ------------------------------------------------------------------
    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def settings(self) -> dict:
        with self._state_lock:
            if self._settings is not None:
                return deepcopy(self._settings)
        return deepcopy(self._default_settings)

    @settings.setter
    def settings(self, value: dict | None):
        with self._state_lock:
            self._settings = deepcopy(value) if value is not None else None
            self._settings_version += 1
            self._settings_modified = self._applied_settings_version != self._settings_version

    @property
    def camera_code(self) -> str | None:
        with self._state_lock:
            return self._camera_code

    @camera_code.setter
    def camera_code(self, value: str | None):
        with self._state_lock:
            self._camera_code = value

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    @property
    def frame(self) -> Frame:
        return self._get_raw_frame()

    @property
    def frame_masked(self) -> Frame:
        return self._get_masked_frame()

    @property
    def frame_masked_ai(self) -> Frame:
        return self._get_masked_ai_frame()

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------
    def _get_pending_settings_update(self) -> tuple[dict | None, int] | None:
        with self._state_lock:
            if self._applied_settings_version == self._settings_version:
                self._settings_modified = False
                return None
            snapshot = deepcopy(self._settings) if self._settings is not None else None
            return snapshot, self._settings_version

    def _mark_settings_applied(self, *, settings_version: int) -> None:
        with self._state_lock:
            if settings_version > self._applied_settings_version:
                self._applied_settings_version = settings_version
            self._settings_modified = self._applied_settings_version != self._settings_version

    def reset_settings(self):
        self.settings = deepcopy(self._default_settings)

    # ------------------------------------------------------------------
    # Stream token management
    # ------------------------------------------------------------------
    def create_stream_token(self) -> int:
        with self._state_lock:
            return self._stream_id

    def _invalidate_stream_tokens(self) -> None:
        with self._state_lock:
            self._stream_id += 1

    def _is_stream_token_current(self, *, stream_token: int | None) -> bool:
        if stream_token is None:
            return True
        with self._state_lock:
            return self._stream_id == stream_token

    # ------------------------------------------------------------------
    # Headless AI check
    # ------------------------------------------------------------------
    def _is_headless_ai_processing_required(self) -> bool:
        if self._camera_code != "scope_camera":
            return False
        mc = self._master_controller
        if mc is None:
            return False
        ai_agent = getattr(mc, "ai_agent", None)
        if ai_agent is None:
            return False
        fn = getattr(ai_agent, "is_fully_activated", None)
        if not callable(fn):
            return False
        try:
            return fn() is True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Frame access (public API)
    # ------------------------------------------------------------------
    def _ensure_device_open(self, *, stream_token: int | None = None) -> bool:
        if not self._is_stream_token_current(stream_token=stream_token):
            return False
        with self._lock_frame:
            self._last_access_time_raw_frame = time.time()
        if not self.is_alive():
            try:
                if not self._is_stream_token_current(stream_token=stream_token):
                    return False
                if not self.is_alive() and not self._active:
                    threading.Thread.__init__(self)
                    self.daemon = True
                self.start()
                for _ in range(int(2 / EPSILON_DELAY)):
                    if not self._is_stream_token_current(stream_token=stream_token):
                        return False
                    if self._active:
                        break
                    time.sleep(EPSILON_DELAY)
            except RuntimeError as err:
                print(f"Error starting camera proxy thread: {err}")
                return False
        return self._is_stream_token_current(stream_token=stream_token)

    def _get_raw_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            with self._lock_frame:
                self._last_access_time_raw_frame = time.time()
                if self._frame.valid:
                    return self._frame.copy()
        return self._create_blank_frame()

    def _get_masked_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            now = time.time()
            with self._lock_frame:
                self._last_access_time_raw_frame = now
            with self._lock_frame_masked:
                self._last_access_time_masked_frame = now
                if self._frame_masked.valid:
                    return self._frame_masked.copy()
        return self._create_blank_frame()

    def _get_masked_ai_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            now = time.time()
            with self._lock_frame:
                self._last_access_time_raw_frame = now
            with self._lock_frame_masked:
                self._last_access_time_masked_frame = now
            with self._lock_frame_masked_ai:
                self._last_access_time_masked_ai_frame = now
                if self._frame_masked_ai.valid:
                    return self._frame_masked_ai.copy()
        return self._create_blank_frame()

    def get_stream_frame(self, *, mode: int, stream_token: int) -> Frame | None:
        if not self._is_stream_token_current(stream_token=stream_token):
            return None
        if mode == 3:
            frame = self._get_masked_ai_frame(stream_token=stream_token)
        elif mode == 1:
            frame = self._get_masked_frame(stream_token=stream_token)
        else:
            frame = self._get_raw_frame(stream_token=stream_token)
        if not self._is_stream_token_current(stream_token=stream_token):
            return None
        return frame

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _start_worker(self) -> bool:
        if self._transport is not None:
            self._cleanup_transport()

        camera_settings_snapshot = self.settings
        global_settings_snapshot = dict(deepcopy(get_settings_sync()))

        self._transport = FrameTransport(
            camera_index=self._index,
            max_width=camera_settings_snapshot.get("width", 1920),
            max_height=camera_settings_snapshot.get("height", 1080),
        )
        worker_args = self._transport.get_worker_init_args()

        self._worker = CameraWorkerProcess(
            camera_type=self._camera_type_str,
            camera_index=self._camera_index,
            index=self._index,
            camera_code=self._camera_code,
            camera_name=self._camera_name,
            initial_camera_settings=camera_settings_snapshot,
            initial_global_settings=global_settings_snapshot,
            **worker_args,
        )
        try:
            self._worker.start()
            self._worker_started = True
            self._last_sent_global_settings = get_settings_sync()
            self._last_seq_raw = 0
            self._last_seq_masked = 0
            self._last_seq_ai = 0
            self._last_heartbeat_send = time.monotonic()
            return True
        except Exception as e:
            print(f"Failed to start worker for {self._camera_name}: {e}")
            self._cleanup_transport()
            return False

    def _stop_worker(self) -> None:
        if self._transport is not None:
            self._transport.send_command({"cmd": "stop"})

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=_WORKER_STOP_TIMEOUT)
            if self._worker.is_alive():
                print(f"Worker {self._camera_name} did not stop, terminating")
                self._worker.terminate()
                self._worker.join(timeout=2.0)
                if self._worker.is_alive():
                    self._worker.kill()
                    self._worker.join(timeout=1.0)

        self._worker = None
        self._worker_started = False
        self._cleanup_transport()

    def _cleanup_transport(self) -> None:
        if self._transport is not None:
            self._transport.cleanup()
            self._transport = None

    def _try_restart_worker(self) -> bool:
        if self._restart_count >= _MAX_RESTART_ATTEMPTS:
            return False
        backoff = min(
            _RESTART_BACKOFF_BASE * (2 ** self._restart_count),
            _RESTART_BACKOFF_MAX,
        )
        now = time.monotonic()
        if now - self._last_restart_time < backoff:
            return False
        self._restart_count += 1
        self._last_restart_time = now
        print(f"Restarting worker for {self._camera_name} "
              f"(attempt {self._restart_count}/{_MAX_RESTART_ATTEMPTS})")
        self._stop_worker()
        return self._start_worker()

    # ------------------------------------------------------------------
    # Main proxy loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._active = True
        if not self._start_worker():
            self._active = False
            return

        try:
            with self._lock_frame:
                self._last_access_time_raw_frame = time.time()

            while self._active:
                # --- 1. Check worker health ---
                if self._worker is None or not self._worker.is_alive():
                    if not self._try_restart_worker():
                        time.sleep(EPSILON_DELAY)
                        if self._restart_count >= _MAX_RESTART_ATTEMPTS:
                            break
                        continue

                # --- 2. Propagate settings changes to worker ---
                self._propagate_settings()

                # --- 3. Send heartbeat ---
                now_mono = time.monotonic()
                if now_mono - self._last_heartbeat_send >= _HEARTBEAT_INTERVAL:
                    if self._transport:
                        self._transport.send_command({"cmd": "heartbeat"})
                    self._last_heartbeat_send = now_mono

                # --- 4. Read frames from shared memory ---
                self._read_frames_from_transport()

                # --- 5. Process pose / engagement for scope_camera ---
                self._process_pose()

                # --- 6. Update demand flags ---
                self._update_demand_flags()

                # --- 7. Check if any consumer is still active ---
                if not self._should_stay_alive():
                    break

                time.sleep(EPSILON_DELAY)

        except Exception as e:
            print(f"Error in camera proxy thread: {e}")
        finally:
            self._stop_worker()
            self._active = False

    # ------------------------------------------------------------------
    # Settings propagation
    # ------------------------------------------------------------------
    def _propagate_settings(self) -> None:
        if self._transport is None:
            return

        # Camera-specific settings
        pending = self._get_pending_settings_update()
        if pending is not None:
            settings_snapshot, version = pending
            self._transport.send_command({
                "cmd": "update_camera_settings",
                "settings": settings_snapshot,
            })
            self._mark_settings_applied(settings_version=version)

        # Global settings (identity check — O(1))
        current_global = get_settings_sync()
        if current_global is not self._last_sent_global_settings:
            self._transport.send_command({
                "cmd": "update_global_settings",
                "settings": dict(deepcopy(current_global)),
            })
            self._last_sent_global_settings = current_global

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------
    def _read_frames_from_transport(self) -> None:
        if self._transport is None:
            return
        t = self._transport

        # Raw
        try:
            _, _, _, _, _, seq_raw, _ = t.slot_raw.read_header()
            if seq_raw != self._last_seq_raw:
                img, ts, valid, seq = t.slot_raw.read_frame()
                if img is not None and valid:
                    with self._lock_frame:
                        self._frame = Frame(valid=True, image=img, time=ts)
                    self._last_seq_raw = seq
        except Exception:
            pass

        # Masked
        try:
            _, _, _, _, _, seq_m, _ = t.slot_masked.read_header()
            if seq_m != self._last_seq_masked:
                img, ts, valid, seq = t.slot_masked.read_frame()
                if img is not None and valid:
                    with self._lock_frame_masked:
                        self._frame_masked = Frame(valid=True, image=img, time=ts)
                    self._last_seq_masked = seq
        except Exception:
            pass

        # AI
        try:
            _, _, _, _, _, seq_a, _ = t.slot_ai.read_header()
            if seq_a != self._last_seq_ai:
                img, ts, valid, seq = t.slot_ai.read_frame()
                if img is not None and valid:
                    with self._lock_frame_masked_ai:
                        self._frame_masked_ai = Frame(valid=True, image=img, time=ts)
                    self._last_seq_ai = seq
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pose / AI engagement
    # ------------------------------------------------------------------
    def _process_pose(self) -> None:
        if self._transport is None:
            return
        if self._camera_code != "scope_camera":
            self._drain_pose_pipe()
            return

        mc = self._master_controller
        if mc is None:
            self._drain_pose_pipe()
            return
        ai_agent = getattr(mc, "ai_agent", None)
        if ai_agent is None:
            self._drain_pose_pipe()
            return

        latest_raw_pose = None
        while self._transport.poll_pose():
            try:
                msg = self._transport.recv_pose()
                if isinstance(msg, dict) and msg.get("type") == "pose":
                    latest_raw_pose = msg.get("pose")
            except Exception:
                break

        if latest_raw_pose is not None:
            try:
                from .cameraai import translate_raw_pose_to_pose_dict

                settings = get_settings_sync()
                ai_setup = settings.get("aiSetup", {}) if isinstance(settings, dict) else {}
                pose = translate_raw_pose_to_pose_dict(
                    raw_pose=latest_raw_pose, ai_setup=ai_setup,
                )

                with self._lock_frame_masked_ai:
                    frame_ai = self._frame_masked_ai
                if frame_ai.valid:
                    frame_with_pose = Frame(
                        valid=True, image=frame_ai.image, time=frame_ai.time,
                        pose=pose,
                    )
                    engagement_result = ai_agent.engage(
                        frame=frame_with_pose, settings=settings,
                    )
                    ai_agent.draw_engagement_result(
                        frame=frame_with_pose,
                        engagement_result=engagement_result,
                    )
                    with self._lock_frame_masked_ai:
                        self._frame_masked_ai = frame_with_pose
            except Exception as e:
                print(f"Error processing pose for {self._camera_name}: {e}")

    def _drain_pose_pipe(self) -> None:
        if self._transport is None:
            return
        while self._transport.poll_pose():
            try:
                self._transport.recv_pose()
            except Exception:
                break

    # ------------------------------------------------------------------
    # Demand flags
    # ------------------------------------------------------------------
    def _update_demand_flags(self) -> None:
        if self._transport is None:
            return
        now = time.time()
        headless = self._is_headless_ai_processing_required()

        with self._lock_frame:
            elapsed_raw = now - self._last_access_time_raw_frame
        with self._lock_frame_masked:
            elapsed_masked = now - self._last_access_time_masked_frame
        with self._lock_frame_masked_ai:
            elapsed_ai = now - self._last_access_time_masked_ai_frame

        demand_raw = elapsed_raw <= self.TIMEOUT_SECONDS or headless
        demand_ai = elapsed_ai <= self.TIMEOUT_SECONDS or headless
        demand_masked = elapsed_masked <= self.TIMEOUT_SECONDS or demand_ai

        self._transport.send_command({
            "cmd": "update_demand",
            "raw": demand_raw,
            "masked": demand_masked,
            "ai": demand_ai,
        })

    def _should_stay_alive(self) -> bool:
        if self._is_headless_ai_processing_required():
            return True
        with self._lock_frame:
            elapsed = time.time() - self._last_access_time_raw_frame
        return elapsed <= self.TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._invalidate_stream_tokens()
        self._active = False
        if threading.current_thread() is self:
            return
        if not self.is_alive():
            self._stop_worker()
            return
        self.join(timeout=self.STOP_TIMEOUT_SECONDS + _WORKER_STOP_TIMEOUT)
        if self.is_alive():
            msg = (
                f"Camera proxy still alive after stop timeout "
                f"(index={self._index}, camera_index={self._camera_index}, "
                f"camera_code={self._camera_code}, camera_name={self._camera_name})"
            )
            print(msg)
