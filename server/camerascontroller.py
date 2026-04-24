"""
Cameras controller — singleton host for the one and only
:class:`~server.cameraworker.CameraWorkerProcess` and its
:class:`~server.cameraframetransport.FrameTransport`.

Responsibilities:

* Create the :class:`Camera` proxies (one per potential input device) at
  startup so the UI can list / select them.
* Own the singleton worker process + transport + pipes.
* Dispatch commands from the active :class:`Camera` proxy to the worker
  (``select_camera``, ``release_camera``, ``update_demand``,
  ``update_camera_settings``, ``update_global_settings``, ``heartbeat``).
* Route pose samples from the worker to the right proxy (keyed by
  ``camera_index`` so stale messages during a switch are filtered out).
* Route engagement snapshots back to the worker through the dedicated
  ``ai_result_pipe``.
* Detect a dead worker and restart it with exponential backoff.
* Pump a heartbeat to the worker from its own background thread so the
  worker can distinguish "main is busy with GC" from "main crashed".
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any

from .camera import Camera
from .cameraframetransport import FrameTransport
from .cameraworker import CameraWorkerProcess
from .settingscontroller import get_settings_sync


_HEARTBEAT_INTERVAL = 1.0       # how often main pings the worker
_WORKER_STOP_TIMEOUT = 4.0
_WORKER_HEARTBEAT_STALE = 5.0   # worker heartbeat considered lost after this
_RESTART_BACKOFF_BASE = 1.0
_RESTART_BACKOFF_MAX = 30.0
_MAX_RESTART_ATTEMPTS = 10


class CamerasController:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self, *, master_controller: "MasterController"):  # pyright: ignore[reportUndefinedVariable]
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._master_controller = master_controller
        self._cameras: list[Camera] = []
        self._lifecycle_lock = threading.RLock()

        # --- singleton worker state ---
        self._worker_lock = threading.RLock()
        self._transport: FrameTransport | None = None
        self._worker: CameraWorkerProcess | None = None
        self._active_camera: Camera | None = None
        self._pending_select_generation = 0
        self._last_sent_global_settings: object = None
        self._last_worker_heartbeat = 0.0
        self._restart_count = 0
        self._last_restart_time = 0.0

        # --- pose demux: one queue per physical camera index ---
        self._pose_lock = threading.Lock()
        self._pose_by_camera: dict[int, list[dict]] = {}

        # --- watchdog status cache (populated by worker heartbeats) ---
        # The worker's dedicated heartbeat thread sends a status snapshot
        # every ~1 s piggybacked onto the heartbeat message (device
        # state, capture health, AI model state, demand_seen, ring
        # sequences).  The HTTP relay's stall watchdog reads the latest
        # snapshot via ``get_worker_status`` to classify stalls.
        self._worker_status_lock = threading.Lock()
        self._last_worker_status: dict | None = None
        self._last_worker_status_mono: float = 0.0

        # --- background threads ---
        self._io_thread: threading.Thread | None = None
        self._io_stop = threading.Event()

        self.reset()

    # ------------------------------------------------------------------
    # Camera list
    # ------------------------------------------------------------------
    @property
    def cameras(self) -> list[Camera]:
        with self._lifecycle_lock:
            return list(self._cameras)

    def _stop_cameras_locked(self) -> None:
        stop_errors: list[tuple[str, Exception]] = []
        for camera in self._cameras:
            try:
                camera.stop()
            except Exception as exc:
                camera_name = getattr(camera, "camera_name", repr(camera))
                print(f"Error stopping camera proxy '{camera_name}': {exc}")
                stop_errors.append((camera_name, exc))

        if stop_errors:
            failed_cameras = ", ".join(name for name, _ in stop_errors)
            raise RuntimeError(
                f"Failed to stop camera proxies: {failed_cameras}"
            ) from stop_errors[0][1]

    def stop(self) -> None:
        with self._lifecycle_lock:
            try:
                self._stop_cameras_locked()
            finally:
                self._stop_worker_and_io()

    def stop_camera(self, *, index: int) -> None:
        with self._lifecycle_lock:
            if index < 0 or index >= len(self._cameras):
                raise IndexError(f"Camera index {index} not found")
            self._cameras[index].stop()

    def reset(self, *, max_index=8, reset_to_default=False):
        """Reload the list of cameras.  The shared worker is also stopped
        and will be restarted lazily the next time a camera is activated.
        """
        with self._lifecycle_lock:
            print("Resetting cameras controller")
            try:
                self._stop_cameras_locked()
            except Exception as exc:
                print(f"Error while stopping cameras during reset: {exc}")
            self._stop_worker_and_io()
            time.sleep(1.0)

            settings = get_settings_sync()
            self._cameras = []

            for i in range(max_index + 6):
                print(f"Creating camera with index \"{i}\"")
                found_camera_settings = None
                found_camera_code = None
                if "cameras" in settings:
                    for camera_code, camera_settings in settings["cameras"].items():
                        if "index" in camera_settings and camera_settings["index"] == i:
                            found_camera_settings = camera_settings
                            found_camera_code = camera_code
                            break
                if reset_to_default:
                    found_camera_settings = None

                if i < 4:
                    camera_index = i
                    camera_type = "dummy"
                    camera_name = f'{i}: dummy_camera_{camera_index}'
                elif i < 6:
                    camera_index = i - 4
                    camera_type = "rpi"
                    camera_name = f'{i}: rpi_camera_{camera_index}'
                else:
                    camera_index = i - 6
                    camera_type = "cv2"
                    camera_name = f'{i}: cv2_camera_{camera_index}'

                try:
                    self._cameras.append(Camera(
                        index=i,
                        camera_index=camera_index,
                        camera_code=found_camera_code,
                        camera_name=camera_name,
                        camera_type=camera_type,
                        settings=found_camera_settings,
                        master_controller=self._master_controller,
                    ))
                except Exception as e:
                    print(
                        f"Error creating camera with index=\"{i}\", camera_index=\"{camera_index}\", "
                        f"camera_name=\"{camera_name}\". error: {e}. Creating dummy camera instead."
                    )
                    self._cameras.append(Camera(
                        index=i,
                        camera_index=-1,
                        camera_code=found_camera_code,
                        camera_name=camera_name,
                        camera_type="dummy",
                        settings=found_camera_settings,
                        master_controller=self._master_controller,
                    ))

    # ==================================================================
    # Worker lifecycle
    # ==================================================================
    def _ensure_worker_running_locked(self) -> bool:
        if self._worker is not None and self._worker.is_alive() and self._transport is not None:
            return True

        # Backoff check
        now_mono = time.monotonic()
        if self._restart_count >= _MAX_RESTART_ATTEMPTS:
            return False
        if self._worker is not None:
            backoff = min(
                _RESTART_BACKOFF_BASE * (2 ** max(self._restart_count - 1, 0)),
                _RESTART_BACKOFF_MAX,
            )
            if now_mono - self._last_restart_time < backoff:
                return False

        self._cleanup_worker_locked()
        try:
            self._transport = FrameTransport()
            args = self._transport.get_worker_init_args()
            self._worker = CameraWorkerProcess(**args)
            self._worker.start()
            self._last_worker_heartbeat = now_mono
            self._last_restart_time = now_mono
            self._restart_count += 1
            self._last_sent_global_settings = None
            self._pending_select_generation += 1
            self._start_io_thread_locked()
            print("[Controller] Camera worker started")
            return True
        except Exception as exc:
            print(f"[Controller] Failed to start camera worker: {exc}")
            self._cleanup_worker_locked()
            return False

    def _cleanup_worker_locked(self) -> None:
        self._stop_io_thread_locked()

        if self._worker is not None:
            try:
                if self._transport is not None:
                    self._transport.send_command({"cmd": "stop"})
                if self._worker.is_alive():
                    self._worker.join(timeout=_WORKER_STOP_TIMEOUT)
                if self._worker.is_alive():
                    print("[Controller] Worker did not stop, terminating")
                    self._worker.terminate()
                    self._worker.join(timeout=2.0)
                if self._worker.is_alive():
                    self._worker.kill()
                    self._worker.join(timeout=1.0)
            except Exception as exc:
                print(f"[Controller] Error stopping worker: {exc}")
            self._worker = None

        if self._transport is not None:
            try:
                self._transport.cleanup()
            except Exception:
                pass
            self._transport = None

        self._active_camera = None
        with self._pose_lock:
            self._pose_by_camera.clear()
        # Drop any cached worker status so the stall watchdog does not
        # classify stalls against a now-stale snapshot from a worker
        # that no longer exists.
        with self._worker_status_lock:
            self._last_worker_status = None
            self._last_worker_status_mono = 0.0

    def _stop_worker_and_io(self) -> None:
        with self._worker_lock:
            self._cleanup_worker_locked()
            self._restart_count = 0
            self._last_restart_time = 0.0

    # ------------------------------------------------------------------
    # Background IO thread: heartbeat + worker->main pipe drain
    # ------------------------------------------------------------------
    def _start_io_thread_locked(self) -> None:
        if self._io_thread is not None and self._io_thread.is_alive():
            return
        self._io_stop.clear()
        t = threading.Thread(
            target=self._io_thread_main, name="cameras-controller-io", daemon=True,
        )
        self._io_thread = t
        t.start()

    def _stop_io_thread_locked(self) -> None:
        self._io_stop.set()
        if self._io_thread is not None and self._io_thread.is_alive():
            # Avoid self-join when called from within the IO thread itself
            # (happens when the IO thread detects the worker died and drives
            # cleanup).  It will exit on its own once it notices the stop flag.
            if self._io_thread is not threading.current_thread():
                self._io_thread.join(timeout=2.0)
        # If called from the IO thread, leave the reference; a fresh thread
        # will be started by ``_ensure_worker_running_locked`` on next
        # activation and it will replace this handle.
        if self._io_thread is not threading.current_thread():
            self._io_thread = None

    def _io_thread_main(self) -> None:
        last_heartbeat_sent = 0.0
        while not self._io_stop.is_set():
            now_mono = time.monotonic()

            transport = self._transport
            worker = self._worker
            if transport is None or worker is None:
                time.sleep(0.05)
                continue

            # --- send heartbeat to worker ---
            if now_mono - last_heartbeat_sent >= _HEARTBEAT_INTERVAL:
                transport.send_command({"cmd": "heartbeat"})
                last_heartbeat_sent = now_mono

            # --- drain pose pipe (worker -> main) ---
            drained = 0
            try:
                while drained < 64 and transport.poll_pose(0):
                    msg = transport.recv_pose()
                    drained += 1
                    if not isinstance(msg, dict):
                        continue
                    mtype = msg.get("type")
                    if mtype == "heartbeat":
                        self._last_worker_heartbeat = now_mono
                        status = msg.get("status")
                        if isinstance(status, dict):
                            with self._worker_status_lock:
                                self._last_worker_status = status
                                self._last_worker_status_mono = now_mono
                        continue
                    if mtype == "pose":
                        cam_idx = int(msg.get("camera_index", -1))
                        with self._pose_lock:
                            self._pose_by_camera.setdefault(cam_idx, []).append(msg)
                            # Keep bounded so we never balloon if main is
                            # briefly stalled.
                            q = self._pose_by_camera[cam_idx]
                            if len(q) > 8:
                                del q[:-8]
            except (EOFError, OSError, BrokenPipeError):
                # Pipe broken -> worker gone.  Let the crash detector handle it.
                pass

            # --- worker health check ---
            if not worker.is_alive():
                print("[Controller] Worker died; will attempt restart")
                with self._worker_lock:
                    self._cleanup_worker_locked()
                # exit this IO thread; a new one will be spawned on next
                # activation.
                return

            if self._last_worker_heartbeat and (now_mono - self._last_worker_heartbeat) > (
                _WORKER_HEARTBEAT_STALE * 3
            ):
                # Worker process is running but not sending heartbeats —
                # probably stuck.  Terminate so _ensure_worker_running picks
                # it up next activation.
                print("[Controller] Worker heartbeat stale; terminating")
                with self._worker_lock:
                    self._cleanup_worker_locked()
                return

            time.sleep(0.05)

    # ==================================================================
    # API used by Camera proxies
    # ==================================================================
    def activate_camera(self, camera: Camera) -> bool:
        with self._worker_lock:
            if not self._ensure_worker_running_locked():
                return False

            if self._active_camera is camera:
                # Already active — just refresh settings if they changed.
                self._push_global_settings_if_changed_locked()
                return True

            # Deactivate previous active camera's stream so it exits cleanly.
            prev = self._active_camera
            if prev is not None:
                try:
                    prev._invalidate_stream_tokens()
                except Exception:
                    pass
                try:
                    prev._active = False
                except Exception:
                    pass

            # Send select_camera command
            cam_settings = deepcopy(camera.settings)
            try:
                global_settings = dict(deepcopy(get_settings_sync()))
            except Exception:
                global_settings = {}
            self._last_sent_global_settings = global_settings
            self._transport.send_command({
                "cmd": "select_camera",
                "camera_type": camera.camera_type_str,
                "camera_index": camera.physical_camera_index,
                "index": camera.index,
                "camera_code": camera.camera_code,
                "camera_name": camera.camera_name,
                "camera_settings": cam_settings,
                "global_settings": global_settings,
            })
            self._active_camera = camera
            self._pending_select_generation += 1

            # Reset pose queues so stale messages from the previous camera
            # never bleed into the new one.
            with self._pose_lock:
                self._pose_by_camera.clear()

            return True

    def is_camera_active(self, camera: Camera) -> bool:
        with self._worker_lock:
            return self._active_camera is camera

    def deactivate_camera(self, camera: Camera) -> None:
        with self._worker_lock:
            if self._active_camera is not camera:
                return
            if self._transport is not None:
                self._transport.send_command({"cmd": "release_camera"})
            self._active_camera = None
            with self._pose_lock:
                self._pose_by_camera.clear()

    def _push_global_settings_if_changed_locked(self) -> None:
        try:
            current = get_settings_sync()
        except Exception:
            return
        if current is self._last_sent_global_settings:
            return
        if self._transport is None:
            return
        try:
            snapshot = dict(deepcopy(current))
        except Exception:
            snapshot = {}
        self._transport.send_command({
            "cmd": "update_global_settings",
            "settings": snapshot,
        })
        self._last_sent_global_settings = current

    def update_camera_settings(self, camera: Camera, settings_snapshot: dict) -> None:
        with self._worker_lock:
            if self._active_camera is not camera or self._transport is None:
                return
            self._transport.send_command({
                "cmd": "update_camera_settings",
                "settings": settings_snapshot,
            })
            self._push_global_settings_if_changed_locked()

    def update_demand_from_camera(self, camera: Camera) -> None:
        with self._worker_lock:
            if self._active_camera is not camera or self._transport is None:
                return
            raw_t, masked_t, ai_t = camera.get_last_access_times()
            q_raw, q_masked, q_ai = camera.get_requested_qualities()
            now = time.time()
            headless = False
            try:
                headless = camera._is_headless_ai_processing_required()
            except Exception:
                pass

            demand_raw = (now - raw_t) <= Camera.TIMEOUT_SECONDS or headless
            demand_ai = (now - ai_t) <= Camera.TIMEOUT_SECONDS or headless
            demand_masked = (now - masked_t) <= Camera.TIMEOUT_SECONDS or demand_ai

            self._transport.send_command({
                "cmd": "update_demand",
                "raw": bool(demand_raw),
                "masked": bool(demand_masked),
                "ai": bool(demand_ai),
                "jpeg_quality_raw": int(q_raw),
                "jpeg_quality_masked": int(q_masked),
                "jpeg_quality_ai": int(q_ai),
            })
            self._push_global_settings_if_changed_locked()

    def read_jpeg_for(
        self,
        *,
        camera: Camera,
        mode_key: str,
        since_sequence: int = 0,
    ) -> tuple[bytes | None, float, int, int, int]:
        """Return the newest JPEG for *camera* in *mode_key* (lossy
        newest-wins).  If the worker published multiple frames since the
        previous call, only the freshest is returned — intermediate
        frames are intentionally dropped so the HTTP relay never lags
        behind the camera.
        """
        with self._worker_lock:
            if self._active_camera is not camera or self._transport is None:
                return None, 0.0, 0, -1, 0
            transport = self._transport
        try:
            return transport.read_jpeg(mode_key=mode_key, since_sequence=since_sequence)
        except Exception as exc:
            print(f"[Controller] read_jpeg error: {exc}")
            return None, 0.0, 0, -1, 0

    def wait_for_new_jpeg_for(
        self,
        *,
        camera: Camera,
        mode_key: str,
        timeout: float,
    ) -> bool:
        """Block briefly until the worker publishes a new frame for the
        active *camera* in *mode_key*.  Returns ``False`` immediately if
        *camera* is not the active camera or the worker is not running.
        """
        with self._worker_lock:
            if self._active_camera is not camera or self._transport is None:
                return False
            transport = self._transport
        try:
            return transport.wait_for_new_jpeg(mode_key=mode_key, timeout=timeout)
        except Exception:
            return False

    def get_worker_status(self) -> tuple[dict | None, float]:
        """Return ``(latest_worker_status_snapshot, age_in_seconds)``.

        The stall watchdog in the HTTP relay calls this to classify
        stalls.  The snapshot itself is the dict emitted by the worker's
        heartbeat thread (see :mod:`server.cameraworker`); age is the
        monotonic-time delta since the snapshot arrived and is
        ``math.inf`` if no status has ever been received (or the cache
        was just cleared after a worker restart).
        """
        with self._worker_status_lock:
            status = self._last_worker_status
            stamp = self._last_worker_status_mono
        if status is None or stamp == 0.0:
            return None, float("inf")
        return status, max(0.0, time.monotonic() - stamp)

    def collect_pose_messages(self, *, camera_index: int) -> list[dict]:
        with self._pose_lock:
            msgs = self._pose_by_camera.pop(camera_index, [])
        return msgs

    def drain_pose_for_non_ai(self, *, camera_index: int) -> None:
        with self._pose_lock:
            self._pose_by_camera.pop(camera_index, None)

    def send_engagement_result(
        self,
        *,
        result_payload: dict,
        pose_dict: Any,
    ) -> None:
        with self._worker_lock:
            if self._transport is None:
                return
            transport = self._transport
        try:
            transport.send_ai_result({
                "type": "engagement",
                "result_payload": result_payload,
                "pose_dict": pose_dict,
            })
        except (BrokenPipeError, OSError, EOFError):
            pass
