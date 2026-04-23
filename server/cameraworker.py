"""
Singleton camera worker process.

There is exactly **one** ``CameraWorkerProcess`` per application.  It does
every expensive step of the camera pipeline inside its own OS process
(independent GIL):

    1.  open a selected camera device
    2.  capture frames
    3.  apply flip / rotate / crop / mask
    4.  optionally run YOLO pose inference (AI mode)
    5.  overlay AI drawings (pose circles + engagement rectangle + text)
    6.  JPEG-encode each demanded mode (raw, masked, AI)
    7.  publish bytes into the shared-memory JPEG triple-buffers

The main process only needs to relay bytes to the HTTP client and run the
AI engagement state machine (sending the result back via ``ai_result_pipe``).

IMPORTANT: This module must never import ``settingscontroller`` or any
other main-process-only singleton.  All runtime settings arrive through
the command pipe.

Created via the explicit ``spawn`` multiprocessing context so behaviour is
identical across Linux, macOS, and Windows.
"""

from __future__ import annotations

import atexit
import collections
import multiprocessing
import threading
import time
from typing import Any

import numpy as np

_mp_ctx = multiprocessing.get_context("spawn")

EPSILON_DELAY = 0.005
HEARTBEAT_SEND_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 5.0  # seconds without main's heartbeat before idling
POLL_BATCH_LIMIT = 32

FLIP_HORIZONTAL = 1
FLIP_VERTICAL = 0
FLIP_BOTH = -1

MODE_RAW = 0
MODE_MASKED = 1
MODE_AI = 3
MODE_KEY_BY_NUM = {MODE_RAW: "raw", MODE_MASKED: "masked", MODE_AI: "ai"}


class RollingFps:
    """Sliding-window FPS counter (default window: 2 s)."""

    __slots__ = ("_window", "_ts")

    def __init__(self, window: float = 2.0) -> None:
        self._window = window
        self._ts: collections.deque[float] = collections.deque()

    def tick(self, now: float) -> None:
        self._ts.append(now)
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    def value(self, now: float) -> float:
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
        return len(self._ts) / self._window

    def reset(self) -> None:
        self._ts.clear()


class CameraWorkerProcess(_mp_ctx.Process):
    """Singleton camera worker.

    Commands (received on ``command_pipe``):

    ``select_camera``
        Switch to a different physical/logical camera.  Payload:
        ``camera_type``, ``camera_index``, ``index``, ``camera_code``,
        ``camera_name``, ``camera_settings``, ``global_settings``.
        If a device is already open it is closed first.
    ``release_camera``
        Close the currently open device (go to idle).  Worker stays alive.
    ``update_demand``
        Update ``raw`` / ``masked`` / ``ai`` booleans and the three JPEG
        quality values (``jpeg_quality_raw``, ``jpeg_quality_masked``,
        ``jpeg_quality_ai``).
    ``update_camera_settings``
        New camera-specific settings (flip, crop, resolution, etc).
    ``update_global_settings``
        New global settings snapshot (contains ``aiSetup``).
    ``heartbeat``
        Keep-alive from main.  Resets the ``HEARTBEAT_TIMEOUT`` idle timer.
    ``stop``
        Graceful shutdown.

    AI engagement results (received on ``ai_result_pipe``):

    ``engagement``  ->  ``{"type": "engagement", "result_payload": {...},
                           "pose_dict": [...], "last_image_seq": N}``
    """

    def __init__(
        self,
        *,
        shm_names: dict,
        max_jpeg_size: int,
        slot_lock_groups: dict,
        index_locks: dict,
        new_frame_events: dict,
        command_pipe_conn: Any,
        pose_pipe_conn: Any,
        ai_result_pipe_conn: Any,
    ) -> None:
        super().__init__(daemon=True)
        self._shm_names = shm_names
        self._max_jpeg_size = max_jpeg_size
        self._slot_lock_groups = slot_lock_groups
        self._index_locks = index_locks
        self._new_frame_events = new_frame_events
        self._command_conn = command_pipe_conn
        self._pose_conn = pose_pipe_conn
        self._ai_result_conn = ai_result_pipe_conn

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901  - state machine, intentionally long
        import cv2
        from .cameradevice import create_camera_device
        from .cameraframetransport import JpegRingBuffer
        from .aidraw import draw_engagement_overlay, draw_pose_overlay
        from .cameraai import translate_raw_pose_to_pose_dict

        rings: dict[str, JpegRingBuffer] = {}
        for mode_key in ("raw", "masked", "ai"):
            rings[mode_key] = JpegRingBuffer(
                name=self._shm_names[mode_key],
                max_jpeg_size=self._max_jpeg_size,
                create=False,
                slot_locks=self._slot_lock_groups[mode_key],
                index_lock=self._index_locks[mode_key],
                new_frame_event=self._new_frame_events[mode_key],
            )

        # --- device + settings state --------------------------------------
        device = None
        active_camera_index = -1  # -1 == idle
        active_camera_code: str | None = None
        active_camera_name: str = ""
        active_camera_type: str = ""
        active_index: int = -1
        camera_settings: dict = {}
        global_settings: dict = {}

        # --- demand / quality --------------------------------------------
        demand_raw = False
        demand_masked = False
        demand_ai = False
        jpeg_quality_raw = 80
        jpeg_quality_masked = 80
        jpeg_quality_ai = 80

        # --- sequences ----------------------------------------------------
        seq_raw = 0
        seq_masked = 0
        seq_ai = 0
        ai_image_seq = 0  # matches pose pipe samples
        last_engagement: dict | None = None
        last_engagement_pose: Any = None

        # --- FPS measurement (producer-side, 2 s sliding window) ---------
        fps_capture = RollingFps()
        fps_raw = RollingFps()
        fps_masked = RollingFps()
        fps_ai = RollingFps()

        def _fps_snapshot(now: float) -> dict:
            return {
                "capture": fps_capture.value(now),
                "raw": fps_raw.value(now),
                "masked": fps_masked.value(now),
                "ai": fps_ai.value(now),
            }

        # --- heartbeat timers --------------------------------------------
        last_heartbeat_recv = time.monotonic()
        idle_logged = False

        # Protects ``self._pose_conn.send`` because it is now written to
        # from both the main loop (pose samples) and the dedicated
        # heartbeat thread below.  ``multiprocessing.Connection.send`` is
        # not thread-safe.
        pose_send_lock = threading.Lock()

        # Dedicated heartbeat thread: keeps sending liveness pings to
        # main every ``HEARTBEAT_SEND_INTERVAL`` seconds even while the
        # main loop is blocked in a slow operation (e.g. the first YOLO
        # inference on CPU can take 15+ s).  Without this, main's
        # watchdog would flag the worker as hung and terminate it.
        heartbeat_stop = threading.Event()

        def _heartbeat_thread_main() -> None:
            while not heartbeat_stop.is_set():
                try:
                    with pose_send_lock:
                        self._pose_conn.send({"type": "heartbeat", "time": time.time()})
                except (BrokenPipeError, OSError, EOFError):
                    return
                except Exception as exc:
                    print(f"[Worker] heartbeat send error: {exc}")
                if heartbeat_stop.wait(timeout=HEARTBEAT_SEND_INTERVAL):
                    return

        heartbeat_thread = threading.Thread(
            target=_heartbeat_thread_main,
            name="camera-worker-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        def _close_device() -> None:
            nonlocal device, active_camera_index, active_camera_code
            nonlocal active_camera_name, active_camera_type, active_index
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
                device = None
            active_camera_index = -1
            active_camera_code = None
            active_camera_name = ""
            active_camera_type = ""
            active_index = -1
            fps_capture.reset()
            fps_raw.reset()
            fps_masked.reset()
            fps_ai.reset()

        def _cleanup() -> None:
            _close_device()
            for ring in rings.values():
                ring.close()
            for conn in (self._command_conn, self._pose_conn, self._ai_result_conn):
                try:
                    conn.close()
                except Exception:
                    pass

        atexit.register(_cleanup)

        # ---------------- helpers accessing the outer state --------------
        def _apply_select(msg: dict) -> None:
            nonlocal device, active_camera_index, active_camera_code
            nonlocal active_camera_name, active_camera_type, active_index
            nonlocal camera_settings, global_settings
            nonlocal seq_raw, seq_masked, seq_ai, ai_image_seq
            nonlocal last_engagement, last_engagement_pose
            _close_device()
            active_camera_type = msg.get("camera_type", "dummy")
            active_camera_index = int(msg.get("camera_index", -1))
            active_index = int(msg.get("index", active_camera_index))
            active_camera_code = msg.get("camera_code")
            active_camera_name = msg.get("camera_name", "") or ""
            camera_settings = dict(msg.get("camera_settings") or {})
            new_global = msg.get("global_settings")
            if new_global is not None:
                global_settings = dict(new_global)
            # Reset per-camera state
            seq_raw = seq_masked = seq_ai = 0
            ai_image_seq = 0
            last_engagement = None
            last_engagement_pose = None
            try:
                device = create_camera_device(
                    camera_type=active_camera_type,
                    camera_index=active_camera_index,
                    camera_name=active_camera_name,
                    settings=camera_settings,
                )
            except Exception as exc:
                print(f"[Worker] create_camera_device failed ({active_camera_name}): {exc}")
                device = None
                return
            if device is None or not device.open():
                print(f"[Worker] Failed to open camera {active_camera_name}")
                _close_device()
                return
            try:
                device.set_properties(camera_settings)
            except Exception as exc:
                print(f"[Worker] set_properties error for {active_camera_name}: {exc}")

        # ---------------- main loop --------------------------------------
        try:
            while True:
                now_mono = time.monotonic()

                # --- 1. drain command pipe (non-blocking) ---
                try:
                    processed = 0
                    while processed < POLL_BATCH_LIMIT and self._command_conn.poll(0):
                        msg = self._command_conn.recv()
                        processed += 1
                        cmd = msg.get("cmd")
                        if cmd == "stop":
                            return
                        elif cmd == "select_camera":
                            _apply_select(msg)
                            last_heartbeat_recv = now_mono
                            idle_logged = False
                        elif cmd == "release_camera":
                            _close_device()
                            idle_logged = True
                        elif cmd == "update_demand":
                            demand_raw = bool(msg.get("raw", demand_raw))
                            demand_masked = bool(msg.get("masked", demand_masked))
                            demand_ai = bool(msg.get("ai", demand_ai))
                            jpeg_quality_raw = int(msg.get("jpeg_quality_raw", jpeg_quality_raw))
                            jpeg_quality_masked = int(msg.get("jpeg_quality_masked", jpeg_quality_masked))
                            jpeg_quality_ai = int(msg.get("jpeg_quality_ai", jpeg_quality_ai))
                        elif cmd == "update_camera_settings":
                            camera_settings = dict(msg.get("settings") or {})
                            if device is not None:
                                try:
                                    device.set_properties(camera_settings)
                                except Exception as exc:
                                    print(f"[Worker] set_properties error: {exc}")
                        elif cmd == "update_global_settings":
                            global_settings = dict(msg.get("settings") or {})
                        elif cmd == "heartbeat":
                            last_heartbeat_recv = now_mono
                            if idle_logged:
                                idle_logged = False
                except (EOFError, OSError):
                    return

                # --- 2. drain ai_result pipe (non-blocking) ---
                try:
                    processed = 0
                    while processed < POLL_BATCH_LIMIT and self._ai_result_conn.poll(0):
                        ai_msg = self._ai_result_conn.recv()
                        processed += 1
                        if isinstance(ai_msg, dict) and ai_msg.get("type") == "engagement":
                            last_engagement = ai_msg.get("result_payload")
                            last_engagement_pose = ai_msg.get("pose_dict")
                except (EOFError, OSError):
                    return

                # --- 3. heartbeat timeout -> idle (close device, stay alive) ---
                if now_mono - last_heartbeat_recv > HEARTBEAT_TIMEOUT:
                    if device is not None:
                        if not idle_logged:
                            print(f"[Worker] No heartbeat for {HEARTBEAT_TIMEOUT}s -> idling")
                            idle_logged = True
                        _close_device()
                    time.sleep(0.05)
                    continue

                # --- 4. heartbeats are now sent by ``heartbeat_thread`` so
                #        the liveness ping keeps flowing even while the main
                #        loop is blocked on a slow operation (YOLO warm-up,
                #        camera capture stall, JPEG encoding).

                # --- 5. if no camera selected, idle sleep ---
                if device is None or active_camera_index < 0:
                    time.sleep(EPSILON_DELAY)
                    continue

                # --- 6. if no consumer demands anything, still tick heartbeat ---
                if not (demand_raw or demand_masked or demand_ai):
                    time.sleep(EPSILON_DELAY)
                    continue

                # --- 7. capture ---
                try:
                    valid, image = device.get_image()
                except Exception as exc:
                    print(f"[Worker] capture exception for {active_camera_name}: {exc}")
                    valid, image = False, None

                if not valid or image is None:
                    try:
                        device.close()
                    except Exception:
                        pass
                    time.sleep(0.25)
                    try:
                        if device is None or not device.open():
                            time.sleep(0.75)
                            continue
                    except Exception:
                        time.sleep(0.75)
                        continue
                    continue

                now_ts = time.time()
                fps_capture.tick(now_ts)

                # --- 8. flip ---
                fh = bool(camera_settings.get("flip_horizontal", False))
                fv = bool(camera_settings.get("flip_vertical", False))
                if fh and fv:
                    image = cv2.flip(image, FLIP_BOTH)
                elif fh:
                    image = cv2.flip(image, FLIP_HORIZONTAL)
                elif fv:
                    image = cv2.flip(image, FLIP_VERTICAL)

                # --- 9. rotate ---
                rot = int(camera_settings.get("rotate", 0))
                if rot == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rot == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif rot == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                # --- 10. crop and resize ---
                image = _crop_and_resize(image, camera_settings)

                # --- 11. encode raw ---
                if demand_raw:
                    try:
                        ok, buf = cv2.imencode(
                            ".jpg", image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(jpeg_quality_raw, 100))],
                        )
                        if ok:
                            seq_raw += 1
                            rings["raw"].write(
                                data=bytes(buf),
                                timestamp=now_ts,
                                mode=MODE_RAW,
                                camera_index=active_index,
                                sequence=seq_raw,
                            )
                            fps_raw.tick(now_ts)
                    except Exception as exc:
                        print(f"[Worker] JPEG encode raw error: {exc}")

                # --- 12. mask (if needed by masked or AI) ---
                image_masked: np.ndarray | None = None
                if demand_masked or demand_ai:
                    try:
                        image_masked = _mask_image(image, camera_settings)
                    except Exception as exc:
                        print(f"[Worker] mask error: {exc}")
                        image_masked = image

                if demand_masked and image_masked is not None:
                    try:
                        ok, buf = cv2.imencode(
                            ".jpg", image_masked,
                            [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(jpeg_quality_masked, 100))],
                        )
                        if ok:
                            seq_masked += 1
                            rings["masked"].write(
                                data=bytes(buf),
                                timestamp=time.time(),
                                mode=MODE_MASKED,
                                camera_index=active_index,
                                sequence=seq_masked,
                            )
                            fps_masked.tick(now_ts)
                    except Exception as exc:
                        print(f"[Worker] JPEG encode masked error: {exc}")

                # --- 13. AI pipeline ---
                if demand_ai and image_masked is not None:
                    try:
                        ai_frame, raw_pose = _run_ai_inference(image_masked, global_settings)
                    except Exception as exc:
                        print(f"[Worker] AI inference error: {exc}")
                        ai_frame, raw_pose = image_masked, None

                    # send raw pose to main (for engagement state machine)
                    ai_image_seq += 1
                    ai_h, ai_w = ai_frame.shape[:2]
                    try:
                        with pose_send_lock:
                            self._pose_conn.send({
                                "type": "pose",
                                "pose": raw_pose,
                                "seq": ai_image_seq,
                                "camera_index": active_index,
                                "camera_code": active_camera_code,
                                "time": now_ts,
                                "image_width": int(ai_w),
                                "image_height": int(ai_h),
                                "fps": _fps_snapshot(now_ts),
                            })
                    except (BrokenPipeError, OSError):
                        return

                    # Translate the *fresh* raw pose (same frame we're about to
                    # draw on) into the dict form used by the overlay.  Using
                    # the pose from this very frame avoids the 1-5 frame lag
                    # that would happen if we waited for main to round-trip
                    # its engagement snapshot back to us.
                    ai_setup_for_draw = (
                        global_settings.get("aiSetup") or {}
                    ) if isinstance(global_settings, dict) else {}
                    fresh_pose_dict: Any = None
                    if raw_pose:
                        try:
                            fresh_pose_dict = translate_raw_pose_to_pose_dict(
                                raw_pose=raw_pose, ai_setup=ai_setup_for_draw,
                            )
                        except Exception as exc:
                            print(f"[Worker] pose translate error: {exc}")
                            fresh_pose_dict = None

                    # overlay engagement state using the latest result from main,
                    # but always draw circles from the *fresh* pose of this frame.
                    if last_engagement is not None:
                        try:
                            draw_engagement_overlay(
                                image=ai_frame,
                                status_value=str(last_engagement.get("status_value", "not_engaging")),
                                engagement_counter=int(last_engagement.get("engagement_counter", 0)),
                                fps=fps_ai.value(now_ts),
                                pose=fresh_pose_dict,
                                ai_setup=ai_setup_for_draw,
                                draw_ai_stats=bool(last_engagement.get("draw_ai_stats", True)),
                            )
                        except Exception as exc:
                            print(f"[Worker] overlay error: {exc}")
                    elif fresh_pose_dict is not None:
                        # Main hasn't produced an engagement snapshot yet (e.g.
                        # the first AI frame after start-up).  Still show the
                        # detected pose immediately so the user doesn't see a
                        # "loading" pause.
                        try:
                            draw_pose_overlay(
                                image=ai_frame,
                                pose=fresh_pose_dict,
                                ai_setup=ai_setup_for_draw,
                            )
                        except Exception as exc:
                            print(f"[Worker] pose-only draw error: {exc}")

                    try:
                        ok, buf = cv2.imencode(
                            ".jpg", ai_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(jpeg_quality_ai, 100))],
                        )
                        if ok:
                            seq_ai += 1
                            rings["ai"].write(
                                data=bytes(buf),
                                timestamp=time.time(),
                                mode=MODE_AI,
                                camera_index=active_index,
                                sequence=seq_ai,
                            )
                            fps_ai.tick(now_ts)
                    except Exception as exc:
                        print(f"[Worker] JPEG encode ai error: {exc}")

                time.sleep(EPSILON_DELAY)

        except Exception as exc:
            print(f"[Worker] Fatal error: {exc}")
        finally:
            heartbeat_stop.set()
            try:
                heartbeat_thread.join(timeout=0.5)
            except Exception:
                pass
            _cleanup()


# =========================================================================
# Image processing helpers (run inside the worker process)
# =========================================================================

def _crop_and_resize(image: np.ndarray, settings: dict) -> np.ndarray:
    import cv2

    crop_top = settings.get("crop_top", 0.0)
    crop_left = settings.get("crop_left", 0.0)
    crop_bottom = settings.get("crop_bottom", 0.0)
    crop_right = settings.get("crop_right", 0.0)

    h, w = image.shape[:2]
    ct = int(round(crop_top * h))
    cl = int(round(crop_left * w))
    cb = int(round(crop_bottom * h))
    cr = int(round(crop_right * w))

    sx, sy = cl, ct
    sw = w - cr - cl
    sh = h - cb - ct

    stretch_enabled = settings.get("stretch_enabled", False)
    stretch_w = settings.get("stretch_width", 0)
    stretch_h = settings.get("stretch_height", 0)
    tw = stretch_w if stretch_enabled else sw
    th = stretch_h if stretch_enabled else sh

    if sx < 0 or sy < 0 or sw <= 0 or sh <= 0 or sx + sw > w or sy + sh > h:
        return image

    return cv2.resize(image[sy:sy + sh, sx:sx + sw], (tw, th))


def _mask_image(image: np.ndarray, settings: dict) -> np.ndarray:
    import cv2
    from .common import POLYGONS_ASPECT_RATIO

    mask_polygons = settings.get("mask_polygons", [])
    if not mask_polygons:
        return image

    height, width = image.shape[:2]
    image_ar = width / height
    polygons_ar = POLYGONS_ASPECT_RATIO
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in mask_polygons:
        points = []
        for pt in polygon:
            if image_ar > polygons_ar:
                xp = int(pt["x"] * width)
                yp = pt["y"] - 0.5
                yp = yp / polygons_ar * image_ar + 0.5
                yp = int(yp * height)
            else:
                xp = pt["x"] - 0.5
                xp = xp * polygons_ar / image_ar + 0.5
                xp = int(xp * width)
                yp = int(pt["y"] * height)
            points.append([xp, yp])
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    return cv2.bitwise_and(image, image, mask=mask)


def _run_ai_inference(
    image: np.ndarray,
    global_settings: dict,
) -> tuple[np.ndarray, list[dict] | None]:
    """Run YOLO inference and return ``(image, raw_pose_or_None)``.

    ``raw_pose`` contains only plain Python types (dicts / tuples / floats)
    so it survives pickling through the pose pipe.  The image is returned
    unchanged — organ circle / engagement overlays are applied afterwards
    in :func:`aidraw.draw_engagement_overlay` using the engagement result
    received from the main process via the ``ai_result_pipe``.
    """
    from .yolomodels import YOLOModels
    from .ai_setup_constants import DEFAULT_DEVICE
    from .cameraai import get_pose_dict

    ai_setup = global_settings.get("aiSetup", {}) if isinstance(global_settings, dict) else {}
    model_name = ai_setup.get("modelName", YOLOModels().default_model_name)
    device = ai_setup.get("device", DEFAULT_DEVICE)

    try:
        results = YOLOModels().predict(
            model_name=model_name,
            preferred_device=device,
            image=image,
            verbose=False,
        )
        result = results[0]

        if hasattr(result, "keypoints") and result.keypoints is not None:
            kpts_data = getattr(result.keypoints, "data", result.keypoints)
            keypoints = kpts_data.cpu().numpy() if hasattr(kpts_data, "cpu") else kpts_data
            raw_pose = get_pose_dict(keypoints=keypoints, ai_setup=ai_setup)
            return image, raw_pose
    except Exception as e:
        print(f"[Worker] AI inference error: {e}")

    return image, None
