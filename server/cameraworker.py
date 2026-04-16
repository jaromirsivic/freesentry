"""
Camera worker process — runs capture + processing in its own OS process / GIL.

Created via the explicit ``spawn`` multiprocessing context so the behaviour
is identical across Linux, macOS, and Windows.

IMPORTANT: This module NEVER imports ``settingscontroller``.
All settings arrive through the ``command_pipe`` from the main process.
"""

from __future__ import annotations

import atexit
import multiprocessing
import multiprocessing.shared_memory
import time
from typing import Any

import numpy as np

_mp_ctx = multiprocessing.get_context("spawn")

EPSILON_DELAY = 0.005
HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_TIMEOUT = 15.0

FLIP_HORIZONTAL = 1
FLIP_VERTICAL = 0
FLIP_BOTH = -1


class CameraWorkerProcess(_mp_ctx.Process):
    """Captures frames, applies transforms / AI, writes to shared memory."""

    def __init__(
        self,
        *,
        camera_type: str,
        camera_index: int,
        index: int,
        camera_code: str | None,
        camera_name: str,
        initial_camera_settings: dict,
        initial_global_settings: dict,
        shm_names: tuple[str, str, str],
        shm_max_data_size: int,
        slot_locks: tuple[Any, Any, Any],
        pose_pipe_conn: Any,
        command_pipe_conn: Any,
    ) -> None:
        super().__init__(daemon=True)
        self._camera_type = camera_type
        self._camera_index = camera_index
        self._index = index
        self._camera_code = camera_code
        self._camera_name = camera_name
        self._initial_camera_settings = initial_camera_settings
        self._initial_global_settings = initial_global_settings
        self._shm_names = shm_names
        self._shm_max_data_size = shm_max_data_size
        self._slot_locks = slot_locks
        self._pose_conn = pose_pipe_conn
        self._command_conn = command_pipe_conn

    def run(self) -> None:
        import cv2
        from .cameradevice import create_camera_device
        from .cameraframetransport import SharedFrameSlot

        device = create_camera_device(
            camera_type=self._camera_type,
            camera_index=self._camera_index,
            camera_name=self._camera_name,
            settings=self._initial_camera_settings,
        )

        slot_raw = SharedFrameSlot(
            name=self._shm_names[0],
            max_data_size=self._shm_max_data_size,
            create=False,
            lock=self._slot_locks[0],
        )
        slot_masked = SharedFrameSlot(
            name=self._shm_names[1],
            max_data_size=self._shm_max_data_size,
            create=False,
            lock=self._slot_locks[1],
        )
        slot_ai = SharedFrameSlot(
            name=self._shm_names[2],
            max_data_size=self._shm_max_data_size,
            create=False,
            lock=self._slot_locks[2],
        )

        def _cleanup():
            slot_raw.close()
            slot_masked.close()
            slot_ai.close()
            try:
                self._pose_conn.close()
            except Exception:
                pass
            try:
                self._command_conn.close()
            except Exception:
                pass
            device.close()

        atexit.register(_cleanup)

        camera_settings: dict = dict(self._initial_camera_settings)
        global_settings: dict = dict(self._initial_global_settings)

        demand_raw = True
        demand_masked = False
        demand_ai = False

        seq_raw = 0
        seq_masked = 0
        seq_ai = 0

        last_heartbeat_recv = time.monotonic()
        last_heartbeat_send = 0.0

        if not device.open():
            print(f"[Worker] Failed to open camera {self._camera_name}")
            _cleanup()
            return

        try:
            while True:
                now_mono = time.monotonic()

                # --- process incoming commands (non-blocking) ---
                try:
                    while self._command_conn.poll(0):
                        msg = self._command_conn.recv()
                        cmd = msg.get("cmd")
                        if cmd == "stop":
                            return
                        elif cmd == "update_camera_settings":
                            camera_settings = msg["settings"]
                            device.set_properties(camera_settings)
                        elif cmd == "update_global_settings":
                            global_settings = msg["settings"]
                        elif cmd == "update_demand":
                            demand_raw = msg.get("raw", demand_raw)
                            demand_masked = msg.get("masked", demand_masked)
                            demand_ai = msg.get("ai", demand_ai)
                        elif cmd == "heartbeat":
                            last_heartbeat_recv = now_mono
                except (EOFError, OSError):
                    return

                # --- heartbeat timeout check ---
                if now_mono - last_heartbeat_recv > HEARTBEAT_TIMEOUT:
                    print(f"[Worker] Heartbeat timeout for {self._camera_name}")
                    return

                # --- send heartbeat back ---
                if now_mono - last_heartbeat_send >= HEARTBEAT_INTERVAL:
                    try:
                        self._pose_conn.send({"type": "heartbeat", "time": time.time()})
                        last_heartbeat_send = now_mono
                    except (BrokenPipeError, OSError):
                        return

                # --- capture ---
                valid, image = device.get_image()
                if not valid or image is None:
                    device.close()
                    time.sleep(0.5)
                    if not device.open():
                        time.sleep(1.0)
                        continue
                    continue

                # --- flip ---
                fh = camera_settings.get("flip_horizontal", False)
                fv = camera_settings.get("flip_vertical", False)
                if fh and fv:
                    image = cv2.flip(image, FLIP_BOTH)
                elif fh:
                    image = cv2.flip(image, FLIP_HORIZONTAL)
                elif fv:
                    image = cv2.flip(image, FLIP_VERTICAL)

                # --- rotate ---
                rot = camera_settings.get("rotate", 0)
                if rot == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rot == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif rot == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                # --- crop and resize ---
                image = _crop_and_resize(image, camera_settings)

                # --- raw slot ---
                if demand_raw:
                    now_ts = time.time()
                    seq_raw += 1
                    slot_raw.write_frame(image, now_ts, True, seq_raw)

                # --- masked slot ---
                if demand_masked:
                    image_masked = _mask_image(image, camera_settings)
                    seq_masked += 1
                    slot_masked.write_frame(image_masked, time.time(), True, seq_masked)
                else:
                    image_masked = None

                # --- AI slot ---
                if demand_ai:
                    try:
                        source_for_ai = image_masked if image_masked is not None else _mask_image(image, camera_settings)
                        ai_frame, raw_pose = _run_ai_inference(source_for_ai, global_settings)
                        seq_ai += 1
                        slot_ai.write_frame(ai_frame, time.time(), True, seq_ai)
                        try:
                            self._pose_conn.send({"type": "pose", "pose": raw_pose, "seq": seq_ai})
                        except (BrokenPipeError, OSError):
                            return
                    except Exception as e:
                        print(f"[Worker] AI processing error for {self._camera_name}: {e}")

                time.sleep(EPSILON_DELAY)

        except Exception as e:
            print(f"[Worker] Fatal error in {self._camera_name}: {e}")
        finally:
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
    """Run YOLO inference and draw pose on a copy of the image.

    Returns ``(drawn_image, raw_pose_or_None)`` where *raw_pose* contains
    only plain Python types (dicts, tuples, floats) so it can be safely
    pickled through a multiprocessing Pipe.
    """
    from .yolomodels import YOLOModels
    from .ai_setup_constants import DEFAULT_DEVICE
    from .cameraai import draw_pose, get_pose_dict, translate_raw_pose_to_pose_dict

    ai_setup = global_settings.get("aiSetup", {})
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
            pose = translate_raw_pose_to_pose_dict(raw_pose=raw_pose, ai_setup=ai_setup)
            draw_stats = ai_setup.get("drawAiStats", True)
            drawn = draw_pose(image=image, pose=pose, ai_setup=ai_setup) if draw_stats else image.copy()
            # Return raw_pose (plain dicts/tuples/floats) — NOT pose
            # (which contains AICircle Pydantic objects that may fail to pickle).
            return drawn, raw_pose
    except Exception as e:
        print(f"[Worker] AI inference error: {e}")

    return image.copy(), None
