"""
REST API for Camera management.
Provides endpoints to list, update, and stream camera feeds.
"""
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import time

from .camera import Camera
from .common import EPSILON_DELAY, _make_loading_jpeg
from . import settingscontroller
from .context import get_master_controller
from .settingserrors import SettingsError

if TYPE_CHECKING:
    from .mastercontroller import MasterController

router = APIRouter()


class CameraUpdateRequest(BaseModel):
    """Request model for updating camera settings."""
    index: int
    name: str = ""
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    hue: float = 0.0
    saturation: float = 0.0
    sharpness: float = 0.0
    gamma: float = 0.0
    white_balance_temperature: float = 0.0
    backlight: float = 0.0
    gain: float = 0.0
    focus: float = 0.0
    exposure: float = 0.0
    auto_white_balance_temperature: bool = False
    auto_focus: bool = False
    auto_exposure: bool = False
    crop_top: float = 0.0
    crop_left: float = 0.0
    crop_bottom: float = 0.0
    crop_right: float = 0.0
    stretch_enabled: bool = False
    stretch_width: int = 0
    stretch_height: int = 0
    static_reticle_x: float = 0.0
    static_reticle_y: float = 0.0
    static_reticle_color: str = "#88ff00cc"
    static_reticle_outline: str = "#009900cc"
    static_reticle_size: float = 0.0
    mask_polygons: List[List[Dict[str, float]]] = []


# Camera code to panel name mapping
CAMERA_NAMES = {
    "scope_camera": "Scope Camera",
    "spotter_camera1": "Spotter Camera 1",
    "spotter_camera2": "Spotter Camera 2",
    "spotter_camera3": "Spotter Camera 3"
}


def _get_configured_camera_index(
    *,
    settings: Mapping[str, Any],
    camera_code: str,
) -> int | None:
    cameras_config = settings.get("cameras", {})
    if not isinstance(cameras_config, Mapping):
        return None

    camera_config = cameras_config.get(camera_code)
    if not isinstance(camera_config, Mapping):
        return None

    camera_index = camera_config.get("index")
    if not isinstance(camera_index, int) or isinstance(camera_index, bool):
        return None
    if camera_index < 0:
        return None
    return camera_index


def _get_camera_from_snapshot(
    *,
    cameras: list[Camera],
    camera_index: int,
) -> Camera:
    if camera_index < 0 or camera_index >= len(cameras):
        raise HTTPException(
            status_code=404,
            detail=f"Camera index {camera_index} not found"
        )
    return cameras[camera_index]


def _get_bound_camera_from_settings(
    *,
    settings: Mapping[str, Any],
    cameras: list[Camera],
    camera_code: str,
) -> tuple[int, Camera]:
    camera_index = _get_configured_camera_index(
        settings=settings,
        camera_code=camera_code,
    )
    if camera_index is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_code}' is not assigned to a live camera index",
        )

    camera = _get_camera_from_snapshot(cameras=cameras, camera_index=camera_index)
    return camera_index, camera


def _build_input_device_info(camera: Camera) -> dict[str, Any]:
    device_info = dict(camera.settings)
    camera_type = getattr(getattr(camera, "_camera_type", None), "value", None)
    supported_resolutions = getattr(camera, "supported_resolutions", []) or []

    if not isinstance(device_info.get("index"), int) or isinstance(device_info.get("index"), bool):
        device_info["index"] = getattr(camera, "_index", None)
    if not isinstance(device_info.get("camera_index"), int) or isinstance(device_info.get("camera_index"), bool):
        device_info["camera_index"] = getattr(camera, "_camera_index", None)
    if not isinstance(device_info.get("camera_type"), str) or not device_info["camera_type"]:
        device_info["camera_type"] = camera_type
    if not isinstance(device_info.get("name"), str) or not device_info["name"].strip():
        device_info["name"] = camera.camera_name
    if not isinstance(device_info.get("supported_resolutions"), list) or not device_info["supported_resolutions"]:
        device_info["supported_resolutions"] = supported_resolutions

    device_info["capabilities"] = camera.capabilities
    return device_info


@router.get("/api/cameras/list")
async def get_cameras_list():
    """
    Get cameras configuration from settings.json.
    Returns the cameras object with all three camera configurations.
    """
    try:
        settings = await settingscontroller.get_settings()
        cameras = settings.get("cameras", {})
        return {"success": True, "cameras": cameras}
    except SettingsError:
        raise
    except Exception as e:
        print(f"Error getting cameras list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/cameras/input_devices")
async def get_input_devices(
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Get list of available input devices from CamerasController.
    Returns device info including index, name, and supported resolutions.
    """
    try:
        input_devices = []

        cameras = master_controller.cameras_controller.cameras
        for camera in cameras:
            device_info = _build_input_device_info(camera)
            input_devices.append(device_info)
        
        return {"success": True, "input_devices": input_devices}
        
    except Exception as e:
        print(f"Error getting input devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/update/{camera_code}")
async def update_camera(
    *,
    camera_code: str,
    request: CameraUpdateRequest,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Update camera settings in settings.json and reset CamerasController.
    
    Parameters:
        camera_code: One of 'scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'
        request: Camera settings to update
    """
    if camera_code not in CAMERA_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_code. Must be one of: {list(CAMERA_NAMES.keys())}"
        )
    
    try:
        new_index = request.index
        cameras_list = master_controller.cameras_controller.cameras
        camera = _get_camera_from_snapshot(cameras=cameras_list, camera_index=new_index)
        updated_camera_config = request.model_dump()
        updated_camera_config["supported_resolutions"] = camera.supported_resolutions

        def update_camera_settings(current_settings: dict[str, Any]) -> int | None:
            cameras = current_settings.setdefault("cameras", {})
            stored_camera_config = cameras.get(camera_code, {})
            old_index = stored_camera_config.get("index")

            for other_code, other_config in cameras.items():
                if other_code != camera_code and other_config.get("index") == new_index:
                    other_config["index"] = -1

            next_camera_config = dict(stored_camera_config)
            next_camera_config.update(updated_camera_config)
            cameras[camera_code] = next_camera_config
            return old_index

        old_index = await settingscontroller.update_settings(update_camera_settings)
        # update the camera settings
        camera.settings = {
            **camera.settings,
            **updated_camera_config,
        }

        # Update Camera instances: old device gets camera_code None, new device gets this camera_code
        n = len(cameras_list)
        if old_index is not None and old_index != new_index and 0 <= old_index < n:
            cameras_list[old_index].camera_code = None
        if 0 <= new_index < n:
            cameras_list[new_index].camera_code = camera_code

        return {"success": True}

    except SettingsError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/reset/{camera_code}")
async def reset_camera(
    *,
    camera_code: str,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Reset camera settings to default values.
    
    Parameters:
        camera_code: One of 'scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'
    """
    if camera_code not in CAMERA_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_code. Must be one of: {list(CAMERA_NAMES.keys())}"
        )

    try:
        current_settings = await settingscontroller.get_settings()
        controller_cameras = master_controller.cameras_controller.cameras
        _, camera = _get_bound_camera_from_settings(
            settings=current_settings,
            cameras=controller_cameras,
            camera_code=camera_code,
        )
        camera.reset_settings()

        def update_reset_camera_settings(settings: dict[str, Any]) -> None:
            cameras = settings.setdefault("cameras", {})
            stored_camera_settings = cameras.setdefault(camera_code, {})
            stored_camera_settings.update(camera.settings)

        await settingscontroller.update_settings(update_reset_camera_settings)
        return {"success": True}

    except SettingsError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error resetting camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/resetall")
async def reset_all_cameras(
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Reset all cameras by calling master_controller.cameras_controller.reset().
    Reloads all cameras from scratch.
    """
    try:
        await asyncio.to_thread(
            master_controller.cameras_controller.reset,
            reset_to_default=True,
        )
        await settingscontroller.clear_cached_settings()
        for camera_name in CAMERA_NAMES:
            await reset_camera(camera_code=camera_name, master_controller=master_controller)
        return {"success": True}
    except SettingsError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error resetting all cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/stop/{camera_code}")
async def stop_camera(
    *,
    camera_code: str,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Stop camera streaming.
    
    Parameters:
        camera_code: One of 'scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'
    """
    if camera_code not in CAMERA_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_code. Must be one of: {list(CAMERA_NAMES.keys())}"
        )
    
    try:
        current_settings = await settingscontroller.get_settings()
        controller_cameras = master_controller.cameras_controller.cameras
        camera_index, _ = _get_bound_camera_from_settings(
            settings=current_settings,
            cameras=controller_cameras,
            camera_code=camera_code,
        )

        await asyncio.to_thread(
            master_controller.cameras_controller.stop_camera,
            index=camera_index,
        )

        return {"success": True}

    except SettingsError:
        raise
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error stopping camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# MJPEG stream stall watchdog
# ----------------------------------------------------------------------
#
# Tunables -- all in seconds.  Documented in the plan at
# .cursor/plans/generic_stream_stall_watchdog_03468505.plan.md.
#
# * _WATCHDOG_STALL_GRACE: no action until the stream has been "not
#   live" (Loading or no-change) for at least this long.
# * _WATCHDOG_GENTLE_PERIOD: rate-limit on gentle-recovery invocations
#   (re-touch demand + bump settings version).
# * _WATCHDOG_HARD_THRESHOLD: escalate from gentle to hard recovery
#   (deactivate + rotate stream token) at this stall age.
# * _WATCHDOG_HARD_COOLDOWN: minimum gap between two hard-recovery
#   invocations on the same connection, to avoid thrashing when the
#   root cause is external (camera unplugged, worker crashed, etc.).
_WATCHDOG_STALL_GRACE = 1.5
_WATCHDOG_GENTLE_PERIOD = 2.01
_WATCHDOG_HARD_THRESHOLD = 4.01
_WATCHDOG_HARD_COOLDOWN = 7.01


def _watchdog_worker_legitimately_not_ready(
    *,
    status: dict | None,
    status_age: float,
    mode: int,
    camera_index: int,
) -> bool:
    """Return True when the worker reports a state that *legitimately*
    explains why no frames are arriving (device still opening, YOLO
    warming up, capture currently failing, select still propagating).

    In those cases the watchdog should *not* take recovery action: the
    stall is expected and the stream will resume on its own once the
    worker finishes what it is doing.
    """
    if status is None or status_age > 3.0:
        # No fresh status -> don't assume desync; give the worker more
        # time.  (If the worker is truly dead the existing heartbeat-
        # stale detector in CamerasController handles it.)
        return True
    if int(status.get("active_camera_index", -1)) != int(camera_index):
        # Worker is not on our camera yet; select_camera is still
        # propagating through the command pipe.
        return True
    device_state = status.get("device_state", "closed")
    if device_state in ("closed", "opening", "failed"):
        return True
    now_mono = time.monotonic()
    select_applied = float(status.get("select_applied_mono", 0.0) or 0.0)
    if select_applied and (now_mono - select_applied) < 1.5:
        # Device was just opened; give capture/encoding a moment to
        # produce the first frames of the new camera.
        return True
    last_fail = float(status.get("last_capture_fail_mono", 0.0) or 0.0)
    last_ok = float(status.get("last_capture_ok_mono", 0.0) or 0.0)
    if last_fail and last_fail > last_ok and (now_mono - last_fail) < 2.0:
        # Recent capture failure (USB hiccup or similar); worker is
        # retrying on its own.
        return True
    if mode == 3:
        started = float(status.get("ai_inference_started_mono", 0.0) or 0.0)
        completed = float(status.get("ai_inference_completed_mono", 0.0) or 0.0)
        if started > completed and (now_mono - started) > 0.5:
            # AI inference has been blocked for >500 ms -- YOLO is
            # loading / first-inference warmup.
            return True
        if status.get("ai_model_state") in ("not_loaded", "loading"):
            return True
    return False


def _watchdog_gentle_recovery(*, camera: Camera, mode: int) -> None:
    """Gentle stage: force the demand flags back through to the worker
    and bump the camera-settings version so the proxy thread re-sends
    ``update_camera_settings`` on its next iteration.  Idempotent and
    cheap enough to invoke every ~1 s while a stall persists.
    """
    try:
        camera._touch_access(mode=mode)
    except Exception as exc:
        print(f"[Watchdog] gentle recovery touch_access failed: {exc}")
    try:
        camera.settings = camera.settings  # bump _settings_version
    except Exception as exc:
        print(f"[Watchdog] gentle recovery settings bump failed: {exc}")


def _watchdog_hard_recovery(
    *,
    master_controller: "MasterController",
    camera: Camera,
    mode: int,
    last_seq_by_mode: dict[str, int],
) -> int:
    """Hard stage: tear down the camera's proxy-thread and current
    worker selection, then rotate the stream token in place so the
    generator loop can transparently resume on the same HTTP
    connection.  The next ``get_stream_frame`` call will re-enter
    ``_ensure_active`` and re-issue ``select_camera`` to the worker.

    Returns the freshly-created stream token that the generator must
    start using.
    """
    print(
        f"[Watchdog] hard recovery on camera index={camera.index} "
        f"camera_code={camera.camera_code} mode={mode}"
    )
    # Pre-touch the REQUESTED mode BEFORE we tear down the proxy
    # thread.  This is what prevents the recovery itself from
    # re-triggering the update_demand(masked=False) race described in
    # discussion.md: if we skipped this, the new proxy thread's init
    # ``_touch_access(mode=0)`` in ``Camera.run`` would run while
    # ``_last_access_time_masked`` is still 0 and would send
    # ``update_demand(masked=False)`` to the worker right after
    # ``select_camera`` -- exactly the bug the watchdog is supposed to
    # recover from.  Setting the timestamps here means every
    # subsequent ``update_demand`` from the restarted thread carries
    # the correct demand flags for the mode we are actually streaming.
    try:
        camera._touch_access(mode=mode)
    except Exception:
        pass
    try:
        master_controller.cameras_controller.deactivate_camera(camera)
    except Exception as exc:
        print(f"[Watchdog] deactivate_camera failed: {exc}")
    try:
        camera.stop()
    except Exception as exc:
        print(f"[Watchdog] camera.stop failed: {exc}")
    for key in last_seq_by_mode:
        last_seq_by_mode[key] = 0
    return camera.create_stream_token()


async def generate_camera_frames(
    *,
    index: int,
    master_controller: "MasterController",
    mode: int = 0,
    quality: int = 90,
):
    """
    Generator that yields MJPEG frames from the camera.
    Uses multipart/x-mixed-replace for browser-native streaming.
    """
    match mode:
        case 0:
            mode_str = "mode=0 (Raw)"
        case 1:
            mode_str = "mode=1 (Masked)"
        case 3:
            mode_str = "mode=3 (Masked + AI)"
        case _:
            mode_str = f"mode={mode} (Unknown)"
    print(f"Streaming camera {index} in {mode_str}")
    time_of_last_loading_sent = 0.0
    last_sent_was_loading = False
    cameras = master_controller.cameras_controller.cameras
    if index < 0 or index >= len(cameras):
        return
    expected_camera = cameras[index]
    stream_token = expected_camera.create_stream_token()
    # Per-connection "last sequence handed to this browser", one cursor
    # per ring.  Starts at 0 on every new HTTP request, so a
    # freshly-mounted <img> (mode switch + Apply, first page load, ...)
    # never inherits a stale counter from a previous session.  Separate
    # cursors per ring are what let Camera.get_stream_frame fall back
    # to a lower-mode ring (e.g. "masked" while "ai" is still
    # loading the YOLO model) without re-emitting frames the browser
    # has already seen and without dropping live frames on the ring
    # it eventually switches back to.
    last_seq_by_mode: dict[str, int] = {"raw": 0, "masked": 0, "ai": 0}
    # Map MODE_RAW / MODE_MASKED / MODE_AI codes to ring keys so we can
    # update the right cursor from the frame the generator actually
    # served (which may differ from the mode the browser requested
    # during the AI-warmup fallback window).
    _MODE_NUM_TO_KEY: dict[int, str] = {0: "raw", 1: "masked", 3: "ai"}
    # --- Stream stall watchdog state (per connection) ---------------
    # ``stall_since_mono`` is set the first iteration the generator
    # sees something other than a live frame (Loading placeholder or
    # no-change sentinel) and cleared back to ``None`` as soon as a
    # fresh frame yields.  The recovery clocks rate-limit each stage
    # so we don't thrash.
    stall_since_mono: float | None = None
    last_gentle_recovery_mono: float = 0.0
    last_hard_recovery_mono: float = 0.0
    # Emit the real "Loading, please wait a minute..." JPEG as the very
    # first multipart part so the browser immediately renders something
    # instead of a blank <img>.  The main loop below will replace it
    # with live frames as soon as the worker publishes them.
    try:
        loading_bytes = _make_loading_jpeg()
        if loading_bytes:
            time_of_last_loading_sent = time.time()
            last_sent_was_loading = True
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + loading_bytes + b'\r\n'
            )
    except Exception as exc:
        print(f"Error emitting initial Loading JPEG for camera {index}: {exc}")
    while True:
        try:
            now = time.time()
            cameras = master_controller.cameras_controller.cameras
            if index < 0 or index >= len(cameras):
                break
            camera = cameras[index]
            if camera is not expected_camera:
                break
            # Worker already JPEG-encoded at the requested quality; we only
            # forward the bytes.  Frame semantics:
            #   frame is None          -> stream token stale, stop
            #   frame.valid            -> fresh JPEG, yield it
            #   not valid and no data  -> no-change sentinel, skip
            #   not valid and data     -> real "Loading..." placeholder,
            #                             rate-limit to avoid flicker
            frame = camera.get_stream_frame(
                mode=mode, stream_token=stream_token, quality=quality,
                since_by_mode=last_seq_by_mode,
            )
            if frame is None:
                break

            if frame.valid and frame.data:
                last_sent_was_loading = False
                # Live frame -> clear any in-flight stall state.
                stall_since_mono = None
                # Update the cursor for the ring that actually served
                # this frame (mode=3 may be getting masked/raw frames
                # while YOLO warms up).  Ignore unknown modes defensively.
                served_key = _MODE_NUM_TO_KEY.get(int(frame.mode))
                if served_key is not None:
                    last_seq_by_mode[served_key] = int(frame.sequence)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame.data + b'\r\n'
                )
            elif frame.data:
                # Real Loading placeholder.  Emit once when we transition
                # into the loading state, then refresh at most every 0.5s
                # to keep the client connection warm without flickering.
                if stall_since_mono is None:
                    stall_since_mono = time.monotonic()
                if (
                    not last_sent_was_loading
                    or now - time_of_last_loading_sent > 0.5
                ):
                    time_of_last_loading_sent = now
                    last_sent_was_loading = True
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame.data + b'\r\n'
                    )
            else:
                # No-change sentinel -- the ring has frames, just none
                # newer than the browser already has.  Normally this is
                # just the inter-frame gap, but if it persists the
                # watchdog below needs to notice.
                if stall_since_mono is None:
                    stall_since_mono = time.monotonic()

            # --- Stream stall watchdog --------------------------------
            # Classify + escalate only if we've been non-live for at
            # least the grace window.  Consults the worker's 1 Hz
            # status heartbeat to distinguish legitimate not-ready
            # states (device opening, YOLO warmup, capture failing)
            # from real pipeline desync that recovery can fix.
            if stall_since_mono is not None:
                stall_age = time.monotonic() - stall_since_mono
                if stall_age >= _WATCHDOG_STALL_GRACE:
                    status, status_age = (
                        master_controller.cameras_controller.get_worker_status()
                    )
                    if not _watchdog_worker_legitimately_not_ready(
                        status=status,
                        status_age=status_age,
                        mode=mode,
                        camera_index=camera.index,
                    ):
                        if stall_age < _WATCHDOG_HARD_THRESHOLD:
                            if (
                                time.monotonic() - last_gentle_recovery_mono
                                >= _WATCHDOG_GENTLE_PERIOD
                            ):
                                _watchdog_gentle_recovery(camera=camera, mode=mode)
                                last_gentle_recovery_mono = time.monotonic()
                        else:
                            if (
                                time.monotonic() - last_hard_recovery_mono
                                >= _WATCHDOG_HARD_COOLDOWN
                            ):
                                stream_token = _watchdog_hard_recovery(
                                    master_controller=master_controller,
                                    camera=camera,
                                    mode=mode,
                                    last_seq_by_mode=last_seq_by_mode,
                                )
                                # Re-point expected_camera to the same
                                # proxy so the identity check above
                                # keeps passing (the Camera object
                                # survives .stop()).
                                expected_camera = camera
                                last_hard_recovery_mono = time.monotonic()
                                # Restart the stall clock so we don't
                                # immediately re-trigger a second hard
                                # recovery before the freshly-restarted
                                # worker pipeline has a chance to
                                # produce frames.
                                stall_since_mono = time.monotonic()

            await asyncio.sleep(EPSILON_DELAY)

        except Exception as e:
            print(f"Error streaming camera {index}: {e}")
            await asyncio.sleep(EPSILON_DELAY)


@router.get("/api/cameras/stream/{item_code}")
async def stream_camera(
    *,
    item_code: str,
    mode: int = 0,
    quality: int = 80,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Stream camera feed as MJPEG.
    
    Parameters:
        item_code: One of 'scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'
        mode: Display mode (0=Raw, 1=Masked, 3=Masked+AI)
        quality: JPEG quality (35-100)
    """
    if item_code not in CAMERA_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid item_code. Must be one of: {list(CAMERA_NAMES.keys())}"
        )
    
    try:
        settings = await settingscontroller.get_settings()
        controller_cameras = master_controller.cameras_controller.cameras
        camera_index, camera = _get_bound_camera_from_settings(
            settings=settings,
            cameras=controller_cameras,
            camera_code=item_code,
        )
        # Refresh the camera settings; otherwise after a resolution change the
        # stream can stay zoomed until the next full refresh.
        camera.settings = camera.settings

        return StreamingResponse(
            generate_camera_frames(
                index=camera_index,
                master_controller=master_controller,
                mode=mode,
                quality=quality - 10,
            ),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )

    except SettingsError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error streaming camera {item_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

