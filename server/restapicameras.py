"""
REST API for Camera management.
Provides endpoints to list, update, and stream camera feeds.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import time
import cv2

from .camera import Camera
from .common import EPSILON_DELAY
from . import settingscontroller
from .context import master_controller

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
    except Exception as e:
        print(f"Error getting cameras list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/cameras/input_devices")
async def get_input_devices():
    """
    Get list of available input devices from CamerasController.
    Returns device info including index, name, and supported resolutions.
    """
    try:
        input_devices = []
        
        for camera in master_controller.cameras_controller.cameras:
            device_info = camera.settings
            device_info["capabilities"] = camera.capabilities
            input_devices.append(device_info)
        
        return {"success": True, "input_devices": input_devices}
        
    except Exception as e:
        print(f"Error getting input devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/update/{camera_code}")
async def update_camera(*, camera_code: str, request: CameraUpdateRequest):
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
        # Get current settings
        current_settings = await settingscontroller.get_settings()
        
        # Ensure cameras object exists
        if "cameras" not in current_settings:
            current_settings["cameras"] = {}
        
        # Capture previous device index for this camera_code (before we overwrite)
        camera_config = current_settings["cameras"].get(camera_code, {})
        old_index = camera_config.get("index")
        new_index = request.index

        # If user selected a different device, clear that device from any other camera_code
        # so only one slot (e.g. scope_camera) owns the device.
        for other_code, other_config in current_settings["cameras"].items():
            if other_code != camera_code and other_config.get("index") == new_index:
                other_config["index"] = -1

        # Update all fields from request
        camera_config["index"] = request.index
        camera_config["name"] = request.name
        camera_config["width"] = request.width
        camera_config["height"] = request.height
        camera_config["fps"] = request.fps
        camera_config["flip_horizontal"] = request.flip_horizontal
        camera_config["flip_vertical"] = request.flip_vertical
        camera_config["rotate"] = request.rotate
        camera_config["brightness"] = request.brightness
        camera_config["contrast"] = request.contrast
        camera_config["hue"] = request.hue
        camera_config["saturation"] = request.saturation
        camera_config["sharpness"] = request.sharpness
        camera_config["gamma"] = request.gamma
        camera_config["white_balance_temperature"] = request.white_balance_temperature
        camera_config["backlight"] = request.backlight
        camera_config["gain"] = request.gain
        camera_config["focus"] = request.focus
        camera_config["exposure"] = request.exposure
        camera_config["auto_white_balance_temperature"] = request.auto_white_balance_temperature
        camera_config["auto_focus"] = request.auto_focus
        camera_config["auto_exposure"] = request.auto_exposure
        camera_config["crop_top"] = request.crop_top
        camera_config["crop_left"] = request.crop_left
        camera_config["crop_bottom"] = request.crop_bottom
        camera_config["crop_right"] = request.crop_right
        camera_config["stretch_enabled"] = request.stretch_enabled
        camera_config["stretch_width"] = request.stretch_width
        camera_config["stretch_height"] = request.stretch_height
        camera_config["static_reticle_x"] = request.static_reticle_x
        camera_config["static_reticle_y"] = request.static_reticle_y
        camera_config["static_reticle_color"] = request.static_reticle_color
        camera_config["static_reticle_outline"] = request.static_reticle_outline
        camera_config["static_reticle_size"] = request.static_reticle_size
        camera_config["mask_polygons"] = request.mask_polygons

        # Reset CamerasController to apply changes
        camera: Camera = master_controller.cameras_controller.cameras[camera_config["index"]]
        # get supported resolutions from the camera
        camera_config["supported_resolutions"] = camera.supported_resolutions
        # Update the cameras object
        current_settings["cameras"][camera_code] = camera_config
        # Save to disk
        await settingscontroller.save_settings(current_settings)
        # update the camera settings
        camera_settings = camera.settings
        camera_settings.update(camera_config)
        camera.settings = camera_settings

        # Update Camera instances: old device gets camera_code None, new device gets this camera_code
        cameras_list = master_controller.cameras_controller.cameras
        n = len(cameras_list)
        if old_index is not None and old_index != new_index and 0 <= old_index < n:
            cameras_list[old_index]._camera_code = None
        if 0 <= new_index < n:
            cameras_list[new_index]._camera_code = camera_code

        return {"success": True}
        
    except Exception as e:
        print(f"Error updating camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/reset/{camera_code}")
async def reset_camera(*, camera_code: str):
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
        # Get current settings to find the camera index
        current_settings = await settingscontroller.get_settings()
        cameras_config = current_settings.get("cameras", {})
        camera_config = cameras_config.get(camera_code, {})
        camera_index = camera_config.get("index", 0)

        error_when_searching_for_camera_index = False        
        if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
            error_when_searching_for_camera_index = True
            print(f"Error: Resetting camera {camera_code}: Camera index {camera_index} not found."
                    "Trying to find the camera index by name.")
            try:
                camera_index = CAMERA_NAMES.keys().index(camera_code)
            except Exception as e:
                camera_index = 0
                print(f"Error:Camera {camera_code} not found in CAMERA_NAMES.keys()")
        
        # Reset the camera settings to default
        camera: Camera = master_controller.cameras_controller.cameras[camera_index]
        camera.reset_settings()
        current_settings["cameras"][camera_code].update(camera.settings)

        # Save to disk
        await settingscontroller.save_settings(current_settings)

        if error_when_searching_for_camera_index:
            raise HTTPException(
                status_code=404,
                detail=f"Camera index {camera_index} not found"
            )
        else:        
            return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error resetting camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/resetall")
async def reset_all_cameras():
    """
    Reset all cameras by calling master_controller.cameras_controller.reset().
    Reloads all cameras from scratch.
    """
    try:
        master_controller.cameras_controller.reset(reset_to_default=True)
        await settingscontroller.clear_cached_settings()
        for camera_name in CAMERA_NAMES:
            await reset_camera(camera_code=camera_name)
        return {"success": True}
    except Exception as e:
        print(f"Error resetting all cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/cameras/stop/{camera_code}")
async def stop_camera(*, camera_code: str):
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
        # Get current settings to find the camera index
        current_settings = await settingscontroller.get_settings()
        cameras_config = current_settings.get("cameras", {})
        camera_config = cameras_config.get(camera_code, {})
        camera_index = camera_config.get("index", 0)
        
        if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
            raise HTTPException(
                status_code=404,
                detail=f"Camera index {camera_index} not found"
            )
        
        # Stop the camera
        camera: Camera = master_controller.cameras_controller.cameras[camera_index]
        camera.stop()
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error stopping camera {camera_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_camera_frames(*, index: int, mode: int=0, quality: int=90):
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
    # uid of last frame sent to the client
    uid_of_last_frame_sent_to_client = -1
    # time of last frame sent to the client
    time_of_last_frame_sent_to_client = 0
    while True:
        try:
            if index < 0 or index >= len(master_controller.cameras_controller.cameras):
                break
            
            now = time.time()
            camera = master_controller.cameras_controller.cameras[index]           
            # get the image based on the mode
            if mode == 3:
                frame = camera.frame_masked_ai
            elif mode == 1:
                frame = camera.frame_masked
            else:
                frame = camera.frame
            # send image to the client if it is new or it has been 0.5 seconds since the last frame was sent
            if frame.image is not None and \
                (frame.uid != uid_of_last_frame_sent_to_client or now - time_of_last_frame_sent_to_client > 0.5):
                # update the uid and time of the last frame sent to the client
                uid_of_last_frame_sent_to_client = frame.uid
                time_of_last_frame_sent_to_client = now
                # encode the image as JPEG
                _, jpeg = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                image_bytes = jpeg.tobytes()                
                # Yield as multipart frame
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
                )
            
            # Small delay to control frame rate
            await asyncio.sleep(EPSILON_DELAY)
            
        except Exception as e:
            print(f"Error streaming camera {index}: {e}")
            await asyncio.sleep(EPSILON_DELAY)


@router.get("/api/cameras/stream/{item_code}")
async def stream_camera(*, item_code: str, mode: int = 0, quality: int = 80):
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
        # Get the camera index from settings
        settings = await settingscontroller.get_settings()
        cameras = settings.get("cameras", {})
        camera_config = cameras.get(item_code, {})
        camera_index = camera_config.get("index", 0)
        
        if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
            raise HTTPException(
                status_code=404,
                detail=f"Camera index {camera_index} not found"
            )

        # refresh the camera settings
        # necessary otherwise after the resolution change the camera is zoomed in
        camera: Camera = master_controller.cameras_controller.cameras[camera_index]
        camera.settings = camera.settings
        
        return StreamingResponse(
            generate_camera_frames(index=camera_index, mode=mode, quality=quality-10),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error streaming camera {item_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

