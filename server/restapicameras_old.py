from typing import TYPE_CHECKING, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import cv2
from . import settingscontroller
from .context import get_master_controller

if TYPE_CHECKING:
    from .mastercontroller import MasterController

router = APIRouter()

class CameraSettings(BaseModel):
    index: int
    name: str = ""
    camera_role: str = "scope_cam_ai"
    width: int
    height: int
    fps: float
    flip_horizontal: bool
    flip_vertical: bool
    rotate: int
    brightness: float
    contrast: float
    hue: float
    saturation: float
    sharpness: float
    gamma: float
    white_balance_temperature: float
    backlight: float
    gain: float
    focus: float
    exposure: float
    auto_white_balance_temperature: bool
    auto_focus: bool
    auto_exposure: bool
    crop_top: float = 0.0
    crop_left: float = 0.0
    crop_bottom: float = 0.0
    crop_right: float = 0.0
    stretch_width: int = 0
    stretch_height: int = 0
    static_reticle_x: float = 0.0
    static_reticle_y: float = 0.0
    static_reticle_color: str = "#ff0000cc"
    static_reticle_size: float = 0.0
    mask_polygons: List[List[Dict[str, float]]] = []
    saveToDisk: bool = False


def _find_camera_config_by_index(cameras: dict[str, dict[str, Any]], index: int) -> dict[str, Any] | None:
    for camera_config in cameras.values():
        if isinstance(camera_config, dict) and camera_config.get("index") == index:
            return camera_config
    return None


@router.get("/api/cameras_old/list")
async def get_cameras_list_endpoint(
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Get connected cameras info and settings"""
    settings = await settingscontroller.get_settings()
    general_settings = settings.get("general", {})
    
    input_devices = []
    
    for cam in master_controller.cameras_controller.cameras:
        device_info = cam.settings
        input_devices.append(device_info)
        
    return {
        **general_settings,
        "input_devices": input_devices
    }

@router.get("/api/cameras_old/primary")
async def get_primary_camera_endpoint():
    """Get the primary camera configuration"""
    settings = await settingscontroller.get_settings()
    primary_camera = settings.get("primaryCamera", {})
    return {"success": True, "primaryCamera": primary_camera}

@router.post("/api/reset")
async def reset_cameras_endpoint(
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Reset camera settings to defaults"""
    # Reset in-memory settings for all cameras
    for camera in master_controller.cameras_controller.cameras:
        camera.reset_settings()

    # Update settings.json to remove general section for each camera
    try:
        def reset_legacy_camera_settings(current_settings: dict[str, Any]) -> None:
            cameras = current_settings.get("cameras", {})
            for camera_config in cameras.values():
                if isinstance(camera_config, dict) and "general" in camera_config:
                    del camera_config["general"]

        await settingscontroller.update_settings(reset_legacy_camera_settings)
    except Exception as e:
        print(f"Error resetting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    master_controller.cameras_controller.reset()
    return {"success": True}

@router.post("/api/cameras_old/savecamera")
async def save_camera_settings(
    settings: CameraSettings,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Save or apply camera settings.
    If saveToDisk is True, saves to settings.json.
    Always applies settings to the running camera instance.
    """
    try:
        # Find the camera by index
        camera = None
        for cam in master_controller.cameras_controller.cameras:
            if cam.index == settings.index:
                camera = cam
                break
        
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with index {settings.index} not found")

        # Apply settings to the running camera instance
        camera.settings = settings.model_dump()
        # camera.width = settings.width
        # camera.height = settings.height
        # camera.fps = settings.fps
        # camera.flip_horizontal = settings.flip_horizontal
        # camera.flip_vertical = settings.flip_vertical
        # camera.rotate = settings.rotate
        # camera.brightness = settings.brightness
        # camera.contrast = settings.contrast
        # camera.hue = settings.hue
        # camera.saturation = settings.saturation
        # camera.sharpness = settings.sharpness
        # camera.gamma = settings.gamma
        # camera.white_balance_temperature = settings.white_balance_temperature
        # camera.backlight = settings.backlight_compensation
        # camera.gain = settings.gain
        # camera.focus = settings.focus
        # camera.exposure = settings.exposure
        # camera.auto_white_balance_temperature = settings.auto_white_balance_temperature
        # camera.auto_focus = settings.auto_focus
        # camera.auto_exposure = settings.auto_exposure

        # If saveToDisk is True, update settings.json
        if settings.saveToDisk:
            def update_legacy_camera_settings(current_settings: dict[str, Any]) -> None:
                cameras = current_settings.setdefault("cameras", {})
                camera_config = _find_camera_config_by_index(cameras, settings.index)
                if camera_config is None:
                    legacy_camera_code = f"legacy_camera_{settings.index}"
                    camera_config = cameras.setdefault(legacy_camera_code, {"index": settings.index})

                if settings.name:
                    current_settings["primaryCamera"] = {
                        "index": settings.index,
                        "name": settings.name,
                    }

                camera_config["index"] = settings.index
                camera_config["camera_role"] = settings.camera_role
                camera_config["width"] = settings.width
                camera_config["height"] = settings.height
                camera_config["fps"] = settings.fps
                camera_config["flip_horizontal"] = settings.flip_horizontal
                camera_config["flip_vertical"] = settings.flip_vertical
                camera_config["rotate"] = settings.rotate
                camera_config["brightness"] = settings.brightness
                camera_config["contrast"] = settings.contrast
                camera_config["hue"] = settings.hue
                camera_config["saturation"] = settings.saturation
                camera_config["sharpness"] = settings.sharpness
                camera_config["gamma"] = settings.gamma
                camera_config["white_balance_temperature"] = settings.white_balance_temperature
                camera_config["backlight"] = settings.backlight
                camera_config["gain"] = settings.gain
                camera_config["focus"] = settings.focus
                camera_config["exposure"] = settings.exposure
                camera_config["auto_white_balance_temperature"] = settings.auto_white_balance_temperature
                camera_config["auto_focus"] = settings.auto_focus
                camera_config["auto_exposure"] = settings.auto_exposure
                camera_config["crop_top"] = settings.crop_top
                camera_config["crop_left"] = settings.crop_left
                camera_config["crop_bottom"] = settings.crop_bottom
                camera_config["crop_right"] = settings.crop_right
                camera_config["stretch_width"] = settings.stretch_width
                camera_config["stretch_height"] = settings.stretch_height
                camera_config["static_reticle_x"] = settings.static_reticle_x
                camera_config["static_reticle_y"] = settings.static_reticle_y
                camera_config["static_reticle_color"] = settings.static_reticle_color
                camera_config["static_reticle_size"] = settings.static_reticle_size
                camera_config["mask_polygons"] = settings.mask_polygons
                camera_config.pop("general", None)

            await settingscontroller.update_settings(update_legacy_camera_settings)

        return {"success": True}

    except Exception as e:
        print(f"Error saving camera settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_camera_frames(*, camera_index: int, master_controller: "MasterController"):
    """
    Generator that yields MJPEG frames from the camera.
    Uses multipart/x-mixed-replace for browser-native streaming.
    """
    while True:
        try:
            # Get the camera by index
            if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
                break
            
            camera = master_controller.cameras_controller.cameras[camera_index]
            image = camera.frame.image
            
            if image is not None:
                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = jpeg.tobytes()
                
                # Yield as multipart frame
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
                )
            
            # Small delay to control frame rate (100 fps max)
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"Error streaming camera {camera_index}: {e}")
            await asyncio.sleep(0.1)

@router.get("/api/cameras_old/stream/{camera_index}")
async def stream_camera(
    camera_index: int,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Stream camera feed as MJPEG (raw camera image).
    Use this URL as an img src for live video streaming.
    """
    if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    return StreamingResponse(
        generate_camera_frames(camera_index=camera_index, master_controller=master_controller),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


async def generate_camera_frames_manual(*, camera_index: int, master_controller: "MasterController"):
    """
    Generator that yields MJPEG frames from the cropped/resized camera image.
    Uses multipart/x-mixed-replace for browser-native streaming.
    Source: camera.image_cropped_resized.frame
    """
    while True:
        try:
            if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
                break
            
            camera = master_controller.cameras_controller.cameras[camera_index]
            image = camera.image_cropped_resized.frame.image
            
            if image is not None:
                _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = jpeg.tobytes()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
                )
            
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"Error streaming manual camera {camera_index}: {e}")
            await asyncio.sleep(0.1)


async def generate_camera_frames_ai(*, camera_index: int, master_controller: "MasterController"):
    """
    Generator that yields MJPEG frames from the AI-processed camera image.
    Uses multipart/x-mixed-replace for browser-native streaming.
    Source: camera.image_ai.frame
    """
    while True:
        try:
            if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
                break
            
            camera = master_controller.cameras_controller.cameras[camera_index]
            image = camera.image_ai.frame.image
            
            if image is not None:
                _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                image_bytes = jpeg.tobytes()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n'
                )
            
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"Error streaming AI camera {camera_index}: {e}")
            await asyncio.sleep(0.1)


@router.get("/api/cameras_old/stream-manual/{camera_index}")
async def stream_camera_manual(
    camera_index: int,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Stream cropped/resized camera feed as MJPEG for Manual Control.
    Source: camera.image_cropped_resized.frame
    """
    if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    return StreamingResponse(
        generate_camera_frames_manual(camera_index=camera_index, master_controller=master_controller),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@router.get("/api/cameras_old/stream-ai/{camera_index}")
async def stream_camera_ai(
    camera_index: int,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Stream AI-processed camera feed as MJPEG for AI Agent.
    Source: camera.image_ai.frame
    """
    if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    return StreamingResponse(
        generate_camera_frames_ai(camera_index=camera_index, master_controller=master_controller),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@router.get("/api/cameras_old/frame/{camera_index}")
async def get_camera_frame(
    camera_index: int,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """
    Get a single frame from the camera as JPEG.
    Use this for polling-based updates.
    """
    if camera_index < 0 or camera_index >= len(master_controller.cameras_controller.cameras):
        raise HTTPException(status_code=404, detail=f"Camera {camera_index} not found")
    
    try:
        camera = master_controller.cameras_controller.cameras[camera_index]
        frame = camera.image.frame
        
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available")
        
        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        return StreamingResponse(
            iter([jpeg.tobytes()]),
            media_type='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
