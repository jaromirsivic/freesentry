"""
REST API for Manual Control page motor settings.
Handles loading and saving motor visibility settings for the Manual Control interface.
"""
from fastapi import APIRouter, HTTPException
from . import settingscontroller
from pydantic import BaseModel
from .context import master_controller
from .common import epsilon, Vector2D, Line2D, fit_vector_to_polygon, rotate_vector
import math

router = APIRouter()


class MotorConfig(BaseModel):
    """Configuration for a single motor in manual control."""
    index: int
    enabled: bool
    name: str = ""
    mode: str = "joystick"  # "joystick" or "slider"
    color: str = "#888888"  # Motor color from settings


class ManualControlMotorsResponse(BaseModel):
    """Response model for getting manual control motors."""
    success: bool
    motors: list[MotorConfig]


class CameraConfig(BaseModel):
    """Camera configuration for manual control."""
    selectedCamera: str = "scope_camera"
    streamQuality: int = 80
    scopeCameraMode: int = 0
    spotterCamera1Mode: int = 0
    spotterCamera2Mode: int = 0
    spotterCamera3Mode: int = 0


class SaveMotorVisibilityRequest(BaseModel):
    """Request model for saving motor visibility settings."""
    motors: list[dict]
    camera: CameraConfig | None = None


class MotorAction(BaseModel):
    """Motor action from Joystick1D component."""
    index: int
    value: float = 0.0


class ManualControlActionRequest(BaseModel):
    """Request model for manual control action."""
    fullscreen: bool = False
    joystick: Vector2D
    motors: list[MotorAction]


class MotorStatus(BaseModel):
    """Motor status in response."""
    index: int
    position: float = 0.0
    speed: float = 0.0
    duty: float = 0.0


class ManualControlActionResponse(BaseModel):
    """Response model for manual control action."""
    success: bool
    motors: list[MotorStatus]


@router.get("/api/manualcontrol/motors")
async def get_manual_control_motors():
    """
    Get motor settings for Manual Control page.
    Returns at most 4 motors with their names (from motors array) and enabled status.
    """
    try:
        settings = await settingscontroller.get_settings()
        
        # Get manualControl.motors array (max 4)
        manual_control = settings.get("manualControl", {})
        motor_configs = manual_control.get("motors", [])[:4]  # Limit to 4 motors
        
        # Get motors array for names
        motors_list = settings.get("motors", [])
        
        result = []
        for config in motor_configs:
            motor_index = config.get("index", 0)
            enabled = config.get("enabled", True)
            mode = config.get("mode", "joystick")
            
            # Get motor name and color from motors array using index
            motor_name = ""
            motor_color = "#888888"
            if 0 <= motor_index < len(motors_list):
                motor_name = motors_list[motor_index].get("name", f"Motor {motor_index}")
                motor_color = motors_list[motor_index].get("color", "#888888")
            else:
                motor_name = f"Motor {motor_index}"
            
            result.append({
                "index": motor_index,
                "enabled": enabled,
                "name": motor_name,
                "mode": mode,
                "color": motor_color
            })
        
        # Get camera settings
        camera_config = manual_control.get("camera", {})
        camera_settings = {
            "selectedCamera": camera_config.get("selectedCamera", "scope_camera"),
            "streamQuality": camera_config.get("streamQuality", 80),
            "scopeCameraMode": camera_config.get("scopeCameraMode", 0),
            "spotterCamera1Mode": camera_config.get("spotterCamera1Mode", 0),
            "spotterCamera2Mode": camera_config.get("spotterCamera2Mode", 0),
            "spotterCamera3Mode": camera_config.get("spotterCamera3Mode", 0)
        }
        
        return {"success": True, "motors": result, "camera": camera_settings}
    
    except Exception as e:
        print(f"Error getting manual control motors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/manualcontrol/motors")
async def save_manual_control_motors(*, request: SaveMotorVisibilityRequest):
    """
    Save motor visibility settings for Manual Control page.
    Updates the enabled status of motors in settings.json -> manualControl -> motors.
    """
    try:
        current_settings = await settingscontroller.get_settings()
        
        # Ensure manualControl section exists
        if "manualControl" not in current_settings:
            current_settings["manualControl"] = {}
        
        # Get current motors config or create empty
        current_motors = current_settings["manualControl"].get("motors", [])
        
        # Update enabled status and mode for each motor in the request
        for motor_update in request.motors:
            motor_index = motor_update.get("index")
            enabled = motor_update.get("enabled", True)
            mode = motor_update.get("mode", "joystick")
            
            # Find and update existing motor config
            found = False
            for motor_config in current_motors:
                if motor_config.get("index") == motor_index:
                    motor_config["enabled"] = enabled
                    motor_config["mode"] = mode
                    found = True
                    break
            
            # Add new motor config if not found
            if not found:
                current_motors.append({
                    "index": motor_index,
                    "enabled": enabled,
                    "mode": mode
                })
        
        # Limit to 4 motors and save
        current_settings["manualControl"]["motors"] = current_motors[:4]
        
        # Update camera settings if provided
        if request.camera is not None:
            if "camera" not in current_settings["manualControl"]:
                current_settings["manualControl"]["camera"] = {}
            
            current_settings["manualControl"]["camera"]["selectedCamera"] = request.camera.selectedCamera
            current_settings["manualControl"]["camera"]["streamQuality"] = request.camera.streamQuality
            current_settings["manualControl"]["camera"]["scopeCameraMode"] = request.camera.scopeCameraMode
            current_settings["manualControl"]["camera"]["spotterCamera1Mode"] = request.camera.spotterCamera1Mode
            current_settings["manualControl"]["camera"]["spotterCamera2Mode"] = request.camera.spotterCamera2Mode
            current_settings["manualControl"]["camera"]["spotterCamera3Mode"] = request.camera.spotterCamera3Mode
        
        await settingscontroller.save_settings(current_settings)
        
        return {"success": True}
    
    except Exception as e:
        print(f"Error saving manual control motors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/manualcontrol/action")
async def manual_control_action(*, request: ManualControlActionRequest):
    """
    Process manual control action from frontend.
    
    This endpoint is called periodically (every 0.25s) and on joystick movement.
    It sets motor speeds based on joystick positions and returns motor status.
    
    - Motors at index 0 and 1 are controlled by the Polygon joystick (x/y)
    - Other motors are controlled by their individual Joystick1D values
    """
    try:
        settings = await settingscontroller.get_settings()
        motors_list = settings.get("motors", [])
        motors_controller = master_controller.motors_controller
        
        # Build index-to-name mapping from settings
        index_to_name = {}
        for idx, motor_cfg in enumerate(motors_list):
            motor_name = motor_cfg.get("name", f"Motor {idx}")
            index_to_name[idx] = motor_name

        # Check if the joystick position is inside the polygon
        # if it is not then return intersection with the polygon
        joystick_position = request.joystick
        # square the joystick position to give more weight to the center of the joystick
        joystick_position.x = abs(joystick_position.x) * joystick_position.x
        joystick_position.y = abs(joystick_position.y) * joystick_position.y
        polygon = settings.get("polygon", [])
        intersection = fit_vector_to_polygon(vector=joystick_position, polygon=polygon)
        if intersection is not None:
            joystick_position = intersection
        joystick_position = rotate_vector(
            vector=joystick_position, 
            angle=settings.get("general", {}).get("joystickSetup", {}).get("rotationAngle", 0)
        )
        request.joystick = joystick_position
        
        # Set speed for motors 0 and 1 based on Polygon joystick
        # Motor 0 uses joystick.x, Motor 1 uses joystick.y
        if 0 in index_to_name:
            motor_name = index_to_name[0]
            if motor_name in motors_controller.motors:
                motors_controller.motors[motor_name].move(speed=request.joystick.x)
        
        if 1 in index_to_name:
            motor_name = index_to_name[1]
            if motor_name in motors_controller.motors:
                motors_controller.motors[motor_name].move(speed=request.joystick.y)
        
        # Set speed for other motors based on their Joystick1D values
        for motor_action in request.motors:
            motor_index = motor_action.index
            if motor_index in index_to_name:
                motor_name = index_to_name[motor_index]
                if motor_name in motors_controller.motors:
                    motors_controller.motors[motor_name].move(speed=motor_action.value)
        
        # Collect motor status for response
        # Include motors 0, 1 and all motors from the request
        motor_indices_to_report = {0, 1}
        for motor_action in request.motors:
            motor_indices_to_report.add(motor_action.index)
        
        result_motors = []
        for motor_index in sorted(motor_indices_to_report):
            if motor_index in index_to_name:
                motor_name = index_to_name[motor_index]
                if motor_name in motors_controller.motors:
                    motor = motors_controller.motors[motor_name]
                    result_motors.append({
                        "index": motor_index,
                        "position": motor.position,
                        "speed": motor.current_speed,
                        "duty": 0.0  # TODO: compute actual duty cycle if needed
                    })
                else:
                    # Motor exists in settings but not in controller (e.g., disabled)
                    result_motors.append({
                        "index": motor_index,
                        "position": 0.0,
                        "speed": 0.0,
                        "duty": 0.0
                    })
        
        return {"success": True, "motors": result_motors}
    
    except Exception as e:
        print(f"Error in manual control action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

