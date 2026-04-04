from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from . import settingscontroller
from .context import get_master_controller
from .motorerrors import MotorOverrideConflictError

if TYPE_CHECKING:
    from .mastercontroller import MasterController

router = APIRouter()

class MotorActionStartRequest(BaseModel):
    pin_index: int
    pwm_multiplier: float

class MotorActionStopRequest(BaseModel):
    pin_index: int

class MotorSpeedRequest(BaseModel):
    motor_name: str
    speed: float

async def get_j8(master_controller: "MasterController"):
    """Get the J8 instance from the motors controller"""
    return master_controller.motors_controller.j8


def _raise_motor_http_error(exc: Exception) -> None:
    if isinstance(exc, MotorOverrideConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get("/api/settings/motors")
async def get_motors_settings_endpoint():
    """Get motors settings from settings.json file"""
    settings = await settingscontroller.get_settings()
    return settings.get("motors", [])

@router.post("/api/settings/motors")
async def save_motors_settings_endpoint(
    motors_settings: list[dict[str, Any]],
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Save motors settings to settings.json file"""
    def update_motors_settings(settings: dict[str, Any]) -> None:
        settings["motors"] = motors_settings

    await settingscontroller.update_settings(update_motors_settings)
    master_controller.motors_controller.reset()
    return {"success": True}

@router.get("/api/controller/status")
async def get_controller_status_endpoint(
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Get J8 controller initialization status"""
    try:
        j8 = await get_j8(master_controller=master_controller)
        return {
            "initialized": j8.initialized,
            "error_message": j8.error_message
        }
    except Exception as e:
        return {
            "initialized": False,
            "error_message": str(e)
        }

@router.post("/api/motors/action/start")
async def start_motor_action(
    request: MotorActionStartRequest,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Start an exclusive manual hardware override for histogram quick-test."""
    try:
        active_pin = master_controller.motors_controller.start_manual_override(
            pin_index=request.pin_index,
            pwm_multiplier=request.pwm_multiplier,
        )
        return {
            "success": True,
            "pin_index": active_pin,
            "message": f"Manual hardware override active on pin {active_pin}",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(str(e))
        _raise_motor_http_error(e)

@router.post("/api/motors/action/stop")
async def stop_motor_action(
    request: MotorActionStopRequest,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Stop the active manual hardware override and resume managed control."""
    try:
        stopped_pin = master_controller.motors_controller.stop_manual_override(
            pin_index=request.pin_index
        )
        if stopped_pin is None:
            return {
                "success": True,
                "message": "No manual hardware override was active.",
            }
        return {
            "success": True,
            "pin_index": stopped_pin,
            "message": f"Manual hardware override on pin {stopped_pin} stopped",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(str(e))
        _raise_motor_http_error(e)

@router.post("/api/motors/speed")
async def set_motor_speed(
    request: MotorSpeedRequest,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Set motor speed via REST API"""
    try:
        applied = master_controller.motors_controller.set_motor_speed_by_name(
            motor_name=request.motor_name,
            speed=request.speed,
        )
        if not applied:
            raise HTTPException(status_code=404, detail=f"Motor not found: {request.motor_name}")
        return {"success": True, "motor_name": request.motor_name, "speed": request.speed}
    except HTTPException:
        raise
    except Exception as e:
        print(str(e))
        _raise_motor_http_error(e)

@router.get("/api/motors/speedhistogram")
async def get_speed_histogram_endpoint():
    """Get speed histogram data for Chart2D visualization"""
    from .speedhistogram import SpeedHistogram
    settings = await settingscontroller.get_settings()
    motors = settings.get("motors", [])
    
    result = []
    for motor in motors:
        histogram_data = motor.get("histogram", [])
        if len(histogram_data) >= 2:
            # Use histogram data directly (already in camelCase format)
            speed_histogram_input = histogram_data
            try:
                speed_histogram = SpeedHistogram(speed_histogram=speed_histogram_input)
                # Convert to Chart2D format: list of {x, y} points
                resolution = speed_histogram.resolution
                forward_data = [
                    {"x": i / resolution, "y": speed_histogram.forward_speed_histogram[i]}
                    for i in range(resolution + 1)
                ]
                reverse_data = [
                    {"x": i / resolution, "y": speed_histogram.reverse_speed_histogram[i]}
                    for i in range(resolution + 1)
                ]
                result.append({
                    "motorName": motor.get("name", "Unknown"),
                    "forward": forward_data,
                    "reverse": reverse_data
                })
            except Exception as e:
                print(f"Error processing histogram for motor {motor.get('name')}: {e}")
                result.append({
                    "motorName": motor.get("name", "Unknown"),
                    "forward": [],
                    "reverse": [],
                    "error": str(e)
                })
        else:
            result.append({
                "motorName": motor.get("name", "Unknown"),
                "forward": [],
                "reverse": [],
                "error": "Insufficient histogram data"
            })
    
    return result

