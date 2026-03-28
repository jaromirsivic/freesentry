from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from . import settingscontroller
from .context import master_controller

router = APIRouter()

class MotorActionStartRequest(BaseModel):
    pin_index: int
    pwm_multiplier: float

class MotorActionStopRequest(BaseModel):
    pin_index: int

class MotorSpeedRequest(BaseModel):
    motor_name: str
    speed: float

async def get_j8():
    """Get the J8 instance from the motors controller"""
    return master_controller.motors_controller.j8

@router.get("/api/settings/motors")
async def get_motors_settings_endpoint():
    """Get motors settings from settings.json file"""
    settings = await settingscontroller.get_settings()
    return settings.get("motors", [])

@router.post("/api/settings/motors")
async def save_motors_settings_endpoint(motors_settings: list[dict[str, Any]]):
    """Save motors settings to settings.json file"""
    settings = await settingscontroller.get_settings()
    settings["motors"] = motors_settings
    await settingscontroller.save_settings(settings)
    master_controller.motors_controller.reset()
    return {"success": True}

@router.get("/api/controller/status")
async def get_controller_status_endpoint():
    """Get J8 controller initialization status"""
    try:
        j8 = await get_j8()
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
async def start_motor_action(request: MotorActionStartRequest):
    """Start motor action: set J8[pin_index] to pwm_multiplier"""
    try:
        j8 = await get_j8()
        # Note: We do not reset the controller here because it would interrupt the automatic execution loop.
        
        j8[request.pin_index].value = request.pwm_multiplier
        return {"success": True, "message": f"Pin {request.pin_index} set to {request.pwm_multiplier}"}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/motors/action/stop")
async def stop_motor_action(request: MotorActionStopRequest):
    """Stop motor action: reset J8[pin_index]"""
    try:
        j8 = await get_j8()
        j8[request.pin_index].reset()
        return {"success": True, "message": f"Pin {request.pin_index} reset"}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/motors/speed")
async def set_motor_speed(request: MotorSpeedRequest):
    """Set motor speed via REST API"""
    try:
        motors = master_controller.motors_controller.motors
        if request.motor_name in motors:
            motors[request.motor_name].move(speed=float(request.speed))
            return {"success": True, "motor_name": request.motor_name, "speed": request.speed}
        else:
            raise HTTPException(status_code=404, detail=f"Motor not found: {request.motor_name}")
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

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

