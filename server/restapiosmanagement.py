"""
REST API for OS management operations.
Handles system date/time and timezone configuration.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from . import osmanagement
from .common import get_platform_info

router = APIRouter()


class SystemInfoResponse(BaseModel):
    """Response model for getting system info."""
    success: bool
    dateTime: str
    dateTimeUtc: str
    timezoneName: str
    timezoneOffsetMinutes: int
    osType: str
    availableTimezones: list[str]


class SetDateTimeRequest(BaseModel):
    """Request model for setting system date/time."""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


class SetTimezoneRequest(BaseModel):
    """Request model for setting system timezone."""
    timezone: str


class SetDateTimeAndTimezoneRequest(BaseModel):
    """Request model for setting both date/time and timezone."""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    timezone: str


class OperationResponse(BaseModel):
    """Response model for operations."""
    success: bool
    message: str


@router.get("/api/system/platform-info")
async def get_system_platform_info():
    """
    Get platform information (OS, hardware, user).
    Returns data from common.get_system_info().
    """
    try:
        info = get_platform_info()
        return info
    except Exception as e:
        print(f"Error getting platform info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/system/info")
async def get_system_info():
    """
    Get current system date/time information.
    Returns datetime, timezone, and available timezones.
    """
    try:
        info = osmanagement.get_system_info()
        available_tz = osmanagement.get_available_timezones()
        
        return {
            "success": True,
            "dateTime": info["dateTime"],
            "dateTimeUtc": info["dateTimeUtc"],
            "timezoneName": info["timezoneName"],
            "timezoneOffsetMinutes": info["timezoneOffsetMinutes"],
            "osType": info["osType"],
            "availableTimezones": available_tz
        }
    except Exception as e:
        print(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/system/datetime")
async def set_system_datetime(*, request: SetDateTimeRequest):
    """
    Set the system date and time.
    Requires elevated privileges.
    """
    try:
        success, message = osmanagement.set_system_datetime(
            year=request.year,
            month=request.month,
            day=request.day,
            hour=request.hour,
            minute=request.minute,
            second=request.second
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return {"success": True, "message": message}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting system datetime: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/system/timezone")
async def set_system_timezone(*, request: SetTimezoneRequest):
    """
    Set the system timezone.
    Requires elevated privileges.
    """
    try:
        success, message = osmanagement.set_system_timezone(
            timezone_name=request.timezone
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return {"success": True, "message": message}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting system timezone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/system/reboot")
async def reboot_system():
    """
    Reboot the system.
    Requires elevated privileges.
    """
    try:
        success, message = osmanagement.reboot_system()

        if not success:
            raise HTTPException(status_code=500, detail=message)

        return {"success": True, "message": message}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error rebooting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/system/datetime-and-timezone")
async def set_system_datetime_and_timezone(*, request: SetDateTimeAndTimezoneRequest):
    """
    Set both the system date/time and timezone.
    First sets timezone, then sets date/time.
    Requires elevated privileges.
    """
    try:
        # First set timezone
        tz_success, tz_message = osmanagement.set_system_timezone(
            timezone_name=request.timezone
        )
        
        if not tz_success:
            raise HTTPException(status_code=500, detail=f"Failed to set timezone: {tz_message}")
        
        # Then set date/time
        dt_success, dt_message = osmanagement.set_system_datetime(
            year=request.year,
            month=request.month,
            day=request.day,
            hour=request.hour,
            minute=request.minute,
            second=request.second
        )
        
        if not dt_success:
            raise HTTPException(status_code=500, detail=f"Timezone updated but failed to set date/time: {dt_message}")
        
        return {
            "success": True, 
            "message": f"Timezone set to {request.timezone} and date/time updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting system datetime and timezone: {e}")
        raise HTTPException(status_code=500, detail=str(e))
