from fastapi import APIRouter
from typing import Any
from . import settingscontroller
from .context import master_controller

router = APIRouter()

@router.post("/api/settings")
async def save_settings_endpoint(settings: dict[str, Any]):
    """Save settings to settings.json file"""
    return await settingscontroller.save_settings(settings)

@router.get("/api/settings")
async def get_settings_endpoint():
    """Get current settings from settings.json file"""
    return await settingscontroller.get_settings()

@router.get("/api/settings/general")
async def get_general_settings_endpoint():
    """Get general settings from settings.json file"""
    settings = await settingscontroller.get_settings()
    return settings.get("general", {})

@router.post("/api/settings/general")
async def save_general_settings_endpoint(general_settings: dict[str, Any]):
    """Save general settings to settings.json file"""
    settings = await settingscontroller.get_settings()
    settings["general"] = general_settings
    await settingscontroller.save_settings(settings)
    
    # Reset controller with new settings
    try:
        master_controller.motors_controller.reset(is_hard_reset=True)
    except Exception as e:
        print(f"Failed to reset controller after settings save: {e}")
        
    return {"success": True}

