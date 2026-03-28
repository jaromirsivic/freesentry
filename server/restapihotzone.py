from fastapi import APIRouter
from typing import Any
from . import settingscontroller

router = APIRouter()

@router.get("/api/settings/hot-zone")
async def get_hot_zone_settings_endpoint():
    """Get hot zone settings from settings.json file"""
    settings = await settingscontroller.get_settings()
    return settings.get("hotZone", {})

@router.post("/api/settings/hot-zone")
async def save_hot_zone_settings_endpoint(hot_zone_settings: dict[str, Any]):
    """Save hot zone settings to settings.json file"""
    settings = await settingscontroller.get_settings()
    settings["hotZone"] = hot_zone_settings
    return await settingscontroller.save_settings(settings)

