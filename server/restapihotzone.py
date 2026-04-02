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
    def update_hot_zone_settings(settings: dict[str, Any]) -> None:
        settings["hotZone"] = hot_zone_settings

    await settingscontroller.update_settings(update_hot_zone_settings)
    return {"success": True, "message": "Settings saved successfully"}

