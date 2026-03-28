from fastapi import HTTPException
from pathlib import Path
from typing import Any
import json
import threading

# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "wwwroot/src/assets/settings.json"
CACHED_SETTINGS_LOCK = threading.RLock()

def get_settings_sync() -> dict[str, Any]:
    """Get current settings from settings.json file"""
    try:
        # Check if cached settings are available
        with CACHED_SETTINGS_LOCK:
            CACHED_SETTINGS = globals().get("CACHED_SETTINGS", None)
            if CACHED_SETTINGS is not None:
                return CACHED_SETTINGS
        # Load settings from file
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        # Cache settings
        with CACHED_SETTINGS_LOCK:
            globals().update({"CACHED_SETTINGS": settings})
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")

async def get_settings() -> dict[str, Any]:
    """Get current settings from settings.json file"""
    try:
        # Check if cached settings are available
        with CACHED_SETTINGS_LOCK:
            CACHED_SETTINGS = globals().get("CACHED_SETTINGS", None)
            if CACHED_SETTINGS is not None:
                return CACHED_SETTINGS
        # Load settings from file
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        # Cache settings
        with CACHED_SETTINGS_LOCK:
            globals().update({"CACHED_SETTINGS": settings})
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


async def save_settings(settings: dict[str, Any]) -> dict[str, str]:
    """Save settings to settings.json file"""
    try:
        # Write to settings.json
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        # Cache settings
        with CACHED_SETTINGS_LOCK:
            globals().update({"CACHED_SETTINGS": settings})
        return {"success": True, "message": "Settings saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


def clear_cached_settings_sync():
    """Drop cached settings"""
    try:        
        with CACHED_SETTINGS_LOCK:
            globals().pop("CACHED_SETTINGS", None)
            return {"success": True, "message": "Cached settings dropped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop cached settings: {str(e)}")


async def clear_cached_settings():
    """Drop cached settings"""
    try:        
        with CACHED_SETTINGS_LOCK:
            globals().pop("CACHED_SETTINGS", None)
            return {"success": True, "message": "Cached settings dropped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop cached settings: {str(e)}")