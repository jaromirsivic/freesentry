"""
REST API for AI Setup page settings.
Handles loading and saving AI Setup configuration for the front end.
"""
from copy import deepcopy
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from . import settingscontroller
from .ai_setup_constants import DEFAULT_DEVICE
from .yolomodels import YOLOModels

router = APIRouter()


class OrganConfig(BaseModel):
    """Configuration for a single organ."""
    enabled: bool = True
    sizeMultiplier: float = 1.0
    confidenceThreshold: float = 0.5
    minimumRadius: int = 1


class OrgansConfig(BaseModel):
    """Configuration for all organs."""
    brain: OrganConfig = OrganConfig()
    chest: OrganConfig = OrganConfig()
    heart: OrganConfig = OrganConfig()
    liver: OrganConfig = OrganConfig()
    abdomen: OrganConfig = OrganConfig()


class MotorConfig(BaseModel):
    """Motor configuration for mission/exit strategy."""
    index: int
    enabled: bool = False
    speed: float = 0.0


class RandomWalkMission(BaseModel):
    """Random Walk mission configuration."""
    enabled: bool = True
    disableDutyCycle: bool = True
    delayBetweenEngagements: float = 0.5
    engagementDuration: float = 0.5
    motors: list[MotorConfig] = []


class MissionsConfig(BaseModel):
    """Missions configuration."""
    randomWalk: RandomWalkMission = RandomWalkMission()


class ExitStrategyConfig(BaseModel):
    """Exit strategy configuration."""
    maxEngagements: int = 1000000000
    timeoutAfterFirstEngagement: float = 10000000000
    fixedDateTime: str = "2199-12-31T23:59:59Z"
    exitStrategyDuration: float = 0.5
    motors: list[MotorConfig] = []


class AISetupConfig(BaseModel):
    """Complete AI Setup configuration."""
    activationDateTime: str = "2199-12-31T23:59:59Z"
    minFpsToAllowEngagement: int = 0
    organMustBeVisibleSeconds: float = 0
    detectionRadiusFromReticle: int = 50
    organs: dict = {}
    missions: dict = {}
    exitStrategy: dict = {}


class AISetupResponse(BaseModel):
    """Response model for getting AI setup."""
    success: bool
    aiSetup: dict
    motors: list[dict] = []


class SaveAISetupRequest(BaseModel):
    """Request model for saving AI setup."""
    aiSetup: dict


def _build_ai_setup_response(ai_setup_settings: dict | None) -> dict:
    ai_setup = deepcopy(ai_setup_settings) if isinstance(ai_setup_settings, dict) else {}

    if "activationDateTime" not in ai_setup:
        ai_setup["activationDateTime"] = "2199-12-31T23:59:59Z"
    if "minFpsToAllowEngagement" not in ai_setup:
        ai_setup["minFpsToAllowEngagement"] = 0
    if "organMustBeVisibleSeconds" not in ai_setup:
        ai_setup["organMustBeVisibleSeconds"] = 0
    if "detectionRadiusFromReticle" not in ai_setup:
        ai_setup["detectionRadiusFromReticle"] = 50
    if "modelName" not in ai_setup:
        ai_setup["modelName"] = YOLOModels.DEFAULT_MODEL_NAME
    if "device" not in ai_setup:
        ai_setup["device"] = DEFAULT_DEVICE
    ai_setup["device"] = YOLOModels.normalize_device_value(ai_setup.get("device"))
    if "organs" not in ai_setup:
        ai_setup["organs"] = {
            "brain": {"enabled": True, "sizeMultiplier": 1.0, "confidenceThreshold": 0.5, "minimumRadius": 1},
            "chest": {"enabled": True, "sizeMultiplier": 1.0, "confidenceThreshold": 0.5, "minimumRadius": 1},
            "heart": {"enabled": True, "sizeMultiplier": 1.0, "confidenceThreshold": 0.5, "minimumRadius": 1},
            "liver": {"enabled": True, "sizeMultiplier": 1.0, "confidenceThreshold": 0.5, "minimumRadius": 1},
            "abdomen": {"enabled": True, "sizeMultiplier": 1.0, "confidenceThreshold": 0.5, "minimumRadius": 1},
        }
    if "missions" not in ai_setup:
        ai_setup["missions"] = {
            "randomWalk": {
                "enabled": True,
                "disableDutyCycle": True,
                "delayBetweenEngagements": 0.5,
                "engagementDuration": 0.5,
                "motors": [],
            }
        }
    if "exitStrategy" not in ai_setup:
        ai_setup["exitStrategy"] = {
            "maxEngagements": 1000000000,
            "timeoutAfterFirstEngagement": 10000000000,
            "fixedDateTime": "2199-12-31T23:59:59Z",
            "exitStrategyDuration": 0.5,
            "motors": [],
        }

    return ai_setup


@router.get("/api/aisetup")
async def get_ai_setup():
    """
    Get AI Setup settings from settings.json.
    Returns aiSetup section along with motor names and colors.
    """
    try:
        settings = await settingscontroller.get_settings()

        # Build defaults on a detached copy so GET requests stay read-only.
        ai_setup = _build_ai_setup_response(settings.get("aiSetup", {}))

        # Get motors array for names and colors
        motors_list = settings.get("motors", [])
        motors_info = []
        for idx, motor_cfg in enumerate(motors_list):
            motors_info.append({
                "index": idx,
                "name": motor_cfg.get("name", f"Motor {idx}"),
                "color": motor_cfg.get("color", "#888888")
            })

        return {
            "success": True,
            "aiSetup": ai_setup,
            "modelNames": list(YOLOModels.MODEL_NAMES),
            "deviceOptions": YOLOModels.get_supported_device_options(),
            "defaultDevice": YOLOModels.get_default_device_value(),
            "motors": motors_info
        }

    except Exception as e:
        print(f"Error getting AI setup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/aisetup")
async def save_ai_setup(*, request: SaveAISetupRequest):
    """
    Save AI Setup settings to settings.json.
    Updates the aiSetup section.
    """
    try:
        def update_ai_setup(settings: dict) -> None:
            settings["aiSetup"] = request.aiSetup

        await settingscontroller.update_settings(update_ai_setup)

        return {"success": True}

    except Exception as e:
        print(f"Error saving AI setup: {e}")
        raise HTTPException(status_code=500, detail=str(e))
