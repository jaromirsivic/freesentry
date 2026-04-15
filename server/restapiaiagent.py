"""
REST API for runtime AI Agent activation.
Keeps activation state in the running backend only.
"""
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .context import get_master_controller

if TYPE_CHECKING:
    from .mastercontroller import MasterController


router = APIRouter()


class AIAgentActivationResponse(BaseModel):
    success: bool
    aiagent_fully_activated: bool


@router.get("/api/aiagent/activation", response_model=AIAgentActivationResponse)
async def get_aiagent_activation(
    *,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Return the runtime AI Agent activation state."""
    try:
        return {
            "success": True,
            "aiagent_fully_activated": master_controller.ai_agent.is_fully_activated(),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting AI agent activation state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/aiagent/activate", response_model=AIAgentActivationResponse)
async def activate_aiagent(
    *,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Enable runtime AI motor control."""
    try:
        master_controller.ai_agent.activate()
        return {
            "success": True,
            "aiagent_fully_activated": master_controller.ai_agent.is_fully_activated(),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error activating AI agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/aiagent/deactivate", response_model=AIAgentActivationResponse)
async def deactivate_aiagent(
    *,
    master_controller: "MasterController" = Depends(get_master_controller),
):
    """Disable runtime AI motor control and stop AI-driven motors."""
    try:
        master_controller.ai_agent.deactivate()
        return {
            "success": True,
            "aiagent_fully_activated": master_controller.ai_agent.is_fully_activated(),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deactivating AI agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
