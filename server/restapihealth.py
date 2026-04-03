from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .context import StartupState, get_startup_state

router = APIRouter()


def _get_readiness_status_code(startup_state: StartupState) -> int:
    return 200 if startup_state.ready else 503


@router.get("/api/health/readiness")
async def get_readiness(
    startup_state: StartupState = Depends(get_startup_state),
):
    return JSONResponse(
        status_code=_get_readiness_status_code(startup_state),
        content={
            "success": startup_state.ready,
            "ready": startup_state.ready,
            "phase": startup_state.phase,
            "lastError": startup_state.last_error,
        },
    )
