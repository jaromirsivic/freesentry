from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request

MASTER_CONTROLLER_STATE_KEY = "master_controller"
STARTUP_STATE_KEY = "startup_state"
STARTUP_PHASE_STARTING = "starting"
STARTUP_PHASE_POST_START = "post_start"
STARTUP_PHASE_READY = "ready"
STARTUP_PHASE_FAILED = "failed"
STARTUP_PHASE_STOPPING = "stopping"

if TYPE_CHECKING:
    from .mastercontroller import MasterController


@dataclass(slots=True)
class StartupState:
    phase: str = STARTUP_PHASE_STARTING
    ready: bool = False
    last_error: str | None = None
    deferred_startup_task: asyncio.Task[None] | None = None

    def mark_post_start(self) -> None:
        self.phase = STARTUP_PHASE_POST_START
        self.ready = False
        self.last_error = None

    def mark_ready(self) -> None:
        self.phase = STARTUP_PHASE_READY
        self.ready = True
        self.last_error = None

    def mark_failed(self, error: BaseException | str) -> None:
        self.phase = STARTUP_PHASE_FAILED
        self.ready = False
        self.last_error = str(error)

    def mark_stopping(self) -> None:
        self.phase = STARTUP_PHASE_STOPPING
        self.ready = False


def set_master_controller(app: Any, controller: "MasterController") -> None:
    setattr(app.state, MASTER_CONTROLLER_STATE_KEY, controller)


def clear_master_controller(app: Any) -> None:
    if hasattr(app.state, MASTER_CONTROLLER_STATE_KEY):
        delattr(app.state, MASTER_CONTROLLER_STATE_KEY)


def get_master_controller_from_app(app: Any) -> "MasterController":
    controller = getattr(app.state, MASTER_CONTROLLER_STATE_KEY, None)
    if controller is None:
        raise RuntimeError("MasterController is not initialized.")
    return cast("MasterController", controller)


def get_master_controller(request: Request) -> "MasterController":
    return get_master_controller_from_app(request.app)


def set_startup_state(app: Any, startup_state: StartupState) -> None:
    setattr(app.state, STARTUP_STATE_KEY, startup_state)


def clear_startup_state(app: Any) -> None:
    if hasattr(app.state, STARTUP_STATE_KEY):
        delattr(app.state, STARTUP_STATE_KEY)


def get_startup_state_from_app(app: Any) -> StartupState:
    startup_state = getattr(app.state, STARTUP_STATE_KEY, None)
    if startup_state is None:
        raise RuntimeError("StartupState is not initialized.")
    return cast(StartupState, startup_state)


def get_startup_state(request: Request) -> StartupState:
    return get_startup_state_from_app(request.app)

