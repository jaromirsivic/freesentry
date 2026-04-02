from typing import TYPE_CHECKING, Any, cast

from fastapi import Request

MASTER_CONTROLLER_STATE_KEY = "master_controller"

if TYPE_CHECKING:
    from .mastercontroller import MasterController


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

