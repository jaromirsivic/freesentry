import asyncio
import importlib
import sys
import types
import unittest
from copy import deepcopy
from unittest import mock

from pydantic import BaseModel
from server.motorerrors import MotorOverrideConflictError


def _reset_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _import_manualcontrol_module(*, initial_settings: dict):
    _reset_modules(
        "server.restapimanualcontrol",
        "server.settingscontroller",
        "server.context",
        "server.common",
    )

    settings_payload = deepcopy(initial_settings)

    settingscontroller = types.ModuleType("server.settingscontroller")

    async def get_settings():
        return deepcopy(settings_payload)

    async def update_settings(mutator):
        mutator(settings_payload)
        return None

    settingscontroller.get_settings = get_settings
    settingscontroller.update_settings = update_settings

    context = types.ModuleType("server.context")
    context.get_master_controller = lambda: None

    common = types.ModuleType("server.common")
    common.epsilon = 0.000001

    class Vector2D(BaseModel):
        x: float = 0.0
        y: float = 0.0

    class Line2D(BaseModel):
        point1: Vector2D | None = None
        point2: Vector2D | None = None

    common.Vector2D = Vector2D
    common.Line2D = Line2D
    common.fit_vector_to_polygon = lambda *, vector, polygon: None
    common.rotate_vector = lambda *, vector, angle: vector

    with mock.patch.dict(
        sys.modules,
        {
            "server.settingscontroller": settingscontroller,
            "server.context": context,
            "server.common": common,
        },
    ):
        module = importlib.import_module("server.restapimanualcontrol")

    return module, settings_payload


class ManualControlApiTests(unittest.TestCase):
    def test_save_manual_control_motors_updates_settings_and_camera(self):
        module, settings_payload = _import_manualcontrol_module(
            initial_settings={
                "manualControl": {
                    "motors": [
                        {"index": 0, "enabled": True, "mode": "joystick"},
                        {"index": 1, "enabled": True, "mode": "slider"},
                    ],
                    "camera": {
                        "selectedCamera": "scope_camera",
                        "streamQuality": 80,
                        "scopeCameraMode": 0,
                        "spotterCamera1Mode": 0,
                        "spotterCamera2Mode": 0,
                        "spotterCamera3Mode": 0,
                    },
                }
            },
        )

        request = module.SaveMotorVisibilityRequest(
            motors=[
                {"index": 0, "enabled": False, "mode": "slider"},
                {"index": 2, "enabled": True, "mode": "joystick"},
            ],
            camera=module.CameraConfig(
                selectedCamera="spotter_camera1",
                streamQuality=65,
                scopeCameraMode=1,
                spotterCamera1Mode=3,
                spotterCamera2Mode=2,
                spotterCamera3Mode=1,
            ),
        )

        response = asyncio.run(module.save_manual_control_motors(request=request))

        self.assertEqual(response, {"success": True})
        self.assertEqual(
            settings_payload["manualControl"]["motors"],
            [
                {"index": 0, "enabled": False, "mode": "slider"},
                {"index": 1, "enabled": True, "mode": "slider"},
                {"index": 2, "enabled": True, "mode": "joystick"},
            ],
        )
        self.assertEqual(
            settings_payload["manualControl"]["camera"],
            {
                "selectedCamera": "spotter_camera1",
                "streamQuality": 65,
                "scopeCameraMode": 1,
                "spotterCamera1Mode": 3,
                "spotterCamera2Mode": 2,
                "spotterCamera3Mode": 1,
            },
        )

    def test_manual_control_action_returns_409_when_override_is_active(self):
        module, _ = _import_manualcontrol_module(
            initial_settings={
                "motors": [
                    {"name": "Left motor", "enabled": True},
                    {"name": "Right motor", "enabled": True},
                ],
                "polygon": [],
                "general": {"joystickSetup": {"rotationAngle": 0}},
            },
        )

        class ConflictMotorsController:
            motors = {}

            def set_motor_speed_by_index(self, *, motor_index: int, speed: float) -> bool:
                raise MotorOverrideConflictError(
                    "Manual hardware override is active on pin 12. "
                    "Stop it before sending managed motor commands."
                )

        request = module.ManualControlActionRequest(
            fullscreen=False,
            joystick=module.Vector2D(x=0.25, y=-0.5),
            motors=[],
        )
        master_controller = types.SimpleNamespace(
            motors_controller=ConflictMotorsController()
        )

        with self.assertRaises(module.HTTPException) as raised:
            asyncio.run(
                module.manual_control_action(
                    request=request,
                    master_controller=master_controller,
                )
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("Manual hardware override is active on pin 12", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
