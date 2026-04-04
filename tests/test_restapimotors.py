import asyncio
import importlib
import sys
import types
import unittest
from unittest import mock

from server.motorerrors import MotorOverrideConflictError


def _reset_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def _import_restapimotors_module():
    _reset_modules(
        "server.restapimotors",
        "server.settingscontroller",
        "server.context",
    )

    settingscontroller = types.ModuleType("server.settingscontroller")

    async def get_settings():
        return {"motors": []}

    async def update_settings(mutator):
        settings = {"motors": []}
        mutator(settings)
        return None

    settingscontroller.get_settings = get_settings
    settingscontroller.update_settings = update_settings

    context = types.ModuleType("server.context")
    context.get_master_controller = lambda: None

    with mock.patch.dict(
        sys.modules,
        {
            "server.settingscontroller": settingscontroller,
            "server.context": context,
        },
    ):
        return importlib.import_module("server.restapimotors")


class FakeMotorsController:
    def __init__(
        self,
        *,
        start_exc: Exception | None = None,
        stop_result: int | None = None,
        stop_exc: Exception | None = None,
        speed_result: bool = True,
        speed_exc: Exception | None = None,
    ):
        self.start_exc = start_exc
        self.stop_result = stop_result
        self.stop_exc = stop_exc
        self.speed_result = speed_result
        self.speed_exc = speed_exc
        self.start_calls: list[tuple[int, float]] = []
        self.stop_calls: list[int] = []
        self.speed_calls: list[tuple[str, float]] = []
        self.j8 = types.SimpleNamespace(initialized=True, error_message="")

    def start_manual_override(self, *, pin_index: int, pwm_multiplier: float) -> int:
        self.start_calls.append((pin_index, pwm_multiplier))
        if self.start_exc is not None:
            raise self.start_exc
        return pin_index

    def stop_manual_override(self, *, pin_index: int | None = None) -> int | None:
        if pin_index is not None:
            self.stop_calls.append(pin_index)
        if self.stop_exc is not None:
            raise self.stop_exc
        return self.stop_result

    def set_motor_speed_by_name(self, *, motor_name: str, speed: float) -> bool:
        self.speed_calls.append((motor_name, float(speed)))
        if self.speed_exc is not None:
            raise self.speed_exc
        return self.speed_result

    def reset(self):
        return None


class RestApiMotorsTests(unittest.TestCase):
    def test_start_motor_action_returns_409_when_override_busy(self):
        module = _import_restapimotors_module()
        controller = FakeMotorsController(
            start_exc=MotorOverrideConflictError(
                "Manual hardware override requires all managed motors to be idle."
            )
        )
        master_controller = types.SimpleNamespace(motors_controller=controller)

        with self.assertRaises(module.HTTPException) as raised:
            asyncio.run(
                module.start_motor_action(
                    request=module.MotorActionStartRequest(pin_index=7, pwm_multiplier=0.4),
                    master_controller=master_controller,
                )
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(
            raised.exception.detail,
            "Manual hardware override requires all managed motors to be idle.",
        )
        self.assertEqual(controller.start_calls, [(7, 0.4)])

    def test_stop_motor_action_reports_stopped_pin(self):
        module = _import_restapimotors_module()
        controller = FakeMotorsController(stop_result=6)
        master_controller = types.SimpleNamespace(motors_controller=controller)

        response = asyncio.run(
            module.stop_motor_action(
                request=module.MotorActionStopRequest(pin_index=6),
                master_controller=master_controller,
            )
        )

        self.assertEqual(
            response,
            {
                "success": True,
                "pin_index": 6,
                "message": "Manual hardware override on pin 6 stopped",
            },
        )
        self.assertEqual(controller.stop_calls, [6])

    def test_set_motor_speed_returns_409_when_override_is_active(self):
        module = _import_restapimotors_module()
        controller = FakeMotorsController(
            speed_exc=MotorOverrideConflictError(
                "Manual hardware override is active on pin 4. "
                "Stop it before sending managed motor commands."
            )
        )
        master_controller = types.SimpleNamespace(motors_controller=controller)

        with self.assertRaises(module.HTTPException) as raised:
            asyncio.run(
                module.set_motor_speed(
                    request=module.MotorSpeedRequest(motor_name="Traverse", speed=0.8),
                    master_controller=master_controller,
                )
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("Manual hardware override is active on pin 4", raised.exception.detail)
        self.assertEqual(controller.speed_calls, [("Traverse", 0.8)])


if __name__ == "__main__":
    unittest.main()
