import importlib
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np

from server import aiagent, common


class FakeMotor:
    def __init__(self):
        self.speeds: list[float] = []

    def move(self, *, speed: float):
        self.speeds.append(speed)


class FakeMotorsController:
    def __init__(self, *, available_indexes: set[int]):
        self.available_indexes = set(available_indexes)
        self.commands: list[tuple[int, float]] = []

    def set_motor_speed_by_index(self, *, motor_index: int, speed: float) -> bool:
        if motor_index not in self.available_indexes:
            return False
        self.commands.append((motor_index, float(speed)))
        return True


def _make_pose(*, visible: bool = True) -> dict[str, common.AICircle]:
    center = common.Vector2D(x=5.0, y=5.0) if visible else common.Vector2D(x=50.0, y=50.0)
    def make_organ() -> common.AICircle:
        return common.AICircle(
            center=center,
            radius=5.0,
            confidence=1.0,
            confidence_threshold=0.5,
            confidence_achieved=True,
        )

    return {
        "brain": make_organ(),
        "chest": make_organ(),
        "abdomen": make_organ(),
        "liver": make_organ(),
        "heart": make_organ(),
    }


def _make_frame() -> common.Frame:
    return common.Frame(
        valid=True,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        time=0.0,
        pose=_make_pose(),
    )


def _make_settings(*, mission_motors: list[dict] | None = None) -> dict:
    return {
        "cameras": {
            "scope_camera": {
                "static_reticle_x": 0.5,
                "static_reticle_y": 0.5,
            }
        },
        "aiSetup": {
            "activationDateTime": "2000-01-01T00:00:00",
            "minFpsToAllowEngagement": 0,
            "organMustBeVisibleSeconds": 0.2,
            "missions": {
                "randomWalk": {
                    "engagementDuration": 0.2,
                    "delayBetweenEngagements": 0.2,
                    "motors": mission_motors
                    if mission_motors is not None
                    else [
                        {"enabled": True, "index": 0, "speed": 0.6},
                        {"enabled": True, "index": 1, "speed": -0.4},
                    ],
                }
            },
            "exitStrategy": {
                "maxEngagements": 100,
                "timeoutAfterFirstEngagement": 1000.0,
                "fixedDateTime": "2199-12-31T23:59:59",
                "motors": [],
            },
        },
    }


class AIAgentMotorTests(unittest.TestCase):
    def _make_agent(self, *, available_indexes: set[int]) -> tuple[aiagent.AIAgent, FakeMotorsController]:
        motors_controller = FakeMotorsController(available_indexes=available_indexes)
        master_controller = types.SimpleNamespace(motors_controller=motors_controller)
        return aiagent.AIAgent(master_controller=master_controller), motors_controller

    def test_engagement_state_machine_applies_start_and_stop_motor_commands(self):
        agent, motors_controller = self._make_agent(available_indexes={0, 1})
        settings = _make_settings()
        timestamps = [1000.0, 1000.1, 1000.4, 1000.65, 1000.9]

        with mock.patch("server.aiagent.time.time", side_effect=timestamps):
            results = [agent.engage(frame=_make_frame(), settings=settings) for _ in timestamps]

        self.assertEqual(
            [result.status for result in results],
            [
                aiagent.EngagementStatus.ARMING,
                aiagent.EngagementStatus.ARMING,
                aiagent.EngagementStatus.ENGAGING,
                aiagent.EngagementStatus.DISENGAGING,
                aiagent.EngagementStatus.NOT_ENGAGING,
            ],
        )
        self.assertEqual(
            motors_controller.commands,
            [
                (0, 0.6),
                (1, -0.4),
                (0, 0.0),
                (1, 0.0),
            ],
        )

    def test_apply_motors_logs_unavailable_index_without_raising(self):
        agent, motors_controller = self._make_agent(available_indexes={0})

        with mock.patch("builtins.print") as print_mock:
            agent._apply_motors(
                motors_config=[{"enabled": True, "index": 99, "speed": 0.6}],
                use_speed=True,
            )

        self.assertEqual(motors_controller.commands, [])
        print_mock.assert_any_call("AI motor config references unavailable motor index 99")


class MotorsControllerApiTests(unittest.TestCase):
    def _import_motorscontroller(self):
        gpiozero = types.ModuleType("gpiozero")

        class PWMOutputDevice:
            def __init__(self, *args, **kwargs):
                self.value = kwargs.get("initial_value", 0)
                self.frequency = kwargs.get("frequency", 0)

            def off(self):
                self.value = 0

            def close(self):
                return None

        gpiozero.PWMOutputDevice = PWMOutputDevice

        for module_name in ("server.motorscontroller", "server.j8", "server.pin"):
            sys.modules.pop(module_name, None)

        with mock.patch.dict(sys.modules, {"gpiozero": gpiozero}):
            return importlib.import_module("server.motorscontroller")

    def test_set_motor_speed_by_index_moves_matching_motor(self):
        motorscontroller = self._import_motorscontroller()
        motor = FakeMotor()
        controller = types.SimpleNamespace(
            _lock=threading.RLock(),
            _motors_by_index={2: motor},
        )

        applied = motorscontroller.MotorsController.set_motor_speed_by_index(
            controller,
            motor_index=2,
            speed=0.75,
        )

        self.assertTrue(applied)
        self.assertEqual(motor.speeds, [0.75])

    def test_set_motor_speed_by_index_returns_false_for_unknown_index(self):
        motorscontroller = self._import_motorscontroller()
        motor = FakeMotor()
        controller = types.SimpleNamespace(
            _lock=threading.RLock(),
            _motors_by_index={0: motor},
        )

        applied = motorscontroller.MotorsController.set_motor_speed_by_index(
            controller,
            motor_index=5,
            speed=0.75,
        )

        self.assertFalse(applied)
        self.assertEqual(motor.speeds, [])


if __name__ == "__main__":
    unittest.main()
