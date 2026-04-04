import importlib
import sys
import threading
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

from server import aiagent, common


class FakeMotor:
    def __init__(self, *, current_speed: float = 0.0, target_speed: float = 0.0):
        self.speeds: list[float] = []
        self.current_speed = current_speed
        self.target_speed = target_speed

    def move(self, *, speed: float):
        resolved_speed = float(speed)
        self.speeds.append(resolved_speed)
        self.target_speed = resolved_speed


class FakePin:
    def __init__(self):
        self.value = 0.0
        self.reset_calls = 0

    def reset(self):
        self.value = 0.0
        self.reset_calls += 1


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


def _make_result(*, timestamp: float = 0.0) -> aiagent.EngagementResult:
    return aiagent.EngagementResult(
        timestamp=timestamp,
        is_valid=False,
        activation_date_time_str="",
        activation_date_time_condition_satisfied=False,
        fps=0,
        min_fps_to_allow_engagement=0,
        fps_satisfied=False,
        immediate_engagement_condition_satisfied=False,
        engaging=False,
        engagement_counter=0,
        time_until_current_engagement_ends=0,
        exit_strategy_under_execution=False,
        exit_strategy_executed=False,
        status=aiagent.EngagementStatus.NOT_ENGAGING,
    )


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

    def test_check_activation_datetime_accepts_z_suffix_and_normalizes_result(self):
        agent, _ = self._make_agent(available_indexes=set())
        result = _make_result()

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                if tz is None:
                    return current.replace(tzinfo=None)
                return current.astimezone(tz)

        with mock.patch("server.aiagent.datetime", FixedDatetime):
            now_dt = agent._check_activation_datetime(
                ai_setup={"activationDateTime": "2026-01-01T00:00:00Z"},
                result=result,
            )

        self.assertIsNotNone(now_dt)
        self.assertEqual(result.activation_date_time_str, "2026-01-01T00:00:00+00:00")
        self.assertTrue(result.activation_date_time_condition_satisfied)

    def test_engage_triggers_exit_strategy_with_z_fixed_datetime(self):
        agent, motors_controller = self._make_agent(available_indexes={0})
        settings = _make_settings(mission_motors=[])
        settings["aiSetup"]["activationDateTime"] = "2000-01-01T00:00:00+00:00"
        settings["aiSetup"]["exitStrategy"] = {
            "maxEngagements": 100,
            "timeoutAfterFirstEngagement": 1000.0,
            "fixedDateTime": "2029-12-31T23:59:59Z",
            "motors": [{"enabled": True, "index": 0, "speed": 0.5}],
        }

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                if tz is None:
                    return current.replace(tzinfo=None)
                return current.astimezone(tz)

        with mock.patch("server.aiagent.datetime", FixedDatetime):
            with mock.patch("server.aiagent.time.time", return_value=1000.0):
                result = agent.engage(frame=_make_frame(), settings=settings)

        self.assertEqual(result.status, aiagent.EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION)
        self.assertEqual(motors_controller.commands, [(0, 0.5)])


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

    def _make_controller(
        self,
        motorscontroller,
        *,
        motors_by_index: dict[int, FakeMotor] | None = None,
        motors_by_name: dict[str, FakeMotor] | None = None,
        pin_count: int = 8,
    ):
        controller = motorscontroller.MotorsController.__new__(motorscontroller.MotorsController)
        controller._lock = threading.RLock()
        controller._j8 = [FakePin() for _ in range(pin_count)]
        controller._motors_by_index = motors_by_index or {}
        controller._motors = motors_by_name or {}
        controller._manual_override_pin_index = None
        controller._manual_override_pwm_multiplier = None
        controller._paused = threading.Event()
        controller._paused.set()
        controller._running = False
        controller._thread_started = False
        return controller

    def test_set_motor_speed_by_index_moves_matching_motor(self):
        motorscontroller = self._import_motorscontroller()
        motor = FakeMotor()
        controller = self._make_controller(
            motorscontroller,
            motors_by_index={2: motor},
        )

        applied = controller.set_motor_speed_by_index(
            motor_index=2,
            speed=0.75,
        )

        self.assertTrue(applied)
        self.assertEqual(motor.speeds, [0.75])

    def test_set_motor_speed_by_index_returns_false_for_unknown_index(self):
        motorscontroller = self._import_motorscontroller()
        motor = FakeMotor()
        controller = self._make_controller(
            motorscontroller,
            motors_by_index={0: motor},
        )

        applied = controller.set_motor_speed_by_index(
            motor_index=5,
            speed=0.75,
        )

        self.assertFalse(applied)
        self.assertEqual(motor.speeds, [])

    def test_start_manual_override_pauses_loop_and_sets_pin(self):
        motorscontroller = self._import_motorscontroller()
        controller = self._make_controller(motorscontroller)

        active_pin = controller.start_manual_override(pin_index=2, pwm_multiplier=0.4)

        self.assertEqual(active_pin, 2)
        self.assertTrue(controller.manual_override_active)
        self.assertEqual(controller.manual_override_pin_index, 2)
        self.assertFalse(controller._paused.is_set())
        self.assertEqual(controller._j8[2].value, 0.4)

    def test_stop_manual_override_resets_pin_and_resumes_loop(self):
        motorscontroller = self._import_motorscontroller()
        controller = self._make_controller(motorscontroller)
        controller.start_manual_override(pin_index=3, pwm_multiplier=0.6)

        stopped_pin = controller.stop_manual_override(pin_index=3)

        self.assertEqual(stopped_pin, 3)
        self.assertFalse(controller.manual_override_active)
        self.assertIsNone(controller.manual_override_pin_index)
        self.assertTrue(controller._paused.is_set())
        self.assertEqual(controller._j8[3].value, 0.0)
        self.assertEqual(controller._j8[3].reset_calls, 1)

    def test_start_manual_override_refuses_busy_motor(self):
        motorscontroller = self._import_motorscontroller()
        busy_motor = FakeMotor(current_speed=0.2, target_speed=0.2)
        controller = self._make_controller(
            motorscontroller,
            motors_by_name={"leftArm": busy_motor},
        )

        with self.assertRaises(motorscontroller.MotorOverrideConflictError):
            controller.start_manual_override(pin_index=1, pwm_multiplier=0.5)

        self.assertTrue(controller._paused.is_set())
        self.assertFalse(controller.manual_override_active)

    def test_set_motor_speed_by_index_refuses_commands_during_manual_override(self):
        motorscontroller = self._import_motorscontroller()
        motor = FakeMotor()
        controller = self._make_controller(
            motorscontroller,
            motors_by_index={0: motor},
            motors_by_name={"leftArm": motor},
        )
        controller.start_manual_override(pin_index=2, pwm_multiplier=0.5)

        with self.assertRaises(motorscontroller.MotorOverrideConflictError):
            controller.set_motor_speed_by_index(motor_index=0, speed=0.75)

        self.assertEqual(motor.speeds, [])


if __name__ == "__main__":
    unittest.main()
