import threading
import time
from .common import epsilon
from .j8 import J8
from .motor import Motor
from .motorerrors import MotorOverrideConflictError
from .motorlinearactuator import MotorLinearActuator
from .speedhistogram import SpeedHistogram
from .pin import PinType
from .settingscontroller import get_settings_sync


class MotorsController(threading.Thread):
    """
    Thread-based controller for motors. Uses Event-based pause/resume pattern.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self._j8 = J8()
        self._motors: dict[str, Motor] = {}
        self._motors_by_index: dict[int, Motor] = {}
        self._manual_override_pin_index: int | None = None
        self._manual_override_pwm_multiplier: float | None = None
        self._running = False
        self._paused = threading.Event()
        self._paused.set()  # Start in "running" (not paused) state
        self._lock = threading.RLock()
        self._thread_started = False  # Track if thread was ever started
        self._initialize_motors()

    @property
    def j8(self) -> J8:
        return self._j8

    @property
    def motors(self) -> dict[str, Motor]:
        with self._lock:
            return self._motors

    @property
    def manual_override_active(self) -> bool:
        with self._lock:
            return self._manual_override_pin_index is not None

    @property
    def manual_override_pin_index(self) -> int | None:
        with self._lock:
            return self._manual_override_pin_index

    def _resolve_motor_speed(self, *, speed: float) -> float:
        try:
            return float(speed)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed must be a number.") from exc

    def _resolve_pin_index(self, *, pin_index: int) -> int:
        try:
            resolved_pin_index = int(pin_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("pin_index must be an integer.") from exc

        if resolved_pin_index < 0:
            raise ValueError("pin_index must be non-negative.")

        try:
            self._j8[resolved_pin_index]
        except IndexError as exc:
            raise ValueError(f"Unknown J8 pin index: {resolved_pin_index}") from exc

        return resolved_pin_index

    def _resolve_pwm_multiplier(self, *, pwm_multiplier: float) -> float:
        try:
            resolved_pwm_multiplier = float(pwm_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError("pwm_multiplier must be a number.") from exc

        if not 0 <= resolved_pwm_multiplier <= 1:
            raise ValueError("pwm_multiplier must be between 0 and 1.")

        return resolved_pwm_multiplier

    def _motor_is_idle_locked(self, *, motor: Motor) -> bool:
        try:
            current_speed = float(getattr(motor, "current_speed"))
            target_speed = float(getattr(motor, "target_speed"))
        except (AttributeError, TypeError, ValueError):
            return False

        return abs(current_speed) <= epsilon and abs(target_speed) <= epsilon

    def _motors_are_idle_locked(self) -> bool:
        return all(self._motor_is_idle_locked(motor=motor) for motor in self._motors.values())

    def _ensure_no_manual_override_locked(self) -> None:
        active_pin = self._manual_override_pin_index
        if active_pin is not None:
            raise MotorOverrideConflictError(
                f"Manual hardware override is active on pin {active_pin}. "
                "Stop it before sending managed motor commands."
            )

    def _set_motor_speed_locked(self, *, motor: Motor | None, speed: float) -> bool:
        self._ensure_no_manual_override_locked()
        if motor is None:
            return False
        motor.move(speed=speed)
        return True

    def _clear_manual_override_locked(self) -> int | None:
        active_pin = self._manual_override_pin_index
        self._manual_override_pin_index = None
        self._manual_override_pwm_multiplier = None
        if active_pin is not None:
            self._j8[active_pin].reset()
        return active_pin

    def _initialize_motors(self, *, is_hard_reset: bool = False):
        """
        Internal method to initialize motors from settings. Called during init and reset.
        """
        # Load settings
        settings = get_settings_sync()

        if "general" in settings and "controllerSetup" in settings["general"]:
            if is_hard_reset:
                # self._j8.hard_reset()
                self._j8.reset()
            else:
                self._j8.reset()

        # Clear existing motors
        self._motors = {}
        self._motors_by_index = {}

        # Create motors
        if "motors" in settings:
            for motor_index, motor_config in enumerate(settings["motors"]):
                if motor_config.get("enabled", False) and motor_config.get("type") == "linear":
                    try:
                        forward_pin_index = motor_config["forwardPin"]
                        reverse_pin_index = motor_config["reversePin"]
                        forward_enable_pin_index = motor_config["forwardEnablePin"]
                        reverse_enable_pin_index = motor_config["reverseEnablePin"]
                        pwm_frequency = motor_config["pwmFrequency"]
                        
                        forward_pin = self._j8[forward_pin_index]
                        reverse_pin = self._j8[reverse_pin_index]
                        forward_enable_pin = self._j8[forward_enable_pin_index]
                        reverse_enable_pin = self._j8[reverse_enable_pin_index]
                        
                        # Configure pins
                        forward_pin.pin_type = PinType.OUTPUT
                        reverse_pin.pin_type = PinType.OUTPUT

                        # Use the simple histogram list
                        hist_data = motor_config.get("histogram", [])
                        resolution = 1000  # Default
                        speed_hist = SpeedHistogram(speed_histogram=hist_data, resolution=resolution)
                        
                        inertia = motor_config.get("inertia", 0.5)
                        
                        motor = MotorLinearActuator(
                            forward_pin=forward_pin, 
                            forward_enable_pin=forward_enable_pin,
                            reverse_pin=reverse_pin,
                            reverse_enable_pin=reverse_enable_pin,
                            speed_histogram=speed_hist,
                            pwm_frequency=pwm_frequency,
                            inertia=inertia
                        )
                        self._motors[motor_config.get('name')] = motor
                        self._motors_by_index[motor_index] = motor
                    except Exception as e:
                        print(f"Error creating motor {motor_config.get('name')}: {e}")

    def set_motor_speed_by_index(self, *, motor_index: int, speed: float) -> bool:
        """Set a motor target speed by settings index.

        Returns False when the index is invalid, disabled, or not available so
        callers can handle malformed AI motor configs without raising.
        """
        try:
            resolved_index = int(motor_index)
        except (TypeError, ValueError):
            return False

        resolved_speed = self._resolve_motor_speed(speed=speed)

        with self._lock:
            motor = self._motors_by_index.get(resolved_index)
            return self._set_motor_speed_locked(motor=motor, speed=resolved_speed)

    def set_motor_speed_by_name(self, *, motor_name: str, speed: float) -> bool:
        """Set a motor target speed by configured name."""
        if not isinstance(motor_name, str) or len(motor_name.strip()) == 0:
            return False

        resolved_speed = self._resolve_motor_speed(speed=speed)

        with self._lock:
            motor = self._motors.get(motor_name)
            return self._set_motor_speed_locked(motor=motor, speed=resolved_speed)

    def start_manual_override(self, *, pin_index: int, pwm_multiplier: float) -> int:
        """Pause managed motor control and drive a single pin directly for test use."""
        resolved_pin_index = self._resolve_pin_index(pin_index=pin_index)
        resolved_pwm_multiplier = self._resolve_pwm_multiplier(pwm_multiplier=pwm_multiplier)

        with self._lock:
            active_pin = self._manual_override_pin_index
            if active_pin is not None:
                if active_pin != resolved_pin_index:
                    raise MotorOverrideConflictError(
                        f"Manual hardware override is already active on pin {active_pin}. "
                        "Stop it before switching pins."
                    )
                self._manual_override_pwm_multiplier = resolved_pwm_multiplier
                self._j8[resolved_pin_index].value = resolved_pwm_multiplier
                return resolved_pin_index

            if not self._motors_are_idle_locked():
                raise MotorOverrideConflictError(
                    "Manual hardware override requires all managed motors to be idle."
                )

            self.pause()
            try:
                self._j8[resolved_pin_index].value = resolved_pwm_multiplier
            except Exception:
                self._paused.set()
                raise

            self._manual_override_pin_index = resolved_pin_index
            self._manual_override_pwm_multiplier = resolved_pwm_multiplier
            return resolved_pin_index

    def stop_manual_override(self, *, pin_index: int | None = None) -> int | None:
        """Stop the active manual override and resume managed motor control."""
        if pin_index is not None:
            self._resolve_pin_index(pin_index=pin_index)

        with self._lock:
            active_pin = self._manual_override_pin_index
            if active_pin is None:
                return None

            cleanup_error: Exception | None = None
            try:
                self._clear_manual_override_locked()
            except Exception as exc:
                cleanup_error = exc
            finally:
                self.resume()

            if cleanup_error is not None:
                raise cleanup_error

            return active_pin

    def reset(self, *, is_hard_reset: bool = False):
        """
        Delete all motors, reset J8, and create new motors based on settings.json.
        Uses pause/resume to safely reset while thread is running.
        """
        with self._lock:
            self.pause()
            try:
                self._clear_manual_override_locked()
                self._initialize_motors(is_hard_reset=is_hard_reset)
            finally:
                self.resume()

    def pause(self):
        """
        Pause the motor execution loop. Thread stays alive but waits.
        """
        self._paused.clear()

    def resume(self):
        """
        Resume the motor execution loop after pause.
        """
        with self._lock:
            if self._manual_override_pin_index is not None:
                return
            self._paused.set()

    def run(self):
        """
        Thread's main loop. Calls go() method on each motor.
        Waits when paused, exits when _running is False.
        """
        while self._running:
            # Wait if paused (blocks until resume is called)
            self._paused.wait()
            # Check _running again after waking up (in case stop was called)
            if not self._running:
                break
            with self._lock:
                for name, motor in self._motors.items():
                    try:
                        motor.go()
                    except Exception as e:
                        print(f"Error in motor execution: {e}")
            # Sleep to avoid 100% CPU usage
            time.sleep(0.001)

    def start(self):
        """
        Start the thread. Can only be called once per instance.
        """
        if self._thread_started:
            return
        self._running = True
        self._thread_started = True
        super().start()

    def stop(self):
        """
        Stop the execution, release motors and J8.
        """
        self._running = False
        # Resume if paused to allow thread to exit
        self._paused.set()
        if self._thread_started and self.is_alive():
            self.join(timeout=1.0)
        
        with self._lock:
            self._clear_manual_override_locked()
            # Delete all motors
            self._motors = {}
            self._motors_by_index = {}
            # Release J8
            self._j8.release()
