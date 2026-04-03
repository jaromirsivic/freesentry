import threading
import time
from .j8 import J8
from .motor import Motor
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
            resolved_speed = float(speed)
        except (TypeError, ValueError):
            return False

        with self._lock:
            motor = self._motors_by_index.get(resolved_index)
            if motor is None:
                return False
            motor.move(speed=resolved_speed)
            return True

    def reset(self, *, is_hard_reset: bool = False):
        """
        Delete all motors, reset J8, and create new motors based on settings.json.
        Uses pause/resume to safely reset while thread is running.
        """
        with self._lock:
            self.pause()
            try:
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
        self.resume()
        if self._thread_started and self.is_alive():
            self.join(timeout=1.0)
        
        with self._lock:
            # Delete all motors
            self._motors = {}
            self._motors_by_index = {}
            # Release J8
            self._j8.release()
