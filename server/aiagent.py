from ast import Return
from collections import deque
from .settingscontroller import get_settings_sync
from .common import Vector2D, Circle, AICircle
import time
import random
import numpy as np
from datetime import datetime
from .common import Frame
from pydantic import BaseModel
from enum import Enum
import cv2
import threading

# class EngagementHistory:
#     def __init__(self):
#         self._timestamp = time.time()

class EngagementStatus(Enum):
    WAITING_TO_START = "waiting_to_start" # the system is waiting to predefined activation date time
    FPS_CONDITION_NOT_SATISFIED = "fps_condition_not_satisfied" # the fps is not satisfied
    NOT_ENGAGING = "not_engaging" # the system is not engaging - default status if the system is not waiting to start and FPS is satisfied
    ARMING = "arming" # the system is arming - meaning the organ is not yet visible for predefined duration "organMustBeVisibleSeconds"
    ENGAGING = "engaging" # the system is engaging - meaning the organ is visible for predefined duration "organMustBeVisibleSeconds"
    DISENGAGING = "disengaging" # the system is disengaging - meaning engaging happened for predefined duration "engagementDuration"
    EXIT_STRATEGY_UNDER_EXECUTION = "exit_strategy_under_execution" # the exit strategy is under execution - meaning the exit strategy is under execution
    EXIT_STRATEGY_EXECUTED = "exit_strategy_executed" # the exit strategy was executed - meaning the exit strategy was executed

class EngagementResult(BaseModel):
    # whether the engagement result is valids
    is_valid: bool
    # timestamp of the engagement result
    timestamp: float
    # activation date time string from the ai setup
    activation_date_time_str: str
    # whether the activation date time condition is satisfied
    activation_date_time_condition_satisfied: bool
    # fps from the last two seconds of the immediate engagement history
    fps: float
    # minimum fps to allow engagement
    min_fps_to_allow_engagement: float
    # whether the fps is greater than the minimum fps to allow engagement
    fps_satisfied: bool
    # whether the immediate engagement condition is satisfied
    immediate_engagement_condition_satisfied: bool
    # whether the AI agent is engaging
    engaging: bool
    # engagement counter
    engagement_counter: int
    # time until the current engagement ends
    time_until_current_engagement_ends: float
    # whether the exit strategy is under execution
    exit_strategy_under_execution: bool
    # whether the exit strategy was executed
    exit_strategy_executed: bool
    # status of the engagement
    status: EngagementStatus

class AIAgent(threading.Thread):
    def __init__(self, *, master_controller: "MasterController"):  # pyright: ignore[reportUndefinedVariable]
        super().__init__(daemon=True)
        self._master_controller = master_controller
        self._engagement_history: list[EngagementResult] = []
        self._immediate_engagement_history: deque[EngagementResult] = deque(maxlen=1000)
        self._latest_status: EngagementStatus = EngagementStatus.NOT_ENGAGING
        self._latest_processed_timestamp: float = 0
        self._engaging_started_at: float = 0
        self._disengaging_started_at: float = 0
        self._running = False
        self._paused = threading.Event()
        self._paused.set()
        self._stop_event = threading.Event()
        self._thread_started = False

    ORGAN_NAMES = ("brain", "chest", "abdomen", "liver", "heart")

    def start(self) -> None:
        """Start the worker thread exactly once for this instance."""
        if self._thread_started:
            return
        self._running = True
        self._stop_event.clear()
        self._paused.set()
        self._thread_started = True
        super().start()

    def _get_motor_by_role(self, *, role: str) -> "Motor | None":  # pyright: ignore[reportUndefinedVariable]
        """Return the Motor instance whose settings entry has the given role, or None."""
        try:
            settings = get_settings_sync()
            for motor_config in settings.get("motors", []):
                if motor_config.get("role") == role and motor_config.get("enabled", False):
                    name = motor_config.get("name")
                    return self._master_controller.motors_controller.motors.get(name)
        except Exception as e:
            print(f"Error looking up motor for role '{role}': {e}")
        return None

    def _stop_arm_motors(self) -> None:
        """Set both arm motors to speed 0."""
        for role in ("leftArm", "rightArm"):
            motor = self._get_motor_by_role(role=role)
            if motor is not None:
                motor.move(speed=0)

    def run(self) -> None:
        """Thread main loop: randomly move leftArm and rightArm motors."""
        loop_count = 0
        direction: Vector2D | None = None

        while self._running:
            if self._stop_event.wait(timeout=1.0):
                break

            self._paused.wait()
            if not self._running:
                break

            duration = random.uniform(1, 4)

            if loop_count == 0 or random.random() >= 0.5:
                direction = Vector2D(
                    x=random.uniform(-1.0, 1.0),
                    y=random.uniform(-1.0, 1.0),
                )

            left_motor = self._get_motor_by_role(role="leftArm")
            right_motor = self._get_motor_by_role(role="rightArm")

            # if left_motor is not None and direction is not None:
            #     left_motor.move(speed=direction.x)
            # if right_motor is not None and direction is not None:
            #     right_motor.move(speed=direction.y)

            # if self._stop_event.wait(timeout=duration):
            #     self._stop_arm_motors()
            #     break

            # self._stop_arm_motors()
            loop_count += 1

    def stop(self) -> None:
        """Stop the thread and ensure motors are halted."""
        self._running = False
        self._stop_event.set()
        self._paused.set()
        if self.is_alive():
            self.join(timeout=2.0)

    def pause(self) -> None:
        """Pause the random movement loop and stop both arm motors."""
        self._paused.clear()
        self._stop_arm_motors()

    def resume(self) -> None:
        """Resume the random movement loop after a pause."""
        self._paused.set()

    def _is_condition_for_immediate_engagement_satisfied(self, *, pose: dict | list, reticle_position: Vector2D, ai_setup: dict) -> bool:
        """Check whether any organ in any detected pose satisfies the immediate engagement condition."""
        poses = pose if isinstance(pose, list) else [pose]
        organs = ai_setup.get("organs", {})
        detection_radius_from_reticle = 0

        for p in poses:
            if p is None:
                continue
            for organ_name in self.ORGAN_NAMES:
                organ_settings = organs.get(organ_name, {})
                if not organ_settings.get("enabled", True):
                    continue
                size_multiplier = float(organ_settings.get("sizeMultiplier", 1.0))
                minimum_radius = int(organ_settings.get("minimumRadius", 1))
                organ: AICircle = p[organ_name]
                if (organ.radius >= minimum_radius
                        and organ.confidence >= organ.confidence_threshold
                        and organ.is_point_inside(point=reticle_position,
                                                  radiusMultiplier=size_multiplier,
                                                  radiusDelta=detection_radius_from_reticle)):
                    return True
        return False

    def _get_fps(self, *, now: float) -> float:
        """
        Get the FPS from the immediate engagement history.
        """
        now_minus_2_seconds = now - 2
        frames_during_last_2_seconds = 0
        for i in range(len(self._immediate_engagement_history)-1, 0, -1):
            if self._immediate_engagement_history[i].timestamp < now_minus_2_seconds:
                break
            frames_during_last_2_seconds += 1
        return frames_during_last_2_seconds / 2

    def _get_organ_visible_duration(self, *, now: float, organ_must_be_visible_seconds: float) -> float:
        """
        Check if the organ is visible for the duration.
        """
        organ_was_visible = now
        # iterate over the immediate engagement history in reverse order
        for i in range(len(self._immediate_engagement_history)-1, 0, -1):
            # first check if the engagement was active
            if self._immediate_engagement_history[i].immediate_engagement_condition_satisfied:
                organ_was_visible = self._immediate_engagement_history[i].timestamp
            else:
                break
            # then check if we are still within the organ must be visible seconds
            if self._immediate_engagement_history[i].timestamp < now - organ_must_be_visible_seconds:
                break
        # return the duration the organ was visible
        return now - organ_was_visible

    def _check_activation_datetime(self, *, ai_setup: dict, result: EngagementResult) -> datetime | None:
        """Check whether the activation datetime has been reached.

        Returns the current datetime if the activation condition is satisfied,
        or None if the activation time is still in the future (result is updated accordingly).
        """
        activation_date_time = ai_setup.get("activationDateTime", "2026-01-01T00:00:00.000Z")
        result.activation_date_time_str = activation_date_time
        activation_date_time_dt = datetime.fromisoformat(activation_date_time)
        try:
            now_dt = datetime.now(activation_date_time_dt.tzinfo) if activation_date_time_dt.tzinfo else datetime.now()
            if now_dt < activation_date_time_dt:
                result.activation_date_time_condition_satisfied = False
                result.status = EngagementStatus.WAITING_TO_START
                return None
        except Exception:
            now_dt = datetime.now()
            if now_dt.replace(tzinfo=None) < activation_date_time_dt.replace(tzinfo=None):
                result.activation_date_time_condition_satisfied = False
                result.status = EngagementStatus.WAITING_TO_START
                return None
        result.activation_date_time_condition_satisfied = True
        return now_dt

    def _compute_next_status(self, *, result: EngagementResult, now: float, now_dt: datetime, ai_setup: dict) -> None:
        """Run the state machine to determine the next engagement status.

        State diagram:
        NOT_ENGAGING -> ARMING (organ visible, immediate_engagement_condition_satisfied)
        ARMING -> NOT_ENGAGING (immediate_engagement_condition_satisfied becomes False)
        ARMING -> ENGAGING (organ visible for organMustBeVisibleSeconds)
        ENGAGING -> DISENGAGING (after engagementDuration elapsed)
        DISENGAGING -> NOT_ENGAGING (after delayBetweenEngagements elapsed)
        Any state -> EXIT_STRATEGY_UNDER_EXECUTION (exit conditions met)
        FPS_CONDITION_NOT_SATISFIED prevents ARMING and ENGAGING.
        """
        random_walk = ai_setup.get("missions", {}).get("randomWalk", {})
        engagement_duration = random_walk.get("engagementDuration", 0.5)
        delay_between_engagements = random_walk.get("delayBetweenEngagements", 0.5)
        organ_must_be_visible_seconds = ai_setup.get("organMustBeVisibleSeconds", 0)

        exit_strategy = ai_setup.get("exitStrategy", {})
        max_engagements = exit_strategy.get("maxEngagements", 1000000)
        timeout_after_first = exit_strategy.get("timeoutAfterFirstEngagement", 1000000)
        fixed_date_time_str = exit_strategy.get("fixedDateTime", "2199-12-31T23:59:59Z")

        total_engagements = len(self._engagement_history)
        first_engagement = self._engagement_history[0] if total_engagements > 0 else None

        # Reset to NOT_ENGAGING if more than 1 second elapsed since last processed frame
        if now - self._latest_processed_timestamp > 1:
            self._latest_status = EngagementStatus.NOT_ENGAGING
        self._latest_processed_timestamp = now

        # --- ENGAGING continuation (duration-based, ignores immediate_engagement_condition) ---
        if self._latest_status == EngagementStatus.ENGAGING:
            if now - self._engaging_started_at < engagement_duration:
                result.status = EngagementStatus.ENGAGING
                return
            # Engagement duration elapsed -> transition to DISENGAGING
            result.status = EngagementStatus.DISENGAGING
            self._latest_status = EngagementStatus.DISENGAGING
            self._disengaging_started_at = now
            return

        # --- DISENGAGING continuation ---
        if self._latest_status == EngagementStatus.DISENGAGING:
            if now - self._disengaging_started_at < delay_between_engagements:
                result.status = EngagementStatus.DISENGAGING
                return
            # Delay elapsed -> transition to NOT_ENGAGING
            result.status = EngagementStatus.NOT_ENGAGING
            self._latest_status = EngagementStatus.NOT_ENGAGING
            return

        # --- Exit strategy (sticky once entered) ---
        if self._latest_status == EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION:
            result.status = EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION
            return

        # --- Check exit strategy triggers ---
        exit_triggered = (
            total_engagements >= max_engagements
            or (first_engagement is not None and (now - first_engagement.timestamp) >= timeout_after_first)
        )
        if not exit_triggered:
            try:
                exit_triggered = datetime.fromisoformat(fixed_date_time_str) <= now_dt
            except Exception:
                pass

        if exit_triggered:
            result.status = EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION
            self._latest_status = EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION
            return

        # --- FPS gate: prevents entering ARMING or ENGAGING ---
        if not result.fps_satisfied:
            result.status = EngagementStatus.FPS_CONDITION_NOT_SATISFIED
            self._latest_status = EngagementStatus.FPS_CONDITION_NOT_SATISFIED
            return

        # --- Organ not visible: reset to NOT_ENGAGING (breaks arming) ---
        if not result.immediate_engagement_condition_satisfied:
            result.status = EngagementStatus.NOT_ENGAGING
            self._latest_status = EngagementStatus.NOT_ENGAGING
            return

        # --- Organ IS visible: check arming duration ---
        organ_visible_duration = self._get_organ_visible_duration(
            now=now,
            organ_must_be_visible_seconds=organ_must_be_visible_seconds
        )

        if organ_visible_duration >= organ_must_be_visible_seconds:
            # Arming complete -> start ENGAGING, record engagement once
            result.status = EngagementStatus.ENGAGING
            self._latest_status = EngagementStatus.ENGAGING
            self._engaging_started_at = now
            self._engagement_history.append(result)
        else:
            # Still arming
            result.status = EngagementStatus.ARMING
            self._latest_status = EngagementStatus.ARMING

    def engage(self, *, frame: Frame, settings: dict) -> EngagementResult:
        """Compute the engagement result."""
        result = EngagementResult(
            timestamp=time.time(),
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
            status=EngagementStatus.FPS_CONDITION_NOT_SATISFIED
        )
        ai_setup = settings.get("aiSetup", {})
        if frame.pose is None:
            return result
        pose = frame.pose

        # Check activation datetime (early return if not yet activated)
        now_dt = self._check_activation_datetime(ai_setup=ai_setup, result=result)
        if now_dt is None:
            return result

        # Check immediate engagement condition
        reticle_position = self._get_reticle_position(image=frame.image, settings=settings)
        result.immediate_engagement_condition_satisfied = self._is_condition_for_immediate_engagement_satisfied(
            pose=pose, reticle_position=reticle_position, ai_setup=ai_setup
        )
        # Add to history (deque auto-evicts oldest)
        self._immediate_engagement_history.append(result)

        # Compute FPS
        now = result.timestamp
        result.min_fps_to_allow_engagement = ai_setup.get("minFpsToAllowEngagement", 0)
        result.fps = self._get_fps(now=now)
        result.fps_satisfied = result.fps >= result.min_fps_to_allow_engagement

        # Compute next status via state machine
        old_status = self._latest_status
        self._compute_next_status(result=result, now=now, now_dt=now_dt, ai_setup=ai_setup)

        result.is_valid = True
        result.engaging = result.status == EngagementStatus.ENGAGING
        result.engagement_counter = len(self._engagement_history)
        result.exit_strategy_under_execution = result.status == EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION

        # Fire status change callback if status changed
        if result.status != old_status:
            self._random_walk_on_status_changed(new_status=result.status, old_status=old_status, settings=settings)
        return result

    def _apply_motors(self, *, motors_config: list[dict], use_speed: bool) -> None:
        """Set positions for all enabled motors in *motors_config*.

        When *use_speed* is True each motor is set to its configured speed;
        otherwise it is set to 0 (stopped).
        """
        for motor in motors_config:
            if motor.get("enabled", False):
                position = motor.get("speed", 0) if use_speed else 0
                self._master_controller.motors_controller.set_motor_position(
                    motor_index=motor.get("index", 0), position=position
                )

    def _random_walk_on_status_changed(self, *, new_status: EngagementStatus, old_status: EngagementStatus, settings: dict):
        """Called when the engagement status changes to start/stop motors."""
        try:
            ai_setup = settings.get("aiSetup", {})
            mission_motors = ai_setup.get("missions", {}).get("randomWalk", {}).get("motors", [])
            exit_strategy_motors = ai_setup.get("exitStrategy", {}).get("motors", [])

            if new_status == EngagementStatus.ENGAGING and old_status != EngagementStatus.ENGAGING:
                self._apply_motors(motors_config=mission_motors, use_speed=True)
            elif new_status == EngagementStatus.NOT_ENGAGING and old_status != EngagementStatus.NOT_ENGAGING:
                self._apply_motors(motors_config=mission_motors, use_speed=False)

            if new_status == EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION and old_status != EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION:
                self._apply_motors(motors_config=exit_strategy_motors, use_speed=True)
            elif new_status == EngagementStatus.EXIT_STRATEGY_EXECUTED and old_status != EngagementStatus.EXIT_STRATEGY_EXECUTED:
                self._apply_motors(motors_config=exit_strategy_motors, use_speed=False)
        except Exception as e:
            print(f"Error in _random_walk_on_status_changed: {e}")

    def _get_reticle_position(self, *, image:np.ndarray, settings: dict) -> Vector2D:
        """
        Get the reticle position in the pixel coordinate system.
        """
        # get the reticle position in the CV2 coordinate system
        reticle_position_cv2_x = settings.get("cameras", {}).get("scope_camera", {}).get("static_reticle_x", 0.5)
        reticle_position_cv2_y = settings.get("cameras", {}).get("scope_camera", {}).get("static_reticle_y", 0.5)
        reticle_position_cv2 = Vector2D(x=reticle_position_cv2_x, y=reticle_position_cv2_y)
        # get the resolution of the image
        resolution = image.shape[:2]
        # convert the reticle position to the pixel coordinate system
        reticle_position = self._cv2_coord_to_pixel_coord(cv2_coord=reticle_position_cv2, resolution=resolution)
        return reticle_position

    def _cv2_coord_to_pixel_coord(self, *, cv2_coord: Vector2D, resolution: tuple[int, int]) -> Vector2D:
        """
        Convert a coordinate from the CV2 coordinate system to the pixel coordinate system.
        """
        return Vector2D(x=cv2_coord.x * resolution[1], y=cv2_coord.y * resolution[0])

    def draw_engagement_result(self, *, frame: Frame, engagement_result: EngagementResult):
        """
        Draw the engagement result on the frame.
        """
        text_color = (0, 0, 0)
        color = (240, 255, 240)
        match engagement_result.status:
            case EngagementStatus.WAITING_TO_START:
                color = (128, 128, 128)
            case EngagementStatus.FPS_CONDITION_NOT_SATISFIED:
                color = (0, 128, 0)
            case EngagementStatus.NOT_ENGAGING:
                color = (0, 255, 0)
            case EngagementStatus.ENGAGING:
                text_color = (0, 0, 255)
                color = (0, 0, 255)
            case EngagementStatus.EXIT_STRATEGY_UNDER_EXECUTION:
                color = (255, 0, 128)
        # draw rectangle around the frame.image
        cv2.rectangle(frame.image, (0, 0), (frame.image.shape[1], frame.image.shape[0]), color, 20)
        # draw text on the frame.image
        status_str = f'Status: {engagement_result.status.value}'
        cv2.putText(frame.image, status_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        # draw engagement counter
        engagement_counter_str = f'Engagement Counter: {len(self._engagement_history)}'
        cv2.putText(frame.image, engagement_counter_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        # draw fps
        fps_str = f'FPS: {engagement_result.fps:.2f}'
        cv2.putText(frame.image, fps_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)