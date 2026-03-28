from time import time
from .motor import Motor
from .common import epsilon, motor_frame
from .speedhistogram import SpeedHistogram
from .pin import Pin

class MotorLinearActuator(Motor):
    def __init__(self, *, forward_pin: Pin, forward_enable_pin: Pin,
                 reverse_pin: Pin, reverse_enable_pin: Pin,
                 pwm_frequency: int,
                 speed_histogram: SpeedHistogram, inertia: float):
        # forward pin and forward enable pin
        self._forward_pin = forward_pin
        self._forward_pin.pwm_frequency = pwm_frequency
        self._forward_enable_pin = forward_enable_pin
        self._forward_enable_pin.pwm_frequency = 0
        self._forward_enable_pin.value = 1
        # reverse pin and reverse enable pin
        self._reverse_pin = reverse_pin
        self._reverse_pin.pwm_frequency = pwm_frequency
        self._reverse_enable_pin = reverse_enable_pin
        self._reverse_enable_pin.pwm_frequency = 0
        self._reverse_enable_pin.value = 1
        # speed histogram describes relation between motor speed and pwm multiplier
        self._speed_histogram = speed_histogram
        # inertia is a time in seconds how long it takes to go from speed 0% to 100%
        self._inertia = inertia
        # compute speed step
        self._speed_step = self._compute_speed_step()
        #self._forward_speed_step_index_increment_in_histogram = self._compute_forward_speed_step_index_increment_in_histogram()
        #self._reverse_speed_step_index_increment_in_histogram = self._compute_reverse_speed_step_index_increment_in_histogram()
        # current position
        self._position = 0
        # target position
        self._target_speed = 0
        #self._target_forward_speed_index_in_histogram = 0
        #self._target_reverse_speed_index_in_histogram = 0
        # current speed
        self._current_speed = 0
        #self._current_forward_speed_index_in_histogram = 0
        #self._current_reverse_speed_index_in_histogram = 0  
        # last time
        self._last_time = time()

    def _compute_speed_step_index_in_histogram(self):
        resolution = self._speed_histogram.resolution
        frames_to_reach_max_speed = self._inertia / motor_frame
        # if inertia is very small then set speed step to a very large number
        if frames_to_reach_max_speed < epsilon:
            result = 1000000000
        # if inertia is not very small then compute speed step
        else:
            result = resolution / frames_to_reach_max_speed
        return result

    def _compute_speed_step(self):
        return (1 / self._speed_histogram.resolution) * self._compute_speed_step_index_in_histogram()    

    @property
    def position(self):
        return self._position

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        self._inertia = value
        self._speed_step = self._compute_speed_step()

    @property
    def speed_histogram(self):
        return self._speed_histogram

    @property
    def speed_step(self):
        return self._speed_step        

    @property
    def target_speed(self):
        return self._target_speed

    @property
    def current_speed(self):
        return self._current_speed

    def move(self, *, speed: float):
        """
        Set the target speed of the motor
        """
        speed = round(speed * self._speed_histogram.resolution) / self._speed_histogram.resolution
        self._target_speed = min(max(speed, -1), 1)
    
    def go(self) -> bool:
        """
        This function is called periodically as quickly as possible.
        It is used to set pwm values of forward or reverse pin on Raspberry Pi
        to move the motor to the target position.
        """
        # if motor_frame time did not elapsed then do nothing
        now = time()
        delta = now - self._last_time
        if delta < motor_frame:
            return False
        # if speed is not changed then do nothing
        if abs(self._target_speed - self._current_speed) < epsilon:
            self._current_speed = self._target_speed
            # compute position
            now = time()
            delta = now - self._last_time
            if self._current_speed < 0:
                self._position += (1 / self.speed_histogram.max_reverse_speed_seconds) * self._current_speed * delta
            else:
                self._position += (1 / self.speed_histogram.max_forward_speed_seconds) * self._current_speed * delta
            self._last_time = now
            return True
        # compute direction
        direction = 1 if self._target_speed > self._current_speed else -1
        # save current speed
        old_current_speed = self._current_speed
        # compute new current speed
        new_current_speed = self._current_speed + self._speed_step * direction
        if abs(self._target_speed - self._current_speed) < (self._speed_step + epsilon):
            new_current_speed = self._target_speed
        # if speed crosses through zero then stop both pwms
        if (old_current_speed <= 0 and new_current_speed >= 0) or \
           (old_current_speed >= 0 and new_current_speed <= 0):
            self._forward_pin.value = 0
            self._reverse_pin.value = 0
        # save new current speed
        self._current_speed = new_current_speed

        # update position and pin pwm
        if self._current_speed < 0:
            # compute reverse pwm
            reverse_pwm = self._speed_histogram.pwm_of_reverse_speed(speed=self._current_speed)
            # compute position
            now = time()
            delta = now - self._last_time
            self._position += (1 / self.speed_histogram.max_reverse_speed_seconds) * old_current_speed * delta
            self._last_time = now
            # set reverse pin to new pwm value
            self._reverse_pin.value = reverse_pwm
        elif self._current_speed > 0:
            # compute forward pwm
            forward_pwm = self._speed_histogram.pwm_of_forward_speed(speed=self._current_speed)
            # compute position
            now = time()
            delta = now - self._last_time
            self._position += (1 / self.speed_histogram.max_forward_speed_seconds) * old_current_speed * delta
            self._last_time = now
            # set forward pin to new pwm value
            self._forward_pin.value = forward_pwm
        return False        
        