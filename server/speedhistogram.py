from copy import deepcopy
from .common import epsilon

class SpeedHistogram:
    def __init__(self, *, speed_histogram: list[dict], resolution: int = 1000):
        if resolution < 2:
            raise ValueError("Resolution must be at least 2")
        self._resolution = resolution
        self._speed_histogram = deepcopy(speed_histogram)
        self._forward_speed_histogram = []
        self._reverse_speed_histogram = []
        self.max_forward_speed_seconds = 0
        self.max_reverse_speed_seconds = 0
        self._init_speed_histograms()

    @property
    def resolution(self):
        return self._resolution

    @property
    def forward_speed_histogram(self):
        return self._forward_speed_histogram

    @property
    def reverse_speed_histogram(self):
        return self._reverse_speed_histogram
    
    def _init_speed_histograms(self):
        """
        Initialize speed histograms in following way:
        It transforms speed histogram to two lists of forward and reverse using linear interpolation.
        It uses given resolution as number of items in forward and reverse lists. Each item in the list
        is a dictionary with pwmMultiplier and distance traveled in one second. Distance is relative
        to the length of the actuator.
        """
        # sort speed histogram by pwmMultiplier
        self._speed_histogram.sort(key=lambda x: x["pwmMultiplier"])
        # check that there are at least two items in speed histogram
        if len(self._speed_histogram) < 2:
            raise ValueError("Speed histogram must contain at least two items")
        # check that pwmMultiplier is in range [0, 1]
        for item in self._speed_histogram:
            if item["pwmMultiplier"] < 0 or item["pwmMultiplier"] > 1:
                raise ValueError("pwmMultiplier must be in range [0, 1]")
        # check that first item has pwmMultiplier 0 and forwardSeconds 0 and reverseSeconds 0
        if self._speed_histogram[0]["pwmMultiplier"] != 0 or self._speed_histogram[0]["forwardSeconds"] != 0 or self._speed_histogram[0]["reverseSeconds"] != 0:
            raise ValueError("First item in speed histogram must have pwmMultiplier 0 and forwardSeconds 0 and reverseSeconds 0")
        # check that last item has pwmMultiplier 1
        if self._speed_histogram[-1]["pwmMultiplier"] != 1:
            raise ValueError("Last item in speed histogram must have pwmMultiplier 1")
        # check that each item has forwardSeconds less than or equal previous item
        for i in range(1, len(self._speed_histogram)):
            if self._speed_histogram[i-1]["forwardSeconds"] > 0 and self._speed_histogram[i]["forwardSeconds"] >= self._speed_histogram[i - 1]["forwardSeconds"]:
                raise ValueError("forwardSeconds must be less than or equal previous item")
        # check that each item has reverseSeconds less than or equal previous item
        for i in range(1, len(self._speed_histogram)):
            if self._speed_histogram[i-1]["reverseSeconds"] > 0 and self._speed_histogram[i]["reverseSeconds"] >= self._speed_histogram[i - 1]["reverseSeconds"]:
                raise ValueError("reverseSeconds must be less than or equal previous item")
        # normalize speed histogram
        self._normalize_speed()
        # compute forward and reverse pwm histograms
        self._forward_speed_histogram.append(0)
        self._reverse_speed_histogram.append(0)
        for i in range(1, self._resolution):
            speed = i / self._resolution
            forward_pwm = self._compute_pwm(speed=speed, which_type_of_speed="forward")
            reverse_pwm = self._compute_pwm(speed=speed, which_type_of_speed="reverse")
            self._forward_speed_histogram.append(forward_pwm)
            self._reverse_speed_histogram.append(reverse_pwm)
        self._forward_speed_histogram.append(1)
        self._reverse_speed_histogram.append(1)

    def _normalize_speed(self):
        # get max forward and reverse seconds
        self.max_forward_speed_seconds = self._speed_histogram[-1]["forwardSeconds"]
        self.max_reverse_speed_seconds = self._speed_histogram[-1]["reverseSeconds"]
        # normalize speed histogram adds new parameter forwardSpeed and reverseSpeed
        for i in range(0, len(self._speed_histogram)):
            if self._speed_histogram[i]["forwardSeconds"] == 0:
                self._speed_histogram[i]["forwardSpeed"] = 0
            else:
                self._speed_histogram[i]["forwardSpeed"] = self.max_forward_speed_seconds / self._speed_histogram[i]["forwardSeconds"]
            if self._speed_histogram[i]["reverseSeconds"] == 0:
                self._speed_histogram[i]["reverseSpeed"] = 0
            else:
                self._speed_histogram[i]["reverseSpeed"] = self.max_reverse_speed_seconds / self._speed_histogram[i]["reverseSeconds"]
    
    def get_speed_index(self, *, speed: float, which_type_of_speed: str="forward"):
        # find the index of the item with pwmMultiplier closest to the given speed
        index = 0
        speed_key = which_type_of_speed + "Speed"
        for i in range(0, len(self._speed_histogram)):
            if self._speed_histogram[i][speed_key] > speed:
                index = i
                break
        return index
    
    def _compute_pwm(self, *, speed: float, which_type_of_speed: str="forward"):
        # find the index of the item with pwmMultiplier closest to the given speed
        index = self.get_speed_index(speed=speed, which_type_of_speed=which_type_of_speed)
        speed_key = which_type_of_speed + "Speed"
        # if the index is 0 then return the first item
        if index <= 0:
            return self._speed_histogram[0]["pwmMultiplier"]
        # if the index is the last item then return the last item
        if index == len(self._speed_histogram):
            return self._speed_histogram[-1]["pwmMultiplier"]
        # if the value with the index is close to the given speed then return it
        if abs(self._speed_histogram[index][speed_key] - speed) < epsilon:
            return self._speed_histogram[index]["pwmMultiplier"]
        # interpolate the value
        lower_speed = self._speed_histogram[index - 1][speed_key]
        higher_speed = self._speed_histogram[index][speed_key]
        speed_delta = higher_speed - lower_speed
        if speed_delta == 0:
            return self._speed_histogram[index]["pwmMultiplier"]
        # relative position of speed between lower and higher speed points
        speed_position = (speed - lower_speed) / speed_delta
        # interpolate the value
        lower_pwm = self._speed_histogram[index - 1]["pwmMultiplier"]
        higher_pwm = self._speed_histogram[index]["pwmMultiplier"]
        pwm_delta = higher_pwm - lower_pwm
        result = lower_pwm + pwm_delta * speed_position
        return result

    def pwm_of_forward_speed(self, *, speed: float):
        index = round(speed * self._resolution)
        return self._forward_speed_histogram[index]
    
    def pwm_of_reverse_speed(self, *, speed: float):
        speed = abs(speed)
        index = round(speed * self._resolution)
        return self._reverse_speed_histogram[index]      
