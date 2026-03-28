from gpiozero import PWMOutputDevice
from time import sleep
import datetime
import math
from enum import Enum
import threading
from .common import epsilon, Device
#from periphery import PWM
import subprocess

class PinType(Enum):
    OUTPUT = 1
    INPUT = 2

class PWMWrapper:
    def __init__(self, *, index: int, gpio_index: int, initial_value: float = 0, pwm_frequency: int = 4000, pin_factory=None, device: Device = Device.RASPBERRY_PI_5):
        # initialize variables
        self._index = index
        self._gpio_index = gpio_index
        self._value = initial_value
        self._pwm_frequency = pwm_frequency
        self._pwm = None
        self._device = device
        self._pin_factory = pin_factory
        periphery_available = False
        # check if periphery is available
        try:
            from periphery import PWM
            periphery_available = True
        except Exception as e:
            print(f"Failed to import periphery: {e} for pin {self._gpio_index}")
            periphery_available = False
        # if periphery is available, use it to create the pwm object for hardware pwm
        if periphery_available and self._device == Device.RASPBERRY_PI_5 and self._gpio_index in [12, 13, 18, 19] and self._force_pin_muxing():
            gpi = self._gpio_index
            PWM_CHIP = 0
            CHANNEL = 0 if gpi == 12 else 1 if gpi == 13 else 2 if gpi == 18 else 3
            self._pwm = PWM(PWM_CHIP, CHANNEL)
            self._pwm.period_ns = 1_000_000_000 // pwm_frequency
            self._pwm.duty_cycle_ns = int(self._pwm.period_ns * self._value)
            self._pwm.enable()
        # if periphery is not available, use gpiozero to create the pwm object for software pwm
        else:
            self._pwm = PWMOutputDevice(f"J8:{self._index}", 
                                        initial_value=self._value,
                                        frequency=self._pwm_frequency,
                                        pin_factory=self._pin_factory)

    def _force_pin_muxing(self) -> bool:
        """
        Uses the 'pinctrl' system command to manually force the Pi 5 pins 
        to connect to the PWM hardware block.
        
        Mapping for RP1 (Pi 5):
        GPIO 12 -> Alt0 (PWM0 Chan 0)
        GPIO 13 -> Alt0 (PWM0 Chan 1)
        GPIO 18 -> Alt3 (PWM0 Chan 2)
        GPIO 19 -> Alt3 (PWM0 Chan 3)
        """
        print("Configuring Pin Muxing via pinctrl...")
        try:
            # We run these shell commands to force the mode
            # 'a0' = Alt0, 'a3' = Alt3
            match self._gpio_index:
                case 12:
                    subprocess.run(["pinctrl", "set", "12", "a0"], check=True)
                case 13:
                    subprocess.run(["pinctrl", "set", "13", "a0"], check=True)
                case 18:
                    subprocess.run(["pinctrl", "set", "18", "a3"], check=True)
                case 19:
                    subprocess.run(["pinctrl", "set", "19", "a3"], check=True)
            print(f"Pin muxing success: Pin {self._gpio_index} mapped to hardware PWM.")
            return True
        except FileNotFoundError:
            print("ERROR: 'pinctrl' command not found.")
            print("Please install it: sudo apt install raspi-utils")
            return False
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to set pin mode. {e}")
            return False

    @property
    def value(self):
        """
        Get the value of the pin
        """
        return self._value
    
    @value.setter
    def value(self, value):
        """
        Set the value of the pin
        """
        self._value = value
        if self._device == Device.RASPBERRY_PI_5 and self._gpio_index in [12, 13, 18, 19]:
            self._pwm.duty_cycle_ns = int(self._pwm.period_ns * self._value)
        else:
            self._pwm.value = value

    @property
    def frequency(self):
        """
        Get the pwm frequency of the pin
        """
        return self._pwm_frequency
    
    @frequency.setter
    def frequency(self, pwm_frequency):
        """
        Set the pwm frequency of the pin
        """
        self._pwm_frequency = pwm_frequency
        if self._device == Device.RASPBERRY_PI_5 and self._gpio_index in [12, 13, 18, 19]:
            self._pwm.period_ns = 1_000_000_000 // pwm_frequency
            self._pwm.duty_cycle_ns = int(self._pwm.period_ns * self._value)
        else:
            self._pwm.frequency = pwm_frequency

    def on(self):
        """
        Turn the pin on
        """
        self.value = 1

    def off(self):
        """
        Turn the pin off
        """
        self.value = 0

    def close(self):
        """
        Close the pin
        """
        if self._device == Device.RASPBERRY_PI_5 and self._gpio_index in [12, 13, 18, 19]:
            self.value = 0
            self._pwm.disable()
            self._pwm.close()
        else:
            self._pwm.value = 0
            self._pwm.off()
            self._pwm.close()

    def __del__(self):
        self.close()


class Pin:
    def __init__(self, *, index: int, gpio_index: int, name:str,
                 pin_factory=None, pwm_frequency: int = 4000, device: Device = Device.RASPBERRY_PI_5):
        """
        Initialize the pin
        """
        self._index = index
        self._gpio_index = gpio_index
        self._name = name
        self._lock = threading.RLock()
        self._pin_type = None
        self._pin_factory = pin_factory
        self._device = device
        self._isDummyPin = not(self._name.startswith("GPIO"))
        self._pwm_frequency = pwm_frequency
        self._value = 0
        self._pwm = None
        self.reset()

    def reset(self):
        """
        Reset the pin
        """
        with self._lock:
            self._release_pwm()
            self._isDummyPin = not(self._name.startswith("GPIO"))
            self._pwm_frequency = self._pwm_frequency
            self._value = 0
            self._pwm = None
            self._pin_type = None
            # pwm instance is created during setup of pin_type
            self.pin_type = PinType.OUTPUT

    def __del__(self):
        self._release_pwm()

    @property
    def index(self):
        """
        Get the index of the pin
        """
        with self._lock:
            return self._index

    @property
    def gpio_index(self) -> int:
        """
        Get the gpio index of the pin
        """
        with self._lock:
            return self._gpio_index

    @property
    def name(self):
        """
        Get the name of the pin
        """
        with self._lock:
            return self._name

    @property
    def pin_type(self):
        """
        Get the pin type of the pin
        """
        with self._lock:
            return self._pin_type

    @pin_type.setter
    def pin_type(self, pin_type):
        """
        Set the pin type of the pin
        """
        with self._lock:
            if self._pin_type == pin_type or self._isDummyPin:
                return
            if pin_type == PinType.OUTPUT:
                self._pin_type = pin_type      
                self._pwm = PWMWrapper(index=self._index,
                                       gpio_index=self._gpio_index,
                                       initial_value=0,
                                       pwm_frequency=self._pwm_frequency,
                                       pin_factory=self._pin_factory,
                                       device=self._device)
                # self._pwm = PWMOutputDevice(f"J8:{self._index}",
                #                             initial_value=0,
                #                             frequency=self._pwm_frequency,
                #                             pin_factory=self._pin_factory)
                self.value = 0
            else:
                # self._pin_type = pin_type
                # self._pwm.value = 0
                # self._pwm.off()
                # # PWM must not be closed because it may return pin to the floating state
                # self._pwm.close()
                # self._pwm = None
                # TODO implement
                return

    @property
    def pwm_frequency(self):
        """
        Get the pwm frequency of the pin
        """
        with self._lock:
            return self._pwm_frequency
    
    @pwm_frequency.setter
    def pwm_frequency(self, pwm_frequency):
        """
        Set the pwm frequency of the pin
        """
        if self._pwm_frequency == pwm_frequency:
            return
        with self._lock:
            self._pwm_frequency = pwm_frequency
            if self._pwm is not None:
                self._pwm.frequency = pwm_frequency

    @property
    def value(self):
        """
        Get the value of the pin
        """
        with self._lock:
            return self._value
    
    @value.setter
    def value(self, value):
        """
        Set the value of the pin
        """
        with self._lock:
            # Dummy GPIO pin can be set to any value
            if self._index == 0:
                return
            # Check if the pin type is output
            if self._pin_type != PinType.OUTPUT or self._isDummyPin:
                raise Exception("Pin type is not output. You cannot set the value of an input pin.")
            if self._pwm_frequency < epsilon:
                value = round(value, 0)
            # Set the value of the pin
            if value < epsilon:
                value = 0
                self._pwm.off()
            elif value > 1 - epsilon:
                value = 1
                self._pwm.on()
            else:
                self._pwm.value = value
            self._value = value
            # print(f"Pin {self._name} value set to {value}")

    def _release_pwm(self):
        """
        Release the pwm object
        """
        with self._lock:
            if self._pin_type == PinType.OUTPUT and not self._isDummyPin and self._pwm is not None:
                self._pwm.value = 0
                self._pwm.off()
                # PWM must not be closed because it may return pin to the floating state
                self._pwm.close()