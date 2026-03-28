from .pin import Pin
from .settingscontroller import get_settings_sync
import threading


# the singleton instance of J8
class J8(list):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(J8, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_lock'):
            self._lock = threading.RLock()
        
        with self._lock:
            # initialize general variables
            self._host = None
            self._port = None
            self._pin_factory = None
            # initialize the pins
            self._pins = []
            # parameters containing information regarding initialization progress
            self._initialized = False
            self._error_message = ""
            # parameter indicating if the connection failed
            # if the connection failed the hard reset is required
            self._hard_reset_required = False
            # initialize pins
            self.reset_finished = False
            self.reset()

    def _get_pin_factory(self, *, host:str | None = None, port:int | None = None):
        """
        Try to get the pin factory.
        """
        with self._lock:
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory
                return PiGPIOFactory(host=host, port=port)
            except Exception as e:
                self._error_message += str(e)
                return None

    def __del__(self):
        self.release()

    def release(self):
        """
        Release the J8.
        """
        with self._lock:
            self._initialized = False
            self._error_message = ""
            self._pins = []
            self._pin_factory = None

    
    def hard_reset(self):
        """
        Hard reset the J8.
        """
        with self._lock:
            self._hard_reset_required = False
            self.reset()

    def reset(self):
        """
        Reset the J8.
        """
        with self._lock:
            # if self._hard_reset_required:
            #     return
            # device can be reset only once
            if self._initialized:
                return
            # try to initialize pins
            try:
                self.release()
                # try to get the controller setup from settings.json
                settings = get_settings_sync()
                controller_setup = settings.get('general', {}).get('controllerSetup', {})
                controller = controller_setup.get('controller')
                self._host = controller_setup.get('remoteHost')
                self._port = controller_setup.get('remotePort')
                # try to get the pin factory using host and port from the function parameters
                # if controller.lower() != "localhost" and self._host is not None and self._port is not None:
                #     self._pin_factory = self._get_pin_factory(host=self._host, port=self._port)
                #     if self._pin_factory is None:
                #         self._error_message = f'Failed to connect to the pigpio daemon running at {self._host}:{self._port}. '
                #         self._error_message += 'Please check if the daemon is running. '
                #         self._error_message += 'Check firewall rules which might be blocking the connection. '
                #         self._error_message += 'Try to ping the host using "ping -c 1 {host}". '
                #         self._error_message += 'Try to run the daemon with sudo. '
                #         self._error_message += 'Try to restart the daemon with sudo pigpiod -x.'
                #         self._hard_reset_required = True
                #         # self._pins = [Pin(index=0, name=f"Dummy GPIO {index}", pin_factory=None) for index in range(41)]
                #         # return
                    
                # # try to get the pin factory using host and port from the settings.json
                # elif controller == "remote":
                #     self._pin_factory = self._get_pin_factory(host=self._host, port=self._port)
                #     if self._pin_factory is None:
                #         self._error_message = f'Failed to connect to the remote pigpio daemon running at {self._host}:{self._port}. '
                #         self._error_message += 'Please check if the daemon is running on the remote host. '
                #         self._error_message += 'Check firewall rules which might be blocking the connection. '
                #         self._error_message += f'Try to ping the remote host using "ping -c 1 {self._host}". '
                #         self._error_message += 'Try to run the daemon with sudo on the remote host. '
                #         self._error_message += 'Try to restart the daemon with sudo pigpiod -x on the remote host.'
                #         self._hard_reset_required = True
                #         # self._pins = [Pin(index=0, name=f"Dummy GPIO {index}", pin_factory=None) for index in range(41)]
                #         # return
                # # there is no pin factory
                # else:
                #     self._pin_factory = None
                try:
                    from gpiozero.pins.lgpio import LGPIOFactory
                    self._pin_factory = LGPIOFactory()
                except Exception as e:
                    self._error_message = f"Failed to initialize J8 pins using lgpio: {e}. Reverting to pin_factory=None."
                    print(self._error_message)
                    self._pin_factory = None
                # setup the pins                
                self._pins.append(Pin(index=0, gpio_index=None, name="Dummy GPIO 0", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=1, gpio_index=None, name="3v3 Power", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=2, gpio_index=None, name="5v Power", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=3, gpio_index=2, name="GPIO 2 (I2C SDA)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=4, gpio_index=None, name="5v Power", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=5, gpio_index=3, name="GPIO 3 (I2C SCL)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=6, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=7, gpio_index=4, name="GPIO 4", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=8, gpio_index=14, name="GPIO 14 (TXD)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=9, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=10, gpio_index=15, name="GPIO 15 (RXD)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=11, gpio_index=17, name="GPIO 17", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=12, gpio_index=18, name="GPIO 18 (PWM0)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=13, gpio_index=27, name="GPIO 27", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=14, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=15, gpio_index=22, name="GPIO 22", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=16, gpio_index=23, name="GPIO 23", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=17, gpio_index=None, name="3v3 Power", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=18, gpio_index=24, name="GPIO 24", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=19, gpio_index=10, name="GPIO 10 (SPI0 MOSI) - Reserved for SPI", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=20, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=21, gpio_index=9, name="GPIO 9 (SPI0 MISO) - Reserved for SPI", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=22, gpio_index=25, name="GPIO 25", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=23, gpio_index=11, name="GPIO 11 (SPI0 SCLK)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=24, gpio_index=8, name="GPIO 8 (SPI0 CE0) - Reserved for SPI", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=25, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=26, gpio_index=7, name="GPIO 7 (SPI0 CE1) - Reserved for SPI", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=27, gpio_index=0, name="Disabled GPIO 0 (I2C SDA)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=28, gpio_index=1, name="Disabled GPIO 1 (I2C SCL)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=29, gpio_index=5, name="GPIO 5", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=30, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=31, gpio_index=6, name="GPIO 6", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=32, gpio_index=12, name="GPIO 12 (PWM0)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=33, gpio_index=13, name="GPIO 13 (PWM1)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=34, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=35, gpio_index=19, name="GPIO 19 (PWM1)", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=36, gpio_index=16, name="GPIO 16", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=37, gpio_index=26, name="GPIO 26", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=38, gpio_index=20, name="GPIO 20", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=39, gpio_index=None, name="GND", pin_factory=self._pin_factory))
                self._pins.append(Pin(index=40, gpio_index=21, name="GPIO 21", pin_factory=self._pin_factory))
                self._initialized = True
            except Exception as e:
                self._error_message = f"Failed to initialize J8 pins: {e}"
                self._hard_reset_required = True
                self._pins = [Pin(index=0, gpio_index=None, name=f"Dummy GPIO {index}", pin_factory=None) for index in range(41)]

    @property
    def initialized(self):
        with self._lock:
            return self._initialized

    @property
    def error_message(self):
        with self._lock:
            return self._error_message

    def __del__(self):
        with self._lock:
            self.release()

    def __getitem__(self, key):
        with self._lock:
            return self._pins[key]

    def __setitem__(self, key, value):
        raise Exception("J8 pins are read-only")