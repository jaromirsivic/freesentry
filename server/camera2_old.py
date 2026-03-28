from pathlib import Path
from typing import Any
from .camerapostprocessing import NodeImage, NodeImageCroppedResized, NodeImageAI
from .common import Frame
import time
import cv2
import threading
import numpy as np


FLIP_HORIZONTAL = 1
FLIP_VERTICAL = 0
FLIP_BOTH = -1

class Camera2(threading.Thread):
    def __init__(
        self, *, index: int,
        settings: dict = None,
        image_cropped_resized_settings: dict = None,
        image_ai_settings: dict = None
    ):
        super().__init__()
        self.daemon = True        
        # OpenCV camera object
        self._camera:cv2.VideoCapture = None
        # Camera index
        self._index = index
        # Camera settings
        self._settings = settings
        # Camera name
        self._name = f'{index}: NO CAMERA DETECTED'
        # Supported resolutions
        self.supported_resolutions = []
        # Flip and rotate settings
        self._flip_horizontal = False
        self._flip_vertical = False
        self._rotate = 0
        # Is camera active
        self._active = False
        # Last frame
        self._frame: Frame = self.create_blank_frame()
        # Create post processing filters
        # NodeImage object
        self._image: NodeImage = NodeImage(parent=self, fps=None)
        # NodeImageCroppedResized object
        self._image_cropped_resized_settings = image_cropped_resized_settings
        self._image_cropped_resized: NodeImageCroppedResized = NodeImageCroppedResized(parent=self, fps=None, settings=self._image_cropped_resized_settings)
        # NodeImageAI object
        self._image_ai_settings = image_ai_settings
        self._image_ai: NodeImageAI = NodeImageAI(parent=self, fps=None, settings=self._image_ai_settings)
        # open camera
        self.open()

    def create_blank_frame(self) -> Frame:
        """
        Create a frame with width 640 and height 480 and fill it with black color.
        There is a text "NO CAMERA DETECTED" in the center of the frame.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "NO CAMERA DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Frame(valid=False, frame=frame, frame_time=0)

    # Basic properties

    @property
    def camera(self) -> cv2.VideoCapture:
        return self._camera

    @property
    def active(self) -> bool:
        return self._active

    @property
    def index(self) -> int:
        return self._index
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def fps(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "fps" in self._settings:
            return self._settings["fps"]
        result = self._camera.get(cv2.CAP_PROP_FPS)
        result = min(1000, max(1, result))
        return result

    @fps.setter
    def fps(self, value: float):
        if not self._active or self._camera is None:
            return
        value = min(1000, max(1, value))
        try:
            self._camera.set(cv2.CAP_PROP_FPS, value)
        except Exception as e:
            print(f"Error setting fps: {e}")
        if self._settings is not None and "fps" in self._settings:
            self._settings["fps"] = value
        # Reset the post processing filters fps
        self.reset_nodes_fps()

    @property
    def width(self) -> int:
        if not self._active or self._camera is None:
            return -1
        if self._settings is not None and "width" in self._settings:
            return self._settings["width"]
        return int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @width.setter
    def width(self, value: int):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, value)
        except Exception as e:
            print(f"Error setting width: {e}")
        if self._settings is not None and "width" in self._settings:
            self._settings["width"] = value

    @property
    def height(self) -> int:
        if not self._active or self._camera is None:
            return -1
        if self._settings is not None and "height" in self._settings:
            return self._settings["height"]
        return int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @height.setter
    def height(self, value: int):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
        except Exception as e:
            print(f"Error setting height: {e}")
        if self._settings is not None and "height" in self._settings:
            self._settings["height"] = value

    @property
    def bitrate(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "bitrate" in self._settings:
            return self._settings["bitrate"]
        return self._camera.get(cv2.CAP_PROP_BITRATE)

    @bitrate.setter
    def bitrate(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_BITRATE, value)
        except Exception as e:
            print(f"Error setting bitrate: {e}")
        if self._settings is not None and "bitrate" in self._settings:
            self._settings["bitrate"] = value

    @property
    def buffer_size(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "buffer_size" in self._settings:
            return self._settings["buffer_size"]
        return self._camera.get(cv2.CAP_PROP_BUFFERSIZE)

    @buffer_size.setter
    def buffer_size(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, value)
        except Exception as e:
            print(f"Error setting buffer_size: {e}")
        if self._settings is not None and "buffer_size" in self._settings:
            self._settings["buffer_size"] = value

    # Post processing filters

    @property
    def image(self) -> NodeImage:
        return self._image

    @property
    def image_cropped_resized(self) -> NodeImageCroppedResized:
        return self._image_cropped_resized

    @property
    def image_ai(self) -> NodeImageAI:
        return self._image_ai

    def reset_nodes_fps(self):
        """
        Reset the fps of the nodes to twice the fps of the camera.
        """
        fps = self.fps
        self._image.fps = fps * 2
        self._image_cropped_resized.fps = fps * 2
        self._image_ai.fps = fps * 2

    # Flip and rotate settings

    @property
    def flip_horizontal(self) -> bool:
        if self._settings is not None and "flip_horizontal" in self._settings:
            return self._settings["flip_horizontal"]
        return self._flip_horizontal
    
    @flip_horizontal.setter
    def flip_horizontal(self, value: bool):
        self._flip_horizontal = value
        if self._settings is not None and "flip_horizontal" in self._settings:
            self._settings["flip_horizontal"] = value
    
    @property
    def flip_vertical(self) -> bool:
        if self._settings is not None and "flip_vertical" in self._settings:
            return self._settings["flip_vertical"]
        return self._flip_vertical
    
    @flip_vertical.setter
    def flip_vertical(self, value: bool):
        self._flip_vertical = value
        if self._settings is not None and "flip_vertical" in self._settings:
            self._settings["flip_vertical"] = value
    
    @property
    def rotate(self) -> int:
        if self._settings is not None and "rotate" in self._settings:
            return self._settings["rotate"]
        return self._rotate
    
    @rotate.setter
    def rotate(self, value: int):
        self._rotate = value
        if self._settings is not None and "rotate" in self._settings:
            self._settings["rotate"] = value

    # Camera settings

    @property
    def brightness(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "brightness" in self._settings:
            return self._settings["brightness"]
        return self._camera.get(cv2.CAP_PROP_BRIGHTNESS)

    @brightness.setter
    def brightness(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_BRIGHTNESS, value)
        except Exception as e:
            print(f"Error setting brightness: {e}")
        if self._settings is not None and "brightness" in self._settings:
            self._settings["brightness"] = value

    @property
    def contrast(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "contrast" in self._settings:
            return self._settings["contrast"]
        return self._camera.get(cv2.CAP_PROP_CONTRAST)

    @contrast.setter
    def contrast(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_CONTRAST, value)
        except Exception as e:
            print(f"Error setting contrast: {e}")
        if self._settings is not None and "contrast" in self._settings:
            self._settings["contrast"] = value

    @property
    def hue(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "hue" in self._settings:
            return self._settings["hue"]
        return self._camera.get(cv2.CAP_PROP_HUE)

    @hue.setter
    def hue(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_HUE, value)
        except Exception as e:
            print(f"Error setting hue: {e}")
        if self._settings is not None and "hue" in self._settings:
            self._settings["hue"] = value

    @property
    def saturation(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "saturation" in self._settings:
            return self._settings["saturation"]
        return self._camera.get(cv2.CAP_PROP_SATURATION)

    @saturation.setter
    def saturation(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_SATURATION, value)
        except Exception as e:
            print(f"Error setting saturation: {e}")
        if self._settings is not None and "saturation" in self._settings:
            self._settings["saturation"] = value

    @property
    def sharpness(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "sharpness" in self._settings:
            return self._settings["sharpness"]
        return self._camera.get(cv2.CAP_PROP_SHARPNESS)

    @sharpness.setter
    def sharpness(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_SHARPNESS, value)
        except Exception as e:
            print(f"Error setting sharpness: {e}")
        if self._settings is not None and "sharpness" in self._settings:
            self._settings["sharpness"] = value

    @property
    def gamma(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "gamma" in self._settings:
            return self._settings["gamma"]
        return self._camera.get(cv2.CAP_PROP_GAMMA)

    @gamma.setter
    def gamma(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_GAMMA, value)
        except Exception as e:
            print(f"Error setting gamma: {e}")
        if self._settings is not None and "gamma" in self._settings:
            self._settings["gamma"] = value

    @property
    def white_balance_temperature(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "white_balance_temperature" in self._settings:
            return self._settings["white_balance_temperature"]
        return self._camera.get(cv2.CAP_PROP_WB_TEMPERATURE)

    @white_balance_temperature.setter
    def white_balance_temperature(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_WB_TEMPERATURE, value)
        except Exception as e:
            print(f"Error setting white_balance_temperature: {e}")
        if self._settings is not None and "white_balance_temperature" in self._settings:
            self._settings["white_balance_temperature"] = value

    @property
    def backlight(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "backlight" in self._settings:
            return self._settings["backlight"]
        return self._camera.get(cv2.CAP_PROP_BACKLIGHT)

    @backlight.setter
    def backlight(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_BACKLIGHT, value)
        except Exception as e:
            print(f"Error setting backlight: {e}")
        if self._settings is not None and "backlight" in self._settings:
            self._settings["backlight"] = value

    @property
    def gain(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "gain" in self._settings:
            return self._settings["gain"]
        return self._camera.get(cv2.CAP_PROP_GAIN)

    @gain.setter
    def gain(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_GAIN, value)
        except Exception as e:
            print(f"Error setting gain: {e}")
        if self._settings is not None and "gain" in self._settings:
            self._settings["gain"] = value

    @property
    def focus(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "focus" in self._settings:
            return self._settings["focus"]
        return self._camera.get(cv2.CAP_PROP_FOCUS)

    @focus.setter
    def focus(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_FOCUS, value)
        except Exception as e:
            print(f"Error setting focus: {e}")
        if self._settings is not None and "focus" in self._settings:
            self._settings["focus"] = value

    @property
    def exposure(self) -> float:
        if not self._active or self._camera is None:
            return -1.0
        if self._settings is not None and "exposure" in self._settings:
            return self._settings["exposure"]
        return self._camera.get(cv2.CAP_PROP_EXPOSURE)

    @exposure.setter
    def exposure(self, value: float):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_EXPOSURE, value)
        except Exception as e:
            print(f"Error setting exposure: {e}")
        if self._settings is not None and "exposure" in self._settings:
            self._settings["exposure"] = value
    
    @property
    def auto_white_balance_temperature(self) -> bool:
        if not self._active or self._camera is None:
            return False
        if self._settings is not None and "auto_white_balance_temperature" in self._settings:
            return self._settings["auto_white_balance_temperature"]
        return bool(self._camera.get(cv2.CAP_PROP_AUTO_WB))

    @auto_white_balance_temperature.setter
    def auto_white_balance_temperature(self, value: bool):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_AUTO_WB, 1.0 if value else 0.0)
        except Exception as e:
            print(f"Error setting auto_white_balance_temperature: {e}")
        if self._settings is not None and "auto_white_balance_temperature" in self._settings:
            self._settings["auto_white_balance_temperature"] = value

    @property
    def auto_focus(self) -> bool:
        if not self._active or self._camera is None:
            return False
        if self._settings is not None and "auto_focus" in self._settings:
            return self._settings["auto_focus"]
        return bool(self._camera.get(cv2.CAP_PROP_AUTOFOCUS))

    @auto_focus.setter
    def auto_focus(self, value: bool):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if value else 0.0)
        except Exception as e:
            print(f"Error setting auto_focus: {e}")
        if self._settings is not None and "auto_focus" in self._settings:
            self._settings["auto_focus"] = value

    @property
    def auto_exposure(self) -> bool:
        if not self._active or self._camera is None:
            return False
        if self._settings is not None and "auto_exposure" in self._settings:
            return self._settings["auto_exposure"]
        return bool(self._camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    @auto_exposure.setter
    def auto_exposure(self, value: bool):
        if not self._active or self._camera is None:
            return
        try:
            self._camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if value else 0.0)
        except Exception as e:
            print(f"Error setting auto_exposure: {e}")
        if self._settings is not None and "auto_exposure" in self._settings:
            self._settings["auto_exposure"] = value

    def open(self):
        self._camera = cv2.VideoCapture(self._index)
        if not self._camera.isOpened():
            self._active = False
            return False
        # Set active
        self._active = True
        # Set name
        self._name = f'{self._index}: GUID_{self._camera.get(cv2.CAP_PROP_GUID)}'

        # Check supported resolutions
        common_resolutions = [
            (320, 240),
            (640, 480),
            (800, 600),
            (960, 720),
            (1024, 768),
            (1280, 960),
            (1280, 720),
            (1600, 1200),
            (1920, 1080),
            (2560, 1440),
            (3200, 1800),
            (3840, 2160),
            (4096, 2160),
            (5120, 2880),
            (6016, 3384),
            (7680, 4320),
            (8192, 4608)
        ]
        supported_resolutions = []
        startup_width = 640
        startup_height = 480
        # Check if supported resolutions are available
        for width, height in common_resolutions:
            self.width = width
            self.height = height
            if self.width == width and self.height == height:
                supported_resolutions.append({"width": width, "height": height, "label": f"{width} x {height}"})
                # check if there is a supported resolution that is 1920x1080 and set it as the startup resolution
                if width ==1920 and height == 1080:
                    startup_width = width
                    startup_height = height
        # set the startup resolution
        self.width = startup_width
        self.height = startup_height
        self.supported_resolutions = supported_resolutions
        # Reset the fps of the nodes (post processing filters)
        self.reset_nodes_fps()
        return True

    def close(self):
        if not self._active:
            return
        self._active = False
        self._camera.release()
        self._camera = None

    def __del__(self):
        self.close()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "supported_resolutions": self.supported_resolutions,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bitrate": self.bitrate,
            "buffer_size": self.buffer_size,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "rotate": self.rotate,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "hue": self.hue,
            "saturation": self.saturation,
            "sharpness": self.sharpness,
            "gamma": self.gamma,
            "white_balance_temperature": self.white_balance_temperature,
            "backlight": self.backlight,
            "gain": self.gain,
            "focus": self.focus,
            "exposure": self.exposure,
            "auto_white_balance_temperature": self.auto_white_balance_temperature,
            "auto_focus": self.auto_focus,
            "auto_exposure": self.auto_exposure
        }

    def reload_settings(self):
        if not self._active or self._camera is None or self._settings is None:
            return
        self.width = self._settings["width"] if "width" in self._settings else self.width
        self.height = self._settings["height"] if "height" in self._settings else self.height
        self.fps = self._settings["fps"] if "fps" in self._settings else self.fps
        self.flip_horizontal = self._settings["flip_horizontal"] if "flip_horizontal" in self._settings else self.flip_horizontal
        self.flip_vertical = self._settings["flip_vertical"] if "flip_vertical" in self._settings else self.flip_vertical
        self.rotate = self._settings["rotate"] if "rotate" in self._settings else self.rotate
        self.brightness = self._settings["brightness"] if "brightness" in self._settings else self.brightness
        self.contrast = self._settings["contrast"] if "contrast" in self._settings else self.contrast
        self.hue = self._settings["hue"] if "hue" in self._settings else self.hue
        self.saturation = self._settings["saturation"] if "saturation" in self._settings else self.saturation
        self.sharpness = self._settings["sharpness"] if "sharpness" in self._settings else self.sharpness
        self.gamma = self._settings["gamma"] if "gamma" in self._settings else self.gamma
        self.white_balance_temperature = self._settings["white_balance_temperature"] if "white_balance_temperature" in self._settings else self.white_balance_temperature
        # Support both "backlight" and "backlight_compensation" for backward compatibility
        self.backlight = self._settings.get("backlight", self._settings.get("backlight_compensation", 0)) if "backlight" in self._settings else self.backlight
        self.gain = self._settings["gain"] if "gain" in self._settings else self.gain
        self.focus = self._settings["focus"] if "focus" in self._settings else self.focus
        self.exposure = self._settings["exposure"] if "exposure" in self._settings else self.exposure
        self.auto_white_balance_temperature = self._settings["auto_white_balance_temperature"] if "auto_white_balance_temperature" in self._settings else self.auto_white_balance_temperature
        self.auto_focus = self._settings["auto_focus"] if "auto_focus" in self._settings else self.auto_focus
        self.auto_exposure = self._settings["auto_exposure"] if "auto_exposure" in self._settings else self.auto_exposure

    def reset_settings(self):
        self._settings = None
        width = self.width
        self.width = width
        height = self.height
        self.height = height
        fps = self.fps
        self.fps = fps
        flip_horizontal = self.flip_horizontal
        self.flip_horizontal = flip_horizontal
        flip_vertical = self.flip_vertical
        self.flip_vertical = flip_vertical
        rotate = self.rotate
        self.rotate = rotate
        brightness = self.brightness
        self.brightness = brightness
        contrast = self.contrast
        self.contrast = contrast
        hue = self.hue
        self.hue = hue
        saturation = self.saturation
        self.saturation = saturation
        sharpness = self.sharpness
        self.sharpness = sharpness
        gamma = self.gamma
        self.gamma = gamma
        white_balance_temperature = self.white_balance_temperature
        self.white_balance_temperature = white_balance_temperature
        backlight = self.backlight
        self.backlight = backlight
        gain = self.gain
        self.gain = gain
        focus = self.focus
        self.focus = focus
        exposure = self.exposure
        self.exposure = exposure
        auto_white_balance_temperature = self.auto_white_balance_temperature
        self.auto_white_balance_temperature = auto_white_balance_temperature
        auto_focus = self.auto_focus
        self.auto_focus = auto_focus
        auto_exposure = self.auto_exposure
        self.auto_exposure = auto_exposure

    # Settings properties

    @property
    def settings(self) -> dict:
        return self._settings

    @settings.setter
    def settings(self, value: dict):
        self._settings = value
        self.reload_settings()
    
    @property
    def image_cropped_resized_settings(self) -> dict:
        return self._image_cropped_resized_settings

    @image_cropped_resized_settings.setter
    def image_cropped_resized_settings(self, value: dict):
        self._image_cropped_resized_settings = value
        self._image_cropped_resized.reload_settings()
    
    @property
    def image_ai_settings(self) -> dict:
        return self._image_ai_settings

    @image_ai_settings.setter
    def image_ai_settings(self, value: dict):
        self._image_ai_settings = value
        self._image_ai.reload_settings()

    # Frame properties

    @property
    def frame(self) -> Frame:
        # if thread is started, return the frame
        if self.is_alive():
            return self._frame
        else:
            try:
                self.start()
            except Exception as e:
                print(f"Error during starting camera thread: {e}")
            return self.create_blank_frame()

        # ret_value, ret_frame = self._get_frame()
        # if ret_value:
        #     self._frame_time = time.time()
        #     return ret_frame
        # return None

    def _get_frame(self):
        # if camera is not active or camera is not opened, return False, None
        if not self._active or self._camera is None:
            return
        # read frame from camera
        valid, frame = self._camera.read()
        if not valid:
            return
        # Flip camera input horizontally and vertically
        if self._flip_horizontal and self._flip_vertical:
            frame = cv2.flip(frame, FLIP_BOTH)
        elif self._flip_horizontal:
            frame = cv2.flip(frame, FLIP_HORIZONTAL)
        elif self._flip_vertical:
            frame = cv2.flip(frame, FLIP_VERTICAL)
        # Rotate camera input
        if self._rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self._rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self._rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # set frame
        self._frame.valid = valid
        self._frame.image = frame
        self._frame.frame_time = time.time()

    def run(self):
        self.open()
        self._frame.frame_time = time.time()
        while self._active and time.time() - self._frame.frame_time < 5:
            self._get_frame()
            time.sleep(0.001)
        self.close()

    def stop(self):
        self._active = False
        self.join()