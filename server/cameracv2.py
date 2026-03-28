import cv2
from .yolomodels import YOLOModels
# from .camerapostprocessing import NodeImage, NodeImageCroppedResized, NodeImageAI
from copy import deepcopy
from .camera import Camera, CameraType


class CameraCV2(Camera):
    """
    Camera class that captures frames in a background thread.
    
    The device is closed by default. When frame property is accessed,
    the thread starts and captures frames. If frame is not accessed
    for more than 5 seconds, the thread stops and closes the device.
    """
    
    def __init__(self, *, index: int,
                 camera_index: int,
                 camera_code: str = None,
                 camera_name: str = None,
                 settings: dict = None,
                 image_cropped_resized_settings: dict = None,
                 image_ai_settings: dict = None,
                 master_controller: "MasterController" = None):  # pyright: ignore[reportUndefinedVariable]
        super().__init__(
            index=index,
            camera_index=camera_index,
            camera_code=camera_code,
            camera_name=camera_name,
            settings=settings,
            image_cropped_resized_settings=image_cropped_resized_settings,
            image_ai_settings=image_ai_settings,
            master_controller=master_controller
        )
        # set the camera type
        self._camera_type = CameraType.CV2

    def _open(self) -> bool:
        """Open the camera device."""
        # close the camera if it is open
        self._close()
        # open the camera
        self._camera = cv2.VideoCapture(self._camera_index)
        if not self._camera.isOpened():
            self._active = False
            return False
        self._active = True
        return True

    def _close(self):
        """Close the camera device."""
        # close the camera
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        # set the camera as inactive
        self._active = False

    def _get_supported_resolutions(self):
        with self._lock:
            #with self._lock_property_manipulation:
            supported_resolutions = []
            # self._ensure_device_open()
            self._open()
            if not self._active or self._camera is None:
                # if the camera is not active, return the default resolution
                supported_resolutions.append({"width": 640, "height": 480, "label": f"640 x 480"})
                return supported_resolutions

            # Check supported resolutions
            common_resolutions = [
                (320, 240),
                (640, 480),
                (800, 600),
                (848, 480),
                (960, 540),
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
            startup_width = 640
            startup_height = 480
            # Check if supported resolutions are available
            for width, height in common_resolutions:
                #if self._active and self._camera is not None:
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                new_width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                new_height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # skip resolutions that are less than 320x240
                if new_width < 320 or new_height < 240:
                    continue
                # check if the resolution is already in the list
                resolution_found = False
                for supported_resolution in supported_resolutions:
                    if supported_resolution["width"] == new_width and supported_resolution["height"] == new_height:
                        resolution_found = True
                        break
                if resolution_found:
                    continue
                # add the resolution to the list
                supported_resolutions.append({"width": new_width, "height": new_height, "label": f"{new_width} x {new_height}"})
                # check if there is a supported resolution that is 1920x1080 and set it as the startup resolution
                if new_width ==1920 and new_height == 1080:
                    startup_width = new_width
                    startup_height = new_height
            # set the startup resolution
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, startup_width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, startup_height)
            # self.width = startup_width
            # self.height = startup_height
            # close the camera
            self._close()
            #self.supported_resolutions = supported_resolutions
            return supported_resolutions

    def _get_capabilities(self) -> dict:
        """
        Gets the capabilities of the camera.
        """
        result = {
            "CAP_PROP_FRAME_WIDTH": {"min": 160, "max": 7680, "minSlider": 640, "maxSlider": 1920, "step": 16, "value": 1280, "enabled": True},
            "CAP_PROP_FRAME_HEIGHT": {"min": 120, "max": 4320, "minSlider": 480, "maxSlider": 1080, "step": 9, "value": 720, "enabled": True},
            "CAP_PROP_FPS": {"min": 1, "max": 240, "minSlider": 15, "maxSlider": 60, "step": 1, "value": 30, "enabled": True},
            "CAP_PROP_BITRATE": {"min": 0, "max": 10000, "minSlider": 2000, "maxSlider": 8000, "step": 500, "value": 4000, "enabled": True},
            "CAP_PROP_BUFFERSIZE": {"min": 1, "max": 10, "minSlider": 1, "maxSlider": 3, "step": 1, "value": 1, "enabled": True},
            "CAP_PROP_BRIGHTNESS": {"min": 0, "max": 255, "minSlider": 50, "maxSlider": 200, "step": 1, "value": 128, "enabled": True},
            "CAP_PROP_CONTRAST": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 100, "step": 1, "value": 32, "enabled": True},
            "CAP_PROP_HUE": {"min": -180, "max": 180, "minSlider": -20, "maxSlider": 20, "step": 1, "value": 0, "enabled": True},
            "CAP_PROP_SATURATION": {"min": 0, "max": 255, "minSlider": 50, "maxSlider": 150, "step": 1, "value": 64, "enabled": True},
            "CAP_PROP_SHARPNESS": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 50, "step": 1, "value": 0, "enabled": True},
            "CAP_PROP_GAMMA": {"min": 1, "max": 500, "minSlider": 100, "maxSlider": 200, "step": 10, "value": 100, "enabled": True},
            "CAP_PROP_WB_TEMPERATURE": {"min": 1000, "max": 10000, "minSlider": 100, "maxSlider": 10000, "step": 100, "value": 4500, "enabled": True},
            "CAP_PROP_BACKLIGHT": {"min": 0, "max": 4, "minSlider": 0, "maxSlider": 2, "step": 1, "value": 0, "enabled": True},
            "CAP_PROP_GAIN": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 128, "step": 1, "value": 0, "enabled": True},
            "CAP_PROP_FOCUS": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 255, "step": 5, "value": 0, "enabled": True},
            "CAP_PROP_EXPOSURE": {"min": -13, "max": 0, "minSlider": -7, "maxSlider": -3, "step": 1, "value": -6, "enabled": True},
            "CAP_PROP_AUTO_WB": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True},
            "CAP_PROP_AUTOFOCUS": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True},
            "CAP_PROP_AUTO_EXPOSURE": {"min": 0, "max": 3, "minSlider": 1, "maxSlider": 3, "step": 1, "value": 3, "enabled": True},
            "CAP_PROP_WHITE_BALANCE": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True}
        }
        return result

    def _get_camera_properties(self) -> dict:
        """
        Gets the camera properties from the camera.
        """
        with self._lock:
            try:
                active = self._active
                if not active:
                    self._open()
                result = {}
                result["index"] = self._index
                result["camera_index"] = self._camera_index
                result["camera_type"] = self._camera_type.value
                result["name"] = self._camera_name
                result["supported_resolutions"] = self.supported_resolutions
                if self._settings is not None:
                    result["flip_horizontal"] = self._settings["flip_horizontal"] if "flip_horizontal" in self._settings else False
                    result["flip_vertical"] = self._settings["flip_vertical"] if "flip_vertical" in self._settings else False
                    result["rotate"] = self._settings["rotate"] if "rotate" in self._settings else 0
                    result["crop_top"] = self._settings["crop_top"] if "crop_top" in self._settings else 0.0
                    result["crop_left"] = self._settings["crop_left"] if "crop_left" in self._settings else 0.0
                    result["crop_bottom"] = self._settings["crop_bottom"] if "crop_bottom" in self._settings else 0.0
                    result["crop_right"] = self._settings["crop_right"] if "crop_right" in self._settings else 0.0
                    result["stretch_enabled"] = self._settings["stretch_enabled"] if "stretch_enabled" in self._settings else False
                    result["stretch_width"] = self._settings["stretch_width"] if "stretch_width" in self._settings else 320
                    result["stretch_height"] = self._settings["stretch_height"] if "stretch_height" in self._settings else 240
                    result["static_reticle_x"] = self._settings["static_reticle_x"] if "static_reticle_x" in self._settings else 0.5
                    result["static_reticle_y"] = self._settings["static_reticle_y"] if "static_reticle_y" in self._settings else 0.5
                    result["static_reticle_color"] = self._settings["static_reticle_color"] if "static_reticle_color" in self._settings else "#88ff00cc"
                    result["static_reticle_size"] = self._settings["static_reticle_size"] if "static_reticle_size" in self._settings else 1.0
                    result["mask_polygons"] = self._settings["mask_polygons"] if "mask_polygons" in self._settings else []
                else:
                    result["flip_horizontal"] = False
                    result["flip_vertical"] = False
                    result["rotate"] = 0
                    result["crop_top"] = 0.0
                    result["crop_left"] = 0.0
                    result["crop_bottom"] = 0.0
                    result["crop_right"] = 0.0
                    result["stretch_enabled"] = False
                    result["stretch_width"] = 320
                    result["stretch_height"] = 240
                    result["static_reticle_x"] = 0.5
                    result["static_reticle_y"] = 0.5
                    result["static_reticle_color"] = "#88ff00cc"
                    result["static_reticle_size"] = 1.0
                    result["mask_polygons"] = []
                result["width"] = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                result["height"] = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                result["fps"] = int(self._camera.get(cv2.CAP_PROP_FPS))
                result["bitrate"] = int(self._camera.get(cv2.CAP_PROP_BITRATE))
                result["buffer_size"] = int(self._camera.get(cv2.CAP_PROP_BUFFERSIZE))
                result["brightness"] = float(self._camera.get(cv2.CAP_PROP_BRIGHTNESS))
                result["contrast"] = float(self._camera.get(cv2.CAP_PROP_CONTRAST))
                result["hue"] = float(self._camera.get(cv2.CAP_PROP_HUE))
                result["saturation"] = float(self._camera.get(cv2.CAP_PROP_SATURATION))
                result["sharpness"] = float(self._camera.get(cv2.CAP_PROP_SHARPNESS))
                result["gamma"] = float(self._camera.get(cv2.CAP_PROP_GAMMA))
                result["white_balance_temperature"] = float(self._camera.get(cv2.CAP_PROP_WB_TEMPERATURE))
                result["backlight"] = float(self._camera.get(cv2.CAP_PROP_BACKLIGHT))
                result["gain"] = float(self._camera.get(cv2.CAP_PROP_GAIN))
                result["focus"] = float(self._camera.get(cv2.CAP_PROP_FOCUS))
                result["exposure"] = float(self._camera.get(cv2.CAP_PROP_EXPOSURE))
                result["auto_white_balance_temperature"] = bool(self._camera.get(cv2.CAP_PROP_AUTO_WB))
                result["auto_focus"] = bool(self._camera.get(cv2.CAP_PROP_AUTOFOCUS))
                result["auto_exposure"] = bool(self._camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
            except Exception as e:
                print(f"Error getting settings: {e}")
            finally:
                if not active:
                    self._close()
                return result

    def _set_camera_properties(self, value: dict | None):
        with self._lock:
            try:
                # if value is None, set the default settings
                default_value = deepcopy(self._default_settings)
                if value is None:
                    value = default_value
                value["name"] = self._camera_name
                # set the default settings if not in the value
                # if "index" not in value:
                #     print(f"Index not in settings. Using default index {self._index}.")
                #     value["index"] = self._index
                # if "name" not in value:
                #     print(f"Name not in settings. Using default name {self._name}.")
                #     value["name"] = self._name
                # if "supported_resolutions" not in value:
                #     print(f"Supported resolutions not in settings. Using default supported resolutions {self.supported_resolutions}.")
                #     value["supported_resolutions"] = self.supported_resolutions
                # set properties
                if "width" not in value:
                    print(f"Width not in settings. Using default width {default_value['width']}.")
                    value["width"] = default_value["width"]
                if "height" not in value:
                    print(f"Height not in settings. Using default height {default_value['height']}.")
                    value["height"] = default_value["height"]
                if "fps" not in value:
                    print(f"FPS not in settings. Using default FPS {default_value['fps']}.")
                    value["fps"] = default_value["fps"]
                # if "bitrate" not in value:
                #     print(f"Bitrate not in settings. Using default bitrate {default_value['bitrate']}.")
                #     value["bitrate"] = default_value["bitrate"]
                # if "buffer_size" not in value:
                #     print(f"Buffer size not in settings. Using default buffer size {default_value['buffer_size']}.")
                #     value["buffer_size"] = default_value["buffer_size"]
                if "brightness" not in value:
                    print(f"Brightness not in settings. Using default brightness {default_value['brightness']}.")
                    value["brightness"] = default_value["brightness"]
                if "contrast" not in value:
                    print(f"Contrast not in settings. Using default contrast {default_value['contrast']}.")
                    value["contrast"] = default_value["contrast"]
                if "hue" not in value:
                    print(f"Hue not in settings. Using default hue {default_value['hue']}.")
                    value["hue"] = default_value["hue"]
                if "saturation" not in value:
                    print(f"Saturation not in settings. Using default saturation {default_value['saturation']}.")
                    value["saturation"] = default_value["saturation"]
                if "sharpness" not in value:
                    print(f"Sharpness not in settings. Using default sharpness {default_value['sharpness']}.")
                    value["sharpness"] = default_value["sharpness"]
                if "gamma" not in value:
                    print(f"Gamma not in settings. Using default gamma {default_value['gamma']}.")
                    value["gamma"] = default_value["gamma"]
                if "white_balance_temperature" not in value:
                    print(f"White balance temperature not in settings. Using default white balance temperature {default_value['white_balance_temperature']}.")
                    value["white_balance_temperature"] = default_value["white_balance_temperature"]
                if "backlight" not in value:
                    print(f"Backlight not in settings. Using default backlight {default_value['backlight']}.")
                    value["backlight"] = default_value["backlight"]
                if "gain" not in value:
                    print(f"Gain not in settings. Using default gain {default_value['gain']}.")
                    value["gain"] = default_value["gain"]
                if "focus" not in value:
                    print(f"Focus not in settings. Using default focus {default_value['focus']}.")
                    value["focus"] = default_value["focus"]
                if "exposure" not in value:
                    print(f"Exposure not in settings. Using default exposure {default_value['exposure']}.")
                    value["exposure"] = default_value["exposure"]
                if "auto_white_balance_temperature" not in value:
                    print(f"Auto white balance temperature not in settings. Using default auto white balance temperature {default_value['auto_white_balance_temperature']}.")
                    value["auto_white_balance_temperature"] = default_value["auto_white_balance_temperature"]
                if "auto_focus" not in value:
                    print(f"Auto focus not in settings. Using default auto focus {default_value['auto_focus']}.")
                    value["auto_focus"] = default_value["auto_focus"]
                if "auto_exposure" not in value:
                    print(f"Auto exposure not in settings. Using default auto exposure {default_value['auto_exposure']}.")
                    value["auto_exposure"] = default_value["auto_exposure"]
                # set the properties
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, value["width"])
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, value["height"])
                self._camera.set(cv2.CAP_PROP_FPS, value["fps"])
                # self._camera.set(cv2.CAP_PROP_BITRATE, value["bitrate"])
                # self._camera.set(cv2.CAP_PROP_BUFFERSIZE, value["buffer_size"])
                self._camera.set(cv2.CAP_PROP_BRIGHTNESS, value["brightness"])
                self._camera.set(cv2.CAP_PROP_CONTRAST, value["contrast"])
                self._camera.set(cv2.CAP_PROP_HUE, value["hue"])
                self._camera.set(cv2.CAP_PROP_SATURATION, value["saturation"])
                self._camera.set(cv2.CAP_PROP_SHARPNESS, value["sharpness"])
                self._camera.set(cv2.CAP_PROP_GAMMA, value["gamma"])
                self._camera.set(cv2.CAP_PROP_WB_TEMPERATURE, value["white_balance_temperature"])
                self._camera.set(cv2.CAP_PROP_BACKLIGHT, value["backlight"])
                self._camera.set(cv2.CAP_PROP_GAIN, value["gain"])
                self._camera.set(cv2.CAP_PROP_FOCUS, value["focus"])
                self._camera.set(cv2.CAP_PROP_EXPOSURE, value["exposure"])
                self._camera.set(cv2.CAP_PROP_AUTO_WB, 1.0 if value["auto_white_balance_temperature"] else 0.0)
                self._camera.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if value["auto_focus"] else 0.0)
                self._camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if value["auto_exposure"] else 0.0)
            except Exception as e:
                print(f"Error setting settings: {e}")

    def _get_image_ndarray(self) -> tuple[bool, cv2.typing.MatLike]:
        return self._camera.read()
