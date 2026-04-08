"""
CameraRPI - Raspberry Pi Camera Module 3 implementation using Picamera2.

This module provides camera control for Raspberry Pi Camera Module 3 (Sony IMX708)
using the Picamera2 library. It supports all camera features including:
- Resolution control (up to 4608x2592 for stills, various video modes)
- Brightness, Contrast, Saturation, Sharpness
- Autofocus and manual lens position control
- Auto/Manual exposure with gain control
- Auto/Manual white balance with colour temperature/gains
"""

from .common import Frame
import time
import cv2
from .yolomodels import YOLOModels
from copy import deepcopy
from .camera import Camera, CameraType

# Try to import Picamera2 - will fail on non-Raspberry Pi systems
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2: 'Picamera2' | None = None
    controls: 'controls' | None = None


# =============================================================================
# CAMERA MODULE 3 RESOLUTION DEFINITIONS
# =============================================================================
CAMERA_MODULE_3_RESOLUTIONS = {
    "still": {
        "full_12mp": (4608, 2592),
        "8mp": (3840, 2160),
        "5mp": (2592, 1944),
        "2mp": (1920, 1080),
        "1mp": (1280, 720),
        "vga": (640, 480),
        "qvga": (320, 240),
    },
    "video": {
        "2304x1296_56fps": (2304, 1296),
        "2304x1296_30fps_hdr": (2304, 1296),
        "1920x1080_30fps": (1920, 1080),
        "1536x864_120fps": (1536, 864),
        "1280x720_60fps": (1280, 720),
        "640x480_120fps": (640, 480),
    },
}


# =============================================================================
# CONTROL MAPPINGS AND RANGES
# =============================================================================
# Picamera2 control ranges for Camera Module 3
CONTROL_RANGES = {
    "brightness": {"min": -1.0, "max": 1.0, "default": 0.0},
    "contrast": {"min": 0.0, "max": 32.0, "default": 1.0},
    "saturation": {"min": 0.0, "max": 32.0, "default": 1.0},
    "sharpness": {"min": 0.0, "max": 16.0, "default": 1.0},
    "exposure_time": {"min": 1, "max": 112000000, "default": 20000},
    "analogue_gain": {"min": 1.0, "max": 16.0, "default": 1.0},
    "exposure_value": {"min": -8.0, "max": 8.0, "default": 0.0},
    "lens_position": {"min": 0.0, "max": 10.0, "default": 1.0},
    "colour_gain_red": {"min": 0.0, "max": 32.0, "default": 1.0},
    "colour_gain_blue": {"min": 0.0, "max": 32.0, "default": 1.0},
}


class CameraRPI(Camera):
    """
    Raspberry Pi Camera Module 3 implementation using Picamera2.
    
    The device is closed by default. When frame property is accessed,
    the thread starts and captures frames. If frame is not accessed
    for more than 5 seconds, the thread stops and closes the device.
    
    Supports all Camera Module 3 features:
    - Resolution up to 4608x2592 (still) or 2304x1296@56fps (video)
    - Motorized autofocus with manual lens position control
    - Full image quality controls (brightness, contrast, saturation, sharpness)
    - Auto/manual exposure and white balance
    """
    
    def __init__(self, *, index: int,
                 camera_index: int,
                 camera_code: str = None,
                 camera_name: str = None,
                 settings: dict = None,
                 image_cropped_resized_settings: dict = None,
                 image_ai_settings: dict = None,
                 master_controller: "MasterController" = None):  # pyright: ignore[reportUndefinedVariable]
        """
        Initialize the Raspberry Pi Camera.
        
        Args:
            index: Camera index (usually 0 for primary camera)
            camera_name: Optional camera name
            settings: Optional initial settings dictionary
            image_cropped_resized_settings: Optional crop/resize settings
            image_ai_settings: Optional AI processing settings
        """
        # Check if Picamera2 is available
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("Picamera2 is not available. "
                               "This camera class requires Raspberry Pi with libcamera.")
        
        # Initialize Picamera2 reference
        self._picam2: Picamera2 = None
        self._current_resolution = (1920, 1080)
        self._current_fps = 30
        
        # Call parent constructor
        super().__init__(
            index=index,
            camera_index=camera_index,
            camera_code=camera_code,
            camera_name=camera_name if camera_name else f"rpi_camera_{index}",
            settings=settings,
            image_cropped_resized_settings=image_cropped_resized_settings,
            image_ai_settings=image_ai_settings,
            master_controller=master_controller
        )
        # set the camera type
        self._camera_type = CameraType.RPI

    def _open(self) -> bool:
        """
        Open the Picamera2 device.
        
        Returns:
            True if camera opened successfully, False otherwise.
        """
        try:
            # Close if already open
            self._close()
            
            # Create Picamera2 instance
            self._picam2 = Picamera2(camera_num=self._camera_index)
            
            # Get resolution from settings or use default
            width = self._settings.get("width", 1920) if self._settings else 1920
            height = self._settings.get("height", 1080) if self._settings else 1080
            self._current_resolution = (width, height)
            
            # Create video configuration for continuous capture
            # Using RGB888 format for direct OpenCV/numpy compatibility
            config = self._picam2.create_video_configuration(
                main={"format": "RGB888", "size": self._current_resolution},
                buffer_count=2
            )
            self._picam2.configure(config)
            
            # Start the camera
            self._picam2.start()
            
            # Allow camera to stabilize
            time.sleep(0.5)
            
            # Set camera reference for parent class compatibility
            self._camera = self._picam2
            self._active = True
            
            return True
            
        except Exception as e:
            print(f"Error opening Raspberry Pi camera index={self._index}, camera_index={self._camera_index} "
                  f"and camera_type={self._camera_type.value}, error opening camera: {e}")
            self._active = False
            self._camera = None
            self._picam2 = None
            return False

    def _close(self):
        """Close the Picamera2 device."""
        try:
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                except Exception:
                    pass
                try:
                    self._picam2.close()
                except Exception:
                    pass
                self._picam2 = None
            self._camera = None
            self._active = False
        except Exception as e:
            print(f"Error closing Raspberry Pi camera: {e}")
            self._active = False

    def _get_supported_resolutions(self) -> list:
        """
        Get list of supported resolutions for Camera Module 3.
        
        Returns:
            List of resolution dictionaries with width, height, and label.
        """
        supported_resolutions = []
        
        # Add common video resolutions that Camera Module 3 supports
        resolutions_to_check = [
            (320, 240, "QVGA"),
            (640, 480, "VGA"),
            (1280, 720, "HD 720p"),
            (1536, 864, "120fps Mode"),
            (1920, 1080, "Full HD"),
            (2304, 1296, "56fps Mode"),
            (2592, 1944, "5MP"),
            (3840, 2160, "4K UHD"),
            (4608, 2592, "Full 12MP"),
        ]
        
        for width, height, label in resolutions_to_check:
            supported_resolutions.append({
                "width": width,
                "height": height,
                "label": f"{width} x {height}"
            })
        
        return supported_resolutions

    def _get_capabilities(self) -> dict:
        """
        Get the capabilities/ranges of Camera Module 3 controls.
        
        Returns:
            Dictionary of control capabilities with min, max, step, default values.
        """
        # https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-2-picamera2-manual.pdf?disposition=inline appendix C
        result = {
            # Resolution and FPS
            "CAP_PROP_FRAME_WIDTH": {
                "min": 320, "max": 4608, "minSlider": 320, "maxSlider": 1920,
                "step": 16, "value": 1920, "enabled": True
            },
            "CAP_PROP_FRAME_HEIGHT": {
                "min": 240, "max": 2592, "minSlider": 240, "maxSlider": 1080,
                "step": 16, "value": 1080, "enabled": True
            },
            "CAP_PROP_FPS": {
                "min": 1, "max": 120, "minSlider": 1, "maxSlider": 60,
                "step": 1, "value": 30, "enabled": True
            },
            # Image Quality (Picamera2 uses different ranges than OpenCV)
            # {"Brightness": 0.0}
            "CAP_PROP_BRIGHTNESS": {
                "min": -100, "max": 100, "minSlider": -1, "maxSlider": 1,
                "step": 0.01, "value": 0, "enabled": True
            },
            # {"Contrast": 1.0}
            "CAP_PROP_CONTRAST": {
                "min": 0, "max": 3200, "minSlider": 0, "maxSlider": 32,
                "step": 0.1, "value": 1, "enabled": True
            },
            "CAP_PROP_HUE": {
                "min": -180, "max": 180, "minSlider": -20, "maxSlider": 20,
                "step": 1, "value": 0, "enabled": False
            },
            # {"Saturation": 1.0}
            "CAP_PROP_SATURATION": {
                "min": 0, "max": 3200, "minSlider": 0, "maxSlider": 32,
                "step": 0.1, "value": 1, "enabled": True
            },
            # {"Sharpness": 1.0}
            "CAP_PROP_SHARPNESS": {
                "min": 0, "max": 1600, "minSlider": 0, "maxSlider": 16,
                "step": 0.1, "value": 1, "enabled": True
            },
            "CAP_PROP_GAMMA": {
                "min": 100, "max": 500, "minSlider": 100, "maxSlider": 500,
                "step": 10, "value": 220, "enabled": False
            },
            "CAP_PROP_GAIN": {
                "min": 0, "max": 255, "minSlider": 0, "maxSlider": 128,
                "step": 1, "value": 0, "enabled": False
            },
            "CAP_PROP_BACKLIGHT": {
                "min": 0, "max": 4, "minSlider": 0, "maxSlider": 2,
                "step": 1, "value": 0, "enabled": False
            },
            # LensPosition (Camera Module 3 has motorized AF)
            "CAP_PROP_FOCUS": {
                "min": 0, "max": 1000, "minSlider": 0, "maxSlider": 32,
                "step": 0.1, "value": 1, "enabled": True
            },
            # {"AfMode": 0 ,"LensPosition": focus value} for manual focus or {"AfMode": 2 ,"AfTrigger": 0} for auto focus
            "CAP_PROP_AUTOFOCUS": {
                "min": 0, "max": 2, "minSlider": 0, "maxSlider": 2,
                "step": 1, "value": 2, "enabled": True
            },
            # ExposureTime
            "CAP_PROP_EXPOSURE": {
                "min": 1, "max": 1000000, "minSlider": 1, "maxSlider": 66666,
                "step": 100, "value": 20000, "enabled": True
            },
            # {'AeEnable': False}
            "CAP_PROP_AUTO_EXPOSURE": {
                "min": 0, "max": 1, "minSlider": 0, "maxSlider": 1,
                "step": 1, "value": 1, "enabled": True
            },
            # {'AnalogueGain': 1.0}
            # "CAP_PROP_GAIN": {
            #     "min": 100, "max": 1600, "minSlider": 100, "maxSlider": 1600,
            #     "step": 10, "value": 100, "enabled": True
            # },
            # {"AwbMode": 0 ,"ColourTemperature": 5500}
            "CAP_PROP_WB_TEMPERATURE": {
                "min": 2000, "max": 10000, "minSlider": 100, "maxSlider": 10000,
                "step": 100, "value": 5500, "enabled": True
            },
            # {'AwbEnable': False}
            "CAP_PROP_AUTO_WB": {
                "min": 0, "max": 1, "minSlider": 0, "maxSlider": 1,
                "step": 1, "value": 1, "enabled": True
            },
        }
        for key, value in result.items():
            value["min"] = -1000000000
            value["max"] = 1000000000
        return result

    def _build_base_camera_properties(self) -> dict:
        settings = self._settings if self._settings is not None else {}
        width = settings.get("width", self._current_resolution[0])
        height = settings.get("height", self._current_resolution[1])
        fps = settings.get("fps", self._current_fps)

        return {
            "index": self._index,
            "camera_index": self._camera_index,
            "camera_type": CameraType.RPI.value,
            "name": self._camera_name,
            "supported_resolutions": self.supported_resolutions,
            "flip_horizontal": settings.get("flip_horizontal", False),
            "flip_vertical": settings.get("flip_vertical", False),
            "rotate": settings.get("rotate", 0),
            "crop_top": settings.get("crop_top", 0.0),
            "crop_left": settings.get("crop_left", 0.0),
            "crop_bottom": settings.get("crop_bottom", 0.0),
            "crop_right": settings.get("crop_right", 0.0),
            "stretch_enabled": settings.get("stretch_enabled", False),
            "stretch_width": settings.get("stretch_width", 320),
            "stretch_height": settings.get("stretch_height", 240),
            "static_reticle_x": settings.get("static_reticle_x", 0.5),
            "static_reticle_y": settings.get("static_reticle_y", 0.5),
            "static_reticle_color": settings.get("static_reticle_color", "#88ff00cc"),
            "static_reticle_outline": settings.get("static_reticle_outline", "#000000cc"),
            "static_reticle_size": settings.get("static_reticle_size", 1.0),
            "mask_polygons": settings.get("mask_polygons", []),
            "width": width,
            "height": height,
            "fps": fps,
            "bitrate": settings.get("bitrate", -1),
            "buffer_size": settings.get("buffer_size", 1),
            "brightness": settings.get("brightness", 0),
            "contrast": settings.get("contrast", 1),
            "hue": settings.get("hue", 0),
            "saturation": settings.get("saturation", 1),
            "sharpness": settings.get("sharpness", 1),
            "gamma": settings.get("gamma", 1),
            "white_balance_temperature": settings.get("white_balance_temperature", 1),
            "backlight": settings.get("backlight", 0),
            "gain": settings.get("gain", 1),
            "focus": settings.get("focus", 1),
            "exposure": settings.get("exposure", 20000),
            "auto_white_balance_temperature": settings.get("auto_white_balance_temperature", True),
            "auto_focus": settings.get("auto_focus", True),
            "auto_exposure": settings.get("auto_exposure", True),
        }

    def _get_camera_properties(self) -> dict:
        """
        Get current camera properties/settings.
        
        Returns:
            Dictionary of current camera settings.
        """
        with self._lock:
            active = self._active
            result = self._build_base_camera_properties()
            
            try:
                if not active:
                    self._open()

                configuration = self._picam2.create_preview_configuration()
                print(f"configuration: {configuration}")
                controls = self._picam2.camera_controls
                print(f"controls: {controls}")

                # Get resolution from current configuration
                if self._picam2 is not None:
                    try:
                        config = self._picam2.camera_configuration()
                        if config and "main" in config:
                            result["width"] = config["main"]["size"][0]
                            result["height"] = config["main"]["size"][1]
                        else:
                            result["width"] = self._current_resolution[0]
                            result["height"] = self._current_resolution[1]
                    except Exception:
                        result["width"] = self._current_resolution[0]
                        result["height"] = self._current_resolution[1]
                
            except Exception as e:
                print(f"Error getting camera properties: {e}")
                
            finally:
                if not active:
                    self._close()
                return result

    def _set_camera_properties(self, value: dict | None):
        """
        Set camera properties/controls.
        
        Args:
            value: Dictionary of settings to apply. If None, uses defaults.
        """
        with self._lock:
            try:
                if self._picam2 is None:
                    return
                
                # Use default settings if value is None
                default_value = deepcopy(self._default_settings) if self._default_settings else {}
                if value is None:
                    value = default_value
                
                value["name"] = self._camera_name
                
                # Apply defaults for missing values
                if "width" not in value:
                    value["width"] = default_value.get("width", 1920)
                if "height" not in value:
                    value["height"] = default_value.get("height", 1080)
                if "fps" not in value:
                    value["fps"] = default_value.get("fps", 30)
                if "brightness" not in value:
                    value["brightness"] = default_value.get("brightness", 0)
                if "contrast" not in value:
                    value["contrast"] = default_value.get("contrast", 1)
                if "saturation" not in value:
                    value["saturation"] = default_value.get("saturation", 1)
                if "sharpness" not in value:
                    value["sharpness"] = default_value.get("sharpness", 1)
                # if "gamma" not in value:
                #     value["gamma"] = default_value.get("gamma", 220)
                # if "gain" not in value:
                #     value["gain"] = default_value.get("gain", 100)
                if "focus" not in value:
                    value["focus"] = default_value.get("focus", 0)
                if "exposure" not in value:
                    value["exposure"] = default_value.get("exposure", 20000)
                if "white_balance_temperature" not in value:
                    value["white_balance_temperature"] = default_value.get("white_balance_temperature", 5500)
                if "auto_white_balance_temperature" not in value:
                    value["auto_white_balance_temperature"] = default_value.get("auto_white_balance_temperature", True)
                if "auto_focus" not in value:
                    value["auto_focus"] = default_value.get("auto_focus", True)
                if "auto_exposure" not in value:
                    value["auto_exposure"] = default_value.get("auto_exposure", True)
                
                # Check if resolution changed - requires reconfiguration
                new_resolution = (value["width"], value["height"])
                if new_resolution != self._current_resolution:
                    self._reconfigure_resolution(width=new_resolution[0], height=new_resolution[1])
                
                # Build controls dictionary for Picamera2
                ctrl = {}
                
                # Image quality controls
                brightness_normalized = value["brightness"]
                #brightness_normalized = max(-1.0, min(1.0, brightness_normalized))
                ctrl["Brightness"] = brightness_normalized
                
                # Contrast: Convert from 0..3200 to 0.0..32.0 (100 = 1.0)
                contrast_normalized = value["contrast"]
                # contrast_normalized = max(0.0, min(32.0, contrast_normalized))
                ctrl["Contrast"] = contrast_normalized
                
                # Saturation: Convert from 0..3200 to 0.0..32.0 (100 = 1.0)
                saturation_normalized = value["saturation"]
                #saturation_normalized = max(0.0, min(32.0, saturation_normalized))
                ctrl["Saturation"] = saturation_normalized
                
                # Sharpness: Convert from 0..1600 to 0.0..16.0 (100 = 1.0)
                sharpness_normalized = value["sharpness"]
                #sharpness_normalized = max(0.0, min(16.0, sharpness_normalized))
                ctrl["Sharpness"] = sharpness_normalized
                
                # Autofocus controls
                auto_focus = value.get("auto_focus", True)
                if auto_focus:
                    ctrl["AfMode"] = controls.AfModeEnum.Continuous
                    ctrl["AfTrigger"] = controls.AfTriggerEnum.Start
                else:
                    ctrl["AfMode"] = controls.AfModeEnum.Manual
                    # Focus: Convert from 0..1000 to 0.0..10.0 diopters
                    lens_position = value["focus"]
                    # lens_position = max(0.0, min(10.0, lens_position))
                    ctrl["LensPosition"] = lens_position
                
                # Exposure controls
                auto_exposure = value.get("auto_exposure", True)
                ctrl["AeEnable"] = auto_exposure
                
                if not auto_exposure:
                    # Manual exposure time in microseconds
                    exposure_time = int(value["exposure"])
                    #exposure_time = max(1, min(112000000, exposure_time))
                    ctrl["ExposureTime"] = exposure_time
                    
                    # Analog gain: Convert from 100..1600 to 1.0..16.0
                    # analogue_gain = (value["gain"] + 255) / 32
                    # analogue_gain = max(1.0, min(16.0, analogue_gain))
                    # ctrl["AnalogueGain"] = analogue_gain
                
                # White balance controls
                auto_wb = value.get("auto_white_balance_temperature", True)
                ctrl["AwbEnable"] = auto_wb
                ctrl["AwbMode"] = controls.AwbModeEnum.Auto
                
                if not auto_wb:
                    temp = value.get("white_balance_temperature", 5500)
                    ctrl["ColourTemperature"] = int(temp)
                    ctrl["AwbMode"] = controls.AwbModeEnum.Custom
                
                # Apply controls
                try:
                    print(f"Setting controls: {ctrl}")
                    self._picam2.set_controls(ctrl)
                except Exception as e:
                    print(f"Warning: Some controls may not be supported: {e}")
                
            except Exception as e:
                print(f"Error setting camera properties: {e}")

    def _reconfigure_resolution(self, *, width: int, height: int):
        """
        Reconfigure camera with new resolution.
        
        Args:
            width: New frame width
            height: New frame height
        """
        try:
            if self._picam2 is None:
                return
            
            # Stop camera
            self._picam2.stop()
            
            # Create new configuration
            config = self._picam2.create_video_configuration(
                main={"format": "RGB888", "size": (width, height)},
                buffer_count=1
            )
            self._picam2.configure(config)
            
            # Restart camera
            self._picam2.start()
            
            # Update current resolution
            self._current_resolution = (width, height)
            
            # Allow camera to stabilize
            time.sleep(1.0)
            
        except Exception as e:
            print(f"Error reconfiguring resolution: {e}")

    def _get_image_ndarray(self) -> tuple[bool, cv2.typing.MatLike]:
        """
        Capture a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: numpy array in BGR format)
        """
        try:
            if self._picam2 is None:
                return False, None
            
            # Capture frame as numpy array (RGB format from Picamera2)
            frame_rgb = self._picam2.capture_array("main")
            
            if frame_rgb is None:
                return False, None

            return True, frame_rgb
            
            # Convert RGB to BGR for OpenCV compatibility
            # frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            # return True, frame_bgr
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False, None

    def trigger_autofocus(self):
        """
        Trigger a single autofocus cycle.
        
        This is useful when using single-shot autofocus mode.
        """
        try:
            if self._picam2 is not None:
                # Set to auto mode and trigger
                self._picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
                self._picam2.autofocus_cycle()
        except Exception as e:
            print(f"Error triggering autofocus: {e}")

    def get_metadata(self) -> dict:
        """
        Get current camera metadata including actual exposure, gain, etc.
        
        Returns:
            Dictionary of current camera metadata.
        """
        try:
            if self._picam2 is not None:
                metadata = self._picam2.capture_metadata()
                return metadata
        except Exception as e:
            print(f"Error getting metadata: {e}")
        return {}
