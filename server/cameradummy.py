import time
import cv2
import numpy as np
from .yolomodels import YOLOModels
# from .camerapostprocessing import NodeImage, NodeImageCroppedResized, NodeImageAI
from .camera import Camera, CameraType
from pathlib import Path

# =============================================================================
# CAMERA DUMMY
# =============================================================================
class CameraDummy(Camera):
    """
    Camera class that returns a dummy frame.
    """
    
    def __init__(self, *, index: int,
                 camera_index: int,
                 camera_code: str = None,
                 camera_name: str = None,
                 settings: dict = None,
                 image_cropped_resized_settings: dict = None,
                 image_ai_settings: dict = None,
                 master_controller: "MasterController" = None):  # pyright: ignore[reportUndefinedVariable]
        self._images = []
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
        self._camera_type = CameraType.DUMMY
        # set the camera to a dummy camera text
        self._camera = "DUMMY CAMERA"
        self._dummy_images = []
        self._error_images = [self._create_error_image()]
        for i in range(1):
            self._dummy_images.append(self._create_dummy_image())

    def _create_dummy_image(self) -> np.ndarray:
        """Create a dummy image with text "Dummy Camera"."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # write the text "Dummy" in the center of the frame
        text = "DUMMY CAMERA"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def _create_error_image(self) -> np.ndarray:
        """Create a dummy image with text "ERROR WHEN CREATING CAMERA"."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # write the text "ERROR WHEN CREATING CAMERA" in the center of the frame
        text = f"ERROR WHEN CREATING CAMERA"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 // 2 - text_size[1] - 5)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # write the text "CAMERA NAME" in the center of the frame
        text = f"{self._camera_name}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 // 2 + text_size[1])
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image


    def _open(self) -> bool:
        """Open the dummy camera."""
        self._active = True
        # load all images from the dummy_cam directory
        self._images = []
        this_file_path = Path(__file__).resolve()
        images_dir = this_file_path.parent / "dummy_cam" / str(self._camera_index)
        image_files = list(images_dir.glob("*.png"))
        sorted_image_files = sorted(image_files, key=lambda x: x.name)
        for image_file in sorted_image_files:
            image = cv2.imread(str(image_file))
            self._images.append(image)
        return True

    def _close(self):
        """Close the dummy camera."""
        self._active = False
        return

    def _get_supported_resolutions(self):
        return [{"width": 640, "height": 480, "label": f"640 x 480"}]

    def _get_capabilities(self) -> dict:
        result = {
            "CAP_PROP_FRAME_WIDTH": {"min": 160, "max": 7680, "minSlider": 640, "maxSlider": 1920, "step": 16, "value": 1280, "enabled": False},
            "CAP_PROP_FRAME_HEIGHT": {"min": 120, "max": 4320, "minSlider": 480, "maxSlider": 1080, "step": 9, "value": 720, "enabled": False},
            "CAP_PROP_FPS": {"min": 1, "max": 240, "minSlider": 15, "maxSlider": 60, "step": 1, "value": 30, "enabled": False},
            "CAP_PROP_BITRATE": {"min": 0, "max": 10000, "minSlider": 2000, "maxSlider": 8000, "step": 500, "value": 4000, "enabled": False},
            "CAP_PROP_BUFFERSIZE": {"min": 1, "max": 10, "minSlider": 1, "maxSlider": 3, "step": 1, "value": 1, "enabled": False},
            "CAP_PROP_BRIGHTNESS": {"min": 0, "max": 255, "minSlider": 50, "maxSlider": 200, "step": 1, "value": 128, "enabled": False},
            "CAP_PROP_CONTRAST": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 100, "step": 1, "value": 32, "enabled": False},
            "CAP_PROP_HUE": {"min": -180, "max": 180, "minSlider": -20, "maxSlider": 20, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_SATURATION": {"min": 0, "max": 255, "minSlider": 50, "maxSlider": 150, "step": 1, "value": 64, "enabled": False},
            "CAP_PROP_SHARPNESS": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 50, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_GAMMA": {"min": 1, "max": 500, "minSlider": 100, "maxSlider": 200, "step": 10, "value": 100, "enabled": False},
            "CAP_PROP_WB_TEMPERATURE": {"min": 1000, "max": 10000, "minSlider": 2800, "maxSlider": 6500, "step": 100, "value": 4500, "enabled": False},
            "CAP_PROP_BACKLIGHT": {"min": 0, "max": 4, "minSlider": 0, "maxSlider": 2, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_GAIN": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 128, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_FOCUS": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 255, "step": 5, "value": 0, "enabled": False},
            "CAP_PROP_EXPOSURE": {"min": -13, "max": 0, "minSlider": -7, "maxSlider": -3, "step": 1, "value": -6, "enabled": False},
            "CAP_PROP_AUTO_WB": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": False},
            "CAP_PROP_AUTOFOCUS": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": False},
            "CAP_PROP_AUTO_EXPOSURE": {"min": 0, "max": 3, "minSlider": 1, "maxSlider": 3, "step": 1, "value": 3, "enabled": False},
            "CAP_PROP_WHITE_BALANCE": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": False}
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
                result["width"] = 640
                result["height"] = 480
                result["fps"] = 30
                result["bitrate"] = 4000
                result["buffer_size"] = 1
                result["brightness"] = 128
                result["contrast"] = 32
                result["hue"] = 0
                result["saturation"] = 64
                result["sharpness"] = 0
                result["gamma"] = 100
                result["white_balance_temperature"] = 4500
                result["backlight"] = 0
                result["gain"] = 0
                result["focus"] = 0
                result["exposure"] = -6
                result["auto_white_balance_temperature"] = True
                result["auto_focus"] = True
                result["auto_exposure"] = True
            except Exception as e:
                print(f"Error getting settings: {e}")
            finally:
                if not active:
                    self._close()
                return result

    def _set_camera_properties(self, value: dict | None):
        return

    def _get_image_ndarray(self) -> tuple[bool, cv2.typing.MatLike]:
        if self._camera_index == -1:
            return (True, self._error_images[0].copy())
        elif len(self._images) > 0:
            now = time.time()
            len_images = len(self._images) - 1
            next_image_index = int(now * 8) % (len_images * 2)
            next_image_index = abs(next_image_index - len_images)
            print(f"next_image_index: {next_image_index}")
            result = self._images[next_image_index].copy()
            return (True, result)
        else:
            return (True, self._dummy_images[0].copy())
