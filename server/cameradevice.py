"""
Pure camera hardware abstraction — no threads, no frame storage.

Each CameraDevice subclass encapsulates one way to acquire images.  Instances
are created inside the worker process (after ``spawn``) so heavy imports
(picamera2) happen only when actually needed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class CameraDevice(ABC):
    """Minimal interface that the worker process calls for every camera."""

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_image(self) -> tuple[bool, np.ndarray | None]: ...

    @abstractmethod
    def get_properties(self) -> dict: ...

    @abstractmethod
    def set_properties(self, settings: dict) -> None: ...

    @abstractmethod
    def get_supported_resolutions(self) -> list[dict]: ...

    @abstractmethod
    def get_capabilities(self) -> dict: ...


# =========================================================================
# CV2 (OpenCV VideoCapture)
# =========================================================================
class CameraCV2Device(CameraDevice):

    def __init__(self, *, camera_index: int, camera_name: str, settings: dict) -> None:
        self._camera_index = camera_index
        self._camera_name = camera_name
        self._cap: cv2.VideoCapture | None = None
        self._settings = settings or {}

    def open(self) -> bool:
        self.close()
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._cap = None
            return False
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_image(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def get_supported_resolutions(self) -> list[dict]:
        supported: list[dict] = []
        opened_here = False
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._camera_index)
            opened_here = True
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                return [{"width": 640, "height": 480, "label": "640 x 480"}]
        try:
            common = [
                (320, 240), (640, 480), (800, 600), (848, 480),
                (960, 540), (960, 720), (1024, 768), (1280, 960),
                (1280, 720), (1600, 1200), (1920, 1080), (2560, 1440),
                (3200, 1800), (3840, 2160), (4096, 2160), (5120, 2880),
                (6016, 3384), (7680, 4320), (8192, 4608),
            ]
            startup_w, startup_h = 640, 480
            for w, h in common:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                nw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                nh = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if nw < 320 or nh < 240:
                    continue
                if any(r["width"] == nw and r["height"] == nh for r in supported):
                    continue
                supported.append({"width": nw, "height": nh, "label": f"{nw} x {nh}"})
                if nw == 1920 and nh == 1080:
                    startup_w, startup_h = nw, nh
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, startup_w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, startup_h)
        finally:
            if opened_here:
                self._cap.release()
                self._cap = None
        return supported or [{"width": 640, "height": 480, "label": "640 x 480"}]

    def get_capabilities(self) -> dict:
        return {
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
            "CAP_PROP_WHITE_BALANCE": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True},
        }

    def get_properties(self) -> dict:
        result: dict = {}
        opened_here = False
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._camera_index)
            opened_here = True
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                return self._get_default_props()
        try:
            result = self._get_base_props()
            result["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            result["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result["fps"] = int(self._cap.get(cv2.CAP_PROP_FPS))
            result["bitrate"] = int(self._cap.get(cv2.CAP_PROP_BITRATE))
            result["buffer_size"] = int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE))
            result["brightness"] = float(self._cap.get(cv2.CAP_PROP_BRIGHTNESS))
            result["contrast"] = float(self._cap.get(cv2.CAP_PROP_CONTRAST))
            result["hue"] = float(self._cap.get(cv2.CAP_PROP_HUE))
            result["saturation"] = float(self._cap.get(cv2.CAP_PROP_SATURATION))
            result["sharpness"] = float(self._cap.get(cv2.CAP_PROP_SHARPNESS))
            result["gamma"] = float(self._cap.get(cv2.CAP_PROP_GAMMA))
            result["white_balance_temperature"] = float(self._cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
            result["backlight"] = float(self._cap.get(cv2.CAP_PROP_BACKLIGHT))
            result["gain"] = float(self._cap.get(cv2.CAP_PROP_GAIN))
            result["focus"] = float(self._cap.get(cv2.CAP_PROP_FOCUS))
            result["exposure"] = float(self._cap.get(cv2.CAP_PROP_EXPOSURE))
            result["auto_white_balance_temperature"] = bool(self._cap.get(cv2.CAP_PROP_AUTO_WB))
            result["auto_focus"] = bool(self._cap.get(cv2.CAP_PROP_AUTOFOCUS))
            result["auto_exposure"] = bool(self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        except Exception as e:
            print(f"Error getting CV2 camera properties: {e}")
        finally:
            if opened_here:
                self._cap.release()
                self._cap = None
        return result

    def set_properties(self, settings: dict) -> None:
        if self._cap is None:
            return
        try:
            s = settings
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.get("width", 640))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.get("height", 480))
            self._cap.set(cv2.CAP_PROP_FPS, s.get("fps", 30))
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, s.get("brightness", 128))
            self._cap.set(cv2.CAP_PROP_CONTRAST, s.get("contrast", 32))
            self._cap.set(cv2.CAP_PROP_HUE, s.get("hue", 0))
            self._cap.set(cv2.CAP_PROP_SATURATION, s.get("saturation", 64))
            self._cap.set(cv2.CAP_PROP_SHARPNESS, s.get("sharpness", 0))
            self._cap.set(cv2.CAP_PROP_GAMMA, s.get("gamma", 100))
            self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s.get("white_balance_temperature", 4500))
            self._cap.set(cv2.CAP_PROP_BACKLIGHT, s.get("backlight", 0))
            self._cap.set(cv2.CAP_PROP_GAIN, s.get("gain", 0))
            self._cap.set(cv2.CAP_PROP_FOCUS, s.get("focus", 0))
            self._cap.set(cv2.CAP_PROP_EXPOSURE, s.get("exposure", -6))
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if s.get("auto_white_balance_temperature", True) else 0.0)
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if s.get("auto_focus", True) else 0.0)
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if s.get("auto_exposure", True) else 0.0)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"Error setting CV2 camera properties: {e}")

    def _get_base_props(self) -> dict:
        s = self._settings
        return {
            "flip_horizontal": s.get("flip_horizontal", False),
            "flip_vertical": s.get("flip_vertical", False),
            "rotate": s.get("rotate", 0),
            "crop_top": s.get("crop_top", 0.0),
            "crop_left": s.get("crop_left", 0.0),
            "crop_bottom": s.get("crop_bottom", 0.0),
            "crop_right": s.get("crop_right", 0.0),
            "stretch_enabled": s.get("stretch_enabled", False),
            "stretch_width": s.get("stretch_width", 320),
            "stretch_height": s.get("stretch_height", 240),
            "static_reticle_x": s.get("static_reticle_x", 0.5),
            "static_reticle_y": s.get("static_reticle_y", 0.5),
            "static_reticle_color": s.get("static_reticle_color", "#88ff00cc"),
            "static_reticle_size": s.get("static_reticle_size", 1.0),
            "mask_polygons": s.get("mask_polygons", []),
        }

    def _get_default_props(self) -> dict:
        result = self._get_base_props()
        result.update({
            "width": 640, "height": 480, "fps": 30,
            "bitrate": 4000, "buffer_size": 1, "brightness": 128,
            "contrast": 32, "hue": 0, "saturation": 64,
            "sharpness": 0, "gamma": 100,
            "white_balance_temperature": 4500, "backlight": 0,
            "gain": 0, "focus": 0, "exposure": -6,
            "auto_white_balance_temperature": True,
            "auto_focus": True, "auto_exposure": True,
        })
        return result


# =========================================================================
# Raspberry Pi (picamera2)
# =========================================================================
class CameraRPIDevice(CameraDevice):

    def __init__(self, *, camera_index: int, camera_name: str, settings: dict) -> None:
        self._camera_index = camera_index
        self._camera_name = camera_name
        self._settings = settings or {}
        self._picam2 = None
        self._current_resolution = (
            self._settings.get("width", 1920),
            self._settings.get("height", 1080),
        )
        self._current_fps = self._settings.get("fps", 30)

    def open(self) -> bool:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            print(f"picamera2 not available: {exc}")
            return False
        try:
            self.close()
            self._picam2 = Picamera2(camera_num=self._camera_index)
            w, h = self._current_resolution
            config = self._picam2.create_video_configuration(
                main={"format": "RGB888", "size": (w, h)},
                buffer_count=2,
            )
            self._picam2.configure(config)
            self._picam2.start()
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error opening RPi camera {self._camera_index}: {e}")
            self.close()
            return False

    def close(self) -> None:
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

    def get_image(self) -> tuple[bool, np.ndarray | None]:
        if self._picam2 is None:
            return False, None
        try:
            frame = self._picam2.capture_array("main")
            if frame is None:
                return False, None
            return True, frame
        except Exception as e:
            print(f"RPi capture error: {e}")
            return False, None

    def get_supported_resolutions(self) -> list[dict]:
        entries = [
            (320, 240), (640, 480), (1280, 720), (1536, 864),
            (1920, 1080), (2304, 1296), (2592, 1944),
            (3840, 2160), (4608, 2592),
        ]
        return [{"width": w, "height": h, "label": f"{w} x {h}"} for w, h in entries]

    def get_capabilities(self) -> dict:
        result = {
            "CAP_PROP_FRAME_WIDTH": {"min": 320, "max": 4608, "minSlider": 320, "maxSlider": 1920, "step": 16, "value": 1920, "enabled": True},
            "CAP_PROP_FRAME_HEIGHT": {"min": 240, "max": 2592, "minSlider": 240, "maxSlider": 1080, "step": 16, "value": 1080, "enabled": True},
            "CAP_PROP_FPS": {"min": 1, "max": 120, "minSlider": 1, "maxSlider": 60, "step": 1, "value": 30, "enabled": True},
            "CAP_PROP_BRIGHTNESS": {"min": -100, "max": 100, "minSlider": -1, "maxSlider": 1, "step": 0.01, "value": 0, "enabled": True},
            "CAP_PROP_CONTRAST": {"min": 0, "max": 3200, "minSlider": 0, "maxSlider": 32, "step": 0.1, "value": 1, "enabled": True},
            "CAP_PROP_HUE": {"min": -180, "max": 180, "minSlider": -20, "maxSlider": 20, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_SATURATION": {"min": 0, "max": 3200, "minSlider": 0, "maxSlider": 32, "step": 0.1, "value": 1, "enabled": True},
            "CAP_PROP_SHARPNESS": {"min": 0, "max": 1600, "minSlider": 0, "maxSlider": 16, "step": 0.1, "value": 1, "enabled": True},
            "CAP_PROP_GAMMA": {"min": 100, "max": 500, "minSlider": 100, "maxSlider": 500, "step": 10, "value": 220, "enabled": False},
            "CAP_PROP_GAIN": {"min": 0, "max": 255, "minSlider": 0, "maxSlider": 128, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_BACKLIGHT": {"min": 0, "max": 4, "minSlider": 0, "maxSlider": 2, "step": 1, "value": 0, "enabled": False},
            "CAP_PROP_FOCUS": {"min": 0, "max": 1000, "minSlider": 0, "maxSlider": 32, "step": 0.1, "value": 1, "enabled": True},
            "CAP_PROP_AUTOFOCUS": {"min": 0, "max": 2, "minSlider": 0, "maxSlider": 2, "step": 1, "value": 2, "enabled": True},
            "CAP_PROP_EXPOSURE": {"min": 1, "max": 1000000, "minSlider": 1, "maxSlider": 66666, "step": 100, "value": 20000, "enabled": True},
            "CAP_PROP_AUTO_EXPOSURE": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True},
            "CAP_PROP_WB_TEMPERATURE": {"min": 2000, "max": 10000, "minSlider": 100, "maxSlider": 10000, "step": 100, "value": 5500, "enabled": True},
            "CAP_PROP_AUTO_WB": {"min": 0, "max": 1, "minSlider": 0, "maxSlider": 1, "step": 1, "value": 1, "enabled": True},
        }
        for v in result.values():
            v["min"] = -1000000000
            v["max"] = 1000000000
        return result

    def get_properties(self) -> dict:
        s = self._settings
        return {
            "flip_horizontal": s.get("flip_horizontal", False),
            "flip_vertical": s.get("flip_vertical", False),
            "rotate": s.get("rotate", 0),
            "crop_top": s.get("crop_top", 0.0),
            "crop_left": s.get("crop_left", 0.0),
            "crop_bottom": s.get("crop_bottom", 0.0),
            "crop_right": s.get("crop_right", 0.0),
            "stretch_enabled": s.get("stretch_enabled", False),
            "stretch_width": s.get("stretch_width", 320),
            "stretch_height": s.get("stretch_height", 240),
            "static_reticle_x": s.get("static_reticle_x", 0.5),
            "static_reticle_y": s.get("static_reticle_y", 0.5),
            "static_reticle_color": s.get("static_reticle_color", "#88ff00cc"),
            "static_reticle_outline": s.get("static_reticle_outline", "#000000cc"),
            "static_reticle_size": s.get("static_reticle_size", 1.0),
            "mask_polygons": s.get("mask_polygons", []),
            "width": self._current_resolution[0],
            "height": self._current_resolution[1],
            "fps": self._current_fps,
            "bitrate": s.get("bitrate", -1),
            "buffer_size": s.get("buffer_size", 1),
            "brightness": s.get("brightness", 0),
            "contrast": s.get("contrast", 1),
            "hue": s.get("hue", 0),
            "saturation": s.get("saturation", 1),
            "sharpness": s.get("sharpness", 1),
            "gamma": s.get("gamma", 1),
            "white_balance_temperature": s.get("white_balance_temperature", 1),
            "backlight": s.get("backlight", 0),
            "gain": s.get("gain", 1),
            "focus": s.get("focus", 1),
            "exposure": s.get("exposure", 20000),
            "auto_white_balance_temperature": s.get("auto_white_balance_temperature", True),
            "auto_focus": s.get("auto_focus", True),
            "auto_exposure": s.get("auto_exposure", True),
        }

    def set_properties(self, settings: dict) -> None:
        if self._picam2 is None:
            return
        try:
            from libcamera import controls as lc_controls
        except ImportError:
            return
        try:
            s = settings
            new_res = (s.get("width", self._current_resolution[0]),
                       s.get("height", self._current_resolution[1]))
            if new_res != self._current_resolution:
                self._reconfigure_resolution(new_res[0], new_res[1])

            ctrl: dict = {}
            ctrl["Brightness"] = s.get("brightness", 0)
            ctrl["Contrast"] = s.get("contrast", 1)
            ctrl["Saturation"] = s.get("saturation", 1)
            ctrl["Sharpness"] = s.get("sharpness", 1)

            if s.get("auto_focus", True):
                ctrl["AfMode"] = lc_controls.AfModeEnum.Continuous
                ctrl["AfTrigger"] = lc_controls.AfTriggerEnum.Start
            else:
                ctrl["AfMode"] = lc_controls.AfModeEnum.Manual
                ctrl["LensPosition"] = s.get("focus", 1)

            auto_exp = s.get("auto_exposure", True)
            ctrl["AeEnable"] = auto_exp
            if not auto_exp:
                ctrl["ExposureTime"] = int(s.get("exposure", 20000))

            auto_wb = s.get("auto_white_balance_temperature", True)
            ctrl["AwbEnable"] = auto_wb
            ctrl["AwbMode"] = lc_controls.AwbModeEnum.Auto
            if not auto_wb:
                ctrl["ColourTemperature"] = int(s.get("white_balance_temperature", 5500))
                ctrl["AwbMode"] = lc_controls.AwbModeEnum.Custom

            self._picam2.set_controls(ctrl)
        except Exception as e:
            print(f"Error setting RPi camera properties: {e}")

    def _reconfigure_resolution(self, width: int, height: int) -> None:
        if self._picam2 is None:
            return
        try:
            self._picam2.stop()
            config = self._picam2.create_video_configuration(
                main={"format": "RGB888", "size": (width, height)},
                buffer_count=1,
            )
            self._picam2.configure(config)
            self._picam2.start()
            self._current_resolution = (width, height)
            time.sleep(1.0)
        except Exception as e:
            print(f"Error reconfiguring RPi resolution: {e}")


# =========================================================================
# Dummy (test / fallback)
# =========================================================================
class CameraDummyDevice(CameraDevice):

    def __init__(self, *, camera_index: int, camera_name: str, settings: dict) -> None:
        self._camera_index = camera_index
        self._camera_name = camera_name
        self._settings = settings or {}
        self._images: list[np.ndarray] = []
        self._dummy_images = [self._create_dummy_image()]
        self._error_images = [self._create_error_image()]

    def _create_dummy_image(self) -> np.ndarray:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "DUMMY CAMERA"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(image, text, ((640 - ts[0]) // 2, (480 + ts[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def _create_error_image(self) -> np.ndarray:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        t1 = "ERROR WHEN CREATING CAMERA"
        ts1 = cv2.getTextSize(t1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(image, t1, ((640 - ts1[0]) // 2, 480 // 2 - ts1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        t2 = str(self._camera_name)
        ts2 = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(image, t2, ((640 - ts2[0]) // 2, 480 // 2 + ts2[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def open(self) -> bool:
        self._images = []
        images_dir = Path(__file__).resolve().parent / "dummy_cam" / str(self._camera_index)
        if images_dir.is_dir():
            for f in sorted(images_dir.glob("*.png"), key=lambda p: p.name):
                img = cv2.imread(str(f))
                if img is not None:
                    self._images.append(img)
        return True

    def close(self) -> None:
        pass

    def get_image(self) -> tuple[bool, np.ndarray | None]:
        if self._camera_index == -1:
            return True, self._error_images[0].copy()
        if self._images:
            n = len(self._images) - 1
            idx = int(time.time() * 8) % (n * 2) if n > 0 else 0
            idx = abs(idx - n) if n > 0 else 0
            return True, self._images[idx].copy()
        return True, self._dummy_images[0].copy()

    def get_supported_resolutions(self) -> list[dict]:
        return [{"width": 640, "height": 480, "label": "640 x 480"}]

    def get_capabilities(self) -> dict:
        base = CameraCV2Device.get_capabilities(CameraCV2Device.__new__(CameraCV2Device))
        for v in base.values():
            v["enabled"] = False
        return base

    def get_properties(self) -> dict:
        s = self._settings
        return {
            "flip_horizontal": s.get("flip_horizontal", False),
            "flip_vertical": s.get("flip_vertical", False),
            "rotate": s.get("rotate", 0),
            "crop_top": s.get("crop_top", 0.0),
            "crop_left": s.get("crop_left", 0.0),
            "crop_bottom": s.get("crop_bottom", 0.0),
            "crop_right": s.get("crop_right", 0.0),
            "stretch_enabled": s.get("stretch_enabled", False),
            "stretch_width": s.get("stretch_width", 320),
            "stretch_height": s.get("stretch_height", 240),
            "static_reticle_x": s.get("static_reticle_x", 0.5),
            "static_reticle_y": s.get("static_reticle_y", 0.5),
            "static_reticle_color": s.get("static_reticle_color", "#88ff00cc"),
            "static_reticle_size": s.get("static_reticle_size", 1.0),
            "mask_polygons": s.get("mask_polygons", []),
            "width": 640, "height": 480, "fps": 30,
            "bitrate": 4000, "buffer_size": 1, "brightness": 128,
            "contrast": 32, "hue": 0, "saturation": 64,
            "sharpness": 0, "gamma": 100,
            "white_balance_temperature": 4500, "backlight": 0,
            "gain": 0, "focus": 0, "exposure": -6,
            "auto_white_balance_temperature": True,
            "auto_focus": True, "auto_exposure": True,
        }

    def set_properties(self, settings: dict) -> None:
        self._settings = settings


# =========================================================================
# Factory
# =========================================================================
def create_camera_device(
    *,
    camera_type: str,
    camera_index: int,
    camera_name: str,
    settings: dict,
) -> CameraDevice:
    """Create the right CameraDevice for the given type string.

    Used by both the main process (probing resolutions/capabilities) and
    the worker process (actual capture).
    """
    if camera_type == "rpi":
        try:
            return CameraRPIDevice(
                camera_index=camera_index,
                camera_name=camera_name,
                settings=settings,
            )
        except Exception as e:
            print(f"CameraRPIDevice unavailable ({e}), falling back to dummy")
            return CameraDummyDevice(
                camera_index=-1,
                camera_name=camera_name,
                settings=settings,
            )
    if camera_type == "cv2":
        return CameraCV2Device(
            camera_index=camera_index,
            camera_name=camera_name,
            settings=settings,
        )
    return CameraDummyDevice(
        camera_index=camera_index,
        camera_name=camera_name,
        settings=settings,
    )
