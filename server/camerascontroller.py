from cv2 import calibrateCamera
from .camera import Camera
from .cameracv2 import CameraCV2
from .cameradummy import CameraDummy
from .settingscontroller import get_settings_sync
import threading
import time

class CamerasController:
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self, *, master_controller: "MasterController"):  # pyright: ignore[reportUndefinedVariable]
        # List of cameras
        self._master_controller = master_controller
        self._cameras: list[Camera] = []
        self._lifecycle_lock = threading.RLock()
        self.reset()

    @property
    def cameras(self) -> list[Camera]:
        with self._lifecycle_lock:
            return list(self._cameras)

    def stop_camera(self, *, index: int) -> None:
        with self._lifecycle_lock:
            if index < 0 or index >= len(self._cameras):
                raise IndexError(f"Camera index {index} not found")
            self._cameras[index].stop()

    def reset(self, *, max_index=8, reset_to_default=False):
        """
        Reload the list of cameras.
        Parameters:
        max_index : int : The maximum index of the cameras to reload.
        reset_to_default : bool : If True, the cameras are reset to default settings.
        """
        with self._lifecycle_lock:
            print("Resetting cameras controller")
            settings = get_settings_sync()
            for camera in self._cameras:
                camera.stop()
            time.sleep(1.0)
            # clear the list of cameras
            self._cameras = []
            # Initialize all cameras from settings
            for i in range(max_index + 6):
                print(f"Creating camera with index \"{i}\"")
                found_camera_settings = None
                found_camera_code = None
                if "cameras" in settings:
                    for camera_code, camera_settings in settings["cameras"].items():
                        if "index" in camera_settings and camera_settings["index"] == i:
                            found_camera_settings = camera_settings
                            found_camera_code = camera_code
                            break
                # if reset_to_default is True, the camera settings are reset to default settings
                if reset_to_default:
                    found_camera_settings = None
                # create cameras
                try:
                    if i < 4:
                        camera_index = i
                        camera_name = f'{i}: dummy_camera_{camera_index}'
                        self._cameras.append(CameraDummy(index=i, 
                                                         camera_index=camera_index,
                                                         camera_code=found_camera_code,
                                                         camera_name=camera_name,
                                                         settings=found_camera_settings,
                                                         master_controller=self._master_controller))
                    elif i < 6:
                        try:
                            from .camerarpi import CameraRPI
                        except Exception as e:
                            print(f"CameraRPI is not available. Creating dummy camera instead. Error: {e}")
                            camera_index = -1
                            camera_name = f'{i}: rpidummy_camera_{camera_index}'
                            self._cameras.append(CameraDummy(index=i, 
                                                             camera_index=camera_index,
                                                             camera_code=found_camera_code,
                                                             camera_name=camera_name,
                                                             settings=found_camera_settings,
                                                             master_controller=self._master_controller))
                            continue
                        camera_index = i-4
                        camera_name = f'{i}: rpi_camera_{camera_index}'
                        self._cameras.append(CameraRPI(index=i, 
                                                       camera_index=camera_index,
                                                       camera_code=found_camera_code,
                                                       camera_name=camera_name,
                                                       settings=found_camera_settings,
                                                       master_controller=self._master_controller))
                    else:
                        camera_index = i-6
                        camera_name = f'{i}: cv2_camera_{camera_index}'
                        self._cameras.append(CameraCV2(index=i, 
                                                       camera_index=camera_index,
                                                       camera_code=found_camera_code,
                                                       camera_name=camera_name,
                                                       settings=found_camera_settings,
                                                       master_controller=self._master_controller))
                except Exception as e:
                    print(f"Error creating camera with index=\"{i}\", camera_index=\"{camera_index}\", camera_name=\"{camera_name}\". "
                          f"error: {e}. Creating dummy camera instead.")
                    self._cameras.append(CameraDummy(index=i, 
                                                     camera_index=-1,
                                                     camera_code=found_camera_code,
                                                     camera_name=camera_name,
                                                     settings=found_camera_settings,
                                                     master_controller=self._master_controller))
