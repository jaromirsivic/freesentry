from .camera import Camera
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
        self._master_controller = master_controller
        self._cameras: list[Camera] = []
        self._lifecycle_lock = threading.RLock()
        self.reset()

    @property
    def cameras(self) -> list[Camera]:
        with self._lifecycle_lock:
            return list(self._cameras)

    def _stop_cameras_locked(self) -> None:
        stop_errors: list[tuple[str, Exception]] = []
        for camera in self._cameras:
            try:
                camera.stop()
            except Exception as exc:
                camera_name = getattr(camera, "camera_name", repr(camera))
                print(f"Error stopping camera worker '{camera_name}': {exc}")
                stop_errors.append((camera_name, exc))

        if stop_errors:
            failed_cameras = ", ".join(name for name, _ in stop_errors)
            raise RuntimeError(f"Failed to stop camera workers: {failed_cameras}") from stop_errors[0][1]

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_cameras_locked()

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
            self._stop_cameras_locked()
            time.sleep(1.0)
            settings = get_settings_sync()
            self._cameras = []

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
                if reset_to_default:
                    found_camera_settings = None

                # Determine camera_type and camera_index based on index range
                if i < 4:
                    camera_index = i
                    camera_type = "dummy"
                    camera_name = f'{i}: dummy_camera_{camera_index}'
                elif i < 6:
                    camera_index = i - 4
                    camera_type = "rpi"
                    camera_name = f'{i}: rpi_camera_{camera_index}'
                else:
                    camera_index = i - 6
                    camera_type = "cv2"
                    camera_name = f'{i}: cv2_camera_{camera_index}'

                try:
                    self._cameras.append(Camera(
                        index=i,
                        camera_index=camera_index,
                        camera_code=found_camera_code,
                        camera_name=camera_name,
                        camera_type=camera_type,
                        settings=found_camera_settings,
                        master_controller=self._master_controller,
                    ))
                except Exception as e:
                    print(f"Error creating camera with index=\"{i}\", camera_index=\"{camera_index}\", "
                          f"camera_name=\"{camera_name}\". error: {e}. Creating dummy camera instead.")
                    self._cameras.append(Camera(
                        index=i,
                        camera_index=-1,
                        camera_code=found_camera_code,
                        camera_name=camera_name,
                        camera_type="dummy",
                        settings=found_camera_settings,
                        master_controller=self._master_controller,
                    ))
