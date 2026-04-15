from .common import Frame
import time
import cv2
import threading
import numpy as np
from .ai_setup_constants import DEFAULT_DEVICE
from .yolomodels import YOLOModels
# from .camerapostprocessing import NodeImage, NodeImageCroppedResized, NodeImageAI
from copy import deepcopy
from .cameraai import draw_pose, get_pose_dict, translate_raw_pose_to_pose_dict
from .common import EPSILON_DELAY
from enum import Enum
from .settingscontroller import get_settings_sync
from .aiagent import AIAgent

class CameraType(Enum):
    CV2 = "cv2"
    RPI = "rpi"
    DUMMY = "dummy"
    UNKNOWN = "unknown"


FLIP_HORIZONTAL = 1
FLIP_VERTICAL = 0
FLIP_BOTH = -1


class Camera(threading.Thread):
    """
    Camera class that captures frames in a background thread.
    
    The device is closed by default. When frame property is accessed,
    the thread starts and captures frames. If frame is not accessed
    for more than 5 seconds, the thread stops and closes the device.
    """
    
    TIMEOUT_SECONDS = 5
    STOP_TIMEOUT_SECONDS = 4.0
    
    def __init__(self, *, index: int,
                 camera_index: int,
                 camera_code: str = None,
                 camera_name: str = None,
                 settings: dict = None,
                 image_cropped_resized_settings: dict = None,
                 image_ai_settings: dict = None,
                 master_controller: "MasterController" = None):  # pyright: ignore[reportUndefinedVariable]
        super().__init__()
        self.daemon = True
        self._master_controller = master_controller
        self._index = index
        self._camera_index = camera_index
        self._camera_code = camera_code
        self._camera_type = CameraType.UNKNOWN
        self._camera_name = camera_name if camera_name is not None else f'{index}: Loadeing, please wait a minute...'
        self._camera: cv2.VideoCapture = None
        self._lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._latest_ai_time = 0
        self._latest_ai_code = ""
        # Frame properties
        self._frame: Frame = self._create_blank_frame()
        self._frame_masked: Frame = self._create_blank_frame()
        self._frame_masked_ai: Frame = self._create_blank_frame()
        self._last_access_time_raw_frame: float = 0
        self._last_access_time_masked_frame: float = 0
        self._last_access_time_masked_ai_frame: float = 0
        self._lock_frame = threading.Lock()
        self._lock_frame_masked = threading.Lock()
        self._lock_frame_masked_ai = threading.Lock()
        #self._lock_property_manipulation = threading.Lock()
        self._active = False
        # Flip and rotate settings (software post-processing)
        self._flip_horizontal = False
        self._flip_vertical = False
        self._rotate = 0
        # Create post processing filters
        self._image: Frame = None
        self._image_cropped_resized: Frame = None
        self._image_ai: Frame = None
        # Initialize settings
        self._settings: dict = None
        self._settings_version = 0
        self._applied_settings_version = -1
        self._settings_modified = False
        self._stream_id = 0
        # Supported resolutions
        self.supported_resolutions = self._get_supported_resolutions()
        # Settings
        self._default_settings = self._get_camera_properties()
        self._default_settings["auto_focus"] = True
        self._default_settings["auto_white_balance_temperature"] = True
        self._default_settings["auto_exposure"] = True
        camera_settings = deepcopy(self._default_settings)
        if settings is not None:
            camera_settings.update(settings)
        self.settings = camera_settings

    def _open(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    def _close(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_supported_resolutions(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_capabilities(self) -> dict:
       raise NotImplementedError("Subclasses must implement this method")

    def _get_camera_properties(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    def _set_camera_properties(self, value: dict | None):
        raise NotImplementedError("Subclasses must implement this method")

    def _get_image_ndarray(self) -> tuple[bool, cv2.typing.MatLike]:
        raise NotImplementedError("Subclasses must implement this method")

    def _create_blank_frame(self) -> Frame:
        """Create a black frame of size 640x480."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # write the text "Loading, please wait a minute..." in the center of the frame
        text = "Loading, please wait a minute..."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return Frame(valid=False, image=image, time=0)

    # @property
    # def index(self) -> int:
    #     return self._index

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def settings(self) -> dict:
        """
        Returns the settings of the camera.
        If settings are not set, gets the default settings.
        """
        with self._state_lock:
            if self._settings is not None:
                return deepcopy(self._settings)
        return deepcopy(self._get_camera_properties())

    @settings.setter
    def settings(self, value: dict | None):
        """
        Set the settings of the camera.
        If value is None, sets the default settings.
        """
        with self._state_lock:
            self._settings = deepcopy(value) if value is not None else None
            self._settings_version += 1
            self._settings_modified = self._applied_settings_version != self._settings_version

    @property
    def camera_code(self) -> str | None:
        with self._state_lock:
            return self._camera_code

    @camera_code.setter
    def camera_code(self, value: str | None):
        with self._state_lock:
            self._camera_code = value

    def _get_state_snapshot(self) -> tuple[dict | None, str | None]:
        with self._state_lock:
            settings_snapshot = deepcopy(self._settings) if self._settings is not None else None
            camera_code = self._camera_code
        return settings_snapshot, camera_code

    def _get_pending_settings_update(self) -> tuple[dict | None, int] | None:
        with self._state_lock:
            if self._applied_settings_version == self._settings_version:
                self._settings_modified = False
                return None
            settings_snapshot = deepcopy(self._settings) if self._settings is not None else None
            return settings_snapshot, self._settings_version

    def _mark_settings_applied(self, *, settings_version: int) -> None:
        with self._state_lock:
            if settings_version > self._applied_settings_version:
                self._applied_settings_version = settings_version
            self._settings_modified = self._applied_settings_version != self._settings_version

    def _is_headless_ai_processing_required(self) -> bool:
        if self._camera_code != "scope_camera":
            return False

        master_controller = self._master_controller
        if master_controller is None:
            return False

        ai_agent = getattr(master_controller, "ai_agent", None)
        if ai_agent is None:
            return False

        is_fully_activated = getattr(ai_agent, "is_fully_activated", None)
        if not callable(is_fully_activated):
            return False

        try:
            return is_fully_activated() is True
        except Exception as err:
            # log the error
            print(f"Error checking if AI agent is fully activated: {err}")
            return False

    def create_stream_token(self) -> int:
        with self._state_lock:
            return self._stream_id

    def _invalidate_stream_tokens(self) -> None:
        with self._state_lock:
            self._stream_id += 1

    def _is_stream_token_current(self, *, stream_token: int | None) -> bool:
        if stream_token is None:
            return True
        with self._state_lock:
            return self._stream_id == stream_token

    @property
    def capabilities(self) -> dict:
        """
        Returns the capabilities of the camera.
        """
        return self._get_capabilities()

    def _ensure_device_open(self, *, stream_token: int | None = None) -> bool:
        """
        Ensure the camera device is open and thread is running.
        Updates the last access time to keep the thread alive.
        """
        if not self._is_stream_token_current(stream_token=stream_token):
            return False
        #self._open()
        with self._lock_frame:
            self._last_access_time_raw_frame = time.time()
        
        if not self.is_alive():
            try:
                # start the thread if it is not already started
                if not self._is_stream_token_current(stream_token=stream_token):
                    return False
                if not self.is_alive() and self._active is False:
                    threading.Thread.__init__(self)
                    self.daemon = True
                self.start()
                # Wait briefly for the device to open
                for _ in range(int(2 / EPSILON_DELAY)):  # Wait up to 2 seconds
                    if not self._is_stream_token_current(stream_token=stream_token):
                        return False
                    if self._active and self._camera is not None:
                        break
                    time.sleep(EPSILON_DELAY)
            except RuntimeError as err:
                # log the error
                print(f"Error starting camera thread: {err}")
                return False
        return self._is_stream_token_current(stream_token=stream_token)

    def _get_raw_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            with self._lock_frame:
                self._last_access_time_raw_frame = time.time()
                if self._frame.valid:
                    return self._frame.copy()
        return self._create_blank_frame()

    def _get_masked_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            now = time.time()
            # Frame is needed to produce correct masked_frame therefore there
            # is a change of last access time for raw frame
            with self._lock_frame:
                self._last_access_time_raw_frame = now
            with self._lock_frame_masked:
                self._last_access_time_masked_frame = now
                if self._frame_masked.valid:
                    return self._frame_masked.copy()
        return self._create_blank_frame()

    def _get_masked_ai_frame(self, *, stream_token: int | None = None) -> Frame:
        if not self._ensure_device_open(stream_token=stream_token):
            return self._create_blank_frame()
        if self._active:
            now = time.time()
            # Frame is needed to produce correct masked_frame
            # therefore there is a change of last access time for raw frame
            with self._lock_frame:
                self._last_access_time_raw_frame = now
            # Masked frame is needed to produce correct masked_frame_ai therefore there
            # is a change of last access time for masked frame
            with self._lock_frame_masked:
                self._last_access_time_masked_frame = now
            with self._lock_frame_masked_ai:
                self._last_access_time_masked_ai_frame = now
                if self._frame_masked_ai.valid:
                    return self._frame_masked_ai.copy()
        return self._create_blank_frame()

    def get_stream_frame(self, *, mode: int, stream_token: int) -> Frame | None:
        if not self._is_stream_token_current(stream_token=stream_token):
            return None

        if mode == 3:
            frame = self._get_masked_ai_frame(stream_token=stream_token)
        elif mode == 1:
            frame = self._get_masked_frame(stream_token=stream_token)
        else:
            frame = self._get_raw_frame(stream_token=stream_token)

        if not self._is_stream_token_current(stream_token=stream_token):
            return None
        return frame

    @property
    def frame(self) -> Frame:
        """
        Returns the current frame. Starts the capture thread if not running.
        Updates the last access time to keep the thread alive.
        """
        return self._get_raw_frame()

    @property
    def frame_masked(self) -> Frame:
        """
        Returns the current frame. Starts the capture thread if not running.
        Updates the last access time to keep the thread alive.
        """
        return self._get_masked_frame()

    @property
    def frame_masked_ai(self) -> Frame:
        """
        Returns the current frame. Starts the capture thread if not running.
        Updates the last access time to keep the thread alive.
        """
        return self._get_masked_ai_frame()

    def _crop_and_resize(self, *, image: np.ndarray, settings: dict | None) -> np.ndarray:
        """
        Crop and resize a frame.
        Parameters:
        src_frame : numpy.ndarray : The source frame.
        Returns:
        bool : True if the frame was cropped and resized successfully, False otherwise.
        np.ndarray : The cropped and resized frame.
        """
        settings = settings or {}
        # convert crop coordinates to pixels
        # crop coordinates are between 0 and 1
        crop_top = settings["crop_top"] if "crop_top" in settings else 0.0
        crop_left = settings["crop_left"] if "crop_left" in settings else 0.0
        crop_bottom = settings["crop_bottom"] if "crop_bottom" in settings else 0.0
        crop_right = settings["crop_right"] if "crop_right" in settings else 0.0
        crop_top_px = int(round(crop_top * image.shape[0]))
        crop_left_px = int(round(crop_left * image.shape[1]))
        crop_bottom_px = int(round(crop_bottom * image.shape[0]))
        crop_right_px = int(round(crop_right * image.shape[1]))
        # calculate source frame coordinates in pixels
        src_x = crop_left_px
        src_y = crop_top_px
        src_width = image.shape[1] - crop_right_px - crop_left_px
        src_height = image.shape[0] - crop_bottom_px - crop_top_px
        # get stretch properties
        stretch_enabled = settings["stretch_enabled"] if "stretch_enabled" in settings else False
        stretch_width = settings["stretch_width"] if "stretch_width" in settings else 0
        stretch_height = settings["stretch_height"] if "stretch_height" in settings else 0
        if stretch_enabled:
            target_width = stretch_width
            target_height = stretch_height
        else:
            target_width = src_width
            target_height = src_height
        # check if source frame coordinates are valid
        is_valid = True
        if src_x < 0 or src_x > image.shape[1]:
            is_valid = False
        if src_y < 0 or src_y > image.shape[0]:
            is_valid = False
        if src_width <= 0:
            is_valid = False
        if src_height <= 0:
            is_valid = False
        if src_x + src_width > image.shape[1]:
            is_valid = False
        if src_y + src_height > image.shape[0]:
            is_valid = False
        # if source frame coordinates are not valid, return copy of the source frame
        if not is_valid:
            return image
        # if source frame coordinates are valid, return cropped and resized frame
        return cv2.resize(image[src_y:src_y + src_height, src_x:src_x + src_width], (target_width, target_height))

    def _mask_image(self, *, image: np.ndarray, settings: dict | None) -> np.ndarray:
        """
        Takes an image and a list of normalized polygon coordinates.
        Returns the image with everything outside the polygons blackened out.
        """
        settings = settings or {}
        # 1. Base case: If no polygons, return original image
        mask_polygons = settings["mask_polygons"] if "mask_polygons" in settings else []
        if not mask_polygons:
            return image
        # 2. Get image dimensions
        height, width = image.shape[:2]
        image_aspect_ratio = width / height
        width_of_image_where_polygons_were_drawn = 640
        height_of_image_where_polygons_were_drawn = 480
        polygons_aspect_ratio = width_of_image_where_polygons_were_drawn / height_of_image_where_polygons_were_drawn        
        # 3. Create a black single-channel mask (same height/width, unsigned 8-bit integer)
        mask = np.zeros((height, width), dtype=np.uint8)
        # 4. Process each polygon
        for polygon in mask_polygons:
            points = []
            for point in polygon:
                # Convert normalized coordinates (0 -> 1) to pixel coordinates
                if image_aspect_ratio > polygons_aspect_ratio:
                    x_pixel = int(point["x"] * width)
                    y_pixel = point["y"] - 0.5
                    y_pixel = y_pixel / polygons_aspect_ratio * image_aspect_ratio
                    y_pixel += 0.5
                    y_pixel = int(y_pixel * height)
                else:
                    x_pixel = point["x"] - 0.5
                    x_pixel = x_pixel * polygons_aspect_ratio / image_aspect_ratio
                    x_pixel += 0.5
                    x_pixel = int(x_pixel * width)
                    y_pixel = int(point["y"] * height)
                points.append([x_pixel, y_pixel])            
            # Convert points to a numpy array of shape (N, 1, 2) required by cv2.fillPoly
            pts_array = np.array(points, dtype=np.int32)
            pts_array = pts_array.reshape((-1, 1, 2))            
            # Fill the polygon on the mask with White (255)
            cv2.fillPoly(mask, [pts_array], 255)
        # 5. Apply the mask to the image
        # cv2.bitwise_and keeps pixels where mask is non-zero (255) and blackens the rest
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    def _mask_ai_image(self, *, image: np.ndarray, settings: dict) -> np.ndarray:
        """Process a frame through the AI pipeline."""
        # get the model name from the settings
        ai_setup = settings.get("aiSetup", {})
        model_name = ai_setup.get("modelName", YOLOModels().default_model_name)
        device = ai_setup.get("device", DEFAULT_DEVICE)

        # Run YOLO inference
        try:
            results = YOLOModels().predict(
                model_name=model_name,
                preferred_device=device,
                image=image,
                verbose=False
            )
            result = results[0]

            # Check for keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # Get keypoints with confidence if possible (N, 17, 3)
                if hasattr(result.keypoints, 'data'):
                    kpts = result.keypoints.data
                else:
                    kpts = result.keypoints
                # Get pose dictionary
                keypoints = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
                # get the AI setup from the settings
                ai_setup = settings.get("aiSetup", {})
                raw_pose = get_pose_dict(keypoints=keypoints, ai_setup=ai_setup)
                pose = translate_raw_pose_to_pose_dict(raw_pose=raw_pose, ai_setup=ai_setup)
                image = draw_pose(image=image, pose=pose, ai_setup=ai_setup)
                # Return frame with pose
                return Frame(valid=True, image=image, time=time.time(), pose=pose)

        except Exception as e:
            print(f"Error during AI processing: {e}")

        # If no keypoints detected or error, return masked frame
        return Frame(valid=True, image=image, time=time.time(), pose=None)

    def _get_frame(self):
        """Capture a frame from the camera and apply transformations."""
        camera_settings, camera_code = self._get_state_snapshot()
        now = time.time()
        headless_ai_processing_required = self._is_headless_ai_processing_required()
        try:
            with self._lock:
                if not self._active or self._camera is None:
                    return
                valid, image = self._get_image_ndarray()
        except Exception as e:
            print(f"Warning: Camera index={self._index}, camera_index={self._camera_index} "
                  f"and camera_type={self._camera_type.value}, error getting frame: {e}")
            valid = False
            image = None
        if not valid:
            with self._lock:
                if self._camera is not None:
                    self._close()
            with self._lock_frame:
                self._frame = self._create_blank_frame()
            return
        # Apply flip transformations
        camera_settings = camera_settings or {}
        flip_horizontal = camera_settings["flip_horizontal"] if "flip_horizontal" in camera_settings else False
        flip_vertical = camera_settings["flip_vertical"] if "flip_vertical" in camera_settings else False
        rotate = camera_settings["rotate"] if "rotate" in camera_settings else 0
        if flip_horizontal and flip_vertical:
            image = cv2.flip(image, FLIP_BOTH)
        elif flip_horizontal:
            image = cv2.flip(image, FLIP_HORIZONTAL)
        elif flip_vertical:
            image = cv2.flip(image, FLIP_VERTICAL)
        # Apply rotation
        if rotate == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotate == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # crop and resize the image
        image = self._crop_and_resize(image=image, settings=camera_settings)
        with self._lock_frame:
            self._frame = Frame(valid=True, image=image, time=time.time())
        # mask the image
        with self._lock_frame_masked:
            elapsed_masked = now - self._last_access_time_masked_frame
            #elapsed_masked_ai = now - self._last_access_time_masked_ai_frame
            # if the masked frame has timed out, return
            if elapsed_masked > self.TIMEOUT_SECONDS and not headless_ai_processing_required:
                return
            image_masked = self._mask_image(image=image, settings=camera_settings)
            self._frame_masked = Frame(valid=True, image=image_masked, time=time.time())
        # apply the AI model to the image
        with self._lock_frame_masked_ai:
            elapsed_masked_ai = now - self._last_access_time_masked_ai_frame
            # if the AI mask has timed out, return
            if elapsed_masked_ai > self.TIMEOUT_SECONDS and not headless_ai_processing_required:
                return
            # get the settings
            settings = get_settings_sync()
            # apply the AI model to the image
            frame_masked_ai = self._mask_ai_image(image=image_masked, settings=settings)
            self._frame_masked_ai = Frame(valid=True, image=frame_masked_ai.image, time=time.time(), pose=frame_masked_ai.pose)
            # check if the AI agent can engage
            # this is only for the scope camera
            if camera_code is not None and camera_code == "scope_camera":
                ai_agent: AIAgent = self._master_controller.ai_agent
                engagement_result = ai_agent.engage(frame=self._frame_masked_ai, settings=settings)
                ai_agent.draw_engagement_result(frame=self._frame_masked_ai, engagement_result=engagement_result)

    def _should_capture_next_frame(self) -> bool:
        with self._lock:
            if not self._active or self._camera is None:
                return False
        if self._is_headless_ai_processing_required():
            return True
        with self._lock_frame:
            elapsed = time.time() - self._last_access_time_raw_frame
        return elapsed <= self.TIMEOUT_SECONDS

    def run(self):
        """Thread main loop: captures frames until timeout."""
        if not self._open():
            return
        try:
            # set the last access time
            with self._lock_frame:
                self._last_access_time_raw_frame = time.time()
            # main loop
            while self._active:
                # if settings were modified, set the properties
                pending_settings = self._get_pending_settings_update()
                if pending_settings is not None:
                    settings_snapshot, settings_version = pending_settings
                    self._set_camera_properties(settings_snapshot)
                    self._mark_settings_applied(settings_version=settings_version)
                if not self._should_capture_next_frame():
                    break
                # Device access stays serialized inside _get_frame, but slow
                # post-processing runs outside the main lifecycle lock.
                self._get_frame()
                time.sleep(EPSILON_DELAY)  # Small delay to prevent CPU overload
        except Exception as e:
            print(f"Error in camera thread: {e}")
        finally:
            # close the camera
            self._close()

    def stop(self):
        """Stop the capture thread."""
        self._invalidate_stream_tokens()
        with self._lock:
            self._active = False
        if threading.current_thread() is self:
            return
        if not self.is_alive():
            return

        self.join(timeout=self.STOP_TIMEOUT_SECONDS)
        if self.is_alive():
            message = (
                f"Camera worker still alive after {self.STOP_TIMEOUT_SECONDS:.1f}s stop timeout "
                f"(index={self._index}, camera_index={self._camera_index}, "
                f"camera_code={self._camera_code}, camera_name={self._camera_name})"
            )
            print(message)
            raise RuntimeError(message)

    def reset_settings(self):
        self.settings = deepcopy(self._default_settings)
