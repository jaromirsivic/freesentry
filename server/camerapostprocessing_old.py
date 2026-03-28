import time
import cv2
import numpy as np
from .common import epsilon, Frame
from .yolomodels import YOLOModels
from ultralytics import YOLO
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .camera import Camera


class NodeImage(threading.Thread):
    """
    Node that provides direct access to parent Camera frame.
    No thread processing needed - frame is returned directly from parent.
    """

    def __init__(self, *, parent: "Camera", settings: dict | None = None):
        super().__init__()
        self.daemon = True
        self._parent = parent
        self._settings = settings if settings is not None else {}

    @property
    def settings(self) -> dict:
        """Get the settings dictionary."""
        return self._settings

    @settings.setter
    def settings(self, value: dict):
        """Set the settings dictionary."""
        self._settings = value

    @property
    def frame(self) -> Frame:
        """Returns the frame directly from parent Camera."""
        return self._parent.frame


class NodeImageCroppedResized(NodeImage):
    """
    Node that crops and resizes frames from parent Camera.
    
    If crop/resize values are all <= epsilon, returns parent frame directly.
    Otherwise, starts a thread to process frames in background.
    Thread stops after 5 seconds of inactivity.
    """

    TIMEOUT_SECONDS = 5

    def __init__(self, *, parent: "Camera", settings: dict | None = None):
        super().__init__(parent=parent, settings=settings)
        # self.daemon = True 
        # self._parent = parent
        # self._settings = settings if settings is not None else self._default_settings()
        self._frame: Frame = Frame(valid=False, image=None, time=0)
        self._last_access_time: float = 0
        self._last_processed_time: float = 0
        self._lock = threading.Lock()
        self._active = False

    def _default_settings(self) -> dict:
        """Return default settings."""
        return {
            "crop_top": 0,
            "crop_left": 0,
            "crop_bottom": 0,
            "crop_right": 0,
            "width": 0,
            "height": 0,
            "static_reticle_x": 0.5,
            "static_reticle_y": 0.5,
            "static_reticle_color": "#ff0000cc",
            "static_reticle_size": 1
        }

    # @property
    # def settings(self) -> dict:
    #     """Get the settings dictionary."""
    #     return self._settings

    # @settings.setter
    # def settings(self, value: dict):
    #     """Set the settings dictionary."""
    #     self._settings = value

    # Crop properties (getters only)

    @property
    def crop_top(self) -> float:
        """Get the top crop value (0-1)."""
        return self._settings.get("crop_top", 0)

    @property
    def crop_left(self) -> float:
        """Get the left crop value (0-1)."""
        return self._settings.get("crop_left", 0)

    @property
    def crop_bottom(self) -> float:
        """Get the bottom crop value (0-1)."""
        return self._settings.get("crop_bottom", 0)

    @property
    def crop_right(self) -> float:
        """Get the right crop value (0-1)."""
        return self._settings.get("crop_right", 0)

    @property
    def width(self) -> int:
        """Get the target width."""
        return self._settings.get("width", 0)

    @property
    def height(self) -> int:
        """Get the target height."""
        return self._settings.get("height", 0)

    # Static reticle properties (getters only)

    @property
    def static_reticle_x(self) -> float:
        """Get the static reticle X position (0-1)."""
        return self._settings.get("static_reticle_x", 0.5)

    @property
    def static_reticle_y(self) -> float:
        """Get the static reticle Y position (0-1)."""
        return self._settings.get("static_reticle_y", 0.5)

    @property
    def static_reticle_color(self) -> str:
        """Get the static reticle color."""
        return self._settings.get("static_reticle_color", "#ff0000cc")

    @property
    def static_reticle_size(self) -> float:
        """Get the static reticle size."""
        return self._settings.get("static_reticle_size", 1)

    def _needs_processing(self) -> bool:
        """Check if crop/resize processing is needed."""
        return (
            self.crop_top > epsilon or
            self.crop_left > epsilon or
            self.crop_bottom > epsilon or
            self.crop_right > epsilon or
            self.width > epsilon or
            self.height > epsilon
        )

    def _ensure_thread_running(self):
        """Ensure the processing thread is running."""
        with self._lock:
            self._last_access_time = time.time()

        if not self.is_alive():
            try:
                if not self.is_alive() and self._active is False:
                    threading.Thread.__init__(self)
                    self.daemon = True
                self._active = True
                self.start()
            except RuntimeError:
                pass

    @property
    def frame(self) -> Frame:
        """
        Returns the processed frame.
        If no processing needed, returns parent frame directly.
        Otherwise, starts thread and returns processed frame.
        """
        with self._lock:
            self._last_access_time = time.time()

        # If no processing needed, return parent frame directly
        if not self._needs_processing():
            return self._parent.frame

        # Processing is needed - ensure thread is running
        self._ensure_thread_running()

        # Return current frame (may be from previous processing or empty)
        if self._frame.valid:
            return self._frame
        return self._parent.frame

    def _crop_and_resize(self, *, src_frame: np.ndarray) -> Frame:
        """Crop and resize a frame."""
        # Convert crop coordinates to pixels (crop values are 0-1)
        crop_top_px = int(round(self.crop_top * src_frame.shape[0]))
        crop_left_px = int(round(self.crop_left * src_frame.shape[1]))
        crop_bottom_px = int(round(self.crop_bottom * src_frame.shape[0]))
        crop_right_px = int(round(self.crop_right * src_frame.shape[1]))

        # Calculate source frame coordinates in pixels
        src_x = crop_left_px
        src_y = crop_top_px
        src_width = src_frame.shape[1] - crop_right_px - crop_left_px
        src_height = src_frame.shape[0] - crop_bottom_px - crop_top_px

        # Validate coordinates
        is_valid = (
            src_x >= 0 and src_x <= src_frame.shape[1] and
            src_y >= 0 and src_y <= src_frame.shape[0] and
            src_width > 0 and src_height > 0 and
            src_x + src_width <= src_frame.shape[1] and
            src_y + src_height <= src_frame.shape[0]
        )

        if not is_valid:
            return Frame(valid=True, image=src_frame.copy(), time=time.time())

        # Crop the frame
        cropped = src_frame[src_y:src_y + src_height, src_x:src_x + src_width]

        # Resize if width and height are specified
        target_width = self.width if self.width > 0 else cropped.shape[1]
        target_height = self.height if self.height > 0 else cropped.shape[0]

        if target_width != cropped.shape[1] or target_height != cropped.shape[0]:
            result = cv2.resize(cropped, (target_width, target_height))
        else:
            result = cropped

        return Frame(valid=True, image=result, time=time.time())

    def run(self):
        """Thread main loop: processes frames until timeout."""
        with self._lock:
            self._last_access_time = time.time()

        while self._active:
            # Check timeout
            with self._lock:
                elapsed = time.time() - self._last_access_time

            if elapsed > self.TIMEOUT_SECONDS:
                break

            # Get parent frame
            parent_frame = self._parent.frame

            # Only process if parent frame time differs from last processed
            if parent_frame.valid and parent_frame.time != self._last_processed_time:
                self._frame = self._crop_and_resize(src_frame=parent_frame.image)
                self._last_processed_time = parent_frame.time

            time.sleep(0.001)  # Small delay to prevent CPU overload

        self._active = False

    def stop(self):
        """Stop the processing thread."""
        self._active = False
        if self.is_alive():
            self.join(timeout=1.0)

    def reset_settings(self):
        self.settings = {
            "crop_top": 0,
            "crop_left": 0,
            "crop_bottom": 0,
            "crop_right": 0,
            "width": 0,
            "height": 0,
            "static_reticle_x": 0.5,
            "static_reticle_y": 0.5,
            "static_reticle_color": "#ff0000cc",
            "static_reticle_size": 1
        }

    def reload_settings(self):
        return

    def to_dict(self) -> dict:
        return self._settings


class NodeImageAI(NodeImageCroppedResized):
    """
    AI processing node that detects pose keypoints using YOLO model.
    
    Thread starts when user accesses "frame" property.
    Thread stops after 5 seconds of inactivity.
    """

    TIMEOUT_SECONDS = 5

    def __init__(self, *, parent: "Camera", settings: dict | None = None):
        super().__init__(parent=parent, settings=settings)
        # self.daemon = True
        # self._parent = parent
        # self._settings = settings if settings is not None else self._default_settings()
        # self._frame: Frame = Frame(valid=False, image=None, time=0)
        # self._last_access_time: float = 0
        # self._last_processed_time: float = 0
        # self._lock = threading.Lock()
        # self._active = False
        self._model: YOLO = None

    def _default_settings(self) -> dict:
        """Return default settings."""
        return {
            "model_name": YOLOModels().default_model_name,
            "mask_polygons": []
        }

    @property
    def settings(self) -> dict:
        """Get the settings dictionary."""
        return self._settings

    @settings.setter
    def settings(self, value: dict):
        """Set the settings dictionary and reload model if needed."""
        old_model_name = self._settings.get("model_name")
        self._settings = value
        new_model_name = self._settings.get("model_name")
        # Reload model if name changed
        if old_model_name != new_model_name:
            self._model = None

    @property
    def model_name(self) -> str:
        """Get the YOLO model name."""
        return self._settings.get("model_name", YOLOModels().default_model_name)

    @property
    def model(self) -> YOLO:
        """Get the YOLO model instance."""
        if self._model is None:
            model_name = self.model_name
            if model_name not in YOLOModels().models:
                model_name = YOLOModels().default_model_name
            self._model = YOLOModels().models[model_name]
        return self._model

    @property
    def mask_polygons(self) -> list:
        """Get the mask polygons list."""
        return self._settings.get("mask_polygons", [])

    def _ensure_thread_running(self):
        """Ensure the processing thread is running."""
        with self._lock:
            self._last_access_time = time.time()

        if not self.is_alive():
            try:
                if not self.is_alive() and self._active is False:
                    threading.Thread.__init__(self)
                    self.daemon = True
                self._active = True
                self.start()
            except RuntimeError:
                pass

    @property
    def frame(self) -> Frame:
        """
        Returns the AI-processed frame.
        Starts thread if not running.
        """
        with self._lock:
            self._last_access_time = time.time()

        # Always ensure thread is running for AI processing
        self._ensure_thread_running()

        # Return current frame (may be from previous processing or parent)
        if self._frame.valid:
            return self._frame
        return self._parent.frame

    def _apply_mask(self, *, src_frame: np.ndarray) -> np.ndarray:
        """
        Apply mask polygons to the frame.
        Areas outside polygons are filled with black.
        
        Polygons are in normalized coordinates (0.0-1.0):
        - x=0.0, y=0.0 is top-left corner
        - x=1.0, y=1.0 is bottom-right corner
        
        Format: [[ {"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, ... ], ...]
        
        Returns the masked frame.
        """
        if not self.mask_polygons or len(self.mask_polygons) == 0:
            return src_frame

        height, width = src_frame.shape[:2]

        # Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Fill polygon areas with white (255) - these areas will be visible
        for polygon in self.mask_polygons:
            if polygon and len(polygon) >= 3:
                # Convert normalized coordinates to pixel coordinates
                pts = []
                for point in polygon:
                    # Handle dict format {"x": float, "y": float}
                    if isinstance(point, dict):
                        x = int(point.get("x", 0) * width)
                        y = int(point.get("y", 0) * height)
                    # Handle object with x, y attributes (Vector2D)
                    elif hasattr(point, "x") and hasattr(point, "y"):
                        x = int(point.x * width)
                        y = int(point.y * height)
                    else:
                        continue
                    pts.append([x, y])
                
                if len(pts) >= 3:
                    pts_array = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask, [pts_array], 255)

        # Apply mask: keep only polygon areas, rest is black
        masked_frame = cv2.bitwise_and(src_frame, src_frame, mask=mask)
        return masked_frame

    def _draw_keypoints(self, *, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw keypoints (pose landmarks) on a transparent label layer.
        Returns the label layer with keypoints drawn.
        """
        # Create transparent label layer (BGRA)
        label_layer = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

        green = (0, 255, 0, 255)
        red = (0, 0, 255, 255)
        blue = (255, 0, 0, 255)
        line_color = (0, 255, 0, 255)

        for person in keypoints:
            # COCO keypoints: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear
            # 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow
            # 9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip
            nose = tuple(map(int, person[0])) if not np.any(np.isnan(person[0])) else None
            left_eye = tuple(map(int, person[1])) if not np.any(np.isnan(person[1])) else None
            right_eye = tuple(map(int, person[2])) if not np.any(np.isnan(person[2])) else None
            left_ear = tuple(map(int, person[3])) if person.shape[0] > 3 and not np.any(np.isnan(person[3])) else None
            right_ear = tuple(map(int, person[4])) if person.shape[0] > 4 and not np.any(np.isnan(person[4])) else None
            left_shoulder = tuple(map(int, person[5])) if person.shape[0] > 5 and not np.any(np.isnan(person[5])) else None
            right_shoulder = tuple(map(int, person[6])) if person.shape[0] > 6 and not np.any(np.isnan(person[6])) else None
            left_hip = tuple(map(int, person[11])) if person.shape[0] > 11 and not np.any(np.isnan(person[11])) else None
            right_hip = tuple(map(int, person[12])) if person.shape[0] > 12 and not np.any(np.isnan(person[12])) else None

            # Draw points
            if nose:
                cv2.circle(label_layer, nose, 5, green, -1)
            if left_eye:
                cv2.circle(label_layer, left_eye, 5, green, -1)
            if right_eye:
                cv2.circle(label_layer, right_eye, 5, green, -1)
            if left_ear:
                cv2.circle(label_layer, left_ear, 5, green, -1)
            if right_ear:
                cv2.circle(label_layer, right_ear, 5, green, -1)
            if left_shoulder:
                cv2.circle(label_layer, left_shoulder, 5, green, -1)
            if right_shoulder:
                cv2.circle(label_layer, right_shoulder, 5, green, -1)
            if left_hip:
                cv2.circle(label_layer, left_hip, 5, green, -1)
            if right_hip:
                cv2.circle(label_layer, right_hip, 5, green, -1)

            # Draw connecting lines
            if left_eye and nose:
                cv2.line(label_layer, left_eye, nose, line_color, 2)
            if right_eye and nose:
                cv2.line(label_layer, right_eye, nose, line_color, 2)
            if left_shoulder and right_shoulder:
                cv2.line(label_layer, left_shoulder, right_shoulder, line_color, 2)
            if left_hip and right_hip:
                cv2.line(label_layer, left_hip, right_hip, line_color, 2)

            # Compute center points
            g1 = None  # Center between shoulders
            g2 = None  # Center between hips
            if left_shoulder and right_shoulder:
                g1 = (int((left_shoulder[0] + right_shoulder[0]) / 2),
                      int((left_shoulder[1] + right_shoulder[1]) / 2))
            if left_hip and right_hip:
                g2 = (int((left_hip[0] + right_hip[0]) / 2),
                      int((left_hip[1] + right_hip[1]) / 2))

            # Alpha point: 1/4 from g1 to g2
            if g1 and g2:
                alpha = (int(g1[0] + 0.25 * (g2[0] - g1[0])),
                         int(g1[1] + 0.25 * (g2[1] - g1[1])))
                cv2.circle(label_layer, alpha, 8, red, -1)

            if g1:
                cv2.circle(label_layer, g1, 7, blue, -1)
            if g2:
                cv2.circle(label_layer, g2, 7, blue, -1)

            # Beta point: center between ears projection
            p1 = None  # Center between ears
            p2 = None  # Projection behind head
            if left_ear and right_ear:
                p1 = (int((left_ear[0] + right_ear[0]) / 2),
                      int((left_ear[1] + right_ear[1]) / 2))
            if left_eye and right_eye and nose:
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                dx = nose[0] - eye_center_x
                dy = nose[1] - eye_center_y
                p2 = (int(eye_center_x - dx), int(eye_center_y - dy))

            if p1 and p2:
                beta = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                cv2.circle(label_layer, beta, 12, red, -1)

            if p1:
                cv2.circle(label_layer, p1, 6, blue, -1)
            if p2:
                cv2.circle(label_layer, p2, 6, blue, -1)

        return label_layer

    def _blend_layers(self, *, base_frame: np.ndarray, label_layer: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend label layer on top of base frame with specified alpha.
        Label layer is BGRA, base_frame is BGR.
        """
        # Create output frame
        result = base_frame.copy()

        # Get the alpha channel from label layer
        label_alpha = label_layer[:, :, 3] / 255.0

        # Where label has content (alpha > 0), blend with base
        for c in range(3):
            result[:, :, c] = (
                (1 - label_alpha * alpha) * base_frame[:, :, c] +
                label_alpha * alpha * label_layer[:, :, c]
            ).astype(np.uint8)

        return result

    def _process_frame(self, *, src_frame: np.ndarray) -> Frame:
        """Process a frame through the AI pipeline."""
        # Apply mask if polygons are defined
        masked_frame = self._apply_mask(src_frame=src_frame)

        # Run YOLO inference
        try:
            results = self.model(masked_frame, verbose=False)
            result = results[0]

            # Check for keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else result.keypoints.xy

                # Draw keypoints on label layer
                label_layer = self._draw_keypoints(frame=masked_frame, keypoints=keypoints)

                # Blend label layer on top of masked frame at 50% transparency
                final_frame = self._blend_layers(base_frame=masked_frame, label_layer=label_layer, alpha=0.5)

                return Frame(valid=True, image=final_frame, time=time.time())

        except Exception as e:
            print(f"Error during AI processing: {e}")

        # If no keypoints detected or error, return masked frame
        return Frame(valid=True, image=masked_frame, time=time.time())

    def run(self):
        """Thread main loop: processes frames until timeout."""
        with self._lock:
            self._last_access_time = time.time()

        while self._active:
            # Check timeout
            with self._lock:
                elapsed = time.time() - self._last_access_time

            if elapsed > self.TIMEOUT_SECONDS:
                break

            # Get parent frame
            parent_frame = self._parent.frame

            # Only process if parent frame time differs from last processed
            if parent_frame.valid and parent_frame.time != self._last_processed_time:
                self._frame = self._process_frame(src_frame=parent_frame.image)
                self._last_processed_time = parent_frame.time

            time.sleep(0.001)  # Small delay to prevent CPU overload

        self._active = False

    def stop(self):
        """Stop the processing thread."""
        self._active = False
        if self.is_alive():
            self.join(timeout=1.0)

    def reload_settings(self):
        if self._settings is None:
            return
        # load model
        model_name = self._settings["model_name"] if "model_name" in self._settings else YOLOModels().default_model_name
        if model_name not in YOLOModels().models:
            model_name = YOLOModels().default_model_name
        self._model_name = model_name
        self._model = YOLOModels().models[model_name]
        # load mask polygons
        self._mask_polygons = self._settings["mask_polygons"] if "mask_polygons" in self._settings else []
    