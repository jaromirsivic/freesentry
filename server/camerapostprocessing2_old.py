import time
import cv2
import numpy as np
from .common import epsilon, Frame
from .yolomodels import YOLOModels
from ultralytics import YOLO


class NodeImage2:
    def __init__(self, *, parent, fps: int | None):
        self._parent: NodeImage = parent
        self._fps = 1000
        self._frame_interval = 1 / self._fps
        self._last_frame_time = 0
        self._frame_count = 0
        self._start_time = time.time()
        # resets all values
        self.fps = fps if fps is not None else 60
        self._frame: Frame = Frame(valid=False, frame=None, frame_time=0)

    @property
    def frame(self) -> Frame:
        self._get_frame()
        return self._frame

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, value: int):
        if value < 0.001:
            value = 1
        self._fps = value        
        self._frame_interval = 1 / value
        self._last_frame_time = 0

    @property
    def frame_interval(self) -> float:
        return self._frame_interval

    @property
    def last_frame_time(self) -> float:
        return self._last_frame_time

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def start_time(self) -> float:
        return self._start_time

    def _get_frame(self):
        # now = time.time()
        # if last frame was valid and not enough time has passed since last frame
        # return False, None
        # if self._frame_valid and now - self._last_frame_time < self._frame_interval:
        #     return True, self._frame
        # otherwise, get frame from parent
        # self._last_frame_time = now
        self._frame_count += 1
        self._frame = self._parent.frame


class NodeImageCroppedResized2(NodeImage):
    def __init__(self, *, parent, fps: int | None, settings: dict | None = None):
        super().__init__(parent=parent, fps=fps)
        if settings is None:
            settings = self.reset_settings()
        else:
            self._settings = settings
        #self.reload_settings()

    @property
    def settings(self) -> dict:
        return self._settings
    
    @settings.setter
    def settings(self, value: dict):
        self._settings = value
        self.reload_settings()

    @property
    def crop_top(self) -> float:
        return self._settings["crop_top"] if "crop_top" in self._settings else 0
    
    @property
    def crop_left(self) -> float:
        return self._settings["crop_left"] if "crop_left" in self._settings else 0
    
    @property
    def crop_bottom(self) -> float:
        return self._settings["crop_bottom"] if "crop_bottom" in self._settings else 0  
    
    @property
    def crop_right(self) -> float:
        return self._settings["crop_right"] if "crop_right" in self._settings else 0

    @property
    def width(self) -> int:
        return self._settings["width"] if "width" in self._settings else 0

    @property
    def height(self) -> int:
        return self._settings["height"] if "height" in self._settings else 0

    @property
    def static_reticle_x(self) -> float:
        return self._settings["static_reticle_x"] if "static_reticle_x" in self._settings else 0.5
    
    @property
    def static_reticle_y(self) -> float:
        return self._settings["static_reticle_y"] if "static_reticle_y" in self._settings else 0.5
    
    @property
    def static_reticle_color(self) -> str:
        return self._settings["static_reticle_color"] if "static_reticle_color" in self._settings else "#ff0000cc"
    
    @property
    def static_reticle_size(self) -> float:
        return self._settings["static_reticle_size"] if "static_reticle_size" in self._settings else 1

    def to_dict(self) -> dict:
        return self._settings
        # return {
        #     "crop_top": self._crop_top,
        #     "crop_left": self._crop_left,
        #     "crop_bottom": self._crop_bottom,
        #     "crop_right": self._crop_right,
        #     "width": self._width,
        #     "height": self._height,
        #     "static_reticle_x": self._static_reticle_x,
        #     "static_reticle_y": self._static_reticle_y,
        #     "static_reticle_color": self._static_reticle_color,
        #     "static_reticle_size": self._static_reticle_size
        # }

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

    def _get_frame(self):
        now = time.time()
        # if last frame was valid and not enough time has passed since last frame
        if self._frame.valid and now - self.frame.frame_time < self._frame_interval:
            return
        # otherwise, get frame from parent
        self._frame_count += 1
        # if crop and resize are enabled, crop and resize the frame
        if self._parent.frame.valid:
            # if crop and resize are not enabled, return the frame as is
            if (self.crop_top == 0 and self.crop_left == 0 and \
                self.crop_bottom == 0 and self.crop_right == 0) and \
                (self.width == 0 or self.width == self._frame.shape[1]) and \
                (self.height == 0 or self.height == self._frame.shape[0]):
                self._frame = self._parent.frame
            else:
                self._frame = self.crop_and_resize_frame(src_frame=self._parent.frame.image)
                self._frame.valid = True
                self._frame.frame_time = now
        return

    def crop_and_resize_frame(self, *, src_frame: np.ndarray) -> np.ndarray:
        """
        Crop and resize a frame.
        Parameters:
        src_frame : numpy.ndarray : The source frame.
        Returns:
        bool : True if the frame was cropped and resized successfully, False otherwise.
        np.ndarray : The cropped and resized frame.
        """
        # convert crop coordinates to pixels
        # crop coordinates are between 0 and 1
        crop_top_px = int(round(self.crop_top * src_frame.shape[0]))
        crop_left_px = int(round(self.crop_left * src_frame.shape[1]))
        crop_bottom_px = int(round(self.crop_bottom * src_frame.shape[0]))
        crop_right_px = int(round(self.crop_right * src_frame.shape[1]))
        # calculate source frame coordinates in pixels
        src_x = crop_left_px
        src_y = crop_top_px
        src_width = src_frame.shape[1] - crop_right_px - crop_left_px
        src_height = src_frame.shape[0] - crop_bottom_px - crop_top_px
        # check if source frame coordinates are valid
        is_valid = True
        if src_x < 0 or src_x > src_frame.shape[1]:
            is_valid = False
        if src_y < 0 or src_y > src_frame.shape[0]:
            is_valid = False
        if src_width <= 0:
            is_valid = False
        if src_height <= 0:
            is_valid = False
        if src_x + src_width > src_frame.shape[1]:
            is_valid = False
        if src_y + src_height > src_frame.shape[0]:
            is_valid = False
        # if source frame coordinates are not valid, return copy of the source frame
        if not is_valid:
            return src_frame.copy()
        # if source frame coordinates are valid, return cropped and resized frame
        return cv2.resize(src_frame[src_y:src_y + src_height, src_x:src_x + src_width], (self.width, self.height))

    # def draw_static_reticle(self, *, frame: np.ndarray) -> np.ndarray:
    #     # draw a cross created by 4 lines with gaps and a dot in the center
    #     # gap size is 5 percent of the bigger dimension of the frame multiplied by static_reticle_size
    #     gap_size = max(frame.shape[0], frame.shape[1]) * 0.05 * self.static_reticle_size
    #     # line length is 14 percent of the bigger dimension of the frame multiplied by static_reticle_size
    #     line_length = max(frame.shape[0], frame.shape[1]) * 0.14 * self.static_reticle_size
    #     # dot radius is 2 percent of the bigger dimension of the frame multiplied by static_reticle_size
    #     dot_radius = max(frame.shape[0], frame.shape[1]) * 0.02 * self.static_reticle_size
    #     # line width is 2 percent of the bigger dimension of the frame multiplied by static_reticle_size
    #     line_width = max(frame.shape[0], frame.shape[1]) * 0.02 * self.static_reticle_size
    #     # draw the cross
    #     cv2.line(frame, (int(self.static_reticle_x * frame.shape[1]), int(self.static_reticle_y * frame.shape[0]) - gap_size), (int(self.static_reticle_x * frame.shape[1]), int(self.static_reticle_y * frame.shape[0]) - gap_size - line_length), self.static_reticle_color, line_width)
    #     cv2.line(frame, (int(self.static_reticle_x * frame.shape[1]), int(self.static_reticle_y * frame.shape[0]) + gap_size), (int(self.static_reticle_x * frame.shape[1]), int(self.static_reticle_y * frame.shape[0]) + gap_size + line_length), self.static_reticle_color, line_width)
    #     cv2.line(frame, (int(self.static_reticle_x * frame.shape[1]) - gap_size, int(self.static_reticle_y * frame.shape[0])), (int(self.static_reticle_x * frame.shape[1]) - gap_size - line_length, int(self.static_reticle_y * frame.shape[0])), self.static_reticle_color, line_width)
    #     cv2.line(frame, (int(self.static_reticle_x * frame.shape[1]) + gap_size, int(self.static_reticle_y * frame.shape[0])), (int(self.static_reticle_x * frame.shape[1]) + gap_size + line_length, int(self.static_reticle_y * frame.shape[0])), self.static_reticle_color, line_width)
    #     # draw the dot
    #     cv2.circle(frame, (int(self.static_reticle_x * frame.shape[1]), int(self.static_reticle_y * frame.shape[0])), dot_radius, self.static_reticle_color, -1)
    #     return frame


class NodeImageAI2(NodeImageCroppedResized):
    def __init__(self, *, parent, fps: int | None, settings: dict | None = None):
        super().__init__(parent=parent, fps=fps, settings=settings)
        self._model_name = YOLOModels().default_model_name
        self._model = None
        self._mask_polygons = None
        self.reload_settings()

    @property
    def model_name(self) -> str:
        return self._settings["model_name"] if "model_name" in self._settings else YOLOModels().default_model_name

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLOModels().models[self.model_name]
        return self._model

    @property
    def mask_polygons(self) -> list[list[tuple[int, int]]]:
        return self._settings["mask_polygons"] if "mask_polygons" in self._settings else []

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

    def _get_frame(self):
        super()._get_frame()
        # run inference on the frame
        results = self._model(self._frame.image, verbose=False)
        # compute points of interest
        result = results[0]
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else result.keypoints.xy
            self._frame = Frame(
                valid=True,
                frame=self.draw_eyes_nose(frame=self._frame.image, keypoints=keypoints),
                frame_time=time.time()
            )
        return

    def draw_eyes_nose(self, *, frame: np.ndarray, keypoints: np.ndarray, color_points=(0,0,255), color_lines=(0,255,0)):
        # COCO order: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear
        # keypoints: [num_persons, num_keypoints, 2]
        if frame is None:
            return None
        frame = frame.copy()
        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)
        for person in keypoints:
            nose = tuple(map(int, person[0]))
            left_eye = tuple(map(int, person[1]))
            right_eye = tuple(map(int, person[2]))
            left_ear = tuple(map(int, person[3])) if person.shape[0] > 3 and not np.any(np.isnan(person[3])) else None
            right_ear = tuple(map(int, person[4])) if person.shape[0] > 4 and not np.any(np.isnan(person[4])) else None
            # Arms: 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow, 9-left_wrist, 10-right_wrist
            left_shoulder = tuple(map(int, person[5])) if person.shape[0] > 5 and not np.any(np.isnan(person[5])) else None
            right_shoulder = tuple(map(int, person[6])) if person.shape[0] > 6 and not np.any(np.isnan(person[6])) else None
            left_hip = tuple(map(int, person[11])) if person.shape[0] > 11 and not np.any(np.isnan(person[11])) else None
            right_hip = tuple(map(int, person[12])) if person.shape[0] > 12 and not np.any(np.isnan(person[12])) else None

            # Draw points (all green)
            cv2.circle(frame, nose, 5, green, -1)
            cv2.circle(frame, left_eye, 5, green, -1)
            cv2.circle(frame, right_eye, 5, green, -1)
            if left_ear:
                cv2.circle(frame, left_ear, 5, green, -1)
            if right_ear:
                cv2.circle(frame, right_ear, 5, green, -1)
            if left_shoulder:
                cv2.circle(frame, left_shoulder, 5, green, -1)
            if right_shoulder:
                cv2.circle(frame, right_shoulder, 5, green, -1)
            if left_hip:
                cv2.circle(frame, left_hip, 5, green, -1)
            if right_hip:
                cv2.circle(frame, right_hip, 5, green, -1)
            # Draw lines
            cv2.line(frame, left_eye, nose, color_lines, 2)
            cv2.line(frame, right_eye, nose, color_lines, 2)
            if left_shoulder and right_shoulder:
                cv2.line(frame, left_shoulder, right_shoulder, color_lines, 2)
            if left_hip and right_hip:
                cv2.line(frame, left_hip, right_hip, color_lines, 2)

            # Compute g1 (center between arms/shoulders) and g2 (center between hips)
            g1 = None
            g2 = None
            alpha = None
            if left_shoulder and right_shoulder:
                g1_x = (left_shoulder[0] + right_shoulder[0]) / 2
                g1_y = (left_shoulder[1] + right_shoulder[1]) / 2
                g1 = (int(g1_x), int(g1_y))
            if left_hip and right_hip:
                g2_x = (left_hip[0] + right_hip[0]) / 2
                g2_y = (left_hip[1] + right_hip[1]) / 2
                g2 = (int(g2_x), int(g2_y))
            # Compute alpha as 1/4 from g1 to g2
            if g1 and g2:
                alpha_x = int(g1[0] + 0.25 * (g2[0] - g1[0]))
                alpha_y = int(g1[1] + 0.25 * (g2[1] - g1[1]))
                alpha = (alpha_x, alpha_y)
            # Draw g1, g2 in blue, alpha in red
            if g1:
                cv2.circle(frame, g1, 7, blue, -1)
            if g2:
                cv2.circle(frame, g2, 7, blue, -1)
            if alpha:
                cv2.circle(frame, alpha, 8, red, -1)

            # --- Existing beta logic below ---
            # Compute p1 (center between ears) and p2 (projection)
            p1 = None
            p2 = None
            beta = None
            if left_ear and right_ear:
                p1_x = (left_ear[0] + right_ear[0]) / 2
                p1_y = (left_ear[1] + right_ear[1]) / 2
                p1 = (int(p1_x), int(p1_y))
            if left_eye and right_eye and nose:
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                dx = nose[0] - eye_center_x
                dy = nose[1] - eye_center_y
                p2_x = eye_center_x - dx
                p2_y = eye_center_y - dy
                p2 = (int(p2_x), int(p2_y))
            # Compute beta as center between p1 and p2 if both exist
            if p1 and p2:
                beta_x = int((p1[0] + p2[0]) / 2)
                beta_y = int((p1[1] + p2[1]) / 2)
                beta = (beta_x, beta_y)
            # Draw p1 and p2 in blue, beta in red
            if p1:
                cv2.circle(frame, p1, 6, blue, -1)
            if p2:
                cv2.circle(frame, p2, 6, blue, -1)
            if beta:
                print(f"Drawing beta at: {beta}")
                cv2.circle(frame, beta, 12, red, -1)
        return frame