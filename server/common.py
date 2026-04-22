import math
from pathlib import Path
from pydantic import BaseModel
from multiprocessing import Process, Queue
import numpy as np
from copy import deepcopy
from threading import Lock
from enum import Enum
import psutil

from .platformcapabilities import get_host_capabilities

# epsilon is used to compare floating point numbers
epsilon = 0.000001
# motor_frame is the time during which pwm value must not be changed
motor_frame = 0.01
# aspect ratio of the rectangle where polygon was drawn
POLYGONS_ASPECT_RATIO = 4 / 3
 # delay in seconds used to protect client from receiving too many frames
EPSILON_DELAY = 0.005

def get_platform_info() -> dict:
    """
    Get system information.
    """
    capabilities = get_host_capabilities()
    # total memory
    TOTAL_MEMORY = psutil.virtual_memory()
    # cpu total system usage
    CPU_TOTAL_SYSTEM_USAGE = psutil.cpu_percent()

    operating_system = f"{capabilities.system_name} {capabilities.release} ({capabilities.version})"
    architecture = f"{capabilities.processor} ({capabilities.machine})"
    user = f"{capabilities.node_name}/{capabilities.user_name} (HomeDir: {capabilities.user_home_directory})"
    cpu = f"{CPU_TOTAL_SYSTEM_USAGE:.2f}%"
    # total system ram usage
    ram_used = TOTAL_MEMORY.total - TOTAL_MEMORY.available
    ram_total = TOTAL_MEMORY.total
    ram_percent = ram_used / ram_total * 100
    ram = f"{ram_percent:.2f}% - {ram_used / 1024 / 1024 / 1024:.2f} GB (used) / {ram_total / 1024 / 1024 / 1024:.2f} GB (total)"
    return {
        "operating_system_code": capabilities.operating_system_code,
        "operating_system": operating_system,
        "architecture": architecture,
        "user": user,
        "cpu": cpu,
        "ram": ram,
    }

class Device(Enum):
    RASPBERRY_PI_5 = "RASPBERRY_PI_5"

class Vector2D(BaseModel):
    """Joystick X/Y position from Polygon component."""
    x: float = 0.0
    y: float = 0.0
    
    
class Line2D(BaseModel):
    """Line defined by two points."""
    point1: Vector2D
    point2: Vector2D

class Circle(BaseModel):
    """Circle defined by a center and a radius."""
    center: Vector2D
    radius: float

    def is_point_inside(self, *, point: Vector2D, radiusMultiplier: float = 1.0, radiusDelta: float = 0.0) -> bool:
        """
        Check if a point is inside the circle of radius self.radius * radiusMultiplier + radiusDelta.
        """
        return (point.x - self.center.x) ** 2 + (point.y - self.center.y) ** 2 <= ((self.radius * radiusMultiplier) + radiusDelta) ** 2

class AICircle(BaseModel):
    """Circle defined by a center, a radius, and a confidence."""
    center: Vector2D
    radius: float
    confidence: float
    confidence_threshold: float
    confidence_achieved: bool

    def is_point_inside(self, *, point: Vector2D, radiusMultiplier: float = 1.0, radiusDelta: float = 0.0) -> bool:
        """
        Check if a point is inside the circle of radius self.radius * radiusMultiplier + radiusDelta.
        """
        return (point.x - self.center.x) ** 2 + (point.y - self.center.y) ** 2 <= ((self.radius * radiusMultiplier) + radiusDelta) ** 2

class Frame():
    def __init__(self, *, valid: bool, image: np.ndarray, time: float, pose: list[dict] = None):
        self.valid = valid
        self.image: np.ndarray = image
        self.time = time
        self.pose: list[dict] = pose
        self.uid = get_uid()

    def copy(self) -> 'Frame':
        return Frame(valid=self.valid, image=self.image.copy(), time=self.time,
                     pose=deepcopy(self.pose) if self.pose is not None else None)


class JpegFrame():
    """Frame already JPEG-encoded by the camera worker.

    Used by the MJPEG relay path: camera worker encodes, main process only
    copies bytes out of shared memory and forwards them to the HTTP client.
    """

    def __init__(
        self,
        *,
        valid: bool,
        data: bytes,
        time: float,
        mode: int = 0,
        sequence: int = 0,
        camera_index: int = -1,
    ):
        self.valid = valid
        self.data: bytes = data
        self.time = time
        self.mode = mode
        self.sequence = sequence
        self.camera_index = camera_index
        self.uid = get_uid()


_loading_jpeg_cache: bytes | None = None


def _make_loading_jpeg() -> bytes:
    """Return a cached JPEG with the 'Loading, please wait a minute...'
    placeholder.  Encoded once on first access.
    """
    global _loading_jpeg_cache
    if _loading_jpeg_cache is not None:
        return _loading_jpeg_cache
    import cv2
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "Loading, please wait a minute..."
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(image, text, ((640 - ts[0]) // 2, (480 + ts[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        _loading_jpeg_cache = b""
    else:
        _loading_jpeg_cache = bytes(buf)
    return _loading_jpeg_cache

_uid_lock = Lock()
_last_used_uid = -1
def get_uid() -> int:
    """
    Generate a unique identifier.
    Which starts at 0 and increments by 1.
    """
    global _last_used_uid
    with _uid_lock:
        _last_used_uid += 1
        return _last_used_uid

# def timeout(seconds, action=None):
#     """Calls any function with timeout after 'seconds'.
#        If a timeout occurs, 'action' will be returned or called if
#        it is a function-like object.
#     """
#     def handler(queue, func, args, kwargs):
#         queue.put(func(*args, **kwargs))

#     def decorator(func):

#         def wraps(*args, **kwargs):
#             q = Queue()
#             p = Process(target=handler, args=(q, func, args, kwargs))
#             p.start()
#             p.join(timeout=seconds)
#             if p.is_alive():
#                 p.terminate()
#                 p.join()
#                 if hasattr(action, '__call__'):
#                     return action()
#                 else:
#                     return action
#             else:
#                 return q.get()

#         return wraps

#     return decorator


# def get_settings():
#     """Returns the latest settings.json file as a dictionary"""
#     # common.py is in package/src/submoamoa/
#     # settings.json is in package/src/submoamoa/wwwroot/src/assets/settings.json
#     base_dir = Path(__file__).resolve().parent
#     settings_path = base_dir / "wwwroot/src/assets/settings.json"
    
#     if not settings_path.exists():
#         # Fallback for alternative structure or check if dev env
#         # Try local execution path or ../../../
#         print(f"Warning: settings.json not found at {settings_path}")
#         return {}
        
#     try:
#         with open(settings_path, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Error loading settings.json: {e}")
#         return {}
        

def rotate_vector(*, vector: Vector2D, angle: float) -> Vector2D:
    """
    Rotate the vector counter clockwise by the given degrees.
    """
    return Vector2D(
        x=vector.x * math.cos(math.radians(angle)) 
        - vector.y * math.sin(math.radians(angle)),
        y=vector.x * math.sin(math.radians(angle))
        + vector.y * math.cos(math.radians(angle))
    )

def is_point_on_line(*, point: Vector2D, line: Line2D) -> bool:
    """
    Check if the point is on the line segment between line.point1 and line.point2.
    """
    if point is None or line is None or line.point1 is None or line.point2 is None:
        return False

    x2, y2 = point.x, point.y
    x3, y3 = line.point1.x, line.point1.y
    x4, y4 = line.point2.x, line.point2.y

    # 1. Check for Collinearity (Is it on the infinite line?)
    # (x2 - x3) * (y4 - y3) == (y2 - y3) * (x4 - x3)
    cross_product = (x2 - x3) * (y4 - y3) - (y2 - y3) * (x4 - x3)
    
    if abs(cross_product) > epsilon:
        return False

    # 2. Check Bounding Box (Is it between the endpoints?)
    # We add epsilon to the bounds to account for floating point errors
    within_x = min(x3, x4) - epsilon <= x2 <= max(x3, x4) + epsilon
    within_y = min(y3, y4) - epsilon <= y2 <= max(y3, y4) + epsilon

    return within_x and within_y

def vector_intersection_with_line(*, vector: Vector2D, line: Line2D) -> Vector2D | None:
    """
    Find the intersection of a vector with a line.
    If there is no intersedtion then return None.
    """
    # Line A: (0,0) to (vector.x, vector.y)
    # Line B: (line.point1) to (line.point2)
    x1, y1 = 0.0, 0.0
    x2, y2 = vector.x, vector.y
    x3, y3 = line.point1.x, line.point1.y
    x4, y4 = line.point2.x, line.point2.y
    # Denominator (Cross product of direction vectors)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # 1. Handle Parallel/Collinear Lines
    if abs(denom) < epsilon:
        # Check if they are collinear (Origin lies on Line B)
        # Using cross product of (P3-Origin) and (P4-Origin)
        is_collinear = abs(x3 * y4 - y3 * x4) < epsilon        
        if is_collinear:
            # Return the point on Line B closest to the origin (0,0)
            dx = x4 - x3
            dy = y4 - y3
            t = -(x3 * dx + y3 * dy) / (dx*dx + dy*dy)
            # Note: For infinite line, we don't clamp t between 0 and 1
            return Vector2D(x=x3 + t * dx, y=y3 + t * dy)
        return None
    # 2. Calculate Intersection Point using Determinants
    intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    # 1. Check for Collinearity (Is it on the infinite line?)
    # (intersect_x - x3) * (y4 - y3) == (intersect_y - y3) * (x4 - x3)
    cross_product = (intersect_x - x3) * (y4 - y3) - (intersect_y - y3) * (x4 - x3)
    cross_product2 = intersect_x * y2 - intersect_y * x2
    # If the cross product is not 0 then the intersection point is not on the line
    if abs(cross_product) > epsilon or abs(cross_product2) > epsilon:
        return None
    # 2. Check Bounding Box (Is it between the endpoints?)
    # We add epsilon to the bounds to account for floating point errors
    within_x = min(x1, x2) - epsilon <= intersect_x <= max(x1, x2) + epsilon and \
               min(x3, x4) - epsilon <= intersect_x <= max(x3, x4) + epsilon
    within_y = min(y1, y2) - epsilon <= intersect_y <= max(y1, y2) + epsilon and \
               min(y3, y4) - epsilon <= intersect_y <= max(y3, y4) + epsilon
    # If the intersection point is on the line then return the intersection point
    if within_x and within_y:
        return Vector2D(x=intersect_x, y=intersect_y)
    else:
        return None

def fit_vector_to_polygon(*, vector: Vector2D, polygon: list[Vector2D]) -> Vector2D:
    """
    Fit the vector to the polygon.
    """
    joystick_line = Line2D(point1=Vector2D(x=0, y=0), point2=vector)
    for i in range(len(polygon)):
        polygon_line = Line2D(point1=polygon[i], point2=polygon[(i+1)%len(polygon)])
        intersection = vector_intersection_with_line(vector=vector, line=polygon_line)
        if intersection is not None:
            return intersection
    return vector