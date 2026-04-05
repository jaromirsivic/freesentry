from PIL import Image
from .common import AICircle, Vector2D
import time
import cv2
import threading
import numpy as np
from .yolomodels import YOLOModels
# from .camerapostprocessing import NodeImage, NodeImageCroppedResized, NodeImageAI
from copy import deepcopy
import random

# def draw_circle(image: np.ndarray, point: dict, confidence_threshold: float, draw_confidence_threshold: bool) -> np.ndarray:
#     px = int(point["point"][0])
#     py = int(point["point"][1])
#     p = (px, py)
#     radius = int(point["radius"])
#     confidence = point["confidence"]
#     cb = 1 - (abs(confidence - 0.5) * 2)  
#     color = (int(cb * 255), 0, int(confidence * 255))    
#     if draw_confidence_threshold and confidence > confidence_threshold:
#         # color definition
#         white = (255, 255, 255)
#         # draw circle
#         if radius > 7:
#             cv2.circle(image, p, radius-3, white, 8)
#             cv2.circle(image, p, radius-4, color, 3)
#         else:
#             cv2.circle(image, p, radius, white, -1)
#             cv2.circle(image, p, radius, color, 4)
#     else:
#         # draw circle
#         cv2.circle(image, p, radius, color, 2)

def draw_aicircle(*, image: np.ndarray, aicircle: AICircle, size_multiplier: float, minimum_radius: int) -> np.ndarray:
    # get the resolution of the image
    resolution_x = image.shape[1]
    resolution_y = image.shape[0]
    resolution_max = max(resolution_x, resolution_y)
    circle_outline_size = max(min(int(resolution_max / 640), 20), 1)    # get the center of the circle
    px = int(aicircle.center.x)
    py = int(aicircle.center.y)
    p = (px, py)
    # get the radius of the circle
    radius = int(aicircle.radius * size_multiplier)
    # get the confidence of the circle
    confidence = aicircle.confidence
    cb = 1 - (abs(confidence - 0.5) * 2)  
    color = (int(cb * 255), 0, int(confidence * 255))    
    # if the confidence is achieved and the radius is greater than the minimum radius, draw the circle
    if aicircle.confidence_achieved and radius > minimum_radius:
        # color definition
        white = (255, 255, 255)
        # draw circle
        if radius > (8 * circle_outline_size):
            cv2.circle(image, p, max(radius - 3 * circle_outline_size, 0), white, 8 * circle_outline_size)
            cv2.circle(image, p, max(radius - 4 * circle_outline_size, 0), color, 3 * circle_outline_size)
        else:
            cv2.circle(image, p, max(radius * circle_outline_size, 0), white, -1)
            cv2.circle(image, p, max(radius * circle_outline_size, 0), color, 4 * circle_outline_size)
    else:
        # draw circle
        cv2.circle(image, p, max(radius * circle_outline_size, 0), color, 2 * circle_outline_size)

def draw_pose(*, image: np.ndarray, pose: dict | list, ai_setup: dict, copy_image: bool = True) -> np.ndarray:
    """Draw keypoints on a frame."""
    # if the pose is a list, iterate over all poses
    poses = pose if isinstance(pose, list) else [pose]
    result = image.copy() if copy_image else image
    # get the organs from the ai setup
    organs = ai_setup.get("organs", {})
    # brain settings
    brain_enabled = bool(organs.get("brain", {}).get("enabled", True))
    brain_size_multiplier = float(organs.get("brain", {}).get("sizeMultiplier", 1.0))
    brain_minimum_radius = int(organs.get("brain", {}).get("minimumRadius", 1))
    # chest settings
    chest_enabled = bool(organs.get("chest", {}).get("enabled", True))
    chest_size_multiplier = float(organs.get("chest", {}).get("sizeMultiplier", 1.0))
    chest_minimum_radius = int(organs.get("chest", {}).get("minimumRadius", 1))
    # abdomen settings
    abdomen_enabled = bool(organs.get("abdomen", {}).get("enabled", True))
    abdomen_size_multiplier = float(organs.get("abdomen", {}).get("sizeMultiplier", 1.0))
    abdomen_minimum_radius = int(organs.get("abdomen", {}).get("minimumRadius", 1))
    # liver settings
    liver_enabled = bool(organs.get("liver", {}).get("enabled", True))
    liver_size_multiplier = float(organs.get("liver", {}).get("sizeMultiplier", 1.0))
    liver_minimum_radius = int(organs.get("liver", {}).get("minimumRadius", 1))
    # heart settings
    heart_enabled = bool(organs.get("heart", {}).get("enabled", True))
    heart_size_multiplier = float(organs.get("heart", {}).get("sizeMultiplier", 1.0))
    heart_minimum_radius = int(organs.get("heart", {}).get("minimumRadius", 1))
    # iterate over all peoples poses
    for p in poses:
        if p is None:
            continue
        # draw selected points
        if brain_enabled:
            brain = p["brain"]
            draw_aicircle(image=result, aicircle=brain, size_multiplier=brain_size_multiplier, minimum_radius=brain_minimum_radius)
        if chest_enabled:
            chest = p["chest"]
            draw_aicircle(image=result, aicircle=chest, size_multiplier=chest_size_multiplier, minimum_radius=chest_minimum_radius)
        if abdomen_enabled:
            abdomen = p["abdomen"]
            draw_aicircle(image=result, aicircle=abdomen, size_multiplier=abdomen_size_multiplier, minimum_radius=abdomen_minimum_radius)
        if liver_enabled:
            liver = p["liver"]
            draw_aicircle(image=result, aicircle=liver, size_multiplier=liver_size_multiplier, minimum_radius=liver_minimum_radius)
        if heart_enabled:
            heart = p["heart"]
            draw_aicircle(image=result, aicircle=heart, size_multiplier=heart_size_multiplier, minimum_radius=heart_minimum_radius)
    return result

def calculate_point(point_start: tuple[float, float], 
                    point_end: tuple[float, float], 
                    fraction: float) -> tuple[float, float]:
    """
    Calculate a point between two points.
    """
    return (point_start[0] + fraction * (point_end[0] - point_start[0]),
            point_start[1] + fraction * (point_end[1] - point_start[1]))

def calculate_distance(point_start: tuple[float, float], 
                       point_end: tuple[float, float]) -> float:
    """
    Calculate the distance between two points.
    """
    return np.sqrt((point_start[0] - point_end[0])**2 + (point_start[1] - point_end[1])**2)

def translate_raw_pose_to_pose_dict(*, raw_pose: dict, ai_setup: dict) -> list[dict]:
    """
    Translate a raw pose to a pose dictionary.
    """
    result = []
    organs = ai_setup.get("organs", {})
    brain_confidence_threshold = float(organs.get("brain", {}).get("confidenceThreshold", 0.01))
    chest_confidence_threshold = float(organs.get("chest", {}).get("confidenceThreshold", 0.01))
    abdomen_confidence_threshold = float(organs.get("abdomen", {}).get("confidenceThreshold", 0.01))
    liver_confidence_threshold = float(organs.get("liver", {}).get("confidenceThreshold", 0.01))
    heart_confidence_threshold = float(organs.get("heart", {}).get("confidenceThreshold", 0.01))
    for pose in raw_pose:
        brain = AICircle(center=Vector2D(x=pose["brain"]["point"][0], y=pose["brain"]["point"][1]),
                     radius=pose["brain"]["radius"],
                     confidence=pose["brain"]["confidence"],
                     confidence_threshold=brain_confidence_threshold,
                     confidence_achieved=pose["brain"]["confidence"] >= brain_confidence_threshold)
        chest = AICircle(center=Vector2D(x=pose["chest"]["point"][0], y=pose["chest"]["point"][1]),
                     radius=pose["chest"]["radius"],
                     confidence=pose["chest"]["confidence"],
                     confidence_threshold=chest_confidence_threshold,
                     confidence_achieved=pose["chest"]["confidence"] >= chest_confidence_threshold)
        abdomen = AICircle(center=Vector2D(x=pose["abdomen"]["point"][0], y=pose["abdomen"]["point"][1]),
                     radius=pose["abdomen"]["radius"],
                     confidence=pose["abdomen"]["confidence"],
                     confidence_threshold=abdomen_confidence_threshold,
                     confidence_achieved=pose["abdomen"]["confidence"] >= abdomen_confidence_threshold)
        liver = AICircle(center=Vector2D(x=pose["liver"]["point"][0], y=pose["liver"]["point"][1]),
                     radius=pose["liver"]["radius"],
                     confidence=pose["liver"]["confidence"],
                     confidence_threshold=liver_confidence_threshold,
                     confidence_achieved=pose["liver"]["confidence"] >= liver_confidence_threshold)
        heart = AICircle(center=Vector2D(x=pose["heart"]["point"][0], y=pose["heart"]["point"][1]),
                     radius=pose["heart"]["radius"],
                     confidence=pose["heart"]["confidence"],
                     confidence_threshold=heart_confidence_threshold,
                     confidence_achieved=pose["heart"]["confidence"] >= heart_confidence_threshold)
        result.append({
            "brain": brain,
            "chest": chest,
            "abdomen": abdomen,
            "liver": liver,
            "heart": heart
        })
    return result
    

def get_pose_dict(*, keypoints: np.ndarray, ai_setup: dict) -> list[dict]:
    """
    Args:
        keypoints: Shape (N, 17, 3) or (N, 17, 2).
                   If (17, 3) or (17, 2) is passed, it is treated as (1, 17, X).
        confidence_threshold: Points below this score will be set to None.
                        (Only applies if input has confidence data)
    """
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    # Handle shape to ensure (N, 17, X)
    if keypoints.ndim == 2:
        keypoints = np.expand_dims(keypoints, axis=0)
    # Check if the number of keypoints is 17
    if keypoints.shape[1] != 17:
        raise ValueError(f"Expected 17 keypoints per person, got {keypoints.shape[1]}")
    # List of peoples pose dictionaries
    results: list[dict] = []
    has_confidence = keypoints.shape[2] >= 3
    num_people = keypoints.shape[0]
    # If the keypoints have confidence data, process the data
    if has_confidence:
        for p in range(num_people):
            pose: dict = {}
            for i, name in enumerate(keypoint_names):
                # Extract x,y and confidence
                point = keypoints[p, i, :2]
                conf = float(keypoints[p, i, 2])
                pose[name] = {
                    "point": (int(point[0]), int(point[1])),
                    "confidence": conf,
                    "radius": 0
                }
            results.append(pose)
    # If the keypoints do not have confidence data, set the confidence to 1.0
    else:
        for p in range(num_people):
            pose: dict = {}
            for i, name in enumerate(keypoint_names):
                point = keypoints[p, i, :2]
                pose[name] = {
                    "point": (int(point[0]), int(point[1])),
                    "confidence": 1.0,
                    "radius": 0
                }
            results.append(pose)
    # Calculate the position of the brain and the chest and abdomen
    _get_brain_position(keypoints=results)
    _get_chest_and_abdomen_position(keypoints=results)

    return results

def _get_brain_position(*, keypoints: list[dict], confidence_threshold: float = 0.01):
    # Calculate the position of the brain
    for pose in keypoints:
        # Set all points in pose
        left_ear_point = pose["left_ear"]["point"]
        right_ear_point = pose["right_ear"]["point"]
        left_eye_point = pose["left_eye"]["point"]
        right_eye_point = pose["right_eye"]["point"]
        nose_point = pose["nose"]["point"]
        # Compute overall confidence
        left_ear_confidence = pose["left_ear"]["confidence"]
        right_ear_confidence = pose["right_ear"]["confidence"]
        left_eye_confidence = pose["left_eye"]["confidence"]
        right_eye_confidence = pose["right_eye"]["confidence"]
        nose_confidence = pose["nose"]["confidence"]
        confidence_of_all_points_is_ok = left_ear_confidence >= confidence_threshold and \
            right_ear_confidence >= confidence_threshold and \
            left_eye_confidence >= confidence_threshold and \
            right_eye_confidence >= confidence_threshold and \
            nose_confidence >= confidence_threshold
        # Calculate the position of the brain from the left ear and right ear
        if (not confidence_of_all_points_is_ok and \
            left_ear_confidence >= confidence_threshold and \
            right_ear_confidence >= confidence_threshold):
            # Calculate center
            brain = calculate_point(left_ear_point, right_ear_point, 0.5)
            # Calculate radius
            d = calculate_distance(brain, left_ear_point)
            radius = d * 0.59
            radius = int(radius)
            # Calculate confidence
            avg_confidence = (left_ear_confidence + right_ear_confidence) / 2
            pose["brain"] = {
                "point": brain,
                "confidence": avg_confidence,
                "radius": radius
            }
        else:
            # Calculate center
            nose_left_ear = calculate_point(nose_point, left_ear_point, 2)
            nose_right_ear = calculate_point(nose_point, right_ear_point, 2)
            nose_left_ear2 = calculate_point(nose_point, left_ear_point, 0.5)
            nose_right_ear2 = calculate_point(nose_point, right_ear_point, 0.5)
            nose_left_eye = calculate_point(nose_left_ear2, left_eye_point, 2)
            nose_right_eye = calculate_point(nose_right_ear2, right_eye_point, 2)
            nose_left_ear_eye = calculate_point(nose_left_ear, nose_left_eye, 0.5)
            nose_right_ear_eye = calculate_point(nose_right_ear, nose_right_eye, 0.5)
            brain = calculate_point(nose_left_ear_eye, nose_right_ear_eye, 0.5)
            # Calculate radius
            distance_brain_nose = calculate_distance(brain, nose_point)
            distance_brain_left_ear = calculate_distance(brain, left_ear_point)
            distance_brain_right_ear = calculate_distance(brain, right_ear_point)
            d = distance_brain_nose if distance_brain_nose > distance_brain_left_ear else distance_brain_left_ear
            d = d if d > distance_brain_right_ear else distance_brain_right_ear
            radius = d * 0.59
            radius = int(radius)
            avg_confidence = (
                nose_confidence + 
                left_ear_confidence + 
                right_ear_confidence + 
                left_eye_confidence + 
                right_eye_confidence) / 5
            min_confidence = min(
                nose_confidence,
                left_ear_confidence,
                right_ear_confidence,
                left_eye_confidence,
                right_eye_confidence
            )
            confidence = min_confidence + (avg_confidence - min_confidence) * 0.8
            pose["brain"] = {
                "point": brain,
                "confidence": confidence,
                "radius": radius
            }
        

def _get_chest_and_abdomen_position(*, keypoints: list[dict], confidence_threshold: float = 0.01):
    # Calculate the position of the chest and abdomen
    for pose in keypoints:
        left_shoulder_right_shoulder = calculate_point(pose["left_shoulder"]["point"], pose["right_shoulder"]["point"], 0.5)
        left_hip_right_hip = calculate_point(pose["left_hip"]["point"], pose["right_hip"]["point"], 0.5)
        left_shoulder_left_hip = calculate_point(pose["left_shoulder"]["point"], pose["left_hip"]["point"], 0.4)
        right_shoulder_right_hip = calculate_point(pose["right_shoulder"]["point"], pose["right_hip"]["point"], 0.4)
        chest = calculate_point(left_shoulder_right_shoulder, left_hip_right_hip, 0.2)
        heart = calculate_point(left_shoulder_left_hip, chest, 0.5)
        abdomen = calculate_point(left_shoulder_right_shoulder, left_hip_right_hip, 0.7)
        liver = calculate_point(left_shoulder_left_hip, right_shoulder_right_hip, 0.7)
        d1 = calculate_distance(pose["left_shoulder"]["point"], pose["right_shoulder"]["point"])
        d2 = calculate_distance(pose["left_hip"]["point"], pose["right_hip"]["point"])
        d3 = calculate_distance(pose["left_shoulder"]["point"], pose["left_hip"]["point"])
        d4 = calculate_distance(pose["right_shoulder"]["point"], pose["right_hip"]["point"])
        davg_left_right = (d1 + d2) * 0.21
        davg_top_bottom = (d3 + d4) * 0.1
        davg = davg_left_right if davg_left_right > davg_top_bottom else davg_top_bottom
        dmin_left_right = d1 if d1 < d2 else d2
        dmin_top_bottom = d3 if d3 < d4 else d4
        dmin_top_bottom = dmin_top_bottom * 0.25
        dmin = dmin_left_right if dmin_left_right > dmin_top_bottom else dmin_top_bottom
        # compute the minimum confidence
        left_shoulder_confidence = pose["left_shoulder"]["confidence"]
        right_shoulder_confidence = pose["right_shoulder"]["confidence"]
        left_hip_confidence = pose["left_hip"]["confidence"]
        right_hip_confidence = pose["right_hip"]["confidence"]
        left_shoulder_confidence_doubled = left_shoulder_confidence * 2
        right_shoulder_confidence_doubled = right_shoulder_confidence * 2
        left_hip_confidence_doubled = left_hip_confidence * 2
        right_hip_confidence_doubled = right_hip_confidence * 2
        min_confidence = min(left_shoulder_confidence, right_shoulder_confidence, left_hip_confidence, right_hip_confidence)
        #avg_confidence_top = (left_shoulder_confidence + right_shoulder_confidence) / 2
        #avg_confidence_bottom = (left_hip_confidence + right_hip_confidence) / 2
        pose["chest"] = {
            "point": chest,
            "confidence": (left_shoulder_confidence_doubled + right_shoulder_confidence_doubled +
                          left_hip_confidence + right_hip_confidence + min_confidence) / 5,
            "radius": int(davg)
        }
        pose["heart"] = {
            "point": heart,
            "confidence": (left_shoulder_confidence_doubled + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence + min_confidence) / 5,
            "radius": int(davg * 0.5)
        }
        pose["abdomen"] = {
            "point": abdomen,
            "confidence": (left_shoulder_confidence + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence_doubled + min_confidence) / 5,
            "radius": int(dmin * 0.5)
        }
        pose["liver"] = {
            "point": liver,
            "confidence": (left_shoulder_confidence + right_shoulder_confidence +
                          left_hip_confidence_doubled + right_hip_confidence_doubled + min_confidence) / 5,
            "radius": int(davg * 0.65)
        }
