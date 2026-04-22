"""
Pure AI drawing helpers — run inside the camera worker process.

This module draws organ circles (pose overlay) and the engagement status
rectangle/text directly onto a frame image.  It intentionally does NOT
import :mod:`server.settingscontroller` or any other main-process
singleton: everything it needs arrives as function arguments.  That keeps
it safe to call from the worker's subprocess, where importing
``settingscontroller`` would try to start its own thread/IO stack.

The drawing logic here is the one previously living in
:meth:`server.aiagent.AIAgent.draw_engagement_result` and
:func:`server.cameraai.draw_pose`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Engagement status color table (kept in sync with AIAgent.draw_engagement_result)
# ---------------------------------------------------------------------------
_STATUS_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    # status_value -> (frame_color, text_background, text_color)
    "waiting_to_start":               ((128, 128, 128), (255, 255, 255), (0, 0, 0)),
    "fps_condition_not_satisfied":    ((0, 128, 0),     (255, 255, 255), (0, 0, 0)),
    "not_engaging":                   ((0, 255, 0),     (255, 255, 255), (0, 0, 0)),
    "arming":                         ((0, 192, 255),   (255, 255, 255), (0, 0, 0)),
    "disengaging":                    ((0, 192, 255),   (255, 255, 255), (0, 0, 0)),
    "engaging":                       ((0, 0, 255),     (0, 0, 255),     (0, 0, 0)),
    "exit_strategy_under_execution":  ((255, 0, 128),   (255, 255, 255), (0, 0, 0)),
    "exit_strategy_executed":         ((240, 255, 240), (255, 255, 255), (0, 0, 0)),
}


def _aicircle_components(organ: Any) -> tuple[int, int, int, float, bool] | None:
    """Extract (cx, cy, radius, confidence, confidence_achieved) from a pose
    organ entry.

    Accepts both the Pydantic :class:`~server.common.AICircle` and a plain
    dict shape ``{"point": (x, y), "radius": r, "confidence": c,
    "confidence_threshold": t}`` (as sent through the pose pipe).
    """
    if organ is None:
        return None
    if hasattr(organ, "center"):
        cx = int(organ.center.x)
        cy = int(organ.center.y)
        radius = int(organ.radius)
        confidence = float(organ.confidence)
        threshold = float(getattr(organ, "confidence_threshold", 0.0))
        confidence_achieved = bool(getattr(organ, "confidence_achieved", confidence >= threshold))
        return cx, cy, radius, confidence, confidence_achieved
    if isinstance(organ, dict):
        point = organ.get("point") or organ.get("center")
        if point is None:
            return None
        if isinstance(point, dict):
            cx = int(point.get("x", 0)); cy = int(point.get("y", 0))
        else:
            cx = int(point[0]); cy = int(point[1])
        radius = int(organ.get("radius", 0))
        confidence = float(organ.get("confidence", 0.0))
        threshold = float(organ.get("confidence_threshold", 0.0))
        confidence_achieved = bool(organ.get("confidence_achieved", confidence >= threshold))
        return cx, cy, radius, confidence, confidence_achieved
    return None


def draw_aicircle(
    *,
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    confidence: float,
    confidence_achieved: bool,
    size_multiplier: float,
    minimum_radius: int,
) -> None:
    resolution_max = max(image.shape[1], image.shape[0])
    circle_outline_size = max(min(int(resolution_max / 640), 20), 1)
    p = (cx, cy)
    radius_eff = int(radius * size_multiplier)
    cb = 1 - (abs(confidence - 0.5) * 2)
    color = (int(cb * 255), 0, int(confidence * 255))

    if confidence_achieved and radius_eff > minimum_radius:
        white = (255, 255, 255)
        if radius_eff > (8 * circle_outline_size):
            cv2.circle(image, p, max(radius_eff - 3 * circle_outline_size, 0), white, 8 * circle_outline_size)
            cv2.circle(image, p, max(radius_eff - 4 * circle_outline_size, 0), color, 3 * circle_outline_size)
        else:
            cv2.circle(image, p, max(radius_eff * circle_outline_size, 0), white, -1)
            cv2.circle(image, p, max(radius_eff * circle_outline_size, 0), color, 4 * circle_outline_size)
    else:
        cv2.circle(image, p, max(radius_eff * circle_outline_size, 0), color, 2 * circle_outline_size)


_ORGAN_NAMES = ("brain", "chest", "abdomen", "liver", "heart")


def draw_pose_overlay(
    *,
    image: np.ndarray,
    pose: Any,
    ai_setup: dict,
) -> None:
    """Draw organ circles for all detected people.

    ``pose`` may be a list of dicts (one per person) or a single dict.
    Each person's dict maps organ name -> :class:`AICircle`-like object.
    """
    if pose is None:
        return
    poses = pose if isinstance(pose, list) else [pose]
    organs_cfg = ai_setup.get("organs", {}) if isinstance(ai_setup, dict) else {}

    organ_params: dict[str, tuple[bool, float, int]] = {}
    for organ_name in _ORGAN_NAMES:
        cfg = organs_cfg.get(organ_name, {}) if isinstance(organs_cfg, dict) else {}
        if not isinstance(cfg, dict):
            cfg = {}
        organ_params[organ_name] = (
            bool(cfg.get("enabled", True)),
            float(cfg.get("sizeMultiplier", 1.0)),
            int(cfg.get("minimumRadius", 1)),
        )

    for person in poses:
        if person is None:
            continue
        for organ_name, (enabled, size_multiplier, minimum_radius) in organ_params.items():
            if not enabled:
                continue
            organ_obj = person.get(organ_name) if isinstance(person, dict) else None
            comp = _aicircle_components(organ_obj)
            if comp is None:
                continue
            cx, cy, radius, confidence, confidence_achieved = comp
            draw_aicircle(
                image=image,
                cx=cx, cy=cy, radius=radius,
                confidence=confidence,
                confidence_achieved=confidence_achieved,
                size_multiplier=size_multiplier,
                minimum_radius=minimum_radius,
            )


def draw_engagement_overlay(
    *,
    image: np.ndarray,
    status_value: str,
    engagement_counter: int,
    fps: float,
    pose: Any,
    ai_setup: dict,
    draw_ai_stats: bool = True,
) -> None:
    """Draw the full AI overlay (pose circles + engagement rect + text).

    Matches the visual output of the original
    :meth:`AIAgent.draw_engagement_result` exactly.
    """
    if not draw_ai_stats:
        return

    try:
        draw_pose_overlay(image=image, pose=pose, ai_setup=ai_setup)
    except Exception as exc:
        print(f"[aidraw] Error drawing pose overlay: {exc}")

    color, text_background, text_color = _STATUS_COLORS.get(
        status_value,
        ((240, 255, 240), (255, 255, 255), (0, 0, 0)),
    )

    h, w = image.shape[:2]
    scale = max(w, 1) / 1280
    cv2.rectangle(image, (0, 0), (int(450 * scale), int(140 * scale)), text_background, -1)
    cv2.rectangle(image, (0, 0), (w, h), color, int(20 * scale))

    status_str = f'Status: {status_value}'
    cv2.putText(image, status_str, (int(20 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1)
    counter_str = f'Engagement Counter: {engagement_counter}'
    cv2.putText(image, counter_str, (int(20 * scale), int(80 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1)
    fps_str = f'FPS: {fps:.2f}'
    cv2.putText(image, fps_str, (int(20 * scale), int(120 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1)
