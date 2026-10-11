"""Overlay rendering: 9-point skeleton, person score, HUD, help and debug views (no boxes)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xtx4.runtime import Pose

from .config import SkeletonPreset

# BGR colours
SIDE_COLORS = {"l": (60, 220, 60), "r": (0, 150, 255), "c": (0, 230, 255)}
VIEW_COLORS = {0: (255, 200, 0), 1: (255, 200, 0), 2: (255, 120, 200), 3: (200, 120, 255)}
TEXT_COLOR = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(slots=True)
class DrawOptions:
    kpt_threshold: float = 0.3
    show_scores: bool = True
    show_labels: bool = False
    point_radius: int = 4
    line_thickness: int = 2


def _scale_for(frame: np.ndarray) -> float:
    return max(0.6, frame.shape[0] / 720.0)


def draw_poses(frame: np.ndarray, poses: list[Pose], skeleton: SkeletonPreset, opts: DrawOptions) -> None:
    """Draw all poses in place: filled dot = visible, hollow ring = occluded; score above the person."""

    scale = _scale_for(frame)
    thickness = max(1, int(round(opts.line_thickness * scale)))
    radius = max(2, int(round(opts.point_radius * scale)))
    font_scale = 0.45 * scale

    for pose in poses:
        kpts = pose.keypoints
        shown = (kpts[:, 2] >= opts.kpt_threshold) & (pose.visibility > 0)

        for a, b in skeleton.edges:
            if not (shown[a] and shown[b]):
                continue
            side_a, side_b = skeleton.sides[a], skeleton.sides[b]
            color = SIDE_COLORS.get(side_a if side_a == side_b else "c", SIDE_COLORS["c"])
            pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
            pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
            cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

        for k in np.flatnonzero(shown):
            center = (int(round(kpts[k, 0])), int(round(kpts[k, 1])))
            color = SIDE_COLORS.get(skeleton.sides[k], SIDE_COLORS["c"])
            if pose.visibility[k] == 2:  # visible: filled dot with dark rim
                cv2.circle(frame, center, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
            else:  # occluded: hollow ring
                cv2.circle(frame, center, radius, color, max(1, thickness - 1), cv2.LINE_AA)
            if opts.show_labels:
                _text(frame, skeleton.keypoints[k], (center[0] + radius + 2, center[1] - radius),
                      font_scale * 0.85, color)

        if opts.show_scores and shown.any():
            xy = kpts[shown, :2]
            x = int(round(float(xy[:, 0].min())))
            y = int(round(float(xy[:, 1].min()))) - radius - int(8 * scale)
            label = f"{pose.score:.2f}" if pose.source_view == 0 else f"{pose.score:.2f} V{pose.source_view}"
            _text(frame, label, (x, max(12, y)), font_scale, VIEW_COLORS.get(pose.source_view, VIEW_COLORS[0]))


def _text(frame: np.ndarray, text: str, org: tuple[int, int], font_scale: float, color) -> None:
    thick = max(1, int(round(font_scale * 2)))
    cv2.putText(frame, text, org, FONT, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, font_scale, color, thick, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str], *, corner: str = "tl") -> None:
    """Semi-transparent text panel in a frame corner."""

    if not lines:
        return
    scale = _scale_for(frame)
    font_scale = 0.5 * scale
    thick = max(1, int(round(scale)))
    line_h = int(22 * scale)
    pad = int(8 * scale)
    width = max(cv2.getTextSize(line, FONT, font_scale, thick)[0][0] for line in lines) + 2 * pad
    height = line_h * len(lines) + pad
    h, w = frame.shape[:2]
    x0 = pad if corner in ("tl", "bl") else max(0, w - width - pad)
    y0 = pad if corner in ("tl", "tr") else max(0, h - height - pad)
    x1, y1 = min(w, x0 + width), min(h, y0 + height)
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)
    for i, line in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - int(6 * scale)
        cv2.putText(frame, line, (x0 + pad, y), FONT, font_scale, TEXT_COLOR, thick, cv2.LINE_AA)


def draw_debug_views(
    frame: np.ndarray,
    views: dict[int, np.ndarray],
    *,
    poses_net: dict[int, list[np.ndarray]] | None = None,
) -> None:
    """Paste the network input view(s) as thumbnails into the bottom-right corner;
    ``poses_net`` (view -> list of (K, 2) network-px keypoints) is drawn as dots."""

    h, w = frame.shape[:2]
    thumb = max(96, min(h // 3, w // (len(views) + 1)))
    margin = 8
    x = w - margin
    for key in sorted(views, reverse=True):
        view = views[key].copy()
        for kp in (poses_net or {}).get(key, []):
            for px, py in kp:
                cv2.circle(view, (int(round(px)), int(round(py))), 2, (0, 230, 255), -1, cv2.LINE_AA)
        view = cv2.resize(view, (thumb, thumb), interpolation=cv2.INTER_AREA)
        x -= thumb
        y = h - margin - thumb
        if x < 0 or y < 0:
            break
        frame[y:y + thumb, x:x + thumb] = view
        cv2.rectangle(frame, (x, y), (x + thumb - 1, y + thumb - 1), (0, 230, 255), 1)
        _text(frame, f"view {key} ({views[key].shape[1]}px)", (x + 4, y + 16), 0.45, (0, 230, 255))
        x -= margin
