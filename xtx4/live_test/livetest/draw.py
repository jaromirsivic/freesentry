"""Overlay rendering: 9-point skeleton, organ circles (brain, chest), person score, RLE sigma, NMS extent, HUD
and debug views (no boxes from the network -- XTX4 is box-free; the optional extent box is the one OKS-NMS
derives from the points).

The organ circles are drawn exactly like FreeSentry ``server/aidraw.py`` (``draw_aicircle``): white rim plus a
colour ring whose colour encodes the confidence, a filled dot for small circles and a thin ring below the
confidence threshold. ``DrawOptions.overlay_mode`` switches between keypoints, organs and both.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np

from xtx4.runtime import Pose
from xtx4.runtime.decode import DEFAULT_NMS_MIN_SIDE_PX
from xtx4.utils.organs import ORGAN_NAMES

from .config import SkeletonPreset

# BGR colours
SIDE_COLORS = {"l": (60, 220, 60), "r": (0, 150, 255), "c": (0, 230, 255)}
PERSON_COLORS = [
    (255, 120, 60), (60, 200, 255), (120, 255, 120), (255, 80, 200), (80, 120, 255), (255, 220, 80),
    (200, 160, 255), (60, 255, 220), (160, 255, 60), (255, 160, 160),
]
VIEW_COLORS = {0: (255, 200, 0), 1: (255, 200, 0), 2: (255, 120, 200), 3: (200, 120, 255)}
HIGHLIGHT = (255, 255, 255)
TEXT_COLOR = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_MODES = ("keypoints", "organs", "both")


@dataclass(slots=True)
class DrawOptions:
    kpt_threshold: float = 0.3
    show_skeleton: bool = True
    show_points: bool = True
    show_occluded: bool = True
    show_scores: bool = True
    show_labels: bool = False
    show_ids: bool = False
    show_sigma: bool = False
    show_extent: bool = False
    extent_kpt_threshold: float = 0.5
    extent_scale: float = 1.5
    color_mode: str = "side"  # "side" | "person"
    point_radius: int = 4
    line_thickness: int = 2
    highlight: int | None = None  # index into ``poses`` drawn emphasised
    overlay_mode: str = "organs"  # "keypoints" | "organs" | "both" (extradetection.md, decision O3)
    # organ settings, as FreeSentry ``ai_setup.organs`` (enabled / sizeMultiplier / confidenceThreshold /
    # minimumRadius); the chest threshold is the server's 0.20 * 5/7 (chest confidence divided by 7)
    brain_enabled: bool = True
    brain_size_multiplier: float = 1.0
    brain_conf_threshold: float = 0.25
    brain_min_radius: int = 1
    chest_enabled: bool = True
    chest_size_multiplier: float = 1.0
    chest_conf_threshold: float = 0.143
    chest_min_radius: int = 1

    @property
    def show_keypoints(self) -> bool:
        return self.overlay_mode in ("keypoints", "both")

    @property
    def show_organs(self) -> bool:
        return self.overlay_mode in ("organs", "both")

    def organ_params(self) -> list[tuple[int, bool, float, float, int]]:
        """``(row, enabled, size_multiplier, confidence_threshold, minimum_radius)`` per organ."""

        return [
            (i, bool(getattr(self, f"{name}_enabled")), float(getattr(self, f"{name}_size_multiplier")),
             float(getattr(self, f"{name}_conf_threshold")), int(getattr(self, f"{name}_min_radius")))
            for i, name in enumerate(ORGAN_NAMES)
        ]


def options_from_settings(settings, *, highlight: int | None = None) -> DrawOptions:
    """``DrawOptions`` from a ``livetest.settings.Settings`` (same field names; the extent box follows
    the OKS-NMS keypoint threshold and scale factor)."""

    values = {f.name: getattr(settings, f.name) for f in fields(DrawOptions) if hasattr(settings, f.name)}
    values.update(extent_kpt_threshold=settings.nms_kpt_threshold, extent_scale=settings.oks_nms_scale_factor,
                  highlight=highlight)
    return DrawOptions(**values)


def _scale_for(frame: np.ndarray) -> float:
    return max(0.6, frame.shape[0] / 720.0)


def extent_box(pose: Pose, kpt_threshold: float, scale: float,
               min_side: float = DEFAULT_NMS_MIN_SIDE_PX) -> tuple[int, int, int, int]:
    """The expanded extent box OKS-NMS uses for this pose (``xtx4.runtime.decode._extent_boxes``)."""

    kpts = pose.keypoints
    use = kpts[:, 2] >= kpt_threshold
    if not use.any():
        use[:] = True
    xy = kpts[use, :2].astype(np.float64)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    w, h = max(scale * (x1 - x0), min_side), max(scale * (y1 - y0), min_side)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))


def draw_aicircle(
    image: np.ndarray,
    *,
    cx: int,
    cy: int,
    radius: int,
    confidence: float,
    confidence_achieved: bool,
    size_multiplier: float,
    minimum_radius: int,
) -> None:
    """Port of FreeSentry ``server/aidraw.py::draw_aicircle`` (1:1, including the ``radius * outline``
    of the small-circle branches, decision O4)."""

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


def draw_pose_organs(frame: np.ndarray, pose: Pose, opts: DrawOptions, *, emphasised: bool = False) -> None:
    """Brain and chest circles of one pose (FreeSentry style); a selected pose gets a thin white halo."""

    if pose.organs is None:
        return
    outline = max(min(int(max(frame.shape[:2]) / 640), 20), 1)
    for row, enabled, size_multiplier, threshold, minimum_radius in opts.organ_params():
        if not enabled:
            continue
        x, y, radius, conf = (float(v) for v in pose.organs[row])
        if not np.isfinite([x, y, radius, conf]).all():
            continue
        cx, cy = int(x), int(y)
        draw_aicircle(frame, cx=cx, cy=cy, radius=int(radius), confidence=conf,
                      confidence_achieved=conf >= threshold, size_multiplier=size_multiplier,
                      minimum_radius=minimum_radius)
        if emphasised:
            cv2.circle(frame, (cx, cy), int(radius * size_multiplier) + 2 * outline, HIGHLIGHT, 1, cv2.LINE_AA)


def _organ_label_anchor(pose: Pose, opts: DrawOptions) -> tuple[int, int] | None:
    """Top-left of the enabled organ circles (where the score goes when keypoints are hidden)."""

    if pose.organs is None:
        return None
    tops = []
    for row, enabled, mult, _threshold, _minimum in opts.organ_params():
        if enabled:
            x, y, r = (float(v) for v in pose.organs[row, :3])
            tops.append((x - r * mult, y - r * mult))
    tops = [t for t in tops if np.isfinite(t).all()]
    if not tops:
        return None
    return int(round(min(t[0] for t in tops))), int(round(min(t[1] for t in tops)))


def draw_poses(frame: np.ndarray, poses: list[Pose], skeleton: SkeletonPreset, opts: DrawOptions) -> None:
    """Draw all poses in place: organ circles (``overlay_mode`` organs / both) first, then the keypoints
    (keypoints / both: filled dot = visible, hollow ring = occluded); score above the person."""

    scale = _scale_for(frame)
    thickness = max(1, int(round(opts.line_thickness * scale)))
    radius = max(2, int(round(opts.point_radius * scale)))
    font_scale = 0.45 * scale

    for index, pose in enumerate(poses):
        kpts = pose.keypoints
        shown = (kpts[:, 2] >= opts.kpt_threshold) & (pose.visibility > 0)
        if not opts.show_occluded:
            shown &= pose.visibility == 2
        person_color = PERSON_COLORS[index % len(PERSON_COLORS)]
        emphasised = opts.highlight == index
        line_t = thickness + (2 if emphasised else 0)

        def color_of(k: int) -> tuple[int, int, int]:
            if opts.color_mode == "person":
                return person_color
            return SIDE_COLORS.get(skeleton.sides[k], SIDE_COLORS["c"])

        if opts.show_extent:
            x0, y0, x1, y1 = extent_box(pose, opts.extent_kpt_threshold, opts.extent_scale)
            cv2.rectangle(frame, (x0, y0), (x1, y1), HIGHLIGHT if emphasised else (180, 180, 180), 1, cv2.LINE_AA)

        # organs below the keypoints, so the white rims never hide a point in "both"
        if opts.show_organs:
            draw_pose_organs(frame, pose, opts, emphasised=emphasised)
        if not opts.show_keypoints:
            shown = np.zeros_like(shown)

        if opts.show_sigma:
            for k in np.flatnonzero(shown):
                center = (int(round(kpts[k, 0])), int(round(kpts[k, 1])))
                sigma = int(round(float(pose.sigma_px[k])))
                if sigma > 0:
                    cv2.circle(frame, center, sigma, color_of(k), 1, cv2.LINE_AA)

        if opts.show_skeleton:
            for a, b in skeleton.edges:
                if not (shown[a] and shown[b]):
                    continue
                if opts.color_mode == "person":
                    color = person_color
                else:
                    side_a, side_b = skeleton.sides[a], skeleton.sides[b]
                    color = SIDE_COLORS.get(side_a if side_a == side_b else "c", SIDE_COLORS["c"])
                pa = (int(round(kpts[a, 0])), int(round(kpts[a, 1])))
                pb = (int(round(kpts[b, 0])), int(round(kpts[b, 1])))
                if emphasised:
                    cv2.line(frame, pa, pb, HIGHLIGHT, line_t + 2, cv2.LINE_AA)
                cv2.line(frame, pa, pb, color, line_t, cv2.LINE_AA)

        for k in np.flatnonzero(shown):
            center = (int(round(kpts[k, 0])), int(round(kpts[k, 1])))
            color = color_of(k)
            if opts.show_points:
                if pose.visibility[k] == 2:  # visible: filled dot with dark rim
                    cv2.circle(frame, center, radius + 1, HIGHLIGHT if emphasised else (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
                else:  # occluded: hollow ring
                    cv2.circle(frame, center, radius, color, max(1, thickness - 1), cv2.LINE_AA)
            if opts.show_labels:
                _text(frame, skeleton.keypoints[k], (center[0] + radius + 2, center[1] - radius),
                      font_scale * 0.85, color)

        anchor = None
        if shown.any():
            xy = kpts[shown, :2]
            anchor = (int(round(float(xy[:, 0].min()))), int(round(float(xy[:, 1].min()))))
        elif opts.show_organs:
            anchor = _organ_label_anchor(pose, opts)
        if (opts.show_scores or opts.show_ids) and anchor is not None:
            x, y = anchor[0], anchor[1] - radius - int(8 * scale)
            parts = []
            if opts.show_ids:
                parts.append(f"#{index}")
            if opts.show_scores:
                parts.append(f"{pose.score:.2f}")
            if pose.source_view != 0:
                parts.append(f"V{pose.source_view}")
            color = person_color if opts.color_mode == "person" else VIEW_COLORS.get(pose.source_view, VIEW_COLORS[0])
            _text(frame, " ".join(parts), (x, max(12, y)), font_scale, HIGHLIGHT if emphasised else color)


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
