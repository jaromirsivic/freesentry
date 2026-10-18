"""XTX4 test application: live video with the pose overlay on the left, every model /
decode / source / display option on the right, applied while the video runs.

Tk owns the main thread; capture and inference run in the threads of
:class:`livetest.engine.LiveEngine`. The GUI polls the newest result every ~10 ms,
draws it at full frame resolution (that is also what screenshots and recordings
contain) and scales it into the video area.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import cv2
import numpy as np
from xtx4.runtime.preprocess import apply_affine_points, invert_affine

from .blit import make_blitter
from .backends import BACKEND_LABELS, RunnerError, backend_installed, vulkan_devices
from .camera import CameraSource, list_cameras
from .config import AppConfig, ConfigError, Preset
from .draw import OVERLAY_MODES, DrawOptions, draw_debug_views, draw_panel, draw_poses, options_from_settings
from .engine import LiveEngine
from .settings import (
    APP_DIR,
    FIELD_TYPES,
    RUNNER_KEYS,
    SOURCE_KEYS,
    Settings,
    all_presets,
    apply_preset,
    config_defaults,
    from_dict,
    manifest_decode,
    save_last,
    save_user_presets,
    validate_for_config,
)

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = APP_DIR / "screenshots"
RECORDING_DIR = APP_DIR / "recordings"
RESOLUTIONS = ["640x480", "800x600", "1280x720", "1920x1080", "2560x1440", "3840x2160"]
PANEL_WIDTH = 440  # at 96 dpi; scaled with the display DPI
VIS_TEXT = {0: "absent", 1: "occluded", 2: "visible"}
# Keys that only change the drawing: no detection re-run needed while paused.
DISPLAY_KEYS = ("kpt_threshold", "show_skeleton", "show_points", "show_occluded", "show_scores", "show_labels",
                "show_ids", "show_sigma", "show_extent", "color_mode", "point_radius", "line_thickness", "show_hud",
                "overlay_mode", "brain_enabled", "brain_size_multiplier", "brain_conf_threshold", "brain_min_radius",
                "chest_enabled", "chest_size_multiplier", "chest_conf_threshold", "chest_min_radius")
ORGAN_ROWS = ("brain", "chest")
OVERLAY_TEXT = {"keypoints": "Keypoints", "organs": "Organs", "both": "Keypoints + organs"}

SHORTCUTS = [
    ("Space", "pause / resume"),
    (".", "next frame (while paused)"),
    ("+ / -", "detection threshold +-0.05"),
    ("[ / ]", "keypoint (drawing) threshold +-0.05"),
    ("1 .. 9", "input size (in the order of the network's sizes)"),
    ("b", "backend ncnn <-> ONNX Runtime"),
    ("w", "view mode whole <-> pyramid"),
    ("n", "OKS-NMS on / off"),
    ("m", "mirror"),
    ("d", "network input view (debug)"),
    ("e / g / l / c", "NMS extent boxes / sigma circles / labels / scores"),
    ("o", "overlay: keypoints -> organs -> both"),
    ("i", "info overlay (HUD)"),
    ("s", "screenshot (PNG + JSON with poses and settings)"),
    ("r", "start / stop recording (MP4)"),
    ("Tab", "hide / show the control panel"),
    ("F11 / f", "fullscreen (Esc leaves)"),
    ("q", "quit"),
    ("click", "select the person under the mouse (Poses tab)"),
]


class RateMeter:
    """Events per second over a sliding ~1 s window."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._count = 0
        self.value = 0.0

    def tick(self) -> None:
        self._count += 1
        elapsed = time.perf_counter() - self._start
        if elapsed >= 1.0:
            self.value = self._count / elapsed
            self._count = 0
            self._start = time.perf_counter()


class MovingAverage:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def add(self, x: float) -> None:
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x

    def text(self, fmt: str = "{:.1f}") -> str:
        return fmt.format(self.value) if self.value is not None else "-"


class ToolTip:
    """Hover help for a widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=360, padx=6, pady=4).pack()

    def _hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def _enable_windows_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - not Windows / not supported
        pass


def _make_var(kind: type) -> tk.Variable:
    if kind is bool:
        return tk.BooleanVar()
    if kind is int:
        return tk.IntVar()
    if kind is float:
        return tk.DoubleVar()
    return tk.StringVar()


def pose_to_dict(pose, names: list[str]) -> dict[str, Any]:
    return {
        "score": round(float(pose.score), 5),
        "source_view": int(pose.source_view),
        "anchor_index": int(pose.anchor_index),
        "keypoints": [
            {"name": names[k] if k < len(names) else str(k), "x": round(float(pose.keypoints[k, 0]), 2),
             "y": round(float(pose.keypoints[k, 1]), 2), "conf": round(float(pose.keypoints[k, 2]), 4),
             "visibility": VIS_TEXT.get(int(pose.visibility[k]), str(int(pose.visibility[k]))),
             "sigma_px": round(float(pose.sigma_px[k]), 2)}
            for k in range(pose.keypoints.shape[0])
        ],
        **({"organs": [
            {"name": name, "x": round(float(pose.organs[i, 0]), 2), "y": round(float(pose.organs[i, 1]), 2),
             "radius": round(float(pose.organs[i, 2]), 2), "conf": round(float(pose.organs[i, 3]), 4)}
            for i, name in enumerate(ORGAN_ROWS)
        ]} if getattr(pose, "organs", None) is not None else {}),
    }


class TestApp:
    def __init__(self, config: AppConfig, settings: Settings) -> None:
        self.config = config
        self.engine = LiveEngine(config)
        _enable_windows_dpi_awareness()
        self.root = tk.Tk()
        self.root.title(config.display.window_title)
        self.ui_scale = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        self.panel_width = int(PANEL_WIDTH * self.ui_scale)
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_w * 0.88)}x{int(screen_h * 0.85)}+{int(screen_w * 0.04)}+{int(screen_h * 0.04)}")
        self.root.minsize(int(900 * self.ui_scale), int(560 * self.ui_scale))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))
        style.configure("Small.TLabel", foreground="#555")
        style.configure("Warn.TLabel", foreground="#b00")
        style.configure("Treeview", rowheight=int(20 * self.ui_scale))

        self.vars: dict[str, tk.Variable] = {name: _make_var(kind) for name, kind in FIELD_TYPES.items()}
        self.resolution_var = tk.StringVar()
        self.video_path_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.preset_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Starting...")
        self.source_info_var = tk.StringVar(value="-")
        self.network_label_var = tk.StringVar()
        self._suspend = 0
        self._apply_job: str | None = None
        self._source_dirty = False
        self.cameras: list[CameraSource] = []
        self.presets: list[Preset] = []

        # render state
        self._placeholder = False
        self._frame: np.ndarray | None = None
        self._result = None
        self._shown_seq = -1
        self._dirty = True
        self._redetect = False
        self._view_geom = (0, 0, 1.0)  # x offset, y offset, scale of the shown image in the canvas
        self.selected_pose: int | None = None
        self.display_rate = RateMeter()
        self.infer_rate = RateMeter()
        self.pre_ms, self.net_ms, self.post_ms, self.age_ms = (MovingAverage() for _ in range(4))
        self._writer: cv2.VideoWriter | None = None
        self._record_path: Path | None = None
        self._last_table = 0.0
        self._last_stats = 0.0
        self._fullscreen = False
        self._panel_visible = True
        self._error_shown = False
        self._flash_until = 0.0
        self._flash_text = ""
        self._size_key: tuple[str, str] | None = None
        self._proc = None
        self._capture_sample: tuple[float, int] | None = None
        self.capture_fps = 0.0

        self._build()
        self._set_vars(settings)
        self._load_presets()
        self._refresh_cameras(select=settings)
        if settings.source_kind == "camera" and self.cameras:
            camera = self.cameras[self.camera_box.current()]
            settings = settings.updated(camera_index=camera.index, camera_name=camera.name)
        self.root.update_idletasks()
        self._start(settings)
        self.root.bind("<KeyPress>", self._on_key)
        self.root.after(15, self._tick)

    # ================================================================== layout
    def _build(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)
        self.video_frame = ttk.Frame(outer)
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(self.video_frame, background="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._mark_dirty())
        self.canvas.bind("<Expose>", lambda _e: self._mark_dirty())
        self.blitter = make_blitter(self.canvas)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-1>", lambda _e: self.canvas.focus_set(), add="+")
        self.status_bar = ttk.Label(self.video_frame, textvariable=self.status_var, anchor="w", padding=(6, 3))
        self.status_bar.pack(fill="x")

        self.panel = ttk.Frame(outer, width=self.panel_width, padding=(6, 6, 6, 6))
        self.panel.pack(side="right", fill="y")
        self.panel.pack_propagate(False)

        presets = ttk.LabelFrame(self.panel, text="Preset", padding=6)
        presets.pack(fill="x")
        self.preset_box = ttk.Combobox(presets, textvariable=self.preset_var, state="readonly")
        self.preset_box.pack(fill="x")
        self.preset_box.bind("<<ComboboxSelected>>", lambda _e: self._update_preset_tip())
        self.preset_desc = ttk.Label(presets, style="Small.TLabel", wraplength=self._wrap(40), justify="left")
        self.preset_desc.pack(fill="x", pady=(3, 3))
        row = ttk.Frame(presets)
        row.pack(fill="x")
        ttk.Button(row, text="Apply preset", style="Accent.TButton", command=self._apply_preset).pack(side="left")
        ttk.Button(row, text="Save current as...", command=self._save_preset).pack(side="left", padx=4)
        ttk.Button(row, text="Delete", command=self._delete_preset).pack(side="left")

        self.notebook = ttk.Notebook(self.panel)
        self.notebook.pack(fill="both", expand=True, pady=(6, 6))
        self._build_model_tab()
        self._build_detection_tab()
        self._build_source_tab()
        self._build_display_tab()
        self._build_poses_tab()
        self._build_stats_tab()
        self._build_help_tab()

        buttons = ttk.Frame(self.panel)
        buttons.pack(fill="x")
        self.pause_btn = ttk.Button(buttons, text="Pause", width=8, command=self.toggle_pause, takefocus=False)
        self.pause_btn.pack(side="left")
        ttk.Button(buttons, text="Step", width=6, command=self.step, takefocus=False).pack(side="left", padx=(3, 0))
        ttk.Button(buttons, text="Screenshot", command=self.screenshot, takefocus=False).pack(side="left",
                                                                                               padx=(3, 0))
        self.record_btn = ttk.Button(buttons, text="Record", width=8, command=self.toggle_record, takefocus=False)
        self.record_btn.pack(side="left", padx=(3, 0))
        ttk.Button(buttons, text="Quit", width=6, command=self.quit, takefocus=False).pack(side="right")

    def _tab(self, title: str) -> ttk.Frame:
        """A scrollable tab; returns the inner frame."""

        holder = ttk.Frame(self.notebook)
        self.notebook.add(holder, text=title)
        canvas = tk.Canvas(holder, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(4, 6, 4, 6))
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

        def wheel(event) -> None:
            canvas.yview_scroll(int(-event.delta / 120), "units")

        inner.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", wheel))
        inner.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        inner.columnconfigure(1, weight=1)
        return inner

    # ------------------------------------------------------------ widget helpers
    def _wrap(self, margin: int) -> int:
        """Label wrap length inside the panel (``margin`` at 96 dpi)."""

        return self.panel_width - int(margin * self.ui_scale)

    def _group(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=(6, 4, 6, 6))
        frame.pack(fill="x", pady=(0, 6))
        frame.columnconfigure(1, weight=1)
        return frame

    def _slider(self, parent, row: int, text: str, key: str, lo: float, hi: float, *, digits: int = 2,
                tip: str = "") -> None:
        var = self.vars[key]
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)

        def rounded(value: str) -> None:
            var.set(round(float(value), digits))

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, command=rounded)
        scale.grid(row=row, column=1, sticky="we", pady=2)
        step = 10 ** -digits
        spin = ttk.Spinbox(parent, from_=lo, to=hi, increment=step * (5 if digits == 2 else 1), textvariable=var,
                           width=6, format=f"%.{digits}f")
        spin.grid(row=row, column=2, sticky="e", padx=(6, 0), pady=2)
        for widget in (label, scale, spin):
            ToolTip(widget, tip)

    def _spin(self, parent, row: int, text: str, key: str, lo: int, hi: int, *, tip: str = "") -> ttk.Spinbox:
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        spin = ttk.Spinbox(parent, from_=lo, to=hi, increment=1, textvariable=self.vars[key], width=6)
        spin.grid(row=row, column=1, sticky="w", pady=2)
        ToolTip(label, tip)
        ToolTip(spin, tip)
        return spin

    def _check(self, parent, text: str, key: str, *, tip: str = "", row: int | None = None,
               column: int = 0, columnspan: int = 3) -> ttk.Checkbutton:
        check = ttk.Checkbutton(parent, text=text, variable=self.vars[key])
        if row is None:
            check.pack(anchor="w", pady=1)
        else:
            check.grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=1)
        ToolTip(check, tip)
        return check

    def _radios(self, parent, key: str, options: list[tuple[str, str]], *, tip: str = "") -> dict[str, ttk.Radiobutton]:
        frame = ttk.Frame(parent)
        out = {}
        for value, text in options:
            radio = ttk.Radiobutton(frame, text=text, value=value, variable=self.vars[key])
            radio.pack(side="left", padx=(0, 10))
            ToolTip(radio, tip)
            out[value] = radio
        frame.radios = out  # type: ignore[attr-defined]
        return out

    # ------------------------------------------------------------------- tabs
    def _build_model_tab(self) -> None:
        tab = self._tab("Model")
        g = self._group(tab, "Network")
        self.network_box = ttk.Combobox(g, textvariable=self.network_label_var, state="readonly",
                                        values=[self._network_label(n) for n in self.config.networks])
        self.network_box.grid(row=0, column=0, columnspan=3, sticky="we")
        self.network_box.bind("<<ComboboxSelected>>", self._on_network_selected)
        self.network_desc = ttk.Label(g, style="Small.TLabel", wraplength=self._wrap(50), justify="left")
        self.network_desc.grid(row=1, column=0, columnspan=3, sticky="we", pady=(3, 0))

        g = self._group(tab, "Backend")
        radios = self._radios(g, "backend", [(b, BACKEND_LABELS[b]) for b in ("ncnn", "onnxruntime")],
                              tip="ncnn = the deployment runtime (Raspberry Pi). ONNX Runtime runs the exported FP32 "
                                  "ONNX graph on the CPU - the reference to compare the ncnn output with.")
        next(iter(radios.values())).master.grid(row=0, column=0, columnspan=3, sticky="w")
        self.backend_radios = radios

        g = self._group(tab, "Input size (letterbox S x S)")
        self.size_frame = ttk.Frame(g)
        self.size_frame.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(g, text="One set of weights serves all sizes (multi-scale training). Bigger = more accurate "
                          "for small people, slower.", style="Small.TLabel", wraplength=self._wrap(50),
                  justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))

        g = self._group(tab, "View mode")
        radios = self._radios(g, "mode", [("whole", "whole frame (1 pass)"), ("pyramid", "pyramid (3 passes)")],
                              tip="whole = the frame letterboxed into one S x S view (XTX4 default). pyramid = "
                                  "XTX3 ablation: three views of a 4S canvas (whole, centre 1/2, centre 1/4 zoomed), "
                                  "merged with view priority; ~3x slower.")
        next(iter(radios.values())).master.grid(row=0, column=0, columnspan=3, sticky="w")

        g = self._group(tab, "Runtime")
        self._spin(g, 0, "CPU threads", "num_threads", 1, max(1, os.cpu_count() or 4),
                   tip="ncnn / ONNX Runtime intra-op threads. Raspberry Pi 5: 3 threads are optimal.")
        self.fp16_check = self._check(g, "FP16 storage + arithmetic (ncnn)", "fp16", row=1,
                                      tip="ncnn use_fp16_storage / packed / arithmetic. On ARMv8.2 (Pi 5) 2.4x faster "
                                          "than FP32. Weights in the .bin are FP16 or FP32 depending on the export "
                                          "(see Model info).")
        self.vulkan_check = self._check(g, "Vulkan GPU (ncnn)", "use_vulkan", row=2,
                                        tip="Run the ncnn graph on a Vulkan GPU instead of the CPU threads.")
        devices = vulkan_devices()
        ttk.Label(g, text="Vulkan: " + (", ".join(devices) if devices else "no device"), style="Small.TLabel",
                  wraplength=self._wrap(50)).grid(row=3, column=0, columnspan=3, sticky="w")
        self._vulkan_available = bool(devices)

        g = self._group(tab, "Model info")
        self.model_info = tk.Text(g, height=19, wrap="word", font=("Consolas", 8), relief="flat",
                                  background=self.root.cget("background"))
        self.model_info.grid(row=0, column=0, columnspan=3, sticky="we")

    def _build_detection_tab(self) -> None:
        tab = self._tab("Detection")
        g = self._group(tab, "Person detection")
        self._slider(g, 0, "Detection threshold", "conf_threshold", 0.01, 0.95,
                     tip="Minimum person score sigmoid(person logit). Raw head output: manifest default 0.25; the "
                         "tuned deployment threshold with OKS-NMS was ~0.15.")
        self._spin(g, 1, "Max detections", "max_detections", 1, 300,
                   tip="Top-k persons by score (per view in pyramid mode).")

        g = self._group(tab, "Keypoints and visibility")
        self._slider(g, 0, "Present threshold", "present_threshold", 0.0, 1.0,
                     tip="A keypoint is absent (class 0, not drawn) when sigmoid(present) < threshold.")
        self._slider(g, 1, "Visible threshold", "visible_threshold", 0.0, 1.0,
                     tip="A present keypoint is visible (filled dot) when sigmoid(visible) >= threshold, otherwise "
                         "occluded (ring). 0.8 gave the best visibility F1 on validation.")
        self._slider(g, 2, "Drawing threshold", "kpt_threshold", 0.0, 1.0,
                     tip="Only draw keypoints with conf >= threshold (display only).")

        g = self._group(tab, "OKS-NMS (duplicate suppression)")
        self._check(g, "Enable OKS-NMS", "oks_nms", row=0,
                    tip="The one-to-one head is NMS-free; OKS-NMS on the host is a safety net for duplicates "
                        "(+0.5 AP on u_b, fewer duplicates in crowds). Always on in pyramid mode across views.")
        self._slider(g, 1, "OKS threshold", "oks_nms_threshold", 0.05, 1.0,
                     tip="Drop a pose when OKS with an already kept pose >= threshold.")
        self._slider(g, 2, "Scale factor", "oks_nms_scale_factor", 0.5, 4.0,
                     tip="The OKS scale is the extent box of the confident points expanded by this factor "
                         "(1.5 ~ head + torso box). Larger = suppress more.")
        self._slider(g, 3, "NMS keypoint conf", "nms_kpt_threshold", 0.0, 1.0,
                     tip="Keypoints with conf >= this value define the extent box and count in the OKS.")
        self._check(g, "Also suppress on extent-box IoU", "oks_nms_iou", row=4,
                    tip="Additionally drop a pose when the IoU of the expanded extent boxes >= threshold "
                        "(XTX3's merge rule used 0.5).")
        self._slider(g, 5, "IoU threshold", "oks_nms_iou_threshold", 0.05, 1.0)

        g = self._group(tab, "Decode contract (advanced)")
        self._slider(g, 0, "Sigma conf coeff", "sigma_conf_coeff", 0.0, 2.0,
                     tip="Keypoint conf = present * exp(-coeff * sigma_units). 0 = conf is the present "
                         "probability only (training default).")
        self._slider(g, 1, "Sigma floor", "sigma_floor", 0.0, 0.5, digits=3,
                     tip="Added to softplus(log_sigma). Part of the trained contract (0.05) - changing it only "
                         "rescales the shown sigma and the sigma-weighted conf.")
        ttk.Button(tab, text="Reset decode / NMS to manifest defaults", command=self._reset_decode).pack(anchor="w")

    def _build_source_tab(self) -> None:
        tab = self._tab("Source")
        g = self._group(tab, "Source type")
        radios = self._radios(g, "source_kind", [("camera", "Camera"), ("video", "Video file / URL"),
                                                ("image", "Image")])
        next(iter(radios.values())).master.grid(row=0, column=0, columnspan=3, sticky="w")

        g = self._group(tab, "Camera")
        self.camera_box = ttk.Combobox(g, state="readonly")
        self.camera_box.grid(row=0, column=0, columnspan=2, sticky="we")
        self.camera_box.bind("<<ComboboxSelected>>", self._on_camera_selected)
        ttk.Button(g, text="Refresh", command=self._refresh_cameras).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(g, text="Backend").grid(row=1, column=0, sticky="w", pady=2)
        radios = self._radios(g, "windows_backend", [("msmf", "Media Foundation"), ("dshow", "DirectShow")],
                              tip="Windows capture API. DirectShow opens some cameras faster / offers more "
                                  "resolutions.")
        next(iter(radios.values())).master.grid(row=1, column=1, columnspan=2, sticky="w")
        ttk.Label(g, text="Resolution").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Combobox(g, textvariable=self.resolution_var, values=RESOLUTIONS, width=12).grid(
            row=2, column=1, sticky="w", pady=2)
        self._spin(g, 3, "FPS", "camera_fps", 1, 240, tip="Requested camera rate (also the pace of an image source).")
        self._check(g, "MJPG (needed for high FPS at high resolution)", "mjpg", row=4)

        g = self._group(tab, "Video file / stream URL")
        ttk.Entry(g, textvariable=self.video_path_var).grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse...", command=self._browse_video).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(g, text="Files loop and play in real time; rtsp:// and http:// streams work too.",
                  style="Small.TLabel").grid(row=1, column=0, columnspan=3, sticky="w")

        g = self._group(tab, "Image")
        ttk.Entry(g, textvariable=self.image_path_var).grid(row=0, column=0, columnspan=2, sticky="we")
        ttk.Button(g, text="Browse...", command=self._browse_image).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(g, text="The image is fed repeatedly at the FPS above, so every change is visible at once.",
                  style="Small.TLabel", wraplength=self._wrap(50)).grid(row=1, column=0, columnspan=3, sticky="w")

        self.open_btn = ttk.Button(tab, text="Open source", style="Accent.TButton", command=self._open_source)
        self.open_btn.pack(anchor="w", pady=(0, 6))
        self._check(tab, "Mirror image (applied live)", "mirror")
        ttk.Label(tab, textvariable=self.source_info_var, style="Small.TLabel", wraplength=self._wrap(40),
                  justify="left").pack(anchor="w", pady=(6, 0))

    def _build_display_tab(self) -> None:
        tab = self._tab("Display")
        g = self._group(tab, "Show")
        radios = self._radios(g, "overlay_mode", [(m, OVERLAY_TEXT[m]) for m in OVERLAY_MODES],
                              tip="Keypoints (skeleton + points), the organ circles (brain, chest) in the FreeSentry "
                                  "style, or both. Key o cycles.")
        next(iter(radios.values())).master.grid(row=0, column=0, columnspan=3, sticky="w")
        g = self._group(tab, "Organs (FreeSentry style)")
        for i, organ in enumerate(ORGAN_ROWS):
            base = i * 4
            self._check(g, f"{organ.capitalize()} circle", f"{organ}_enabled", row=base,
                        tip="Draw this organ (FreeSentry ai_setup.organs.<organ>.enabled).")
            self._slider(g, base + 1, "  size multiplier", f"{organ}_size_multiplier", 0.1, 3.0,
                         tip="Radius multiplier (sizeMultiplier).")
            self._slider(g, base + 2, "  confidence threshold", f"{organ}_conf_threshold", 0.0, 1.0,
                         tip="Below it the circle is drawn thin (confidenceThreshold). The chest default 0.143 is "
                             "the server's 0.20 * 5/7: XTX4 divides the chest confidence by 7, not 5.")
            self._spin(g, base + 3, "  minimum radius", f"{organ}_min_radius", 0, 200,
                       tip="Smaller circles are drawn thin (minimumRadius, frame pixels).")
        g = self._group(tab, "Overlay")
        self._check(g, "Skeleton lines", "show_skeleton", row=0)
        self._check(g, "Keypoints (filled = visible, ring = occluded)", "show_points", row=1)
        self._check(g, "Draw occluded keypoints", "show_occluded", row=2)
        self._check(g, "Person score", "show_scores", row=3)
        self._check(g, "Person number (#)", "show_ids", row=4)
        self._check(g, "Keypoint names", "show_labels", row=5)
        self._check(g, "Sigma circles (RLE localisation uncertainty)", "show_sigma", row=6,
                    tip="Radius = predicted sigma in frame pixels: how unsure the network is about the position.")
        self._check(g, "NMS extent boxes", "show_extent", row=7,
                    tip="The expanded extent box of the confident points that OKS-NMS uses as the person scale "
                        "(XTX4 itself predicts no boxes).")
        g = self._group(tab, "Colours and sizes")
        ttk.Label(g, text="Colour by").grid(row=0, column=0, sticky="w")
        radios = self._radios(g, "color_mode", [("side", "body side (L green / R orange)"), ("person", "person")])
        next(iter(radios.values())).master.grid(row=0, column=1, columnspan=2, sticky="w")
        self._spin(g, 1, "Point radius", "point_radius", 1, 20)
        self._spin(g, 2, "Line thickness", "line_thickness", 1, 12)
        g = self._group(tab, "Panels")
        self._check(g, "Info overlay (HUD) in the image", "show_hud", row=0)
        self._check(g, "Network input view(s) with keypoints (debug)", "show_debug_view", row=1,
                    tip="Shows the letterboxed S x S view(s) the network sees, with the keypoints in network "
                        "pixels.")

    def _build_poses_tab(self) -> None:
        tab = self._tab("Poses")
        ttk.Label(tab, text="Persons in the shown frame (click a row or a person in the video):",
                  style="Small.TLabel").pack(anchor="w")
        cols = ("id", "score", "view", "vis", "occ", "abs", "anchor")
        self.pose_tree = ttk.Treeview(tab, columns=cols, show="headings", height=8, selectmode="browse")
        for col, text, width in zip(cols, ("#", "score", "view", "visible", "occl.", "absent", "anchor"),
                                    (30, 60, 40, 55, 50, 50, 60)):
            self.pose_tree.heading(col, text=text)
            self.pose_tree.column(col, width=width, anchor="center")
        self.pose_tree.pack(fill="x")
        self.pose_tree.bind("<<TreeviewSelect>>", self._on_pose_selected)
        ttk.Label(tab, text="Keypoints of the selected person:", style="Small.TLabel").pack(anchor="w", pady=(8, 0))
        cols = ("name", "x", "y", "conf", "vis", "sigma")
        self.kpt_tree = ttk.Treeview(tab, columns=cols, show="headings", height=11, selectmode="none")
        for col, text, width in zip(cols, ("keypoint / organ", "x", "y", "conf", "visibility", "sigma / r px"),
                                    (95, 55, 55, 50, 65, 60)):
            self.kpt_tree.heading(col, text=text)
            self.kpt_tree.column(col, width=width, anchor="center")
        self.kpt_tree.pack(fill="x")
        self.pose_summary = ttk.Label(tab, style="Small.TLabel", wraplength=self._wrap(40), justify="left")
        self.pose_summary.pack(anchor="w", pady=(6, 0))

    def _build_stats_tab(self) -> None:
        tab = self._tab("Stats")
        self.stats_text = tk.Text(tab, height=26, wrap="none", font=("Consolas", 8), relief="flat")
        self.stats_text.pack(fill="both", expand=True)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Reset statistics", command=self._reset_stats).pack(side="left")
        ttk.Button(row, text="Copy", command=self._copy_stats).pack(side="left", padx=(4, 0))

    def _build_help_tab(self) -> None:
        tab = self._tab("Help")
        for row, (key, text) in enumerate(SHORTCUTS):
            ttk.Label(tab, text=key, font=("Consolas", 9, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 10))
            ttk.Label(tab, text=text, wraplength=self._wrap(130), justify="left").grid(row=row, column=1, sticky="w")
        ttk.Label(tab, text="Every option applies immediately, except the Source tab (press 'Open source'). "
                            "While paused, the frozen frame is re-detected after every change. Screenshots go to "
                            "live_test/screenshots, recordings to live_test/recordings.",
                  style="Small.TLabel", wraplength=self._wrap(40), justify="left").grid(
            row=len(SHORTCUTS), column=0, columnspan=2, sticky="w", pady=(10, 0))

    # =========================================================== vars <-> settings
    def _network_label(self, network) -> str:
        return network.label if network.available_sizes() else f"{network.label}  [model files missing]"

    def _set_vars(self, settings: Settings) -> None:
        self._suspend += 1
        try:
            for key, value in settings.to_dict().items():
                self.vars[key].set(value)
            self.resolution_var.set(f"{settings.width}x{settings.height}")
            if settings.source_kind == "video":
                self.video_path_var.set(settings.source_path)
            elif settings.source_kind == "image":
                self.image_path_var.set(settings.source_path)
            ids = [n.id for n in self.config.networks]
            if settings.network_id in ids:
                self.network_box.current(ids.index(settings.network_id))
        finally:
            self._suspend -= 1
        self._refresh_model_widgets(settings)

    def _collect(self, *, include_source: bool) -> Settings:
        """Settings from the widgets; source fields only when ``include_source``."""

        current = self.engine.settings or Settings()
        values: dict[str, Any] = {}
        for key, var in self.vars.items():
            if key in SOURCE_KEYS and not include_source:
                continue
            try:
                values[key] = var.get()
            except (tk.TclError, ValueError):
                pass  # half-typed number in a spinbox: keep the current value
        if include_source:
            text = self.resolution_var.get().lower().replace(" ", "")
            width, _, height = text.partition("x")
            if not (width.isdigit() and height.isdigit()):
                raise ValueError(f"Resolution must look like 1280x720, got {self.resolution_var.get()!r}")
            values["width"], values["height"] = int(width), int(height)
            kind = values.get("source_kind", current.source_kind)
            if kind == "video":
                values["source_path"] = self.video_path_var.get().strip()
            elif kind == "image":
                values["source_path"] = self.image_path_var.get().strip()
            index = self.camera_box.current()
            if kind == "camera" and 0 <= index < len(self.cameras):
                values["camera_index"] = self.cameras[index].index
                values["camera_name"] = self.cameras[index].name
        return validate_for_config(from_dict(values, current), self.config)

    def _install_traces(self) -> None:
        for key, var in self.vars.items():
            var.trace_add("write", lambda *_a, k=key: self._on_var(k))
        for var in (self.resolution_var, self.video_path_var, self.image_path_var):
            var.trace_add("write", lambda *_a: self._on_var("source_path"))

    def _on_var(self, key: str) -> None:
        if self._suspend:
            return
        if key == "windows_backend":
            self.root.after_idle(self._refresh_cameras)
        if key in SOURCE_KEYS:
            self._source_dirty = True
            self.open_btn.configure(text="Open source  (changed - press to apply)")
            return
        if key in DISPLAY_KEYS:
            # Drawing only: no engine round trip, just redraw.
            self._schedule_apply(delay=60)
            return
        self._schedule_apply()

    def _schedule_apply(self, delay: int = 150) -> None:
        if self._apply_job is not None:
            self.root.after_cancel(self._apply_job)
        self._apply_job = self.root.after(delay, self._apply)

    def _apply(self, *, include_source: bool = False) -> None:
        self._apply_job = None
        try:
            new = self._collect(include_source=include_source)
        except ValueError as exc:
            messagebox.showerror("XTX4 test app", str(exc), parent=self.root)
            return
        old = self.engine.settings
        only_display = old is not None and not include_source and all(
            getattr(new, k) == getattr(old, k) for k in FIELD_TYPES if k not in DISPLAY_KEYS)
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            messages = self.engine.apply(new, reopen_source=True if include_source else None)
        except (RunnerError, ConfigError, OSError) as exc:
            messagebox.showerror("XTX4 test app", f"Cannot load the graph:\n{exc}", parent=self.root)
            if old is not None:
                self._set_vars(old)
            return
        finally:
            self.root.config(cursor="")
        if new.differs(old, RUNNER_KEYS) or include_source:
            self._reset_meters()
        self._set_vars(self.engine.settings)
        if include_source:
            self._source_dirty = False
            self.open_btn.configure(text="Open source")
            self._update_source_info()
            self._frame = self._result = None
            self._shown_seq = -1
        for message in messages:
            self.flash(message)
        if self.engine.paused and not only_display:
            self._redetect = True
        self._mark_dirty()
        save_last(self.engine.settings)

    def _start(self, settings: Settings) -> None:
        """First apply: the requested settings, else config defaults."""

        for attempt in (settings, config_defaults(self.config)):
            try:
                messages = self.engine.apply(validate_for_config(attempt, self.config))
                break
            except (RunnerError, ConfigError, OSError) as exc:
                messagebox.showerror("XTX4 test app", f"Cannot load {attempt.network_id} @{attempt.size} "
                                                      f"({attempt.backend}):\n{exc}", parent=self.root)
        else:
            raise SystemExit(2)
        self._set_vars(self.engine.settings)
        self._install_traces()
        self._update_source_info()
        for message in messages:
            self.flash(message)

    # ======================================================= model tab helpers
    def _on_network_selected(self, _event=None) -> None:
        index = self.network_box.current()
        if index < 0:
            return
        network = self.config.networks[index]
        if not network.available_sizes():
            messagebox.showwarning("XTX4 test app", f"{network.label}: model files missing"
                                   + (f"\n{network.manifest_error}" if network.manifest_error else ""),
                                   parent=self.root)
            self._set_vars(self.engine.settings)
            return
        self.vars["network_id"].set(network.id)

    def _refresh_model_widgets(self, settings: Settings) -> None:
        network = self.config.network(settings.network_id) if self.config.has_network(settings.network_id) else None
        if network is None:
            return
        self.network_desc.configure(text=network.description or network.id)
        if self._size_key != (network.id, settings.backend):
            self._size_key = (network.id, settings.backend)
            for widget in self.size_frame.winfo_children():
                widget.destroy()
            available = set(network.available_sizes(settings.backend))
            for index, size in enumerate(network.sizes):
                radio = ttk.Radiobutton(self.size_frame, text=f"{size}  [{index + 1}]", value=size,
                                        variable=self.vars["size"])
                radio.pack(side="left", padx=(0, 10))
                if size not in available:
                    radio.state(["disabled"])
        onnx_ok = bool(network.available_sizes("onnxruntime")) and backend_installed("onnxruntime")
        self.backend_radios["onnxruntime"].state(["!disabled"] if onnx_ok else ["disabled"])
        ncnn_backend = settings.backend == "ncnn"
        self.fp16_check.state(["!disabled"] if ncnn_backend else ["disabled"])
        self.vulkan_check.state(["!disabled"] if ncnn_backend and self._vulkan_available else ["disabled"])
        self._update_model_info(network, settings)

    def _update_model_info(self, network, settings: Settings) -> None:
        runner = self.engine.runner
        lines = [f"network id   {network.id}"]
        manifest = network.manifest_data or {}
        if manifest:
            spec = manifest.get("variant_spec", {})
            lines += [
                f"model id     {manifest.get('model_id', '?')}",
                f"variant      {spec.get('name', '?')}  widths {spec.get('widths')}  depths {spec.get('depths')}",
                f"block / act  {spec.get('block')} / {spec.get('act')}  levels {manifest.get('levels')}  "
                f"strides {manifest.get('strides')}",
                f"precision    {json.dumps(manifest.get('precision'))}",
                f"checkpoint   sha256 {str(manifest.get('checkpoint_sha256', ''))[:16]}...",
                f"exported     {manifest.get('created', '?')}",
            ]
            parity = manifest.get("parity", {}).get("per_size", {}).get(str(settings.size))
            if parity:
                lines.append(f"parity @{settings.size}  ncnn xy p95 {parity.get('ncnn_xy_p95_px', 0):.3f} px, "
                             f"max {parity.get('ncnn_xy_max_px', 0):.3f} px; onnx xy max "
                             f"{parity.get('onnx_xy_max_px', 0):.4f} px")
        elif network.manifest_error:
            lines.append(f"manifest     {network.manifest_error}")
        if runner is not None:
            lines.append(f"running      {runner.describe()}")
            lines.append(f"anchors      N = {getattr(runner, 'num_anchors', '?')} @ {runner.size}")
            for attr in ("param_path", "bin_path", "onnx_path"):
                path = getattr(runner, attr, None)
                if path is not None:
                    size_mb = Path(path).stat().st_size / 2**20 if Path(path).is_file() else 0
                    lines.append(f"{attr[:-5]:<12} {path}  ({size_mb:.1f} MB)")
        self.model_info.configure(state="normal")
        self.model_info.delete("1.0", "end")
        self.model_info.insert("1.0", "\n".join(lines))
        self.model_info.configure(state="disabled")

    def _reset_decode(self) -> None:
        current = self._collect(include_source=False)
        defaults = manifest_decode(self.config, current.network_id)
        self._suspend += 1
        try:
            for key, value in from_dict(defaults, current).to_dict().items():
                if key in defaults or key == "oks_nms_iou":
                    self.vars[key].set(value)
        finally:
            self._suspend -= 1
        self._apply()

    # ============================================================ presets
    def _load_presets(self, select: str | None = None) -> None:
        self.presets = all_presets(self.config)
        names = [p.name if p.builtin else f"{p.name}  (user)" for p in self.presets]
        self.preset_box.configure(values=names)
        if names:
            index = next((i for i, p in enumerate(self.presets) if p.name == select), 0)
            self.preset_box.current(index)
        self._update_preset_tip()

    def _selected_preset(self) -> Preset | None:
        index = self.preset_box.current()
        return self.presets[index] if 0 <= index < len(self.presets) else None

    def _update_preset_tip(self) -> None:
        preset = self._selected_preset()
        self.preset_desc.configure(text=(preset.description or "") if preset else "")

    def _apply_preset(self) -> None:
        preset = self._selected_preset()
        if preset is None:
            return
        current = self._collect(include_source=False)
        self._set_vars(apply_preset(current, preset, self.config))
        self._apply()
        self.flash(f"Preset: {preset.name}")

    def _save_preset(self) -> None:
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if any(p.name == name and p.builtin for p in self.presets):
            messagebox.showerror("XTX4 test app", "A built-in preset has this name.", parent=self.root)
            return
        user = [p for p in self.presets if not p.builtin and p.name != name]
        settings = self._collect(include_source=False)
        user.append(Preset(name=name, settings=settings.preset_dict(),
                           description=f"Saved {time.strftime('%Y-%m-%d %H:%M')}", builtin=False))
        save_user_presets(user)
        self._load_presets(select=name)
        self.flash(f"Saved preset '{name}'")

    def _delete_preset(self) -> None:
        preset = self._selected_preset()
        if preset is None:
            return
        if preset.builtin:
            messagebox.showinfo("XTX4 test app", "Built-in presets live in config.json and cannot be deleted here.",
                                parent=self.root)
            return
        if not messagebox.askyesno("XTX4 test app", f"Delete preset '{preset.name}'?", parent=self.root):
            return
        save_user_presets([p for p in self.presets if not p.builtin and p.name != preset.name])
        self._load_presets()

    # ============================================================ source tab
    def _refresh_cameras(self, select: Settings | None = None) -> None:
        backend = self.vars["windows_backend"].get() or self.config.camera.windows_backend
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.cameras = list_cameras(windows_backend=backend, max_probe_index=self.config.camera.max_probe_index)
        finally:
            self.root.config(cursor="")
        self.camera_box.configure(values=[c.describe() for c in self.cameras] or ["(no camera found)"])
        want = select or self.engine.settings
        index = 0
        if want is not None:
            for i, cam in enumerate(self.cameras):
                if cam.name == want.camera_name and cam.index == want.camera_index:
                    index = i
                    break
            else:
                index = next((i for i, c in enumerate(self.cameras) if c.index == want.camera_index), 0)
        self.camera_box.current(index)

    def _on_camera_selected(self, _event=None) -> None:
        self._suspend += 1
        try:
            self.vars["source_kind"].set("camera")
        finally:
            self._suspend -= 1
        self._on_var("camera_index")

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Select a video file", filetypes=[
            ("Video", "*.mp4 *.avi *.mkv *.mov *.webm *.m4v"), ("All files", "*.*")])
        if path:
            self.video_path_var.set(path)
            self.vars["source_kind"].set("video")

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Select an image", filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.image_path_var.set(path)
            self.vars["source_kind"].set("image")

    def _open_source(self) -> None:
        self._apply(include_source=True)
        if self.engine.source_error:
            messagebox.showerror("XTX4 test app", f"Cannot open the source:\n{self.engine.source_error}",
                                 parent=self.root)

    def _update_source_info(self) -> None:
        if self.engine.source_error:
            self.source_info_var.set(f"Source error: {self.engine.source_error}")
        else:
            self.source_info_var.set(f"Open: {self.engine.describe_source()}")

    # ============================================================== render loop
    def _mark_dirty(self) -> None:
        self._dirty = True

    def _reset_meters(self) -> None:
        self.pre_ms, self.net_ms, self.post_ms, self.age_ms = (MovingAverage() for _ in range(4))

    def flash(self, text: str, seconds: float = 3.0) -> None:
        logger.info("%s", text)
        self._flash_text = text
        self._flash_until = time.perf_counter() + seconds

    def _tick(self) -> None:
        try:
            self._update()
        except Exception:  # noqa: BLE001 - keep the GUI alive, log the problem
            logger.exception("Render loop error")
        self.root.after(10, self._tick)

    def _update(self) -> None:
        new_frame = False
        item = self.engine.latest()
        if item is not None and item.seq != self._shown_seq:
            self._shown_seq = item.seq
            self._frame, self._result = item.frame, item.result
            t = item.result.timings
            self.infer_rate.tick()
            self.pre_ms.add(t.preprocess_ms)
            self.net_ms.add(t.inference_ms)
            self.post_ms.add(t.postprocess_ms)
            self.age_ms.add(item.frame_age_ms)
            new_frame = self._dirty = True
        if self._redetect and self._frame is not None:
            self._redetect = False
            self._result = self.engine.redetect(self._frame)
            self._dirty = True
        if self.engine.error is not None and not self._error_shown:
            self._error_shown = True
            messagebox.showerror("XTX4 test app", f"Pipeline failed:\n{self.engine.error!r}", parent=self.root)
        if self._dirty and self._frame is not None:
            self._dirty = False
            canvas = self._render()
            if new_frame and self._writer is not None:
                self._write_video(canvas)
            self._show(canvas)
        elif self._frame is None:
            self._show_placeholder()
        now = time.perf_counter()
        if now - self._last_table > 0.25:
            self._last_table = now
            self._update_capture_fps()
            self._update_status()
            self._update_pose_tables()
        if now - self._last_stats > 1.0:
            self._last_stats = now
            self._update_stats()

    def _draw_options(self, settings: Settings) -> DrawOptions:
        return options_from_settings(settings, highlight=self.selected_pose)

    def _display_settings(self) -> Settings:
        """Engine settings with the display fields straight from the widgets (no engine round trip)."""

        settings = self.engine.settings
        values = {}
        for key in DISPLAY_KEYS:
            try:
                values[key] = self.vars[key].get()
            except (tk.TclError, ValueError):
                pass
        return from_dict(values, settings)

    def _render(self) -> np.ndarray:
        settings = self._display_settings()
        canvas = self._frame.copy()
        result = self._result
        network = self.engine.network
        if result is not None and network is not None:
            draw_poses(canvas, result.poses, network.skeleton, self._draw_options(settings))
            if settings.show_debug_view and result.prepared is not None:
                net_points: dict[int, list[np.ndarray]] = {}
                for pose in result.poses:
                    inv = invert_affine(result.prepared.transforms[pose.source_view])
                    shown = pose.keypoints[:, 2] >= settings.kpt_threshold
                    net_points.setdefault(pose.source_view, []).append(
                        apply_affine_points(inv, pose.keypoints[shown, :2].astype(np.float64)))
                draw_debug_views(canvas, result.prepared.views, poses_net=net_points)
        if settings.show_hud:
            draw_panel(canvas, self._hud_lines(), corner="tl")
        if time.perf_counter() < self._flash_until:
            draw_panel(canvas, [self._flash_text], corner="bl")
            self._dirty = True  # keep redrawing until the message expires
        if self._writer is not None:
            cv2.circle(canvas, (canvas.shape[1] - 24, 24), 9, (0, 0, 230), -1, cv2.LINE_AA)
        return canvas

    def _show(self, frame: np.ndarray) -> None:
        if self._placeholder:
            self.canvas.delete("placeholder")
            self._placeholder = False
        self._view_geom = self.blitter.show(frame)
        self.display_rate.tick()

    def _show_placeholder(self) -> None:
        if not self._placeholder or self._dirty:
            self.canvas.delete("all")
            self.blitter.clear()
            self._placeholder = True
            text = (f"No source: {self.engine.source_error}" if self.engine.source_error
                    else "Waiting for frames...")
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text=text,
                                    fill="#ccc", font=("Segoe UI", 12), tags="placeholder",
                                    width=max(200, self.canvas.winfo_width() - 80))
            self._dirty = False

    def _hud_lines(self) -> list[str]:
        settings = self.engine.settings
        runner = self.engine.runner
        result = self._result
        reader = self.engine.reader
        stats = self.engine.stats
        people = len(result.poses) if result is not None else 0
        candidates = result.raw_candidates if result is not None else 0
        backend = ("ONNX Runtime" if runner.backend == "onnxruntime" else
                   "Vulkan" if runner.vulkan_active else f"CPU x{runner.num_threads}")
        lines = [
            f"{settings.network_id} @{runner.size} {runner.mode} | {runner.backend} {backend} | "
            f"fp16 {'on' if runner.fp16 else 'off'}",
            f"source {self.capture_fps:4.1f} | infer {self.infer_rate.value:4.1f} | "
            f"display {self.display_rate.value:4.1f} fps | dropped {stats.dropped if stats else 0}",
            f"pre {self.pre_ms.text()} | net {self.net_ms.text()} | post {self.post_ms.text()} ms | "
            f"age {self.age_ms.text()} ms",
            f"people {people} (cand {candidates}) | conf >= {settings.conf_threshold:.2f} | "
            f"NMS {'on' if settings.oks_nms else 'off'}",
            f"show {OVERLAY_TEXT.get(settings.overlay_mode, settings.overlay_mode).lower()} | "
            f"organs from {getattr(runner, 'organ_source', 'host')}",
        ]
        if self.engine.paused:
            lines.append("PAUSED")
        if reader is not None and getattr(reader, "status", "ok") != "ok":
            lines.append(f"camera: {reader.status}")
        return lines

    def _update_status(self) -> None:
        runner = self.engine.runner
        if runner is None:
            return
        result = self._result
        people = len(result.poses) if result is not None else 0
        paused = "PAUSED | " if self.engine.paused else ""
        rec = f" | REC {self._record_path.name}" if self._writer is not None and self._record_path else ""
        self.status_var.set(
            f"{paused}{runner.describe()} | source {self.capture_fps:.1f} fps | "
            f"inference {self.infer_rate.value:.1f} fps | net {self.net_ms.text()} ms | "
            f"total {self._total_ms()} ms | people {people}{rec}")

    def _update_capture_fps(self) -> None:
        """Source rate: the camera reader's meter, else the pipeline's capture count (image source)."""

        reader, stats = self.engine.reader, self.engine.stats
        if reader is not None and reader.fps > 0:
            self.capture_fps = reader.fps
            return
        if stats is None:
            self.capture_fps = 0.0
            return
        now = time.perf_counter()
        sample = self._capture_sample
        if sample is not None and now - sample[0] >= 1.0:
            self.capture_fps = max(0.0, (stats.captured - sample[1]) / (now - sample[0]))
        if sample is None or now - sample[0] >= 1.0:
            self._capture_sample = (now, stats.captured)

    def _total_ms(self) -> str:
        parts = [m.value for m in (self.pre_ms, self.net_ms, self.post_ms)]
        return f"{sum(parts):.1f}" if all(p is not None for p in parts) else "-"

    # ============================================================== poses tab
    def _update_pose_tables(self) -> None:
        result = self._result
        poses = result.poses if result is not None else []
        if self.selected_pose is not None and self.selected_pose >= len(poses):
            self.selected_pose = None
        rows = self.pose_tree.get_children()
        for index, pose in enumerate(poses):
            vis = pose.visibility
            values = (index, f"{pose.score:.3f}", pose.source_view, int((vis == 2).sum()), int((vis == 1).sum()),
                      int((vis == 0).sum()), pose.anchor_index)
            iid = str(index)
            if iid in rows:
                self.pose_tree.item(iid, values=values)
            else:
                self.pose_tree.insert("", "end", iid=iid, values=values)
        for iid in rows:
            if int(iid) >= len(poses):
                self.pose_tree.delete(iid)
        if self.selected_pose is not None and str(self.selected_pose) not in self.pose_tree.selection():
            self._suspend += 1
            self.pose_tree.selection_set(str(self.selected_pose))
            self._suspend -= 1

        for iid in self.kpt_tree.get_children():
            self.kpt_tree.delete(iid)
        network = self.engine.network
        if self.selected_pose is not None and network is not None:
            pose = poses[self.selected_pose]
            for k in range(pose.keypoints.shape[0]):
                x, y, conf = pose.keypoints[k]
                self.kpt_tree.insert("", "end", values=(
                    network.skeleton.keypoints[k], f"{x:.1f}", f"{y:.1f}", f"{conf:.3f}",
                    VIS_TEXT.get(int(pose.visibility[k]), "?"), f"{pose.sigma_px[k]:.1f}"))
            if pose.organs is not None:
                for i, organ in enumerate(ORGAN_ROWS):
                    x, y, radius, conf = pose.organs[i]
                    self.kpt_tree.insert("", "end", values=(
                        organ, f"{x:.1f}", f"{y:.1f}", f"{conf:.3f}", "organ", f"r {radius:.1f}"))
        if result is not None:
            self.pose_summary.configure(
                text=f"{len(poses)} persons shown, {result.raw_candidates} anchors above the detection threshold "
                     f"(before top-k / NMS).")

    def _on_pose_selected(self, _event=None) -> None:
        if self._suspend:
            return
        selection = self.pose_tree.selection()
        self.selected_pose = int(selection[0]) if selection else None
        self._last_table = 0.0
        self._mark_dirty()

    def _on_click(self, event) -> None:
        if self._result is None:
            return
        x0, y0, scale = self._view_geom
        fx, fy = (event.x - x0) / scale, (event.y - y0) / scale
        best, best_d = None, 40.0 / max(scale, 1e-6)
        for index, pose in enumerate(self._result.poses):
            pts = pose.keypoints[pose.visibility > 0, :2]
            if pose.organs is not None:  # the organ centres are clickable too (organs-only overlay)
                pts = np.concatenate([pts, pose.organs[:, :2]], axis=0)
            if pts.size == 0:
                continue
            d = float(np.min(np.hypot(pts[:, 0] - fx, pts[:, 1] - fy)))
            if d < best_d:
                best, best_d = index, d
        self.selected_pose = best
        if best is not None:
            self.notebook.select(4)
        self._last_table = 0.0
        self._mark_dirty()

    # ============================================================== stats tab
    def _update_stats(self) -> None:
        stats = self.engine.stats
        runner = self.engine.runner
        lines = []
        if runner is not None:
            lines.append(runner.describe())
        lines.append(f"source: {self.engine.describe_source()}")
        if stats is not None:
            lines += ["", stats.report()]
        try:
            import psutil

            if self._proc is None:
                self._proc = psutil.Process(os.getpid())
            proc = self._proc
            lines += ["", f"process RSS {proc.memory_info().rss / 2**20:.0f} MB, CPU {proc.cpu_percent():.0f} % "
                          f"(100 % = one core), threads {proc.num_threads()}"]
        except ImportError:
            pass
        text = "\n".join(lines)
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.configure(state="disabled")

    def _reset_stats(self) -> None:
        self.engine.reset_stats()
        self._reset_meters()
        self._last_stats = 0.0

    def _copy_stats(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.stats_text.get("1.0", "end"))
        self.flash("Statistics copied")

    # ============================================================= actions
    def toggle_pause(self) -> None:
        paused = not self.engine.paused
        self.engine.set_paused(paused)
        self.pause_btn.configure(text="Resume" if paused else "Pause")
        self._mark_dirty()

    def step(self) -> None:
        if not self.engine.paused:
            self.toggle_pause()
        self.engine.step()

    def _stem(self) -> str:
        settings = self.engine.settings
        return f"{settings.network_id}_{settings.size}_{settings.backend}_{time.strftime('%Y%m%d-%H%M%S')}"

    def screenshot(self) -> None:
        if self._frame is None:
            return
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self._stem()
        png, js = SCREENSHOT_DIR / f"{stem}.png", SCREENSHOT_DIR / f"{stem}.json"
        cv2.imwrite(str(png), self._render())
        names = self.engine.network.skeleton.keypoints if self.engine.network else []
        result = self._result
        payload = {
            "image": png.name,
            "frame_size": [int(self._frame.shape[1]), int(self._frame.shape[0])],
            "runner": self.engine.runner.describe() if self.engine.runner else None,
            "settings": self.engine.settings.to_dict(),
            "timings_ms": ({"preprocess": result.timings.preprocess_ms, "inference": result.timings.inference_ms,
                            "postprocess": result.timings.postprocess_ms} if result is not None else None),
            "raw_candidates": result.raw_candidates if result is not None else 0,
            "poses": [pose_to_dict(p, names) for p in (result.poses if result is not None else [])],
        }
        js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.flash(f"Saved {png.name} (+ .json)")

    def toggle_record(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.record_btn.configure(text="Record")
            self.flash(f"Recording saved: {self._record_path}")
            return
        if self._frame is None:
            return
        RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        self._record_path = RECORDING_DIR / f"{self._stem()}.mp4"
        h, w = self._frame.shape[:2]
        fps = self.infer_rate.value or float(self.engine.settings.camera_fps)
        writer = cv2.VideoWriter(str(self._record_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, round(fps)),
                                 (w, h))
        if not writer.isOpened():
            messagebox.showerror("XTX4 test app", f"Cannot write {self._record_path}", parent=self.root)
            return
        self._writer = writer
        self.record_btn.configure(text="Stop rec.")
        self.flash(f"Recording at {round(fps)} fps -> {self._record_path.name}")

    def _write_video(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        if self._frame is not None and (w, h) == (self._frame.shape[1], self._frame.shape[0]):
            self._writer.write(canvas)

    def toggle_panel(self) -> None:
        self._panel_visible = not self._panel_visible
        if self._panel_visible:
            self.panel.pack(side="right", fill="y")
        else:
            self.panel.pack_forget()
        self._mark_dirty()

    def set_fullscreen(self, on: bool) -> None:
        self._fullscreen = on
        self.root.attributes("-fullscreen", on)
        if on and self._panel_visible:
            self.toggle_panel()
        elif not on and not self._panel_visible:
            self.toggle_panel()
        self._mark_dirty()

    def _nudge(self, key: str, delta: float, lo: float, hi: float) -> None:
        value = round(min(hi, max(lo, float(self.vars[key].get()) + delta)), 2)
        self.vars[key].set(value)
        self.flash(f"{key} = {value:.2f}")

    def _toggle(self, key: str, text: str) -> None:
        value = not bool(self.vars[key].get())
        self.vars[key].set(value)
        self.flash(f"{text} {'on' if value else 'off'}")

    def _on_key(self, event) -> None:
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Spinbox, tk.Text)):
            if not (isinstance(widget, ttk.Combobox) and "readonly" in str(widget.cget("state"))):
                return
        key, ch = event.keysym, event.char
        if key == "space" and isinstance(widget, (ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
            return
        settings = self.engine.settings
        if key == "space":
            self.toggle_pause()
        elif ch == ".":
            self.step()
        elif ch in ("+", "="):
            self._nudge("conf_threshold", 0.05, 0.01, 0.95)
        elif ch in ("-", "_"):
            self._nudge("conf_threshold", -0.05, 0.01, 0.95)
        elif ch == "]":
            self._nudge("kpt_threshold", 0.05, 0.0, 1.0)
        elif ch == "[":
            self._nudge("kpt_threshold", -0.05, 0.0, 1.0)
        elif ch.isdigit() and ch != "0" and self.engine.network is not None:
            sizes = self.engine.network.sizes
            if int(ch) - 1 < len(sizes):
                self.vars["size"].set(sizes[int(ch) - 1])
        elif ch == "b":
            other = "onnxruntime" if settings.backend == "ncnn" else "ncnn"
            if self.backend_radios[other].instate(["!disabled"]):
                self.vars["backend"].set(other)
        elif ch == "w":
            self.vars["mode"].set("pyramid" if settings.mode == "whole" else "whole")
        elif ch == "n":
            self._toggle("oks_nms", "OKS-NMS")
        elif ch == "m":
            self._toggle("mirror", "Mirror")
        elif ch == "d":
            self._toggle("show_debug_view", "Network input view")
        elif ch == "e":
            self._toggle("show_extent", "NMS extent boxes")
        elif ch == "g":
            self._toggle("show_sigma", "Sigma circles")
        elif ch == "l":
            self._toggle("show_labels", "Keypoint names")
        elif ch == "c":
            self._toggle("show_scores", "Scores")
        elif ch == "o":
            current = str(self.vars["overlay_mode"].get())
            nxt = OVERLAY_MODES[(OVERLAY_MODES.index(current) + 1) % len(OVERLAY_MODES)] \
                if current in OVERLAY_MODES else OVERLAY_MODES[0]
            self.vars["overlay_mode"].set(nxt)
            self.flash(f"Show {OVERLAY_TEXT[nxt].lower()}")
        elif ch == "i":
            self._toggle("show_hud", "HUD")
        elif ch == "s":
            self.screenshot()
        elif ch == "r":
            self.toggle_record()
        elif key == "Tab":
            self.toggle_panel()
            return "break"  # type: ignore[return-value]
        elif key == "F11" or ch == "f":
            self.set_fullscreen(not self._fullscreen)
        elif key == "Escape" and self._fullscreen:
            self.set_fullscreen(False)
        elif ch == "q":
            self.quit()

    # ================================================================ life
    def run(self) -> None:
        self.root.mainloop()

    def quit(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        settings = self.engine.settings
        if self.engine.stats is not None:
            logger.info("Pipeline statistics:\n%s", self.engine.stats.report())
        self.engine.close()
        if settings is not None:
            save_last(settings)
        self.root.destroy()


def run_gui(config: AppConfig, settings: Settings) -> int:
    app = TestApp(config, settings)
    app.run()
    return 0

