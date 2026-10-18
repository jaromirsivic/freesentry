"""Everything the test app lets the user change, in one flat dataclass.

``Settings`` is immutable in practice: the GUI builds a new instance (``updated``)
and hands it to :class:`livetest.engine.LiveEngine.apply`, which works out what
has to happen -- reload the graph (``RUNNER_KEYS``), set decode attributes on the
running graph (``DECODE_ATTRS``), reopen the source (``SOURCE_KEYS``) or nothing
(per-call detection and display values are read on every frame).

Start-up values come from ``config.json`` (+ the network's manifest ``decode``
block), then the last session (``.last_settings.json``), then a preset and the
command line. Presets are partial settings: built-in ones live in ``config.json``,
user ones in ``user_presets.json``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .config import AppConfig, ConfigError, Preset

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
LAST_SETTINGS_PATH = APP_DIR / ".last_settings.json"
USER_PRESETS_PATH = APP_DIR / "user_presets.json"

BACKENDS = ("ncnn", "onnxruntime")
SOURCE_KINDS = ("camera", "video", "image")
COLOR_MODES = ("side", "person")
OVERLAY_MODES = ("keypoints", "organs", "both")


@dataclass(frozen=True, slots=True)
class Settings:
    # ---- model / graph (changing these reloads the graph)
    network_id: str = ""
    size: int = 320
    backend: str = "ncnn"  # "ncnn" (deployment) | "onnxruntime" (FP32 reference)
    num_threads: int = 4
    fp16: bool = True  # ncnn FP16 storage + arithmetic
    use_vulkan: bool = False  # ncnn Vulkan GPU
    # ---- decode attributes of the running graph
    mode: str = "whole"  # "whole" | "pyramid" (3 views, XTX3 ablation)
    present_threshold: float = 0.5
    visible_threshold: float = 0.5
    sigma_conf_coeff: float = 0.0
    sigma_floor: float = 0.05
    # ---- per-call detection parameters
    conf_threshold: float = 0.25
    max_detections: int = 50
    oks_nms: bool = False
    oks_nms_threshold: float = 0.5
    oks_nms_scale_factor: float = 1.5
    oks_nms_iou: bool = False
    oks_nms_iou_threshold: float = 0.5
    nms_kpt_threshold: float = 0.5
    # ---- source (changing these reopens the source)
    source_kind: str = "camera"  # "camera" | "video" (file or URL) | "image"
    camera_index: int = 0
    camera_name: str = ""
    source_path: str = ""
    width: int = 1280
    height: int = 720
    camera_fps: int = 30
    mjpg: bool = True
    windows_backend: str = "msmf"  # "msmf" | "dshow"
    mirror: bool = True
    # ---- display
    kpt_threshold: float = 0.3
    show_skeleton: bool = True
    show_points: bool = True
    show_occluded: bool = True
    show_scores: bool = True
    show_labels: bool = False
    show_ids: bool = False
    show_sigma: bool = False
    show_extent: bool = False
    color_mode: str = "side"  # "side" (left green / right orange) | "person"
    point_radius: int = 4
    line_thickness: int = 2
    show_hud: bool = True
    show_debug_view: bool = False
    # ---- organ overlay (brain, chest; FreeSentry ai_setup.organs semantics, extradetection.md)
    overlay_mode: str = "organs"  # "keypoints" | "organs" | "both"
    brain_enabled: bool = True
    brain_size_multiplier: float = 1.0
    brain_conf_threshold: float = 0.25
    brain_min_radius: int = 1
    chest_enabled: bool = True
    chest_size_multiplier: float = 1.0
    chest_conf_threshold: float = 0.143  # server 0.20 * 5/7 (chest confidence divided by 7)
    chest_min_radius: int = 1

    # ------------------------------------------------------------------ helpers
    def updated(self, **changes: Any) -> "Settings":
        return coerce(dataclasses.replace(self, **changes))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def differs(self, other: "Settings | None", keys: tuple[str, ...]) -> bool:
        return other is None or any(getattr(self, k) != getattr(other, k) for k in keys)

    def detect_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``PoseRunnerBase.detect`` (besides the frame)."""

        return {
            "conf_threshold": self.conf_threshold,
            "max_detections": self.max_detections,
            "oks_nms": self.oks_nms,
            "oks_nms_threshold": self.oks_nms_threshold,
            "oks_nms_scale_factor": self.oks_nms_scale_factor,
            "oks_nms_iou_threshold": self.oks_nms_iou_threshold if self.oks_nms_iou else None,
            "nms_kpt_threshold": self.nms_kpt_threshold,
            "keep_views": self.show_debug_view,
        }

    def decode_attrs(self) -> dict[str, Any]:
        """Runner attributes that may change without reloading the graph."""

        return {key: getattr(self, key) for key in DECODE_ATTRS}

    def preset_dict(self) -> dict[str, Any]:
        """Everything except the source (a preset should not switch the camera)."""

        return {k: v for k, v in self.to_dict().items() if k not in SOURCE_KEYS}


DEFAULTS: dict[str, Any] = {f.name: f.default for f in fields(Settings)}
FIELD_TYPES: dict[str, type] = {name: type(value) for name, value in DEFAULTS.items()}
RUNNER_KEYS = ("network_id", "size", "backend", "num_threads", "fp16", "use_vulkan")
DECODE_ATTRS = ("mode", "present_threshold", "visible_threshold", "sigma_conf_coeff", "sigma_floor")
SOURCE_KEYS = ("source_kind", "camera_index", "camera_name", "source_path", "width", "height", "camera_fps", "mjpg",
               "windows_backend")
# Keys a "decode": "manifest" preset resets to the network's manifest values.
MANIFEST_DECODE_KEYS = ("conf_threshold", "present_threshold", "visible_threshold", "sigma_conf_coeff", "sigma_floor",
                        "oks_nms", "oks_nms_threshold", "oks_nms_scale_factor")

_CHOICES = {"backend": BACKENDS, "mode": ("whole", "pyramid"), "source_kind": SOURCE_KINDS,
            "color_mode": COLOR_MODES, "windows_backend": ("msmf", "dshow"), "overlay_mode": OVERLAY_MODES}
_RANGES = {
    "size": (32, 4096), "num_threads": (1, 256), "max_detections": (1, 1000), "width": (16, 8192),
    "height": (16, 8192), "camera_fps": (1, 240), "point_radius": (1, 30), "line_thickness": (1, 20),
    "camera_index": (0, 100000),
    "brain_size_multiplier": (0.1, 5.0), "chest_size_multiplier": (0.1, 5.0),
    "brain_min_radius": (0, 200), "chest_min_radius": (0, 200),
}


def _convert(key: str, value: Any) -> Any:
    kind = FIELD_TYPES[key]
    if kind is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind is int:
        return int(round(float(value)))
    if kind is float:
        return float(value)
    return str(value)


def coerce(settings: Settings) -> Settings:
    """Clamp numbers and fix invalid choices (values may come from JSON or Tk widgets)."""

    changes: dict[str, Any] = {}
    for key, (lo, hi) in _RANGES.items():
        value = getattr(settings, key)
        if not lo <= value <= hi:
            changes[key] = min(hi, max(lo, value))
    for key in ("conf_threshold", "present_threshold", "visible_threshold", "oks_nms_threshold",
                "oks_nms_iou_threshold", "nms_kpt_threshold", "kpt_threshold", "brain_conf_threshold",
                "chest_conf_threshold"):
        value = getattr(settings, key)
        if not 0.0 <= value <= 1.0:
            changes[key] = min(1.0, max(0.0, value))
    for key, choices in _CHOICES.items():
        if getattr(settings, key) not in choices:
            changes[key] = DEFAULTS[key] if DEFAULTS[key] in choices else choices[0]
    return dataclasses.replace(settings, **changes) if changes else settings


def from_dict(data: dict[str, Any], base: Settings | None = None) -> Settings:
    """``base`` updated with the known keys of ``data`` (unknown keys and bad values are ignored)."""

    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in FIELD_TYPES or value is None:
            continue
        try:
            values[key] = _convert(key, value)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid setting %s=%r", key, value)
    return coerce(dataclasses.replace(base or Settings(), **values))


# ------------------------------------------------------------------ defaults
def manifest_decode(config: AppConfig, network_id: str) -> dict[str, Any]:
    """Decode / NMS values of ``network_id``: manifest ``decode`` block, overridden by
    explicit numbers in ``config.json`` ``inference`` (``"manifest"`` = keep the manifest)."""

    out: dict[str, Any] = {}
    if config.has_network(network_id):
        decode = config.network(network_id).decode_defaults()
        for key in MANIFEST_DECODE_KEYS:
            if key in decode and decode[key] is not None:
                out[key] = decode[key]
        iou = decode.get("oks_nms_iou_threshold")
        out["oks_nms_iou"] = iou is not None
        if iou is not None:
            out["oks_nms_iou_threshold"] = float(iou)
    overrides = config.inference.decode_overrides()
    if "oks_nms_iou_threshold" in overrides:
        iou = overrides.pop("oks_nms_iou_threshold")
        out["oks_nms_iou"] = iou is not None
        if iou is not None:
            out["oks_nms_iou_threshold"] = iou
    out.update(overrides)
    return out


def config_defaults(config: AppConfig) -> Settings:
    """Start-up settings from config.json alone (first network with model files)."""

    network = next((n for n in config.networks if n.available_sizes()), config.networks[0])
    cam, inf = config.camera, config.inference
    base = Settings(
        network_id=network.id,
        size=network.preferred_size() or network.default_size,
        mode=network.mode,
        num_threads=inf.num_threads,
        fp16=inf.fp16,
        use_vulkan=inf.use_vulkan,
        max_detections=inf.max_detections,
        oks_nms=inf.oks_nms,
        kpt_threshold=inf.kpt_threshold,
        width=cam.width,
        height=cam.height,
        camera_fps=cam.fps,
        mjpg=cam.mjpg,
        windows_backend=cam.windows_backend,
        mirror=cam.mirror,
    )
    decode = manifest_decode(config, network.id)
    decode["conf_threshold"] = network.conf_threshold if network.conf_threshold is not None else inf.conf_threshold
    decode["oks_nms"] = inf.oks_nms
    return from_dict({**decode, **config.display.defaults}, base)


def validate_for_config(settings: Settings, config: AppConfig) -> Settings:
    """Make the network / size / backend combination valid for this config."""

    if not config.has_network(settings.network_id):
        settings = settings.updated(network_id=config_defaults(config).network_id)
    network = config.network(settings.network_id)
    backend = settings.backend
    if not network.available_sizes(backend):
        backend = "ncnn"
    sizes = network.available_sizes(backend)
    size = settings.size if settings.size in sizes else (network.preferred_size(backend) or settings.size)
    return settings.updated(backend=backend, size=size)


def apply_preset(settings: Settings, preset: Preset | dict[str, Any], config: AppConfig) -> Settings:
    """``settings`` with a preset applied (unknown networks are skipped with a warning)."""

    values = dict(preset.settings if isinstance(preset, Preset) else preset)
    network_id = str(values.get("network_id", settings.network_id))
    if not config.has_network(network_id):
        logger.warning("Preset refers to unknown network %r; keeping %r", network_id, settings.network_id)
        values.pop("network_id", None)
        network_id = settings.network_id
    if values.pop("decode", None) == "manifest":
        settings = from_dict(manifest_decode(config, network_id), settings)
    return validate_for_config(from_dict(values, settings), config)


# --------------------------------------------------------------- persistence
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_last(config: AppConfig, path: Path = LAST_SETTINGS_PATH) -> Settings:
    """Config defaults updated with the last session (when there was one)."""

    base = config_defaults(config)
    data = _read_json(path)
    if not isinstance(data, dict):
        return base
    return validate_for_config(from_dict(data, base), config)


def save_last(settings: Settings, path: Path = LAST_SETTINGS_PATH) -> None:
    try:
        path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not save the last settings: %s", exc)


def load_user_presets(path: Path = USER_PRESETS_PATH) -> list[Preset]:
    data = _read_json(path)
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("name") and isinstance(item.get("settings"), dict):
            out.append(Preset(name=str(item["name"]), settings=dict(item["settings"]),
                              description=str(item.get("description", "")), builtin=False))
    return out


def save_user_presets(presets: list[Preset], path: Path = USER_PRESETS_PATH) -> None:
    payload = [{"name": p.name, "description": p.description, "settings": p.settings}
               for p in presets if not p.builtin]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def all_presets(config: AppConfig, user_path: Path = USER_PRESETS_PATH) -> list[Preset]:
    return [*config.presets, *load_user_presets(user_path)]


def find_preset(name: str, presets: list[Preset]) -> Preset:
    for preset in presets:
        if preset.name == name:
            return preset
    lowered = name.lower()
    matches = [p for p in presets if p.name.lower().startswith(lowered)]
    if len(matches) == 1:
        return matches[0]
    raise ConfigError(f"Unknown preset {name!r}; available: " + "; ".join(p.name for p in presets))
