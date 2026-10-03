"""Configuration loading for the live test (``config.json``).

Paths inside the config are resolved relative to the config file itself, so the
app works regardless of the current working directory. Network file paths may
contain a ``{size}`` placeholder that is substituted with the selected input size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

VIEW_MODES = ("whole", "pyramid")
DEFAULT_OUTPUT_BLOBS = {"box": "out0", "cls": "out1", "kpt": "out2"}


class ConfigError(ValueError):
    """Raised when config.json is missing fields or inconsistent."""


@dataclass(slots=True)
class SkeletonPreset:
    keypoints: list[str]
    edges: list[tuple[int, int]]
    sides: list[str]  # per keypoint: "l" | "r" | "c"
    oks_sigmas: list[float]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)


@dataclass(slots=True)
class NetworkConfig:
    id: str
    label: str
    format: str
    param: str  # raw template, may contain {size}
    bin: str
    sizes: list[int]
    default_size: int
    mode: str
    strides: list[int]
    input_blob: str
    output_blobs: dict[str, str]
    skeleton: SkeletonPreset
    conf_threshold: float | None
    base_dir: Path

    def param_path(self, size: int) -> Path:
        return (self.base_dir / self.param.format(size=size)).resolve()

    def bin_path(self, size: int) -> Path:
        return (self.base_dir / self.bin.format(size=size)).resolve()

    def size_available(self, size: int) -> bool:
        return self.param_path(size).is_file() and self.bin_path(size).is_file()

    def available_sizes(self) -> list[int]:
        return [s for s in self.sizes if self.size_available(s)]

    def preferred_size(self) -> int | None:
        available = self.available_sizes()
        if not available:
            return None
        return self.default_size if self.default_size in available else available[0]


@dataclass(slots=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True
    max_probe_index: int = 8
    mirror: bool = True
    windows_backend: str = "msmf"  # "msmf" | "dshow"


@dataclass(slots=True)
class InferenceConfig:
    conf_threshold: float = 0.25
    kpt_threshold: float = 0.3
    max_detections: int = 50
    num_threads: int = 4
    use_vulkan: bool = False
    async_inference: bool = False
    dedup: bool = False
    oks_match_thr: float = 0.5
    iou_match_thr: float = 0.5


@dataclass(slots=True)
class DisplayConfig:
    window_title: str = "XTX3 live test"
    show_bbox: bool = True
    show_labels: bool = False
    show_hud: bool = True
    point_radius: int = 4
    line_thickness: int = 2


@dataclass(slots=True)
class AppConfig:
    path: Path
    networks: list[NetworkConfig]
    camera: CameraConfig
    inference: InferenceConfig
    display: DisplayConfig = field(default_factory=DisplayConfig)

    def network(self, network_id: str) -> NetworkConfig:
        for net in self.networks:
            if net.id == network_id:
                return net
        known = ", ".join(n.id for n in self.networks)
        raise ConfigError(f"Unknown network {network_id!r}; configured: {known}")


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ConfigError(f"Missing '{key}' in {where}")
    return data[key]


def _parse_skeleton(name: str, data: dict[str, Any]) -> SkeletonPreset:
    where = f"skeleton_presets.{name}"
    keypoints = [str(k) for k in _require(data, "keypoints", where)]
    count = len(keypoints)
    edges = [(int(a), int(b)) for a, b in data.get("edges", [])]
    for a, b in edges:
        if not (0 <= a < count and 0 <= b < count):
            raise ConfigError(f"{where}: edge ({a}, {b}) out of range for {count} keypoints")
    sides = [str(s) for s in data.get("sides", ["c"] * count)]
    sigmas = [float(s) for s in data.get("oks_sigmas", [0.05] * count)]
    if len(sides) != count or len(sigmas) != count:
        raise ConfigError(f"{where}: 'sides' and 'oks_sigmas' must have {count} entries")
    return SkeletonPreset(keypoints=keypoints, edges=edges, sides=sides, oks_sigmas=sigmas)


def _parse_network(
    data: dict[str, Any], presets: dict[str, SkeletonPreset], base_dir: Path
) -> NetworkConfig:
    net_id = str(_require(data, "id", "networks[]"))
    where = f"network '{net_id}'"
    fmt = str(data.get("format", "ncnn"))
    if fmt != "ncnn":
        raise ConfigError(f"{where}: unsupported format {fmt!r} (only 'ncnn' is implemented)")
    mode = str(data.get("mode", "whole"))
    if mode not in VIEW_MODES:
        raise ConfigError(f"{where}: mode must be one of {VIEW_MODES}, got {mode!r}")
    sizes = [int(s) for s in _require(data, "sizes", where)]
    if not sizes or any(s % 32 for s in sizes):
        raise ConfigError(f"{where}: 'sizes' must be a non-empty list of multiples of 32")
    preset_name = str(data.get("skeleton_preset", "xtx3_9"))
    if preset_name not in presets:
        raise ConfigError(f"{where}: unknown skeleton_preset {preset_name!r}")
    outputs = dict(DEFAULT_OUTPUT_BLOBS)
    outputs.update({str(k): str(v) for k, v in data.get("output_blobs", {}).items()})
    conf = data.get("conf_threshold")
    return NetworkConfig(
        id=net_id,
        label=str(data.get("label", net_id)),
        format=fmt,
        param=str(_require(data, "param", where)),
        bin=str(_require(data, "bin", where)),
        sizes=sizes,
        default_size=int(data.get("default_size", sizes[0])),
        mode=mode,
        strides=[int(s) for s in data.get("strides", [8, 16, 32])],
        input_blob=str(data.get("input_blob", "in0")),
        output_blobs=outputs,
        skeleton=presets[preset_name],
        conf_threshold=float(conf) if conf is not None else None,
        base_dir=base_dir,
    )


def _dataclass_from(cls: type, data: dict[str, Any]) -> Any:
    names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.json``."""

    cfg_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {cfg_path}: {exc}") from exc

    base_dir = cfg_path.parent
    presets = {
        name: _parse_skeleton(name, data) for name, data in raw.get("skeleton_presets", {}).items()
    }
    networks = [_parse_network(n, presets, base_dir) for n in raw.get("networks", [])]
    if not networks:
        raise ConfigError("config.json defines no networks")
    ids = [n.id for n in networks]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"Duplicate network ids in config.json: {ids}")

    inference_raw = dict(raw.get("inference", {}))
    merge_raw = inference_raw.pop("merge", {}) or {}
    inference_raw.update({k: v for k, v in merge_raw.items() if k in ("oks_match_thr", "iou_match_thr")})

    return AppConfig(
        path=cfg_path,
        networks=networks,
        camera=_dataclass_from(CameraConfig, raw.get("camera", {})),
        inference=_dataclass_from(InferenceConfig, inference_raw),
        display=_dataclass_from(DisplayConfig, raw.get("display", {})),
    )
