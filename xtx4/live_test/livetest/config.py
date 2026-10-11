"""Configuration loading for the live test (``config.json``).

Paths are resolved relative to the config file. A network entry points either
to an export folder's ``manifest.json`` (preferred: sizes, strides, blob names,
decode parameters and keypoint metadata come from it) and/or to explicit
``param``/``bin`` templates containing ``{size}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xtx4.export.manifest import ManifestError, load_manifest, ncnn_files
from xtx4.utils.keypoints import KEYPOINT_NAMES, OKS_SIGMAS, SKELETON_EDGES

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
VIEW_MODES = ("whole", "pyramid")
DEFAULT_OUTPUT_BLOBS = ("out0", "out1")


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


def _side(name: str) -> str:
    return "l" if name.startswith("left") else "r" if name.startswith("right") else "c"


def skeleton_from(names: list[str], edges: list[Any], sigmas: list[float]) -> SkeletonPreset:
    count = len(names)
    parsed = [(int(a), int(b)) for a, b in edges]
    for a, b in parsed:
        if not (0 <= a < count and 0 <= b < count):
            raise ConfigError(f"skeleton edge ({a}, {b}) out of range for {count} keypoints")
    if len(sigmas) != count:
        raise ConfigError(f"{len(sigmas)} OKS sigmas for {count} keypoints")
    return SkeletonPreset(keypoints=list(names), edges=parsed, sides=[_side(n) for n in names],
                          oks_sigmas=[float(s) for s in sigmas])


DEFAULT_SKELETON = skeleton_from(list(KEYPOINT_NAMES), [list(e) for e in SKELETON_EDGES], OKS_SIGMAS.tolist())


@dataclass(slots=True)
class NetworkConfig:
    id: str
    label: str
    param: str | None  # template with {size}, or None when taken from the manifest
    bin: str | None
    manifest: str | None
    sizes: list[int]
    default_size: int
    mode: str
    strides: list[int] | None
    input_blob: str
    output_blobs: tuple[str, str]
    skeleton: SkeletonPreset
    conf_threshold: float | None
    base_dir: Path
    manifest_data: dict[str, Any] | None = None
    manifest_error: str | None = None

    @property
    def manifest_path(self) -> Path | None:
        return (self.base_dir / self.manifest).resolve() if self.manifest else None

    def param_path(self, size: int) -> Path:
        if self.param:
            return (self.base_dir / self.param.format(size=size)).resolve()
        if self.manifest_data is not None:
            return ncnn_files(self.manifest_data, size, self.manifest_path.parent)[0]
        raise ConfigError(f"network '{self.id}': no param template and no manifest")

    def bin_path(self, size: int) -> Path:
        if self.bin:
            return (self.base_dir / self.bin.format(size=size)).resolve()
        if self.manifest_data is not None:
            return ncnn_files(self.manifest_data, size, self.manifest_path.parent)[1]
        raise ConfigError(f"network '{self.id}': no bin template and no manifest")

    def size_available(self, size: int) -> bool:
        try:
            return self.param_path(size).is_file() and self.bin_path(size).is_file()
        except (ConfigError, ManifestError):
            return False

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
    fp16: bool = True
    oks_nms: bool = False
    # Decode / NMS parameters: "manifest" (or key absent) = take the value from the network's
    # manifest ``decode`` block (defaults when there is no manifest); a number overrides it.
    oks_nms_threshold: float | str = "manifest"
    oks_nms_iou_threshold: float | str | None = "manifest"  # null = IoU test off
    oks_nms_scale_factor: float | str = "manifest"
    present_threshold: float | str = "manifest"
    visible_threshold: float | str = "manifest"

    def decode_overrides(self) -> dict[str, Any]:
        """Runner keyword arguments for the values set explicitly in config.json."""

        out: dict[str, Any] = {}
        for key in ("oks_nms_threshold", "oks_nms_scale_factor", "present_threshold", "visible_threshold"):
            value = getattr(self, key)
            if value is not None and value != "manifest":
                out[key] = float(value)
        if self.oks_nms_iou_threshold != "manifest":
            iou = self.oks_nms_iou_threshold
            out["oks_nms_iou_threshold"] = None if iou is None else float(iou)
        return out


@dataclass(slots=True)
class DisplayConfig:
    window_title: str = "XTX4 live test"
    show_scores: bool = True
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


def _parse_network(data: dict[str, Any], base_dir: Path) -> NetworkConfig:
    net_id = str(_require(data, "id", "networks[]"))
    where = f"network '{net_id}'"
    fmt = str(data.get("format", "ncnn"))
    if fmt != "ncnn":
        raise ConfigError(f"{where}: unsupported format {fmt!r} (only 'ncnn' is implemented)")
    manifest_rel = data.get("manifest")
    manifest_data: dict[str, Any] | None = None
    manifest_error: str | None = None
    if manifest_rel:
        try:
            manifest_data = load_manifest((base_dir / str(manifest_rel)).resolve())
        except ManifestError as exc:
            manifest_error = str(exc)
    if not manifest_rel and not (data.get("param") and data.get("bin")):
        raise ConfigError(f"{where}: needs 'manifest' or both 'param' and 'bin'")

    sizes = data.get("sizes") or (manifest_data or {}).get("input", {}).get("sizes") or []
    sizes = [int(s) for s in sizes]
    if not sizes and manifest_data is None:
        sizes = [320]  # manifest missing (not exported yet); shown as "model files missing"
    if any(s % 32 for s in sizes):
        raise ConfigError(f"{where}: 'sizes' must be multiples of 32")
    mode = str(data.get("mode", "whole"))
    if mode not in VIEW_MODES:
        raise ConfigError(f"{where}: mode must be one of {VIEW_MODES}, got {mode!r}")

    strides = data.get("strides") or (manifest_data or {}).get("strides")
    blobs = (manifest_data or {}).get("outputs", {}).get("ncnn", {}).get("blobs", {})
    outputs = data.get("output_blobs") or ([blobs["person"], blobs["pose"]] if blobs else DEFAULT_OUTPUT_BLOBS)
    if isinstance(outputs, dict):
        outputs = [outputs["person"], outputs["pose"]]
    kp_meta = (manifest_data or {}).get("keypoints")
    skeleton = (
        skeleton_from(kp_meta["names"], kp_meta["skeleton_edges"], kp_meta["sigmas"]) if kp_meta else DEFAULT_SKELETON
    )
    conf = data.get("conf_threshold")
    return NetworkConfig(
        id=net_id,
        label=str(data.get("label", net_id)),
        param=data.get("param"),
        bin=data.get("bin"),
        manifest=str(manifest_rel) if manifest_rel else None,
        sizes=sizes,
        default_size=int(data.get("default_size", 320 if 320 in sizes else (sizes[0] if sizes else 320))),
        mode=mode,
        strides=[int(s) for s in strides] if strides else None,
        input_blob=str(data.get("input_blob", "in0")),
        output_blobs=(str(outputs[0]), str(outputs[1])),
        skeleton=skeleton,
        conf_threshold=float(conf) if conf is not None else None,
        base_dir=base_dir,
        manifest_data=manifest_data,
        manifest_error=manifest_error,
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

    networks = [_parse_network(n, cfg_path.parent) for n in raw.get("networks", [])]
    if not networks:
        raise ConfigError("config.json defines no networks")
    ids = [n.id for n in networks]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"Duplicate network ids in config.json: {ids}")
    return AppConfig(
        path=cfg_path,
        networks=networks,
        camera=_dataclass_from(CameraConfig, raw.get("camera", {})),
        inference=_dataclass_from(InferenceConfig, raw.get("inference", {})),
        display=_dataclass_from(DisplayConfig, raw.get("display", {})),
    )
