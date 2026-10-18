"""Deployment manifest (CONTRACTS.md section 12): everything a torch-free runtime
needs to run an exported XTX4 graph correctly, in one JSON next to the graphs.

Why a manifest: the raw graph outputs are meaningless without the decode
contract (strides, anchor order, channel layout, sigma floor, colour/scale
convention, letterbox geometry). Recording these with hashes of the checkpoint
and config makes every deployed file traceable and lets the runtime refuse a
graph whose contract it does not implement (``decode_version``).

This module must stay **torch-free** -- ``xtx4.runtime.ncnn_runner`` imports it
on the Raspberry Pi.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from ..utils.keypoints import (
    COCO_KEYPOINT_INDICES,
    FLIP_PAIRS,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    OKS_SIGMAS,
    SKELETON_EDGES,
)
from ..utils.organs import (
    ORGAN_CHANNELS,
    ORGAN_CHANNELS_PER_ORGAN,
    ORGAN_NAMES,
    ORGAN_SOURCE,
    ORGAN_VERSION,
    organ_constants,
)

SCHEMA_VERSION = 2  # 2: optional third output ``organs`` + ``organs`` block (extradetection.md)
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
DECODE_VERSION = 1  # == xtx4.runtime.decode.DECODE_VERSION
MANIFEST_NAME = "manifest.json"
PARITY_NAME = "parity.json"

REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version", "model_id", "checkpoint_sha256", "config_sha256", "variant_spec", "keypoints",
    "input", "strides", "levels", "anchor_order", "outputs", "decode_version", "decode", "precision",
    "tools", "created",
)

DEFAULT_DECODE: dict[str, Any] = {
    "sigma_floor": 0.05,
    "sigma_conf_coeff": 0.0,
    "conf_threshold": 0.25,
    "max_detections": 100,
    "oks_nms": False,
    "oks_nms_threshold": 0.5,
    "oks_nms_iou_threshold": None,
    "oks_nms_scale_factor": 1.5,
    "present_threshold": 0.5,
    "visible_threshold": 0.5,
}

LETTERBOX_DESCRIPTION: dict[str, Any] = {
    "mode": "whole",
    "aspect": "preserved",
    "upscale": False,
    "placement": "centred",
    "pad_value": 0,
    "offset_align": 4,
    "interpolation": "INTER_AREA when downscaling (single resize from the source frame)",
    "transform": "2x3 affine, network px -> frame px (pure scale + translation)",
    "reference": "xtx4.data.preprocess (whole) == xtx4.runtime.preprocess",
}


class ManifestError(ValueError):
    """Raised for a missing / incompatible deployment manifest."""


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    """Hash of the canonical JSON encoding (sorted keys, no whitespace)."""

    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tool_versions(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Versions of the export/runtime toolchain (read from package metadata, no imports)."""

    versions = {"python": platform.python_version()}
    for pkg in ("torch", "onnx", "onnxsim", "onnxruntime", "pnnx", "ncnn", "numpy", "opencv-python"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    if extra:
        versions.update({str(k): str(v) for k, v in extra.items()})
    return versions


def keypoint_meta(num_keypoints: int = NUM_KEYPOINTS) -> dict[str, Any]:
    if num_keypoints != NUM_KEYPOINTS:
        raise ManifestError(f"keypoint metadata is defined for K = {NUM_KEYPOINTS}, got {num_keypoints}")
    return {
        "count": NUM_KEYPOINTS,
        "names": list(KEYPOINT_NAMES),
        "coco_indices": list(COCO_KEYPOINT_INDICES),
        "flip_pairs": [list(p) for p in FLIP_PAIRS],
        "sigmas": [float(s) for s in OKS_SIGMAS],
        "skeleton_edges": [list(e) for e in SKELETON_EDGES],
    }


def anchor_counts(size: int, strides: Sequence[int]) -> list[int]:
    return [(int(size) // int(s)) ** 2 for s in strides]


def build_manifest(
    *,
    model_id: str,
    variant_spec: Mapping[str, Any],
    strides: Sequence[int],
    levels: str,
    sizes: Sequence[int],
    num_keypoints: int = NUM_KEYPOINTS,
    checkpoint_sha256: str | None = None,
    config: Mapping[str, Any] | None = None,
    config_sha256: str | None = None,
    onnx_files: Mapping[int, Mapping[str, Any]] | None = None,
    ncnn_files: Mapping[int, Mapping[str, Any]] | None = None,
    onnx_opset: int = 17,
    input_name: str = "in0",
    onnx_output_names: Sequence[str] = ("person", "pose", "organs"),
    ncnn_blobs: Mapping[str, str] | None = None,
    converter: str = "pnnx",
    fp16: bool = True,
    decode: Mapping[str, Any] | None = None,
    tools: Mapping[str, str] | None = None,
    parity: Mapping[str, Any] | None = None,
    created: str | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict.

    ``onnx_files[S] = {"path": "<file name relative to the manifest>"}`` and
    ``ncnn_files[S] = {"param": ..., "bin": ...}``; shapes are filled in from
    ``N(S)``. ``decode`` overrides :data:`DEFAULT_DECODE` (typically
    ``config["decode"]``). A third ONNX output name (``organs``) declares the organ
    output of graphs exported with the organ layer.
    """

    sizes = [int(s) for s in sizes]
    strides = [int(s) for s in strides]
    channels = num_keypoints * 5
    if config_sha256 is None and config is not None:
        config_sha256 = sha256_json(config)
    decode_params = dict(DEFAULT_DECODE)
    if decode:
        decode_params.update({k: decode[k] for k in DEFAULT_DECODE if k in decode})
    with_organs = len(onnx_output_names) >= 3
    organ_channels = len(ORGAN_NAMES) * ORGAN_CHANNELS_PER_ORGAN
    default_blobs = {"input": input_name, "person": "out0", "pose": "out1"}
    if with_organs:
        default_blobs["organs"] = "out2"
    blobs = dict(ncnn_blobs or default_blobs)
    if with_organs and "organs" not in blobs:
        raise ManifestError("ncnn_blobs must name the 'organs' blob of a graph with the organ output")

    def shapes(n: int, batch: bool) -> dict[str, list[int]]:
        lead = [1, n] if batch else [n]
        out = {"person": lead + [1], "pose": lead + [channels]}
        if with_organs:
            out["organs"] = lead + [organ_channels]
        return out

    n_per_size = {str(s): sum(anchor_counts(s, strides)) for s in sizes}
    names = {"person": onnx_output_names[0], "pose": onnx_output_names[1]}
    if with_organs:
        names["organs"] = onnx_output_names[2]
    onnx_section: dict[str, Any] = {
        "input": input_name,
        "names": names,
        "opset": int(onnx_opset),
        "files": {},
    }
    for s, info in (onnx_files or {}).items():
        n = n_per_size.get(str(s), sum(anchor_counts(int(s), strides)))
        onnx_section["files"][str(s)] = {
            **dict(info),
            "input_shape": [1, 3, int(s), int(s)],
            "shapes": shapes(n, True),
        }
    ncnn_section: dict[str, Any] = {
        "input": blobs.get("input", input_name),
        "blobs": {k: blobs[k] for k in names},
        "converter": converter,
        "fp16_weights": bool(fp16),
        "files": {},
    }
    for s, info in (ncnn_files or {}).items():
        n = n_per_size.get(str(s), sum(anchor_counts(int(s), strides)))
        ncnn_section["files"][str(s)] = {
            **dict(info),
            "input_shape": [3, int(s), int(s)],
            "shapes": shapes(n, False),
        }
    layout: dict[str, Any] = {
        "person": "[1, N, 1] raw person logit",
        "pose": f"[1, N, K*5] raw, keypoint-major (dx, dy, log_sigma, present_logit, visible_logit), K={num_keypoints}",
        "channels_per_keypoint": ["dx", "dy", "log_sigma", "present_logit", "visible_logit"],
    }
    if with_organs:
        layout["organs"] = (
            f"[1, N, {organ_channels}] organ-major ({', '.join(ORGAN_NAMES)}), per organ "
            f"({', '.join(ORGAN_CHANNELS)}); dx, dy, r in stride units, conf a probability in [0, 1]"
        )
        layout["channels_per_organ"] = list(ORGAN_CHANNELS)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "variant_spec": dict(variant_spec),
        "num_keypoints": int(num_keypoints),
        "keypoints": keypoint_meta(num_keypoints),
        "input": {
            "name": input_name,
            "layout": "NCHW",
            "dtype": "float32",
            "color": "BGR",
            "scale": "1/255",
            "mean": [0.0, 0.0, 0.0],
            "sizes": sizes,
            "letterbox": dict(LETTERBOX_DESCRIPTION),
        },
        "strides": strides,
        "levels": levels,
        "anchor_order": {
            "levels": "ascending stride",
            "within_level": "row-major (y outer, x inner)",
            "centre": "(i + 0.5) * stride",
            "per_level": {str(s): anchor_counts(s, strides) for s in sizes},
            "N": n_per_size,
        },
        "outputs": {
            "layout": layout,
            "onnx": onnx_section,
            "ncnn": ncnn_section,
        },
        "decode_version": DECODE_VERSION,
        "decode": {
            **decode_params,
            "xy": "anchor + (dx, dy) * stride",
            "sigma": "sigma_units = softplus(log_sigma) + sigma_floor; sigma_px = sigma_units * stride",
            "conf": "sigmoid(present) * exp(-sigma_conf_coeff * sigma_units)",
            "visibility": "0 if sigmoid(present) < present_threshold else (2 if sigmoid(visible) >= visible_threshold else 1)",
            "oks_nms_rule": (
                "greedy by score; drop j if OKS(kept, j) >= oks_nms_threshold with s^2 = max((f w)(f h), 16^2) of j's "
                "confident-point extent (f = oks_nms_scale_factor), or expanded-box IoU >= oks_nms_iou_threshold (null = off)"
            ),
            "score_filter": "sigmoid(person) >= conf_threshold before keypoint decode; top max_detections by score",
        },
        "precision": {
            "onnx": "fp32",
            "ncnn_weights": "fp16" if fp16 else "fp32",
            "ncnn_runtime_default": "fp16 storage + arithmetic" if fp16 else "fp32",
        },
        "tools": dict(tools) if tools is not None else tool_versions(),
    }
    if with_organs:
        manifest["organs"] = {
            "version": ORGAN_VERSION,
            "names": list(ORGAN_NAMES),
            "channels": list(ORGAN_CHANNELS),
            "units": {"dx": "stride", "dy": "stride", "r_units": "stride", "conf": "probability [0, 1]"},
            "decode": "x = ax + dx * stride, y = ay + dy * stride, r_px = r_units * stride",
            "source": ORGAN_SOURCE,
            "reference": "xtx4.utils.organs.compute_organs (chest conf / 7 instead of the server's / 5)",
            "constants": organ_constants(),
        }
    if parity is not None:
        manifest["parity"] = dict(parity)
    return manifest


def manifest_has_organs(manifest: Mapping[str, Any]) -> bool:
    """True when the exported graphs carry the ``organs`` output (schema 2 with the organ layer)."""

    return "organs" in manifest.get("outputs", {}).get("ncnn", {}).get("blobs", {})


def validate_manifest(manifest: Mapping[str, Any], *, source: str = "manifest") -> None:
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ManifestError(f"{source}: missing keys {missing}")
    if int(manifest["schema_version"]) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(
            f"{source}: schema_version {manifest['schema_version']} not in supported {SUPPORTED_SCHEMA_VERSIONS}"
        )
    if int(manifest["decode_version"]) != DECODE_VERSION:
        raise ManifestError(
            f"{source}: decode_version {manifest['decode_version']} is not implemented by this runtime "
            f"(supports {DECODE_VERSION})"
        )


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    validate_manifest(manifest, source=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate a manifest; adds nothing to the dict (paths stay relative)."""

    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {path}: {exc}") from exc
    validate_manifest(manifest, source=str(path))
    return manifest


def ncnn_files(manifest: Mapping[str, Any], size: int, base_dir: Path | str) -> tuple[Path, Path]:
    """Absolute ``(param, bin)`` paths of the ncnn graph for ``size``."""

    files = manifest["outputs"]["ncnn"]["files"]
    entry = files.get(str(int(size)))
    if entry is None:
        raise ManifestError(f"No ncnn graph for size {size} in the manifest (sizes: {sorted(files)})")
    base = Path(base_dir)
    return (base / entry["param"]).resolve(), (base / entry["bin"]).resolve()
