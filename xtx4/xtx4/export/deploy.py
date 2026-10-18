"""Deployment export of a trained XTX4 checkpoint: fused graph -> ONNX + ncnn per
input size -> parity report -> manifest (CONTRACTS.md section 12,
``implementation_plan_fable.md`` 10.1).

Output folder (``<run>/export/`` by default)::

    xtx4_256.onnx  xtx4_256.ncnn.param  xtx4_256.ncnn.bin   (one triple per size)
    manifest.json   decode contract, hashes, blob names, shapes, tool versions
    parity.json     PyTorch vs ONNX Runtime vs ncnn on real-ish inputs

Parity protocol (per size, per input): inputs are a letterboxed smoke image, a
constant grey image and uniform noise. Raw outputs are compared element-wise
(max abs). Keypoints are compared **by anchor id before any filter** (plan
10.1): the compared anchors are the reference detections (``score >=
conf_threshold``) or, when there are fewer than ``min_compare_anchors`` of
them (always the case for random weights, whose person bias starts at -4.6),
the top ``min_compare_anchors`` anchors by reference score. Both raw sets are
decoded with the runtime NumPy decoder and the Euclidean xy error (network
px) is summarised as max/p95/p99.

Tolerances: ONNX raw max abs <= 1e-3, ONNX xy max <= 0.05 px, ONNX score max
<= 1e-4; ncnn FP16 xy p95 <= 0.25 px and p99 <= 1 px.

Organs (brain, chest; ``extradetection.md``) are compared on the same anchors:
centre / radius error in network px and confidence difference. The PyTorch organ
output is additionally checked against the host reference
(``organs_from_pose_selected`` on the reference pose rows), so the graph and the
NumPy fallback provably agree. Tolerances: host vs graph <= 0.01 px, ONNX organs
max <= 0.05 px, ncnn FP16 organs p99 <= 1 px.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..runtime.decode import decode_organs_selected, decode_selected, organs_from_pose_selected, person_scores
from ..runtime.preprocess import letterbox, to_blob
from ..utils.logging import get_logger
from ..utils.organs import gate_ramp_rows
from .manifest import (
    MANIFEST_NAME,
    PARITY_NAME,
    build_manifest,
    sha256_file,
    sha256_json,
    tool_versions,
    write_manifest,
)
from .onnx_ncnn import (
    INPUT_NAME,
    OUTPUT_NAMES,
    ExportModel,
    convert_onnx_to_ncnn,
    export_onnx,
    verify_onnx,
)

logger = get_logger(__name__)

DEFAULT_SIZES: tuple[int, ...] = (256, 320, 384)
DEFAULT_PARITY_IMAGES = Path(__file__).resolve().parents[3] / "xtx3" / ".smoke_dataset" / "test"
MIN_COMPARE_ANCHORS = 100

TOLERANCES: dict[str, float] = {
    "onnx_raw_max_abs": 1e-3,
    "onnx_xy_max_px": 0.05,
    "onnx_score_max_abs": 1e-4,
    "ncnn_xy_p95_px": 0.25,
    "ncnn_xy_p99_px": 1.0,
    "host_organs_max_px": 0.01,
    "onnx_organs_max_px": 0.05,
    "ncnn_organs_p99_px": 1.0,
}


class ParityError(RuntimeError):
    """Exported graph does not reproduce PyTorch within the CONTRACTS tolerances."""


# ------------------------------------------------------------------ helpers
def _first_image(image_dir: Path | None) -> Path | None:
    if image_dir is None or not Path(image_dir).is_dir():
        return None
    for path in sorted(Path(image_dir).iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return path
    return None


def parity_inputs(size: int, *, image_dir: Path | None = DEFAULT_PARITY_IMAGES, seed: int = 0) -> dict[str, np.ndarray]:
    """``{name: (1, 3, S, S) float32}``: letterboxed smoke image, constant grey, uniform noise."""

    import cv2

    inputs: dict[str, np.ndarray] = {}
    image_path = _first_image(image_dir)
    if image_path is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            view, _ = letterbox(image, size)
            inputs[f"image:{image_path.name}"] = to_blob(view)
    if not inputs:
        logger.warning("No parity image found in %s; using synthetic inputs only", image_dir)
    inputs["constant"] = np.full((1, 3, size, size), 114.0 / 255.0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    inputs["random"] = rng.random((1, 3, size, size), dtype=np.float32)
    return inputs


def _compare(
    ref: tuple[np.ndarray, ...],
    test: tuple[np.ndarray, ...],
    *,
    size: int,
    strides: Sequence[int],
    num_keypoints: int,
    conf_threshold: float,
    sigma_conf_coeff: float,
    sigma_floor: float,
    present_threshold: float = 0.5,
    visible_threshold: float = 0.5,
    min_compare: int = MIN_COMPARE_ANCHORS,
) -> dict[str, Any]:
    """Raw + decoded comparison of ``test`` against the reference raw outputs."""

    person_r = np.asarray(ref[0], np.float32).reshape(-1)
    person_t = np.asarray(test[0], np.float32).reshape(-1)
    pose_r = np.asarray(ref[1], np.float32).reshape(person_r.size, -1)
    pose_t = np.asarray(test[1], np.float32).reshape(person_t.size, -1)
    if person_r.shape != person_t.shape or pose_r.shape != pose_t.shape:
        raise ParityError(f"output shapes differ: {person_t.shape}/{pose_t.shape} vs {person_r.shape}/{pose_r.shape}")
    scores_r = person_scores(person_r)
    scores_t = person_scores(person_t)
    selected = np.flatnonzero(scores_r >= conf_threshold)
    selection = "threshold"
    if selected.size < min_compare:
        selected = np.argsort(-scores_r, kind="stable")[: min(min_compare, scores_r.size)]
        selection = f"top{selected.size}"
    kw = dict(network_size=size, strides=strides, num_keypoints=num_keypoints,
              sigma_conf_coeff=sigma_conf_coeff, sigma_floor=sigma_floor,
              present_threshold=present_threshold, visible_threshold=visible_threshold)
    kp_r, vis_r, _ = decode_selected(pose_r, selected, **kw)
    kp_t, vis_t, _ = decode_selected(pose_t, selected, **kw)
    err = np.hypot(kp_t[..., 0] - kp_r[..., 0], kp_t[..., 1] - kp_r[..., 1]).reshape(-1)
    det_r = set(np.flatnonzero(scores_r >= conf_threshold).tolist())
    det_t = set(np.flatnonzero(scores_t >= conf_threshold).tolist())
    union = det_r | det_t
    organs = _compare_organs(ref, test, selected, pose_r, size=size, strides=strides, num_keypoints=num_keypoints)
    return {
        **organs,
        "person_raw_max_abs": float(np.abs(person_t - person_r).max()),
        "pose_raw_max_abs": float(np.abs(pose_t - pose_r).max()),
        "raw_max_abs": float(max(np.abs(person_t - person_r).max(), np.abs(pose_t - pose_r).max())),
        "score_max_abs": float(np.abs(scores_t - scores_r).max()),
        "selected_score_max_abs": float(np.abs(scores_t[selected] - scores_r[selected]).max()) if selected.size else 0.0,
        "compared_anchors": int(selected.size),
        "selection": selection,
        "xy_err_max_px": float(err.max()) if err.size else 0.0,
        "xy_err_p95_px": float(np.percentile(err, 95)) if err.size else 0.0,
        "xy_err_p99_px": float(np.percentile(err, 99)) if err.size else 0.0,
        "kpt_conf_max_abs": float(np.abs(kp_t[..., 2] - kp_r[..., 2]).max()) if err.size else 0.0,
        "visibility_agreement": float((vis_t == vis_r).mean()) if err.size else 1.0,
        "detections_ref": len(det_r),
        "detections_test": len(det_t),
        "detection_jaccard": float(len(det_r & det_t) / len(union)) if union else 1.0,
    }


def _compare_organs(
    ref: tuple[np.ndarray, ...],
    test: tuple[np.ndarray, ...],
    selected: np.ndarray,
    pose_r: np.ndarray,
    *,
    size: int,
    strides: Sequence[int],
    num_keypoints: int,
) -> dict[str, Any]:
    """Organ centre/radius error (network px) and confidence difference on the compared anchors."""

    if len(ref) < 3 or len(test) < 3 or ref[2] is None or test[2] is None or selected.size == 0:
        return {}
    # anchors in the brain gate's ramp blend the branches on float32 rounding; leave them out
    present_logit = pose_r.reshape(pose_r.shape[0], num_keypoints, -1)[selected, :, 3].astype(np.float64)
    present = 1.0 / (1.0 + np.exp(-present_logit))
    ramp = gate_ramp_rows(present)
    selected = selected[~ramp]
    if selected.size == 0:
        return {"organs_gate_ramp_rows": int(ramp.sum())}
    geo = dict(network_size=size, strides=strides)
    org_r = decode_organs_selected(ref[2], selected, **geo)
    org_t = decode_organs_selected(test[2], selected, **geo)
    host = organs_from_pose_selected(pose_r, selected, num_keypoints=num_keypoints, **geo)
    centre = np.hypot(org_t[..., 0] - org_r[..., 0], org_t[..., 1] - org_r[..., 1])
    err = np.maximum(centre, np.abs(org_t[..., 2] - org_r[..., 2])).reshape(-1)
    host_err = np.maximum(np.hypot(host[..., 0] - org_r[..., 0], host[..., 1] - org_r[..., 1]),
                          np.abs(host[..., 2] - org_r[..., 2]))
    return {
        "organs_err_max_px": float(err.max()),
        "organs_err_p99_px": float(np.percentile(err, 99)),
        "organs_conf_max_abs": float(np.abs(org_t[..., 3] - org_r[..., 3]).max()),
        "organs_host_err_max_px": float(host_err.max()),
        "organs_host_conf_max_abs": float(np.abs(host[..., 3] - org_r[..., 3]).max()),
        "organs_gate_ramp_rows": int(ramp.sum()),
    }


def _torch_raw(export_model: ExportModel, blob: np.ndarray) -> tuple[np.ndarray, ...]:
    with torch.no_grad():
        outs = export_model(torch.from_numpy(blob))
    return tuple(t.cpu().numpy() for t in outs)


def _worst(entries: list[dict[str, Any]], key: str) -> float:
    values = [e[key] for e in entries if key in e]
    return float(max(values)) if values else float("nan")


# ------------------------------------------------------------------- export
def export_model(
    model: torch.nn.Module,
    config: Mapping[str, Any] | None,
    out_dir: Path | str,
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    fp16: bool = True,
    opset: int = 17,
    converter: str = "pnnx",
    verify: bool = True,
    strict: bool = True,
    model_id: str | None = None,
    checkpoint_sha256: str | None = None,
    prefix: str = "xtx4",
    parity_images: Path | str | None = DEFAULT_PARITY_IMAGES,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Export ``model`` (an ``XTX4Model``; a fused deep copy is used, the argument is not
    modified) to ONNX + ncnn for every size, check parity and write the manifest.

    Returns ``{"out_dir", "manifest", "parity", "files": {S: {...}}, "parity_summary", "ok"}``.
    Raises :class:`ParityError` (after writing ``parity.json``) when ``verify`` and
    ``strict`` and a tolerance is violated.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config or {})
    decode_cfg = dict(config.get("decode", {}))
    conf_threshold = float(decode_cfg.get("conf_threshold", 0.25))
    sigma_conf_coeff = float(decode_cfg.get("sigma_conf_coeff", 0.0))
    sigma_floor = float(decode_cfg.get("sigma_floor", 0.05))
    vis_thresholds = dict(present_threshold=float(decode_cfg.get("present_threshold", 0.5)),
                          visible_threshold=float(decode_cfg.get("visible_threshold", 0.5)))

    deploy = copy.deepcopy(model).cpu().eval()
    deploy._quiet_fuse = True  # noqa: SLF001 - avoid a second "Fused ..." log line
    deploy.fuse()
    spec = deploy.spec
    strides = list(deploy.strides)
    num_keypoints = int(deploy.num_keypoints)
    export_wrapper = ExportModel(deploy).eval()
    if model_id is None:
        run_name = str(config.get("run", {}).get("name", "xtx4"))
        model_id = f"{run_name}-{spec.name}" + (f"-{checkpoint_sha256[:8]}" if checkpoint_sha256 else "")

    onnx_files: dict[int, dict[str, Any]] = {}
    ncnn_files: dict[int, dict[str, Any]] = {}
    parity: dict[str, Any] = {"tolerances": dict(TOLERANCES), "conf_threshold": conf_threshold, "sizes": {}}
    blobs = {"input": INPUT_NAME, "person": "out0", "pose": "out1", "organs": "out2"}
    all_ok = True
    t_start = time.time()
    for size in [int(s) for s in sizes]:
        stem = f"{prefix}_{size}"
        onnx_path = export_onnx(deploy, out_dir / f"{stem}.onnx", input_size=size, opset=opset)
        onnx_files[size] = {"path": onnx_path.name, "sha256": sha256_file(onnx_path)}
        conv = convert_onnx_to_ncnn(onnx_path, out_dir / stem, converter=converter, fp16=fp16)
        if len(conv.output_names) != 3 or len(conv.input_names) != 1:
            raise RuntimeError(f"Unexpected ncnn blobs: inputs {conv.input_names}, outputs {conv.output_names}")
        blobs = {"input": conv.input_names[0], "person": conv.output_names[0], "pose": conv.output_names[1],
                 "organs": conv.output_names[2]}
        ncnn_files[size] = {
            "param": conv.param_path.name,
            "bin": conv.bin_path.name,
            "sha256_param": sha256_file(conv.param_path),
            "sha256_bin": sha256_file(conv.bin_path),
            "fp16": bool(fp16),
            "layers": dict(sorted(conv.layer_types.items())),
        }
        if not verify:
            continue

        size_report: dict[str, Any] = {"onnx_verify_max_abs": verify_onnx(deploy, onnx_path, input_size=size),
                                       "inputs": {}}
        import onnxruntime as ort

        from ..runtime.ncnn_runner import NcnnPoseRunner

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        runner = NcnnPoseRunner(
            conv.param_path, conv.bin_path, size=size, strides=strides, num_keypoints=num_keypoints,
            num_threads=num_threads, fp16=fp16, input_blob=blobs["input"],
            output_blobs=(blobs["person"], blobs["pose"]), organs_blob=blobs["organs"],
        )
        compare_kw = dict(size=size, strides=strides, num_keypoints=num_keypoints, conf_threshold=conf_threshold,
                          sigma_conf_coeff=sigma_conf_coeff, sigma_floor=sigma_floor, **vis_thresholds)
        try:
            for name, blob in parity_inputs(size, image_dir=Path(parity_images) if parity_images else None).items():
                ref = _torch_raw(export_wrapper, blob)
                ort_out = session.run(list(OUTPUT_NAMES), {INPUT_NAME: blob})
                ncnn_out = runner.forward_blob_all(blob)
                size_report["inputs"][name] = {
                    "onnx": _compare(ref, tuple(ort_out), **compare_kw),
                    "ncnn": _compare(ref, ncnn_out, **compare_kw),
                }
        finally:
            runner.close()
        onnx_entries = [v["onnx"] for v in size_report["inputs"].values()]
        ncnn_entries = [v["ncnn"] for v in size_report["inputs"].values()]
        summary = {
            "onnx_raw_max_abs": _worst(onnx_entries, "raw_max_abs"),
            "onnx_xy_max_px": _worst(onnx_entries, "xy_err_max_px"),
            "onnx_score_max_abs": _worst(onnx_entries, "score_max_abs"),
            "ncnn_raw_max_abs": _worst(ncnn_entries, "raw_max_abs"),
            "ncnn_score_max_abs": _worst(ncnn_entries, "score_max_abs"),
            "ncnn_xy_p95_px": _worst(ncnn_entries, "xy_err_p95_px"),
            "ncnn_xy_p99_px": _worst(ncnn_entries, "xy_err_p99_px"),
            "ncnn_xy_max_px": _worst(ncnn_entries, "xy_err_max_px"),
            "host_organs_max_px": _worst(onnx_entries, "organs_host_err_max_px"),
            "onnx_organs_max_px": _worst(onnx_entries, "organs_err_max_px"),
            "ncnn_organs_p99_px": _worst(ncnn_entries, "organs_err_p99_px"),
            "ncnn_organs_max_px": _worst(ncnn_entries, "organs_err_max_px"),
            "ncnn_organs_conf_max_abs": _worst(ncnn_entries, "organs_conf_max_abs"),
        }
        failures = [key for key, tol in TOLERANCES.items() if not summary[key] <= tol]
        summary["pass"] = not failures
        summary["failures"] = failures
        size_report["summary"] = summary
        parity["sizes"][str(size)] = size_report
        all_ok &= not failures
        logger.info(
            "Parity @%d: onnx raw %.1e xy %.3f px score %.1e | ncnn(fp16=%s) raw %.1e xy p95 %.3f p99 %.3f max %.3f px"
            " | organs host %.4f onnx %.4f ncnn p99 %.3f px%s",
            size, summary["onnx_raw_max_abs"], summary["onnx_xy_max_px"], summary["onnx_score_max_abs"], fp16,
            summary["ncnn_raw_max_abs"], summary["ncnn_xy_p95_px"], summary["ncnn_xy_p99_px"],
            summary["ncnn_xy_max_px"], summary["host_organs_max_px"], summary["onnx_organs_max_px"],
            summary["ncnn_organs_p99_px"], "" if not failures else f"  FAIL {failures}",
        )

    parity["pass"] = bool(all_ok) if verify else None
    parity_path = None
    if verify:
        parity_path = out_dir / PARITY_NAME
        parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")

    parity_summary = {str(s): r["summary"] for s, r in parity["sizes"].items()} if verify else None
    manifest = build_manifest(
        model_id=model_id,
        variant_spec=spec.to_dict(),
        strides=strides,
        levels=spec.levels,
        sizes=list(int(s) for s in sizes),
        num_keypoints=num_keypoints,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=sha256_json(config) if config else None,
        onnx_files=onnx_files,
        ncnn_files=ncnn_files,
        onnx_opset=opset,
        input_name=INPUT_NAME,
        onnx_output_names=OUTPUT_NAMES,
        ncnn_blobs=blobs,
        converter=converter,
        fp16=fp16,
        decode=decode_cfg,
        tools=tool_versions({"torch": torch.__version__}),
        parity={"pass": parity["pass"], "file": PARITY_NAME if verify else None, "per_size": parity_summary},
    )
    manifest_path = write_manifest(out_dir / MANIFEST_NAME, manifest)
    logger.info("Export finished in %.1f s -> %s", time.time() - t_start, out_dir)
    if verify and strict and not all_ok:
        raise ParityError(f"Parity check failed, see {parity_path}")
    return {
        "out_dir": out_dir,
        "manifest": manifest_path,
        "parity": parity_path,
        "files": {s: {"onnx": onnx_files[s], "ncnn": ncnn_files[s]} for s in onnx_files},
        "parity_summary": parity_summary,
        "ok": bool(all_ok) if verify else None,
        "model_id": model_id,
    }


def _load_checkpoint_model(weights: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Model (EMA weights, unfused) + frozen config of a training checkpoint."""

    from ..engine.checkpoint import load_checkpoint

    ckpt = load_checkpoint(weights, map_location="cpu")
    config = dict(ckpt.get("config") or {})
    model = None
    try:  # preferred path: the public API (CONTRACTS 11)
        from ..api import load_model

        loaded = load_model(weights, device="cpu", fuse=False)
        model = getattr(loaded, "model", loaded)
        if not hasattr(model, "fuse") or not hasattr(model, "spec"):
            model = None
    except (ImportError, AttributeError, TypeError) as exc:
        logger.warning("xtx4.api.load_model unavailable (%s); rebuilding the model from the checkpoint", exc)
    if model is None:
        from ..models import build_model
        from ..models.scaling import VariantSpec

        if ckpt.get("variant_spec"):
            from ..models.xtx4 import XTX4Model

            model = XTX4Model(spec=VariantSpec.from_dict(ckpt["variant_spec"]))
        else:
            model = build_model(config)
        state = ckpt.get("ema_state") or ckpt.get("model_state")
        if state is None:
            raise RuntimeError(f"{weights} contains neither ema_state nor model_state")
        model.load_state_dict(state)
    return model.cpu().eval(), config


def export_checkpoint(
    weights: Path | str,
    out_dir: Path | str | None = None,
    sizes: Sequence[int] | None = None,
    fp16: bool = True,
    opset: int = 17,
    converter: str = "pnnx",
    verify: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export a training checkpoint (``best.pt`` / ``last.pt``) for deployment.

    ``out_dir`` defaults to ``<checkpoint dir>/export``; ``sizes`` default to
    ``config["export"]["sizes"]`` (else 256/320/384). See :func:`export_model`.
    """

    weights = Path(weights)
    checkpoint_sha256 = sha256_file(weights)
    model, config = _load_checkpoint_model(weights)
    if sizes is None:
        sizes = config.get("export", {}).get("sizes", list(DEFAULT_SIZES))
    out = Path(out_dir) if out_dir is not None else weights.parent / "export"
    logger.info("Exporting %s (sha256 %s) -> %s, sizes %s", weights, checkpoint_sha256[:12], out, list(sizes))
    return export_model(
        model, config, out, sizes=sizes, fp16=fp16, opset=opset, converter=converter, verify=verify,
        checkpoint_sha256=checkpoint_sha256, **kwargs,
    )
