"""Export randomly initialised XTX4 graphs for Raspberry Pi 5 latency benchmarks (T2).

Latency depends on topology, not on trained weights, so every architecture
candidate is built with deterministic random weights (seed 0), fused, exported
to ONNX (onnxsim + checker + onnxruntime parity) and converted to ncnn with
pnnx (fp16=1 by default, additionally fp16=0 with ``--fp32``). Every graph is
described in ``<out>/index.json`` (spec, size, N, strides, blob names, file
paths, GMAC, parity numbers) for the Pi benchmark harness.

Spec strings: ``<preset>[:key=value[,key=value...]]`` with the CONTRACTS 4.1
override keys, e.g. ``u_b:levels=P3P5``, ``u_b:act=hswish``, ``u_b:p2_assist=true``.

Usage:
    python scripts/export_random.py --specs u_a u_b:levels=P3P5 --sizes 256 320 384 --out bench/graphs [--fp32]
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from xtx4.export.onnx_ncnn import (  # noqa: E402
    INPUT_NAME,
    OUTPUT_NAMES,
    convert_onnx_to_ncnn,
    export_onnx,
    find_pnnx,
    onnx_op_types,
    verify_ncnn,
    verify_onnx,
)
from xtx4.models import build_model  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("export_random")

DEFAULT_SPECS = [
    "u_a", "u_b", "u_c", "xtx3u_ctrl",
    "u_b:levels=P3P5", "u_b:levels=P4P5", "u_b:act=hswish", "u_b:act=silu", "u_b:p2_assist=true",
]
DEFAULT_SIZES = (256, 320, 384)


def parse_spec(text: str) -> tuple[str, dict[str, Any]]:
    """``"u_b:levels=P3P5,act=silu"`` -> ``("u_b", {"levels": "P3P5", "act": "silu"})``."""

    name, _, rest = text.partition(":")
    overrides: dict[str, Any] = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key in ("p2_assist",):
            overrides[key] = value.lower() in ("1", "true", "yes", "on")
        elif key in ("head_hidden", "pose_hidden", "num_keypoints", "default_size"):
            overrides[key] = int(value)
        elif key in ("widths", "depths"):
            overrides[key] = [int(v) for v in value.replace("/", "-").split("-")]
        else:
            overrides[key] = value
    return name.strip(), overrides


def graph_id(spec_name: str, overrides: dict[str, Any], size: int) -> str:
    parts = [spec_name]
    for key in ("levels", "act", "block", "p2_assist", "head_hidden", "pose_hidden", "widths", "depths"):
        if key not in overrides:
            continue
        value = overrides[key]
        if key == "p2_assist":
            parts.append("p2assist" if value else "nop2assist")
        elif key in ("widths", "depths"):
            parts.append(f"{key[0]}" + "-".join(str(v) for v in value))
        else:
            parts.append(str(value))
    parts.append(str(size))
    return "_".join(parts)


def _rel(path: Path, out_dir: Path) -> str:
    """Path relative to the parent of the output dir (``graphs/x.onnx``) when possible."""

    base = out_dir.resolve().parent
    resolved = path.resolve()
    return str(resolved.relative_to(base)) if resolved.is_relative_to(base) else str(resolved)


def tool_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__, "python": platform.python_version()}
    for pkg in ("onnx", "onnxsim", "onnxruntime", "pnnx", "ncnn"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


def export_one(
    *, spec_text: str, size: int, out_dir: Path, seed: int, fp32: bool, skip_ncnn: bool,
    ncnn_verify: bool, opset: int,
) -> dict[str, Any]:
    name, overrides = parse_spec(spec_text)
    gid = graph_id(name, overrides, size)
    entry: dict[str, Any] = {"id": gid, "spec_string": spec_text, "size": size, "errors": []}
    t0 = time.time()
    torch.manual_seed(seed)
    model = build_model({"model": {"variant": name, **overrides}}).eval()
    spec = model.spec
    strides = model.strides
    per_level = [(size // s) ** 2 for s in strides]
    entry.update({
        "spec": spec.to_dict(), "strides": strides, "per_level_anchors": per_level, "N": sum(per_level),
        "num_keypoints": spec.num_keypoints, "output_layout": {"person": [1, sum(per_level), 1],
        "pose": [1, sum(per_level), spec.num_keypoints * 5]},
        "params_train": model.num_parameters(), "params_deploy": model.num_parameters(deploy=True),
        "gmac": model.mac_count(size), "seed": seed,
    })
    model.fuse()
    onnx_path = out_dir / f"{gid}.onnx"
    export_onnx(model, onnx_path, input_size=size, opset=opset)
    entry["onnx"] = {
        "path": _rel(onnx_path, out_dir),
        "input_name": INPUT_NAME, "output_names": list(OUTPUT_NAMES), "opset": opset,
        "ops": dict(sorted(onnx_op_types(onnx_path).items())),
        "verify_max_abs": verify_onnx(model, onnx_path, input_size=size),
    }
    entry["ncnn"] = {}
    if not skip_ncnn:
        for precision, fp16 in (("fp16", True), ("fp32", False)):
            if precision == "fp32" and not fp32:
                continue
            prefix = out_dir / (gid if fp16 else f"{gid}_fp32")
            try:
                conv = convert_onnx_to_ncnn(onnx_path, prefix, fp16=fp16)
                rec: dict[str, Any] = {
                    "param": _rel(conv.param_path, out_dir),
                    "bin": _rel(conv.bin_path, out_dir),
                    "input_names": conv.input_names, "output_names": conv.output_names,
                    "layer_types": dict(sorted(conv.layer_types.items())), "fp16": fp16,
                    "bin_bytes": conv.bin_path.stat().st_size,
                }
                if ncnn_verify:
                    try:
                        rec["parity_vs_torch"] = verify_ncnn(model, conv, input_size=size)
                    except ImportError:
                        rec["parity_vs_torch"] = None
                    except Exception as exc:  # noqa: BLE001 - parity is diagnostic only
                        rec["parity_error"] = f"{type(exc).__name__}: {exc}"
                        logger.warning("ncnn parity check failed for %s (%s): %s", gid, precision, exc)
                entry["ncnn"][precision] = rec
            except Exception as exc:  # noqa: BLE001 - keep the ONNX, continue with the next graph
                msg = f"pnnx {precision} failed: {type(exc).__name__}: {str(exc)[-800:]}"
                entry["errors"].append(msg)
                logger.error("%s: %s", gid, msg)
    entry["seconds"] = round(time.time() - t0, 1)
    logger.info(
        "%s: N=%d GMAC=%.3f params(deploy)=%.3fM onnx diff=%.1e ncnn=%s (%.1fs)",
        gid, entry["N"], entry["gmac"], entry["params_deploy"] / 1e6, entry["onnx"]["verify_max_abs"],
        sorted(entry["ncnn"]) or "skipped", entry["seconds"],
    )
    return entry


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--out", default="bench/graphs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp32", action="store_true", help="also convert with pnnx fp16=0")
    parser.add_argument("--no-ncnn", action="store_true", help="ONNX only")
    parser.add_argument("--no-ncnn-verify", action="store_true", help="skip the ncnn-vs-torch parity check")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_ncnn and find_pnnx() is None:
        logger.warning("pnnx not found; exporting ONNX only")
        args.no_ncnn = True

    graphs: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec_text in args.specs:
        for size in args.sizes:
            try:
                graphs.append(export_one(
                    spec_text=spec_text, size=int(size), out_dir=out_dir, seed=args.seed, fp32=args.fp32,
                    skip_ncnn=args.no_ncnn, ncnn_verify=not args.no_ncnn_verify, opset=args.opset,
                ))
            except Exception as exc:  # noqa: BLE001 - one broken graph must not stop the batch
                failures.append(f"{spec_text}@{size}: {type(exc).__name__}: {exc}")
                logger.error("Export failed for %s @%d:\n%s", spec_text, size, traceback.format_exc())
    index = {
        "schema_version": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "input": {"name": INPUT_NAME, "layout": "NCHW", "color": "BGR", "scale": "1/255"},
        "outputs": {"person": "[1, N, 1] raw logits", "pose": "[1, N, K*5] raw (dx, dy, log_sigma, present, visible)"},
        "ncnn_blob_names": {"input": "in0", "person": "out0", "pose": "out1"},
        "tools": tool_versions(),
        "graphs": graphs,
        "failures": failures,
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Wrote %s with %d graphs (%d failures)", index_path, len(graphs), len(failures))
    for f in failures:
        logger.warning("FAILED: %s", f)
    bad = [g["id"] for g in graphs if g["errors"]]
    if bad:
        logger.warning("Graphs with ncnn conversion errors (ONNX kept): %s", bad)


if __name__ == "__main__":
    main()
