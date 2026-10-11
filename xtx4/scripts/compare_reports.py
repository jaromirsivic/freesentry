"""Side-by-side markdown comparison of two ``scripts/evaluate.py`` reports.

Fixed-threshold columns are misleading when the score calibration differs between models (e.g. the
XTX4 control run is more conservative than XTX3), so the table shows the detection metrics at
@0.25 and at the sweep operating points ``f1_max`` and ``fp_match`` when the report has them.
Older reports without those blocks are tolerated: missing cells print as ``-``.

Usage::

    python scripts/compare_reports.py reports/B0_xtx3u_whole_320.json reports/f3_ctrl_ep25_diag.json [--size 320]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

OPS = ("0.25", "f1_max", "fp_match")


def _get(obj: Any, *path: Any) -> Any:
    for key in path:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return "-" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _delta(a: Any, b: Any, digits: int = 4) -> str:
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
        return f"{b - a:+d}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isfinite(a) and math.isfinite(b):
            return f"{b - a:+.{digits}f}"
    return ""


def load_run(path: Path, size: int | None, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    runs = report.get("runs", [])
    if not runs:
        raise SystemExit(f"{path}: no runs")
    run = next((r for r in runs if size is None or r.get("size") == size), None)
    if run is None:
        raise SystemExit(f"{path}: no run with size {size} (have {[r.get('size') for r in runs]})")
    if variant not in run:
        raise SystemExit(f"{path}: run has no '{variant}' block (have {[k for k in run if isinstance(run[k], dict)]})")
    return report, run


def _detailed(block: dict[str, Any], op: str) -> dict[str, Any] | None:
    return block.get(f"detailed@{op}")


def _op_threshold(block: dict[str, Any], op: str) -> float | None:
    if op == "0.25":
        return 0.25
    return _get(block, "sweep", "operating_points", op, "threshold") or _get(block, f"detailed@{op}", "conf_threshold")


def build_rows(a: dict[str, Any], b: dict[str, Any], bucket_op: str) -> list[tuple[str, Any, Any, int]]:
    """(label, value_a, value_b, digits) rows; a/b are the metric blocks (``run['api']``)."""

    rows: list[tuple[str, Any, Any, int]] = []

    def add(label: str, fn: Callable[[dict[str, Any]], Any], digits: int = 4) -> None:
        rows.append((label, fn(a), fn(b), digits))

    def section(title: str) -> None:
        rows.append((f"**{title}**", "", "", 0))

    section("AP (OKS 9 sigmas)")
    for key in ("AP", "AP50", "AP75", "AR"):
        add(key, lambda m, k=key: _get(m, "AP9_project", k))
    add("AP maxDet20", lambda m: _get(m, "AP9_coco20", "AP"))
    add("GT persons", lambda m: _get(m, "AP9_project", "num_gt"), 0)
    add("detections (score>=thr)", lambda m: _get(m, "AP9_project", "num_det"), 0)

    for op in OPS:
        section(f"detection @{op}")
        add("conf threshold", lambda m, o=op: _op_threshold(m, o), 2)
        add("precision", lambda m, o=op: _get(_detailed(m, o), "detection", "precision"))
        add("recall", lambda m, o=op: _get(_detailed(m, o), "detection", "recall"))
        add("F1", lambda m, o=op: _get(_detailed(m, o), "detection", "f1"))
        add("dup/img", lambda m, o=op: _get(_detailed(m, o), "duplicates_per_image"), 3)

        def fp_per_img(m: dict[str, Any], o: str = op) -> Any:
            d = _detailed(m, o)
            fp, n = _get(d, "detection", "fp"), _get(d, "num_images")
            return fp / n if fp is not None and n else None

        def fp_split(m: dict[str, Any], o: str = op) -> Any:
            d = _detailed(m, o)
            n = _get(d, "num_images")
            parts = [_get(d, "detection", k) for k in ("fp_duplicate", "fp_localisation", "fp_background")]
            if not n or any(p is None for p in parts):
                return None
            return " / ".join(f"{p / n:.3f}" for p in parts)

        add("FP/img", fp_per_img, 3)
        add("FP/img dup / loc / bg", fp_split)

    d = lambda m: _detailed(m, bucket_op) or _detailed(m, "0.25")  # noqa: E731
    section(f"keypoints @{bucket_op}")
    add("mean matched OKS", lambda m: _get(d(m), "mean_matched_oks"))
    add("face PCK@0.1", lambda m: _get(d(m), "pck", "face@0.1"))
    add("face PCK@2px (net)", lambda m: _get(d(m), "pck", "face@2px_net"))
    add("torso PCK@0.1", lambda m: _get(d(m), "pck", "torso@0.1"))
    add("face NME median (by spread)", lambda m: _get(d(m), "face_nme_by_spread", "median"))

    section(f"face spread buckets (net px) @{bucket_op}: recall | PCK@0.1")
    for bucket in ("<8", "8-16", "16-32", ">=32", "none"):
        add(f"face {bucket}: gt", lambda m, k=bucket: _get(d(m), "face_spread_net", k, "gt"), 0)
        add(f"face {bucket}: recall", lambda m, k=bucket: _get(d(m), "face_spread_net", k, "recall"), 3)
        add(f"face {bucket}: PCK@0.1", lambda m, k=bucket: _get(d(m), "face_spread_net", k, "pck@0.1"), 3)
    section(f"torso spread buckets (net px) @{bucket_op}: recall")
    for bucket in ("<16", "16-32", "32-64", ">=64", "none"):
        add(f"torso {bucket}: recall", lambda m, k=bucket: _get(d(m), "torso_spread_net", k, "recall"), 3)

    section(f"crowd buckets @{bucket_op}: precision | recall | dup")
    for bucket in ("1", "2-4", "5+"):
        add(f"crowd {bucket}: precision", lambda m, k=bucket: _get(d(m), "crowd", k, "precision"), 3)
        add(f"crowd {bucket}: recall", lambda m, k=bucket: _get(d(m), "crowd", k, "recall"), 3)
        add(f"crowd {bucket}: duplicates", lambda m, k=bucket: _get(d(m), "crowd", k, "duplicates"), 0)

    section(f"edge @{bucket_op}")
    for bucket in ("edge", "interior"):
        add(f"{bucket}: recall", lambda m, k=bucket: _get(d(m), "edge", k, "recall"), 3)
        add(f"{bucket}: face PCK@0.1", lambda m, k=bucket: _get(d(m), "edge", k, "face_pck@0.1"), 3)

    section("visibility")
    add("macro-F1 @0.25 (default rule)", lambda m: _get(m, "detailed@0.25", "visibility", "macro_f1"))
    add("macro-F1 calibrated", lambda m: _get(m, "visibility_calibrated", "macro_f1"))
    add("calibrated present / visible thr", lambda m: (
        None if _get(m, "visibility_calibrated", "present_threshold") is None
        else f"{m['visibility_calibrated']['present_threshold']:.1f} / {m['visibility_calibrated']['visible_threshold']:.1f}"
    ))
    return rows


def render(path_a: Path, path_b: Path, size: int | None, variant: str, bucket_op: str) -> str:
    rep_a, run_a = load_run(path_a, size, variant)
    rep_b, run_b = load_run(path_b, size, variant)
    label_a = f"A: {run_a.get('name', path_a.stem)} S={run_a.get('size')} n={run_a.get('num_images')}"
    label_b = f"B: {run_b.get('name', path_b.stem)} S={run_b.get('size')} n={run_b.get('num_images')}"
    lines = [
        f"A = `{path_a}`  ",
        f"B = `{path_b}`",
        "",
    ]
    if run_a.get("num_images") != run_b.get("num_images"):
        lines += [f"> warning: different image counts ({run_a.get('num_images')} vs {run_b.get('num_images')}); "
                  "deltas compare different subsets.", ""]
    lines += [f"| metric | {label_a} | {label_b} | Δ (B−A) |", "|---|---|---|---|"]
    for label, va, vb, digits in build_rows(run_a[variant], run_b[variant], bucket_op):
        if label.startswith("**"):
            lines.append(f"| {label} | | | |")
            continue
        lines.append(f"| {label} | {_fmt(va, digits)} | {_fmt(vb, digits)} | {_delta(va, vb, max(digits, 3))} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluate.py reports as a markdown table.")
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--size", type=int, default=None, help="network size of the run to compare (default: first)")
    parser.add_argument("--variant", default="api", help="metric block of the run: api (default) or raw_o2o")
    parser.add_argument("--bucket-op", choices=OPS, default="0.25",
                        help="operating point for the keypoint/bucket sections (falls back to 0.25)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(render(args.report_a, args.report_b, args.size, args.variant, args.bucket_op))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
