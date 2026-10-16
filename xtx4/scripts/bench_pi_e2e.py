"""End-to-end latency of a deployed XTX4 export on the device (torch-free).

Unlike ``bench_pi.py`` (network only) this times the whole ``frame -> poses``
path of :class:`xtx4.runtime.NcnnPoseRunner.detect` on real frames: letterbox
preprocessing, ncnn inference, decode, optional host OKS-NMS. Timings come from
``DetectionResult.timings`` (preprocess / inference / postprocess) plus the wall
clock around the call. Run it from the repository root (or any directory that
contains the ``xtx4`` package) on the Pi::

    python scripts/bench_pi_e2e.py --export ~/xtx4_bench/export --images ~/xtx4_bench/frames \
        --sizes 256 320 384 --threads 2 3 4 --nms on off --out reports/pi_latency_e2e.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from xtx4.runtime.ncnn_runner import NcnnPoseRunner  # noqa: E402


def _vcgencmd(arg: str) -> str:
    tool = shutil.which("vcgencmd")
    if tool is None:
        return ""
    try:
        return subprocess.run([tool, arg], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""



def _cpu_temp_c() -> float | None:
    """Pi CPU temperature from ``vcgencmd measure_temp`` (``temp=54.3'C``); None when unavailable."""

    text = _vcgencmd("measure_temp")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def wait_until_cool(threshold_c: float | None, max_wait_s: float = 180.0, poll_s: float = 5.0) -> None:
    """Sleep until the SoC is below ``threshold_c`` so each config starts from a comparable thermal state."""

    if threshold_c is None:
        return
    deadline = time.time() + max_wait_s
    temp = _cpu_temp_c()
    while temp is not None and temp > threshold_c and time.time() < deadline:
        print(f"cooling: {temp:.1f} C > {threshold_c:.0f} C, waiting ...", file=sys.stderr)
        time.sleep(poll_s)
        temp = _cpu_temp_c()


def _pct(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="Export directory with manifest.json and xtx4_<size>.ncnn.*")
    parser.add_argument("--images", required=True, help="Directory with real frames (any resolution)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320])
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--nms", nargs="+", choices=["on", "off"], default=["on"])
    parser.add_argument("--conf", type=float, default=None, help="Score threshold (default: manifest decode.conf_threshold)")
    parser.add_argument("--fp16", nargs="+", choices=["on", "off"], default=["on"])
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--loops", type=int, default=200)
    parser.add_argument("--out", default="reports/pi_latency_e2e.csv")
    parser.add_argument("--cool-below", type=float, default=None, metavar="C",
                        help="Before each config wait until the SoC temperature (vcgencmd) is below C")
    parser.add_argument("--cool-max-wait", type=float, default=180.0, help="Max seconds to wait for cooling")
    args = parser.parse_args()

    export_dir = Path(args.export)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    frames = [img for img in (cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths[: args.num_frames]) if img is not None]
    if not frames:
        raise SystemExit(f"No readable images in {args.images}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    fields = [
        "timestamp", "host", "machine", "export", "size", "threads", "fp16", "nms", "conf", "frames", "warmup", "loops",
        "total_p50", "total_p90", "total_p95", "total_p99", "total_mean",
        "pre_p50", "pre_mean", "infer_p50", "infer_p95", "infer_mean", "post_p50", "post_p95", "post_mean",
        "wall_p50", "wall_p95", "fps_mean", "poses_per_frame", "temp_before", "temp_after", "throttled_after",
    ]
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for size in args.sizes:
            param = export_dir / f"xtx4_{size}.ncnn.param"
            bin_path = export_dir / f"xtx4_{size}.ncnn.bin"
            for threads in args.threads:
                for fp16 in args.fp16:
                    runner = NcnnPoseRunner(
                        param, bin_path, size=size, manifest=manifest, num_threads=threads, fp16=(fp16 == "on")
                    )
                    for nms in args.nms:
                        use_nms = nms == "on"
                        conf = args.conf if args.conf is not None else runner.default_conf_threshold
                        wait_until_cool(args.cool_below, args.cool_max_wait)
                        temp_before = _vcgencmd("measure_temp")
                        for i in range(args.warmup):
                            runner.detect(frames[i % len(frames)], conf_threshold=conf, oks_nms=use_nms, keep_views=False)
                        pre = np.empty(args.loops)
                        infer = np.empty(args.loops)
                        post = np.empty(args.loops)
                        wall = np.empty(args.loops)
                        poses = 0
                        for i in range(args.loops):
                            frame = frames[i % len(frames)]
                            t0 = time.perf_counter()
                            result = runner.detect(frame, conf_threshold=conf, oks_nms=use_nms, keep_views=False)
                            wall[i] = (time.perf_counter() - t0) * 1000.0
                            pre[i] = result.timings.preprocess_ms
                            infer[i] = result.timings.inference_ms
                            post[i] = result.timings.postprocess_ms
                            poses += len(result.poses)
                        total = pre + infer + post
                        tp, pp, ip, qp, wp = _pct(total), _pct(pre), _pct(infer), _pct(post), _pct(wall)
                        row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "host": platform.node(),
                            "machine": platform.machine(),
                            "export": str(export_dir),
                            "size": size,
                            "threads": threads,
                            "fp16": fp16,
                            "nms": nms,
                            "conf": conf,
                            "frames": len(frames),
                            "warmup": args.warmup,
                            "loops": args.loops,
                            "total_p50": tp["p50"], "total_p90": tp["p90"], "total_p95": tp["p95"], "total_p99": tp["p99"],
                            "total_mean": tp["mean"],
                            "pre_p50": pp["p50"], "pre_mean": pp["mean"],
                            "infer_p50": ip["p50"], "infer_p95": ip["p95"], "infer_mean": ip["mean"],
                            "post_p50": qp["p50"], "post_p95": qp["p95"], "post_mean": qp["mean"],
                            "wall_p50": wp["p50"], "wall_p95": wp["p95"],
                            "fps_mean": 1000.0 / wp["mean"] if wp["mean"] > 0 else float("nan"),
                            "poses_per_frame": poses / args.loops,
                            "temp_before": temp_before,
                            "temp_after": _vcgencmd("measure_temp"),
                            "throttled_after": _vcgencmd("get_throttled"),
                        }
                        writer.writerow(row)
                        handle.flush()
                        print(
                            f"S={size} thr={threads} fp16={fp16} nms={nms}: total p50={tp['p50']:.2f} ms "
                            f"(pre {pp['p50']:.2f} + infer {ip['p50']:.2f} + post {qp['p50']:.2f}), "
                            f"wall p95={wp['p95']:.2f} ms, {row['fps_mean']:.1f} fps, {row['poses_per_frame']:.2f} poses/frame",
                            file=sys.stderr,
                        )
                    runner.close()


if __name__ == "__main__":
    main()
