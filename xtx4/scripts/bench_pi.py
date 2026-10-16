"""Raspberry Pi 5 latency benchmark for exported ncnn graphs (torch-free).

Implements the measurement protocol of ``implementation_plan_fable.md`` section 4
for the **network only** (``extractor.input`` + ``extract`` of every output):

* every graph listed in ``bench/graphs/index.json`` (produced by
  ``scripts/export_random.py``; entries ``graphs[*].ncnn.fp16`` / ``.fp32``,
  paths relative to ``bench/``) or given explicitly via ``--param/--bin``;
* thread counts 1..4, FP16 arithmetic on/off (storage stays FP16 when the
  weights are FP16);
* warm-up iterations, then ``--loops`` timed iterations on real frames (a
  directory of images, letterboxed to the graph input size) or on synthetic
  noise (``--synthetic``);
* p50/p90/p95/p99/mean in ms, RSS, CPU temperature and throttling flags
  (``vcgencmd``) before and after each run.

Results are appended to a CSV (``--out``). Only ``numpy``, ``cv2``, ``ncnn``
and ``psutil`` are required on the device::

    python scripts/bench_pi.py --index bench/graphs/index.json --images ~/frames \
        --threads 1 2 3 4 --fp16 on off --loops 300 --out reports/pi_latency.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - synthetic mode works without OpenCV
    cv2 = None


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


def _cpu_governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Canonical XTX letterbox (downscale only, centred, offsets aligned to 4)."""

    height, width = image.shape[:2]
    scale = min(1.0, size / float(max(width, height)))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    paste_x = ((size - resized_w) // 2 // 4) * 4
    paste_y = ((size - resized_h) // 2 // 4) * 4
    out = np.zeros((size, size, 3), dtype=np.uint8)
    if (resized_w, resized_h) != (width, height):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    out[paste_y : paste_y + resized_h, paste_x : paste_x + resized_w] = image
    return out


def _load_frames(images_dir: Path | None, size: int, count: int, seed: int = 0) -> list[np.ndarray]:
    if images_dir is None or cv2 is None:
        rng = np.random.default_rng(seed)
        return [rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8) for _ in range(count)]
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")
    frames = []
    for path in paths[:count]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frames.append(_letterbox(image, size))
    if not frames:
        raise SystemExit(f"No readable images in {images_dir}")
    return frames


def _output_names(param_path: Path) -> list[str]:
    """Return the output blob names of an ncnn param file (blobs never consumed)."""

    produced: list[str] = []
    consumed: set[str] = set()
    lines = param_path.read_text(encoding="utf-8").splitlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        num_in, num_out = int(parts[2]), int(parts[3])
        blobs = parts[4 : 4 + num_in + num_out]
        consumed.update(blobs[:num_in])
        produced.extend(blobs[num_in:])
    return [b for b in produced if b not in consumed]



def _index_graph_files(entry: dict) -> list[tuple[str, str, str]]:
    """Return ``(precision, param, bin)`` triples for one index entry.

    Supports the ``export_random.py`` schema (``entry["ncnn"]["fp16"|"fp32"]``)
    and the flat legacy form (``ncnn_param`` / ``ncnn_bin``).
    """

    out: list[tuple[str, str, str]] = []
    ncnn_block = entry.get("ncnn")
    if isinstance(ncnn_block, dict):
        for precision in ("fp16", "fp32"):
            block = ncnn_block.get(precision)
            if isinstance(block, dict) and block.get("param") and block.get("bin"):
                out.append((precision, str(block["param"]), str(block["bin"])))
    if not out and entry.get("ncnn_param") and entry.get("ncnn_bin"):
        out.append(("fp16", str(entry["ncnn_param"]), str(entry["ncnn_bin"])))
    return out


def _resolve_index_path(index_path: Path, rel: str) -> Path | None:
    """Index paths are relative to ``bench/`` (the index lives in ``bench/graphs``);
    fall back to the file name next to the index so a copied ``graphs/`` dir works too."""

    rel = rel.replace("\\", "/")
    candidates = [Path(rel)] if Path(rel).is_absolute() else [
        index_path.parent.parent / rel,
        index_path.parent / rel,
        index_path.parent / Path(rel).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def bench_graph(
    *,
    param: Path,
    bin_path: Path,
    size: int,
    threads: int,
    fp16: bool,
    frames: list[np.ndarray],
    warmup: int,
    loops: int,
    input_blob: str = "in0",
) -> dict:
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = fp16
    net.opt.use_fp16_storage = fp16
    net.opt.use_fp16_arithmetic = fp16
    net.opt.use_packing_layout = True
    net.opt.lightmode = True
    if net.load_param(str(param)) != 0 or net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"ncnn failed to load {param}")
    outputs = _output_names(param)
    norm = [1 / 255.0] * 3

    def run(frame: np.ndarray) -> None:
        mat = ncnn.Mat.from_pixels(frame, ncnn.Mat.PixelType.PIXEL_BGR, frame.shape[1], frame.shape[0])
        mat.substract_mean_normalize([], norm)
        with net.create_extractor() as ex:
            ex.input(input_blob, mat)
            for name in outputs:
                ret, _ = ex.extract(name)
                if ret != 0:
                    raise RuntimeError(f"extract {name} failed ({ret})")

    temp_before = _vcgencmd("measure_temp")
    throttled_before = _vcgencmd("get_throttled")
    for i in range(warmup):
        run(frames[i % len(frames)])
    times = np.empty(loops, dtype=np.float64)
    for i in range(loops):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        run(frame)
        times[i] = (time.perf_counter() - t0) * 1000.0
    temp_after = _vcgencmd("measure_temp")
    throttled_after = _vcgencmd("get_throttled")
    net.clear()
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(times.mean()),
        "min_ms": float(times.min()),
        "rss_mb": _rss_mb(),
        "temp_before": temp_before,
        "temp_after": temp_after,
        "throttled_before": throttled_before,
        "throttled_after": throttled_after,
        "outputs": ";".join(outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=None, help="bench/graphs/index.json from export_random.py")
    parser.add_argument("--param", default=None, help="Single graph .ncnn.param (with --bin and --size)")
    parser.add_argument("--bin", default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--name", default=None, help="Label for the single-graph mode")
    parser.add_argument("--images", default=None, help="Directory with real frames (letterboxed per graph size)")
    parser.add_argument("--synthetic", action="store_true", help="Use random noise frames")
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--fp16", nargs="+", choices=["on", "off"], default=["on"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--loops", type=int, default=300)
    parser.add_argument("--filter", default=None, help="Substring filter on graph names from the index")
    parser.add_argument("--precision", choices=["fp16", "fp32", "all"], default="fp16",
                        help="Which weight precision from the index to use (fp32 graphs only if exported)")
    parser.add_argument("--out", default="reports/pi_latency.csv")
    parser.add_argument("--cool-below", type=float, default=None, metavar="C",
                        help="Before each config wait until the SoC temperature (vcgencmd) is below C")
    parser.add_argument("--cool-max-wait", type=float, default=180.0, help="Max seconds to wait for cooling")
    args = parser.parse_args()

    graphs: list[dict] = []
    if args.index:
        index_path = Path(args.index)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index["graphs"] if isinstance(index, dict) and "graphs" in index else index
        for entry in entries:
            name = entry.get("name") or entry.get("id", "")
            if args.filter and args.filter not in name:
                continue
            for precision, param_rel, bin_rel in _index_graph_files(entry):
                if args.precision != "all" and precision != args.precision:
                    continue
                param_path = _resolve_index_path(index_path, param_rel)
                bin_path = _resolve_index_path(index_path, bin_rel)
                if param_path is None or bin_path is None:
                    print(f"skip {name} ({precision}): files not found ({param_rel})", file=sys.stderr)
                    continue
                graphs.append(
                    {
                        "name": f"{name}" if precision == "fp16" else f"{name}_{precision}w",
                        "size": int(entry["size"]),
                        "param": param_path,
                        "bin": bin_path,
                        "gmac": entry.get("gmac"),
                    }
                )
    if args.param:
        if not (args.bin and args.size):
            parser.error("--param requires --bin and --size")
        graphs.append({"name": args.name or Path(args.param).stem, "size": args.size, "param": Path(args.param), "bin": Path(args.bin), "gmac": None})
    if not graphs:
        parser.error("Nothing to benchmark: pass --index or --param/--bin/--size")

    images_dir = None if args.synthetic or not args.images else Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    fields = [
        "timestamp", "host", "machine", "governor", "ncnn_version", "graph", "size", "gmac", "threads", "fp16",
        "frames", "warmup", "loops", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "min_ms", "rss_mb",
        "temp_before", "temp_after", "throttled_before", "throttled_after", "outputs",
    ]
    import ncnn

    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for graph in graphs:
            frames = _load_frames(images_dir, graph["size"], args.num_frames)
            for threads in args.threads:
                for fp16 in args.fp16:
                    wait_until_cool(args.cool_below, args.cool_max_wait)
                    result = bench_graph(
                        param=graph["param"], bin_path=graph["bin"], size=graph["size"], threads=threads,
                        fp16=(fp16 == "on"), frames=frames, warmup=args.warmup, loops=args.loops,
                    )
                    row = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "host": platform.node(),
                        "machine": platform.machine(),
                        "governor": _cpu_governor(),
                        "ncnn_version": getattr(ncnn, "__version__", ""),
                        "graph": graph["name"],
                        "size": graph["size"],
                        "gmac": graph["gmac"],
                        "threads": threads,
                        "fp16": fp16,
                        "frames": len(frames),
                        "warmup": args.warmup,
                        "loops": args.loops,
                        **result,
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(
                        f"{graph['name']:32s} S={graph['size']} thr={threads} fp16={fp16}: "
                        f"p50={result['p50_ms']:.2f} ms p95={result['p95_ms']:.2f} ms mean={result['mean_ms']:.2f} ms",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
