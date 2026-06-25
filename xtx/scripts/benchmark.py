"""Benchmark the full 3-crop XTX pipeline (FPS / ms-per-frame).

Usage:
    python scripts/benchmark.py --weights best.pt --variant n --device cpu
    python scripts/benchmark.py --weights xtx_n --device ncnn        # Raspberry Pi
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from xtx.utils.logging import configure_logging, get_logger

logger = get_logger("benchmark")


def _make_image(size: int = 720) -> np.ndarray:
    return (np.random.rand(size, int(size * 1.5), 3) * 255).astype(np.uint8)


def benchmark_torch(args: argparse.Namespace) -> None:
    from xtx import detect_poses, load_model

    model = load_model(args.weights, variant=args.variant, device=args.device)
    image = _make_image()
    for _ in range(args.warmup):
        detect_poses(image, model=model, conf_threshold=args.conf)

    start = time.perf_counter()
    for _ in range(args.iters):
        detect_poses(image, model=model, conf_threshold=args.conf)
    elapsed = time.perf_counter() - start
    _report(elapsed, args.iters)


def benchmark_ncnn(args: argparse.Namespace) -> None:
    from xtx.export.ncnn_export import NCNNPoseRunner

    runner = NCNNPoseRunner(
        param_path=f"{args.weights}.ncnn.param",
        bin_path=f"{args.weights}.ncnn.bin",
        strides=[8, 16, 32],
    )
    image = _make_image()
    for _ in range(args.warmup):
        runner.detect(image, conf_threshold=args.conf)

    start = time.perf_counter()
    for _ in range(args.iters):
        runner.detect(image, conf_threshold=args.conf)
    elapsed = time.perf_counter() - start
    _report(elapsed, args.iters)


def _report(elapsed: float, iters: int) -> None:
    ms_per_frame = elapsed / iters * 1000.0
    fps = iters / elapsed
    logger.info("Full 3-crop pipeline: %.2f ms/frame | %.2f FPS over %d iters", ms_per_frame, fps, iters)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Checkpoint path or NCNN prefix")
    parser.add_argument("--variant", choices=["n", "m", "l"], default="n")
    parser.add_argument("--device", choices=["cpu", "cuda", "ncnn"], default="cpu")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.device == "ncnn":
        benchmark_ncnn(args)
    else:
        benchmark_torch(args)


if __name__ == "__main__":
    main()
