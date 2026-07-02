"""Benchmark the full 3-level XTX2 pipeline: ms/frame and FPS (spec section 10.2).

Backends:
  * ``torch``  -- PyTorch (fused model, one batched (3,3,384,384) forward).
  * ``ncnn``   -- raw NCNN (*.ncnn.param/*.ncnn.bin), for the Raspberry Pi path.
  * ``xtx``    -- the old ./xtx project's PyTorch pipeline, for the
                  "faster than XTX" comparison (skipped when unavailable).

Also includes a preprocessing microbenchmark validating the one-shot level
rendering against the naive materialise-full-canvas path (acceptance
criterion 3).

Usage:
    python scripts/benchmark.py --backend torch [--variant n] [--weights best.pt]
    python scripts/benchmark.py --backend ncnn --param xtx2_n.ncnn.param --bin xtx2_n.ncnn.bin
    python scripts/benchmark.py --preprocess-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from xtx2.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("benchmark")


def _test_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _report(name: str, times_ms: list[float]) -> None:
    arr = np.asarray(times_ms)
    logger.info(
        "%s: mean=%.2f ms/frame (%.1f FPS) | median=%.2f | p90=%.2f | n=%d",
        name, arr.mean(), 1000.0 / max(arr.mean(), 1e-9), np.median(arr),
        np.percentile(arr, 90), arr.size,
    )


def benchmark_preprocess(*, iterations: int = 200) -> None:
    """Microbenchmark: one-shot level rendering vs naive full-canvas path."""

    from xtx2.data.preprocess import build_canvas, build_levels, compute_placement

    image = _test_image()
    height, width = image.shape[:2]
    placement = compute_placement(width=width, height=height)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        build_levels(image, placement)
        times.append((time.perf_counter() - t0) * 1000)
    _report("preprocess one-shot (levels 1/2/3)", times)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        canvas = build_canvas(image, placement)
        cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_AREA)
        cv2.resize(canvas[384:1152, 384:1152], (384, 384), interpolation=cv2.INTER_AREA)
        np.ascontiguousarray(canvas[576:960, 576:960])
        times.append((time.perf_counter() - t0) * 1000)
    _report("preprocess naive (full canvas + 3 resizes)", times)


def benchmark_torch(
    *, variant: str, weights: str | None, device: str, iterations: int, warmup: int
) -> None:
    from xtx2.api import detect_poses, load_model

    model = load_model(weights, variant=variant, device=device, fuse=True)
    image = _test_image()
    for _ in range(warmup):
        detect_poses(image, model=model)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detect_poses(image, model=model)
        times.append((time.perf_counter() - t0) * 1000)
    _report(f"XTX2-{variant} torch/{device} full 3-level pipeline", times)


def benchmark_ncnn(*, param: str, bin_path: str, iterations: int, warmup: int) -> None:
    from xtx2.export.ncnn_export import NCNNPoseRunner

    runner = NCNNPoseRunner(Path(param), Path(bin_path), strides=[8, 16, 32])
    image = _test_image()
    for _ in range(warmup):
        runner.detect(image)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        runner.detect(image)
        times.append((time.perf_counter() - t0) * 1000)
    _report("XTX2-n ncnn full 3-level pipeline", times)


def benchmark_old_xtx(*, variant: str, device: str, iterations: int, warmup: int) -> None:
    """Benchmark the old ./xtx pipeline on the same input (comparison baseline)."""

    xtx_root = Path(__file__).resolve().parents[2] / "xtx"
    if not (xtx_root / "xtx").is_dir():
        logger.warning("Old ./xtx project not found; skipping comparison")
        return
    sys.path.insert(0, str(xtx_root))
    try:
        from xtx.api import detect_poses as xtx_detect  # type: ignore[import-not-found]
        from xtx.api import load_model as xtx_load  # type: ignore[import-not-found]

        model = xtx_load(None, variant=variant, device=device, fuse=True)
        image = _test_image()
        for _ in range(warmup):
            xtx_detect(image, model=model)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            xtx_detect(image, model=model)
            times.append((time.perf_counter() - t0) * 1000)
        _report(f"old XTX-{variant} torch/{device} full A/B/C pipeline", times)
    except Exception as exc:  # noqa: BLE001 - comparison is best-effort
        logger.warning("Could not benchmark old XTX-%s: %s", variant, exc)
    finally:
        sys.path.remove(str(xtx_root))
        for name in [n for n in sys.modules if n == "xtx" or n.startswith("xtx.")]:
            del sys.modules[name]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "ncnn"], default="torch")
    parser.add_argument("--variant", choices=["n", "m", "l"], default="n")
    parser.add_argument("--weights", default=None, help="Checkpoint (.pt); random weights if omitted")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--param", default=None, help="NCNN .param path (backend=ncnn)")
    parser.add_argument("--bin", dest="bin_path", default=None, help="NCNN .bin path (backend=ncnn)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--no-xtx-compare", action="store_true", help="Skip the old-XTX comparison")
    args = parser.parse_args()

    benchmark_preprocess()
    if args.preprocess_only:
        return

    if args.backend == "torch":
        benchmark_torch(
            variant=args.variant,
            weights=args.weights,
            device=args.device,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        if not args.no_xtx_compare:
            benchmark_old_xtx(
                variant=args.variant,
                device=args.device,
                iterations=args.iterations,
                warmup=args.warmup,
            )
    else:
        if not args.param or not args.bin_path:
            parser.error("--param and --bin are required for backend=ncnn")
        benchmark_ncnn(
            param=args.param, bin_path=args.bin_path,
            iterations=args.iterations, warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
