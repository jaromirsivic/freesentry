"""Benchmark the XTX3 pipeline: ms/frame and FPS per phase.

Backends:
  * ``torch``  -- PyTorch (fused model). Benchmarks the model's configured
                  view mode; for u this is the single whole-frame pass.
  * ``ncnn``   -- raw NCNN (*.ncnn.param/*.ncnn.bin), the Raspberry Pi path.
  * ``xtx2``   -- the ../xtx2 project's PyTorch pipeline for the
                  "at least as fast as XTX2" comparison (skipped when absent).

Also includes a preprocessing microbenchmark of the one-shot view rendering.

Usage:
    python scripts/benchmark.py --backend torch --variant u [--weights best.pt] [--size 320]
    python scripts/benchmark.py --backend torch --variant n --mode pyramid
    python scripts/benchmark.py --backend ncnn --param xtx3_u_320.ncnn.param --bin xtx3_u_320.ncnn.bin --size 320
    python scripts/benchmark.py --preprocess-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from xtx3.utils.logging import configure_logging, get_logger  # noqa: E402

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
    """Microbenchmark: one-shot view rendering for both modes."""

    from xtx3.data.preprocess import preprocess_image

    image = _test_image()
    for mode, size in (("whole", 320), ("whole", 256), ("pyramid", 384)):
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            preprocess_image(image, mode=mode, network_size=size)
            times.append((time.perf_counter() - t0) * 1000)
        _report(f"preprocess {mode}@{size}", times)


def benchmark_torch(
    *,
    variant: str,
    weights: str | None,
    device: str,
    mode: str | None,
    size: int | None,
    iterations: int,
    warmup: int,
) -> None:
    from xtx3.api import detect_poses, load_model

    model = load_model(weights, variant=variant, device=device, fuse=True, mode=mode, network_size=size)
    image = _test_image()
    for _ in range(warmup):
        detect_poses(image, model=model)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detect_poses(image, model=model)
        times.append((time.perf_counter() - t0) * 1000)
    _report(
        f"XTX3-{variant} torch/{device} {model.view_mode}@{model.network_size} pipeline", times
    )


def benchmark_ncnn(
    *, param: str, bin_path: str, size: int, mode: str, strides: list[int],
    iterations: int, warmup: int,
) -> None:
    from xtx3.export.ncnn_export import NCNNPoseRunner

    runner = NCNNPoseRunner(
        Path(param), Path(bin_path), strides=strides, input_size=size, mode=mode
    )
    image = _test_image()
    for _ in range(warmup):
        runner.detect(image)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        runner.detect(image)
        times.append((time.perf_counter() - t0) * 1000)
    _report(f"XTX3 ncnn {mode}@{size} pipeline", times)


def benchmark_xtx2(*, variant: str, device: str, iterations: int, warmup: int) -> None:
    """Benchmark the ../xtx2 pipeline on the same input (comparison baseline)."""

    if variant == "u":
        return  # XTX2 has no u tier
    xtx2_root = Path(__file__).resolve().parents[2] / "xtx2"
    if not (xtx2_root / "xtx2").is_dir():
        logger.warning("../xtx2 project not found; skipping comparison")
        return
    sys.path.insert(0, str(xtx2_root))
    try:
        from xtx2.api import detect_poses as xtx2_detect  # type: ignore[import-not-found]
        from xtx2.api import load_model as xtx2_load  # type: ignore[import-not-found]

        model = xtx2_load(None, variant=variant, device=device, fuse=True)
        image = _test_image()
        for _ in range(warmup):
            xtx2_detect(image, model=model)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            xtx2_detect(image, model=model)
            times.append((time.perf_counter() - t0) * 1000)
        _report(f"XTX2-{variant} torch/{device} full 3-level pipeline", times)
    except Exception as exc:  # noqa: BLE001 - comparison is best-effort
        logger.warning("Could not benchmark XTX2-%s: %s", variant, exc)
    finally:
        sys.path.remove(str(xtx2_root))
        for name in [n for n in sys.modules if n == "xtx2" or n.startswith("xtx2.")]:
            del sys.modules[name]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "ncnn"], default="torch")
    parser.add_argument("--variant", choices=["u", "n", "m", "l"], default="n")
    parser.add_argument("--weights", default=None, help="Checkpoint (.pt); random weights if omitted")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mode", choices=["whole", "pyramid"], default=None)
    parser.add_argument("--size", type=int, default=None, help="Network input size")
    parser.add_argument("--param", default=None, help="NCNN .param path (backend=ncnn)")
    parser.add_argument("--bin", dest="bin_path", default=None, help="NCNN .bin path (backend=ncnn)")
    parser.add_argument("--strides", type=int, nargs="+", default=[8, 16, 32],
                        help="Head strides (add 4 for the l variant's P2 level)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--no-xtx2-compare", action="store_true", help="Skip the XTX2 comparison")
    args = parser.parse_args()

    benchmark_preprocess()
    if args.preprocess_only:
        return

    if args.backend == "torch":
        benchmark_torch(
            variant=args.variant,
            weights=args.weights,
            device=args.device,
            mode=args.mode,
            size=args.size,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        if not args.no_xtx2_compare:
            benchmark_xtx2(
                variant=args.variant,
                device=args.device,
                iterations=args.iterations,
                warmup=args.warmup,
            )
    else:
        if not args.param or not args.bin_path:
            parser.error("--param and --bin are required for backend=ncnn")
        benchmark_ncnn(
            param=args.param,
            bin_path=args.bin_path,
            size=args.size or 320,
            mode=args.mode or "whole",
            strides=args.strides,
            iterations=args.iterations,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
