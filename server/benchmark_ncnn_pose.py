from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import ncnn
except Exception:  # pragma: no cover - optional on non-ARM/dev environments
    ncnn = None


def _load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at '{image_path}'")
    return image


def _format_result(*, label: str, timings_ms: list[float]) -> str:
    average_ms = mean(timings_ms)
    fps = 1000.0 / average_ms if average_ms > 0 else 0.0
    return f"{label}: avg={average_ms:.2f}ms fps={fps:.2f} runs={len(timings_ms)}"


def benchmark_ultralytics(
    *,
    model_dir: Path,
    image: np.ndarray,
    imgsz: int,
    warmup: int,
    runs: int,
) -> list[float]:
    model = YOLO(model_dir)
    for _ in range(warmup):
        model.predict(source=image, verbose=False, imgsz=imgsz)

    timings_ms: list[float] = []
    for _ in range(runs):
        started = perf_counter()
        model.predict(source=image, verbose=False, imgsz=imgsz)
        timings_ms.append((perf_counter() - started) * 1000.0)
    return timings_ms


def _prepare_ncnn_input(*, image: np.ndarray, imgsz: int) -> np.ndarray:
    resized = cv2.resize(image, (imgsz, imgsz))
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw


def benchmark_raw_ncnn(
    *,
    model_dir: Path,
    image: np.ndarray,
    imgsz: int,
    warmup: int,
    runs: int,
) -> list[float]:
    if ncnn is None:
        raise RuntimeError("The optional 'ncnn' Python package is not available in this environment.")

    param_path = model_dir / "model.ncnn.param"
    bin_path = model_dir / "model.ncnn.bin"
    if not param_path.exists() or not bin_path.exists():
        raise FileNotFoundError(
            f"Missing NCNN files under '{model_dir}'. Expected both 'model.ncnn.param' and 'model.ncnn.bin'."
        )

    input_tensor = _prepare_ncnn_input(image=image, imgsz=imgsz)
    timings_ms: list[float] = []
    with ncnn.Net() as net:
        net.load_param(param_path)
        net.load_model(bin_path)

        def run_once() -> None:
            with net.create_extractor() as extractor:
                extractor.input("in0", ncnn.Mat(input_tensor).clone())
                extractor.extract("out0")

        for _ in range(warmup):
            run_once()
        for _ in range(runs):
            started = perf_counter()
            run_once()
            timings_ms.append((perf_counter() - started) * 1000.0)
    return timings_ms


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Benchmark the current Ultralytics NCNN pose path and, optionally, "
            "a raw NCNN forward pass for the same exported model directory."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path, help="Path to the exported *_ncnn_model directory.")
    parser.add_argument("--image", required=True, type=Path, help="Path to an input image used for benchmarking.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations. Default: 5.")
    parser.add_argument("--runs", type=int, default=30, help="Number of measured runs. Default: 30.")
    parser.add_argument(
        "--skip-raw-ncnn",
        action="store_true",
        help="Skip the raw ncnn.Net benchmark and only measure the Ultralytics wrapper path.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    image = _load_image(args.image)
    ultralytics_timings = benchmark_ultralytics(
        model_dir=args.model_dir,
        image=image,
        imgsz=args.imgsz,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(_format_result(label="ultralytics_ncnn_wrapper", timings_ms=ultralytics_timings))

    if args.skip_raw_ncnn:
        return

    try:
        raw_ncnn_timings = benchmark_raw_ncnn(
            model_dir=args.model_dir,
            image=image,
            imgsz=args.imgsz,
            warmup=args.warmup,
            runs=args.runs,
        )
    except Exception as exc:
        print(f"raw_ncnn_forward: unavailable ({exc})")
        return

    print(_format_result(label="raw_ncnn_forward", timings_ms=raw_ncnn_timings))
    print(
        "Note: raw_ncnn_forward measures the exported network forward pass only. "
        "It excludes Ultralytics postprocess, pose decoding, drawing, and engagement logic."
    )


if __name__ == "__main__":
    main()
