"""Export a trained XTX4 checkpoint for deployment (ONNX + ncnn + manifest + parity).

Usage:
    python scripts/export_checkpoint.py --weights D:/xtx4_runs/xtx4-u/best.pt
    python scripts/export_checkpoint.py --weights best.pt --out D:/xtx4_runs/xtx4-u/export \
        --sizes 256 320 384 --fp16 --converter pnnx

Writes ``xtx4_<S>.onnx``, ``xtx4_<S>.ncnn.param/.bin``, ``manifest.json`` and
``parity.json`` into ``--out`` (default ``<checkpoint dir>/export``). Exit code 3
when the parity check fails (files are still written for inspection).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtx4.export.deploy import DEFAULT_PARITY_IMAGES, ParityError, export_checkpoint  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("export_checkpoint")


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Training checkpoint (best.pt / last.pt)")
    parser.add_argument("--out", default=None, help="Output folder (default: <checkpoint dir>/export)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="Input sizes (default: config export.sizes, else 256 320 384)")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="FP16 ncnn weights (default)")
    precision.add_argument("--fp32", dest="fp16", action="store_false", help="FP32 ncnn weights (reference)")
    parser.add_argument("--converter", default="pnnx", choices=["pnnx", "onnx2ncnn"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true", help="Skip the ONNX/ncnn parity checks")
    parser.add_argument("--no-strict", action="store_true", help="Do not fail on parity tolerance violations")
    parser.add_argument("--parity-images", default=str(DEFAULT_PARITY_IMAGES),
                        help="Folder with images for the parity check (first image is used)")
    parser.add_argument("--prefix", default="xtx4", help="File name prefix (<prefix>_<size>.*)")
    args = parser.parse_args()

    try:
        result = export_checkpoint(
            args.weights, args.out, args.sizes, fp16=args.fp16, opset=args.opset, converter=args.converter,
            verify=not args.no_verify, strict=not args.no_strict, parity_images=args.parity_images,
            prefix=args.prefix,
        )
    except ParityError as exc:
        logger.error("%s", exc)
        return 3
    logger.info("Manifest: %s", result["manifest"])
    if result["parity_summary"]:
        logger.info("Parity summary:\n%s", json.dumps(result["parity_summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
