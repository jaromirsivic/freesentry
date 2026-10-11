"""Pre-build the downscaled image cache for a dataset root with ``train/`` and ``test/`` splits.

Usage::

    .venv/Scripts/python.exe scripts/build_cache.py --root D:/xtxtraining --cache-dir E:/xtx_cache

Each split is cached into ``<cache-dir>/<split>`` (the layout the trainer uses:
``build_cache(samples, Path(cache_dir) / split, max_side)``). Re-running is
cheap: already cached files are reused via the cache index.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtx4.data.annotations import DatasetSpec, scan_split  # noqa: E402
from xtx4.data.cache import build_cache  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("xtx4.scripts.build_cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path, help="dataset root containing split folders")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--format", default="coco17", help="DatasetSpec format")
    parser.add_argument("--name", default="xtx", help="DatasetSpec name")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    configure_logging()

    spec = DatasetSpec(args.name, args.format, None)
    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.is_dir():
            logger.warning("Split %s missing under %s, skipped", split, args.root)
            continue
        t0 = time.perf_counter()
        samples = scan_split(split_dir, spec)
        cached = build_cache(samples, args.cache_dir / split, args.max_side, num_threads=args.threads)
        moved = sum(1 for a, b in zip(samples, cached) if a.image_path != b.image_path)
        logger.info("%s: %d/%d images cached in %.1fs", split, moved, len(samples), time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
