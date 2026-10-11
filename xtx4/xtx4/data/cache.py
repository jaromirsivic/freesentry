"""One-off image cache: downscale large sources so decoding is not the training bottleneck.

Images whose longer side exceeds ``max_side`` are resized with ``INTER_AREA``
and re-encoded as JPEG q95; smaller images are copied byte-for-byte. Because
annotations are normalised (``x/W``, ``y/H``), they stay valid; only
``image_path`` / ``width`` / ``height`` of the returned samples change.

The cache is idempotent: an index JSON in ``cache_dir`` records, per source
``path|mtime_ns|size|max_side``, the cached file and its size; entries whose
cached file still exists are reused without decoding anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..utils.logging import get_logger
from .annotations import ImageSample, read_image_size

logger = get_logger(__name__)

INDEX_NAME = "_cache_index.json"
JPEG_QUALITY = 95


def _cache_name(src: Path, resized: bool) -> str:
    folder = hashlib.sha1(str(src.parent).encode("utf-8")).hexdigest()[:10]
    ext = src.suffix.lower()
    if not resized:
        return f"{folder}/{src.stem}{ext}"
    stem = src.stem if ext in (".jpg", ".jpeg") else f"{src.stem}_{ext.lstrip('.')}"
    return f"{folder}/{stem}.jpg"


def _process(src: Path, dst_root: Path, max_side: int, width: int, height: int) -> tuple[str, int, int]:
    if width <= 0 or height <= 0:
        width, height = read_image_size(src)
    if max(width, height) <= max_side:
        rel = _cache_name(src, resized=False)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return rel, width, height
    buf = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot decode {src}")
    h, w = img.shape[:2]
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"cannot encode {src}")
    rel = _cache_name(src, resized=True)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp{threading.get_ident()}")
    enc.tofile(str(tmp))
    os.replace(tmp, dst)
    return rel, nw, nh


def build_cache(
    samples: Sequence[ImageSample],
    cache_dir: Path | str,
    max_side: int = 640,
    *,
    num_threads: int = 8,
) -> list[ImageSample]:
    """Return samples pointing at cached images (same order); unreadable sources keep their path."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / INDEX_NAME
    index: dict[str, list] = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Cache index unreadable, rebuilding: %s", index_path)

    def work(sample: ImageSample) -> tuple[str | None, list | None]:
        src = Path(sample.image_path)
        try:
            st = os.stat(src)
        except OSError:
            return None, None
        key = f"{src}|{st.st_mtime_ns}|{st.st_size}|{max_side}"
        hit = index.get(key)
        if hit is not None and (cache_dir / hit[0]).is_file():
            return key, hit
        try:
            rel, w, h = _process(src, cache_dir, max_side, sample.width, sample.height)
        except OSError as exc:
            logger.warning("Cache: %s", exc)
            return None, None
        return key, [rel, w, h]

    out: list[ImageSample] = []
    created = reused = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_threads))) as pool:
        for sample, (key, entry) in zip(samples, pool.map(work, samples)):
            if key is None or entry is None:
                failed += 1
                out.append(sample)
                continue
            if key in index:
                reused += 1
            else:
                created += 1
                index[key] = entry
            rel, w, h = entry
            out.append(replace(sample, image_path=(cache_dir / rel).resolve(), width=int(w), height=int(h)))
    index_path.write_text(json.dumps(index), encoding="utf-8")
    logger.info("Cache %s: %d created, %d reused, %d failed (max_side=%d)", cache_dir, created, reused, failed, max_side)
    return out
