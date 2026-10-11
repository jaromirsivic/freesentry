"""Content manifests: what exactly a run trained / evaluated on.

Each image is identified by the sha1 of its file bytes, so renamed copies and
cross-split duplicates are detected (``check_overlap``) and every report can
carry the manifest hash. Hashing ~45k files is I/O bound, so it runs in a
thread pool and results are cached in a sidecar JSON keyed by
``path|mtime_ns|size`` -- a re-run only hashes new or modified files.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from ..utils.logging import get_logger
from .annotations import ImageSample

logger = get_logger(__name__)

SCHEMA_VERSION = 1
_CHUNK = 1 << 20


def file_sha1(path: Path | str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stat_key(path: Path) -> tuple[str, int]:
    st = os.stat(path)
    return f"{path}|{st.st_mtime_ns}|{st.st_size}", int(st.st_size)


def hash_files(
    paths: Sequence[Path | str], *, num_threads: int = 16, cache_path: Path | str | None = None
) -> dict[str, tuple[str, int]]:
    """Return ``{str(path): (sha1, bytes)}``; missing files map to ``("", 0)``.

    ``cache_path`` (sidecar JSON) is read and rewritten with the current entries.
    """

    cache: dict[str, str] = {}
    if cache_path is not None and Path(cache_path).is_file():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable sha1 cache %s", cache_path)
            cache = {}

    def work(p: Path | str) -> tuple[str, str, str, int]:
        path = Path(p)
        try:
            key, size = _stat_key(path)
        except OSError:
            return str(path), "", "", 0
        sha = cache.get(key)
        if sha is None:
            sha = file_sha1(path)
        return str(path), key, sha, size

    results: dict[str, tuple[str, int]] = {}
    new_cache: dict[str, str] = {}
    hashed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_threads))) as pool:
        for path, key, sha, size in pool.map(work, paths, chunksize=64):
            results[path] = (sha, size)
            if key:
                if key not in cache:
                    hashed += 1
                new_cache[key] = sha
    logger.info("sha1: %d files (%d hashed, %d from cache)", len(results), hashed, len(results) - hashed)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        merged = {**cache, **new_cache}
        Path(cache_path).write_text(json.dumps(merged), encoding="utf-8")
    return results


def build_manifest(
    samples: Sequence[ImageSample],
    split_name: str,
    *,
    num_threads: int = 16,
    cache_path: Path | str | None = None,
    compute_sha1: bool = True,
) -> dict:
    """Manifest dict: per image path, sha1, bytes, size, person count, dataset.

    ``manifest_sha1`` hashes the sorted ``(sha1, persons)`` list, i.e. it changes
    whenever content or labels change but not when files are moved.
    """

    hashes = (
        hash_files([s.image_path for s in samples], num_threads=num_threads, cache_path=cache_path)
        if compute_sha1
        else {}
    )
    entries = []
    for s in samples:
        sha, size = hashes.get(str(s.image_path), ("", 0))
        entries.append({
            "path": str(s.image_path),
            "sha1": sha,
            "bytes": size,
            "width": int(s.width),
            "height": int(s.height),
            "persons": sum(1 for a in s.annotations if not a.iscrowd),
            "crowd": sum(1 for a in s.annotations if a.iscrowd),
            "dataset": s.dataset,
        })
    digest = hashlib.sha1()
    for sha, persons, path in sorted((e["sha1"], e["persons"], e["path"]) for e in entries):
        digest.update(f"{sha or path}:{persons};".encode())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "num_images": len(entries),
        "num_persons": sum(e["persons"] for e in entries),
        "datasets": sorted({e["dataset"] for e in entries}),
        "has_sha1": bool(compute_sha1),
        "manifest_sha1": digest.hexdigest(),
        "images": entries,
    }
    return manifest


def write_manifest(manifest: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("Wrote manifest %s (%d images, sha1 %s)", path, manifest["num_images"], manifest["manifest_sha1"])
    return path


def load_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_overlap(manifest_a: dict, manifest_b: dict) -> list[str]:
    """Error strings for byte-identical images shared by two manifests (empty -> no leak)."""

    if not manifest_a.get("has_sha1", True) or not manifest_b.get("has_sha1", True):
        raise ValueError("check_overlap needs manifests built with compute_sha1=True")
    by_sha: dict[str, list[str]] = {}
    for e in manifest_a["images"]:
        if e["sha1"]:
            by_sha.setdefault(e["sha1"], []).append(e["path"])
    errors = []
    for e in manifest_b["images"]:
        if e["sha1"] and e["sha1"] in by_sha:
            errors.append(
                f"identical content {e['sha1']}: {manifest_a.get('split')}:{by_sha[e['sha1']][0]} "
                f"== {manifest_b.get('split')}:{e['path']}"
            )
    if errors:
        logger.error("%d images shared between %s and %s", len(errors), manifest_a.get("split"), manifest_b.get("split"))
    return errors
