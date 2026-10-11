"""Audit one dataset split: what the nine-landmark task actually sees.

Usage::

    .venv/Scripts/python.exe scripts/data_audit.py --root D:/xtxtraining --split test [--sha1]

Reports (written to ``reports/data_audit_<split>.json``):

* counts: images, persons, crowd regions, images without persons;
* labeled points per landmark and the visibility histogram (v = -1/0/1/2);
* labeled points per person;
* face spread = max pairwise distance of labeled face points (>= 2 needed), in
  source px and in network px at ``--size`` after the canonical letterbox
  (scale ``min(1, S / max(W, H))``), bucketed like the metrics
  (<8, 8-16, 16-32, >=32 px); torso spread likewise;
* persons-per-image histogram (exact and 1 / 2-4 / 5+ buckets);
* edge-truncated persons (any labeled point within 16 network px of the image
  border; bbox touching the border);
* images that qualify for oversampling (small face < 16 px and >= 3 persons);
* optional sha1 manifest (``--sha1``; ``reports/manifest_<split>.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtx4.data.annotations import DatasetSpec, read_image_size, scan_split  # noqa: E402
from xtx4.data.manifest import build_manifest, write_manifest  # noqa: E402
from xtx4.utils.keypoints import FACE_INDICES, KEYPOINT_NAMES, NUM_KEYPOINTS, TORSO_INDICES  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("xtx4.scripts.data_audit")

NET_BUCKETS = ((0, 8, "<8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, float("inf"), ">=32"))
SRC_BUCKETS = ((0, 8, "<8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 128, "64-128"),
               (128, float("inf"), ">=128"))
EDGE_PX = 16.0


def _spread(points: np.ndarray) -> float:
    d = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def _bucket(value: float, buckets) -> str:
    for lo, hi, name in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][2]


def _hist(values: list[float], buckets) -> dict[str, int]:
    counts = {name: 0 for _, _, name in buckets}
    for v in values:
        counts[_bucket(v, buckets)] += 1
    return counts


def audit(samples, *, size: int) -> dict:
    vis_hist = np.zeros((NUM_KEYPOINTS, 4), np.int64)  # columns v = -1, 0, 1, 2
    labeled_per_person = Counter()
    persons_per_image = Counter()
    face_src, face_net, torso_net = [], [], []
    no_face = 0
    edge_points = edge_box = 0
    oversample = 0
    max_sides = []
    persons = crowd = 0
    for s in samples:
        w, h = s.width, s.height
        if w <= 0 or h <= 0:
            w, h = read_image_size(s.image_path)
        max_sides.append(max(w, h))
        scale = min(1.0, size / float(max(w, h)))
        n_persons = 0
        small_face = False
        for a in s.annotations:
            if a.iscrowd:
                crowd += 1
                continue
            n_persons += 1
            kp = a.keypoints_norm.astype(np.float64)
            v = kp[:, 2].astype(np.int64)
            vis_hist[np.arange(NUM_KEYPOINTS), v + 1] += 1
            lab = v > 0
            labeled_per_person[int(lab.sum())] += 1
            px = kp[:, :2] * np.array([w, h], np.float64)
            face = [k for k in FACE_INDICES if lab[k]]
            if len(face) >= 2:
                spread = _spread(px[face])
                face_src.append(spread)
                face_net.append(spread * scale)
                small_face |= spread * scale < 16
            else:
                no_face += 1
            torso = [k for k in TORSO_INDICES if lab[k]]
            if len(torso) >= 2:
                torso_net.append(_spread(px[torso]) * scale)
            net = px[lab] * scale
            lim_w, lim_h = w * scale, h * scale
            if ((net[:, 0] < EDGE_PX) | (net[:, 1] < EDGE_PX) | (net[:, 0] > lim_w - EDGE_PX)
                    | (net[:, 1] > lim_h - EDGE_PX)).any():
                edge_points += 1
            if a.bbox_xywh_norm is not None:
                bx, by, bw, bh = (float(x) for x in a.bbox_xywh_norm)
                tol_x, tol_y = 1.0 / w, 1.0 / h
                if bx <= tol_x or by <= tol_y or bx + bw >= 1 - tol_x or by + bh >= 1 - tol_y:
                    edge_box += 1
        persons += n_persons
        persons_per_image[n_persons] += 1
        if small_face and n_persons >= 3:
            oversample += 1

    labeled = vis_hist[:, 2] + vis_hist[:, 3]
    crowd_buckets = {
        "0": persons_per_image.get(0, 0),
        "1": persons_per_image.get(1, 0),
        "2-4": sum(c for n, c in persons_per_image.items() if 2 <= n <= 4),
        "5+": sum(c for n, c in persons_per_image.items() if n >= 5),
    }
    return {
        "network_size": size,
        "counts": {
            "images": len(samples),
            "persons": persons,
            "crowd_regions": crowd,
            "images_without_persons": persons_per_image.get(0, 0),
            "datasets": sorted({s.dataset for s in samples}),
        },
        "labeled_per_landmark": {
            name: {
                "labeled": int(labeled[k]),
                "labeled_frac": float(labeled[k] / max(1, persons)),
                "v=-1": int(vis_hist[k, 0]), "v=0": int(vis_hist[k, 1]),
                "v=1": int(vis_hist[k, 2]), "v=2": int(vis_hist[k, 3]),
            }
            for k, name in enumerate(KEYPOINT_NAMES)
        },
        "visibility_histogram_total": {
            "v=-1": int(vis_hist[:, 0].sum()), "v=0": int(vis_hist[:, 1].sum()),
            "v=1": int(vis_hist[:, 2].sum()), "v=2": int(vis_hist[:, 3].sum()),
        },
        "labeled_points_per_person": {str(k): int(labeled_per_person.get(k, 0)) for k in range(1, NUM_KEYPOINTS + 1)},
        "face_spread": {
            "definition": "max pairwise distance of labeled face points (>=2 labeled)",
            "persons_without_face_spread": no_face,
            "source_px": _hist(face_src, SRC_BUCKETS),
            f"net_px_at_{size}": _hist(face_net, NET_BUCKETS),
            f"median_net_px_at_{size}": float(np.median(face_net)) if face_net else None,
        },
        f"torso_spread_net_px_at_{size}": _hist(torso_net, NET_BUCKETS),
        "persons_per_image": {
            "exact": {str(k): int(v) for k, v in sorted(persons_per_image.items())},
            "buckets": crowd_buckets,
        },
        "edge_truncated": {
            "persons_labeled_point_within_16px_at_net": edge_points,
            "persons_bbox_touching_border": edge_box,
        },
        "oversample_candidates_small_face_and_3plus_persons": oversample,
        "image_max_side": {
            "min": int(min(max_sides)) if max_sides else 0,
            "median": float(np.median(max_sides)) if max_sides else 0,
            "max": int(max(max_sides)) if max_sides else 0,
            "gt_640": int(sum(1 for m in max_sides if m > 640)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--format", default="coco17")
    parser.add_argument("--name", default="xtx")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--sha1", action="store_true", help="also hash every image (manifest)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--no-check-files", action="store_true", help="skip per-image existence check")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = parser.parse_args(argv)
    configure_logging()

    t0 = time.perf_counter()
    samples = scan_split(args.root / args.split, DatasetSpec(args.name, args.format, None),
                         check_files=not args.no_check_files)
    report = audit(samples, size=args.size)
    report["split"] = args.split
    report["root"] = str(args.root)
    if args.sha1:
        manifest = build_manifest(samples, args.split, num_threads=args.threads,
                                  cache_path=args.out_dir / f".sha1cache_{args.split}.json")
        write_manifest(manifest, args.out_dir / f"manifest_{args.split}.json")
        dup = Counter(e["sha1"] for e in manifest["images"] if e["sha1"])
        report["manifest_sha1"] = manifest["manifest_sha1"]
        report["duplicate_images_within_split"] = int(sum(c - 1 for c in dup.values() if c > 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"data_audit_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["counts"]
    logger.info("images=%d persons=%d crowd=%d empty=%d (%.1fs)", c["images"], c["persons"],
                c["crowd_regions"], c["images_without_persons"], time.perf_counter() - t0)
    logger.info("labeled per landmark: %s",
                ", ".join(f"{k}={v['labeled']}" for k, v in report["labeled_per_landmark"].items()))
    logger.info("face spread @%d: %s (no face: %d)", args.size, report["face_spread"][f"net_px_at_{args.size}"],
                report["face_spread"]["persons_without_face_spread"])
    logger.info("persons per image: %s", report["persons_per_image"]["buckets"])
    logger.info("edge truncated: %s", report["edge_truncated"])
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
