"""Convert official COCO 2017 person-keypoints annotations into the XTX3 layout.

Reads ``annotations/person_keypoints_{train,val}2017.json`` plus the image
folders from an official COCO root, keeps people that retain at least one
labelled XTX3 landmark (nose, eyes, ears, shoulders, hips), normalises boxes
and keypoints to [0, 1], stores the nine landmarks (27 values), and writes::

    <out>/train/annotations.json + images
    <out>/test/annotations.json  + images   (from val2017)

The resulting folders are directly consumable by ``train.py``. Note the XTX3
annotation scanner also reads XTX2-style datasets (17 keypoints, normalised)
and raw official COCO JSONs placed next to their images, so running this
converter is optional when such data already exists (e.g. D:/xtxtraining).

Usage:
    python scripts/convert_coco.py --coco-root D:/coco --out ./dataset
    python scripts/convert_coco.py --coco-root D:/coco --out ./dataset --max-images 500 --link
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtx3.utils.keypoints import COCO_KEYPOINT_INDICES, NUM_COCO_KEYPOINTS  # noqa: E402
from xtx3.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("convert_coco")

_SPLITS = {
    "train": ("person_keypoints_train2017.json", "train2017"),
    "test": ("person_keypoints_val2017.json", "val2017"),
}


def _convert_annotation(ann: dict, width: int, height: int) -> dict | None:
    """Return a normalised nine-point XTX3 annotation dict, or None to skip."""

    if ann.get("iscrowd"):
        return None
    kpts = ann.get("keypoints")
    bbox = ann.get("bbox")
    if not isinstance(kpts, list) or len(kpts) != NUM_COCO_KEYPOINTS * 3:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    selected: list[float] = []
    labelled = 0
    for idx in COCO_KEYPOINT_INDICES:
        x, y, v = kpts[idx * 3 : idx * 3 + 3]
        v = int(v)
        if v > 0:
            labelled += 1
        selected.extend(
            [
                round(min(max(x / width, 0.0), 1.0), 6),
                round(min(max(y / height, 0.0), 1.0), 6),
                v,
            ]
        )
    if labelled == 0:
        return None  # person has no head/torso landmark labelled

    bx, by, bw, bh = bbox
    return {
        "bbox": [
            round(min(max(bx / width, 0.0), 1.0), 6),
            round(min(max(by / height, 0.0), 1.0), 6),
            round(min(max(bw / width, 0.0), 1.0), 6),
            round(min(max(bh / height, 0.0), 1.0), 6),
        ],
        "keypoints": selected,
        "iscrowd": 0,
        "category_id": 1,
    }


def convert_split(
    *,
    coco_root: Path,
    out_dir: Path,
    split: str,
    max_images: int | None,
    link: bool,
) -> None:
    ann_name, image_dir_name = _SPLITS[split]
    ann_path = coco_root / "annotations" / ann_name
    image_dir = coco_root / image_dir_name
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COCO image folder: {image_dir}")

    logger.info("Loading %s ...", ann_path)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}

    per_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        img = images.get(ann.get("image_id"))
        if img is None:
            continue
        converted = _convert_annotation(ann, int(img["width"]), int(img["height"]))
        if converted is not None:
            per_image.setdefault(int(ann["image_id"]), []).append(converted)

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    copied = 0
    for image_id, ann_list in sorted(per_image.items()):
        if max_images is not None and copied >= max_images:
            break
        img = images[image_id]
        src = image_dir / img["file_name"]
        if not src.exists():
            logger.warning("Image missing on disk, skipping: %s", src)
            continue
        dst = split_dir / img["file_name"]
        if not dst.exists():
            if link:
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        out_images.append(
            {
                "id": image_id,
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
            }
        )
        for ann in ann_list:
            out_annotations.append({"image_id": image_id, **ann})
        copied += 1

    (split_dir / "annotations.json").write_text(
        json.dumps({"images": out_images, "annotations": out_annotations}), encoding="utf-8"
    )
    logger.info(
        "%s: wrote %d images, %d people -> %s",
        split, len(out_images), len(out_annotations), split_dir,
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Convert official COCO 2017 to the XTX3 dataset layout.")
    parser.add_argument("--coco-root", required=True, help="COCO root with annotations/ train2017/ val2017/")
    parser.add_argument("--out", required=True, help="Output dataset root (train/ + test/ are created)")
    parser.add_argument("--max-images", type=int, default=None, help="Cap images per split (smoke tests)")
    parser.add_argument("--link", action="store_true", help="Hardlink images instead of copying")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_dir = Path(args.out)
    for split in ("train", "test"):
        convert_split(
            coco_root=coco_root,
            out_dir=out_dir,
            split=split,
            max_images=args.max_images,
            link=args.link,
        )


if __name__ == "__main__":
    main()
