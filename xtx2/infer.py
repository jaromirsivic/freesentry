"""XTX2 Pose - inference example / CLI (spec section 10.1).

Usage:
    python infer.py --weights best.pt --image scene.jpg [--variant n]
                    [--device cpu|cuda] [--conf 0.25] [--save out.jpg] [--json out.json]

When launched with no arguments (the infer.bat case) it prompts interactively
for the weights path (default: newest best.pt under ./runs) and the image path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from xtx2 import Pose, detect_poses, load_model
from xtx2.utils.keypoints import SKELETON_EDGES
from xtx2.utils.logging import configure_logging, get_logger

logger = get_logger("infer")

_COLOR_BY_LEVEL = {1: (0, 255, 0), 2: (0, 200, 255), 3: (255, 128, 0)}


def draw_poses(image: np.ndarray, poses: list[Pose], *, kpt_thr: float = 0.3) -> np.ndarray:
    """Draw bounding boxes and the 17-keypoint skeleton onto a copy of ``image``."""

    canvas = image.copy()
    for pose in poses:
        color = _COLOR_BY_LEVEL.get(pose.source_level, (0, 255, 0))
        x1, y1, x2, y2 = pose.bbox_xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"{pose.score:.2f}/L{pose.source_level}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        kpts = pose.keypoints
        for a, b in SKELETON_EDGES:
            if kpts[a, 2] >= kpt_thr and kpts[b, 2] >= kpt_thr:
                pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
        for k in range(kpts.shape[0]):
            if kpts[k, 2] >= kpt_thr:
                cv2.circle(canvas, (int(kpts[k, 0]), int(kpts[k, 1])), 3, (0, 0, 255), -1)
    return canvas


def _newest_best_checkpoint(runs_dir: Path = Path("./runs")) -> Path | None:
    if not runs_dir.is_dir():
        return None
    candidates = sorted(runs_dir.glob("**/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _prompt_weights() -> Path:
    default = _newest_best_checkpoint()
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"Weights path{suffix}: ").strip()
        if not raw and default is not None:
            return default
        path = Path(raw)
        if path.is_file():
            return path
        print("Checkpoint not found; try again.")


def _prompt_image() -> Path:
    while True:
        raw = input("Image path: ").strip()
        path = Path(raw)
        if path.is_file():
            return path
        print("Image not found; try again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XTX2 pose inference on an image.")
    parser.add_argument("--weights", default=None, help="Path to a trained checkpoint (.pt)")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--variant", choices=["n", "m", "l"], default="n")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save", default=None, help="Path to write the annotated image")
    parser.add_argument("--json", dest="json_out", default=None, help="Path to write JSON poses")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    interactive = len(sys.argv) == 1
    weights = Path(args.weights) if args.weights else _prompt_weights()
    image_path = Path(args.image) if args.image else _prompt_image()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = load_model(weights, variant=args.variant, device=args.device)
    poses = detect_poses(image, model=model, conf_threshold=args.conf, max_detections=args.max_det)
    logger.info("Detected %d people", len(poses))
    for i, pose in enumerate(poses):
        logger.info("  #%d score=%.3f level=%d bbox=%s", i, pose.score, pose.source_level,
                    np.round(pose.bbox_xyxy, 1).tolist())

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([p.to_dict() for p in poses], indent=2), encoding="utf-8"
        )
        logger.info("Wrote poses JSON -> %s", args.json_out)

    save_path = args.save
    if save_path is None and interactive:
        save_path = str(image_path.with_name(image_path.stem + "_poses.jpg"))
    if save_path:
        cv2.imwrite(save_path, draw_poses(image, poses))
        logger.info("Wrote annotated image -> %s", save_path)


if __name__ == "__main__":
    main()
