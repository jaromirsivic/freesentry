"""XTX3 Pose - interactive console training entry point.

Usage:
    python train.py                              # interactive prompts
    python train.py --variant u --dataset D:/xtxtraining
    python train.py --resume runs/xtx3-n/last.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Multi-scale training (u) constantly changes activation shapes, which fragments
# the CUDA caching allocator on Windows/WDDM ("OOM" with gigabytes reported
# free). Expandable segments avoid that; must be set before the first CUDA
# allocation. Ignored (with a warning) on builds that do not support it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from xtx3.engine.checkpoint import load_checkpoint
from xtx3.engine.trainer import (
    Trainer,
    load_config,
    prompt_dataset,
    prompt_output,
    prompt_resume,
    prompt_variant,
)
from xtx3.utils.logging import configure_logging, get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XTX3 pose model (u/n/m/l).")
    parser.add_argument("--variant", choices=["u", "n", "m", "l"], default=None)
    parser.add_argument("--dataset", default=None, help="Dataset root containing train/ and test/")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--output", default=None, help="Checkpoint output directory")
    parser.add_argument("--no-resume", action="store_true", help="Skip the resume prompt")
    parser.add_argument("--calibrate", action="store_true", help="Only run merge calibration")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(Path(args.config) if args.config else None)

    # Resume question first; silently skipped when no checkpoint exists. On
    # resume, variant and config come from the checkpoint and the
    # variant/output prompts are skipped (dataset path is still confirmed).
    resume_path = None if args.no_resume else prompt_resume(provided=args.resume)

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        variant = ckpt.get("variant", args.variant or "n")
        config = ckpt.get("config") or config
        dataset_path = prompt_dataset(args.dataset)
        output_dir = Path(args.output) if args.output else resume_path.parent
        logger.info("Resuming XTX3-%s from %s", variant, resume_path)
    else:
        variant = prompt_variant(args.variant)
        dataset_path = prompt_dataset(args.dataset)
        output_dir = prompt_output(args.output, variant=variant)

    trainer = Trainer(
        variant=variant,
        dataset_path=dataset_path,
        config=config,
        device=args.device,
        output_dir=output_dir,
    )
    if resume_path is not None:
        trainer.resume(resume_path)

    if args.calibrate:
        merge_params = trainer.calibrate()
        logger.info("Calibration complete: %s", merge_params.to_dict())
        return

    summary = trainer.train()
    logger.info("Training complete. Best OKS AP=%.4f", summary["best_ap"])


if __name__ == "__main__":
    main()
