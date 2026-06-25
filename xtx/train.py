"""XTX Pose - interactive console training entry point (spec section 9).

Usage:
    python train.py                              # interactive prompts
    python train.py --variant n --dataset ./dataset
    python train.py --resume runs/xtx-n/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xtx.engine.trainer import Trainer, load_config, prompt_dataset, prompt_variant
from xtx.utils.logging import configure_logging, get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XTX pose model (n/m/l).")
    parser.add_argument("--variant", choices=["n", "m", "l"], default=None)
    parser.add_argument("--dataset", default=None, help="Dataset root containing train/ and test/")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--output", default=None, help="Output run directory")
    parser.add_argument("--calibrate", action="store_true", help="Only run merge calibration")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(Path(args.config) if args.config else None)

    variant = prompt_variant(args.variant)
    dataset_path = prompt_dataset(args.dataset)

    trainer = Trainer(
        variant=variant,
        dataset_path=dataset_path,
        config=config,
        device=args.device,
        output_dir=Path(args.output) if args.output else None,
    )
    if args.resume:
        trainer.resume(Path(args.resume))

    if args.calibrate:
        merge_params = trainer.calibrate()
        logger.info("Calibration complete: %s", merge_params.to_dict())
        return

    summary = trainer.train()
    logger.info("Training complete. Best OKS AP=%.4f", summary["best_ap"])


if __name__ == "__main__":
    main()
