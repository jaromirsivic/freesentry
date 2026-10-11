"""XTX4 Pose - training entry point (non-interactive CLI, interactive prompts without flags).

Usage:
    python train.py                                   # interactive prompts (variant, dataset, output)
    python train.py --config configs/u.json --dataset D:/xtxtraining
    python train.py --variant u_b --epochs 100 --set model.levels=P3P5 --set training.lr0=0.02
    python train.py --resume D:/xtx4_runs/xtx4-u/last.pt
    python train.py --init-weights D:/xtx4_runs/xtx4-u/best.pt --output D:/xtx4_runs/finetune
    python train.py --calibrate-loss 50 ...           # writes loss_calibration.json first

Runs are written to ``config.run.output_root / config.run.name`` (``D:/xtx4_runs`` by
default - drive C: is nearly full) unless ``--output`` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Multi-scale training constantly changes activation shapes, which fragments the
# CUDA caching allocator on Windows/WDDM. Expandable segments avoid that; must be
# set before the first CUDA allocation (i.e. before importing torch-heavy code).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xtx4.engine.checkpoint import discover_checkpoints, load_checkpoint  # noqa: E402
from xtx4.engine.trainer import Trainer, apply_overrides, load_config, resolve_output_dir  # noqa: E402
from xtx4.models.scaling import PRESETS  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XTX4 box-free pose model.")
    parser.add_argument("--config", default=None, help="User config JSON (deep-merged over configs/u.json)")
    parser.add_argument("--dataset", default=None, help="Dataset root containing train/ and test/")
    parser.add_argument("--variant", default=None, help=f"Model preset: {', '.join(PRESETS)}")
    parser.add_argument("--output", default=None, help="Run directory (default: run.output_root/run.name)")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from (full state)")
    parser.add_argument("--init-weights", default=None, help="Checkpoint whose weights initialise a new run")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--calibrate-loss", type=int, default=None, metavar="N",
                        help="Run the loss-weight calibration on N batches before training")
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                        help="Config override, repeatable (e.g. --set model.levels=P3P5)")
    parser.add_argument("--no-resume", action="store_true", help="Skip the interactive resume prompt")
    parser.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)  # throughput checks
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- prompts


def prompt_resume(*, runs_dir: Path) -> Path | None:
    checkpoints = discover_checkpoints(runs_dir)
    if not checkpoints:
        return None
    answer = input("Resume training from a checkpoint? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return None
    print("Discovered checkpoints:")
    for i, path in enumerate(checkpoints[:20]):
        print(f"  [{i}] {path}")
    while True:
        raw = input("Pick a number or type a checkpoint path: ").strip()
        if raw.isdigit() and int(raw) < len(checkpoints):
            return checkpoints[int(raw)]
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
        print("Not a valid selection; try again.")


def prompt_variant(default: str) -> str:
    while True:
        answer = input(f"Which variant do you want to train? [{default}] ({', '.join(PRESETS)}): ").strip()
        if not answer:
            return default
        if answer in PRESETS:
            return answer
        print(f"Please enter one of: {', '.join(PRESETS)}")


def prompt_dataset(default: str) -> Path:
    while True:
        raw = input(f"Dataset path [{default}]: ").strip()
        path = Path(raw) if raw else Path(default)
        if (path / "train").is_dir() and (path / "test").is_dir():
            return path
        print(f"Expected '{path}/train' and '{path}/test' to exist; try again.")


def prompt_output(default: Path) -> Path:
    raw = input(f"Run directory [{default}]: ").strip()
    return Path(raw) if raw else default


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    interactive = len(sys.argv) == 1 and argv is None
    config = load_config(args.config)

    resume_path: Path | None = Path(args.resume) if args.resume else None
    if interactive and not args.no_resume and resume_path is None:
        resume_path = prompt_resume(runs_dir=Path(str(config.get("run", {}).get("output_root", "D:/xtx4_runs"))))

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        if ckpt.get("config"):
            config = ckpt["config"]  # the frozen run config; CLI overrides below still apply
        del ckpt
        output_dir = Path(args.output) if args.output else resume_path.parent
    else:
        output_dir = Path(args.output) if args.output else None

    if args.variant:
        config.setdefault("model", {})["variant"] = args.variant
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    apply_overrides(config, args.set)

    dataset_default = str(args.dataset or config.get("data", {}).get("dataset_path") or "D:/xtxtraining")
    if interactive and resume_path is None:
        config.setdefault("model", {})["variant"] = prompt_variant(str(config.get("model", {}).get("variant", "u_b")))
        dataset_path = prompt_dataset(dataset_default)
        output_dir = prompt_output(resolve_output_dir(config, None))
    else:
        dataset_path = Path(dataset_default)
    if not ((dataset_path / "train").is_dir() and (dataset_path / "test").is_dir()):
        raise SystemExit(f"Dataset root must contain train/ and test/: {dataset_path}")

    trainer = Trainer(config, dataset_path, output_dir, args.device, max_steps=args.max_steps)
    if args.init_weights:
        trainer.load_init_weights(args.init_weights)
    if resume_path is not None:
        trainer.resume(resume_path)
    if args.calibrate_loss:
        trainer.calibrate_loss(args.calibrate_loss)

    summary = trainer.train()
    logger.info("Training complete. Best AP=%.4f after %d steps", summary["best_ap"], summary["global_step"])


if __name__ == "__main__":
    main()
