"""Stand-alone loss-weight calibration (CONTRACTS §7, ``xtx4/losses/calibrate.py``).

Builds the model + criterion + train loader exactly like ``train.py`` and
measures, for every loss term, the gradient norm it induces on the head
parameters over ``--steps`` batches. Writes ``loss_calibration.json`` into the
run directory and prints the proposed ``loss.*`` config; nothing is trained and
the config is not modified.

    python scripts/calibrate_loss.py --config configs/u.json --dataset D:/xtxtraining --steps 50
    python scripts/calibrate_loss.py --weights D:/xtx4_runs/xtx4-u/last.pt --steps 20   # on trained weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xtx4.engine.trainer import Trainer, apply_overrides, load_config  # noqa: E402
from xtx4.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("calibrate_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate XTX4 loss weights by head gradient norms.")
    parser.add_argument("--config", default=None, help="User config JSON (deep-merged over configs/u.json)")
    parser.add_argument("--dataset", default=None, help="Dataset root containing train/ and test/")
    parser.add_argument("--output", default=None, help="Directory for loss_calibration.json (default: run dir)")
    parser.add_argument("--weights", default=None, help="Optional checkpoint whose weights are calibrated")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.set)
    config.setdefault("training", {})["data_manifest"] = False  # hashing the split is not needed here
    config["training"].setdefault("tensorboard", {})["enabled"] = False
    trainer = Trainer(config, args.dataset, args.output, args.device)
    if args.weights:
        trainer.load_init_weights(args.weights)
    path = trainer.calibrate_loss(args.steps)
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    logger.info("Current effective contributions: %s", {k: round(v, 5) for k, v in result["effective"].items()})
    logger.info("Proposed loss config: %s", json.dumps(result["proposed_config"], indent=2))
    logger.info("Report -> %s", path)


if __name__ == "__main__":
    main()
