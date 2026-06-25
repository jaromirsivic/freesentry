"""Checkpoint management + EMA (spec section 9.3).

Checkpoints store model/EMA/optimizer/scaler state, epoch/step, a frozen config
snapshot, merge params (filled after calibration), metrics, and keypoint metadata.
EMA weights are used for validation and export.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.keypoints import KEYPOINT_NAMES, OKS_SIGMAS
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelEMA:
    """Exponential moving average of model parameters (used for val/export)."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, warmup_steps: int = 2000) -> None:
        self.ema = deepcopy(self._unwrap(model)).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = max(1, warmup_steps)
        self.updates = 0

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        msd = self._unwrap(model).state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(msd[key].detach(), alpha=1 - decay)
            else:
                value.copy_(msd[key])

    def state_dict(self) -> dict[str, Any]:
        return self.ema.state_dict()


def build_checkpoint(
    *,
    variant: str,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    merge_params: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    criterion_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the checkpoint dict (section 9.3 layout)."""

    model_inner = model.module if hasattr(model, "module") else model
    return {
        "variant": variant,
        "model_state": model_inner.state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "criterion_state": criterion_state,
        "epoch": epoch,
        "global_step": global_step,
        "config": deepcopy(config),
        "merge_params": merge_params or {},
        "metrics": metrics or {},
        "keypoint_meta": {
            "names": list(KEYPOINT_NAMES),
            "oks_sigmas": OKS_SIGMAS.tolist(),
        },
    }


def save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    """Save a checkpoint atomically (write to a temp file then rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    logger.info("Saved checkpoint -> %s", path)


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict, failing fast with a clear error if missing."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
