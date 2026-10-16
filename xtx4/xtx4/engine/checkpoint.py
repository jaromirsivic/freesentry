"""Checkpoint management + EMA (CONTRACTS.md section 11).

A checkpoint is a plain dict with ``variant_spec`` (resolved architecture, so a
checkpoint rebuilds without the preset table), ``model_state``, ``ema_state``,
``optimizer_state``, ``scaler_state``, ``criterion_state`` (RLE flow), ``epoch``,
``global_step``, ``ema_updates``, ``rng_state`` (python/numpy/torch/cuda), a
frozen ``config``, ``metrics``, ``keypoint_meta``, ``data_manifest_sha1`` and
``parent_checkpoint_sha256`` (set when fine-tuning from another checkpoint).

The EMA keeps a deep copy of the model and blends floating-point tensors with
``torch._foreach_mul_/_foreach_add_`` (one kernel launch per op instead of one
per tensor); ``updates`` drives the XTX3 warm-up ``decay * (1 - exp(-updates /
warmup))`` so the first steps track the model closely. The trainer only calls
``update`` after an optimizer step that was actually applied (GradScaler skips
are excluded), which is why ``updates`` is persisted separately.

Saves are atomic (temp file + rename) so an interrupted write never leaves a
truncated ``last.pt`` behind.

Rolling per-epoch checkpoints (crash resilience): after every completed epoch
the trainer also keeps the full checkpoint under ``<run>/epochs/epoch_NNNN.pt``.
**Numbering convention**: ``NNNN`` is the 1-based number of the *completed*
epoch exactly as the training log prints it (``Epoch 7/300 done`` ->
``epoch_0007.pt``), i.e. ``NNNN == checkpoint["epoch"] + 1`` because the dict
stores the 0-based epoch index. The content is byte-identical to ``last.pt``
(``last.pt`` is produced by copying the epoch file), so ``--resume`` accepts any
of them and continues with the next epoch. :func:`prune_epoch_checkpoints`
keeps only the newest ``training.keep_epochs`` files **by modification time**,
see its docstring for why not by epoch number.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..utils.keypoints import COCO_KEYPOINT_INDICES, FLIP_PAIRS, KEYPOINT_NAMES, OKS_SIGMAS
from ..utils.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_SCHEMA_VERSION = 2
EPOCH_DIR = "epochs"
EPOCH_FILE_RE = re.compile(r"^epoch_(\d{4,})\.pt$")


class ModelEMA:
    """Exponential moving average of model parameters and buffers (validation / export weights)."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9998, warmup_steps: int = 2000) -> None:
        self.ema = deepcopy(self._unwrap(model)).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.decay = float(decay)
        self.warmup_steps = max(1, int(warmup_steps))
        self.updates = 0
        self._ema_float: list[torch.Tensor] = []
        self._model_float: list[torch.Tensor] = []
        self._ema_other: list[torch.Tensor] = []
        self._model_other: list[torch.Tensor] = []
        self._bind(model)

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _bind(self, model: nn.Module) -> None:
        """Cache aligned tensor lists (state_dict order) for the foreach update."""

        msd = self._unwrap(model).state_dict()
        self._ema_float, self._model_float, self._ema_other, self._model_other = [], [], [], []
        for key, value in self.ema.state_dict().items():
            source = msd[key]
            if value.dtype.is_floating_point:
                self._ema_float.append(value)
                self._model_float.append(source)
            else:
                self._ema_other.append(value)
                self._model_other.append(source)

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup_steps))

    def rebind(self, model: nn.Module) -> None:
        """Re-cache tensor views after the model's storage changed (``.to(memory_format)``, load_state_dict)."""

        self._bind(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model into the EMA; call only after an applied optimizer step."""

        self.updates += 1
        decay = self.current_decay()
        torch._foreach_mul_(self._ema_float, decay)
        torch._foreach_add_(self._ema_float, self._model_float, alpha=1.0 - decay)
        for dst, src in zip(self._ema_other, self._model_other):
            dst.copy_(src)

    def state_dict(self) -> dict[str, Any]:
        return self.ema.state_dict()

    def load_state_dict(self, state: dict[str, Any], *, updates: int | None = None) -> None:
        self.ema.load_state_dict(state)
        if updates is not None:
            self.updates = int(updates)


# --------------------------------------------------------------------------- RNG


def capture_rng_state() -> dict[str, Any]:
    """Snapshot python / numpy / torch (CPU + all CUDA devices) RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
        cuda = state.get("cuda")
        if cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s for s in cuda][: torch.cuda.device_count()])
    except (RuntimeError, ValueError, TypeError) as exc:  # a different device layout is not fatal
        logger.warning("Could not fully restore RNG state: %s", exc)


# --------------------------------------------------------------------------- build / io


def keypoint_meta() -> dict[str, Any]:
    return {
        "names": list(KEYPOINT_NAMES),
        "coco_indices": list(COCO_KEYPOINT_INDICES),
        "flip_pairs": [list(p) for p in FLIP_PAIRS],
        "oks_sigmas": OKS_SIGMAS.tolist(),
    }


def build_checkpoint(
    *,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
    criterion_state: dict[str, Any] | None = None,
    data_manifest_sha1: str | None = None,
    parent_checkpoint_sha256: str | None = None,
    include_rng: bool = True,
) -> dict[str, Any]:
    """Assemble the section-11 checkpoint dict."""

    model_inner = model.module if hasattr(model, "module") else model
    spec = getattr(model_inner, "spec", None)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "variant_spec": spec.to_dict() if spec is not None else None,
        "model_state": model_inner.state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "ema_updates": int(ema.updates) if ema is not None else 0,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "criterion_state": criterion_state,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng_state": capture_rng_state() if include_rng else None,
        "config": deepcopy(config),
        "metrics": dict(metrics or {}),
        "keypoint_meta": keypoint_meta(),
        "data_manifest_sha1": data_manifest_sha1,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }


def save_checkpoint(checkpoint: dict[str, Any], path: Path | str) -> Path:
    """Save a checkpoint atomically (write to a temp file then rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    logger.info("Saved checkpoint -> %s", path)
    return path


def load_checkpoint(path: Path | str, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict, failing fast with a clear error if missing."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def sha256_of_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_checkpoint(src: Path | str, dst: Path | str) -> Path:
    """Atomically duplicate an already written checkpoint (copy to ``dst.tmp`` + replace).

    Used to derive ``last.pt`` / ``best.pt`` from the epoch file without a second
    ``build_checkpoint`` + ``torch.save`` (a file copy is a fraction of the cost of
    re-serialising the state dicts).
    """

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)
    logger.info("Saved checkpoint -> %s (copy of %s)", dst, src.name)
    return dst


def epoch_checkpoint_path(run_dir: Path | str, epoch_index: int) -> Path:
    """``<run>/epochs/epoch_NNNN.pt`` for the 0-based ``epoch_index`` (NNNN = index + 1, as logged)."""

    return Path(run_dir) / EPOCH_DIR / f"epoch_{int(epoch_index) + 1:04d}.pt"


def list_epoch_checkpoints(run_dir: Path | str) -> list[Path]:
    """Epoch files of one run, newest (by mtime) first."""

    epoch_dir = Path(run_dir) / EPOCH_DIR
    if not epoch_dir.is_dir():
        return []
    files = [p for p in epoch_dir.iterdir() if p.is_file() and EPOCH_FILE_RE.match(p.name)]
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def prune_epoch_checkpoints(run_dir: Path | str, keep: int, *, protect: Path | None = None) -> list[Path]:
    """Delete all but the ``keep`` newest epoch files; returns the removed paths.

    Pruning goes by **modification time**, not by the epoch number in the file
    name: after ``--resume epochs/epoch_0098.pt`` while stale ``epoch_0099..0105``
    still exist, the run re-creates ``epoch_0099.pt`` (fresh mtime) and the stale
    higher-numbered files are the *oldest* ones, so they are the first to go.
    Ordering by number would instead keep the stale files and delete the ones the
    resumed run is actually producing. Call this only after the new file is
    completely written (``protect`` is never removed, whatever its mtime).
    """

    if keep <= 0:
        return []
    files = list_epoch_checkpoints(run_dir)
    protect = Path(protect).resolve() if protect is not None else None
    removed: list[Path] = []
    for path in files[keep:]:
        if protect is not None and path.resolve() == protect:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:  # a locked file (antivirus, viewer) is not worth failing the run
            logger.warning("Could not remove old epoch checkpoint %s: %s", path, exc)
    if removed:
        logger.info("Pruned %d old epoch checkpoint(s): %s", len(removed), ", ".join(p.name for p in removed))
    return removed


def run_dir_of_checkpoint(path: Path | str) -> Path:
    """Run directory that owns a checkpoint (``<run>/last.pt`` or ``<run>/epochs/epoch_NNNN.pt``)."""

    path = Path(path)
    parent = path.parent
    if parent.name == EPOCH_DIR and EPOCH_FILE_RE.match(path.name):
        return parent.parent
    return parent


def discover_checkpoints(runs_dir: Path | str) -> list[Path]:
    """Find resumable checkpoints (``**/last.pt``, ``**/best.pt``, ``**/epochs/epoch_*.pt``), newest first."""

    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    found = set(runs_dir.glob("**/last.pt")) | set(runs_dir.glob("**/best.pt"))
    found |= {p for p in runs_dir.glob(f"**/{EPOCH_DIR}/epoch_*.pt") if EPOCH_FILE_RE.match(p.name)}
    return sorted(found, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
