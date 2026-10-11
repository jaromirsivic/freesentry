"""Optimizers, parameter groups and the warm-up + cosine LR schedule.

Why this module exists separately from the trainer:

* **Parameter groups** decide which tensors receive weight decay. Decaying
  BatchNorm gains, biases or the RLE flow parameters (the criterion owns them)
  shrinks quantities that are not weights and hurts both accuracy and the
  calibration of the RLE sigma, so only ``>= 2-D`` *model* weights decay.
* **SGD + Nesterov** is the XTX4 default (implementation_plan_fable.md §8.1).
  **MuSGD** is kept as a reference-style ablation: on every ``>= 2-D`` weight the
  update mixes an orthogonalised (Muon) momentum direction and the plain
  Nesterov-SGD direction with ``muon_w`` / ``sgd_w``; 1-D tensors use SGD. The
  Newton-Schulz iteration runs batched over all same-shape matrices with
  ``torch.bmm`` so hundreds of small conv kernels cost a handful of kernels.
* **AdamW** is available for quick experiments only.

The LR schedule is stepped **per iteration** by the trainer (``lr_at(epoch_float)``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

OPTIMIZERS: tuple[str, ...] = ("sgd", "musgd", "adamw")


# --------------------------------------------------------------------------- groups


def split_parameter_groups(
    model: nn.Module,
    criterion: nn.Module | None = None,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Three groups: model weights (decay, ``muon=True``), model 1-D (no decay), criterion (no decay).

    Empty groups are dropped so optimizers never see a group without parameters.
    """

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.ndim >= 2 else no_decay).append(param)
    crit: list[nn.Parameter] = []
    if criterion is not None:
        for param in criterion.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            crit.append(param)
    groups = [
        {"params": decay, "weight_decay": float(weight_decay), "muon": True, "name": "model_weights"},
        {"params": no_decay, "weight_decay": 0.0, "muon": False, "name": "model_1d"},
        {"params": crit, "weight_decay": 0.0, "muon": False, "name": "criterion"},
    ]
    return [g for g in groups if g["params"]]


# --------------------------------------------------------------------------- MuSGD


@torch.no_grad()
def newton_schulz_batched(mats: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalise a stack of matrices ``(G, m, n)`` with the quintic Newton-Schulz iteration.

    Every matrix is normalised by its own Frobenius norm first; the iteration
    uses the reference Muon coefficients and ``torch.bmm`` on the (transposed
    when ``m > n``) stack. Returns ``(G, m, n)`` in the input dtype.
    """

    if mats.ndim != 3:
        raise ValueError(f"expected (G, m, n), got {tuple(mats.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = mats.to(torch.float32)
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    norm = x.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1)
    x = x / norm
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        poly = b * gram + c * torch.bmm(gram, gram)
        x = a * x + torch.bmm(poly, x)
    if transposed:
        x = x.transpose(1, 2)
    return x.to(mats.dtype)


class MuSGD(Optimizer):
    """Reference-style MuSGD: Muon + SGD mixed on ``>= 2-D`` weights, SGD elsewhere.

    For a weight ``W`` with Nesterov momentum direction ``d``:
    ``W -= lr * (muon_w * scale * NS(d) + sgd_w * d)`` where ``NS`` is the
    batched Newton-Schulz orthogonalisation of ``d`` viewed as ``(O, I*kH*kW)``
    and ``scale = sqrt(max(1, rows / cols))`` keeps the RMS comparable to SGD.
    Groups created with ``muon=False`` (biases, BN, criterion) get plain SGD.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        muon_w: float = 0.5,
        sgd_w: float = 0.5,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "muon": False,
            "muon_w": float(muon_w),
            "sgd_w": float(sgd_w),
            "ns_steps": int(ns_steps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            use_muon = bool(group.get("muon", False))
            muon_w = float(group.get("muon_w", 0.5))
            sgd_w = float(group.get("sgd_w", 0.5))
            ns_steps = int(group.get("ns_steps", 5))

            # Nesterov-SGD direction for every parameter (also feeds the Muon branch).
            directions: list[tuple[nn.Parameter, torch.Tensor]] = []
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                direction = grad.add(buf, alpha=momentum) if nesterov else buf
                directions.append((param, direction))

            if not use_muon or muon_w == 0.0:
                for param, direction in directions:
                    param.add_(direction, alpha=-lr)
                continue

            # Muon branch: batch all same-shape matrices into one Newton-Schulz run.
            by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
            for i, (param, _direction) in enumerate(directions):
                if param.ndim >= 2:
                    by_shape[(param.shape[0], param[0].numel())].append(i)
                else:
                    param.add_(directions[i][1], alpha=-lr)
            for (rows, cols), indices in by_shape.items():
                stack = torch.stack([directions[i][1].reshape(rows, cols) for i in indices], dim=0)
                ortho = newton_schulz_batched(stack, steps=ns_steps)
                scale = math.sqrt(max(1.0, rows / cols))
                for j, i in enumerate(indices):
                    param, direction = directions[i]
                    update = ortho[j].reshape_as(param).mul(muon_w * scale)
                    if sgd_w != 0.0:
                        update.add_(direction, alpha=sgd_w)
                    param.add_(update, alpha=-lr)
        return loss


# --------------------------------------------------------------------------- builder


def build_optimizer(
    model: nn.Module,
    criterion: nn.Module | None = None,
    *,
    name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    nesterov: bool = True,
    muon_w: float = 0.5,
    sgd_w: float = 0.5,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """Build ``"sgd"`` (default), ``"musgd"`` or ``"adamw"`` over model + criterion groups."""

    groups = split_parameter_groups(model, criterion, weight_decay=weight_decay)
    key = name.strip().lower()
    if key == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=bool(nesterov and momentum > 0))
    if key == "musgd":
        return MuSGD(
            groups, lr=lr, momentum=momentum, nesterov=nesterov, muon_w=muon_w, sgd_w=sgd_w, ns_steps=ns_steps
        )
    if key == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


def build_optimizer_from_config(model: nn.Module, criterion: nn.Module | None, train_cfg: dict[str, Any]) -> Optimizer:
    """Convenience: read ``training.*`` keys (``optimizer``, ``lr0``, ``momentum``, ...)."""

    return build_optimizer(
        model,
        criterion,
        name=str(train_cfg.get("optimizer", "sgd")),
        lr=float(train_cfg.get("lr0", 0.01)),
        momentum=float(train_cfg.get("momentum", 0.937)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
        nesterov=bool(train_cfg.get("nesterov", True)),
        muon_w=float(train_cfg.get("muon_w", 0.5)),
        sgd_w=float(train_cfg.get("sgd_w", 0.5)),
        ns_steps=int(train_cfg.get("ns_steps", 5)),
    )


# --------------------------------------------------------------------------- schedule


class WarmupCosineSchedule:
    """Linear warm-up to ``lr0`` over ``warmup_epochs`` then cosine to ``lr0 * lr_final_factor``.

    ``lr_at`` takes a fractional epoch so the trainer can step it every iteration;
    ``apply`` writes the value into every optimizer group (groups may carry an
    ``lr_scale`` key for per-group multipliers).
    """

    def __init__(
        self,
        *,
        lr0: float,
        lr_final_factor: float,
        warmup_epochs: float,
        epochs: int,
        warmup_start_factor: float = 0.0,
    ) -> None:
        self.lr0 = float(lr0)
        self.lr_final_factor = float(lr_final_factor)
        self.warmup_epochs = float(warmup_epochs)
        self.epochs = max(int(epochs), 1)
        self.warmup_start_factor = float(warmup_start_factor)

    def lr_at(self, epoch_float: float) -> float:
        if self.warmup_epochs > 0 and epoch_float < self.warmup_epochs:
            t = epoch_float / self.warmup_epochs
            return self.lr0 * (self.warmup_start_factor + (1.0 - self.warmup_start_factor) * t)
        progress = (epoch_float - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1e-9)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return self.lr0 * (self.lr_final_factor + (1.0 - self.lr_final_factor) * cosine)

    def apply(self, optimizer: Optimizer, epoch_float: float) -> float:
        lr = self.lr_at(epoch_float)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        return lr

    @classmethod
    def from_config(cls, train_cfg: dict[str, Any], *, epochs: int | None = None) -> "WarmupCosineSchedule":
        return cls(
            lr0=float(train_cfg.get("lr0", 0.01)),
            lr_final_factor=float(train_cfg.get("lr_final_factor", 0.01)),
            warmup_epochs=float(train_cfg.get("warmup_epochs", 3)),
            epochs=int(epochs if epochs is not None else train_cfg.get("epochs", 100)),
        )
