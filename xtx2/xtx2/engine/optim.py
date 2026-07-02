"""MuSGD optimizer (spec section 7.5): a YOLO26-style hybrid of Muon and SGD.

**Muon branch** (weight matrices): momentum is accumulated as usual, then the
update direction is **orthogonalised** with a Newton-Schulz iteration (5 steps)
over the momentum matrix, following the Muon optimizer. Conv kernels
``(O, I, kH, kW)`` are treated as ``(O, I*kH*kW)`` matrices. The orthogonalised
update is rescaled by ``sqrt(max(1, rows/cols))`` (Muon convention) so the
update RMS is comparable to SGD.

**SGD branch** (biases, gains, BatchNorm parameters, and any 0/1-D tensor):
plain SGD with Nesterov momentum, identical to the fallback.

``build_optimizer(..., name="musgd"|"sgd")`` builds either optimizer with the
same parameter-group layout (decay / no-decay) so the trainer's LR schedule
(warmup + cosine) applies uniformly.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz_orthogonalize(matrix: torch.Tensor, *, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise ``matrix`` (2D) via Newton-Schulz iteration.

    Uses the quintic coefficients from the reference Muon implementation. The
    result approximates ``U @ V^T`` of the SVD of ``matrix``.
    """

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(matrix.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.to(torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class MuSGD(Optimizer):
    """Hybrid Muon + SGD optimizer.

    Parameters are routed per group: groups created with ``muon=True`` receive
    orthogonalised-momentum updates, others plain SGD with Nesterov momentum.
    All groups share ``lr`` so a single external LR schedule drives both.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "muon": False,
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
            ns_steps = int(group.get("ns_steps", 5))

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

                if use_muon and param.ndim >= 2:
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                    mat = update.reshape(update.shape[0], -1)
                    ortho = newton_schulz_orthogonalize(mat, steps=ns_steps)
                    # Scale so the update RMS matches the SGD branch (Muon convention).
                    scale = max(1.0, mat.shape[0] / mat.shape[1]) ** 0.5
                    param.add_(ortho.reshape_as(param), alpha=-lr * scale)
                else:
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                    param.add_(update, alpha=-lr)
        return loss


def split_parameter_groups(
    modules: Iterable[torch.nn.Module], *, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into (muon-decay, sgd-no-decay) groups.

    Weight matrices (ndim >= 2: conv/linear kernels) go to the Muon branch with
    weight decay; biases, gains, and BatchNorm parameters (ndim <= 1) go to the
    plain-SGD branch without decay.
    """

    muon_params: list[torch.nn.Parameter] = []
    sgd_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (muon_params if param.ndim >= 2 else sgd_params).append(param)
    return [
        {"params": muon_params, "muon": True, "weight_decay": weight_decay},
        {"params": sgd_params, "muon": False, "weight_decay": 0.0},
    ]


def build_optimizer(
    modules: Iterable[torch.nn.Module],
    *,
    name: str = "musgd",
    lr: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
) -> torch.optim.Optimizer:
    """Build the configured optimizer (``"musgd"`` default, ``"sgd"`` fallback)."""

    groups = split_parameter_groups(modules, weight_decay=weight_decay)
    key = name.strip().lower()
    if key == "musgd":
        return MuSGD(groups, lr=lr, momentum=momentum, nesterov=True)
    if key == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer {name!r}; expected 'musgd' or 'sgd'")
