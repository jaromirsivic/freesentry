"""Gradient-norm calibration of the loss term weights (CONTRACTS.md section 7,
implementation_plan_fable.md section 6.3).

Each loss term pushes the head parameters with a different gradient magnitude;
hand-tuned lambdas from another detector (YOLO26, ER-Pose) do not transfer to a
new head layout. This module measures, for each term ``T`` in
``person, oks, rle, present, visible``, the norm ``g_T`` of ``d term_T / d theta_head``
with unit weight over a few batches, then proposes lambdas so that the effective
contributions ``lambda_T * g_T`` lie within a factor ``max_ratio`` (3 by default)
of each other. It only reports; the config is changed by a human.

The function is generic: it needs a module whose forward returns ``HeadOutput``
and a criterion exposing ``compute_terms`` and the ``*_weight`` attributes of
:class:`xtx4.losses.pose_loss.XTX4Loss`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.logging import get_logger

log = get_logger(__name__)

CALIBRATED_TERMS: tuple[str, ...] = ("person", "oks", "rle", "present", "visible")


def _head_parameters(model: nn.Module, param_filter: Callable[[str], bool] | None) -> list[torch.Tensor]:
    if param_filter is None:
        def param_filter(name: str) -> bool:  # noqa: E306 - local default
            return "head" in name.lower()

    params = [p for n, p in model.named_parameters() if p.requires_grad and param_filter(n)]
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    return params


def _grad_norm(term: torch.Tensor, params: list[torch.Tensor]) -> float:
    if not term.requires_grad:
        return 0.0
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum())
    return math.sqrt(total)


def current_term_lambdas(criterion: nn.Module) -> dict[str, float]:
    """Effective lambda per term implied by the criterion's current weights."""

    return {
        "person": float(criterion.person_weight),
        "oks": float(criterion.kpt_weight) * float(criterion.oks_weight),
        "rle": float(criterion.kpt_weight) * float(criterion.rle_weight),
        "present": float(criterion.present_weight),
        "visible": float(criterion.visible_weight),
    }


def propose_lambdas(
    grad_norms: dict[str, float],
    current: dict[str, float],
    *,
    max_ratio: float = 3.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Clip the effective contributions ``lambda_T * g_T`` into ``[ref / sqrt(r), ref * sqrt(r)]``.

    ``ref`` is the median effective contribution of the measurable terms, so terms
    already in range keep their lambda and only the outliers move.
    """

    eff = {k: current[k] * grad_norms[k] for k in grad_norms if grad_norms[k] > eps}
    if not eff:
        return dict(current)
    values = sorted(eff.values())
    mid = len(values) // 2
    ref = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    half = math.sqrt(max_ratio)
    lo, hi = ref / half, ref * half
    proposed = dict(current)
    for key, e in eff.items():
        clipped = min(max(e, lo), hi)
        proposed[key] = clipped / grad_norms[key]
    return proposed


def lambdas_to_config(lambdas: dict[str, float]) -> dict[str, float]:
    """Map per-term lambdas back to ``config["loss"]`` keys (``oks_weight`` fixed at 1)."""

    kpt = max(lambdas["oks"], 1e-12)
    return {
        "person": lambdas["person"],
        "kpt": kpt,
        "oks_weight": 1.0,
        "rle_weight": lambdas["rle"] / kpt,
        "present": lambdas["present"],
        "visible": lambdas["visible"],
    }


def calibrate_loss_weights(
    model: nn.Module,
    criterion: nn.Module,
    batches: Iterable[Any],
    steps: int = 50,
    *,
    max_ratio: float = 3.0,
    param_filter: Callable[[str], bool] | None = None,
    image_size: int | None = None,
) -> dict[str, Any]:
    """Measure per-term head gradient norms over ``steps`` batches and propose lambdas.

    ``batches`` yields ``(images, targets, ...)`` as the collate function does. The
    model is put in training mode for the measurement (both heads) and restored.
    Returns a JSON-serialisable dict with ``grad_norms`` (mean per term),
    ``grad_norms_per_step``, ``current``, ``effective``, ``proposed`` (lambdas) and
    ``proposed_config`` (``loss.*`` keys).
    """

    params = _head_parameters(model, param_filter)
    device = params[0].device if params else torch.device("cpu")
    was_training = model.training
    model.train()
    per_step: dict[str, list[float]] = {k: [] for k in CALIBRATED_TERMS}
    seen = 0
    try:
        for batch in batches:
            if seen >= steps:
                break
            images, targets = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            size = int(image_size) if image_size is not None else int(images.shape[-1])
            output = model(images)
            terms = criterion.compute_terms(output, targets, size)
            for key in CALIBRATED_TERMS:
                per_step[key].append(_grad_norm(terms[key], params))
            seen += 1
            model.zero_grad(set_to_none=True)
    finally:
        model.train(was_training)
    if seen == 0:
        raise ValueError("calibrate_loss_weights received no batches")

    grad_norms = {k: float(sum(v) / len(v)) for k, v in per_step.items()}
    current = current_term_lambdas(criterion)
    effective = {k: current[k] * grad_norms[k] for k in CALIBRATED_TERMS}
    proposed = propose_lambdas(grad_norms, current, max_ratio=max_ratio)
    result = {
        "steps": seen,
        "max_ratio": max_ratio,
        "grad_norms": grad_norms,
        "grad_norms_per_step": per_step,
        "current": current,
        "effective": effective,
        "proposed": proposed,
        "proposed_effective": {k: proposed[k] * grad_norms[k] for k in CALIBRATED_TERMS},
        "proposed_config": lambdas_to_config(proposed),
    }
    log.info(
        "loss calibration over %d steps: grad norms %s -> proposed loss config %s",
        seen,
        {k: round(v, 5) for k, v in grad_norms.items()},
        {k: round(v, 5) for k, v in result["proposed_config"].items()},
    )
    return result


def write_calibration_json(result: dict[str, Any], path: str | Path) -> Path:
    """Write the calibration report as JSON and return the path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out
