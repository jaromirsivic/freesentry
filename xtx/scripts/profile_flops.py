"""Profile XTX params and FLOPs per crop, and compare to the section 6.6 budgets.

Usage:
    python scripts/profile_flops.py [--variant n|m|l|all] [--size 384]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from xtx.models import build_model
from xtx.utils.logging import configure_logging, get_logger

logger = get_logger("profile")

# (params_low_M, params_high_M, gflops_low_per_crop, gflops_high_per_crop) at 384.
TARGETS = {
    "n": (2.5, 3.0, 2.0, 2.8),
    "m": (18.0, 22.0, 22.0, 28.0),
    "l": (24.0, 28.0, 30.0, 38.0),
}

# YOLO26-pose single 640-pass GFLOPs: the XTX 3-crop total must stay at-or-below this
# (spec section 6.6 headline: equal-or-lower compute than YOLO26 of the same tier).
YOLO26_SINGLE_PASS_GFLOPS = {"n": 7.5, "m": 73.1, "l": 91.3}


def count_macs(model: nn.Module, *, input_size: int) -> float:
    """Count multiply-accumulates (GMACs) for a single forward via hooks."""

    total = 0.0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal total
        out_h, out_w = output.shape[-2:]
        kh, kw = module.kernel_size
        total += (
            out_h * out_w * module.out_channels * (module.in_channels // module.groups) * kh * kw
        )

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total
        total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size))
    for handle in handles:
        handle.remove()
    return total / 1e9


def profile_variant(variant: str, *, input_size: int) -> None:
    model = build_model(variant=variant)
    params_m = model.num_parameters() / 1e6
    gmacs = count_macs(model, input_size=input_size)
    lo_p, hi_p, _lo_f, hi_f = TARGETS[variant]
    three_crop = gmacs * 3
    yolo_budget = YOLO26_SINGLE_PASS_GFLOPS[variant]
    params_ok = lo_p <= params_m <= hi_p
    # The compute constraint is an upper budget (equal-or-lower), so <= high is good.
    per_crop_status = "OK" if gmacs <= hi_f else "over"
    budget_status = "OK" if three_crop <= yolo_budget else "OVER"
    logger.info(
        "XTX-%s | params=%.2fM (target %.1f-%.1f %s) | %.2f GFLOPs/crop (budget <=%.1f %s) | "
        "3-crop=%.2f GFLOPs vs YOLO26 single-pass %.1f -> %s",
        variant, params_m, lo_p, hi_p, "OK" if params_ok else "out",
        gmacs, hi_f, per_crop_status,
        three_crop, yolo_budget, budget_status,
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=["n", "m", "l", "all"])
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    variants = ["n", "m", "l"] if args.variant == "all" else [args.variant]
    for variant in variants:
        profile_variant(variant, input_size=args.size)


if __name__ == "__main__":
    main()
