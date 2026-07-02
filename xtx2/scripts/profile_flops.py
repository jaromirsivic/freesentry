"""Profile XTX2 params and FLOPs per level, verify the section 6.6 budgets, and
compare against the old ./xtx models when that project is available.

Usage:
    python scripts/profile_flops.py [--variant n|m|l|all] [--size 384]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from xtx2.models import build_model  # noqa: E402
from xtx2.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("profile")

# (params_low_M, params_high_M, gflops_low_per_level, gflops_high_per_level) at 384.
TARGETS = {
    "n": (2.5, 3.0, 1.8, 2.5),
    "m": (18.0, 22.0, 20.0, 24.0),
    "l": (24.0, 28.0, 26.0, 30.0),
}

# YOLO26-pose single 640-pass GFLOPs: the XTX2 3-level total must stay at-or-below
# this (section 6.6: equal-or-lower compute than YOLO26 of the same tier).
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


def profile_old_xtx(variant: str, *, input_size: int) -> tuple[float, float] | None:
    """Params/GFLOPs of the old ./xtx model, or None when unavailable."""

    xtx_root = Path(__file__).resolve().parents[2] / "xtx"
    if not (xtx_root / "xtx").is_dir():
        return None
    sys.path.insert(0, str(xtx_root))
    try:
        from xtx.models import build_model as build_xtx  # type: ignore[import-not-found]

        model = build_xtx(variant=variant)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gmacs = count_macs(model, input_size=input_size)
        return params_m, gmacs
    except Exception as exc:  # noqa: BLE001 - old project is best-effort reference
        logger.warning("Could not profile old XTX-%s: %s", variant, exc)
        return None
    finally:
        sys.path.remove(str(xtx_root))
        # Purge the old project's modules so they cannot shadow ours.
        for name in [n for n in sys.modules if n == "xtx" or n.startswith("xtx.")]:
            del sys.modules[name]


def profile_variant(variant: str, *, input_size: int) -> None:
    model = build_model(variant=variant)
    params_m = model.num_parameters() / 1e6
    gmacs = count_macs(model, input_size=input_size)
    lo_p, hi_p, _lo_f, hi_f = TARGETS[variant]
    three_level = gmacs * 3
    yolo_budget = YOLO26_SINGLE_PASS_GFLOPS[variant]
    params_ok = lo_p <= params_m <= hi_p
    per_level_status = "OK" if gmacs <= hi_f else "over"
    budget_status = "OK" if three_level <= yolo_budget else "OVER"
    logger.info(
        "XTX2-%s | params=%.2fM (target %.1f-%.1f %s) | %.2f GFLOPs/level (budget <=%.1f %s) | "
        "3-level=%.2f GFLOPs vs YOLO26 single-pass %.1f -> %s",
        variant, params_m, lo_p, hi_p, "OK" if params_ok else "out",
        gmacs, hi_f, per_level_status,
        three_level, yolo_budget, budget_status,
    )

    old = profile_old_xtx(variant, input_size=input_size)
    if old is not None:
        old_params, old_gmacs = old
        faster = gmacs < old_gmacs
        logger.info(
            "  vs old XTX-%s: params %.2fM -> %.2fM | GFLOPs/crop %.2f -> %.2f (%s)",
            variant, old_params, params_m, old_gmacs, gmacs,
            "FASTER" if faster else "NOT faster",
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
