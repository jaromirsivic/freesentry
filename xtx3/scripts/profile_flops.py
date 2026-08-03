"""Profile XTX3 params and FLOPs, verify the per-variant budgets, and compare
against the XTX2 models when that project is available (../xtx2).

Budgets (binding constraints for the scaling table):

* u: <= 1.6M params, <= 0.75 GFLOPs single pass at 320 (Raspberry Pi 5
  real-time target: >= 25 FPS whole-frame).
* n/m/l: params and per-pass GFLOPs at 384 must be at or below the XTX2
  variant of the same tier (the nine-point head guarantees the head is
  cheaper; the backbone/neck are identical in scale).

Usage:
    python scripts/profile_flops.py [--variant u|n|m|l|all] [--size N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from xtx3.models import build_model  # noqa: E402
from xtx3.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("profile")

# variant -> (params_low_M, params_high_M, gflops_cap_per_pass, reference_size)
TARGETS = {
    "u": (0.6, 1.6, 0.75, 320),
    "n": (2.0, 3.0, 2.5, 384),
    "m": (16.0, 22.0, 24.0, 384),
    "l": (22.0, 28.0, 30.0, 384),
}

# Sizes reported for the multi-resolution u variant.
U_SIZES = (256, 320, 384)


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


def profile_xtx2(variant: str, *, input_size: int) -> tuple[float, float] | None:
    """Params/GFLOPs of the XTX2 model of the same tier, or None when unavailable."""

    xtx2_root = Path(__file__).resolve().parents[2] / "xtx2"
    if not (xtx2_root / "xtx2").is_dir():
        return None
    sys.path.insert(0, str(xtx2_root))
    try:
        from xtx2.models import build_model as build_xtx2  # type: ignore[import-not-found]

        model = build_xtx2(variant=variant)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        gmacs = count_macs(model, input_size=input_size)
        return params_m, gmacs
    except Exception as exc:  # noqa: BLE001 - the baseline is best-effort reference
        logger.warning("Could not profile XTX2-%s: %s", variant, exc)
        return None
    finally:
        sys.path.remove(str(xtx2_root))
        # Purge the baseline project's modules so they cannot shadow ours.
        for name in [n for n in sys.modules if n == "xtx2" or n.startswith("xtx2.")]:
            del sys.modules[name]


def profile_variant(variant: str, *, size_override: int | None = None) -> bool:
    lo_p, hi_p, flop_cap, ref_size = TARGETS[variant]
    size = size_override or ref_size

    model = build_model(variant=variant)
    params_m = model.num_parameters() / 1e6
    gmacs = count_macs(model, input_size=size)
    params_ok = lo_p <= params_m <= hi_p
    flops_ok = gmacs <= flop_cap or size != ref_size
    logger.info(
        "XTX3-%s @%d | params=%.2fM (target %.1f-%.1f %s) | %.3f GFLOPs/pass (cap %.2f @%d -> %s)",
        variant, size, params_m, lo_p, hi_p, "OK" if params_ok else "OUT",
        gmacs, flop_cap, ref_size, "OK" if flops_ok else "OVER",
    )

    if variant == "u":
        for s in U_SIZES:
            if s != size:
                logger.info("  XTX3-u @%d: %.3f GFLOPs/pass", s, count_macs(model, input_size=s))

    ok = params_ok and flops_ok
    if variant != "u":
        baseline = profile_xtx2(variant, input_size=384)
        if baseline is not None:
            b_params, b_gmacs = baseline
            not_slower = gmacs <= b_gmacs + 1e-6
            not_bigger = params_m <= b_params + 1e-6
            logger.info(
                "  vs XTX2-%s: params %.2fM -> %.2fM (%s) | GFLOPs/pass %.3f -> %.3f (%s)",
                variant, b_params, params_m, "OK" if not_bigger else "BIGGER",
                b_gmacs, gmacs, "FASTER/EQUAL" if not_slower else "SLOWER",
            )
            ok = ok and not_slower and not_bigger
    return ok


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=["u", "n", "m", "l", "all"])
    parser.add_argument("--size", type=int, default=None, help="Override the profiled input size")
    args = parser.parse_args()
    variants = ["u", "n", "m", "l"] if args.variant == "all" else [args.variant]
    all_ok = all([profile_variant(v, size_override=args.size) for v in variants])
    if not all_ok:
        logger.warning("One or more variants are outside their budgets")
        sys.exit(1)


if __name__ == "__main__":
    main()
