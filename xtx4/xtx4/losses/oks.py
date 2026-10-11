"""Object Keypoint Similarity (OKS) in torch: pairwise matrices for the assigner
and per-point Smooth-OKS / OKS loss terms (CONTRACTS.md sections 6 and 7).

The denominator follows the COCO evaluator exactly, ``2 * s^2 * (2 * sigma_k)^2``
with ``s^2`` the instance area (or the extent fallback), so the training signal
and the metric agree. XTX3's ``OKSL1Loss`` used ``2 * sigma^2 * area`` instead
(a different, non-standard falloff).

Smooth-OKS (ER-Pose): with ``u^2 = d^2 / (2 s^2 (2 sigma_k)^2)`` the similarity
is ``exp(-u^2)`` inside the unit radius and ``exp(-(2u - 1))`` beyond it. Both
value and derivative are continuous at ``u = 1`` (``e^-1`` and ``-2 e^-1``),
so far-off points keep a non-vanishing gradient instead of the Gaussian tail.
"""

from __future__ import annotations

import torch

DEFAULT_OKS_CHUNK_BYTES = 512 * 1024 * 1024


def oks_denominator(scale2: torch.Tensor, sigmas: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """``2 * s^2 * (2 sigma_k)^2`` broadcast to ``scale2.shape + (K,)``."""

    k2 = (2.0 * sigmas) ** 2
    return (2.0 * scale2.unsqueeze(-1) * k2).clamp(min=eps)


def pairwise_oks(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    labeled: torch.Tensor,
    scale2: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    max_chunk_bytes: int = DEFAULT_OKS_CHUNK_BYTES,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Batched OKS between every anchor prediction and every GT (dense reference path).

    pred_xy: (B, N, K, 2)   decoded keypoints (px)
    gt_xy:   (B, M, K, 2)   GT keypoints (px)
    labeled: (B, M, K) bool labeled points (``v > 0`` and a real GT row)
    scale2:  (B, M)         OKS scale ``s^2`` (px^2)
    sigmas:  (K,)           per-landmark sigmas
    Returns (B, M, N) float32; GTs without labeled points get 0 everywhere.

    The ``(B, m, N, K)`` intermediate is chunked over ``M`` so it never exceeds
    ``max_chunk_bytes``. The assigner itself uses :func:`candidate_oks` (the same
    arithmetic on the candidate pairs only); this dense form is kept for tests and
    stand-alone analysis.
    """

    pred_xy = pred_xy.float()
    gt_xy = gt_xy.float()
    scale2 = scale2.float()
    sigmas = sigmas.to(device=pred_xy.device, dtype=torch.float32)
    batch, num_anchors, num_kpts, _ = pred_xy.shape
    num_gt = gt_xy.shape[1]
    out = pred_xy.new_zeros((batch, num_gt, num_anchors))
    if num_gt == 0 or num_anchors == 0:
        return out

    per_gt_bytes = batch * num_anchors * num_kpts * 4
    chunk = max(1, int(max_chunk_bytes // max(per_gt_bytes, 1)))
    px = pred_xy[..., 0].unsqueeze(1)  # (B,1,N,K)
    py = pred_xy[..., 1].unsqueeze(1)
    labeled_f = labeled.to(torch.float32)
    counts = labeled_f.sum(-1).clamp(min=1.0)  # (B,M)
    # exp(-d2 / denom) == exp(d2 / -denom) bit for bit; negating the small tensor saves a pass
    neg_denom = -oks_denominator(scale2, sigmas, eps=eps)  # (B,M,K)
    for m0 in range(0, num_gt, chunk):
        m1 = min(num_gt, m0 + chunk)
        gx = gt_xy[:, m0:m1, :, 0].unsqueeze(2)  # (B,m,1,K)
        gy = gt_xy[:, m0:m1, :, 1].unsqueeze(2)
        d2 = (px - gx) ** 2 + (py - gy) ** 2  # (B,m,N,K)
        e = torch.exp(d2 / neg_denom[:, m0:m1].unsqueeze(2))
        e = e * labeled_f[:, m0:m1].unsqueeze(2)
        out[:, m0:m1] = e.sum(-1) / counts[:, m0:m1].unsqueeze(-1)
    return out


def candidate_oks(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    labeled: torch.Tensor,
    scale2: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """OKS of explicit (anchor, GT) pairs -- the sparse form of :func:`pairwise_oks`.

    pred_xy: (..., T, K, 2) predictions gathered at each pair's anchor; leading dims
             (e.g. the head index) broadcast against the shared GT side
    gt_xy:   (T, K, 2)      GT keypoints of each pair
    labeled: (T, K) bool
    scale2:  (T,)
    Returns (..., T). Per element this is exactly the arithmetic of the dense path, so
    the assigner's result does not depend on which one produced the values.
    """

    pred_xy = pred_xy.float()
    gt_xy = gt_xy.float()
    sigmas = sigmas.to(device=pred_xy.device, dtype=torch.float32)
    if pred_xy.shape[-3] == 0:
        return pred_xy.new_zeros(pred_xy.shape[:-2])
    labeled_f = labeled.to(torch.float32)
    counts = labeled_f.sum(-1).clamp(min=1.0)  # (T,)
    neg_denom = -oks_denominator(scale2.float(), sigmas, eps=eps)  # (T,K)
    d2 = (pred_xy[..., 0] - gt_xy[..., 0]) ** 2 + (pred_xy[..., 1] - gt_xy[..., 1]) ** 2
    e = torch.exp(d2 / neg_denom)
    e = e * labeled_f
    return e.sum(-1) / counts


def smooth_oks(u2: torch.Tensor) -> torch.Tensor:
    """Smooth-OKS similarity from ``u^2``: ``exp(-u^2)`` for ``u^2 < 1``, else ``exp(-(2u - 1))``."""

    u = torch.sqrt(u2.clamp(min=1e-12))
    inner = torch.exp(-u2)
    outer = torch.exp(-(2.0 * u - 1.0))
    return torch.where(u2 < 1.0, inner, outer)


def plain_oks(u2: torch.Tensor) -> torch.Tensor:
    """Plain OKS similarity ``exp(-u^2)``."""

    return torch.exp(-u2)


def keypoint_u2(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    scale2: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """``u^2 = d^2 / (2 s^2 (2 sigma_k)^2)`` for matched points.

    pred_xy, gt_xy: (P, K, 2); scale2: (P,); sigmas: (K,) -> (P, K).
    """

    d2 = ((pred_xy - gt_xy) ** 2).sum(-1)
    denom = oks_denominator(scale2, sigmas.to(device=pred_xy.device, dtype=pred_xy.dtype), eps=eps)
    return d2 / denom


def oks_loss_terms(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    scale2: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    kind: str = "soks",
    eps: float = 1e-9,
) -> torch.Tensor:
    """Per-point ``1 - S(u)`` with ``S`` Smooth-OKS (``kind="soks"``) or plain OKS.

    Shapes as :func:`keypoint_u2`; returns ``(P, K)``. The caller masks unlabeled points.
    """

    u2 = keypoint_u2(pred_xy, gt_xy, scale2, sigmas, eps=eps)
    if kind == "soks":
        sim = smooth_oks(u2)
    elif kind == "oks":
        sim = plain_oks(u2)
    else:
        raise ValueError(f"unknown kpt_loss kind {kind!r} (expected 'soks' or 'oks')")
    return 1.0 - sim
