"""Residual Log-Likelihood Estimation (RLE) keypoint loss (arXiv:2107.11291).

The head predicts, per keypoint, a mean ``(mu_x, mu_y)`` and a scale ``sigma``. RLE
models the residual ``error = (gt - mu) / sigma`` as ``Q(error) = G(error) * Phi(error)``
where ``G`` is a simple Laplace prior and ``Phi`` is a learned normalising flow
(RealNVP). The negative log-likelihood in the reference convention is::

    loss = sum_d [ log(sigma_d) + log(2 sigma_d) + |error_d| ] - log Phi(error)

(XTX3 used a single ``log sigma`` term; both are selectable, see :class:`RLELoss`).
The flow is the XTX3 RealNVP kept verbatim. :class:`OKSL1Loss` is the XTX3 fallback
kept for reference only; XTX4 uses ``xtx4.losses.oks`` (COCO denominator).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _CouplingLayer(nn.Module):
    """Affine coupling for 2D inputs; transforms one dim conditioned on the other."""

    def __init__(self, *, transform_second: bool, hidden: int = 64) -> None:
        super().__init__()
        self.transform_second = transform_second
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2),
        )

    def transform(self, cond: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the ``target`` column (P,1) conditioned on ``cond`` (P,1).

        Returns ``(z_target, scale)`` with ``log_det = -scale`` (the caller negates once).
        """

        s_t = self.net(cond)
        scale = torch.tanh(s_t[:, :1])  # bounded log-scale for stability
        shift = s_t[:, 1:]
        z_target = (target - shift) * torch.exp(-scale)
        return z_target, scale.squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data ``x`` (P,2) -> latent ``z`` (P,2); return ``(z, log_det)``."""

        if self.transform_second:
            cond, target = x[:, :1], x[:, 1:]
        else:
            cond, target = x[:, 1:], x[:, :1]
        z_target, scale = self.transform(cond, target)
        log_det = -scale
        if self.transform_second:
            z = torch.cat([cond, z_target], dim=1)
        else:
            z = torch.cat([z_target, cond], dim=1)
        return z, log_det


class RealNVPFlow(nn.Module):
    """A small RealNVP flow over 2D residuals with a standard Laplace base."""

    def __init__(self, *, num_layers: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _CouplingLayer(transform_second=(i % 2 == 0), hidden=hidden) for i in range(num_layers)
        )

    @staticmethod
    def _base_log_prob(z: torch.Tensor) -> torch.Tensor:
        # Standard Laplace per dim: log p(z) = -log 2 - |z|
        return (-math.log(2.0) - z.abs()).sum(dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``log Phi(x)`` for residuals ``x`` of shape ``(P, 2)``."""

        if x.numel() == 0:
            return torch.zeros((0,), device=x.device, dtype=x.dtype)
        # the two columns are carried separately (no cat / re-slice per layer); the
        # arithmetic per element is exactly that of layer-by-layer ``forward``
        z0, z1 = x[:, :1], x[:, 1:]
        scale_total: torch.Tensor | None = None  # sum of the scales; log_det_total = -scale_total exactly
        for layer in self.layers:
            if layer.transform_second:
                z1, scale = layer.transform(z0, z1)
            else:
                z0, scale = layer.transform(z1, z0)
            scale_total = scale if scale_total is None else scale_total + scale
        base = ((-math.log(2.0) - z0.abs()) + (-math.log(2.0) - z1.abs())).squeeze(1)  # == _base_log_prob(z)
        return base if scale_total is None else base - scale_total


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation loss with an internal RealNVP flow.

    Two conventions for the per-coordinate terms (CONTRACTS.md section 7):

    * ``"reference"`` (original RLE code, YOLO26):
      ``log sigma - log phi(eps) + log(2 sigma) + |eps|`` -- two ``log sigma`` terms,
      the empirically validated form (sigma is penalised more strongly);
    * ``"xtx3"``: ``log sigma + log 2 + |eps| - log phi(eps)`` -- one ``log sigma``
      term, the properly normalised density XTX3 trained with.

    ``eps = (gt - mu) / sigma``; the caller passes ``mu``, ``sigma`` and ``gt`` in the
    same units (stride units in XTX4, so ``log sigma`` is scale-free per level).
    The two conventions differ by exactly ``sum_d log sigma_d``.

    Numerical safety (FP32 only; the caller disables autocast): ``sigma`` is clamped
    from below, ``eps`` is clamped to ``[-eps_clamp, eps_clamp]`` before the flow,
    and rows whose loss is non-finite are replaced by 0 so one broken point cannot
    poison the step.
    """

    def __init__(
        self,
        *,
        convention: str = "reference",
        num_layers: int = 4,
        hidden: int = 64,
        sigma_eps: float = 1e-4,
        eps_clamp: float = 100.0,
    ) -> None:
        super().__init__()
        if convention not in ("reference", "xtx3"):
            raise ValueError(f"unknown RLE convention {convention!r} (expected 'reference' or 'xtx3')")
        self.convention = convention
        self.flow = RealNVPFlow(num_layers=num_layers, hidden=hidden)
        self.sigma_eps = float(sigma_eps)
        self.eps_clamp = float(eps_clamp)

    def forward(
        self,
        *,
        pred_xy: torch.Tensor,  # (P, 2)
        pred_sigma: torch.Tensor,  # (P, 2) or (P, 1) or (P,), positive, same units as xy
        gt_xy: torch.Tensor,  # (P, 2)
    ) -> torch.Tensor:
        """Return per-point RLE loss ``(P,)`` (caller applies the visibility mask)."""

        if pred_xy.shape[0] == 0:
            return torch.zeros((0,), device=pred_xy.device, dtype=torch.float32)
        pred_xy = pred_xy.float()
        gt_xy = gt_xy.float()
        sigma = pred_sigma.float().reshape(pred_xy.shape[0], -1).expand(-1, 2).clamp(min=self.sigma_eps)
        error = ((gt_xy - pred_xy) / sigma).clamp(-self.eps_clamp, self.eps_clamp)
        log_phi = self.flow.log_prob(error)
        log_sigma = torch.log(sigma).sum(dim=1)
        laplace_term = (error.abs() + math.log(2.0)).sum(dim=1)
        if self.convention == "reference":
            loss = log_sigma + laplace_term + log_sigma - log_phi
        else:
            loss = log_sigma + laplace_term - log_phi
        # branch-free guard (a host sync here would stall the whole loss pipeline)
        finite = torch.isfinite(loss)
        return torch.where(finite, loss, torch.zeros_like(loss))


class OKSL1Loss(nn.Module):
    """Fallback keypoint loss: ``(1 - OKS) + lambda * L1`` (per point)."""

    def __init__(self, *, kpt_sigmas: torch.Tensor, l1_weight: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("kpt_sigmas", kpt_sigmas)
        self.l1_weight = l1_weight

    def forward(
        self,
        *,
        pred_xy: torch.Tensor,  # (P, 2) px
        gt_xy: torch.Tensor,  # (P, 2) px
        area: torch.Tensor,  # (P,) instance area px^2
        kpt_index: torch.Tensor,  # (P,) which keypoint each row is
    ) -> torch.Tensor:
        if pred_xy.shape[0] == 0:
            return torch.zeros((0,), device=pred_xy.device, dtype=pred_xy.dtype)
        sigmas = self.kpt_sigmas[kpt_index]
        d2 = ((pred_xy - gt_xy) ** 2).sum(dim=1)
        scale = 2.0 * (sigmas**2) * (area.clamp(min=1.0))
        oks = torch.exp(-d2 / scale.clamp(min=1e-9))
        l1 = (pred_xy - gt_xy).abs().sum(dim=1)
        return (1.0 - oks) + self.l1_weight * l1
