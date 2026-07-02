"""Residual Log-Likelihood Estimation (RLE) keypoint loss (arXiv:2107.11291).

The head predicts, per keypoint, a mean ``(mu_x, mu_y)`` and a scale ``sigma``. RLE
models the residual ``error = (gt - mu) / sigma`` as ``Q(error) = G(error) * Phi(error)``
where ``G`` is a simple Laplace prior and ``Phi`` is a learned normalising flow
(RealNVP). The negative log-likelihood is::

    loss = sum_d [ log(sigma_d) + log 2 + |error_d| ] - log Phi(error)

A robust **OKS + L1** fallback (:class:`OKSL1Loss`) is provided behind the
``loss.keypoint_loss: "rle" | "oks_l1"`` config flag (spec section 7.4) for cases
where flow training is unstable.
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data ``x`` (P,2) -> latent ``z`` (P,2); return ``(z, log_det)``."""

        if self.transform_second:
            cond, target = x[:, :1], x[:, 1:]
        else:
            cond, target = x[:, 1:], x[:, :1]
        s_t = self.net(cond)
        scale = torch.tanh(s_t[:, :1])  # bounded log-scale for stability
        shift = s_t[:, 1:]
        z_target = (target - shift) * torch.exp(-scale)
        log_det = -scale.squeeze(1)
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
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return self._base_log_prob(z) + log_det_total


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation loss with an internal RealNVP flow."""

    def __init__(self, *, num_layers: int = 4, hidden: int = 64, sigma_eps: float = 1e-4) -> None:
        super().__init__()
        self.flow = RealNVPFlow(num_layers=num_layers, hidden=hidden)
        self.sigma_eps = sigma_eps

    def forward(
        self,
        *,
        pred_xy: torch.Tensor,  # (P, 2) px
        pred_sigma: torch.Tensor,  # (P, 2) px, positive
        gt_xy: torch.Tensor,  # (P, 2) px
    ) -> torch.Tensor:
        """Return per-point RLE loss ``(P,)`` (caller applies the visibility mask)."""

        if pred_xy.shape[0] == 0:
            return torch.zeros((0,), device=pred_xy.device, dtype=pred_xy.dtype)
        sigma = pred_sigma.clamp(min=self.sigma_eps)
        error = (gt_xy - pred_xy) / sigma
        log_phi = self.flow.log_prob(error)
        laplace_term = (error.abs() + math.log(2.0)).sum(dim=1)
        sigma_term = torch.log(sigma).sum(dim=1)
        return sigma_term + laplace_term - log_phi


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
