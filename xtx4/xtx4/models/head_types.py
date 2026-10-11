"""Stable tensor contracts shared by the model, the losses, the trainer and the
export code (see ``CONTRACTS.md`` section 3).

Everything in this module is framework glue with no learnable parameters:

* :class:`RawPredictions` / :class:`HeadOutput` -- raw head tensors.
* :func:`build_anchors` -- anchor centres ``(i + 0.5) * stride`` per level,
  levels ordered by ascending stride, row-major within a level.
* :func:`sigma_units_from_raw` / :func:`decode_keypoints` -- the one and only
  decode of the ``(dx, dy, log_sigma, present_logit, visible_logit)`` layout.
* :func:`decode_detections` -- per-image detection lists for evaluation.

Keep this file dependency-free beyond torch so every package can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

KPT_CHANNELS_PER_POINT = 5  # dx, dy, log_sigma, present_logit, visible_logit
SIGMA_FLOOR_UNITS = 0.05  # added to softplus(log_sigma); sigma is in stride units
PRESENT_THRESHOLD = 0.5  # default decision thresholds of the visibility class (decode.present_threshold)
VISIBLE_THRESHOLD = 0.5  # decode.visible_threshold


@dataclass(slots=True)
class RawPredictions:
    """Flattened raw head outputs across all pyramid levels.

    Shapes (``N`` = total anchors across levels, ``K`` = keypoints):
        person:  (B, N, 1)      person logits
        kpt:     (B, N, K, 5)   raw (dx, dy, log_sigma, present_logit, visible_logit)
        anchors: (N, 2)         anchor centre in network pixels (x, y)
        strides: (N,)           stride (px) of each anchor's level
    """

    person: torch.Tensor
    kpt: torch.Tensor
    anchors: torch.Tensor
    strides: torch.Tensor


@dataclass(slots=True)
class HeadOutput:
    """Head forward result. ``one2many`` is populated only in training mode."""

    one2one: RawPredictions
    one2many: RawPredictions | None


def build_anchors(
    *,
    feature_sizes: list[tuple[int, int]],
    strides: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(anchors (N,2) centre px, strides (N,))`` for the given level grids."""

    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []
    for (height, width), stride in zip(feature_sizes, strides):
        shift_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        shift_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        anchor_points.append(points)
        stride_tensor.append(torch.full((height * width,), float(stride), device=device, dtype=dtype))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)


def flatten_level_outputs(
    persons: list[torch.Tensor],
    kpts: list[torch.Tensor],
    num_keypoints: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Flatten per-level maps ``(B, C, H, W)`` into ``(B, N, ...)`` tensors."""

    person_flat, kpt_flat, sizes = [], [], []
    for person, kpt in zip(persons, kpts):
        b, _, h, w = person.shape
        sizes.append((h, w))
        person_flat.append(person.permute(0, 2, 3, 1).reshape(b, h * w, 1))
        kpt_flat.append(
            kpt.permute(0, 2, 3, 1).reshape(b, h * w, num_keypoints, KPT_CHANNELS_PER_POINT)
        )
    return torch.cat(person_flat, dim=1), torch.cat(kpt_flat, dim=1), sizes


def sigma_units_from_raw(log_sigma: torch.Tensor, *, floor: float = SIGMA_FLOOR_UNITS) -> torch.Tensor:
    """RLE scale in stride units: ``softplus(raw) + floor`` (strictly positive)."""

    return F.softplus(log_sigma) + floor


def visibility_class(
    present_prob: torch.Tensor,
    visible_prob: torch.Tensor,
    *,
    present_threshold: float = PRESENT_THRESHOLD,
    visible_threshold: float = VISIBLE_THRESHOLD,
) -> torch.Tensor:
    """int64 class: 0 if ``present < present_threshold`` else 2 if ``visible >= visible_threshold`` else 1."""

    return torch.where(
        present_prob < present_threshold,
        torch.zeros_like(present_prob, dtype=torch.int64),
        torch.where(
            visible_prob >= visible_threshold,
            torch.full_like(present_prob, 2, dtype=torch.int64),
            torch.ones_like(present_prob, dtype=torch.int64),
        ),
    )


def decode_keypoints(
    raw_kpt: torch.Tensor,
    anchors: torch.Tensor,
    strides: torch.Tensor,
    *,
    sigma_conf_coeff: float = 0.0,
    sigma_floor: float = SIGMA_FLOOR_UNITS,
    present_threshold: float = PRESENT_THRESHOLD,
    visible_threshold: float = VISIBLE_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Decode raw keypoint maps ``(B, N, K, 5)``.

    Returns a dict with
        xy            (B, N, K, 2)  network pixels
        sigma_units   (B, N, K)     RLE scale in stride units
        sigma_px      (B, N, K)     RLE scale in pixels
        present_prob  (B, N, K)     sigmoid(present_logit)
        visible_prob  (B, N, K)     sigmoid(visible_logit) (P(visible | present))
        conf          (B, N, K)     present_prob * exp(-sigma_conf_coeff * sigma_units)
        visibility    (B, N, K)     int64 class (see :func:`visibility_class`; thresholds
                                    ``present_threshold`` / ``visible_threshold``)
    """

    ax = anchors[:, 0].reshape(1, -1, 1)
    ay = anchors[:, 1].reshape(1, -1, 1)
    stride = strides.reshape(1, -1, 1)
    kx = ax + raw_kpt[..., 0] * stride
    ky = ay + raw_kpt[..., 1] * stride
    xy = torch.stack([kx, ky], dim=-1)
    sigma_units = sigma_units_from_raw(raw_kpt[..., 2], floor=sigma_floor)
    present_prob = torch.sigmoid(raw_kpt[..., 3])
    visible_prob = torch.sigmoid(raw_kpt[..., 4])
    conf = present_prob
    if sigma_conf_coeff != 0.0:
        conf = conf * torch.exp(-float(sigma_conf_coeff) * sigma_units)
    return {
        "xy": xy,
        "sigma_units": sigma_units,
        "sigma_px": sigma_units * stride,
        "present_prob": present_prob,
        "visible_prob": visible_prob,
        "conf": conf,
        "visibility": visibility_class(
            present_prob, visible_prob, present_threshold=present_threshold, visible_threshold=visible_threshold
        ),
    }


@torch.no_grad()
def decode_detections(
    raw: RawPredictions,
    *,
    conf_threshold: float,
    max_detections: int,
    sigma_conf_coeff: float = 0.0,
    sigma_floor: float = SIGMA_FLOOR_UNITS,
    present_threshold: float = PRESENT_THRESHOLD,
    visible_threshold: float = VISIBLE_THRESHOLD,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Decode one head's raw predictions into per-image detections (no NMS).

    Returns ``(dets, vis, sigma)``:
        dets[i]  (D, 1 + 3K)  ``[score, K * (x, y, conf)]`` in network pixels,
                              sorted by descending score
        vis[i]   (D, K)       int64 visibility class: 0 absent, 1 occluded, 2 visible
                              (0 if present < present_threshold, else 2 if
                              visible >= visible_threshold, else 1)
        sigma[i] (D, K)       sigma in pixels (for diagnostics / confidence calibration)
    """

    num_keypoints = raw.kpt.shape[2]
    scores = torch.sigmoid(raw.person).squeeze(-1)  # (B, N)
    dec = decode_keypoints(
        raw.kpt, raw.anchors, raw.strides, sigma_conf_coeff=sigma_conf_coeff, sigma_floor=sigma_floor
    )
    dets: list[torch.Tensor] = []
    vis: list[torch.Tensor] = []
    sig: list[torch.Tensor] = []
    for b in range(scores.shape[0]):
        keep = torch.nonzero(scores[b] >= conf_threshold).flatten()
        if keep.numel() == 0:
            dets.append(torch.zeros((0, 1 + 3 * num_keypoints), device=scores.device))
            vis.append(torch.zeros((0, num_keypoints), dtype=torch.int64, device=scores.device))
            sig.append(torch.zeros((0, num_keypoints), device=scores.device))
            continue
        s = scores[b][keep]
        # Stable sort keeps equal scores in anchor-index order (matches the NumPy runtime decode).
        order = torch.argsort(s, descending=True, stable=True)[:max_detections]
        keep = keep[order]
        s = scores[b][keep]
        xy = dec["xy"][b][keep]
        conf = dec["conf"][b][keep]
        present = dec["present_prob"][b][keep]
        visible = dec["visible_prob"][b][keep]
        vis_class = visibility_class(
            present, visible, present_threshold=present_threshold, visible_threshold=visible_threshold
        )
        kpt_flat = torch.cat([xy, conf.unsqueeze(-1)], dim=-1).reshape(s.shape[0], -1)
        dets.append(torch.cat([s.unsqueeze(-1), kpt_flat], dim=-1))
        vis.append(vis_class)
        sig.append(dec["sigma_px"][b][keep])
    return dets, vis, sig
