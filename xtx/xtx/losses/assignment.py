"""Label assignment (spec section 7.4).

* **TaskAligned one-to-many** assigner: alignment metric ``score^alpha * iou^beta``
  with top-k candidates per GT (rich training signal).
* **Consistent one-to-one** assignment: the same metric with ``top-k = 1`` so the
  one-to-one head agrees with the one-to-many head, removing the need for NMS.
* **XTX centre-prior**: anchors near the image centre receive a small bonus in the
  alignment metric, aligning the network with the A/B/C centre-recovery objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .detection_loss import pairwise_iou


@dataclass(slots=True)
class AssignResult:
    """Per-anchor assignment for a single image.

    fg_mask:       (N,) bool, True where the anchor is a positive sample.
    gt_idx:        (N,) int64, GT index assigned to each anchor (0 where background).
    target_scores: (N,) float, normalised alignment metric (soft label for cls).
    """

    fg_mask: torch.Tensor
    gt_idx: torch.Tensor
    target_scores: torch.Tensor


class TaskAlignedAssigner:
    """Task-aligned assigner with an optional centre prior."""

    def __init__(
        self,
        *,
        topk: int = 10,
        alpha: float = 1.0,
        beta: float = 6.0,
        center_prior_weight: float = 0.1,
        network_size: int = 384,
        eps: float = 1e-9,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_prior_weight = center_prior_weight
        self.network_size = network_size
        self.eps = eps
        self._max_dist = (network_size / 2.0) * (2.0**0.5)

    def _center_bonus(self, anchors: torch.Tensor) -> torch.Tensor:
        center = self.network_size / 2.0
        dist = torch.sqrt(
            (anchors[:, 0] - center) ** 2 + (anchors[:, 1] - center) ** 2
        )
        dist_norm = (dist / self._max_dist).clamp(0.0, 1.0)
        return 1.0 - dist_norm  # 1 at centre, 0 at corner

    @torch.no_grad()
    def assign(
        self,
        *,
        pred_scores: torch.Tensor,  # (N,) sigmoid prob
        pred_boxes: torch.Tensor,  # (N, 4) xyxy px
        anchors: torch.Tensor,  # (N, 2) centre px
        gt_boxes: torch.Tensor,  # (M, 4) xyxy px
    ) -> AssignResult:
        num_anchors = anchors.shape[0]
        device = anchors.device
        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            return AssignResult(
                fg_mask=torch.zeros(num_anchors, dtype=torch.bool, device=device),
                gt_idx=torch.zeros(num_anchors, dtype=torch.int64, device=device),
                target_scores=torch.zeros(num_anchors, device=device),
            )

        # anchor-centre-inside-gt mask (M, N)
        ax = anchors[:, 0].unsqueeze(0)
        ay = anchors[:, 1].unsqueeze(0)
        in_gts = (
            (ax >= gt_boxes[:, 0:1])
            & (ax <= gt_boxes[:, 2:3])
            & (ay >= gt_boxes[:, 1:2])
            & (ay <= gt_boxes[:, 3:4])
        )

        iou = pairwise_iou(gt_boxes, pred_boxes)  # (M, N)
        scores = pred_scores.unsqueeze(0).expand(num_gt, -1)  # (M, N)
        align = scores.clamp(min=self.eps).pow(self.alpha) * iou.clamp(min=0).pow(self.beta)

        center_bonus = self._center_bonus(anchors).unsqueeze(0)  # (1, N)
        align = align * (1.0 + self.center_prior_weight * center_bonus)
        align = align * in_gts.float()

        # top-k candidates per GT
        topk = min(self.topk, align.shape[1])
        topk_vals, topk_idx = align.topk(topk, dim=1)
        cand_mask = torch.zeros_like(align, dtype=torch.bool)
        valid = topk_vals > self.eps
        rows = torch.arange(num_gt, device=device).unsqueeze(1).expand(-1, topk)
        cand_mask[rows[valid], topk_idx[valid]] = True
        cand_mask = cand_mask & in_gts

        # resolve anchors claimed by multiple GTs -> assign to highest IoU GT
        assigned_count = cand_mask.sum(dim=0)  # (N,)
        multi = assigned_count > 1
        if multi.any():
            iou_masked = iou * cand_mask.float()
            best_gt = iou_masked.argmax(dim=0)  # (N,)
            new_mask = torch.zeros_like(cand_mask)
            anchor_ids = torch.arange(num_anchors, device=device)
            new_mask[best_gt[multi], anchor_ids[multi]] = True
            cand_mask = torch.where(multi.unsqueeze(0), new_mask, cand_mask)

        fg_mask = cand_mask.any(dim=0)  # (N,)
        gt_idx = cand_mask.float().argmax(dim=0)  # (N,) valid only where fg

        # normalised alignment metric as soft cls target (YOLOv8 style)
        align_masked = align * cand_mask.float()
        max_align_per_gt = align_masked.amax(dim=1, keepdim=True).clamp(min=self.eps)  # (M,1)
        max_iou_per_gt = (iou * cand_mask.float()).amax(dim=1, keepdim=True)  # (M,1)
        norm_align = align_masked / max_align_per_gt * max_iou_per_gt  # (M,N)
        target_scores = norm_align.amax(dim=0)  # (N,)
        target_scores = target_scores * fg_mask.float()

        return AssignResult(fg_mask=fg_mask, gt_idx=gt_idx, target_scores=target_scores)
