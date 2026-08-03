"""Label assignment for the dual head.

* **TaskAligned one-to-many** assigner: alignment metric ``score^alpha * iou^beta``
  with top-k candidates per GT (rich training signal).
* **Consistent one-to-one** assignment: the same metric with ``top-k = 1`` so the
  one-to-one head agrees with the one-to-many head, removing the need for NMS.
* **STAL (Small-Target-Aware Label Assignment)**: for GT boxes smaller than
  ``stal_size_px`` (default 8 px in network space, i.e. smaller than one cell at
  the finest stride), the geometry used for candidate *selection* is decoupled
  from the geometry used for *regression*: the assignment surrogate box is
  inflated so the GT is guaranteed a minimum of ``stal_min_anchors`` (default 4)
  positive candidates, while regression still targets the true box. Critical
  for distant people in the pyramid's centre views and tiny people in the
  whole-frame view.
* **Centre prior**: anchors near the image centre receive a bonus in the
  alignment metric (the deployment cameras centre their subjects; in pyramid
  mode this also aligns with the centre-recovery objective).

Because XTX3-u runs at multiple input sizes (256/320/384), ``assign`` accepts
the actual ``network_size`` of the current batch so the centre prior scales
correctly.
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


def inflate_small_boxes(
    gt_boxes: torch.Tensor, *, stal_size_px: float, min_side: float
) -> torch.Tensor:
    """Return selection-surrogate boxes: small GTs inflated to ``min_side`` square.

    Boxes whose smaller side is below ``stal_size_px`` are expanded symmetrically
    about their centre so both sides are at least ``min_side``. Larger boxes are
    returned unchanged. Only used for candidate *selection* (STAL); regression
    always targets the true box.
    """

    if gt_boxes.shape[0] == 0:
        return gt_boxes
    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    small = torch.minimum(w, h) < stal_size_px
    if not small.any():
        return gt_boxes
    out = gt_boxes.clone()
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    new_w = torch.maximum(w, torch.full_like(w, min_side))
    new_h = torch.maximum(h, torch.full_like(h, min_side))
    out[small, 0] = (cx - new_w / 2.0)[small]
    out[small, 1] = (cy - new_h / 2.0)[small]
    out[small, 2] = (cx + new_w / 2.0)[small]
    out[small, 3] = (cy + new_h / 2.0)[small]
    return out


class TaskAlignedAssigner:
    """Task-aligned assigner with centre prior and STAL small-target guarantee."""

    def __init__(
        self,
        *,
        topk: int = 10,
        alpha: float = 1.0,
        beta: float = 6.0,
        center_prior_weight: float = 0.1,
        network_size: int = 384,
        stal_min_anchors: int = 4,
        stal_size_px: float = 8.0,
        min_stride: int = 8,
        eps: float = 1e-9,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_prior_weight = center_prior_weight
        self.network_size = network_size
        # STAL only makes sense for the one-to-many head; a top-1 assigner keeps
        # exactly one candidate per GT by construction.
        self.stal_min_anchors = min(stal_min_anchors, topk) if topk > 1 else 1
        self.stal_size_px = stal_size_px
        # Surrogate side guaranteeing >= 2 anchor columns and rows at the finest
        # stride, hence >= 4 anchor centres inside the inflated box.
        self.stal_min_side = 2.0 * float(min_stride)
        self.eps = eps

    def _center_bonus(self, anchors: torch.Tensor, network_size: int) -> torch.Tensor:
        center = network_size / 2.0
        max_dist = (network_size / 2.0) * (2.0**0.5)
        dist = torch.sqrt((anchors[:, 0] - center) ** 2 + (anchors[:, 1] - center) ** 2)
        dist_norm = (dist / max_dist).clamp(0.0, 1.0)
        return 1.0 - dist_norm  # 1 at centre, 0 at corner

    @torch.no_grad()
    def assign(
        self,
        *,
        pred_scores: torch.Tensor,  # (N,) sigmoid prob
        pred_boxes: torch.Tensor,  # (N, 4) xyxy px
        anchors: torch.Tensor,  # (N, 2) centre px
        gt_boxes: torch.Tensor,  # (M, 4) xyxy px (TRUE boxes; regression geometry)
        network_size: int | None = None,
    ) -> AssignResult:
        num_anchors = anchors.shape[0]
        device = anchors.device
        num_gt = gt_boxes.shape[0]
        size = int(network_size) if network_size is not None else self.network_size

        if num_gt == 0:
            return AssignResult(
                fg_mask=torch.zeros(num_anchors, dtype=torch.bool, device=device),
                gt_idx=torch.zeros(num_anchors, dtype=torch.int64, device=device),
                target_scores=torch.zeros(num_anchors, device=device),
            )

        # --- STAL: selection geometry uses inflated surrogates for small GTs ---
        sel_boxes = inflate_small_boxes(
            gt_boxes, stal_size_px=self.stal_size_px, min_side=self.stal_min_side
        )
        w = gt_boxes[:, 2] - gt_boxes[:, 0]
        h = gt_boxes[:, 3] - gt_boxes[:, 1]
        is_small = torch.minimum(w, h) < self.stal_size_px  # (M,)

        # anchor-centre-inside-selection-box mask (M, N)
        ax = anchors[:, 0].unsqueeze(0)
        ay = anchors[:, 1].unsqueeze(0)
        in_gts = (
            (ax >= sel_boxes[:, 0:1])
            & (ax <= sel_boxes[:, 2:3])
            & (ay >= sel_boxes[:, 1:2])
            & (ay <= sel_boxes[:, 3:4])
        )

        # Alignment metric on the selection geometry (score^a * iou^b, + centre prior).
        iou_sel = pairwise_iou(sel_boxes, pred_boxes)  # (M, N)
        iou_true = pairwise_iou(gt_boxes, pred_boxes)  # (M, N) for tie-break/targets
        scores = pred_scores.unsqueeze(0).expand(num_gt, -1)  # (M, N)
        align = scores.clamp(min=self.eps).pow(self.alpha) * iou_sel.clamp(min=0).pow(self.beta)

        center_bonus = self._center_bonus(anchors, size).unsqueeze(0)  # (1, N)
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

        # --- STAL guarantee: small GTs keep >= stal_min_anchors candidates. ---
        # Even when the align metric is ~0 (untrained network), the nearest
        # anchors inside (or closest to) the surrogate box are forced positive.
        if self.stal_min_anchors > 1 and bool(is_small.any()):
            gcx = ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0).unsqueeze(1)  # (M,1)
            gcy = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0).unsqueeze(1)
            dist2 = (ax - gcx) ** 2 + (ay - gcy) ** 2  # (M, N)
            # Prefer anchors inside the surrogate box; penalise the rest heavily
            # so they are only used when the box lies at the frame border.
            dist2 = dist2 + (~in_gts).float() * 1e12
            for g in torch.nonzero(is_small).flatten().tolist():
                need = self.stal_min_anchors - int(cand_mask[g].sum())
                if need <= 0:
                    continue
                k = min(self.stal_min_anchors, num_anchors)
                nearest = torch.topk(dist2[g], k, largest=False).indices
                cand_mask[g, nearest] = True

        # resolve anchors claimed by multiple GTs -> assign to highest selection IoU
        assigned_count = cand_mask.sum(dim=0)  # (N,)
        multi = assigned_count > 1
        if multi.any():
            iou_masked = iou_sel * cand_mask.float()
            best_gt = iou_masked.argmax(dim=0)  # (N,)
            new_mask = torch.zeros_like(cand_mask)
            anchor_ids = torch.arange(num_anchors, device=device)
            new_mask[best_gt[multi], anchor_ids[multi]] = True
            cand_mask = torch.where(multi.unsqueeze(0), new_mask, cand_mask)

        fg_mask = cand_mask.any(dim=0)  # (N,)
        gt_idx = cand_mask.float().argmax(dim=0)  # (N,) valid only where fg

        # normalised alignment metric as soft cls target (YOLOv8 style). Give the
        # STAL-forced anchors at least the per-GT floor so tiny targets receive a
        # usable positive signal even before the score/IoU warm up.
        align_masked = align * cand_mask.float()
        floor = 0.05 * cand_mask.float() * is_small.float().unsqueeze(1)
        align_masked = torch.maximum(align_masked, floor)
        max_align_per_gt = align_masked.amax(dim=1, keepdim=True).clamp(min=self.eps)  # (M,1)
        max_iou_per_gt = (iou_true * cand_mask.float()).amax(dim=1, keepdim=True)  # (M,1)
        max_iou_per_gt = torch.maximum(
            max_iou_per_gt, 0.05 * is_small.float().unsqueeze(1)
        )
        norm_align = align_masked / max_align_per_gt * max_iou_per_gt  # (M,N)
        target_scores = norm_align.amax(dim=0)  # (N,)
        target_scores = target_scores * fg_mask.float()

        return AssignResult(fg_mask=fg_mask, gt_idx=gt_idx, target_scores=target_scores)
