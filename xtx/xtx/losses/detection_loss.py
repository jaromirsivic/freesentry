"""Box (CIoU) and classification losses + torch box utilities (spec section 7.4)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Area of ``(..., 4)`` xyxy boxes (clamped non-negative)."""

    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU between ``(M,4)`` and ``(N,4)`` xyxy boxes -> ``(M, N)``."""

    area_a = box_area_xyxy(boxes_a).unsqueeze(1)
    area_b = box_area_xyxy(boxes_b).unsqueeze(0)
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


def ciou(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise Complete IoU between matched ``(M,4)`` xyxy box pairs -> ``(M,)``."""

    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    inter_w = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0)
    inter_h = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
    ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
    c2 = cw * cw + ch * ch + eps

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    wp = (px2 - px1).clamp(min=eps)
    hp = (py2 - py1).clamp(min=eps)
    wt = (tx2 - tx1).clamp(min=eps)
    ht = (ty2 - ty1).clamp(min=eps)
    v = (4 / (math.pi**2)) * (torch.atan(wt / ht) - torch.atan(wp / hp)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``1 - CIoU`` per matched pair."""

    return 1.0 - ciou(pred, target)


class VarifocalLoss(nn.Module):
    """Varifocal classification loss (IoU-aware), with a plain BCE fallback."""

    def __init__(self, *, alpha: float = 0.75, gamma: float = 2.0, use_varifocal: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_varifocal = use_varifocal

    def forward(
        self, pred_logits: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        """``pred_logits`` and ``target_score`` are broadcastable; returns scalar sum."""

        if not self.use_varifocal:
            return F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="sum")
        prob = pred_logits.sigmoid()
        weight = self.alpha * prob.pow(self.gamma) * (1 - target_score) + target_score
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none")
        return (loss * weight).sum()
