"""Losses: CIoU detection, RLE keypoints, visibility CE, assignment, total loss."""

from __future__ import annotations

from .assignment import TaskAlignedAssigner
from .detection_loss import VarifocalLoss, ciou, ciou_loss, pairwise_iou
from .pose_loss import XTXLoss
from .rle import OKSL1Loss, RealNVPFlow, RLELoss

__all__ = [
    "TaskAlignedAssigner",
    "VarifocalLoss",
    "ciou",
    "ciou_loss",
    "pairwise_iou",
    "XTXLoss",
    "OKSL1Loss",
    "RealNVPFlow",
    "RLELoss",
]
