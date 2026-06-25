"""Total XTX loss (spec section 7.4): box + cls + keypoint + visibility.

Combines the task-aligned one-to-many and consistent one-to-one assignments over
both heads, CIoU box loss, varifocal classification, RLE (or OKS+L1) keypoint
localisation, and 3-way visibility cross-entropy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.head import HeadOutput, RawPredictions, decode_boxes, decode_keypoints
from ..utils.keypoints import OKS_SIGMAS
from .assignment import TaskAlignedAssigner
from .detection_loss import box_area_xyxy, ciou_loss, VarifocalLoss
from .rle import OKSL1Loss, RLELoss

Target = dict[str, torch.Tensor]  # {"boxes": (M,4) px, "keypoints": (M,K,3) px}


class XTXLoss(nn.Module):
    """Weighted multi-task loss for the XTX dual head."""

    def __init__(
        self,
        *,
        num_keypoints: int = 17,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        kpt_weight: float = 12.0,
        vis_weight: float = 1.0,
        center_prior_weight: float = 0.1,
        keypoint_loss: str = "rle",
        network_size: int = 384,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.kpt_weight = kpt_weight
        self.vis_weight = vis_weight
        self.keypoint_loss = keypoint_loss

        self.assigner_o2m = TaskAlignedAssigner(
            topk=10, center_prior_weight=center_prior_weight, network_size=network_size
        )
        self.assigner_o2o = TaskAlignedAssigner(
            topk=1, center_prior_weight=center_prior_weight, network_size=network_size
        )
        self.cls_loss = VarifocalLoss(use_varifocal=True)
        if keypoint_loss == "rle":
            self.rle = RLELoss()
            self.oks_l1: OKSL1Loss | None = None
        else:
            self.rle = None  # type: ignore[assignment]
            self.oks_l1 = OKSL1Loss(kpt_sigmas=torch.as_tensor(OKS_SIGMAS, dtype=torch.float32))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "XTXLoss":
        loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
        return cls(
            num_keypoints=int(model_cfg.get("num_keypoints", 17)),
            box_weight=float(loss_cfg.get("box", 7.5)),
            cls_weight=float(loss_cfg.get("cls", 0.5)),
            kpt_weight=float(loss_cfg.get("kpt", 12.0)),
            vis_weight=float(loss_cfg.get("vis", 1.0)),
            center_prior_weight=float(loss_cfg.get("center_prior_weight", 0.1)),
            keypoint_loss=str(loss_cfg.get("keypoint_loss", "rle")),
            network_size=int(data_cfg.get("network_size", 384)),
        )

    def forward(self, output: HeadOutput, targets: list[Target]) -> dict[str, torch.Tensor]:
        heads: list[RawPredictions] = [output.one2one]
        if output.one2many is not None:
            heads.append(output.one2many)

        device = output.one2one.box.device
        assigners = [self.assigner_o2o, self.assigner_o2m]
        agg = {k: torch.zeros((), device=device) for k in ("box", "cls", "kpt", "vis")}
        for raw, assigner in zip(heads, assigners):
            head_losses = self._head_loss(raw, targets, assigner)
            for key, value in head_losses.items():
                agg[key] = agg[key] + value

        total = (
            self.box_weight * agg["box"]
            + self.cls_weight * agg["cls"]
            + self.kpt_weight * agg["kpt"]
            + self.vis_weight * agg["vis"]
        )
        return {
            "total": total,
            "box": agg["box"].detach(),
            "cls": agg["cls"].detach(),
            "kpt": agg["kpt"].detach(),
            "vis": agg["vis"].detach(),
        }

    def _head_loss(
        self, raw: RawPredictions, targets: list[Target], assigner: TaskAlignedAssigner
    ) -> dict[str, torch.Tensor]:
        device = raw.box.device
        pred_boxes = decode_boxes(raw.box, raw.anchors, raw.strides)  # (B,N,4)
        pred_scores = torch.sigmoid(raw.cls).squeeze(-1)  # (B,N)
        mu_xy, _, _ = decode_keypoints(raw.kpt, raw.anchors, raw.strides)  # (B,N,K,2)
        strides = raw.strides  # (N,)

        cls_loss = torch.zeros((), device=device)
        box_loss = torch.zeros((), device=device)
        kpt_loss = torch.zeros((), device=device)
        vis_loss = torch.zeros((), device=device)
        score_sum = torch.zeros((), device=device)
        kpt_count = torch.zeros((), device=device)
        vis_count = torch.zeros((), device=device)

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)
            gt_kpts = target["keypoints"].to(device)
            result = assigner.assign(
                pred_scores=pred_scores[b],
                pred_boxes=pred_boxes[b],
                anchors=raw.anchors,
                gt_boxes=gt_boxes,
            )
            target_scores = result.target_scores
            cls_loss = cls_loss + self.cls_loss(raw.cls[b].squeeze(-1), target_scores)
            score_sum = score_sum + target_scores.sum()

            fg = result.fg_mask
            if fg.any() and gt_boxes.shape[0] > 0:
                assigned = result.gt_idx[fg]
                weight = target_scores[fg]
                box_loss = box_loss + (ciou_loss(pred_boxes[b][fg], gt_boxes[assigned]) * weight).sum()

                gt_k = gt_kpts[assigned]  # (P,K,3)
                pred_mu = mu_xy[b][fg]  # (P,K,2)
                vis_gt = gt_k[..., 2].long().clamp(0, 2)  # (P,K)

                vis_logits = raw.kpt[b][fg][..., 3:6]  # (P,K,3)
                vis_loss = vis_loss + F.cross_entropy(
                    vis_logits.reshape(-1, 3), vis_gt.reshape(-1), reduction="sum"
                )
                vis_count = vis_count + vis_gt.numel()

                present = vis_gt > 0  # localisation only on present points
                if present.any():
                    sel_mu = pred_mu[present]  # (Q,2)
                    sel_gt = gt_k[..., :2][present]  # (Q,2)
                    if self.keypoint_loss == "rle" and self.rle is not None:
                        anchor_stride = strides[fg].reshape(-1, 1).expand(-1, self.num_keypoints)
                        log_sigma = raw.kpt[b][fg][..., 2]  # (P,K)
                        sigma_px = (F.softplus(log_sigma) * anchor_stride + 1.0)
                        sel_sigma = sigma_px[present].unsqueeze(-1).expand(-1, 2)
                        per_pt = self.rle(pred_xy=sel_mu, pred_sigma=sel_sigma, gt_xy=sel_gt)
                    else:
                        area = box_area_xyxy(gt_boxes[assigned]).unsqueeze(1).expand(-1, self.num_keypoints)
                        kpt_idx = (
                            torch.arange(self.num_keypoints, device=device)
                            .unsqueeze(0)
                            .expand(present.shape[0], -1)
                        )
                        per_pt = self.oks_l1(  # type: ignore[union-attr]
                            pred_xy=sel_mu,
                            gt_xy=sel_gt,
                            area=area[present],
                            kpt_index=kpt_idx[present],
                        )
                    kpt_loss = kpt_loss + per_pt.sum()
                    kpt_count = kpt_count + present.sum()

        score_norm = score_sum.clamp(min=1.0)
        return {
            "cls": cls_loss / score_norm,
            "box": box_loss / score_norm,
            "kpt": kpt_loss / kpt_count.clamp(min=1.0),
            "vis": vis_loss / vis_count.clamp(min=1.0),
        }
