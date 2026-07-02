"""Total XTX2 loss (spec section 7.4): box + cls + keypoint + visibility, with
**ProgLoss (Progressive Loss Balancing)** across the dual head.

The total is ``w_o2m(t) * L_one2many + w_o2o(t) * L_one2one`` where the weights
shift linearly over training progress ``t in [0,1]`` from emphasis on the
one-to-many head (early: rich dense signal) toward the one-to-one inference
head (late: align training with the deployed path). The trainer advances ``t``
via :meth:`XTX2Loss.set_progress`; the current weights are exposed for the
per-step log (acceptance criterion 7).

Per head: task-aligned assignment (with STAL + centre prior), CIoU box loss,
varifocal classification, RLE (or OKS+L1) keypoint localisation, and 3-way
visibility cross-entropy. Points with GT visibility 0 are masked out of the
localisation loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.head import HeadOutput, RawPredictions, decode_boxes, decode_keypoints
from ..utils.keypoints import OKS_SIGMAS
from .assignment import TaskAlignedAssigner
from .detection_loss import VarifocalLoss, box_area_xyxy, ciou_loss
from .rle import OKSL1Loss, RLELoss

Target = dict[str, torch.Tensor]  # {"boxes": (M,4) px, "keypoints": (M,K,3) px}


class ProgLossSchedule:
    """Linear ProgLoss weight schedule over training progress ``t in [0,1]``."""

    def __init__(
        self,
        *,
        o2m_start: float = 1.0,
        o2m_end: float = 0.25,
        o2o_start: float = 0.5,
        o2o_end: float = 1.0,
    ) -> None:
        self.o2m_start = o2m_start
        self.o2m_end = o2m_end
        self.o2o_start = o2o_start
        self.o2o_end = o2o_end

    def weights(self, t: float) -> tuple[float, float]:
        """Return ``(w_o2m, w_o2o)`` at progress ``t`` (clamped to [0,1])."""

        t = min(max(t, 0.0), 1.0)
        w_o2m = self.o2m_start + (self.o2m_end - self.o2m_start) * t
        w_o2o = self.o2o_start + (self.o2o_end - self.o2o_start) * t
        return w_o2m, w_o2o

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProgLossSchedule":
        data = data or {}
        return cls(
            o2m_start=float(data.get("o2m_start", 1.0)),
            o2m_end=float(data.get("o2m_end", 0.25)),
            o2o_start=float(data.get("o2o_start", 0.5)),
            o2o_end=float(data.get("o2o_end", 1.0)),
        )


class XTX2Loss(nn.Module):
    """Weighted multi-task loss for the XTX2 dual head with ProgLoss balancing."""

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
        stal_min_anchors: int = 4,
        stal_size_px: float = 8.0,
        min_stride: int = 8,
        progloss: ProgLossSchedule | None = None,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.kpt_weight = kpt_weight
        self.vis_weight = vis_weight
        self.keypoint_loss = keypoint_loss
        self.progloss = progloss or ProgLossSchedule()
        self._progress = 0.0

        self.assigner_o2m = TaskAlignedAssigner(
            topk=10,
            center_prior_weight=center_prior_weight,
            network_size=network_size,
            stal_min_anchors=stal_min_anchors,
            stal_size_px=stal_size_px,
            min_stride=min_stride,
        )
        # Consistent one-to-one: top-1 by the same metric (STAL not applicable).
        self.assigner_o2o = TaskAlignedAssigner(
            topk=1,
            center_prior_weight=center_prior_weight,
            network_size=network_size,
            stal_min_anchors=1,
            stal_size_px=stal_size_px,
            min_stride=min_stride,
        )
        self.cls_loss = VarifocalLoss(use_varifocal=True)
        if keypoint_loss == "rle":
            self.rle: RLELoss | None = RLELoss()
            self.oks_l1: OKSL1Loss | None = None
        else:
            self.rle = None
            self.oks_l1 = OKSL1Loss(kpt_sigmas=torch.as_tensor(OKS_SIGMAS, dtype=torch.float32))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "XTX2Loss":
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
            stal_min_anchors=int(loss_cfg.get("stal_min_anchors", 4)),
            stal_size_px=float(loss_cfg.get("stal_size_px", 8)),
            progloss=ProgLossSchedule.from_dict(loss_cfg.get("progloss")),
        )

    def set_progress(self, t: float) -> None:
        """Advance the ProgLoss schedule (t = global training progress in [0,1])."""

        self._progress = min(max(float(t), 0.0), 1.0)

    @property
    def progress(self) -> float:
        return self._progress

    def current_weights(self) -> tuple[float, float]:
        """Current ``(w_o2m, w_o2o)`` ProgLoss weights, for logging."""

        return self.progloss.weights(self._progress)

    def forward(self, output: HeadOutput, targets: list[Target]) -> dict[str, torch.Tensor]:
        device = output.one2one.box.device
        w_o2m, w_o2o = self.current_weights()

        o2o_losses = self._head_loss(output.one2one, targets, self.assigner_o2o)
        if output.one2many is not None:
            o2m_losses = self._head_loss(output.one2many, targets, self.assigner_o2m)
        else:
            o2m_losses = {k: torch.zeros((), device=device) for k in ("box", "cls", "kpt", "vis")}
            w_o2m = 0.0

        agg: dict[str, torch.Tensor] = {}
        for key in ("box", "cls", "kpt", "vis"):
            agg[key] = w_o2o * o2o_losses[key] + w_o2m * o2m_losses[key]

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
            "w_o2m": torch.tensor(w_o2m),
            "w_o2o": torch.tensor(w_o2o),
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
                # Regression targets the TRUE box even for STAL-inflated GTs.
                box_loss = box_loss + (ciou_loss(pred_boxes[b][fg], gt_boxes[assigned]) * weight).sum()

                gt_k = gt_kpts[assigned]  # (P,K,3)
                pred_mu = mu_xy[b][fg]  # (P,K,2)
                vis_gt = gt_k[..., 2].long().clamp(0, 2)  # (P,K)

                vis_logits = raw.kpt[b][fg][..., 3:6]  # (P,K,3)
                vis_loss = vis_loss + F.cross_entropy(
                    vis_logits.reshape(-1, 3), vis_gt.reshape(-1), reduction="sum"
                )
                vis_count = vis_count + vis_gt.numel()

                present = vis_gt > 0  # localisation only on labeled-present points
                if present.any():
                    sel_mu = pred_mu[present]  # (Q,2)
                    sel_gt = gt_k[..., :2][present]  # (Q,2)
                    if self.keypoint_loss == "rle" and self.rle is not None:
                        anchor_stride = strides[fg].reshape(-1, 1).expand(-1, self.num_keypoints)
                        log_sigma = raw.kpt[b][fg][..., 2]  # (P,K)
                        sigma_px = F.softplus(log_sigma) * anchor_stride + 1.0
                        sel_sigma = sigma_px[present].unsqueeze(-1).expand(-1, 2)
                        per_pt = self.rle(pred_xy=sel_mu, pred_sigma=sel_sigma, gt_xy=sel_gt)
                    else:
                        area = box_area_xyxy(gt_boxes[assigned]).unsqueeze(1).expand(-1, self.num_keypoints)
                        kpt_idx = (
                            torch.arange(self.num_keypoints, device=device)
                            .unsqueeze(0)
                            .expand(present.shape[0], -1)
                        )
                        per_pt = self.oks_l1(  # type: ignore[misc]
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
