"""Decoupled, dual, NMS-free pose head with RLE keypoints (spec section 6.4).

Per pyramid level a decoupled head emits three branches:

* **box**: 4 direct distance-to-edge values ``(l, t, r, b)`` -- no DFL.
* **cls/score**: 1 person-confidence logit.
* **keypoint (RLE)**: per keypoint ``(dx, dy, log_sigma, vis0, vis1, vis2)`` where
  ``(dx, dy)`` is a stride-normalised offset, ``log_sigma`` parameterises the RLE
  residual scale, and the three ``vis`` logits give the {absent, occluded, visible}
  class (section 7.2).

Two structurally identical heads are trained jointly (YOLOv10/YOLO26 style):

* **one-to-many**: dense predictions with a one-to-many assigner (training only).
* **one-to-one**: produces the final <=300 detections; the only head active at
  inference/export, giving native NMS-free output.

``drop_one2many()`` removes the auxiliary head for deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DWConv

KPT_CHANNELS_PER_POINT = 6  # dx, dy, log_sigma, vis0, vis1, vis2


@dataclass(slots=True)
class RawPredictions:
    """Flattened raw head outputs across all pyramid levels.

    Shapes (``N`` = total anchors across levels):
        box:     (B, N, 4)        raw ltrb (pre-activation)
        cls:     (B, N, 1)        logits
        kpt:     (B, N, K, 6)     raw (dx, dy, log_sigma, vis0, vis1, vis2)
        anchors: (N, 2)           anchor centre in network (384) pixels
        strides: (N,)             stride (px) of each anchor's level
    """

    box: torch.Tensor
    cls: torch.Tensor
    kpt: torch.Tensor
    anchors: torch.Tensor
    strides: torch.Tensor


@dataclass(slots=True)
class HeadOutput:
    """Head forward result. During training both heads are populated."""

    one2one: RawPredictions
    one2many: RawPredictions | None


def _branch(*, in_channels: int, hidden: int, out_channels: int, depthwise: bool) -> nn.Sequential:
    conv_cls = (lambda c_in, c_out: DWConv(in_channels=c_in, out_channels=c_out, kernel_size=3)) if depthwise \
        else (lambda c_in, c_out: ConvBNAct(in_channels=c_in, out_channels=c_out, kernel_size=3))
    return nn.Sequential(
        conv_cls(in_channels, hidden),
        conv_cls(hidden, hidden),
        nn.Conv2d(hidden, out_channels, kernel_size=1),
    )


class SingleHead(nn.Module):
    """One decoupled head (box + cls + kpt) shared structure across levels."""

    def __init__(
        self,
        *,
        in_channels_list: list[int],
        num_keypoints: int,
        depthwise: bool,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_levels = len(in_channels_list)
        kpt_out = num_keypoints * KPT_CHANNELS_PER_POINT

        # Cap head hidden widths so the (expensive) per-level branches do not blow
        # up the parameter/FLOP budget on wide deep stages (section 6.6 budgets).
        box_cap = 80
        kpt_cap = 128
        self.box_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        self.kpt_branches = nn.ModuleList()
        for c in in_channels_list:
            box_hidden = max(16, min(c // 2, box_cap))
            kpt_hidden = max(32, min(c, kpt_cap))
            self.box_branches.append(
                _branch(in_channels=c, hidden=box_hidden, out_channels=4, depthwise=depthwise)
            )
            self.cls_branches.append(
                _branch(in_channels=c, hidden=box_hidden, out_channels=1, depthwise=depthwise)
            )
            self.kpt_branches.append(
                _branch(in_channels=c, hidden=kpt_hidden, out_channels=kpt_out, depthwise=depthwise)
            )

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        boxes, clss, kpts = [], [], []
        for i, feat in enumerate(features):
            boxes.append(self.box_branches[i](feat))
            clss.append(self.cls_branches[i](feat))
            kpts.append(self.kpt_branches[i](feat))
        return boxes, clss, kpts


def build_anchors(
    *, feature_sizes: list[tuple[int, int]], strides: list[int], device: torch.device, dtype: torch.dtype
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


def _flatten_level_outputs(
    boxes: list[torch.Tensor],
    clss: list[torch.Tensor],
    kpts: list[torch.Tensor],
    num_keypoints: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    box_flat, cls_flat, kpt_flat, sizes = [], [], [], []
    for box, cls, kpt in zip(boxes, clss, kpts):
        b, _, h, w = box.shape
        sizes.append((h, w))
        box_flat.append(box.permute(0, 2, 3, 1).reshape(b, h * w, 4))
        cls_flat.append(cls.permute(0, 2, 3, 1).reshape(b, h * w, 1))
        kpt_flat.append(kpt.permute(0, 2, 3, 1).reshape(b, h * w, num_keypoints, KPT_CHANNELS_PER_POINT))
    return (
        torch.cat(box_flat, dim=1),
        torch.cat(cls_flat, dim=1),
        torch.cat(kpt_flat, dim=1),
        sizes,
    )


def decode_boxes(raw_box: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """Decode raw ltrb to xyxy boxes in network pixels.

    ``raw_box`` (B, N, 4); ``anchors`` (N, 2); ``strides`` (N,). Distances are made
    positive with softplus and scaled by the level stride (DFL-free direct regression).
    """

    dist = F.softplus(raw_box) * strides.reshape(1, -1, 1)
    ax = anchors[:, 0].reshape(1, -1)
    ay = anchors[:, 1].reshape(1, -1)
    x1 = ax - dist[..., 0]
    y1 = ay - dist[..., 1]
    x2 = ax + dist[..., 2]
    y2 = ay + dist[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def decode_keypoints(
    raw_kpt: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw keypoint maps.

    Returns ``(xy (B,N,K,2) px, conf (B,N,K), vis_class (B,N,K))`` where ``conf`` is
    the probability the point is present (occluded or visible) per section 7.2.
    """

    ax = anchors[:, 0].reshape(1, -1, 1)
    ay = anchors[:, 1].reshape(1, -1, 1)
    stride = strides.reshape(1, -1, 1)
    kx = ax + raw_kpt[..., 0] * stride
    ky = ay + raw_kpt[..., 1] * stride
    xy = torch.stack([kx, ky], dim=-1)
    vis_logits = raw_kpt[..., 3:6]
    vis_prob = torch.softmax(vis_logits, dim=-1)
    conf = vis_prob[..., 1] + vis_prob[..., 2]
    vis_class = vis_logits.argmax(dim=-1)
    return xy, conf, vis_class


class PoseHead(nn.Module):
    """Dual decoupled pose head (one-to-one + one-to-many)."""

    def __init__(
        self,
        *,
        in_channels_list: list[int],
        strides: list[int],
        num_keypoints: int,
        depthwise: bool,
    ) -> None:
        super().__init__()
        if len(in_channels_list) != len(strides):
            raise ValueError("in_channels_list and strides must have equal length")
        self.strides = list(strides)
        self.num_keypoints = num_keypoints
        self.one2one = SingleHead(
            in_channels_list=in_channels_list, num_keypoints=num_keypoints, depthwise=depthwise
        )
        self.one2many: SingleHead | None = SingleHead(
            in_channels_list=in_channels_list, num_keypoints=num_keypoints, depthwise=depthwise
        )
        self._init_bias()

    def _init_bias(self) -> None:
        # Initialise cls bias to a low prior so early training is stable.
        prior = -4.6  # sigmoid(-4.6) ~= 0.01
        for head in (self.one2one, self.one2many):
            if head is None:
                continue
            for branch in head.cls_branches:
                final = branch[-1]
                if isinstance(final, nn.Conv2d) and final.bias is not None:
                    nn.init.constant_(final.bias, prior)

    def _assemble(
        self, head: SingleHead, features: list[torch.Tensor]
    ) -> RawPredictions:
        boxes, clss, kpts = head(features)
        box, cls, kpt, sizes = _flatten_level_outputs(boxes, clss, kpts, self.num_keypoints)
        anchors, strides = build_anchors(
            feature_sizes=sizes,
            strides=self.strides,
            device=box.device,
            dtype=box.dtype,
        )
        return RawPredictions(box=box, cls=cls, kpt=kpt, anchors=anchors, strides=strides)

    def forward(self, features: list[torch.Tensor]) -> HeadOutput:
        o2o = self._assemble(self.one2one, features)
        if self.training and self.one2many is not None:
            o2m = self._assemble(self.one2many, features)
        else:
            o2m = None
        return HeadOutput(one2one=o2o, one2many=o2m)

    def drop_one2many(self) -> None:
        """Remove the auxiliary one-to-many head for deployment."""

        self.one2many = None


@torch.no_grad()
def decode_detections(
    raw: RawPredictions,
    *,
    num_keypoints: int,
    conf_threshold: float,
    max_detections: int,
) -> list[torch.Tensor]:
    """Decode one head's raw predictions into per-image ``(num_det, 56)`` tensors.

    Output layout per detection (section 6.5):
        [0:4]  bbox xyxy in 384 network pixels
        [4]    person score (sigmoid of cls logit)
        [5:56] 17 * (x, y, conf)
    Visibility classes are returned via :func:`decode_detections_full` when needed.
    """

    results, _ = decode_detections_full(
        raw,
        num_keypoints=num_keypoints,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
    )
    return results


@torch.no_grad()
def decode_detections_full(
    raw: RawPredictions,
    *,
    num_keypoints: int,
    conf_threshold: float,
    max_detections: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Like :func:`decode_detections` but also returns per-detection visibility classes.

    Returns ``(dets, vis)`` where ``dets[i]`` is ``(num_det, 56)`` and ``vis[i]`` is
    ``(num_det, K)`` int64 visibility classes for image ``i``.
    """

    scores = torch.sigmoid(raw.cls).squeeze(-1)  # (B, N)
    boxes = decode_boxes(raw.box, raw.anchors, raw.strides)  # (B, N, 4)
    kxy, kconf, kvis = decode_keypoints(raw.kpt, raw.anchors, raw.strides)

    batch_size = scores.shape[0]
    dets: list[torch.Tensor] = []
    vis: list[torch.Tensor] = []
    for b in range(batch_size):
        keep = scores[b] >= conf_threshold
        if keep.sum() == 0:
            dets.append(torch.zeros((0, 5 + 3 * num_keypoints), device=scores.device))
            vis.append(torch.zeros((0, num_keypoints), dtype=torch.int64, device=scores.device))
            continue
        s = scores[b][keep]
        bx = boxes[b][keep]
        kp = kxy[b][keep]
        kc = kconf[b][keep]
        kv = kvis[b][keep]
        if s.numel() > max_detections:
            topk = torch.topk(s, max_detections)
            idx = topk.indices
            s, bx, kp, kc, kv = s[idx], bx[idx], kp[idx], kc[idx], kv[idx]
        kpt_flat = torch.cat([kp, kc.unsqueeze(-1)], dim=-1).reshape(s.shape[0], -1)
        det = torch.cat([bx, s.unsqueeze(-1), kpt_flat], dim=-1)
        dets.append(det)
        vis.append(kv)
    return dets, vis
