"""Dual (one2one + one2many) box-free pose head with nine RLE landmarks.

Per level a :class:`LevelHead` emits two maps from a shared 3x3 conv:

* **person**: 1 logit (``1x1 head_hidden -> 1``, bias init -4.6).
* **pose**: ``K * 5`` channels ``(dx, dy, log_sigma, present_logit, visible_logit)``
  per keypoint (``3x3 head_hidden(+assist) -> pose_hidden``, ``1x1 -> K*5``); the
  final 1x1 is initialised with small weights and bias ``[0, 0, 0.6, 0, 0]`` per
  keypoint so that at start ``sigma_units ~ 1`` and ``present = visible = 0.5``.

Two structurally identical :class:`SingleHead` sets are trained jointly (YOLO26
style): ``one2many`` (dense assignment, training only) and ``one2one`` (the only
head kept after :meth:`PoseHead.fuse`; NMS-free inference). The optional
P2-assist tensor is concatenated to the *pose* branch input of level 0 (P3)
only, in both heads, so the person branch and the shared conv stay unchanged.

``block="rep"`` uses :class:`RepConv3x3` for the 3x3 convs, ``block="dw"`` uses
``DWConv`` towers as in XTX3-u.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DWConv, RepConv3x3, fuse_children
from .head_types import (
    KPT_CHANNELS_PER_POINT,
    HeadOutput,
    RawPredictions,
    build_anchors,
    flatten_level_outputs,
)
from .scaling import VariantSpec

PERSON_BIAS_PRIOR = -4.6  # sigmoid(-4.6) ~= 0.01
POSE_BIAS_PER_POINT = (0.0, 0.0, 0.6, 0.0, 0.0)  # softplus(0.6) + 0.05 ~= 1.09 stride units


def _conv3x3(spec: VariantSpec, *, in_channels: int, out_channels: int) -> nn.Module:
    if spec.block == "rep":
        return RepConv3x3(in_channels=in_channels, out_channels=out_channels, act=spec.act)
    return DWConv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, act=spec.act)


class LevelHead(nn.Module):
    """Head of one pyramid level: shared 3x3 -> {person 1x1, pose 3x3 + 1x1}."""

    def __init__(
        self,
        *,
        spec: VariantSpec,
        in_channels: int,
        assist_channels: int,
        num_keypoints: int,
    ) -> None:
        super().__init__()
        self.assist_channels = int(assist_channels)
        self.shared = _conv3x3(spec, in_channels=in_channels, out_channels=spec.head_hidden)
        self.person = nn.Conv2d(spec.head_hidden, 1, kernel_size=1, bias=True)
        self.pose_conv = _conv3x3(
            spec, in_channels=spec.head_hidden + self.assist_channels, out_channels=spec.pose_hidden
        )
        self.pose_out = nn.Conv2d(
            spec.pose_hidden, num_keypoints * KPT_CHANNELS_PER_POINT, kernel_size=1, bias=True
        )
        self._init_outputs(num_keypoints)

    def _init_outputs(self, num_keypoints: int) -> None:
        nn.init.constant_(self.person.bias, PERSON_BIAS_PRIOR)
        nn.init.normal_(self.pose_out.weight, mean=0.0, std=0.01)
        bias = torch.tensor(POSE_BIAS_PER_POINT, dtype=torch.float32).repeat(num_keypoints)
        with torch.no_grad():
            self.pose_out.bias.copy_(bias)

    def forward(self, x: torch.Tensor, assist: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        person = self.person(shared)
        pose_in = shared if assist is None else torch.cat([shared, assist], dim=1)
        pose = self.pose_out(self.pose_conv(pose_in))
        return person, pose

    def fuse(self) -> "LevelHead":
        fuse_children(self)
        return self


class SingleHead(nn.Module):
    """One set of per-level heads (used for both one2one and one2many)."""

    def __init__(
        self,
        *,
        spec: VariantSpec,
        in_channels_list: list[int],
        assist_channels: int,
        num_keypoints: int,
    ) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            LevelHead(
                spec=spec,
                in_channels=c,
                assist_channels=assist_channels if i == 0 else 0,
                num_keypoints=num_keypoints,
            )
            for i, c in enumerate(in_channels_list)
        )

    def forward(
        self, features: list[torch.Tensor], assist: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        persons: list[torch.Tensor] = []
        poses: list[torch.Tensor] = []
        for i, (level, feat) in enumerate(zip(self.levels, features)):
            person, pose = level(feat, assist if i == 0 else None)
            persons.append(person)
            poses.append(pose)
        return persons, poses

    def fuse(self) -> "SingleHead":
        for level in self.levels:
            level.fuse()
        return self


class PoseHead(nn.Module):
    """Dual head: ``one2one`` (always) + ``one2many`` (training only, dropped by fuse)."""

    def __init__(
        self,
        *,
        spec: VariantSpec,
        in_channels_list: list[int],
        strides: list[int],
        assist_channels: int = 0,
    ) -> None:
        super().__init__()
        if len(in_channels_list) != len(strides):
            raise ValueError("in_channels_list and strides must have equal length")
        self.strides = list(strides)
        self.num_keypoints = spec.num_keypoints
        self.assist_channels = int(assist_channels)
        self.one2one = SingleHead(
            spec=spec, in_channels_list=in_channels_list,
            assist_channels=assist_channels, num_keypoints=spec.num_keypoints,
        )
        self.one2many: SingleHead | None = SingleHead(
            spec=spec, in_channels_list=in_channels_list,
            assist_channels=assist_channels, num_keypoints=spec.num_keypoints,
        )

    def assemble(
        self, persons: list[torch.Tensor], poses: list[torch.Tensor]
    ) -> RawPredictions:
        person, kpt, sizes = flatten_level_outputs(persons, poses, self.num_keypoints)
        anchors, strides = build_anchors(
            feature_sizes=sizes, strides=self.strides, device=person.device, dtype=person.dtype
        )
        return RawPredictions(person=person, kpt=kpt, anchors=anchors, strides=strides)

    def forward(self, features: list[torch.Tensor], assist: torch.Tensor | None = None) -> HeadOutput:
        o2o = self.assemble(*self.one2one(features, assist))
        o2m = None
        if self.training and self.one2many is not None:
            o2m = self.assemble(*self.one2many(features, assist))
        return HeadOutput(one2one=o2o, one2many=o2m)

    def drop_one2many(self) -> None:
        self.one2many = None

    def fuse(self) -> "PoseHead":
        """Fuse the one2one towers and drop the one2many head (idempotent)."""

        self.one2one.fuse()
        self.drop_one2many()
        return self
