"""XTX4 model assembly + builder (CONTRACTS.md section 4.1).

``XTX4Model`` wires backbone -> PAN neck (-> optional P2-assist) -> dual pose
head. ``build_model(config)`` resolves ``config["model"]`` through
:func:`resolve_spec`. ``fuse()`` produces the deployment graph (Conv+BN folded,
RepConv reparameterised, one2many head dropped); ``num_parameters(deploy=True)``
and ``mac_count(S)`` are measured on a fused *copy* so the training model is
never modified by profiling.

The network is fully convolutional: any input size divisible by 32 is valid
(256/320/384 for the ``u`` presets).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from ..utils.logging import get_logger
from .backbone import Backbone
from .head import PoseHead
from .head_types import (
    PRESENT_THRESHOLD,
    SIGMA_FLOOR_UNITS,
    VISIBLE_THRESHOLD,
    HeadOutput,
    decode_detections,
)
from .neck import P2Assist, PANNeck
from .organs import OrganLayer, decode_organs
from .scaling import VariantSpec, resolve_spec

logger = get_logger(__name__)


class XTX4Model(nn.Module):
    """End-to-end box-free XTX4 pose network (single class, anchor-free, NMS-free)."""

    def __init__(self, *, spec: VariantSpec) -> None:
        super().__init__()
        self.spec = spec.validate()
        self.num_keypoints = spec.num_keypoints
        self._fused = False

        self.backbone = Backbone(spec=spec)
        self.neck = PANNeck(spec=spec, backbone_channels=self.backbone.out_channels)
        self.p2_assist: P2Assist | None = (
            P2Assist(p2_channels=self.backbone.out_channels["P2"]) if spec.p2_assist else None
        )
        self.head = PoseHead(
            spec=spec,
            in_channels_list=self.neck.out_channels,
            strides=self.neck.out_strides,
            assist_channels=self.p2_assist.out_channels if self.p2_assist is not None else 0,
        )
        # Parameter-free brain/chest layer after the one2one head (export and predict only, never trained).
        self.organs: OrganLayer | None = OrganLayer(spec.num_keypoints) if spec.num_keypoints == 9 else None

    # ------------------------------------------------------------------ API
    @property
    def strides(self) -> list[int]:
        return list(self.neck.out_strides)

    @property
    def fused(self) -> bool:
        return self._fused

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Backbone + neck (+ assist); shared by training forward and export."""

        features = self.backbone(x)
        pyramid = self.neck(features)
        assist = self.p2_assist(features["P2"]) if self.p2_assist is not None else None
        return pyramid, assist

    def forward(self, x: torch.Tensor) -> HeadOutput:
        pyramid, assist = self.forward_features(x)
        return self.head(pyramid, assist)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        max_detections: int = 100,
        sigma_conf_coeff: float = 0.0,
        sigma_floor: float = SIGMA_FLOOR_UNITS,
        present_threshold: float = PRESENT_THRESHOLD,
        visible_threshold: float = VISIBLE_THRESHOLD,
        with_organs: bool = False,
    ) -> tuple[list[torch.Tensor], ...]:
        """Eval-mode inference: per-image ``(dets, vis, sigma_px)`` (see head_types); with
        ``with_organs`` also ``organs`` ``(D, 2, 4)`` ``(x, y, radius, conf)`` in network pixels for the
        same detections (brain, chest)."""

        was_training = self.training
        self.eval()
        try:
            output = self.forward(x)
        finally:
            if was_training:
                self.train()
        dets, vis, sigma = decode_detections(
            output.one2one,
            conf_threshold=conf_threshold,
            max_detections=max_detections,
            sigma_conf_coeff=sigma_conf_coeff,
            sigma_floor=sigma_floor,
            present_threshold=present_threshold,
            visible_threshold=visible_threshold,
        )
        if not with_organs:
            return dets, vis, sigma
        if self.organs is None:
            raise ValueError("Organs need the nine XTX4 landmarks")
        raw = output.one2one
        b, n = raw.kpt.shape[0], raw.kpt.shape[1]
        organs_px = decode_organs(self.organs(raw.kpt.reshape(b, n, -1)), raw.anchors, raw.strides)
        scores = torch.sigmoid(raw.person).squeeze(-1)
        organs: list[torch.Tensor] = []
        for i in range(b):
            # the same selection as decode_detections (threshold, stable descending sort, top-k)
            keep = torch.nonzero(scores[i] >= conf_threshold).flatten()
            order = torch.argsort(scores[i][keep], descending=True, stable=True)[:max_detections]
            organs.append(organs_px[i][keep[order]])
        return dets, vis, sigma, organs

    @torch.no_grad()
    def fuse(self) -> "XTX4Model":
        """Fold Conv+BN, reparameterise RepConv, drop one2many (in place, idempotent)."""

        if self._fused:
            return self
        self.backbone.fuse()
        self.neck.fuse()
        if self.p2_assist is not None:
            self.p2_assist.fuse()
        self.head.fuse()
        self._fused = True
        if not getattr(self, "_quiet_fuse", False):
            logger.info("Fused XTX4-%s for deployment (one2many head dropped)", self.spec.name)
        return self

    def num_parameters(self, deploy: bool = False) -> int:
        """Parameter count; ``deploy=True`` counts the fused one2one graph only."""

        if not deploy or self._fused:
            return sum(p.numel() for p in self.parameters())
        return self._deploy_copy().num_parameters()

    def mac_count(self, input_size: int) -> float:
        """GMAC of the fused one2one graph for a ``(1, 3, S, S)`` input (conv/linear hooks)."""

        model = self if self._fused else self._deploy_copy()
        return count_macs(model, input_size=input_size)

    def _deploy_copy(self) -> "XTX4Model":
        """Fused CPU copy for profiling (the training model is left untouched)."""

        clone = copy.deepcopy(self).cpu()
        clone._quiet_fuse = True  # profiling copies should not spam the training log
        clone.fuse()
        return clone.eval()


def count_macs(model: nn.Module, *, input_size: int) -> float:
    """Multiply-accumulates (GMAC) of one forward pass via Conv2d/Linear hooks."""

    total = 0.0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        out_h, out_w = output.shape[-2:]
        kh, kw = module.kernel_size
        total += out_h * out_w * module.out_channels * (module.in_channels // module.groups) * kh * kw

    def linear_hook(module: nn.Linear, inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return total / 1e9


def build_model(config: dict[str, Any] | None = None, **overrides: Any) -> XTX4Model:
    """Build an :class:`XTX4Model` from ``config["model"]`` (plus keyword overrides)."""

    config = config or {}
    model_cfg = dict(config.get("model", {})) if isinstance(config, dict) else {}
    model_cfg.update({k: v for k, v in overrides.items() if v is not None})
    spec = resolve_spec(model_cfg)
    model = XTX4Model(spec=spec)
    logger.info(
        "Built XTX4-%s: levels=%s block=%s act=%s p2_assist=%s params=%.3fM",
        spec.name, spec.levels, spec.block, spec.act, spec.p2_assist, model.num_parameters() / 1e6,
    )
    return model
