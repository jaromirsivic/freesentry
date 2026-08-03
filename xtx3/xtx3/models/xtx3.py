"""XTX3 model assembly + builder.

``XTX3Model`` wires backbone -> PAN neck -> dual pose head. ``build_model``
constructs a variant from the scaling table and config. ``fuse`` produces the
deployment graph (Conv+BN folded, RepConv reparameterised, one-to-many head
dropped) so inference is NMS-free via the one-to-one head only.

The network is fully convolutional; any input size divisible by 32 is valid
(the ``u`` variant is exercised at 256/320/384).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..utils.keypoints import NUM_KEYPOINTS
from ..utils.logging import get_logger
from .backbone import Backbone
from .head import HeadOutput, PoseHead, decode_detections_full
from .neck import PANNeck
from .scaling import get_variant

logger = get_logger(__name__)

DEFAULT_NETWORK_SIZE = 384


class XTX3Model(nn.Module):
    """End-to-end XTX3 pose network (single class, anchor-free, NMS-free)."""

    def __init__(
        self,
        *,
        variant: str = "n",
        num_keypoints: int = NUM_KEYPOINTS,
        use_p2: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = get_variant(variant)
        self.variant = cfg.name
        self.num_keypoints = num_keypoints
        self.use_p2 = cfg.use_p2_default if use_p2 is None else use_p2
        self._fused = False

        self.backbone = Backbone(cfg=cfg)
        self.neck = PANNeck(
            cfg=cfg,
            backbone_channels=self.backbone.out_channels,
            use_p2=self.use_p2,
        )
        self.head = PoseHead(
            in_channels_list=self.neck.out_channels,
            strides=self.neck.out_strides,
            num_keypoints=num_keypoints,
            depthwise=cfg.depthwise,
            box_cap=cfg.head_box_cap,
            kpt_cap=cfg.head_kpt_cap,
        )

    @property
    def strides(self) -> list[int]:
        return self.neck.out_strides

    def forward(self, x: torch.Tensor) -> HeadOutput:
        """Run backbone -> neck -> head. Returns :class:`HeadOutput`."""

        features = self.backbone(x)
        pyramid = self.neck(features)
        return self.head(pyramid)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        *,
        conf_threshold: float = 0.25,
        max_detections: int = 300,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Eval-mode inference returning decoded ``(dets, visibility)`` per image."""

        was_training = self.training
        self.eval()
        try:
            output = self.forward(x)
        finally:
            if was_training:
                self.train()
        return decode_detections_full(
            output.one2one,
            num_keypoints=self.num_keypoints,
            conf_threshold=conf_threshold,
            max_detections=max_detections,
        )

    @torch.no_grad()
    def fuse(self) -> "XTX3Model":
        """Fold Conv+BN, reparameterise RepConv, drop the one-to-many head."""

        if self._fused:
            return self
        for module in (self.backbone, self.neck):
            module.fuse()
        self._fuse_head_branches()
        self.head.drop_one2many()
        self._fused = True
        logger.info("Fused XTX3-%s for deployment (one-to-many head dropped)", self.variant)
        return self

    def _fuse_head_branches(self) -> None:
        for branch in (
            list(self.head.one2one.box_branches)
            + list(self.head.one2one.cls_branches)
            + list(self.head.one2one.kpt_branches)
        ):
            for layer in branch:
                if hasattr(layer, "fuse"):
                    layer.fuse()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    *,
    variant: str = "n",
    config: dict[str, Any] | None = None,
    use_p2: bool | None = None,
) -> XTX3Model:
    """Build an :class:`XTX3Model` for ``variant`` honouring ``config['model']``."""

    config = config or {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    num_keypoints = int(model_cfg.get("num_keypoints", NUM_KEYPOINTS))

    if use_p2 is None:
        use_p2_map = model_cfg.get("use_p2", {})
        if isinstance(use_p2_map, dict) and variant.lower() in use_p2_map:
            use_p2 = bool(use_p2_map[variant.lower()])

    return XTX3Model(variant=variant, num_keypoints=num_keypoints, use_p2=use_p2)
