"""Parameter-free organ layer: brain and chest circles appended after the one2one head.

``OrganLayer`` turns the raw pose tensor ``[B, N, K*5]`` into ``organs [B, N, 2*4]`` (organ-major
``brain, chest``; per organ ``(dx, dy, r_units, conf)``) with the fixed formulas of
:mod:`xtx4.utils.organs` (ported from FreeSentry ``server/cameraai.py``). Nothing is trained:

* the constants are non-persistent buffers, so checkpoints and ``load_state_dict(strict=True)`` are
  unaffected, and there is no ``nn.Linear``/``nn.Conv`` (parameter and MAC counts stay unchanged);
* everything runs on the anchor-relative raw ``(dx, dy)`` (stride units): both centres are affine
  combinations and the radii are distances, so ``x = ax + dx * stride``, ``r_px = r_units * stride``
  decode exactly like a keypoint. The graph needs no anchors and the squared distances stay far below
  the FP16 limit (pixel distances of ~384 px would overflow it);
* every linear combination the formulas need (three centres and eight difference vectors) is one
  matmul with a constant ``(11, K)`` matrix; the brain branch is a branch-free gate (clamp), so the
  ncnn graph uses only MatMul / BinaryOp / UnaryOp / Reduction / Clip / Sigmoid / Crop / Concat.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..utils import organs as org
from ..utils.keypoints import NUM_KEYPOINTS
from .head_types import KPT_CHANNELS_PER_POINT

# Rows of the combination matrix.
_ROW_BRAIN_FULL, _ROW_BRAIN_EAR, _ROW_CHEST = 0, 1, 2
_ROWS_BRAIN_FULL_DIST = slice(3, 6)  # brain_full - nose, - left ear, - right ear
_ROW_BRAIN_EAR_DIST = 6  # brain_ear - left ear
_ROWS_CHEST_DIST = slice(7, 11)  # LS-RS, LH-RH, LS-LH, RS-RH
_NUM_ROWS = 11


def combination_matrix() -> np.ndarray:
    """``(11, 9)``: centres (rows 0-2) and the difference vectors whose lengths the radii need."""

    eye = np.eye(NUM_KEYPOINTS)
    m = np.zeros((_NUM_ROWS, NUM_KEYPOINTS), dtype=np.float64)
    m[_ROW_BRAIN_FULL] = org.BRAIN_FULL_WEIGHTS
    m[_ROW_BRAIN_EAR] = org.BRAIN_EAR_WEIGHTS
    m[_ROW_CHEST] = org.CHEST_WEIGHTS
    m[3] = org.BRAIN_FULL_WEIGHTS - eye[org.NOSE]
    m[4] = org.BRAIN_FULL_WEIGHTS - eye[org.LEFT_EAR]
    m[5] = org.BRAIN_FULL_WEIGHTS - eye[org.RIGHT_EAR]
    m[6] = org.BRAIN_EAR_WEIGHTS - eye[org.LEFT_EAR]
    m[7] = eye[org.LEFT_SHOULDER] - eye[org.RIGHT_SHOULDER]
    m[8] = eye[org.LEFT_HIP] - eye[org.RIGHT_HIP]
    m[9] = eye[org.LEFT_SHOULDER] - eye[org.LEFT_HIP]
    m[10] = eye[org.RIGHT_SHOULDER] - eye[org.RIGHT_HIP]
    return m


class OrganLayer(nn.Module):
    """``pose [B, N, K*5]`` raw -> ``organs [B, N, 8]`` (see module docstring)."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS) -> None:
        super().__init__()
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError(f"OrganLayer needs the nine XTX4 landmarks, got num_keypoints={num_keypoints}")
        self.num_keypoints = num_keypoints
        # (K, 11): applied to the x and y rows separately, ``(..., K) @ (K, 11)``
        self.register_buffer(
            "combine_t", torch.tensor(combination_matrix().T.copy(), dtype=torch.float32), persistent=False
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        # Every tensor stays (..., N, C) (3-D in the exported graph): pnnx/ncnn mis-convert reductions over the
        # last axis of 4-D tensors, and the per-axis split keeps the ncnn graph to MatMul/BinaryOp/Crop/
        # Reduction/Concat.
        p = pose.reshape(*pose.shape[:-1], self.num_keypoints, KPT_CHANNELS_PER_POINT)  # any leading dims
        dx, dy = p[..., 0], p[..., 1]  # (B, N, 9) anchor-relative stride units
        c = torch.sigmoid(p[..., 3])  # (B, N, 9) present probability

        vx = torch.matmul(dx, self.combine_t)  # (B, N, 11)
        vy = torch.matmul(dy, self.combine_t)
        ex, ey = vx[..., 3:_NUM_ROWS], vy[..., 3:_NUM_ROWS]
        lengths = torch.sqrt(ex * ex + ey * ey)  # (B, N, 8)

        # ---- brain: full and ear branch, blended by the branch-free gate
        full_d = lengths[..., 0:3]
        r_full = org.BRAIN_RADIUS_FACTOR * torch.maximum(torch.maximum(full_d[..., 0:1], full_d[..., 1:2]),
                                                         full_d[..., 2:3])
        r_ear = org.BRAIN_RADIUS_FACTOR * lengths[..., 3:4]
        face = c[..., 0:5]
        c_full = (org.BRAIN_CONF_MIN_WEIGHT * torch.amin(face, dim=-1, keepdim=True)
                  + (1.0 - org.BRAIN_CONF_MIN_WEIGHT) * face.mean(dim=-1, keepdim=True))
        c_ear = 0.5 * (face[..., 3:4] + face[..., 4:5])
        g = torch.clamp((face - org.BRAIN_EAR_THRESHOLD) * org.GATE_GAIN + 1.0, 0.0, 1.0)
        m = g[..., 3:4] * g[..., 4:5] * (1.0 - torch.prod(g, dim=-1, keepdim=True))  # (B, N, 1)
        full, ear = slice(_ROW_BRAIN_FULL, _ROW_BRAIN_FULL + 1), slice(_ROW_BRAIN_EAR, _ROW_BRAIN_EAR + 1)
        brain_x = m * vx[..., ear] + (1.0 - m) * vx[..., full]
        brain_y = m * vy[..., ear] + (1.0 - m) * vy[..., full]
        brain_r = m * r_ear + (1.0 - m) * r_full
        brain_c = torch.clamp(m * c_ear + (1.0 - m) * c_full, 0.0, 1.0)

        # ---- chest
        tb = lengths[..., 4:8]  # LS-RS, LH-RH, LS-LH, RS-RH
        chest_r = torch.maximum(org.CHEST_RADIUS_LR * (tb[..., 0:1] + tb[..., 1:2]),
                                org.CHEST_RADIUS_TB * (tb[..., 2:3] + tb[..., 3:4]))
        torso = c[..., 5:9]
        w = [x / org.CHEST_CONF_DIVISOR for x in org.CHEST_CONF_WEIGHTS]
        chest_c = (w[0] * torso[..., 0:1] + w[1] * torso[..., 1:2] + w[2] * torso[..., 2:3]
                   + w[3] * torso[..., 3:4] + w[4] * torch.amin(torso, dim=-1, keepdim=True))
        chest_c = torch.clamp(chest_c, 0.0, 1.0)

        chest = slice(_ROW_CHEST, _ROW_CHEST + 1)
        return torch.cat([brain_x, brain_y, brain_r, brain_c, vx[..., chest], vy[..., chest], chest_r, chest_c],
                         dim=-1)


def decode_organs(organs: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """``organs [B, N, 8]`` raw -> ``(B, N, 2, 4)`` ``(x, y, radius, conf)`` in network pixels."""

    b, n = organs.shape[0], organs.shape[1]
    o = organs.reshape(b, n, org.NUM_ORGANS, org.ORGAN_CHANNELS_PER_ORGAN)
    stride = strides.reshape(1, n, 1)
    x = anchors[:, 0].reshape(1, n, 1) + o[..., 0] * stride
    y = anchors[:, 1].reshape(1, n, 1) + o[..., 1] * stride
    return torch.stack([x, y, o[..., 2] * stride, o[..., 3]], dim=-1)
