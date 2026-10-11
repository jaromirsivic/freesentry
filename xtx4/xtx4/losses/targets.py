"""Batch padding of per-image targets (CONTRACTS.md section 5).

The dataset keeps targets as a list of per-image dicts because images carry a
variable number of persons. The assigner and the losses are fully batched, so
the list is padded once per step into ``(B, M, ...)`` tensors with a boolean
``mask`` marking the real rows. ``M`` is the largest person count in the batch
(at least 1, so a zero-GT batch still yields well-formed tensors).

Missing or unknown ``area`` / ``boxes`` are ``NaN``; the assigner derives the
OKS scale from the labeled extent in that case (section 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..utils.keypoints import NUM_KEYPOINTS


@dataclass(slots=True)
class PaddedTargets:
    """Padded ground truth for one batch (all tensors on the same device).

    kpts:  (B, M, K, 3) float32  x, y, v  (v in {-1, 0, 1, 2}; padded rows are 0)
    area:  (B, M)       float32  OKS scale area in px^2, NaN when unknown
    mask:  (B, M)       bool     True for real persons, False for padding
    boxes: (B, M, 4)    float32  xyxy px or NaN (never used by the assignment)
    """

    kpts: torch.Tensor
    area: torch.Tensor
    mask: torch.Tensor
    boxes: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.kpts.shape[0])

    @property
    def max_persons(self) -> int:
        return int(self.kpts.shape[1])

    @property
    def num_keypoints(self) -> int:
        return int(self.kpts.shape[2])


def _as_float_tensor(value: Any, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(value).to(device=device, dtype=torch.float32)


def pad_targets(
    targets: list[dict[str, Any]],
    device: torch.device | str,
    *,
    num_keypoints: int | None = None,
) -> PaddedTargets:
    """Pad a list of per-image target dicts into batched tensors.

    Each dict provides ``keypoints (M, K, 3)``; ``area (M,)`` and ``boxes (M, 4)``
    are optional (missing -> NaN). ``M`` may be 0 for images without persons.

    The padding is one ``cat`` plus one ``index_put`` per field on the device the
    targets already live on (the trainer moves them per image, asynchronously),
    followed by at most one transfer; per-image slice copies would cost ~3 device
    copies per image and step.
    """

    device = torch.device(device)
    batch = len(targets)
    kpts_list: list[torch.Tensor] = []
    counts: list[int] = []
    k_dim = num_keypoints
    src: torch.device | None = None
    for target in targets:
        kp = torch.as_tensor(target["keypoints"])
        if kp.numel() == 0:
            kp = kp.reshape(0, k_dim if k_dim is not None else NUM_KEYPOINTS, 3)
        elif kp.dim() == 2:
            kp = kp.unsqueeze(0)
        if k_dim is None:
            k_dim = int(kp.shape[1])
        elif kp.numel() > 0 and int(kp.shape[1]) != k_dim:
            raise ValueError(f"inconsistent keypoint count: {kp.shape[1]} vs {k_dim}")
        if kp.numel() > 0:
            if src is None:
                src = kp.device
            elif kp.device != src:
                src = device  # mixed devices: build directly on the target device
        kpts_list.append(kp)
        counts.append(int(kp.shape[0]))
    if k_dim is None:
        k_dim = NUM_KEYPOINTS
    max_persons = max(1, max(counts) if counts else 0)
    total = sum(counts)
    build = src if src is not None else device

    kpts = torch.zeros((batch, max_persons, k_dim, 3), dtype=torch.float32, device=build)
    area = torch.full((batch, max_persons), float("nan"), dtype=torch.float32, device=build)
    boxes = torch.full((batch, max_persons, 4), float("nan"), dtype=torch.float32, device=build)
    mask = torch.zeros((batch, max_persons), dtype=torch.bool, device=build)

    if total > 0:
        b_rows: list[int] = []
        s_rows: list[int] = []
        area_list: list[torch.Tensor] = []
        area_rows: list[int] = []
        box_list: list[torch.Tensor] = []
        box_rows: list[int] = []
        row = 0
        for b, (target, m) in enumerate(zip(targets, counts)):
            if m == 0:
                continue
            b_rows.extend([b] * m)
            s_rows.extend(range(m))
            raw_area = target.get("area")
            if raw_area is not None:
                a = _as_float_tensor(raw_area, build).reshape(-1)
                if a.numel() == m:
                    area_list.append(a)
                    area_rows.extend(range(row, row + m))
            raw_boxes = target.get("boxes")
            if raw_boxes is not None:
                bx = _as_float_tensor(raw_boxes, build)
                if bx.numel() == 4 * m:
                    box_list.append(bx.reshape(m, 4))
                    box_rows.extend(range(row, row + m))
            row += m
        # one pinned, asynchronous transfer for every index row (no host sync)
        rows = [b_rows, s_rows]
        if area_list and len(area_rows) != total:
            rows.append(area_rows + [0] * (total - len(area_rows)))
        if box_list and len(box_rows) != total:
            rows.append(box_rows + [0] * (total - len(box_rows)))
        index = torch.tensor(rows, dtype=torch.int64)
        if build.type == "cuda":
            index = index.pin_memory().to(build, non_blocking=True)
        else:
            index = index.to(build)
        b_idx, s_idx = index[0], index[1]
        next_row = 2
        all_kpts = torch.cat([kp.reshape(-1, k_dim, 3) for kp in kpts_list if kp.numel() > 0], dim=0)
        kpts.index_put_((b_idx, s_idx), all_kpts.to(device=build, dtype=torch.float32))
        mask.index_put_((b_idx, s_idx), torch.ones((), dtype=torch.bool, device=build))
        if area_list:
            if len(area_rows) == total:
                area.index_put_((b_idx, s_idx), torch.cat(area_list, dim=0))
            else:
                sub = index[next_row, : len(area_rows)]
                next_row += 1
                area.index_put_((b_idx[sub], s_idx[sub]), torch.cat(area_list, dim=0))
        if box_list:
            if len(box_rows) == total:
                boxes.index_put_((b_idx, s_idx), torch.cat(box_list, dim=0))
            else:
                sub = index[next_row, : len(box_rows)]
                boxes.index_put_((b_idx[sub], s_idx[sub]), torch.cat(box_list, dim=0))

    if build != device:
        if device.type == "cuda" and build.type == "cpu":
            kpts, area, boxes, mask = (t.pin_memory().to(device, non_blocking=True) for t in (kpts, area, boxes, mask))
        else:
            kpts, area, boxes, mask = (t.to(device) for t in (kpts, area, boxes, mask))
    return PaddedTargets(kpts=kpts, area=area, mask=mask, boxes=boxes)
