"""Pose-TAL: task-aligned label assignment driven by OKS instead of box IoU
(CONTRACTS.md section 6, implementation_plan_fable.md section 6.2).

XTX4 has no box branch, so the assignment geometry is derived from the labeled
keypoints only:

* **extent box** ``E`` -- envelope of the labeled points, expanded by
  ``extent_expand`` per side, each side at least ``extent_min_px`` (two cells at
  stride 8, the STAL guarantee); a single labeled point gets a
  ``single_point_box_px`` square;
* **scale** ``s^2`` -- the COCO ``area`` when the dataset has one, otherwise the
  area of the expanded envelope ``(1.5 w_E)(1.5 h_E)``, floored at ``extent_min_px^2``;
* **root** ``r`` -- shoulder centre, else torso centre, else face centre; used by
  the candidate guarantee and by the warm-up prior.

Alignment ``align = p^alpha * OKS^beta`` restricted to anchor centres inside
``E`` (plus the nearest ``min_candidates`` anchors to ``r`` when ``E`` holds fewer),
plus a root prior ``root_prior_scale * root_prior_weight * exp(-|a-r|^2 / (2 (0.5 s)^2))``
that the trainer fades out over the first epochs (OKS is ~0 for every anchor of an
untrained network, so without the prior the first positives would be random).

Performance layout (one batch, both heads):

* :meth:`PoseTAL.gt_geometry` and :meth:`PoseTAL.anchor_geometry` depend on the
  targets only, so the loss computes them once and shares them between the
  one-to-many and the one-to-one assigner;
* OKS and the alignment are only ever consumed on *candidate* anchors (inside
  ``E`` or guaranteed), about 4 % of ``(B, M, N)``, so they are computed on the
  candidate pairs (:func:`candidate_oks`) and scattered into the dense
  ``(B, M, N)`` selection tensor -- the ``(B, M, N, K)`` intermediate of the dense
  form never exists;
* the selection itself (top-k, conflict resolution, normalisation) stays dense and
  batched; the only host syncs are the two ``nonzero`` calls of
  :meth:`anchor_geometry` (shared by the heads) and, for the stand-alone
  :meth:`assign`, the stats transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..utils.keypoints import FACE_INDICES, OKS_SIGMAS, TORSO_INDICES
from .oks import candidate_oks

SHOULDER_INDICES: tuple[int, ...] = (5, 6)
STAT_KEYS: tuple[str, ...] = ("gt_total", "gt_covered", "pos_total", "cand_total", "small_gt")


@dataclass(slots=True)
class AssignResult:
    """Batched assignment (CONTRACTS.md section 6).

    fg_mask:      (B, N) bool     positive anchors
    gt_idx:       (B, N) int64    assigned GT index (0 where background)
    target_score: (B, N) float32  soft person target in [0, 1] (0 for background)
    assigned_oks: (B, N) float32  OKS of the assigned pair (0 for background)
    stats:        dict            'gt_total', 'gt_covered', 'pos_per_gt', 'cand_per_gt', 'small_gt'
                                  (empty when the caller asked for ``stats_tensor`` instead)
    stats_tensor: (5,) int64      raw counts ``STAT_KEYS`` on the device (no host sync)
    """

    fg_mask: torch.Tensor
    gt_idx: torch.Tensor
    target_score: torch.Tensor
    assigned_oks: torch.Tensor
    stats: dict[str, float]
    stats_tensor: torch.Tensor | None = None


@dataclass(slots=True)
class GTGeometry:
    """Box-free target geometry derived from the labeled keypoints.

    labeled: (B, M, K) bool   v > 0 and a real GT row
    extent:  (B, M, 4)        expanded extent box x0, y0, x1, y1 (px)
    raw_wh:  (B, M, 2)        width/height of the raw labeled envelope (px)
    scale2:  (B, M)           OKS scale s^2 (px^2)
    root:    (B, M, 2)        root point r (px)
    small:   (B, M) bool      max(w_E, h_E) < extent_min_px
    valid:   (B, M) bool      real GT row with at least one labeled point
    """

    labeled: torch.Tensor
    extent: torch.Tensor
    raw_wh: torch.Tensor
    scale2: torch.Tensor
    root: torch.Tensor
    small: torch.Tensor
    valid: torch.Tensor


@dataclass(slots=True)
class AnchorGeometry:
    """Anchor-side candidate structure of one batch (shared by every head).

    in_extent:     (B, M, N) bool  anchor centre inside the GT's extent box
    cand:          (B, M, N) bool  candidates: in_extent | guaranteed nearest anchors
    cand_idx:      (b, m, n) int64 index tensors of the candidate pairs (T,)
    in_any_extent: (B, N) bool     anchor inside the extent box of at least one GT
    d2_root_cand:  (T,)            squared distance anchor -> root for the candidate pairs
    """

    in_extent: torch.Tensor
    cand: torch.Tensor
    cand_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    in_any_extent: torch.Tensor
    d2_root_cand: torch.Tensor
    num_candidates: int = field(default=0)
    gt_total: torch.Tensor | None = None  # 0-dim int64: valid GTs (shared by the heads)
    small_total: torch.Tensor | None = None  # 0-dim int64: valid small GTs


class PoseTAL:
    """OKS-driven task-aligned assigner for box-free pose (one-to-many or one-to-one)."""

    def __init__(
        self,
        sigmas=OKS_SIGMAS,
        *,
        alpha: float = 0.5,
        beta: float = 6.0,
        topk: int = 10,
        topk2: int | None = None,
        extent_expand: float = 0.25,
        extent_min_px: float = 16.0,
        single_point_box_px: float = 24.0,
        min_candidates: int = 4,
        root_prior_weight: float = 0.1,
        eps: float = 1e-9,
        small_target_floor: float = 0.05,
        oks_chunk_bytes: int | None = None,
        shoulder_indices: tuple[int, ...] = SHOULDER_INDICES,
        torso_indices: tuple[int, ...] = TORSO_INDICES,
        face_indices: tuple[int, ...] = FACE_INDICES,
    ) -> None:
        self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32).reshape(-1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.topk = int(topk)
        self.topk2 = int(topk2) if topk2 is not None else None
        self.extent_expand = float(extent_expand)
        self.extent_min_px = float(extent_min_px)
        self.single_point_box_px = float(single_point_box_px)
        self.min_candidates = int(min_candidates)
        self.root_prior_weight = float(root_prior_weight)
        self.eps = float(eps)
        self.small_target_floor = float(small_target_floor)
        self.oks_chunk_bytes = oks_chunk_bytes  # kept for API compatibility; the sparse path needs no chunking
        self.shoulder_indices = tuple(shoulder_indices)
        self.torso_indices = tuple(torso_indices)
        self.face_indices = tuple(face_indices)
        self._sigma_cache: dict[torch.device, torch.Tensor] = {}
        self._group_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    # ------------------------------------------------------------------ geometry
    def _sigmas_on(self, device: torch.device) -> torch.Tensor:
        cached = self._sigma_cache.get(device)
        if cached is None:
            cached = self.sigmas.to(device)
            self._sigma_cache[device] = cached
        return cached

    def _group_masks(self, num_kpts: int, device: torch.device) -> torch.Tensor:
        """``(G, K)`` bool: row 0 = every landmark, then the non-empty root groups in
        fallback order (face, torso, shoulders)."""

        key = (num_kpts, device)
        cached = self._group_cache.get(key)
        if cached is None:
            rows = [torch.ones((num_kpts,), dtype=torch.bool)]
            for group in (self.face_indices, self.torso_indices, self.shoulder_indices):
                valid = [i for i in group if i < num_kpts]
                if not valid:
                    continue
                mask = torch.zeros((num_kpts,), dtype=torch.bool)
                mask[valid] = True
                rows.append(mask)
            cached = torch.stack(rows).to(device)
            self._group_cache[key] = cached
        return cached

    def gt_geometry(self, gt_kpts: torch.Tensor, gt_area: torch.Tensor, gt_mask: torch.Tensor) -> GTGeometry:
        """Extent box, OKS scale, root point and small flag per GT (all batched, FP32)."""

        gt_kpts = gt_kpts.float()
        gt_area = gt_area.float()
        gt_mask = gt_mask.bool()
        xy = gt_kpts[..., :2]
        v = gt_kpts[..., 2]
        labeled = (v > 0.5) & gt_mask.unsqueeze(-1)
        lab4 = labeled.unsqueeze(-1)  # (B,M,K,1)
        n_lab = labeled.sum(-1)  # (B,M)
        has = n_lab > 0
        has2 = has.unsqueeze(-1)

        # x/y handled together as (B,M,2): same per-element arithmetic as separate x and y
        big = 1e12
        lo = torch.where(has2, torch.where(lab4, xy, big).amin(2), 0.0)  # (B,M,2) = x0, y0
        hi = torch.where(has2, torch.where(lab4, xy, -big).amax(2), 0.0)  # x1, y1
        wh = (hi - lo).clamp(min=0.0)
        centre = 0.5 * (lo + hi)

        ext_wh = (wh * (1.0 + 2.0 * self.extent_expand)).clamp(min=self.extent_min_px)
        ext_wh = torch.where((n_lab == 1).unsqueeze(-1), self.single_point_box_px, ext_wh)
        half = 0.5 * ext_wh
        extent = torch.cat([centre - half, centre + half], dim=-1)  # x0, y0, x1, y1

        fallback = (1.5 * wh[..., 0]) * (1.5 * wh[..., 1])
        area_ok = (gt_area > 0) & (gt_area < float("inf"))  # == isfinite & > 0 (NaN fails both)
        scale2 = torch.where(area_ok, gt_area, fallback).clamp(min=self.extent_min_px**2)

        small = wh.amax(-1) < self.extent_min_px

        # root: shoulders -> torso -> face -> any labeled point; all group means in one pass
        groups = self._group_masks(gt_kpts.shape[2], gt_kpts.device)  # (G,K), row 0 = all
        m = (labeled.unsqueeze(2) & groups.reshape(1, 1, *groups.shape)).to(xy.dtype)  # (B,M,G,K)
        count = m.sum(-1)  # (B,M,G)
        mean = (xy.unsqueeze(2) * m.unsqueeze(-1)).sum(3) / count.clamp(min=1.0).unsqueeze(-1)  # (B,M,G,2)
        present = (count > 0).unsqueeze(-1)
        root = mean[:, :, 0]
        for g in range(1, groups.shape[0]):
            root = torch.where(present[:, :, g], mean[:, :, g], root)
        return GTGeometry(
            labeled=labeled,
            extent=extent,
            raw_wh=wh,
            scale2=scale2,
            root=root,
            small=small,
            valid=gt_mask & has,
        )

    @torch.no_grad()
    def anchor_geometry(self, anchors: torch.Tensor, geo: GTGeometry) -> AnchorGeometry:
        """Candidate anchors per GT: inside the extent box, plus the nearest ``min_candidates``
        anchors to the root for GTs whose box holds fewer (two host syncs)."""

        anchors = anchors.float()
        num_anchors = anchors.shape[0]
        valid_gt = geo.valid
        ax = anchors[:, 0].reshape(1, 1, num_anchors)
        ay = anchors[:, 1].reshape(1, 1, num_anchors)
        ext = geo.extent.unsqueeze(-1)  # (B,M,4,1)
        in_extent = (ax >= ext[:, :, 0]) & (ax <= ext[:, :, 2]) & (ay >= ext[:, :, 1]) & (ay <= ext[:, :, 3])
        in_extent = in_extent & valid_gt.unsqueeze(-1)  # (B,M,N)

        k_guard = min(self.min_candidates, num_anchors)
        cand = in_extent
        if k_guard > 0:
            need = (in_extent.sum(-1) < self.min_candidates) & valid_gt  # (B,M)
            nb, nm = need.nonzero(as_tuple=True)
            if nb.numel() > 0:
                root = geo.root[nb, nm]  # (R,2)
                d2 = (anchors[:, 0].reshape(1, -1) - root[:, 0:1]) ** 2 + (
                    anchors[:, 1].reshape(1, -1) - root[:, 1:2]
                ) ** 2  # (R,N)
                nearest = d2.topk(k_guard, dim=-1, largest=False).indices  # (R,k)
                true = torch.ones((), dtype=torch.bool, device=anchors.device)  # device scalar: no host sync
                cand = in_extent.clone().index_put_(
                    (nb.unsqueeze(-1).expand_as(nearest), nm.unsqueeze(-1).expand_as(nearest), nearest), true
                )
        cb, cm, cn = cand.nonzero(as_tuple=True)
        root_c = geo.root[cb, cm]
        a_c = anchors[cn]
        d2_root_cand = (a_c[:, 0] - root_c[:, 0]) ** 2 + (a_c[:, 1] - root_c[:, 1]) ** 2
        return AnchorGeometry(
            in_extent=in_extent,
            cand=cand,
            cand_idx=(cb, cm, cn),
            in_any_extent=in_extent.any(dim=1),
            d2_root_cand=d2_root_cand,
            num_candidates=int(cb.numel()),
            gt_total=valid_gt.sum(),
            small_total=(geo.small & valid_gt).sum(),
        )

    # ---------------------------------------------------------------- assignment
    @torch.no_grad()
    def assign(
        self,
        person_prob: torch.Tensor,
        pred_xy: torch.Tensor,
        anchors: torch.Tensor,
        strides: torch.Tensor,
        gt_kpts: torch.Tensor,
        gt_area: torch.Tensor,
        gt_mask: torch.Tensor,
        root_prior_scale: float,
        *,
        geometry: GTGeometry | None = None,
        anchor_geometry: AnchorGeometry | None = None,
        sync_stats: bool = True,
    ) -> AssignResult:
        """Assign anchors to GTs for one batch (see module docstring).

        ``geometry`` / ``anchor_geometry`` let the caller share the target-side work
        between heads; ``sync_stats=False`` leaves the counts on the device
        (``AssignResult.stats_tensor``) and returns an empty ``stats`` dict.
        """

        person_prob = person_prob.float()
        pred_xy = pred_xy.float()
        anchors = anchors.float()
        gt_kpts = gt_kpts.float()
        gt_area = gt_area.float()
        gt_mask = gt_mask.bool()
        geo = geometry if geometry is not None else self.gt_geometry(gt_kpts, gt_area, gt_mask)
        ag = anchor_geometry if anchor_geometry is not None else self.anchor_geometry(anchors, geo)
        cb, cm, cn = ag.cand_idx
        oks_c = candidate_oks(
            pred_xy[cb, cn], gt_kpts[cb, cm, :, :2], geo.labeled[cb, cm], geo.scale2[cb, cm],
            self._sigmas_on(person_prob.device), eps=self.eps,
        )
        return self.assign_from_candidates(person_prob, oks_c, geo, ag, root_prior_scale, sync_stats=sync_stats)

    @torch.no_grad()
    def assign_from_candidates(
        self,
        person_prob: torch.Tensor,
        oks_cand: torch.Tensor,
        geo: GTGeometry,
        ag: AnchorGeometry,
        root_prior_scale: float,
        *,
        sync_stats: bool = True,
    ) -> AssignResult:
        """Selection given the OKS of the candidate pairs ``oks_cand (T,)`` (no host sync unless asked)."""

        person_prob = person_prob.float()
        device = person_prob.device
        batch, num_anchors = person_prob.shape
        num_gt = geo.valid.shape[1]
        valid_gt = geo.valid
        cb, cm, cn = ag.cand_idx

        # alignment on the candidate pairs only (identical arithmetic to the dense form)
        oks_c = oks_cand.float()
        align_c = person_prob[cb, cn].clamp(min=self.eps).pow(self.alpha) * oks_c.clamp(min=0.0).pow(self.beta)
        prior_scale = float(root_prior_scale) * self.root_prior_weight
        if prior_scale > 0.0:
            # exp(-|a-r|^2 / (2 (0.5 s)^2)) = exp(-|a-r|^2 / (0.5 s^2))
            align_c = align_c + prior_scale * torch.exp(-ag.d2_root_cand / (0.5 * geo.scale2[cb, cm]))

        # dense selection tensor: align on candidates, -1 elsewhere
        shape = (batch, num_gt, num_anchors)
        sel_score = torch.full(shape, -1.0, device=device, dtype=torch.float32).index_put_((cb, cm, cn), align_c)
        oks = torch.zeros(shape, device=device, dtype=torch.float32).index_put_((cb, cm, cn), oks_c)

        # top-k (then top-k2) among candidates by align
        k1 = min(self.topk, num_anchors)
        vals, idx = sel_score.topk(k1, dim=-1)
        sel = torch.zeros(shape, dtype=torch.bool, device=device).scatter_(-1, idx, vals >= 0.0)
        if self.topk2 is not None and self.topk2 < k1:
            k2 = max(1, self.topk2)
            vals2, idx2 = torch.where(sel, sel_score, -1.0).topk(k2, dim=-1)
            sel = torch.zeros(shape, dtype=torch.bool, device=device).scatter_(-1, idx2, vals2 >= 0.0)

        # conflicts: an anchor claimed by several GTs goes to the one with the larger align
        best_gt = torch.where(sel, sel_score, -1.0).argmax(dim=1)  # (B,N)
        multi = sel.sum(dim=1) > 1  # (B,N)
        gt_ids = torch.arange(num_gt, device=device).reshape(1, num_gt, 1)
        one_hot = (gt_ids == best_gt.unsqueeze(1)) & sel
        sel = torch.where(multi.unsqueeze(1), one_hot, sel)

        fg_mask = sel.any(dim=1)
        gt_idx = torch.where(fg_mask, best_gt, 0)

        # YOLOv8-style normalisation: align / max_align_g * max_OKS_g
        align_sel = torch.where(sel, sel_score, 0.0)
        oks_sel = torch.where(sel, oks, 0.0)
        max_align = align_sel.amax(dim=-1, keepdim=True).clamp(min=self.eps)
        max_oks = oks_sel.amax(dim=-1, keepdim=True)
        norm = align_sel / max_align * max_oks
        # floor for small GTs: their OKS is extremely sensitive (s^2 floored at extent_min_px^2),
        # so without a floor the person head would get ~0 targets for tiny persons for a long time.
        # Applied to every positive of a small GT (a superset of the guaranteed candidates).
        norm = torch.maximum(norm, torch.where(sel & geo.small.unsqueeze(-1), self.small_target_floor, 0.0))
        target_score = torch.where(fg_mask, norm.amax(dim=1).clamp(0.0, 1.0), 0.0)
        assigned_oks = torch.where(fg_mask, oks_sel.amax(dim=1), 0.0)

        # stats after conflict resolution (packed int64 counts; one host sync only when asked).
        # After conflict resolution every positive anchor belongs to exactly one GT, so the
        # number of (GT, anchor) pairs equals the number of positive anchors.
        gt_total = ag.gt_total if ag.gt_total is not None else valid_gt.sum()
        small_total = ag.small_total if ag.small_total is not None else (geo.small & valid_gt).sum()
        covered = (sel.any(dim=-1) & valid_gt).sum()
        pos_total = fg_mask.sum()
        cand_total = torch.full((), ag.num_candidates, device=device, dtype=torch.int64)
        packed = torch.stack([gt_total, covered, pos_total, cand_total, small_total])
        stats: dict[str, float] = {}
        if sync_stats:
            stats = self.stats_from_tensor(packed.tolist())
        return AssignResult(
            fg_mask=fg_mask,
            gt_idx=gt_idx,
            target_score=target_score,
            assigned_oks=assigned_oks,
            stats=stats,
            stats_tensor=packed,
        )

    @staticmethod
    def stats_from_tensor(values: list[float]) -> dict[str, float]:
        """Turn the packed ``STAT_KEYS`` counts into the public stats dict."""

        values = [float(x) for x in values]
        n_gt = max(values[0], 1.0)
        return {
            "gt_total": values[0],
            "gt_covered": values[1],
            "pos_per_gt": values[2] / n_gt,
            "cand_per_gt": values[3] / n_gt,
            "small_gt": values[4],
        }
