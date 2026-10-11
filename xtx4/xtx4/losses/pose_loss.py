"""Total XTX4 loss (CONTRACTS.md section 7): Pose-TAL assignment per head,
person BCE/VFL on the soft target score, Smooth-OKS + RLE keypoint localisation,
present/visible BCE with the four-state visibility semantics, ProgLoss across
the dual head.

Design points (implementation_plan_fable.md section 6.3):

* everything after the raw head tensors runs in FP32 with autocast disabled --
  the OKS exponentials, ``pow(beta=6)`` and the RealNVP flow are not FP16-safe;
* the keypoint, present and visible terms are normalised **per positive anchor
  first** (mean over its labeled / valid points) and then averaged over the
  positives weighted by ``target_score``; XTX3 averaged over all points globally,
  which favoured persons with many positives;
* the person term is normalised by ``sum(target_score).clamp(1)`` (YOLOv8);
* ``v = -1`` (landmark not annotated by the dataset) contributes to nothing;
  ``v = 0`` trains ``present -> 0`` with ``unlabeled_present_weight``;
* the warm-up root prior of the assigner decays linearly over
  ``assign.root_warmup_epochs`` epochs, driven by :meth:`XTX4Loss.set_progress`
  and the total epoch count (``training.epochs`` or :meth:`set_schedule`);
* optional one-to-one anti-duplicate term (``loss.o2o_negative_boost`` > 1):
  candidate anchors inside any GT extent box that are *not* the selected positive
  and already fire (``person_prob >= loss.o2o_negative_min_prob``) get their person
  BCE/VFL weight multiplied by the boost (target stays 0); ``loss.o2o_single_stage``
  makes the o2o assigner pick ``topk=1`` directly instead of ``7 -> 1``.

Per batch the target-side work (GT geometry, candidate anchors) is computed once
and shared by both heads, the candidate OKS exponent runs once for both heads,
and the per-point terms of both heads are evaluated in one concatenated pass
(per-head reductions are contiguous slice sums). Host syncs per step: the two
``nonzero`` of the candidate search, the ``nonzero`` of the positives and one
packed stats transfer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.head_types import HeadOutput, RawPredictions, sigma_units_from_raw
from ..utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS
from ..utils.logging import get_logger
from .assign import AssignResult, PoseTAL
from .oks import candidate_oks, oks_loss_terms
from .rle import RLELoss
from .targets import PaddedTargets, pad_targets

log = get_logger(__name__)

TERM_KEYS: tuple[str, ...] = ("person", "oks", "rle", "present", "visible")
HEAD_DIAG_KEYS: tuple[str, ...] = (
    "coverage", "pos_per_gt", "cand_per_gt", "num_pos", "mean_sigma_units", "mean_target_score", "boosted"
)


class ProgLossSchedule:
    """Linear ProgLoss weights: ``w_o2m`` from ``o2m_start`` to ``o2m_end``, ``w_o2o = 1 - w_o2m``."""

    def __init__(self, *, o2m_start: float = 0.8, o2m_end: float = 0.1) -> None:
        self.o2m_start = float(o2m_start)
        self.o2m_end = float(o2m_end)

    def weights(self, t: float) -> tuple[float, float]:
        t = min(max(float(t), 0.0), 1.0)
        w_o2m = self.o2m_start + (self.o2m_end - self.o2m_start) * t
        return w_o2m, 1.0 - w_o2m

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProgLossSchedule":
        data = data or {}
        return cls(o2m_start=float(data.get("o2m_start", 0.8)), o2m_end=float(data.get("o2m_end", 0.1)))


def _sliced_weighted_means(
    values: torch.Tensor, weights: torch.Tensor, slices: list[slice], *, eps: float = 1e-9
) -> list[torch.Tensor]:
    """``sum(values * weights) / sum(weights)`` of several terms at once over contiguous
    slices (one per head; 0 for an empty slice).

    ``values`` and ``weights`` are ``(R, P)`` (one row per term); returns one ``(R,)`` tensor per slice.
    """

    vw = values * weights
    return [vw[:, sl].sum(-1) / weights[:, sl].sum(-1).clamp(min=eps) for sl in slices]


class XTX4Loss(nn.Module):
    """Weighted multi-task loss for the XTX4 dual pose head with ProgLoss balancing."""

    def __init__(
        self,
        *,
        num_keypoints: int = NUM_KEYPOINTS,
        sigmas=OKS_SIGMAS,
        person_weight: float = 1.0,
        kpt_weight: float = 15.0,
        present_weight: float = 1.0,
        visible_weight: float = 0.5,
        oks_weight: float = 1.0,
        rle_weight: float = 0.0417,
        kpt_loss: str = "soks",
        rle_convention: str = "reference",
        person_loss: str = "bce",
        unlabeled_present_weight: float = 1.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        assign: dict[str, Any] | None = None,
        progloss: ProgLossSchedule | None = None,
        epochs: int = 0,
        sigma_floor: float = 0.05,
        rle_layers: int = 4,
        rle_hidden: int = 64,
        o2o_negative_boost: float = 1.0,
        o2o_negative_min_prob: float = 0.1,
        o2o_single_stage: bool = False,
    ) -> None:
        super().__init__()
        if kpt_loss not in ("soks", "oks"):
            raise ValueError(f"loss.kpt_loss must be 'soks' or 'oks', got {kpt_loss!r}")
        if person_loss not in ("bce", "vfl"):
            raise ValueError(f"loss.person_loss must be 'bce' or 'vfl', got {person_loss!r}")
        self.num_keypoints = int(num_keypoints)
        self.register_buffer("sigmas", torch.as_tensor(sigmas, dtype=torch.float32).reshape(-1), persistent=False)
        if self.sigmas.numel() != self.num_keypoints:
            raise ValueError(f"{self.sigmas.numel()} sigmas for {self.num_keypoints} keypoints")
        self.person_weight = float(person_weight)
        self.kpt_weight = float(kpt_weight)
        self.present_weight = float(present_weight)
        self.visible_weight = float(visible_weight)
        self.oks_weight = float(oks_weight)
        self.rle_weight = float(rle_weight)
        self.kpt_loss = kpt_loss
        self.person_loss = person_loss
        self.unlabeled_present_weight = float(unlabeled_present_weight)
        self.vfl_alpha = float(vfl_alpha)
        self.vfl_gamma = float(vfl_gamma)
        self.sigma_floor = float(sigma_floor)
        if o2o_negative_boost < 0.0:
            raise ValueError(f"loss.o2o_negative_boost must be >= 0, got {o2o_negative_boost!r}")
        self.o2o_negative_boost = float(o2o_negative_boost)
        self.o2o_negative_min_prob = float(o2o_negative_min_prob)
        self.o2o_single_stage = bool(o2o_single_stage)
        self.progloss = progloss or ProgLossSchedule()
        self.epochs = int(epochs)
        self._progress = 0.0

        assign = dict(assign or {})
        self.root_warmup_epochs = float(assign.get("root_warmup_epochs", 3))
        common = dict(
            alpha=float(assign.get("alpha", 0.5)),
            beta=float(assign.get("beta", 6.0)),
            extent_expand=float(assign.get("extent_expand", 0.25)),
            extent_min_px=float(assign.get("extent_min_px", 16.0)),
            single_point_box_px=float(assign.get("single_point_box_px", 24.0)),
            min_candidates=int(assign.get("min_candidates", 4)),
            root_prior_weight=float(assign.get("root_prior_weight", 0.1)),
            eps=float(assign.get("eps", 1e-9)),
        )
        self.assigner_o2m = PoseTAL(self.sigmas, topk=int(assign.get("topk_o2m", 10)), topk2=None, **common)
        if self.o2o_single_stage:
            self.assigner_o2o = PoseTAL(self.sigmas, topk=1, topk2=None, **common)
        else:
            self.assigner_o2o = PoseTAL(
                self.sigmas,
                topk=int(assign.get("topk_o2o", 7)),
                topk2=int(assign.get("topk2_o2o", 1)),
                **common,
            )
        self.rle = RLELoss(convention=rle_convention, num_layers=rle_layers, hidden=rle_hidden)

    # ----------------------------------------------------------------- config
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "XTX4Loss":
        loss_cfg = dict(config.get("loss", {}))
        model_cfg = dict(config.get("model", {}))
        train_cfg = dict(config.get("training", {}))
        decode_cfg = dict(config.get("decode", {}))
        return cls(
            num_keypoints=int(model_cfg.get("num_keypoints", NUM_KEYPOINTS)),
            person_weight=float(loss_cfg.get("person", 1.0)),
            kpt_weight=float(loss_cfg.get("kpt", 15.0)),
            present_weight=float(loss_cfg.get("present", 1.0)),
            visible_weight=float(loss_cfg.get("visible", 0.5)),
            oks_weight=float(loss_cfg.get("oks_weight", 1.0)),
            rle_weight=float(loss_cfg.get("rle_weight", 0.0417)),
            kpt_loss=str(loss_cfg.get("kpt_loss", "soks")),
            rle_convention=str(loss_cfg.get("rle_convention", "reference")),
            person_loss=str(loss_cfg.get("person_loss", "bce")),
            unlabeled_present_weight=float(loss_cfg.get("unlabeled_present_weight", 1.0)),
            vfl_alpha=float(loss_cfg.get("vfl_alpha", 0.75)),
            vfl_gamma=float(loss_cfg.get("vfl_gamma", 2.0)),
            assign=loss_cfg.get("assign"),
            progloss=ProgLossSchedule.from_dict(loss_cfg.get("progloss")),
            epochs=int(train_cfg.get("epochs", 0)),
            sigma_floor=float(decode_cfg.get("sigma_floor", 0.05)),
            o2o_negative_boost=float(loss_cfg.get("o2o_negative_boost", 1.0)),
            o2o_negative_min_prob=float(loss_cfg.get("o2o_negative_min_prob", 0.1)),
            o2o_single_stage=bool(loss_cfg.get("o2o_single_stage", False)),
        )

    # --------------------------------------------------------------- schedule
    def set_progress(self, t: float) -> None:
        """Training progress ``t = epoch / epochs`` in [0, 1] (drives ProgLoss and the root prior)."""

        self._progress = min(max(float(t), 0.0), 1.0)

    def set_schedule(self, epochs: int) -> None:
        """Total epoch count, needed to turn progress into epochs for the root-prior warm-up."""

        self.epochs = int(epochs)

    @property
    def progress(self) -> float:
        return self._progress

    def current_weights(self) -> tuple[float, float]:
        """``(w_o2m, w_o2o)`` at the current progress."""

        return self.progloss.weights(self._progress)

    def root_prior_scale(self) -> float:
        """Linear 1 -> 0 over ``root_warmup_epochs`` epochs (0 once the schedule is unknown or over)."""

        if self.root_warmup_epochs <= 0.0 or self.epochs <= 0:
            return 0.0
        epoch = self._progress * self.epochs
        return float(min(max(1.0 - epoch / self.root_warmup_epochs, 0.0), 1.0))

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        output: HeadOutput,
        targets: list[dict[str, torch.Tensor]] | PaddedTargets,
        image_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        terms = self.compute_terms(output, targets, image_size)
        out: dict[str, torch.Tensor] = {"total": terms["total"]}
        for key, value in terms.items():
            if key == "total":
                continue
            out[key] = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
        return out

    def compute_terms(
        self,
        output: HeadOutput,
        targets: list[dict[str, torch.Tensor]] | PaddedTargets,
        image_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Like :meth:`forward` but the per-term tensors keep their graph (used by calibration).

        Returned per-term values (``person``, ``oks``, ``rle``, ``present``, ``visible``) are the
        ProgLoss-weighted sums over the heads *without* the ``lambda`` weights, so
        ``total = lambda_person * person + lambda_kpt * (oks_w * oks + rle_w * rle) + ...``.
        """

        raw_o2o = output.one2one
        device = raw_o2o.person.device
        w_o2m, w_o2o = self.current_weights()
        if output.one2many is None:
            w_o2m = 0.0
        prior = self.root_prior_scale()

        with torch.autocast(device_type=device.type, enabled=False):
            padded = targets if isinstance(targets, PaddedTargets) else pad_targets(
                targets, device, num_keypoints=self.num_keypoints
            )
            names = ["o2o"]
            raws = [raw_o2o]
            assigners = [self.assigner_o2o]
            if output.one2many is not None:
                names.append("o2m")
                raws.append(output.one2many)
                assigners.append(self.assigner_o2m)
            heads = self._heads_loss(names, raws, assigners, padded, prior)

            weights = {"o2o": w_o2o, "o2m": w_o2m}
            # ProgLoss mix of all five terms at once: (H,5) -> (5,); 0 + a + b == a + b exactly
            term_mat = torch.stack([head[key] for head in heads.values() for key in TERM_KEYS]).reshape(
                len(heads), len(TERM_KEYS)
            )
            mixed: torch.Tensor | None = None
            for h, name in enumerate(heads):
                part = weights[name] * term_mat[h]
                mixed = part if mixed is None else mixed + part
            agg: dict[str, torch.Tensor] = {key: mixed[i] for i, key in enumerate(TERM_KEYS)}
            kpt = self.oks_weight * agg["oks"] + self.rle_weight * agg["rle"]
            total = (
                self.person_weight * agg["person"]
                + self.kpt_weight * kpt
                + self.present_weight * agg["present"]
                + self.visible_weight * agg["visible"]
            )

        out: dict[str, torch.Tensor] = {"total": total, "kpt": kpt}
        out.update(agg)
        out["w_o2m"] = torch.tensor(w_o2m)
        out["w_o2o"] = torch.tensor(w_o2o)
        out["root_prior_scale"] = torch.tensor(prior)
        ref = heads["o2m"] if "o2m" in heads else heads["o2o"]
        out["coverage"] = torch.tensor(float(ref["coverage"]))
        out["pos_per_gt"] = torch.tensor(float(ref["pos_per_gt"]))
        for name, head in heads.items():
            for key in TERM_KEYS:
                out[f"{name}_{key}"] = head[key]
            for key in HEAD_DIAG_KEYS:
                out[f"{name}_{key}"] = torch.tensor(float(head[key]))
        out["gt_total"] = torch.tensor(float(ref["gt_total"]))
        out["small_gt"] = torch.tensor(float(ref["small_gt"]))
        return out

    # ---------------------------------------------------------------- per head
    def _person_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        *,
        prob: torch.Tensor | None = None,
        dims: tuple[int, ...] | None = None,
        target_sum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE / VFL on the soft target, normalised by ``sum(target).clamp(1)``.

        With ``dims`` the reduction runs over those dims only (one head per leading index);
        ``prob`` is the detached sigmoid of ``logits`` when already available.
        """

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.person_loss == "vfl":
            if prob is None:
                prob = torch.sigmoid(logits.detach())
            w = self.vfl_alpha * prob.pow(self.vfl_gamma) * (1.0 - target) + target
            bce = bce * w
        if weight is not None:
            bce = bce * weight
        if dims is None:
            return bce.sum() / target.sum().clamp(min=1.0)
        if target_sum is None:
            target_sum = target.sum(dim=dims)
        return bce.sum(dim=dims) / target_sum.clamp(min=1.0)

    def _heads_loss(
        self,
        names: list[str],
        raws: list[RawPredictions],
        assigners: list[PoseTAL],
        padded: PaddedTargets,
        root_prior_scale: float,
    ) -> dict[str, dict[str, torch.Tensor | float]]:
        """Assignment and loss terms of every head of one batch (see the module docstring)."""

        num_heads = len(names)
        ref = raws[0]
        device = ref.person.device
        batch = ref.person.shape[0]
        anchors = ref.anchors.float()
        strides = ref.strides.float()
        logits = torch.stack([raw.person.float().reshape(batch, -1) for raw in raws])  # (H,B,N)
        kpt_raws = [raw.kpt.float() for raw in raws]  # (B,N,K,5) each
        probs = torch.sigmoid(logits).detach()  # (H,B,N)

        # ---- shared target-side geometry and candidate pairs (two host syncs)
        geo = assigners[0].gt_geometry(padded.kpts, padded.area, padded.mask)
        ag = assigners[0].anchor_geometry(anchors, geo)
        cb, cm, cn = ag.cand_idx

        # ---- candidate OKS of every head in one pass (GT side shared)
        with torch.no_grad():
            ax_c = anchors[cn, 0].reshape(-1, 1)
            ay_c = anchors[cn, 1].reshape(-1, 1)
            st_c = strides[cn].reshape(-1, 1)
            raw_c = torch.stack([kpt_raw[cb, cn] for kpt_raw in kpt_raws])  # (H,T,K,5)
            pred_c = torch.stack([ax_c + raw_c[..., 0] * st_c, ay_c + raw_c[..., 1] * st_c], dim=-1)  # (H,T,K,2)
            oks_c = candidate_oks(
                pred_c, padded.kpts[cb, cm, :, :2], geo.labeled[cb, cm], geo.scale2[cb, cm], self.sigmas,
                eps=assigners[0].eps,
            )  # (H,T)

        results: list[AssignResult] = [
            assigner.assign_from_candidates(probs[h], oks_c[h], geo, ag, root_prior_scale, sync_stats=False)
            for h, assigner in enumerate(assigners)
        ]

        fg_all = torch.stack([r.fg_mask for r in results], dim=0)  # (H,B,N)
        gi_all = torch.stack([r.gt_idx for r in results], dim=0)
        ts_all = torch.stack([r.target_score for r in results], dim=0)
        ts_sum = ts_all.sum(dim=(1, 2))  # (H,)

        # ---- person term of every head in one pass (+ optional o2o anti-duplicate boost)
        boosted_counts: list[torch.Tensor] = []  # one int64 count per head, only when the boost is on
        weight = None
        if self.o2o_negative_boost != 1.0:
            weight = torch.ones_like(ts_all)
            for h, name in enumerate(names):
                if name != "o2o":
                    boosted_counts.append(torch.zeros((), device=device, dtype=torch.int64))
                    continue
                boosted = ag.in_any_extent & ~fg_all[h] & (probs[h] >= self.o2o_negative_min_prob)
                weight[h] = torch.where(boosted, self.o2o_negative_boost, 1.0)
                boosted_counts.append(boosted.sum())
        person_vec = self._person_loss(logits, ts_all, weight, prob=probs, dims=(1, 2), target_sum=ts_sum)  # (H,)

        # ---- positives of all heads: one nonzero, then one packed stats transfer
        ph, pb, pn = fg_all.nonzero(as_tuple=True)
        num_pos_total = int(ph.numel())
        gid = gi_all[ph, pb, pn]  # (P,)
        w_anchor = ts_all[ph, pb, pn]  # (P,)
        gt = padded.kpts[pb, gid]  # (P,K,3)
        v = gt[..., 2]
        labeled = v > 0.5
        labeled_f = labeled.float()
        n_labeled = labeled_f.sum(-1)  # (P,)

        # int64 per-head stats (the per-head positive count is the stats' pos_total) and the
        # float32 diagnostics (counts are exact in float32 here): one device->host transfer
        counts = torch.cat([r.stats_tensor for r in results]).to(torch.float32)
        float_parts = [
            torch.stack([
                sigma_units_from_raw(kpt_raw[..., 2], floor=self.sigma_floor).mean() if kpt_raw.numel()
                else torch.zeros((), device=device)
                for kpt_raw in kpt_raws
            ]),
            ts_sum,
            n_labeled.sum().reshape(1),
        ]
        if boosted_counts:
            float_parts.append(torch.stack(boosted_counts).to(torch.float32))
        packed = torch.cat([counts] + float_parts).tolist()
        pos = 0
        stats_per_head = []
        num_pos_per_head = []
        for h in range(num_heads):
            stats_per_head.append(PoseTAL.stats_from_tensor(packed[pos:pos + 5]))
            num_pos_per_head.append(int(packed[pos + 2]))
            pos += 5
        mean_sigma = packed[pos:pos + num_heads]
        pos += num_heads
        sum_target = packed[pos:pos + num_heads]
        pos += num_heads
        num_labeled = int(round(packed[pos]))
        pos += 1
        boosted_list = packed[pos:pos + num_heads] if boosted_counts else [0.0] * num_heads

        offsets = [0]
        for n in num_pos_per_head:
            offsets.append(offsets[-1] + n)
        slices = [slice(offsets[h], offsets[h + 1]) for h in range(num_heads)]
        zero = torch.zeros((), device=device, dtype=torch.float32)
        heads: dict[str, dict[str, torch.Tensor | float]] = {}
        for h, name in enumerate(names):
            stats = stats_per_head[h]
            n_pos = num_pos_per_head[h]
            heads[name] = {
                "person": person_vec[h],
                "oks": zero,
                "rle": zero,
                "present": zero,
                "visible": zero,
                "coverage": stats["gt_covered"] / max(stats["gt_total"], 1.0) if stats["gt_total"] > 0 else 1.0,
                "pos_per_gt": stats["pos_per_gt"],
                "cand_per_gt": stats["cand_per_gt"],
                "gt_total": stats["gt_total"],
                "small_gt": stats["small_gt"],
                "num_pos": float(n_pos),
                "mean_sigma_units": float(mean_sigma[h]),
                "mean_target_score": float(sum_target[h] / n_pos) if n_pos else 0.0,
                "boosted": float(boosted_list[h]),
            }
        if num_pos_total == 0:
            return heads

        # ---- per-point terms of all positives at once (heads concatenated: o2o first)
        p_raw = torch.cat([kpt_raws[h][pb[slices[h]], pn[slices[h]]] for h in range(num_heads)], dim=0)  # (P,K,5)
        p_stride = strides[pn].reshape(-1, 1, 1)  # (P,1,1)
        p_xy = torch.stack(
            [anchors[pn, 0].reshape(-1, 1) + p_raw[..., 0] * p_stride[:, :, 0],
             anchors[pn, 1].reshape(-1, 1) + p_raw[..., 1] * p_stride[:, :, 0]],
            dim=-1,
        )  # (P,K,2) decoded exactly as decode_keypoints does
        p_sigma = sigma_units_from_raw(p_raw[..., 2], floor=self.sigma_floor)  # (P,K)
        gt_xy = gt[..., :2]
        unknown = v < -0.5
        visible = v > 1.5
        w_anchor = w_anchor.detach()
        w_kpt = w_anchor * (n_labeled > 0).float()
        n_labeled_c = n_labeled.clamp(min=1.0)
        scale2 = geo.scale2[pb, gid]  # (P,)

        # OKS / Smooth-OKS term, per point -> per anchor -> weighted over anchors
        oks_pt = oks_loss_terms(p_xy, gt_xy, scale2, self.sigmas, kind=self.kpt_loss)  # (P,K)
        oks_anchor = (oks_pt * labeled_f).sum(-1) / n_labeled_c

        # RLE in stride units on labeled points only; the labeled rows are compacted with a
        # stable sort (same row order as boolean indexing) so no extra host sync is needed
        num_points = p_xy.shape[0] * p_xy.shape[1]
        order = labeled.reshape(-1).to(torch.int8).argsort(descending=True, stable=True)[:num_labeled]
        mu_units = (p_xy / p_stride).reshape(num_points, 2)[order]  # (Q,2)
        gt_units = (gt_xy / p_stride).reshape(num_points, 2)[order]
        sig_units = p_sigma.reshape(num_points)[order]  # (Q,)
        rle_q = self.rle(pred_xy=mu_units, pred_sigma=sig_units, gt_xy=gt_units)  # (Q,)
        rle_pt = torch.zeros((num_points,), device=device, dtype=torch.float32).index_put((order,), rle_q)
        rle_anchor = rle_pt.reshape(-1, p_xy.shape[1]).sum(-1) / n_labeled_c

        # present: BCE(present_logit, 1[v>0]); v=-1 ignored; v=0 weighted
        present_logit = p_raw[..., 3]
        present_w = torch.where(unknown, 0.0, torch.where(labeled, 1.0, self.unlabeled_present_weight)).to(v.dtype)
        present_bce = F.binary_cross_entropy_with_logits(present_logit, labeled_f, reduction="none") * present_w
        n_valid = (~unknown).float().sum(-1)
        present_anchor = present_bce.sum(-1) / n_valid.clamp(min=1.0)
        w_present = w_anchor * (n_valid > 0).float()

        # visible: BCE(visible_logit, 1[v==2]) where v > 0
        visible_logit = p_raw[..., 4]
        visible_bce = F.binary_cross_entropy_with_logits(visible_logit, visible.float(), reduction="none") * labeled_f
        visible_anchor = visible_bce.sum(-1) / n_labeled_c

        # the four target_score-weighted means of both heads in one pass: rows oks, rle, present, visible
        term_rows = torch.stack([oks_anchor, rle_anchor, present_anchor, visible_anchor])  # (4,P)
        weight_rows = torch.stack([w_kpt, w_kpt, w_present, w_kpt])
        per_head = _sliced_weighted_means(term_rows, weight_rows, slices)  # list of (4,)
        for h, name in enumerate(names):
            if num_pos_per_head[h] == 0:
                continue
            terms = per_head[h]
            heads[name].update({"oks": terms[0], "rle": terms[1], "present": terms[2], "visible": terms[3]})
        return heads
