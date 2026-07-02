"""Training-component tests: STAL guarantee, ProgLoss schedule, MuSGD (criterion 7)."""

from __future__ import annotations

import torch

from xtx2.losses.assignment import TaskAlignedAssigner, inflate_small_boxes
from xtx2.losses.pose_loss import ProgLossSchedule, XTX2Loss
from xtx2.engine.optim import MuSGD, build_optimizer, newton_schulz_orthogonalize
from xtx2.models.head import build_anchors


def _anchors() -> tuple[torch.Tensor, torch.Tensor]:
    return build_anchors(
        feature_sizes=[(48, 48), (24, 24), (12, 12)],
        strides=[8, 16, 32],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestSTAL:
    def test_six_px_box_gets_min_anchors(self) -> None:
        """A 6-px synthetic GT box receives >= 4 positive candidates (criterion 7)."""

        anchors, _ = _anchors()
        n = anchors.shape[0]
        assigner = TaskAlignedAssigner(topk=10, stal_min_anchors=4, stal_size_px=8.0)
        gt = torch.tensor([[190.0, 190.0, 196.0, 196.0]])  # 6 px box near centre
        # Untrained-network conditions: near-zero scores, degenerate boxes.
        result = assigner.assign(
            pred_scores=torch.full((n,), 0.001),
            pred_boxes=anchors.repeat_interleave(2, dim=1) * 1.0,  # zero-area boxes
            anchors=anchors,
            gt_boxes=gt,
        )
        assert int(result.fg_mask.sum()) >= 4
        # All positives carry a usable (non-zero) soft target.
        assert float(result.target_scores[result.fg_mask].min()) > 0

    def test_six_px_box_at_corner(self) -> None:
        anchors, _ = _anchors()
        n = anchors.shape[0]
        assigner = TaskAlignedAssigner(topk=10, stal_min_anchors=4, stal_size_px=8.0)
        gt = torch.tensor([[1.0, 1.0, 7.0, 7.0]])
        result = assigner.assign(
            pred_scores=torch.full((n,), 0.001),
            pred_boxes=anchors.repeat_interleave(2, dim=1),
            anchors=anchors,
            gt_boxes=gt,
        )
        assert int(result.fg_mask.sum()) >= 4

    def test_large_box_unchanged_by_stal(self) -> None:
        boxes = torch.tensor([[100.0, 100.0, 200.0, 220.0]])
        inflated = inflate_small_boxes(boxes, stal_size_px=8.0, min_side=16.0)
        assert torch.equal(boxes, inflated)

    def test_small_box_inflated_around_centre(self) -> None:
        boxes = torch.tensor([[100.0, 100.0, 106.0, 106.0]])
        inflated = inflate_small_boxes(boxes, stal_size_px=8.0, min_side=16.0)
        assert torch.allclose(inflated, torch.tensor([[95.0, 95.0, 111.0, 111.0]]))

    def test_one_to_one_stays_single(self) -> None:
        """The consistent one-to-one assigner keeps at most 1 anchor per GT."""

        anchors, _ = _anchors()
        n = anchors.shape[0]
        assigner = TaskAlignedAssigner(topk=1, stal_min_anchors=1)
        gt = torch.tensor([[100.0, 100.0, 180.0, 260.0]])
        scores = torch.rand(n) * 0.5 + 0.2
        boxes = torch.cat([anchors - 40.0, anchors + 40.0], dim=1)
        result = assigner.assign(
            pred_scores=scores, pred_boxes=boxes, anchors=anchors, gt_boxes=gt
        )
        assert int(result.fg_mask.sum()) <= 1


class TestProgLoss:
    def test_linear_schedule_defaults(self) -> None:
        schedule = ProgLossSchedule()
        assert schedule.weights(0.0) == (1.0, 0.5)
        assert schedule.weights(1.0) == (0.25, 1.0)
        w_o2m, w_o2o = schedule.weights(0.5)
        assert abs(w_o2m - 0.625) < 1e-9
        assert abs(w_o2o - 0.75) < 1e-9

    def test_progress_clamped(self) -> None:
        schedule = ProgLossSchedule()
        assert schedule.weights(-1.0) == schedule.weights(0.0)
        assert schedule.weights(2.0) == schedule.weights(1.0)

    def test_criterion_exposes_weights(self) -> None:
        criterion = XTX2Loss()
        criterion.set_progress(0.5)
        w_o2m, w_o2o = criterion.current_weights()
        assert 0.25 < w_o2m < 1.0
        assert 0.5 < w_o2o < 1.0


class TestMuSGD:
    def test_orthogonalisation(self) -> None:
        """Newton-Schulz drives the singular values towards ~1 (Muon converges
        them into roughly [0.7, 1.2], not exactly to identity)."""

        torch.manual_seed(0)
        mat = torch.randn(16, 32)
        ortho = newton_schulz_orthogonalize(mat, steps=5)
        sv_in = torch.linalg.svdvals(mat)
        sv_out = torch.linalg.svdvals(ortho)
        assert sv_in.max() / sv_in.min() > 3  # input clearly non-orthogonal
        assert float(sv_out.min()) > 0.5
        assert float(sv_out.max()) < 1.5

    def test_musgd_reduces_quadratic(self) -> None:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(8, 8))
        bias = torch.nn.Parameter(torch.randn(8))
        opt = MuSGD(
            [
                {"params": [weight], "muon": True, "weight_decay": 0.0},
                {"params": [bias], "muon": False, "weight_decay": 0.0},
            ],
            lr=0.05,
            momentum=0.9,
        )
        with torch.no_grad():
            initial = float((weight**2).sum() + (bias**2).sum())
        for _ in range(50):
            opt.zero_grad()
            loss = (weight**2).sum() + (bias**2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            assert float((weight**2).sum() + (bias**2).sum()) < initial * 0.5

    def test_build_optimizer_variants(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.BatchNorm2d(8))
        musgd = build_optimizer([model], name="musgd")
        assert isinstance(musgd, MuSGD)
        sgd = build_optimizer([model], name="sgd")
        assert isinstance(sgd, torch.optim.SGD)
        # Weight matrices routed to the muon/decay group, 1-D params to the other.
        assert musgd.param_groups[0]["muon"] is True
        assert all(p.ndim >= 2 for p in musgd.param_groups[0]["params"])
        assert all(p.ndim <= 1 for p in musgd.param_groups[1]["params"])
