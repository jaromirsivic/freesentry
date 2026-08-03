"""Training-component tests: STAL guarantee, ProgLoss schedule, MuSGD, and an
end-to-end loss forward/backward on a tiny batch."""

from __future__ import annotations

import torch

from xtx3.engine.optim import MuSGD, build_optimizer, newton_schulz_orthogonalize
from xtx3.losses.assignment import TaskAlignedAssigner, inflate_small_boxes
from xtx3.losses.pose_loss import ProgLossSchedule, XTX3Loss
from xtx3.models import build_model
from xtx3.models.head import build_anchors
from xtx3.utils.keypoints import NUM_KEYPOINTS


def _anchors() -> tuple[torch.Tensor, torch.Tensor]:
    return build_anchors(
        feature_sizes=[(48, 48), (24, 24), (12, 12)],
        strides=[8, 16, 32],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestSTAL:
    def test_six_px_box_gets_min_anchors(self) -> None:
        """A 6-px synthetic GT box receives >= 4 positive candidates."""

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

    def test_network_size_override(self) -> None:
        """Multi-scale training passes the actual size; the assigner must accept it."""

        anchors256, _ = build_anchors(
            feature_sizes=[(32, 32), (16, 16), (8, 8)],
            strides=[8, 16, 32],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        n = anchors256.shape[0]
        assigner = TaskAlignedAssigner(topk=10, network_size=384)
        gt = torch.tensor([[100.0, 100.0, 160.0, 200.0]])
        result = assigner.assign(
            pred_scores=torch.rand(n) * 0.5,
            pred_boxes=torch.cat([anchors256 - 30.0, anchors256 + 30.0], dim=1),
            anchors=anchors256,
            gt_boxes=gt,
            network_size=256,
        )
        assert int(result.fg_mask.sum()) >= 1


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
        criterion = XTX3Loss()
        criterion.set_progress(0.5)
        w_o2m, w_o2o = criterion.current_weights()
        assert 0.25 < w_o2m < 1.0
        assert 0.5 < w_o2o < 1.0


class TestLossForward:
    def _targets(self, size: int) -> list[dict[str, torch.Tensor]]:
        boxes = torch.tensor([[size * 0.25, size * 0.2, size * 0.7, size * 0.9]])
        kpts = torch.zeros(1, NUM_KEYPOINTS, 3)
        kpts[0, :, 0] = torch.linspace(size * 0.3, size * 0.6, NUM_KEYPOINTS)
        kpts[0, :, 1] = torch.linspace(size * 0.25, size * 0.8, NUM_KEYPOINTS)
        kpts[0, :, 2] = 2
        return [{"boxes": boxes, "keypoints": kpts}]

    def test_backward_through_dual_head(self) -> None:
        torch.manual_seed(0)
        model = build_model(variant="u").train()
        criterion = XTX3Loss(network_size=320)
        out = model(torch.rand(1, 3, 320, 320))
        losses = criterion(out, self._targets(320))
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_multi_scale_image_size(self) -> None:
        """The u variant trains at 256/320/384: loss must honour image_size."""

        torch.manual_seed(0)
        model = build_model(variant="u").train()
        criterion = XTX3Loss(network_size=320)
        for size in (256, 384):
            out = model(torch.rand(1, 3, size, size))
            losses = criterion(out, self._targets(size), image_size=size)
            assert torch.isfinite(losses["total"])

    def test_empty_targets(self) -> None:
        model = build_model(variant="u").train()
        criterion = XTX3Loss(network_size=256)
        out = model(torch.rand(1, 3, 256, 256))
        empty = [{"boxes": torch.zeros(0, 4), "keypoints": torch.zeros(0, NUM_KEYPOINTS, 3)}]
        losses = criterion(out, empty)
        assert torch.isfinite(losses["total"])


class TestMuSGD:
    def test_orthogonalisation(self) -> None:
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
        assert musgd.param_groups[0]["muon"] is True
        assert all(p.ndim >= 2 for p in musgd.param_groups[0]["params"])
        assert all(p.ndim <= 1 for p in musgd.param_groups[1]["params"])
