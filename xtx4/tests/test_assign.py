"""Pose-TAL assignment tests (CONTRACTS.md sections 6 and 13). CPU only, no model."""

from __future__ import annotations

import torch

from xtx4.losses.assign import PoseTAL
from xtx4.losses.targets import pad_targets
from xtx4.models.head_types import build_anchors, decode_keypoints
from xtx4.utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS

S = 320
STRIDES = [8, 16, 32]
K = NUM_KEYPOINTS


def make_anchors(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return build_anchors(
        feature_sizes=[(S // s, S // s) for s in STRIDES],
        strides=STRIDES,
        device=torch.device(device),
        dtype=torch.float32,
    )


def person_kpts(cx: float, cy: float, span: float, *, visible: int = 2) -> torch.Tensor:
    """A nine-point person whose labeled points lie in a ``span``-sized square around (cx, cy)."""

    offsets = torch.tensor(
        [[0.0, -0.5], [-0.2, -0.6], [0.2, -0.6], [-0.4, -0.5], [0.4, -0.5],
         [-0.5, 0.0], [0.5, 0.0], [-0.3, 0.5], [0.3, 0.5]]
    )
    pts = torch.zeros(K, 3)
    pts[:, 0] = cx + offsets[:, 0] * span
    pts[:, 1] = cy + offsets[:, 1] * span
    pts[:, 2] = float(visible)
    return pts


def random_raw(batch: int, num_anchors: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    person = torch.randn(batch, num_anchors, generator=g) * 0.5 - 4.0
    kpt = torch.randn(batch, num_anchors, K, 5, generator=g) * 0.3
    return person, kpt


def oracle_raw(anchors: torch.Tensor, strides: torch.Tensor, gt_kpts: torch.Tensor, gt_mask: torch.Tensor):
    """GT-aligned raw predictions: every anchor regresses the keypoints of the nearest GT root."""

    batch, num_gt = gt_mask.shape
    num_anchors = anchors.shape[0]
    assigner = PoseTAL(OKS_SIGMAS)
    geo = assigner.gt_geometry(gt_kpts, torch.full((batch, num_gt), float("nan")), gt_mask)
    d2 = ((anchors.reshape(1, 1, num_anchors, 2) - geo.root.unsqueeze(2)) ** 2).sum(-1)  # (B,M,N)
    d2 = d2.masked_fill(~gt_mask.unsqueeze(-1), float("inf"))
    nearest = d2.argmin(dim=1)  # (B,N)
    target = torch.gather(gt_kpts, 1, nearest.reshape(batch, num_anchors, 1, 1).expand(-1, -1, K, 3))
    kpt = torch.zeros(batch, num_anchors, K, 5)
    kpt[..., 0] = (target[..., 0] - anchors[:, 0].reshape(1, -1, 1)) / strides.reshape(1, -1, 1)
    kpt[..., 1] = (target[..., 1] - anchors[:, 1].reshape(1, -1, 1)) / strides.reshape(1, -1, 1)
    kpt[..., 2] = 0.6
    ext = torch.gather(geo.extent, 1, nearest.unsqueeze(-1).expand(-1, -1, 4))  # (B,N,4)
    ax = anchors[:, 0].reshape(1, -1)
    ay = anchors[:, 1].reshape(1, -1)
    inside = (ax >= ext[..., 0]) & (ax <= ext[..., 2]) & (ay >= ext[..., 1]) & (ay <= ext[..., 3])
    person = torch.where(inside & gt_mask.any(1, keepdim=True), torch.full_like(ax, 4.0), torch.full_like(ax, -4.0))
    return person, kpt


def run_assign(assigner: PoseTAL, person, kpt, anchors, strides, padded, prior: float = 0.0):
    dec = decode_keypoints(kpt.float(), anchors, strides)
    return assigner.assign(
        torch.sigmoid(person.float()), dec["xy"], anchors, strides, padded.kpts, padded.area, padded.mask, prior
    )


def positives_per_gt(result, padded) -> torch.Tensor:
    counts = torch.zeros(padded.mask.shape, dtype=torch.int64)
    for b in range(padded.mask.shape[0]):
        fg = result.fg_mask[b]
        counts[b] = torch.bincount(result.gt_idx[b][fg], minlength=padded.mask.shape[1])
    return counts


def test_tiny_person_between_cells_gets_min_candidates():
    anchors, strides = make_anchors()
    assigner = PoseTAL(OKS_SIGMAS, topk=10)
    # centre exactly at a cell corner (between the stride-8 centres 4 and 12), 6 px extent
    for cx, cy in ((8.0, 8.0), (2.0, 2.0), (S - 2.0, 160.0)):
        padded = pad_targets([{"keypoints": person_kpts(cx, cy, 6.0).unsqueeze(0)}], "cpu")
        person, kpt = random_raw(1, anchors.shape[0], seed=1)
        res = run_assign(assigner, person, kpt, anchors, strides, padded)
        assert res.stats["small_gt"] == 1
        assert res.stats["gt_covered"] == 1
        assert int(res.fg_mask.sum()) >= assigner.min_candidates
        # guaranteed candidates of a small GT carry at least the 0.05 target floor
        assert float(res.target_score[res.fg_mask].min()) >= 0.05 - 1e-6
        assert torch.all(res.gt_idx[res.fg_mask] == 0)


def test_two_persons_in_one_cell_get_distinct_anchors():
    anchors, strides = make_anchors()
    a = person_kpts(100.0, 100.0, 4.0)
    b = person_kpts(103.0, 103.0, 4.0)
    padded = pad_targets([{"keypoints": torch.stack([a, b])}], "cpu")
    assigner = PoseTAL(OKS_SIGMAS, topk=10)

    person, kpt = oracle_raw(anchors, strides, padded.kpts, padded.mask)
    res = run_assign(assigner, person, kpt, anchors, strides, padded)
    counts = positives_per_gt(res, padded)
    assert res.stats["gt_total"] == 2
    assert res.stats["gt_covered"] == 2
    assert torch.all(counts[0] >= 1)
    assert int(res.fg_mask.sum()) == int(counts.sum())  # each anchor counted once

    # random predictions: invariants still hold (an anchor belongs to at most one GT)
    person, kpt = random_raw(1, anchors.shape[0], seed=3)
    res = run_assign(assigner, person, kpt, anchors, strides, padded)
    counts = positives_per_gt(res, padded)
    assert int(res.fg_mask.sum()) == int(counts.sum())
    assert res.stats["cand_per_gt"] >= 2


def test_zero_gt_batch_has_no_positives():
    anchors, strides = make_anchors()
    padded = pad_targets([{"keypoints": torch.zeros(0, K, 3)}, {"keypoints": torch.zeros(0, K, 3)}], "cpu")
    assert padded.max_persons == 1 and not padded.mask.any()
    person, kpt = random_raw(2, anchors.shape[0], seed=4)
    res = run_assign(PoseTAL(OKS_SIGMAS), person, kpt, anchors, strides, padded, prior=1.0)
    assert not res.fg_mask.any()
    assert float(res.target_score.abs().sum()) == 0.0
    assert res.stats["gt_total"] == 0 and res.stats["gt_covered"] == 0


def test_single_point_person():
    anchors, strides = make_anchors()
    pts = torch.zeros(K, 3)
    pts[0, :2] = torch.tensor([150.0, 90.0])
    pts[0, 2] = 2.0
    padded = pad_targets([{"keypoints": pts.unsqueeze(0)}], "cpu")
    assigner = PoseTAL(OKS_SIGMAS)
    geo = assigner.gt_geometry(padded.kpts, padded.area, padded.mask)
    extent = geo.extent[0, 0]
    assert torch.allclose(extent, torch.tensor([138.0, 78.0, 162.0, 102.0]))
    assert float(geo.scale2[0, 0]) == 16.0**2
    assert torch.allclose(geo.root[0, 0], torch.tensor([150.0, 90.0]))
    person, kpt = random_raw(1, anchors.shape[0], seed=5)
    res = run_assign(assigner, person, kpt, anchors, strides, padded)
    assert res.stats["gt_covered"] == 1
    assert int(res.fg_mask.sum()) >= assigner.min_candidates
    # no labeled point at all -> never a target
    pts[0, 2] = 0.0
    padded = pad_targets([{"keypoints": pts.unsqueeze(0)}], "cpu")
    res = run_assign(assigner, person, kpt, anchors, strides, padded, prior=1.0)
    assert not res.fg_mask.any() and res.stats["gt_total"] == 0


def test_padded_rows_never_positive():
    anchors, strides = make_anchors()
    img0 = torch.stack([person_kpts(60, 60, 40), person_kpts(200, 100, 60), person_kpts(250, 250, 30)])
    img1 = person_kpts(160, 160, 50).unsqueeze(0)
    padded = pad_targets([{"keypoints": img0}, {"keypoints": img1}], "cpu")
    assert padded.max_persons == 3 and padded.mask.tolist() == [[True, True, True], [True, False, False]]
    person, kpt = random_raw(2, anchors.shape[0], seed=6)
    for prior in (0.0, 1.0):
        res = run_assign(PoseTAL(OKS_SIGMAS), person, kpt, anchors, strides, padded, prior=prior)
        assert torch.all(res.gt_idx[1][res.fg_mask[1]] == 0)
        assert res.stats["gt_total"] == 4


def test_o2o_assigns_exactly_one_anchor_per_covered_gt():
    anchors, strides = make_anchors()
    rows = [torch.stack([person_kpts(60, 60, 40), person_kpts(200, 100, 60), person_kpts(250, 250, 30)]),
            person_kpts(160, 160, 50).unsqueeze(0)]
    padded = pad_targets([{"keypoints": r} for r in rows], "cpu")
    o2o = PoseTAL(OKS_SIGMAS, topk=7, topk2=1)
    for seed, use_oracle in ((7, False), (8, True)):
        if use_oracle:
            person, kpt = oracle_raw(anchors, strides, padded.kpts, padded.mask)
        else:
            person, kpt = random_raw(2, anchors.shape[0], seed=seed)
        res = run_assign(o2o, person, kpt, anchors, strides, padded)
        counts = positives_per_gt(res, padded)
        assert torch.all(counts[padded.mask] <= 1)
        assert int(res.fg_mask.sum()) == res.stats["gt_covered"]
        if use_oracle:
            assert res.stats["gt_covered"] == 4
            assert torch.all(res.assigned_oks[res.fg_mask] > 0.99)


def test_coverage_on_random_separated_batch_with_oracle_predictions():
    anchors, strides = make_anchors()
    g = torch.Generator().manual_seed(11)
    batch, per_image = 8, 4
    rows = []
    for _ in range(batch):
        persons = []
        for i in range(per_image):
            cx = 60.0 + (i % 2) * 180.0 + float(torch.rand(1, generator=g)) * 40.0
            cy = 60.0 + (i // 2) * 180.0 + float(torch.rand(1, generator=g)) * 40.0
            span = 6.0 + float(torch.rand(1, generator=g)) * 60.0
            persons.append(person_kpts(cx, cy, span))
        rows.append(torch.stack(persons))
    padded = pad_targets([{"keypoints": r} for r in rows], "cpu")
    person, kpt = oracle_raw(anchors, strides, padded.kpts, padded.mask)
    for assigner in (PoseTAL(OKS_SIGMAS, topk=10), PoseTAL(OKS_SIGMAS, topk=7, topk2=1)):
        res = run_assign(assigner, person, kpt, anchors, strides, padded)
        assert res.stats["gt_total"] == batch * per_image
        assert res.stats["gt_covered"] / res.stats["gt_total"] >= 0.995
        assert torch.all(res.target_score[res.fg_mask] > 0.0)
        assert torch.all(res.assigned_oks[res.fg_mask] > 0.5)


def test_fp16_inputs_match_fp32_assignment():
    anchors, strides = make_anchors()
    rows = [torch.stack([person_kpts(60, 60, 40), person_kpts(200, 100, 60)]), person_kpts(160, 160, 50).unsqueeze(0)]
    padded = pad_targets([{"keypoints": r} for r in rows], "cpu")
    person, kpt = random_raw(2, anchors.shape[0], seed=12)
    person16, kpt16 = person.half(), kpt.half()
    assigner = PoseTAL(OKS_SIGMAS, topk=10)
    res16 = run_assign(assigner, person16, kpt16, anchors, strides, padded, prior=0.5)
    res32 = run_assign(assigner, person16.float(), kpt16.float(), anchors, strides, padded, prior=0.5)
    assert torch.equal(res16.fg_mask, res32.fg_mask)
    assert torch.equal(res16.gt_idx, res32.gt_idx)
    assert torch.allclose(res16.target_score, res32.target_score)
    assert res16.target_score.dtype == torch.float32
    # the half-rounded inputs give the same positives set as the unrounded ones up to a few anchors
    res_full = run_assign(assigner, person, kpt, anchors, strides, padded, prior=0.5)
    assert int((res_full.fg_mask != res16.fg_mask).sum()) <= 8


def test_geometry_scale_and_root_rules():
    assigner = PoseTAL(OKS_SIGMAS)
    full = person_kpts(100, 100, 40)  # raw extent 40 x 44
    no_shoulders = full.clone()
    no_shoulders[5:7, 2] = 0.0
    face_only = full.clone()
    face_only[5:, 2] = -1.0
    kpts = torch.stack([full, no_shoulders, face_only]).unsqueeze(0)
    area = torch.tensor([[1234.0, float("nan"), 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    geo = assigner.gt_geometry(kpts, area, mask)
    assert float(geo.scale2[0, 0]) == 1234.0
    w, h = geo.raw_wh[0, 1].tolist()
    assert abs(float(geo.scale2[0, 1]) - (1.5 * w) * (1.5 * h)) < 1e-3
    assert torch.allclose(geo.root[0, 0], torch.tensor([100.0, 100.0]))  # shoulder centre
    assert torch.allclose(geo.root[0, 1], torch.tensor([100.0, 120.0]))  # hip centre
    assert torch.allclose(geo.root[0, 2], full[:5, :2].mean(0))  # face centre
    assert geo.labeled[0, 2].tolist() == [True] * 5 + [False] * 4
    # extent expansion: 40 px raw width -> 60 px, centred
    assert torch.allclose(geo.extent[0, 0, [0, 2]], torch.tensor([70.0, 130.0]))
    assert not bool(geo.small.any())


def test_root_prior_concentrates_positives_near_root_when_oks_is_zero():
    anchors, strides = make_anchors()
    padded = pad_targets([{"keypoints": person_kpts(160, 160, 80).unsqueeze(0)}], "cpu")
    num_anchors = anchors.shape[0]
    person = torch.full((1, num_anchors), -4.0)
    kpt = torch.zeros(1, num_anchors, K, 5)
    kpt[..., 0] = 200.0  # predictions 1600 px away -> OKS underflows to 0 for every anchor
    assigner = PoseTAL(OKS_SIGMAS, topk=10)
    res = run_assign(assigner, person, kpt, anchors, strides, padded, prior=1.0)
    assert int(res.fg_mask.sum()) == 10
    geo = assigner.gt_geometry(padded.kpts, padded.area, padded.mask)
    d2 = ((anchors - geo.root[0, 0]) ** 2).sum(-1)
    chosen = d2[res.fg_mask[0]]
    # the chosen anchors are the 10 nearest to the root
    assert float(chosen.max()) <= float(d2.topk(10, largest=False).values.max()) + 1e-6
    assert torch.all(res.target_score[res.fg_mask] >= 0.0)


def test_batched_result_matches_per_image_result():
    anchors, strides = make_anchors()
    rows = [torch.stack([person_kpts(60, 60, 40), person_kpts(200, 100, 60)]), person_kpts(160, 160, 50).unsqueeze(0)]
    padded = pad_targets([{"keypoints": r} for r in rows], "cpu")
    person, kpt = random_raw(2, anchors.shape[0], seed=13)
    assigner = PoseTAL(OKS_SIGMAS, topk=10)
    res = run_assign(assigner, person, kpt, anchors, strides, padded, prior=0.3)
    for b in range(2):
        single = pad_targets([{"keypoints": rows[b]}], "cpu")
        res_b = run_assign(assigner, person[b : b + 1], kpt[b : b + 1], anchors, strides, single, prior=0.3)
        assert torch.equal(res.fg_mask[b], res_b.fg_mask[0])
        assert torch.equal(res.gt_idx[b][res.fg_mask[b]], res_b.gt_idx[0][res_b.fg_mask[0]])
        assert torch.allclose(res.target_score[b], res_b.target_score[0], atol=1e-6)
