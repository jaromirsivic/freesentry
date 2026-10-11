"""XTX4 loss tests (CONTRACTS.md sections 7 and 13). CPU only, synthetic raw tensors, no model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from xtx4.losses import PoseTAL, XTX4Loss, pad_targets
from xtx4.losses.calibrate import calibrate_loss_weights, propose_lambdas, write_calibration_json
from xtx4.losses.oks import keypoint_u2, oks_loss_terms, pairwise_oks, smooth_oks
from xtx4.losses.pose_loss import ProgLossSchedule
from xtx4.losses.rle import RLELoss
from xtx4.models.head_types import HeadOutput, RawPredictions, build_anchors
from xtx4.utils.keypoints import NUM_KEYPOINTS, OKS_SIGMAS

S = 320
STRIDES = [8, 16, 32]
K = NUM_KEYPOINTS
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "u.json"


def make_anchors() -> tuple[torch.Tensor, torch.Tensor]:
    return build_anchors(
        feature_sizes=[(S // s, S // s) for s in STRIDES],
        strides=STRIDES,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def person_kpts(cx: float, cy: float, span: float, *, visible: int = 2) -> torch.Tensor:
    offsets = torch.tensor(
        [[0.0, -0.5], [-0.2, -0.6], [0.2, -0.6], [-0.4, -0.5], [0.4, -0.5],
         [-0.5, 0.0], [0.5, 0.0], [-0.3, 0.5], [0.3, 0.5]]
    )
    pts = torch.zeros(K, 3)
    pts[:, 0] = cx + offsets[:, 0] * span
    pts[:, 1] = cy + offsets[:, 1] * span
    pts[:, 2] = float(visible)
    return pts


def random_raw(batch: int, num_anchors: int, *, seed: int = 0, requires_grad: bool = False):
    g = torch.Generator().manual_seed(seed)
    person = (torch.randn(batch, num_anchors, 1, generator=g) * 0.5 - 4.0).requires_grad_(requires_grad)
    kpt = (torch.randn(batch, num_anchors, K, 5, generator=g) * 0.3).requires_grad_(requires_grad)
    return person, kpt


def default_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def make_criterion(**overrides) -> XTX4Loss:
    cfg = default_config()
    cfg["loss"].update(overrides)
    crit = XTX4Loss.from_config(cfg)
    crit.set_progress(0.0)
    return crit


def two_person_targets() -> list[dict]:
    return [
        {"keypoints": torch.stack([person_kpts(60, 60, 40), person_kpts(200, 120, 70)]), "area": torch.tensor([2000.0, float("nan")])},
        {"keypoints": person_kpts(160, 200, 50).unsqueeze(0)},
    ]


# ----------------------------------------------------------------------------- OKS / SOKS
def test_soks_continuous_and_differentiable_at_unit_radius():
    u = torch.tensor([1.0 - 1e-4, 1.0, 1.0 + 1e-4], dtype=torch.float64)
    sim = smooth_oks(u**2)
    assert torch.allclose(sim, torch.full_like(sim, math.exp(-1.0)), atol=5e-4)
    # derivative wrt u is -2/e on both sides
    u_req = torch.tensor([0.9, 1.1], dtype=torch.float64, requires_grad=True)
    smooth_oks(u_req**2).sum().backward()
    assert torch.allclose(u_req.grad, torch.tensor([-2 * 0.9 * math.exp(-0.81), -2 * math.exp(-1.2)], dtype=torch.float64))
    # finite gradient everywhere, including exactly zero distance and huge distances
    u2 = torch.tensor([0.0, 1e-12, 0.5, 1.0, 4.0, 1e6], requires_grad=True)
    out = smooth_oks(u2)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(u2.grad).all()
    assert torch.all(out[1:] <= out[:-1])  # monotone decreasing


def test_pairwise_oks_matches_coco_formula_and_chunking():
    g = torch.Generator().manual_seed(0)
    pred = torch.rand(2, 50, K, 2, generator=g) * 100
    gt = torch.rand(2, 3, K, 2, generator=g) * 100
    labeled = torch.rand(2, 3, K, generator=g) > 0.3
    labeled[1, 2] = False
    scale2 = torch.tensor([[500.0, 1000.0, 300.0], [800.0, 20.0, 400.0]])
    sig = torch.as_tensor(OKS_SIGMAS)
    full = pairwise_oks(pred, gt, labeled, scale2, sig)
    chunked = pairwise_oks(pred, gt, labeled, scale2, sig, max_chunk_bytes=1)
    assert torch.allclose(full, chunked)
    # reference (loops)
    for b in range(2):
        for m in range(3):
            for n in (0, 17):
                lab = labeled[b, m]
                if not lab.any():
                    assert full[b, m, n] == 0.0
                    continue
                d2 = ((pred[b, n] - gt[b, m]) ** 2).sum(-1)
                e = torch.exp(-d2 / (2 * scale2[b, m] * (2 * sig) ** 2))
                assert abs(float(e[lab].mean()) - float(full[b, m, n])) < 1e-5
    # per-point terms agree with the pairwise denominator
    u2 = keypoint_u2(pred[0, :3], gt[0], scale2[0], sig)
    assert u2.shape == (3, K)
    assert torch.all(oks_loss_terms(pred[0, :3], gt[0], scale2[0], sig, kind="oks") <= 1.0)


# ----------------------------------------------------------------------------- RLE
def test_rle_conventions_differ_by_log_sigma_per_coordinate():
    torch.manual_seed(0)
    ref = RLELoss(convention="reference")
    x3 = RLELoss(convention="xtx3")
    x3.load_state_dict(ref.state_dict())
    pred = torch.randn(32, 2)
    gt = torch.randn(32, 2)
    sigma = torch.rand(32, 2) * 2 + 0.1
    l_ref = ref(pred_xy=pred, pred_sigma=sigma, gt_xy=gt)
    l_x3 = x3(pred_xy=pred, pred_sigma=sigma, gt_xy=gt)
    assert torch.allclose(l_ref - l_x3, torch.log(sigma).sum(1), atol=1e-5)
    # xtx3 convention without the flow equals the Laplace NLL: log sigma + log 2 + |eps|
    flow_free = x3.flow.log_prob((gt - pred) / sigma)
    laplace = (torch.log(sigma) + math.log(2.0) + ((gt - pred) / sigma).abs()).sum(1)
    assert torch.allclose(l_x3 + flow_free, laplace, atol=1e-5)
    # a single sigma per point broadcasts to both coordinates
    l_single = ref(pred_xy=pred, pred_sigma=sigma[:, 0], gt_xy=gt)
    l_pair = ref(pred_xy=pred, pred_sigma=sigma[:, :1].expand(-1, 2), gt_xy=gt)
    assert torch.allclose(l_single, l_pair)


def test_rle_is_safe_for_extreme_inputs_and_keeps_flow_learnable():
    rle = RLELoss(convention="reference")
    assert sum(p.numel() for p in rle.parameters() if p.requires_grad) > 0
    pred = torch.tensor([[0.0, 0.0], [1e6, -1e6], [float("nan"), 0.0]])
    gt = torch.zeros(3, 2)
    sigma = torch.tensor([[1e-9, 1e-9], [0.05, 0.05], [1.0, 1.0]])
    loss = rle(pred_xy=pred, pred_sigma=sigma, gt_xy=gt)
    assert torch.isfinite(loss).all()
    assert float(loss.detach()[2]) == 0.0  # non-finite row filtered
    assert rle(pred_xy=torch.zeros(0, 2), pred_sigma=torch.zeros(0), gt_xy=torch.zeros(0, 2)).shape == (0,)


# ----------------------------------------------------------------------------- schedules
def test_progloss_weights_and_root_prior_schedule():
    sched = ProgLossSchedule(o2m_start=0.8, o2m_end=0.1)
    w_o2m, w_o2o = sched.weights(0.0)
    assert abs(w_o2m - 0.8) < 1e-9 and abs(w_o2o - 0.2) < 1e-9
    w_o2m, w_o2o = sched.weights(1.0)
    assert abs(w_o2m - 0.1) < 1e-9 and abs(w_o2o - 0.9) < 1e-9
    w_o2m, w_o2o = sched.weights(0.5)
    assert abs(w_o2m - 0.45) < 1e-9 and abs(w_o2o - 0.55) < 1e-9

    crit = make_criterion()
    crit.set_schedule(10)
    assert crit.root_warmup_epochs == 3
    crit.set_progress(0.0)
    assert crit.root_prior_scale() == 1.0
    crit.set_progress(0.15)  # epoch 1.5 of 10 -> half way through the 3-epoch warm-up
    assert abs(crit.root_prior_scale() - 0.5) < 1e-6
    crit.set_progress(0.3)
    assert crit.root_prior_scale() == 0.0
    crit.set_progress(1.0)
    assert crit.root_prior_scale() == 0.0
    w_o2m, w_o2o = crit.current_weights()
    assert abs(w_o2m - 0.1) < 1e-9 and abs(w_o2o - 0.9) < 1e-9
    crit.set_schedule(0)  # unknown schedule -> no prior rather than a stuck prior
    crit.set_progress(0.0)
    assert crit.root_prior_scale() == 0.0


def test_from_config_reads_u_json():
    cfg = default_config()
    crit = XTX4Loss.from_config(cfg)
    assert crit.epochs == cfg["training"]["epochs"]
    assert crit.assigner_o2m.topk == 10 and crit.assigner_o2m.topk2 is None
    assert crit.assigner_o2o.topk == 7 and crit.assigner_o2o.topk2 == 1
    assert crit.kpt_loss == "soks" and crit.rle.convention == "reference" and crit.person_loss == "bce"
    assert crit.kpt_weight == 15.0 and abs(crit.rle_weight - 0.0417) < 1e-9
    assert any(n.startswith("rle.flow") for n, _ in crit.named_parameters())
    assert "sigmas" not in crit.state_dict()  # non-persistent buffer


# ----------------------------------------------------------------------------- forward semantics
def make_output(person, kpt, anchors, strides, *, dual: bool = True) -> HeadOutput:
    raw = RawPredictions(person=person, kpt=kpt, anchors=anchors, strides=strides)
    return HeadOutput(one2one=raw, one2many=raw if dual else None)


def test_forward_keys_and_finite_gradients():
    anchors, strides = make_anchors()
    person, kpt = random_raw(2, anchors.shape[0], seed=1, requires_grad=True)
    crit = make_criterion()
    crit.set_schedule(300)
    out = crit(make_output(person, kpt, anchors, strides), two_person_targets(), S)
    for key in ("total", "person", "kpt", "present", "visible", "oks", "rle", "w_o2m", "w_o2o", "coverage", "pos_per_gt"):
        assert key in out, key
        assert torch.isfinite(out[key]).all(), key
    assert out["total"].requires_grad and not out["person"].requires_grad
    assert out["total"].dtype == torch.float32
    assert abs(float(out["w_o2m"]) - 0.8) < 1e-6 and abs(float(out["w_o2o"]) - 0.2) < 1e-6
    assert float(out["coverage"]) == 1.0 and float(out["pos_per_gt"]) > 1.0
    assert float(out["root_prior_scale"]) == 1.0
    out["total"].backward()
    assert torch.isfinite(person.grad).all() and torch.isfinite(kpt.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in crit.parameters())
    # eval-mode output (one2many=None) still works and carries only the o2o head
    out_eval = crit(make_output(person, kpt, anchors, strides, dual=False), two_person_targets(), S)
    assert float(out_eval["w_o2m"]) == 0.0 and torch.isfinite(out_eval["total"])


def test_zero_gt_batch_trains_background_only():
    anchors, strides = make_anchors()
    person, kpt = random_raw(2, anchors.shape[0], seed=2, requires_grad=True)
    crit = make_criterion()
    targets = [{"keypoints": torch.zeros(0, K, 3)}, {"keypoints": torch.zeros(0, K, 3)}]
    out = crit(make_output(person, kpt, anchors, strides), targets, S)
    assert torch.isfinite(out["total"])
    assert float(out["pos_per_gt"]) == 0.0 and float(out["o2m_num_pos"]) == 0.0
    assert float(out["kpt"]) == 0.0 and float(out["present"]) == 0.0 and float(out["visible"]) == 0.0
    assert float(out["person"]) > 0.0
    out["total"].backward()
    assert torch.all(person.grad > 0)  # pushes every person logit down
    assert kpt.grad is None or float(kpt.grad.abs().sum()) == 0.0  # no graph reaches the keypoints


def test_unknown_visibility_contributes_nothing():
    anchors, strides = make_anchors()
    person, kpt = random_raw(1, anchors.shape[0], seed=3)
    crit = make_criterion()
    base = person_kpts(160, 160, 80)
    base[0, 2] = -1.0  # nose unknown
    base[3, 2] = 0.0  # left ear not labeled
    targets = [{"keypoints": base.unsqueeze(0)}]

    def total_of(kpt_tensor):
        out = crit(make_output(person, kpt_tensor, anchors, strides), targets, S)
        return out["total"], out

    ref_total, ref_out = total_of(kpt)
    # perturb the unknown point's prediction everywhere: xy, sigma, present, visible
    kpt_mod = kpt.clone()
    kpt_mod[..., 0, :] += torch.tensor([3.0, -2.0, 1.5, 5.0, -5.0])
    mod_total, _ = total_of(kpt_mod)
    assert torch.allclose(ref_total, mod_total)
    # the v=0 point: xy / visible are ignored, present trains with the configured weight
    kpt_mod = kpt.clone()
    kpt_mod[..., 3, :2] += 4.0
    kpt_mod[..., 3, 4] += 4.0
    assert torch.allclose(total_of(kpt_mod)[0], ref_total)
    kpt_mod = kpt.clone()
    kpt_mod[..., 3, 3] += 4.0
    assert not torch.allclose(total_of(kpt_mod)[0], ref_total)
    crit_w0 = make_criterion(unlabeled_present_weight=0.0)
    a = crit_w0(make_output(person, kpt, anchors, strides), targets, S)["present"]
    b = crit_w0(make_output(person, kpt_mod, anchors, strides), targets, S)["present"]
    assert torch.allclose(a, b)
    # gradients on unknown points are exactly zero in every channel
    kpt_g = kpt.clone().requires_grad_(True)
    total_of(kpt_g)[0].backward()
    assert float(kpt_g.grad[..., 0, :].abs().sum()) == 0.0
    assert float(kpt_g.grad[..., 3, [0, 1, 2, 4]].abs().sum()) == 0.0
    assert float(kpt_g.grad[..., 3, 3].abs().sum()) > 0.0


def test_present_visible_targets_follow_visibility_table():
    anchors, strides = make_anchors()
    num_anchors = anchors.shape[0]
    base = person_kpts(160, 160, 80)
    base[:, 2] = torch.tensor([2, 2, 2, 1, 1, 0, 0, -1, -1], dtype=torch.float32)
    targets = [{"keypoints": base.unsqueeze(0)}]
    crit = make_criterion()
    person = torch.full((1, num_anchors, 1), -4.0)
    kpt = torch.zeros(1, num_anchors, K, 5)
    # correct logits: present where v>0, visible where v==2
    kpt[..., 3] = torch.where(base[:, 2] > 0, 8.0, -8.0)
    kpt[..., 4] = torch.where(base[:, 2] == 2, 8.0, -8.0)
    good = crit(make_output(person, kpt, anchors, strides), targets, S)
    kpt_bad = kpt.clone()
    kpt_bad[..., 3:5] *= -1
    bad = crit(make_output(person, kpt_bad, anchors, strides), targets, S)
    assert float(good["present"]) < 1e-3 and float(good["visible"]) < 1e-3
    assert float(bad["present"]) > 5.0 and float(bad["visible"]) > 5.0


def test_person_loss_variants_and_normalisation():
    anchors, strides = make_anchors()
    person, kpt = random_raw(2, anchors.shape[0], seed=4)
    out_bce = make_criterion(person_loss="bce")(make_output(person, kpt, anchors, strides), two_person_targets(), S)
    out_vfl = make_criterion(person_loss="vfl")(make_output(person, kpt, anchors, strides), two_person_targets(), S)
    assert torch.isfinite(out_bce["person"]) and torch.isfinite(out_vfl["person"])
    assert float(out_vfl["person"]) < float(out_bce["person"])  # VFL down-weights easy negatives
    out_oks = make_criterion(kpt_loss="oks")(make_output(person, kpt, anchors, strides), two_person_targets(), S)
    assert float(out_oks["oks"]) >= float(out_bce["oks"])  # plain OKS saturates faster than SOKS


def test_fp32_under_autocast():
    anchors, strides = make_anchors()
    person, kpt = random_raw(2, anchors.shape[0], seed=5, requires_grad=True)
    crit = make_criterion()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        p16 = (person * 1.0).to(torch.bfloat16)
        k16 = (kpt * 1.0).to(torch.bfloat16)
        out = crit(make_output(p16, k16, anchors, strides), two_person_targets(), S)
    assert out["total"].dtype == torch.float32 and torch.isfinite(out["total"])
    out["total"].backward()
    assert torch.isfinite(person.grad).all() and torch.isfinite(kpt.grad).all()


def test_loss_decreases_with_learnable_raw_tensor():
    anchors, strides = make_anchors()
    num_anchors = anchors.shape[0]
    torch.manual_seed(0)
    person = nn.Parameter(torch.full((1, num_anchors, 1), -4.6) + 0.01 * torch.randn(1, num_anchors, 1))
    kpt = nn.Parameter(torch.zeros(1, num_anchors, K, 5) + 0.01 * torch.randn(1, num_anchors, K, 5))
    with torch.no_grad():
        kpt[..., 2] = 0.6
    targets = [{"keypoints": torch.stack([person_kpts(80, 90, 60), person_kpts(230, 200, 90)])}]
    crit = make_criterion()
    crit.set_schedule(300)
    crit.set_progress(0.02)  # warm-up prior active (epoch 6 of 300 -> 0)
    opt = torch.optim.Adam([person, kpt, *crit.parameters()], lr=0.1)
    history, oks_hist = [], []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        out = crit(make_output(person, kpt, anchors, strides), targets, S)
        out["total"].backward()
        assert torch.isfinite(out["total"])
        opt.step()
        history.append(float(out["total"].detach()))
        oks_hist.append(float(out["oks"]))
    assert history[-1] < 0.7 * history[0], history
    assert min(history[-5:]) < min(history[:5])
    assert oks_hist[-1] < 0.8 * oks_hist[0], oks_hist
    final = crit(make_output(person, kpt, anchors, strides), targets, S)
    assert float(final["coverage"]) == 1.0


def test_pad_targets_shapes_and_defaults():
    padded = pad_targets(
        [
            {"keypoints": torch.zeros(0, K, 3)},
            {"keypoints": person_kpts(10, 10, 5).unsqueeze(0), "area": torch.tensor([42.0]), "boxes": torch.tensor([[0.0, 0.0, 5.0, 5.0]])},
            {"keypoints": torch.stack([person_kpts(50, 50, 5), person_kpts(90, 90, 5)])},
        ],
        "cpu",
    )
    assert padded.kpts.shape == (3, 2, K, 3) and padded.area.shape == (3, 2) and padded.boxes.shape == (3, 2, 4)
    assert padded.mask.tolist() == [[False, False], [True, False], [True, True]]
    assert float(padded.area[1, 0]) == 42.0 and torch.isnan(padded.area[2]).all()
    assert torch.isnan(padded.boxes[2]).all() and float(padded.boxes[1, 0, 2]) == 5.0
    empty = pad_targets([{"keypoints": torch.zeros(0, K, 3)}], "cpu")
    assert empty.kpts.shape == (1, 1, K, 3) and not empty.mask.any()


class _ToyHead(nn.Module):
    """A 'model' whose forward returns a HeadOutput from learnable raw tensors."""

    def __init__(self, num_anchors: int, anchors: torch.Tensor, strides: torch.Tensor) -> None:
        super().__init__()
        self.head_person = nn.Parameter(torch.full((1, num_anchors, 1), -4.6))
        self.head_kpt = nn.Parameter(torch.zeros(1, num_anchors, K, 5))
        self.backbone_dummy = nn.Parameter(torch.zeros(1))
        self.register_buffer("anchors", anchors)
        self.register_buffer("strides", strides)

    def forward(self, images: torch.Tensor) -> HeadOutput:
        b = images.shape[0]
        person = self.head_person.expand(b, -1, -1) + 0.0 * self.backbone_dummy
        kpt = self.head_kpt.expand(b, -1, -1, -1)
        raw = RawPredictions(person=person, kpt=kpt, anchors=self.anchors, strides=self.strides)
        return HeadOutput(one2one=raw, one2many=raw)


def test_calibrate_loss_weights(tmp_path: Path):
    anchors, strides = make_anchors()
    model = _ToyHead(anchors.shape[0], anchors, strides)
    crit = make_criterion()
    batches = [(torch.zeros(2, 3, S, S), two_person_targets(), None) for _ in range(3)]
    result = calibrate_loss_weights(model, crit, batches, steps=2)
    assert result["steps"] == 2
    for key in ("person", "oks", "rle", "present", "visible"):
        assert result["grad_norms"][key] > 0.0
        assert math.isfinite(result["proposed"][key])
    eff = [v for v in result["proposed_effective"].values() if v > 0]
    assert max(eff) / min(eff) <= 3.0 + 1e-6
    cfg = result["proposed_config"]
    assert set(cfg) == {"person", "kpt", "oks_weight", "rle_weight", "present", "visible"}
    assert crit.kpt_weight == 15.0  # the criterion itself is untouched
    path = write_calibration_json(result, tmp_path / "cal.json")
    assert json.loads(path.read_text(encoding="utf-8"))["steps"] == 2
    # terms with a zero gradient keep their lambda
    proposed = propose_lambdas({"a": 1.0, "b": 100.0, "c": 0.0}, {"a": 1.0, "b": 1.0, "c": 7.0})
    assert proposed["c"] == 7.0 and proposed["b"] * 100.0 / (proposed["a"] * 1.0) <= 3.0 + 1e-9


def test_assigner_objects_are_reused_per_head():
    crit = make_criterion()
    assert isinstance(crit.assigner_o2m, PoseTAL) and isinstance(crit.assigner_o2o, PoseTAL)
    assert crit.assigner_o2m is not crit.assigner_o2o


# ----------------------------------------------------------------------------- frozen reference / o2o options
REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "loss_reference_default.pt"


def reference_case(seed: int = 1234, batch: int = 3, max_persons: int = 5):
    """The fixed-seed case behind ``tests/data/loss_reference_default.pt`` (generated by the
    pre-optimisation implementation; the generator order must not change)."""

    anchors, strides = make_anchors()
    num_anchors = anchors.shape[0]
    g = torch.Generator().manual_seed(seed)
    person = torch.randn(batch, num_anchors, 1, generator=g) * 1.5 - 2.0
    kpt = torch.randn(batch, num_anchors, K, 5, generator=g) * 0.5
    person_o2m = torch.randn(batch, num_anchors, 1, generator=g) * 1.5 - 2.0
    kpt_o2m = torch.randn(batch, num_anchors, K, 5, generator=g) * 0.5
    targets = []
    counts = [max_persons, 0, 2]
    for b in range(batch):
        m = counts[b]
        k = torch.zeros(m, K, 3)
        c = torch.rand(m, 2, generator=g) * S
        span = torch.rand(m, 1, generator=g) * 90 + 6
        off = torch.randn(m, K, 2, generator=g) * 0.3
        k[..., :2] = c[:, None, :] + off * span[:, :, None]
        k[..., 2] = torch.randint(-1, 3, (m, K), generator=g).float()
        area = torch.rand(m, generator=g) * 5000 + 100
        area[::2] = float("nan")
        targets.append({"keypoints": k, "area": area})
    out = HeadOutput(
        one2one=RawPredictions(person, kpt, anchors, strides),
        one2many=RawPredictions(person_o2m, kpt_o2m, anchors, strides),
    )
    return out, targets


def test_default_path_matches_frozen_reference():
    ref = torch.load(REFERENCE_PATH)
    out, targets = reference_case(seed=ref["seed"])
    crit = XTX4Loss(epochs=300)  # constructor defaults == configs/u.json loss block
    crit.rle.load_state_dict(ref["rle_state"])
    exact_keys = ("coverage", "pos_per_gt", "gt_total", "small_gt", "o2o_num_pos", "o2m_num_pos", "w_o2m", "w_o2o")
    for progress, row in ref["rows"].items():
        crit.set_progress(progress)
        result = crit(out, targets, S)
        for key, expected in row.items():
            assert key in result, key
            got = result[key].detach()
            if key in exact_keys:
                assert torch.equal(got, expected), (progress, key, got, expected)
            else:
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6, msg=f"{progress} {key}")
        assert float(result["o2o_boosted"]) == 0.0  # boost off by default


def test_from_config_reads_o2o_options():
    cfg = default_config()
    crit = XTX4Loss.from_config(cfg)
    assert crit.o2o_negative_boost == 1.0 and crit.o2o_negative_min_prob == 0.1 and crit.o2o_single_stage is False
    assert crit.assigner_o2o.topk == 7 and crit.assigner_o2o.topk2 == 1
    cfg["loss"].update({"o2o_negative_boost": 3.0, "o2o_negative_min_prob": 0.2, "o2o_single_stage": True})
    crit = XTX4Loss.from_config(cfg)
    assert crit.o2o_negative_boost == 3.0 and crit.o2o_negative_min_prob == 0.2 and crit.o2o_single_stage is True
    assert crit.assigner_o2o.topk == 1 and crit.assigner_o2o.topk2 is None


def test_o2o_negative_boost_raises_only_the_o2o_person_term():
    anchors, strides = make_anchors()
    person, kpt = random_raw(1, anchors.shape[0], seed=7)
    gt = person_kpts(160, 160, 80)
    targets = [{"keypoints": gt.unsqueeze(0)}]
    # duplicate-like anchors: every anchor inside the extent box fires with a confident logit
    inside = (anchors[:, 0] > 110) & (anchors[:, 0] < 210) & (anchors[:, 1] > 110) & (anchors[:, 1] < 210)
    person = person.clone()
    person[0, inside, 0] = 2.0
    output = make_output(person, kpt, anchors, strides)

    def seeded_criterion(**overrides) -> XTX4Loss:  # identical RLE flow init for every variant
        torch.manual_seed(0)
        return make_criterion(**overrides)

    base = seeded_criterion()(output, targets, S)
    boosted = seeded_criterion(o2o_negative_boost=3.0)(output, targets, S)
    assert float(boosted["o2o_boosted"]) > 0
    assert float(boosted["o2o_boosted"]) < float(inside.sum())  # the selected positive is never boosted
    assert float(boosted["o2o_person"]) > float(base["o2o_person"]) * 1.1
    for key in ("o2o_oks", "o2o_rle", "o2o_present", "o2o_visible", "o2o_num_pos", "o2m_person", "o2m_oks", "o2m_rle"):
        assert torch.equal(boosted[key], base[key]), key
    # boost < 1 lowers the term; min_prob above every probability disables it
    lowered = seeded_criterion(o2o_negative_boost=0.5)(output, targets, S)
    assert float(lowered["o2o_person"]) < float(base["o2o_person"])
    off = seeded_criterion(o2o_negative_boost=3.0, o2o_negative_min_prob=0.999)(output, targets, S)
    assert float(off["o2o_boosted"]) == 0.0 and torch.equal(off["o2o_person"], base["o2o_person"])
    # the boost also applies to the VFL variant and keeps gradients finite
    vfl_base = seeded_criterion(person_loss="vfl")(output, targets, S)
    vfl_boost = seeded_criterion(person_loss="vfl", o2o_negative_boost=3.0)(output, targets, S)
    assert float(vfl_boost["o2o_person"]) > float(vfl_base["o2o_person"])
    person_g = person.clone().requires_grad_(True)
    out_g = seeded_criterion(o2o_negative_boost=3.0)(make_output(person_g, kpt, anchors, strides), targets, S)
    out_g["total"].backward()
    assert torch.isfinite(person_g.grad).all()


def test_o2o_single_stage_assigns_at_most_one_anchor_per_gt():
    anchors, strides = make_anchors()
    person, kpt = random_raw(2, anchors.shape[0], seed=8)
    output = make_output(person, kpt, anchors, strides)
    targets = two_person_targets()
    crit = make_criterion(o2o_single_stage=True)
    out = crit(output, targets, S)
    assert float(out["o2o_num_pos"]) <= float(out["gt_total"]) and float(out["o2o_pos_per_gt"]) <= 1.0
    assert float(out["o2o_coverage"]) == 1.0
    two_stage = make_criterion()(output, targets, S)
    assert float(two_stage["o2o_pos_per_gt"]) <= 1.0  # 7 -> 1 is still one anchor per GT
    assert float(out["o2m_pos_per_gt"]) == float(two_stage["o2m_pos_per_gt"])  # the o2m head is untouched


def test_pad_targets_partial_area_and_boxes():
    padded = pad_targets(
        [
            {"keypoints": torch.ones(2, K, 3), "area": torch.tensor([10.0, 20.0]), "boxes": torch.arange(8.0).reshape(2, 4)},
            {"keypoints": torch.zeros(0, K, 3)},
            {"keypoints": torch.full((1, K, 3), 2.0), "area": torch.tensor([1.0, 2.0])},  # wrong size -> NaN
            {"keypoints": torch.full((3, K, 3), 3.0), "boxes": torch.ones(3, 4)},
        ],
        "cpu",
    )
    assert padded.kpts.shape == (4, 3, K, 3) and padded.mask.sum() == 6
    assert torch.equal(padded.area[0, :2], torch.tensor([10.0, 20.0])) and torch.isnan(padded.area[0, 2])
    assert torch.isnan(padded.area[2, 0]) and torch.isnan(padded.area[3]).all()
    assert torch.equal(padded.boxes[0, :2], torch.arange(8.0).reshape(2, 4)) and torch.isnan(padded.boxes[2, 0]).all()
    assert torch.equal(padded.boxes[3], torch.ones(3, 4)) and torch.isnan(padded.boxes[1]).all()
    assert float(padded.kpts[3, 2, 0, 0]) == 3.0 and float(padded.kpts[2, 0, 0, 0]) == 2.0
    assert not padded.mask[1].any() and not padded.mask[2, 1:].any()
