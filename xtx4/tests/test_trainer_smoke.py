"""End-to-end smoke test of the trainer on the XTX3 smoke dataset (CPU or CUDA).

Two epochs with ``configs/smoke.json`` into ``D:/xtx4_runs/_tests/<tmp>``: finite
losses, ``last.pt``/``best.pt`` in the CONTRACTS §11 format, a profile, and a
resume for one more epoch that continues ``global_step``. Auto-export stays off.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

import pytest
import torch

from xtx4.engine.checkpoint import load_checkpoint
from xtx4.engine.trainer import Trainer, apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATASET = ROOT.parent / "xtx3" / ".smoke_dataset"
RUNS_ROOT = Path("D:/xtx4_runs/_tests") if Path("D:/").exists() else ROOT / "runs" / "_tests"

SECTION_11_KEYS = {
    "variant_spec", "model_state", "ema_state", "optimizer_state", "scaler_state", "criterion_state",
    "epoch", "global_step", "ema_updates", "rng_state", "config", "metrics", "keypoint_meta",
    "data_manifest_sha1", "parent_checkpoint_sha256",
}


@pytest.fixture()
def run_dir():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    path = RUNS_ROOT / f"smoke-{uuid.uuid4().hex[:8]}"
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _smoke_config(**overrides) -> dict:
    config = load_config(ROOT / "configs" / "smoke.json")
    config["data"]["dataset_path"] = str(SMOKE_DATASET)
    config["export"]["auto_export"] = False
    config["training"]["num_workers"] = 2
    config["training"]["tensorboard"]["graph"] = False
    apply_overrides(config, [f"{k}={json.dumps(v)}" for k, v in overrides.items()])
    return config


def _record_losses(trainer: Trainer) -> list[float]:
    totals: list[float] = []
    original = trainer.criterion.forward

    def forward(output, targets, image_size=None):
        out = original(output, targets, image_size)
        totals.append(float(out["total"].detach()))
        return out

    trainer.criterion.forward = forward  # type: ignore[method-assign]
    return totals


@pytest.mark.skipif(not SMOKE_DATASET.is_dir(), reason="smoke dataset not available")
def test_two_epochs_then_resume(run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _smoke_config()
    trainer = Trainer(config, SMOKE_DATASET, run_dir, device)
    assert trainer.output_dir == run_dir and (run_dir / "train.log").exists()
    assert len(trainer.val_subset_idx) == min(40, len(trainer.val_paths))
    steps_per_epoch = trainer.steps_per_epoch
    assert steps_per_epoch == 120 // 8

    totals = _record_losses(trainer)
    summary = trainer.train()

    assert totals and all(math.isfinite(t) for t in totals), "non-finite training loss"
    # GradScaler may skip a few early steps (overflow); EMA and global_step count applied steps only.
    skips = trainer._amp_skips
    applied = 2 * steps_per_epoch - skips
    assert skips < steps_per_epoch, f"too many skipped optimizer steps: {skips}"
    assert trainer.global_step == applied
    assert trainer.ema.updates == trainer.global_step
    assert summary["export"] is None  # auto_export disabled
    assert math.isfinite(summary["best_ap"]) and summary["best_ap"] >= 0.0
    assert (run_dir / "profile.json").exists()
    profile = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    assert set(profile["fractions"]) == {"data_wait", "h2d", "forward", "loss", "backward", "optimizer"}
    assert abs(sum(profile["fractions"].values()) - 1.0) < 1e-6
    assert (run_dir / "eval_report.json").exists()
    assert (run_dir / "tensorboard").is_dir()

    for name in ("last.pt", "best.pt"):
        ckpt = load_checkpoint(run_dir / name)
        assert SECTION_11_KEYS <= set(ckpt), f"{name} misses {SECTION_11_KEYS - set(ckpt)}"
        assert ckpt["variant_spec"]["name"] == config["model"]["variant"]
        assert ckpt["global_step"] == applied
        assert ckpt["ema_updates"] == applied
        assert ckpt["epoch"] == 1
        assert isinstance(ckpt["data_manifest_sha1"], str) and len(ckpt["data_manifest_sha1"]) == 40
        assert ckpt["parent_checkpoint_sha256"] is None
        assert ckpt["rng_state"]["torch"] is not None
        assert ckpt["keypoint_meta"]["names"][0] == "nose"
        assert math.isfinite(ckpt["metrics"]["AP"])
        assert any(k.startswith("rle.") for k in ckpt["criterion_state"]), "RLE flow params missing"

    # Resume for one more epoch: global_step continues, the epoch counter advances.
    config3 = _smoke_config(**{"training.epochs": 3})
    trainer2 = Trainer(config3, SMOKE_DATASET, run_dir, device)
    trainer2.resume(run_dir / "last.pt")
    assert trainer2.start_epoch == 2 and trainer2.global_step == applied
    assert trainer2.ema.updates == applied
    totals2 = _record_losses(trainer2)
    trainer2.train()
    assert len(totals2) == steps_per_epoch and all(math.isfinite(t) for t in totals2)
    expected = applied + steps_per_epoch - trainer2._amp_skips
    assert trainer2.global_step == expected and trainer2.global_step > applied
    last = load_checkpoint(run_dir / "last.pt")
    assert last["epoch"] == 2 and last["global_step"] == expected


@pytest.mark.skipif(not SMOKE_DATASET.is_dir(), reason="smoke dataset not available")
def test_init_weights_records_parent_hash(run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _smoke_config(**{"training.epochs": 1, "training.profile_steps": 0, "training.num_workers": 0})
    trainer = Trainer(config, SMOKE_DATASET, run_dir / "a", device)
    trainer.train()
    child = Trainer(config, SMOKE_DATASET, run_dir / "b", device)
    child.load_init_weights(run_dir / "a" / "last.pt")
    assert child.parent_checkpoint_sha256 and len(child.parent_checkpoint_sha256) == 64
    assert child.global_step == 0 and child.ema.updates == 0
    parent_state = load_checkpoint(run_dir / "a" / "last.pt")["ema_state"]
    key = next(k for k in parent_state if k.endswith("weight"))
    torch.testing.assert_close(child.model.state_dict()[key].cpu(), parent_state[key].cpu())
