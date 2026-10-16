"""End-to-end smoke test of the trainer on the XTX3 smoke dataset (CPU or CUDA).

Two epochs with ``configs/smoke.json`` into ``D:/xtx4_runs/_tests/<tmp>``: finite
losses, ``last.pt``/``best.pt`` in the CONTRACTS §11 format, a profile, and a
resume for one more epoch that continues ``global_step``. Auto-export stays off.
A second scenario covers the rolling ``epochs/epoch_NNNN.pt`` files (rotation by
``keep_epochs``, resume from an epoch file) and the thermal guard with a scripted
fake sensor (at least one pause, training still completes).
"""

from __future__ import annotations

import json
import math
import shutil
import threading
import uuid
from pathlib import Path

import pytest
import torch

from xtx4.engine.checkpoint import discover_checkpoints, list_epoch_checkpoints, load_checkpoint, run_dir_of_checkpoint
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
    config = _smoke_config(**{"training.epochs": 1, "training.profile_steps": 0, "training.num_workers": 0,
                              "training.keep_epochs": 0})
    trainer = Trainer(config, SMOKE_DATASET, run_dir / "a", device)
    trainer.train()
    assert not (run_dir / "a" / "epochs").exists(), "keep_epochs=0 must disable the rolling epoch files"
    child = Trainer(config, SMOKE_DATASET, run_dir / "b", device)
    child.load_init_weights(run_dir / "a" / "last.pt")
    assert child.parent_checkpoint_sha256 and len(child.parent_checkpoint_sha256) == 64
    assert child.global_step == 0 and child.ema.updates == 0
    parent_state = load_checkpoint(run_dir / "a" / "last.pt")["ema_state"]
    key = next(k for k in parent_state if k.endswith("weight"))
    torch.testing.assert_close(child.model.state_dict()[key].cpu(), parent_state[key].cpu())


class _ScriptedSensor:
    """Hot (90 C) for the first ``hot_samples`` readings, then cool (60 C)."""

    def __init__(self, hot_samples: int) -> None:
        self.hot_samples = hot_samples
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> dict:
        with self._lock:
            self.calls += 1
            return {"gpu_temp_c": 90.0 if self.calls <= self.hot_samples else 60.0, "hw_slowdown": False}


def _fast_config(**overrides) -> dict:
    """Smallest settings the smoke config allows: one size, 5 steps/epoch, tiny validation."""

    base = {
        "training.epochs": 3, "training.keep_epochs": 2, "training.batch_size": 24, "training.num_workers": 0,
        "training.profile_steps": 0, "training.val_images_per_epoch": 8, "training.full_val_every": 100,
        "training.close_mosaic_epochs": 1, "training.tensorboard.histogram_every_epochs": 100,
        "training.thermal.interval_s": 0.02, "training.thermal.max_pause_min": 60,
        "data.sizes": [256], "data.size_probs": [1.0], "data.eval_size": 256, "eval.sizes": [256],
    }
    base.update(overrides)
    return _smoke_config(**base)


@pytest.mark.skipif(not SMOKE_DATASET.is_dir(), reason="smoke dataset not available")
def test_epoch_checkpoints_rotation_resume_and_thermal_pause(run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs_dir = run_dir / "epochs"

    # (a) + (c): 3 epochs, keep_epochs=2, scripted sensor hot for the first samples.
    sensor = _ScriptedSensor(hot_samples=6)
    trainer = Trainer(_fast_config(), SMOKE_DATASET, run_dir, device, thermal_sensor=sensor)
    assert trainer.keep_epochs == 2 and trainer.steps_per_epoch == 120 // 24
    summary = trainer.train()
    guard = trainer.thermal
    assert guard is not None and guard.active and guard._thread is None, "sampling thread must be stopped"
    assert guard.pause_count >= 1 and guard.pause_seconds > 0.0 and summary["thermal_pauses"] >= 1
    assert not guard.hot and sensor.calls > sensor.hot_samples
    assert guard.max_gpu_temp_run == 90.0
    applied = 3 * trainer.steps_per_epoch - trainer._amp_skips
    assert trainer.global_step == applied, "training must complete after the pause"

    assert sorted(p.name for p in epochs_dir.iterdir()) == ["epoch_0002.pt", "epoch_0003.pt"]
    assert [p.name for p in list_epoch_checkpoints(run_dir)] == ["epoch_0003.pt", "epoch_0002.pt"]
    newest = load_checkpoint(epochs_dir / "epoch_0003.pt")
    last = load_checkpoint(run_dir / "last.pt")
    assert newest["epoch"] == 2 and last["epoch"] == newest["epoch"] and last["global_step"] == newest["global_step"]
    assert SECTION_11_KEYS <= set(newest)
    assert (run_dir / "last.pt").read_bytes() == (epochs_dir / "epoch_0003.pt").read_bytes()
    assert (run_dir / "best.pt").exists()
    second = load_checkpoint(epochs_dir / "epoch_0002.pt")
    assert second["epoch"] == 1 and second["metrics"]["best_ap"] <= last["metrics"]["best_ap"]
    found = discover_checkpoints(run_dir)
    assert {p.name for p in found} == {"last.pt", "best.pt", "epoch_0002.pt", "epoch_0003.pt"}
    assert run_dir_of_checkpoint(epochs_dir / "epoch_0002.pt") == run_dir

    # (b) resume from the epoch-2 file: continues with epoch 3, same best_ap bookkeeping as last.pt.
    stale_mtime = (epochs_dir / "epoch_0003.pt").stat().st_mtime
    trainer2 = Trainer(_fast_config(), SMOKE_DATASET, run_dir, device)  # no sensor -> real probe / inactive on CPU
    trainer2.resume(epochs_dir / "epoch_0002.pt")
    assert trainer2.start_epoch == 2 and trainer2.global_step == second["global_step"]
    assert trainer2.ema.updates == second["ema_updates"]
    assert trainer2.best_ap == second["metrics"]["best_ap"]
    totals = _record_losses(trainer2)
    trainer2.train()
    assert len(totals) == trainer2.steps_per_epoch and all(math.isfinite(t) for t in totals)
    assert trainer2.global_step == second["global_step"] + trainer2.steps_per_epoch - trainer2._amp_skips
    assert trainer2.thermal is not None and trainer2.thermal.pause_count == 0
    final = load_checkpoint(run_dir / "last.pt")
    assert final["epoch"] == 2 and final["global_step"] == trainer2.global_step
    # Rotation by mtime: the re-created epoch_0003.pt is the newest, the stale copy is gone, epoch_0002 survives.
    assert sorted(p.name for p in epochs_dir.iterdir()) == ["epoch_0002.pt", "epoch_0003.pt"]
    assert (epochs_dir / "epoch_0003.pt").stat().st_mtime > stale_mtime
    assert load_checkpoint(epochs_dir / "epoch_0003.pt")["global_step"] == trainer2.global_step
