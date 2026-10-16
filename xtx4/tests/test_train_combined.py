"""Resume logic of scripts/train_combined.py (no training; checkpoints are small torch.save files)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_combined.py"

_spec = importlib.util.spec_from_file_location("train_combined", SCRIPT)
tc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tc)


def _save(path: Path, epoch: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "weights": torch.randn(64, 64)}, path)
    return path


def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # what a power cut during the write leaves behind


def test_fresh_run_starts_from_scratch(tmp_path: Path) -> None:
    assert tc.resume_point(tmp_path / "run", None) == (0, None)


def test_newest_intact_epoch_file_wins(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for n in (7, 8, 9):
        _save(run / "epochs" / f"epoch_{n:04d}.pt", n - 1)
    _save(run / "last.pt", 8)
    completed, ckpt = tc.resume_point(run, None)
    assert (completed, ckpt.name) == (9, "epoch_0009.pt")


def test_damaged_newest_epoch_falls_back_to_older(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for n in (7, 8, 9):
        _save(run / "epochs" / f"epoch_{n:04d}.pt", n - 1)
    _truncate(run / "epochs" / "epoch_0009.pt")
    assert not tc.checkpoint_is_intact(run / "epochs" / "epoch_0009.pt")
    completed, ckpt = tc.resume_point(run, None)
    assert (completed, ckpt.name) == (8, "epoch_0008.pt")


def test_last_pt_used_without_epoch_files_and_damaged_last_ignored(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _save(run / "last.pt", 4)
    assert tc.resume_point(run, None) == (-1, run / "last.pt")
    _truncate(run / "last.pt")
    assert tc.resume_point(run, None) == (0, None)


def test_numbering_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _save(run / "epochs" / "epoch_9999.pt", 9998)
    _save(run / "epochs" / "epoch_10000.pt", 9999)
    assert tc.resume_point(run, None)[0] == 10000


def test_dry_run_skips_finished_and_resumes_interrupted(tmp_path: Path) -> None:
    (tmp_path / "f6_u_b_300ep").mkdir()
    (tmp_path / "f6_u_b_300ep" / tc.DONE_MARKER).write_text("{}", encoding="utf-8")
    _save(tmp_path / "f6_u_c_300ep" / "epochs" / "epoch_0042.pt", 41)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--runs-root", str(tmp_path), "--variants", "u_b", "u_c"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "[u_b] already finished" in out
    assert "[u_c] resuming from" in out and "epoch_0042.pt" in out and "epoch 43/300" in out
    assert "--resume" in out
