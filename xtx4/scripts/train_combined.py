"""Train several XTX4 variants one after another, resuming automatically after a crash.

Used by ``train_combined.bat`` (default: ``u_b`` then ``u_c``, 300 epochs each). Every
start of this script works out, per variant, where to continue:

* the run is **finished** (``combined_done.json`` in its run directory, written after
  ``train.py`` exited successfully, i.e. after training, final evaluation and export)
  -> skipped;
* otherwise the **newest intact** ``epochs/epoch_NNNN.pt`` (= the last epoch that fully
  completed) -> ``train.py --resume`` continues with epoch ``NNNN + 1``; a file damaged
  by a power cut is detected (zip CRC check) and the next older one is used instead;
* no epoch file but an intact ``last.pt`` (old runs, ``keep_epochs = 0``) -> resume from it;
* nothing usable -> fresh start.

Variants run **sequentially** on purpose: two concurrent runs exhausted Windows virtual
memory and rebooted the PC, and were not faster in total (EXPERIMENTS.md, "Paralelní
trénink"). If ``train.py`` fails with an error (not Ctrl+C) it is retried from the last
completed epoch up to ``--retries`` times; after that the script stops so the problem gets
noticed instead of silently moving on to the next variant. After a PC crash just start
``train_combined.bat`` again.

    python scripts/train_combined.py                          # u_b then u_c, 300 epochs
    python scripts/train_combined.py --dry-run                # only print what would happen
    python scripts/train_combined.py --variants u_c --epochs 300 --set training.batch_size=48

Unknown options (e.g. ``--set key=value``, ``--no-thermal-guard``) are passed to ``train.py``.
This parent process deliberately does not import torch (keeps ~1 GB of memory free).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "train.py"
EPOCH_FILE_RE = re.compile(r"^epoch_(\d{4,})\.pt$")  # same convention as xtx4.engine.checkpoint
DONE_MARKER = "combined_done.json"
# Windows status for a console Ctrl+C / Ctrl+Break (STATUS_CONTROL_C_EXIT) and its signed form.
CTRL_C_EXIT = 3221225786
CTRL_C_CODES = {CTRL_C_EXIT, -1073741510}


def log(message: str, log_file: Path | None) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def checkpoint_is_intact(path: Path) -> bool:
    """``torch.save`` writes a zip archive; a file cut short by a crash fails the CRC check."""

    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError, ValueError):
        return False


def epoch_files(run_dir: Path) -> list[tuple[int, Path]]:
    """``(completed_epochs, path)`` of ``<run>/epochs/epoch_NNNN.pt``, newest epoch first."""

    epochs_dir = run_dir / "epochs"
    if not epochs_dir.is_dir():
        return []
    found = []
    for path in epochs_dir.iterdir():
        match = EPOCH_FILE_RE.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found, reverse=True)


def resume_point(run_dir: Path, log_file: Path | None) -> tuple[int, Path | None]:
    """Return ``(completed_epochs, checkpoint)``; ``(0, None)`` means start from scratch."""

    for completed, path in epoch_files(run_dir):
        if checkpoint_is_intact(path):
            return completed, path
        log(f"  {path.name} is damaged (incomplete write?) - trying an older epoch", log_file)
    last = run_dir / "last.pt"
    if last.is_file():
        if checkpoint_is_intact(last):
            return -1, last  # epoch unknown without torch; train.py reads it from the file
        log("  last.pt is damaged - ignoring it", log_file)
    return 0, None


def run_variant(variant: str, args: argparse.Namespace, passthrough: list[str], log_file: Path) -> bool:
    """Train one variant to completion (with resume/retry). Returns False if it should stop."""

    run_dir = Path(args.runs_root) / f"{args.prefix}_{variant}_{args.epochs}ep"
    done = run_dir / DONE_MARKER
    if done.is_file():
        log(f"[{variant}] already finished ({run_dir}) - skipping", log_file)
        return True

    attempt = 0
    while True:
        completed, checkpoint = resume_point(run_dir, log_file)
        cmd = [sys.executable, str(TRAIN_PY), "--variant", variant, "--epochs", str(args.epochs), "--output", str(run_dir)]
        if args.config:
            cmd += ["--config", args.config]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if checkpoint is not None:
            cmd += ["--resume", str(checkpoint)]
            if completed < 0:
                where = "the epoch stored in last.pt"
            elif completed >= args.epochs:
                where = "the final evaluation and export (all epochs done)"
            else:
                where = f"epoch {completed + 1}/{args.epochs}"
            log(f"[{variant}] resuming from {checkpoint} -> continuing with {where}", log_file)
        else:
            log(f"[{variant}] starting from scratch -> {run_dir}", log_file)
        cmd += passthrough
        log(f"[{variant}] command: {subprocess.list2cmdline(cmd)}", log_file)
        if args.dry_run:
            return True

        started = time.time()
        try:
            rc = subprocess.call(cmd, cwd=str(ROOT))
        except KeyboardInterrupt:  # subprocess.call kills the child before re-raising
            rc = CTRL_C_EXIT
        hours = (time.time() - started) / 3600.0
        if rc == 0:
            done.write_text(
                json.dumps({"variant": variant, "epochs": args.epochs, "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}"},
                           indent=2),
                encoding="utf-8",
            )
            log(f"[{variant}] finished after {hours:.1f} h of this session -> {run_dir}", log_file)
            return True
        if rc in CTRL_C_CODES:
            log(f"[{variant}] interrupted by the user (Ctrl+C) - start train_combined.bat again to continue", log_file)
            return False
        attempt += 1
        if attempt > args.retries:
            log(f"[{variant}] train.py failed with exit code {rc} ({attempt} attempts) - stopping; "
                f"see {run_dir / 'train.log'}. Start train_combined.bat again to continue.", log_file)
            return False
        log(f"[{variant}] train.py failed with exit code {rc} after {hours:.1f} h - retry {attempt}/{args.retries} "
            f"in {args.retry_wait_s} s from the last completed epoch", log_file)
        time.sleep(args.retry_wait_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["u_b", "u_c"], help="Variants in training order")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs-root", default="D:/xtx4_runs", help="Parent directory of the run directories")
    parser.add_argument("--prefix", default="f6", help="Run directory = <runs-root>/<prefix>_<variant>_<epochs>ep")
    parser.add_argument("--config", default=None, help="Config JSON for train.py (default configs/u.json)")
    parser.add_argument("--dataset", default=None, help="Dataset root for train.py (default from the config)")
    parser.add_argument("--retries", type=int, default=2, help="Automatic retries after a train.py error")
    parser.add_argument("--retry-wait-s", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan")
    args, passthrough = parser.parse_known_args()

    log_file = None if args.dry_run else Path(args.runs_root) / f"{args.prefix}_combined.log"
    log(f"train_combined: variants {' -> '.join(args.variants)}, {args.epochs} epochs each, runs in {args.runs_root}"
        + (" (dry run)" if args.dry_run else ""), log_file)
    for variant in args.variants:
        if not run_variant(variant, args, passthrough, log_file):
            return 1
    log("train_combined: all variants finished" if not args.dry_run else "train_combined: dry run done", log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
