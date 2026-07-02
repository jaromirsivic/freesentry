"""Training pipeline (spec section 9): console app with dual head, ProgLoss,
STAL assignment, MuSGD optimizer, EMA, AMP, warmup + cosine LR, time-based
checkpointing, per-epoch 3-level merged OKS validation, and merge calibration.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.annotations import dataset_statistics
from ..data.augment import AugmentConfig
from ..data.dataset import BalancedSampler, XTX2PoseDataset, collate_fn, denormalise_sample
from ..losses import XTX2Loss
from ..models import build_model
from ..utils.keypoints import KEYPOINT_NAMES
from ..utils.logging import get_logger
from .checkpoint import ModelEMA, build_checkpoint, discover_checkpoints, load_checkpoint, save_checkpoint
from .merge import MergeParams, calibrate_thresholds
from .metrics import ImageGroundTruth, ImagePredictions, evaluate
from .optim import build_optimizer

logger = get_logger(__name__)

VALID_VARIANTS = ("n", "m", "l")
DEFAULT_RUNS_DIR = Path("./runs")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load config.json from the app root, deep-merging a user config over it."""

    defaults_path = Path(__file__).resolve().parents[2] / "config.json"
    config: dict[str, Any] = {}
    if defaults_path.exists():
        config = json.loads(defaults_path.read_text(encoding="utf-8"))
    if config_path is not None and Path(config_path).exists() and Path(config_path) != defaults_path:
        config = _deep_update(config, json.loads(Path(config_path).read_text(encoding="utf-8")))
    return config


def prompt_resume(*, runs_dir: Path = DEFAULT_RUNS_DIR, provided: str | None = None) -> Path | None:
    """Resume prompt (section 9.1, asked first).

    Returns the chosen checkpoint path or None. Silently skipped when no
    checkpoint exists under ``runs_dir`` and none was provided on the CLI.
    """

    if provided:
        return Path(provided)
    checkpoints = discover_checkpoints(runs_dir)
    if not checkpoints:
        return None
    answer = input("Resume training from a checkpoint? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return None
    print("Discovered checkpoints:")
    for i, path in enumerate(checkpoints):
        print(f"  [{i}] {path}")
    while True:
        raw = input("Pick a number or type a checkpoint path: ").strip()
        if raw.isdigit() and int(raw) < len(checkpoints):
            return checkpoints[int(raw)]
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
        print("Not a valid selection; try again.")


def prompt_variant(provided: str | None) -> str:
    """Interactive variant prompt, validated, with CLI override."""

    if provided in VALID_VARIANTS:
        return provided
    while True:
        answer = input("Which model do you want to train? [n/m/l]: ").strip().lower()
        if answer in VALID_VARIANTS:
            return answer
        print("Please enter one of: n, m, l")


def prompt_dataset(provided: str | None) -> Path:
    """Interactive dataset-path prompt (Enter selects ./dataset), validated."""

    while True:
        if provided:
            path = Path(provided)
            provided = None
        else:
            raw = input("Dataset path [./dataset]: ").strip()
            path = Path(raw) if raw else Path("./dataset")
        if (path / "train").is_dir() and (path / "test").is_dir():
            return path
        print(f"Expected '{path}/train' and '{path}/test' to exist; try again.")


def prompt_output(provided: str | None, *, variant: str) -> Path:
    """Interactive checkpoint-directory prompt (Enter accepts the default)."""

    if provided:
        return Path(provided)
    default = f"./runs/xtx2-{variant}"
    raw = input(f"Checkpoint directory [{default}]: ").strip()
    return Path(raw) if raw else Path(default)


def count_gflops(model: torch.nn.Module, *, input_size: int = 384) -> float:
    """Count GMACs (~GFLOPs as commonly quoted) for one 384 forward via hooks."""

    total = 0.0
    handles = []

    def conv_hook(module: torch.nn.Conv2d, inputs, output) -> None:
        nonlocal total
        out_h, out_w = output.shape[-2:]
        kh, kw = module.kernel_size
        total += out_h * out_w * module.out_channels * (module.in_channels // module.groups) * kh * kw

    def linear_hook(module: torch.nn.Linear, inputs, output) -> None:
        nonlocal total
        total += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return total / 1e9


class Trainer:
    """Encapsulates the full training loop for one XTX2 variant."""

    def __init__(
        self,
        *,
        variant: str,
        dataset_path: Path,
        config: dict[str, Any],
        device: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.variant = variant
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir or f"runs/xtx2-{variant}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        self.epochs = int(train_cfg.get("epochs", 50))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.optimizer_name = str(train_cfg.get("optimizer", "musgd"))
        self.lr0 = float(train_cfg.get("lr0", 0.01))
        self.lr_final_factor = float(train_cfg.get("lr_final_factor", 0.01))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0005))
        self.momentum = float(train_cfg.get("momentum", 0.937))
        self.use_amp = bool(train_cfg.get("amp", True)) and self.device == "cuda"
        self.num_workers = int(train_cfg.get("num_workers", 8))
        self.ckpt_interval_min = float(train_cfg.get("checkpoint_interval_minutes", 5))
        self.close_mosaic_epochs = int(data_cfg.get("close_mosaic_epochs", 10))
        self.level_ratio = tuple(data_cfg.get("level_ratio", [1, 1, 1]))
        self.max_val_images = int(config.get("validation", {}).get("max_images", 200))

        self._build()

    def _build(self) -> None:
        data_cfg = self.config.get("data", {})
        augment_cfg = AugmentConfig.from_dict(data_cfg.get("augment"))
        self.train_dataset = XTX2PoseDataset(
            split_dir=self.dataset_path / "train",
            augment_config=augment_cfg,
            augment=True,
            min_visible_keypoints=int(data_cfg.get("min_visible_keypoints", 1)),
        )
        self.test_dataset = XTX2PoseDataset(
            split_dir=self.dataset_path / "test",
            augment=False,
            min_visible_keypoints=int(data_cfg.get("min_visible_keypoints", 1)),
        )
        self.sampler = BalancedSampler(
            index=self.train_dataset.index,
            level_ratio=self.level_ratio,
            seed=int(self.config.get("training", {}).get("seed", 0)),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=(self.device == "cuda"),
            drop_last=True,
        )

        self._log_dataset_stats()

        self.model = build_model(variant=self.variant, config=self.config).to(self.device)
        self.criterion = XTX2Loss.from_config(self.config).to(self.device)
        self.ema = ModelEMA(self.model, decay=float(self.config["training"].get("ema_decay", 0.9999)))
        self.optimizer = build_optimizer(
            [self.model, self.criterion],
            name=self.optimizer_name,
            lr=self.lr0,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        self.global_step = 0
        self.best_ap = -1.0

        params = self.model.num_parameters() / 1e6
        gflops = count_gflops(self.model)
        logger.info(
            "Built XTX2-%s (%.2fM params, %.2f GFLOPs/level, %.2f GFLOPs 3-level) "
            "optimizer=%s on %s | train=%d imgs test=%d imgs",
            self.variant, params, gflops, gflops * 3, self.optimizer_name, self.device,
            len(self.train_dataset.samples), len(self.test_dataset.samples),
        )

    def _log_dataset_stats(self) -> None:
        for name, dataset in (("train", self.train_dataset), ("test", self.test_dataset)):
            stats = dataset_statistics(dataset.samples)
            logger.info(
                "%s split: %d images, %d people", name, stats["num_images"], stats["num_people"]
            )
            hist = stats["visibility_histogram"]
            for k, kpt_name in enumerate(KEYPOINT_NAMES):
                logger.info(
                    "  %-14s vis0=%-6d vis1=%-6d vis2=%-6d",
                    kpt_name, hist[k, 0], hist[k, 1], hist[k, 2],
                )
            level_counts = dataset.per_level_target_counts()
            logger.info(
                "  per-level targets: level_1=%d level_2=%d level_3=%d",
                level_counts[1], level_counts[2], level_counts[3],
            )

    def _lr_at(self, epoch_float: float) -> float:
        if epoch_float < self.warmup_epochs:
            return self.lr0 * (epoch_float / max(self.warmup_epochs, 1e-9))
        progress = (epoch_float - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return self.lr0 * (self.lr_final_factor + (1 - self.lr_final_factor) * cosine)

    def resume(self, checkpoint_path: Path) -> None:
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("ema_state"):
            self.ema.ema.load_state_dict(ckpt["ema_state"])
        if ckpt.get("optimizer_state"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if ckpt.get("criterion_state"):
            self.criterion.load_state_dict(ckpt["criterion_state"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_ap = float(ckpt.get("metrics", {}).get("oks_map", -1.0))
        # Restore the ProgLoss schedule position (section 9.2 resume contract).
        self.criterion.set_progress(self.start_epoch / max(self.epochs, 1))
        logger.info("Resumed from %s at epoch %d", checkpoint_path, self.start_epoch)

    def train(self) -> dict[str, Any]:
        last_ckpt_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            if epoch >= self.epochs - self.close_mosaic_epochs:
                self.train_dataset.set_mosaic(enabled=False)
            self.sampler.set_epoch(epoch)
            last_ckpt_time = self._train_epoch(epoch, last_ckpt_time)

            metrics = self.validate()
            logger.info(
                "Epoch %d/%d val: AP=%.4f AP50=%.4f AP75=%.4f AR=%.4f",
                epoch + 1, self.epochs, metrics["AP"], metrics["AP50"], metrics["AP75"], metrics["AR"],
            )
            self._save("last.pt", epoch=epoch, metrics={"oks_map": metrics["AP"], **metrics})
            if metrics["AP"] > self.best_ap:
                self.best_ap = metrics["AP"]
                self._save("best.pt", epoch=epoch, metrics={"oks_map": metrics["AP"], **metrics})

        merge_params = self.calibrate()
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            # Embed calibrated merge params into best.pt without touching weights.
            ckpt = load_checkpoint(best_path, map_location="cpu")
            ckpt["merge_params"] = merge_params.to_dict()
            save_checkpoint(ckpt, best_path)
        else:
            self._save(
                "best.pt",
                epoch=self.epochs - 1,
                metrics={"oks_map": self.best_ap},
                merge_params=merge_params.to_dict(),
            )
        self._mirror_merge_params(merge_params)
        return {"best_ap": self.best_ap, "merge_params": merge_params.to_dict()}

    def _train_epoch(self, epoch: int, last_ckpt_time: float) -> float:
        self.model.train()
        num_batches = len(self.train_loader)
        epoch_start = time.time()

        for step, (images, targets, _levels, _metas) in enumerate(self.train_loader):
            epoch_float = epoch + step / max(num_batches, 1)
            lr = self._lr_at(epoch_float)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            # ProgLoss: shift head emphasis linearly over global progress.
            self.criterion.set_progress(epoch_float / max(self.epochs, 1))

            images = images.to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets
            ]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(images)
                losses = self.criterion(output, targets)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)
            self.global_step += 1

            if step % 10 == 0 or step == num_batches - 1:
                elapsed = max(time.time() - epoch_start, 1e-6)
                imgs_per_s = (step + 1) * self.batch_size / elapsed
                eta = (num_batches - step - 1) * elapsed / max(step + 1, 1)
                w_o2m, w_o2o = self.criterion.current_weights()
                logged = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                          for k, v in losses.items()}
                logger.info(
                    "E%d/%d S%d/%d | total=%.3f box=%.3f cls=%.3f kpt=%.3f vis=%.3f | "
                    "prog w_o2m=%.2f w_o2o=%.2f | lr=%.5f | %.1f img/s | ETA %.0fs",
                    epoch + 1, self.epochs, step + 1, num_batches,
                    logged["total"], logged["box"], logged["cls"], logged["kpt"], logged["vis"],
                    w_o2m, w_o2o, lr, imgs_per_s, eta,
                )

            if (time.time() - last_ckpt_time) >= self.ckpt_interval_min * 60:
                self._save("last.pt", epoch=epoch, metrics={"oks_map": self.best_ap})
                last_ckpt_time = time.time()
        return last_ckpt_time

    def _eval_model(self):
        """Return the EMA model configured for inference (un-fused, o2o head)."""

        model = self.ema.ema
        model.xtx2_config = self.config  # type: ignore[attr-defined]
        model.merge_params = MergeParams.from_dict(self.config.get("merge"))  # type: ignore[attr-defined]
        model.device_str = self.device  # type: ignore[attr-defined]
        return model

    def _ground_truths(self, samples) -> tuple[list, list[ImageGroundTruth]]:
        import cv2

        images, gts = [], []
        for sample in samples[: self.max_val_images]:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes, kpts = denormalise_sample(sample, width=width, height=height)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            images.append(image)
            gts.append(ImageGroundTruth(keypoints=kpts, areas=areas))
        return images, gts

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Per-epoch validation via the full 3-level merged pipeline (section 9.2)."""

        from ..api import detect_poses

        model = self._eval_model()
        images, gts = self._ground_truths(self.test_dataset.samples)
        if not images:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
        predictions: list[ImagePredictions] = []
        for image in images:
            poses = detect_poses(image, model=model, conf_threshold=0.05, max_detections=100)
            if poses:
                kpts = np.stack([p.keypoints for p in poses], axis=0)
                scores = np.array([p.score for p in poses])
            else:
                kpts = np.zeros((0, 17, 3))
                scores = np.zeros((0,))
            predictions.append(ImagePredictions(keypoints=kpts, scores=scores))
        return evaluate(predictions, gts)

    @torch.no_grad()
    def calibrate(self) -> MergeParams:
        """Merge calibration (section 5.3) on the test split."""

        from ..api import _detections_per_level

        model = self._eval_model()
        images, gts = self._ground_truths(self.test_dataset.samples)
        if not images:
            return MergeParams.from_dict(self.config.get("merge"))
        per_image_levels = [
            _detections_per_level(
                model, image, conf_threshold=0.05, max_detections=100, device=self.device
            )[0]
            for image in images
        ]
        base = MergeParams.from_dict(self.config.get("merge"))
        best, _ap = calibrate_thresholds(per_image_levels, gts, base_params=base)
        return best

    def _mirror_merge_params(self, merge_params: MergeParams) -> None:
        """Mirror calibrated merge params into the app-root config.json (5.3)."""

        config_path = Path(__file__).resolve().parents[2] / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            config["merge"] = merge_params.to_dict()
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            logger.info("Mirrored merge params into %s", config_path)
        except OSError as exc:
            logger.warning("Could not mirror merge params into config.json: %s", exc)

    def _save(
        self,
        filename: str,
        *,
        epoch: int,
        metrics: dict[str, Any],
        merge_params: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = build_checkpoint(
            variant=self.variant,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            config=self.config,
            merge_params=merge_params,
            metrics=metrics,
            criterion_state=self.criterion.state_dict(),
        )
        save_checkpoint(checkpoint, self.output_dir / filename)
