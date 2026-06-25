"""Training pipeline (spec section 9): console app with dual head, EMA, AMP,
time-based checkpointing, per-epoch OKS validation, and merge calibration.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.augment import AugmentConfig
from ..data.dataset import BalancedSampler, XTXPoseDataset, collate_fn, denormalise_sample
from ..losses import XTXLoss
from ..models import build_model
from ..utils.logging import get_logger
from .checkpoint import ModelEMA, build_checkpoint, load_checkpoint, save_checkpoint
from .merge import MergeParams, calibrate_thresholds
from .metrics import ImageGroundTruth, ImagePredictions, evaluate

logger = get_logger(__name__)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load config.json from disk, falling back to packaged defaults for missing keys."""

    import json

    defaults_path = Path(__file__).resolve().parents[2] / "config.json"
    config: dict[str, Any] = {}
    if defaults_path.exists():
        config = json.loads(defaults_path.read_text(encoding="utf-8"))
    if config_path is not None and Path(config_path).exists() and Path(config_path) != defaults_path:
        config = _deep_update(config, json.loads(Path(config_path).read_text(encoding="utf-8")))
    return config


def prompt_variant(provided: str | None) -> str:
    """Interactive variant prompt (section 9.1), validated, with CLI override."""

    if provided in {"n", "m", "l"}:
        return provided
    while True:
        answer = input("Which model do you want to train? [n/m/l]: ").strip().lower()
        if answer in {"n", "m", "l"}:
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


class Trainer:
    """Encapsulates the full training loop for one XTX variant."""

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
        self.output_dir = Path(output_dir or f"runs/xtx-{variant}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        self.epochs = int(train_cfg.get("epochs", 100))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.lr0 = float(train_cfg.get("lr0", 0.01))
        self.lr_final_factor = float(train_cfg.get("lr_final_factor", 0.01))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0005))
        self.momentum = float(train_cfg.get("momentum", 0.937))
        self.use_amp = bool(train_cfg.get("amp", True)) and self.device == "cuda"
        self.num_workers = int(train_cfg.get("num_workers", 8))
        self.ckpt_interval_min = float(train_cfg.get("checkpoint_interval_minutes", 5))
        self.close_mosaic_epochs = int(data_cfg.get("close_mosaic_epochs", 10))
        self.abc_ratio = tuple(data_cfg.get("abc_ratio", [1, 1, 1]))
        self.max_val_images = int(config.get("validation", {}).get("max_images", 200))

        self._build()

    def _build(self) -> None:
        data_cfg = self.config.get("data", {})
        augment_cfg = AugmentConfig.from_dict(data_cfg.get("augment"))
        self.train_dataset = XTXPoseDataset(
            split_dir=self.dataset_path / "train",
            augment_config=augment_cfg,
            augment=True,
            min_visible_keypoints=int(data_cfg.get("min_visible_keypoints", 1)),
        )
        self.test_dataset = XTXPoseDataset(
            split_dir=self.dataset_path / "test",
            augment=False,
            min_visible_keypoints=int(data_cfg.get("min_visible_keypoints", 1)),
        )
        self.sampler = BalancedSampler(
            index=self.train_dataset.index,
            abc_ratio=self.abc_ratio,
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

        self.model = build_model(variant=self.variant, config=self.config).to(self.device)
        self.criterion = XTXLoss.from_config(self.config).to(self.device)
        self.ema = ModelEMA(self.model, decay=float(self.config["training"].get("ema_decay", 0.9999)))
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        self.global_step = 0
        self.best_ap = -1.0

        params = self.model.num_parameters() / 1e6
        logger.info(
            "Built XTX-%s (%.2fM params) on %s | train=%d imgs test=%d imgs",
            self.variant,
            params,
            self.device,
            len(self.train_dataset.samples),
            len(self.test_dataset.samples),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for module in list(self.model.modules()) + list(self.criterion.modules()):
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith("bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
        return torch.optim.SGD(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr0,
            momentum=self.momentum,
            nesterov=True,
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
        self._save(
            "best.pt",
            epoch=self.epochs - 1,
            metrics={"oks_map": self.best_ap},
            merge_params=merge_params.to_dict(),
        )
        return {"best_ap": self.best_ap, "merge_params": merge_params.to_dict()}

    def _train_epoch(self, epoch: int, last_ckpt_time: float) -> float:
        self.model.train()
        num_batches = len(self.train_loader)
        epoch_start = time.time()
        running = {"total": 0.0, "box": 0.0, "cls": 0.0, "kpt": 0.0, "vis": 0.0}

        for step, (images, targets, _crops, _metas) in enumerate(self.train_loader):
            epoch_float = epoch + step / max(num_batches, 1)
            lr = self._lr_at(epoch_float)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

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

            for key in running:
                running[key] += float(losses[key].detach())

            if step % 10 == 0 or step == num_batches - 1:
                elapsed = max(time.time() - epoch_start, 1e-6)
                imgs_per_s = (step + 1) * self.batch_size / elapsed
                eta = (num_batches - step - 1) * elapsed / max(step + 1, 1)
                logger.info(
                    "E%d/%d S%d/%d | total=%.3f box=%.3f cls=%.3f kpt=%.3f vis=%.3f | lr=%.5f | %.1f img/s | ETA %.0fs",
                    epoch + 1, self.epochs, step + 1, num_batches,
                    losses["total"], losses["box"], losses["cls"], losses["kpt"], losses["vis"],
                    lr, imgs_per_s, eta,
                )

            if (time.time() - last_ckpt_time) >= self.ckpt_interval_min * 60:
                self._save("last.pt", epoch=epoch, metrics={"oks_map": self.best_ap})
                last_ckpt_time = time.time()
        return last_ckpt_time

    def _eval_model(self):
        """Return the EMA model configured for inference (un-fused, o2o head)."""

        model = self.ema.ema
        model.xtx_config = self.config  # type: ignore[attr-defined]
        model.merge_params = MergeParams.from_dict(self.config.get("merge"))  # type: ignore[attr-defined]
        model.device_str = self.device  # type: ignore[attr-defined]
        return model

    def _ground_truths(self, samples) -> tuple[list, list[ImageGroundTruth]]:
        import cv2

        paths, gts = [], []
        for sample in samples[: self.max_val_images]:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes, kpts = denormalise_sample(sample, width=width, height=height)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            paths.append(image)
            gts.append(ImageGroundTruth(keypoints=kpts, areas=areas))
        return paths, gts

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
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
        from ..api import _detections_per_crop

        model = self._eval_model()
        images, gts = self._ground_truths(self.test_dataset.samples)
        if not images:
            return MergeParams.from_dict(self.config.get("merge"))
        per_image_crops = [
            _detections_per_crop(
                model, image, conf_threshold=0.05, max_detections=100, device=self.device
            )[0]
            for image in images
        ]
        base = MergeParams.from_dict(self.config.get("merge"))
        best, _ap = calibrate_thresholds(per_image_crops, gts, base_params=base)
        return best

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
