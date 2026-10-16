"""Training engine for XTX4 (implementation_plan_fable.md §8-9, CONTRACTS §10-11).

What the loop does and why:

* **Multi-scale batches** come from :class:`MultiScaleBatchSampler` (one ``S`` per
  batch, rendered directly at ``S`` by the dataset); the loader is rebuilt for
  the close-mosaic phase because persistent workers keep their own dataset copy.
* **AMP + GradScaler**: the model runs in FP16, the loss forces FP32 internally.
  EMA and the step counters advance only when the optimizer step was actually
  applied (a GradScaler overflow skip lowers the scale, which we detect).
* **Grad-norm clipping** covers the model *and* the criterion (RLE flow).
* **Profiling**: the first ``training.profile_steps`` iterations are timed per
  phase with CUDA synchronisation and written to ``profile.json``; without this
  evidence no throughput work is done (§8.1).
* **Validation** runs the EMA model through the public API (`detect_poses_batch`)
  on a fixed, stratified subset of the test split every epoch, on the full
  split every ``training.full_val_every`` epochs and at the end, and writes a
  detailed report (`evaluate_detailed`) at the end. ``best.pt`` is selected by
  the per-epoch subset AP at ``data.eval_size``.
* **Checkpoints** follow CONTRACTS §11: ``last.pt`` every epoch and every
  ``checkpoint_interval_minutes``; a mid-epoch checkpoint stores the last
  *completed* epoch, so resuming restarts the interrupted epoch (§8.3).
  After every completed epoch the full state is also kept as
  ``epochs/epoch_NNNN.pt`` (``NNNN`` = the 1-based epoch number the log prints,
  i.e. ``checkpoint["epoch"] + 1``); the dict is built once, written to the
  epoch file and *copied* to ``last.pt`` / ``best.pt``. Only the newest
  ``training.keep_epochs`` files survive (pruned by mtime, see
  ``checkpoint.prune_epoch_checkpoints``), so a reboot mid-run costs at most one
  epoch and a corrupted ``last.pt`` still leaves older full states.
* **Thermal guard** (``training.thermal``, on by default): a sampling thread
  (:mod:`xtx4.engine.thermal`) sets a flag when the GPU/CPU is too hot; the
  train loop checks it once per step, synchronises CUDA and sleeps until the
  sensors are back under the resume thresholds. Pause time is excluded from
  img/s and ETA, logged per epoch and written to TensorBoard ``sys/*``.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.annotations import DatasetSpec, ImageSample, dataset_statistics, scan_split
from ..data.augment import AugmentConfig
from ..data.dataset import XTX4PoseDataset, build_eval_ground_truth, collate_fn, read_image, worker_init_fn
from ..data.manifest import build_manifest
from ..data.sampler import MultiScaleBatchSampler
from ..losses import XTX4Loss
from ..models import build_model
from ..utils.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
from ..utils.logging import add_file_handler, get_logger
from .checkpoint import (
    ModelEMA,
    build_checkpoint,
    copy_checkpoint,
    epoch_checkpoint_path,
    load_checkpoint,
    prune_epoch_checkpoints,
    restore_rng_state,
    save_checkpoint,
    sha256_of_file,
)
from .optim import WarmupCosineSchedule, build_optimizer_from_config
from .thermal import SensorFn, ThermalConfig, ThermalGuard

logger = get_logger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "u.json"
DEFAULT_OUTPUT_ROOT = "D:/xtx4_runs"
PERSON_BUCKETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 4), (5, 10**9))
_PROFILE_SKIP = 3  # first steps include cudnn autotuning / allocator warm-up


# --------------------------------------------------------------------------- config


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge (``override`` wins); returns ``base`` mutated."""

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """``configs/u.json`` deep-merged with the user JSON (CONTRACTS §10)."""

    config: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        if path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            config = deep_update(config, json.loads(path.read_text(encoding="utf-8")))
    return config


def parse_override_value(raw: str) -> Any:
    """JSON when it parses (``3``, ``0.5``, ``true``, ``null``, ``[1,2]``), else the string."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` assignments in place (``--set model.levels=P3P5``)."""

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = parse_override_value(raw.strip())
    return config


def resolve_output_dir(config: dict[str, Any], output: str | Path | None) -> Path:
    if output:
        return Path(output)
    run_cfg = config.get("run", {})
    root = Path(str(run_cfg.get("output_root") or DEFAULT_OUTPUT_ROOT))
    return root / str(run_cfg.get("name") or "xtx4")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _losses_to_float(losses: dict[str, Any]) -> dict[str, float]:
    """Convert the loss dict to floats with a single device->host transfer."""

    device_keys: list[str] = []
    device_vals: list[torch.Tensor] = []
    out: dict[str, float] = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            t = value.detach()
            if t.device.type == "cpu":
                out[key] = float(t)
            else:
                device_keys.append(key)
                device_vals.append(t.reshape(()).float())
        else:
            out[key] = float(value)
    if device_vals:
        host = torch.stack(device_vals).cpu().tolist()
        out.update(dict(zip(device_keys, host)))
    return out


def _dataset_spec(data_cfg: dict[str, Any]) -> DatasetSpec:
    spec = data_cfg.get("dataset_spec")
    return DatasetSpec.from_dict(spec) if isinstance(spec, dict) else DatasetSpec("xtx", "coco17", None)


# --------------------------------------------------------------------------- trainer


class Trainer:
    """Full training loop for one XTX4 configuration."""

    def __init__(
        self,
        config: dict[str, Any],
        dataset_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        device: str | None = None,
        *,
        max_steps: int | None = None,
        thermal_sensor: SensorFn | None = None,
    ) -> None:
        self.config = config
        data_cfg = config.setdefault("data", {})
        train_cfg = config.setdefault("training", {})
        run_cfg = config.setdefault("run", {})
        self.dataset_path = Path(dataset_path or data_cfg.get("dataset_path") or "./dataset")
        data_cfg["dataset_path"] = str(self.dataset_path)
        self.output_dir = resolve_output_dir(config, output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        add_file_handler(self.output_dir / "train.log")

        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            requested = "cpu"
        self.device = requested
        self.is_cuda = self.device.startswith("cuda")
        self.max_steps = int(max_steps) if max_steps else None

        self.seed = int(run_cfg.get("seed", 0))
        set_global_seed(self.seed)
        if self.is_cuda:
            torch.backends.cudnn.benchmark = True

        self.epochs = int(train_cfg.get("epochs", 100))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.accumulate = max(1, int(train_cfg.get("accumulate", 1)))
        self.use_amp = bool(train_cfg.get("amp", True)) and self.is_cuda
        self.channels_last = bool(train_cfg.get("channels_last", True)) and self.is_cuda
        self.num_workers = int(train_cfg.get("num_workers", 6))
        self.ckpt_interval_min = float(train_cfg.get("checkpoint_interval_minutes", 10))
        self.keep_epochs = max(0, int(train_cfg.get("keep_epochs", 5)))
        self.thermal_cfg = ThermalConfig.from_dict(train_cfg.get("thermal"))
        self._thermal_sensor = thermal_sensor  # injected (tests); None = auto-detect in train()
        self.thermal: ThermalGuard | None = None
        self.close_mosaic_epochs = int(train_cfg.get("close_mosaic_epochs", 15))
        self.val_images_per_epoch = int(train_cfg.get("val_images_per_epoch", 2000))
        self.full_val_every = int(train_cfg.get("full_val_every", 10))
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 10.0))
        self.profile_steps = int(train_cfg.get("profile_steps", 50))
        self.log_every = max(1, int(train_cfg.get("log_every_steps", 10)))
        self.eval_batch = int(train_cfg.get("eval_batch_size", 32))
        self.sizes = [int(s) for s in data_cfg.get("sizes", [256, 320, 384])]
        self.size_probs = data_cfg.get("size_probs")
        self.eval_size = int(data_cfg.get("eval_size", max(self.sizes)))
        eval_cfg = config.setdefault("eval", {})
        self.eval_score_threshold = float(eval_cfg.get("score_threshold", 0.001))
        self.eval_max_det = int(eval_cfg.get("max_det", 100))
        self.eval_match_threshold = float(eval_cfg.get("match_threshold", 0.5))

        self.start_epoch = 0
        self.global_step = 0
        self.best_ap = -1.0
        self.data_manifest_sha1: str | None = None
        self.parent_checkpoint_sha256: str | None = None
        self.monitor = None
        self.profile: dict[str, Any] | None = None
        self.throughput: dict[str, Any] = {}
        self._amp_skips = 0

        self._build_data()
        self._build_model()

    # ------------------------------------------------------------------ build
    def _scan(self, split: str) -> list[ImageSample]:
        data_cfg = self.config["data"]
        spec = _dataset_spec(data_cfg)
        split_dir = self.dataset_path / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected '{split_dir}' to exist")
        samples = scan_split(split_dir, spec)
        if split == "train":
            for extra in data_cfg.get("extra_datasets") or []:
                extra = dict(extra)
                path = Path(extra.get("path", ""))
                if not path.is_dir():
                    logger.warning("extra dataset path missing, skipped: %s", path)
                    continue
                espec = DatasetSpec.from_dict(extra.get("spec") or extra)
                extra_dir = path / "train" if (path / "train").is_dir() else path
                extra_samples = scan_split(extra_dir, espec)
                logger.info("extra dataset %s: %d images from %s", espec.name, len(extra_samples), extra_dir)
                samples.extend(extra_samples)
        return samples

    def _build_data(self) -> None:
        data_cfg = self.config["data"]
        t0 = time.perf_counter()
        self.train_samples = self._scan("train")
        self.test_samples = self._scan("test")
        for name, samples in (("train", self.train_samples), ("test", self.test_samples)):
            stats = dataset_statistics(samples)
            logger.info(
                "%s split: %d images, %d persons, %d crowd (scan %.1fs)",
                name, stats["num_images"], stats["num_persons"], stats["num_crowd"], time.perf_counter() - t0,
            )
        cache_dir = data_cfg.get("cache_dir")
        if cache_dir:
            from ..data.cache import build_cache

            self.train_samples = build_cache(
                self.train_samples, cache_dir, max_side=int(data_cfg.get("cache_max_side", 640))
            )
        if bool(self.config["training"].get("data_manifest", True)):
            t0 = time.perf_counter()
            cache_root = self.output_dir.parent / "_manifest_cache"
            manifest = build_manifest(
                self.train_samples, "train", cache_path=cache_root / "train_sha1.json"
            )
            self.data_manifest_sha1 = manifest["manifest_sha1"]
            (self.output_dir / "manifest_train.json").write_text(
                json.dumps({k: v for k, v in manifest.items() if k != "images"}, indent=2), encoding="utf-8"
            )
            logger.info("train manifest sha1=%s (%.1fs)", self.data_manifest_sha1, time.perf_counter() - t0)

        self.augment_cfg = AugmentConfig.from_dict(data_cfg.get("augment"))
        self.train_dataset = XTX4PoseDataset(
            self.train_samples,
            self.sizes,
            self.augment_cfg,
            augment=True,
            min_labeled_keypoints=int(data_cfg.get("min_labeled_keypoints", 1)),
            mosaic=True,
            seed=self.seed,
        )
        self.sampler = MultiScaleBatchSampler(
            len(self.train_dataset), self.batch_size, self.sizes, self.size_probs, self.seed, drop_last=True
        )
        self.train_loader = self._make_loader()
        self.steps_per_epoch = len(self.sampler)
        if self.steps_per_epoch == 0:
            raise ValueError(f"batch_size {self.batch_size} exceeds the train split ({len(self.train_dataset)} images)")

        self._build_val_sets()

    def _make_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        kwargs: dict[str, Any] = dict(
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.is_cuda,
            generator=generator,
        )
        if self.num_workers > 0:
            kwargs.update(worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=4)
        return DataLoader(self.train_dataset, **kwargs)

    def _build_val_sets(self) -> None:
        """Ground truth for the full test split and the fixed stratified per-epoch subset."""

        from .metrics import ImageGroundTruth

        paths, gts = build_eval_ground_truth(self.test_samples, load_images=False, area_fallback="bbox")
        self.val_paths: list[str] = [str(p) for p in paths]
        self.val_gts: list[ImageGroundTruth] = []
        persons = np.zeros(len(gts), dtype=np.int64)
        for i, (sample, gt) in enumerate(zip(self.test_samples, gts)):
            self.val_gts.append(
                ImageGroundTruth(
                    keypoints=gt["keypoints"],
                    areas=gt["areas"],
                    iscrowd=gt["iscrowd"],
                    image_size=(int(sample.width), int(sample.height)),
                )
            )
            persons[i] = int((~gt["iscrowd"]).sum())
        self.val_subset_idx = self._stratified_subset(persons, self.val_images_per_epoch)
        logger.info(
            "validation: %d test images; per-epoch subset %d (buckets %s)",
            len(self.val_paths), len(self.val_subset_idx),
            {f"{lo}-{hi if hi < 10**9 else '+'}": int(((persons[self.val_subset_idx] >= lo) & (persons[self.val_subset_idx] <= hi)).sum())
             for lo, hi in PERSON_BUCKETS},
        )

    def _stratified_subset(self, persons: np.ndarray, count: int) -> np.ndarray:
        n = len(persons)
        if count <= 0 or count >= n:
            return np.arange(n)
        rng = np.random.default_rng([self.seed, 0xA11])
        chosen: list[np.ndarray] = []
        remainders: list[tuple[float, int]] = []
        allocated = 0
        bucket_members: list[np.ndarray] = []
        for b, (lo, hi) in enumerate(PERSON_BUCKETS):
            members = np.nonzero((persons >= lo) & (persons <= hi))[0]
            bucket_members.append(members)
            exact = count * len(members) / n
            take = int(math.floor(exact))
            remainders.append((exact - take, b))
            chosen.append(rng.permutation(members)[:take])
            allocated += min(take, len(members))
        # Largest-remainder rounding to hit ``count`` exactly.
        for _frac, b in sorted(remainders, reverse=True):
            if allocated >= count:
                break
            members = bucket_members[b]
            if len(chosen[b]) < len(members):
                remaining = np.setdiff1d(members, chosen[b])
                chosen[b] = np.concatenate([chosen[b], rng.permutation(remaining)[:1]])
                allocated += 1
        return np.sort(np.concatenate(chosen))

    def _build_model(self) -> None:
        train_cfg = self.config["training"]
        self.model = build_model(self.config).to(self.device)
        if self.channels_last and bool(train_cfg.get("channels_last_autocheck", True)):
            # NHWC is 5x slower than NCHW for this network on some cuDNN/torch builds (measured
            # on torch 2.14+cu130); never pay that silently.
            self.channels_last = self._channels_last_is_faster()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.criterion = XTX4Loss.from_config(self.config).to(self.device)
        self.criterion.set_schedule(self.epochs)
        self.ema = ModelEMA(
            self.model,
            decay=float(train_cfg.get("ema_decay", 0.9998)),
            warmup_steps=int(train_cfg.get("ema_warmup_steps", 2000)),
        )
        self.optimizer = build_optimizer_from_config(self.model, self.criterion, train_cfg)
        self.schedule = WarmupCosineSchedule.from_config(train_cfg, epochs=self.epochs)
        # A lower initial scale than torch's 2**16 avoids skipping the first handful of steps
        # (raw gradient norms at init are O(100) and overflow FP16 at the default scale).
        self.scaler = torch.amp.GradScaler(
            "cuda", init_scale=float(train_cfg.get("amp_init_scale", 2**13)), enabled=self.use_amp
        )
        spec = self.model.spec
        logger.info(
            "Built XTX4-%s (%s, %s, %s%s): %.3fM params (%.3fM deploy), %s GMAC | optimizer=%s lr0=%g "
            "batch=%d x%d sizes=%s eval=%d amp=%s channels_last=%s device=%s workers=%d | run dir %s",
            spec.name, spec.levels, spec.block, spec.act, ", p2_assist" if spec.p2_assist else "",
            self.model.num_parameters() / 1e6, self.model.num_parameters(deploy=True) / 1e6,
            "/".join(f"{self.model.mac_count(s):.3f}" for s in self.sizes),
            train_cfg.get("optimizer", "sgd"), self.schedule.lr0, self.batch_size, self.accumulate, self.sizes,
            self.eval_size, self.use_amp, self.channels_last, self.device, self.num_workers, self.output_dir,
        )

    def _channels_last_is_faster(self, *, iters: int = 4) -> bool:
        """Time fwd+bwd of a training batch in NCHW vs NHWC on a throw-away copy; keep NHWC only if faster."""

        size = max(self.sizes)
        was_benchmark = torch.backends.cudnn.benchmark
        timings: dict[str, float] = {}
        try:
            for fmt in ("contiguous", "channels_last"):
                model = copy.deepcopy(self.model).train()
                x = torch.rand(self.batch_size, 3, size, size, device=self.device)
                if fmt == "channels_last":
                    model = model.to(memory_format=torch.channels_last)
                    x = x.contiguous(memory_format=torch.channels_last)
                best = float("inf")
                for i in range(iters + 2):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                        out = model(x)
                    total = out.one2one.person.float().mean() + out.one2one.kpt.float().mean()
                    if out.one2many is not None:
                        total = total + out.one2many.person.float().mean() + out.one2many.kpt.float().mean()
                    total.backward()
                    torch.cuda.synchronize()
                    if i >= 2:
                        best = min(best, time.perf_counter() - t0)
                    model.zero_grad(set_to_none=True)
                timings[fmt] = best
                del model, x, out, total
            torch.cuda.empty_cache()
        except RuntimeError as exc:  # OOM etc.: fall back to NCHW, which is never the slow path
            logger.warning("channels_last check failed (%s); using NCHW", exc)
            return False
        finally:
            torch.backends.cudnn.benchmark = was_benchmark
        faster = timings["channels_last"] < 0.95 * timings["contiguous"]
        logger.info(
            "channels_last check @%d batch %d: NCHW %.1f ms, NHWC %.1f ms per fwd+bwd -> %s",
            size, self.batch_size, timings["contiguous"] * 1e3, timings["channels_last"] * 1e3,
            "channels_last" if faster else "NCHW (channels_last disabled)",
        )
        self.config["training"]["channels_last_measured_ms"] = {k: v * 1e3 for k, v in timings.items()}
        return faster

    # ------------------------------------------------------------------ resume / init
    def resume(self, checkpoint_path: Path | str) -> None:
        """Continue a run: model, EMA (+updates), optimizer, scaler, criterion, RNG, epoch, step."""

        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if ckpt.get("ema_state"):
            self.ema.load_state_dict(ckpt["ema_state"], updates=ckpt.get("ema_updates"))
        else:
            self.ema = ModelEMA(self.model, decay=self.ema.decay, warmup_steps=self.ema.warmup_steps)
        self.ema.rebind(self.model)
        if ckpt.get("optimizer_state"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, KeyError) as exc:
                logger.warning("Optimizer state not restored (%s); starting the optimizer fresh", exc)
        if ckpt.get("scaler_state") and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if ckpt.get("criterion_state"):
            self.criterion.load_state_dict(ckpt["criterion_state"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_ap = float(ckpt.get("metrics", {}).get("best_ap", ckpt.get("metrics", {}).get("AP", -1.0)))
        self.data_manifest_sha1 = self.data_manifest_sha1 or ckpt.get("data_manifest_sha1")
        self.parent_checkpoint_sha256 = ckpt.get("parent_checkpoint_sha256")
        if ckpt.get("data_manifest_sha1") and self.data_manifest_sha1 != ckpt.get("data_manifest_sha1"):
            logger.warning(
                "data manifest changed since the checkpoint (%s -> %s)",
                ckpt.get("data_manifest_sha1"), self.data_manifest_sha1,
            )
        restore_rng_state(ckpt.get("rng_state"))
        self.criterion.set_progress(self.start_epoch / max(self.epochs, 1))
        logger.info(
            "Resumed from %s: next epoch %d, global_step %d, ema_updates %d, best AP %.4f",
            checkpoint_path, self.start_epoch, self.global_step, self.ema.updates, self.best_ap,
        )

    def load_init_weights(self, checkpoint_path: Path | str) -> None:
        """Fine-tune start: weights only (EMA preferred), fresh optimizer/EMA, parent hash recorded."""

        path = Path(checkpoint_path)
        ckpt = load_checkpoint(path, map_location=self.device)
        state = ckpt.get("ema_state") or ckpt.get("model_state")
        if state is None:
            raise ValueError(f"{path} has neither ema_state nor model_state")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("init-weights: %d missing, %d unexpected keys (shape-compatible subset loaded)",
                           len(missing), len(unexpected))
        self.ema = ModelEMA(self.model, decay=self.ema.decay, warmup_steps=self.ema.warmup_steps)
        self.optimizer = build_optimizer_from_config(self.model, self.criterion, self.config["training"])
        self.parent_checkpoint_sha256 = sha256_of_file(path)
        logger.info("Initialised weights from %s (sha256 %s)", path, self.parent_checkpoint_sha256[:16])

    # ------------------------------------------------------------------ calibration
    def calibrate_loss(self, steps: int) -> Path:
        """Gradient-norm calibration of the loss weights (report only) -> ``loss_calibration.json``."""

        from ..losses.calibrate import calibrate_loss_weights, write_calibration_json

        def batches() -> Iterator[Any]:
            self.sampler.set_epoch(10**6)  # a dedicated permutation, not the first training epoch
            for images, targets, _metas in self.train_loader:
                if self.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                yield images, [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]

        result = calibrate_loss_weights(self.model, self.criterion, batches(), steps=int(steps))
        self.model.zero_grad(set_to_none=True)
        path = write_calibration_json(result, self.output_dir / "loss_calibration.json")
        logger.info("Loss calibration written -> %s (proposed loss config: %s)", path, result["proposed_config"])
        return path

    # ------------------------------------------------------------------ train
    def train(self) -> dict[str, Any]:
        from ..utils.tensorboard import TrainingMonitor

        self.monitor = None
        options = self.config["training"].get("tensorboard", {})
        if options.get("enabled", True):
            self.monitor = TrainingMonitor(self.output_dir, self.config, self.model, self.eval_size, self.global_step)
        (self.output_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self.thermal = ThermalGuard(self.thermal_cfg, sensor=self._thermal_sensor, is_cuda=self.is_cuda)
        self.thermal.start()
        try:
            return self._train_loop()
        finally:
            self.thermal.stop()  # also on exceptions / KeyboardInterrupt
            if self.monitor is not None:
                self.monitor.close()

    def _train_loop(self) -> dict[str, Any]:
        last_ckpt_time = time.time()
        mosaic_on = True
        stopped_early = False
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.epochs):
            if mosaic_on and epoch >= self.epochs - self.close_mosaic_epochs:
                logger.info("Closing mosaic for the last %d epochs (rebuilding the loader)", self.epochs - epoch)
                self.train_dataset.set_mosaic(False)
                del self.train_loader
                self.train_loader = self._make_loader()
                mosaic_on = False
            self.sampler.set_epoch(epoch)
            last_ckpt_time, finished = self._train_epoch(epoch, last_ckpt_time)
            if not finished:
                stopped_early = True
                self._save("last.pt", epoch=epoch - 1, metrics={"best_ap": self.best_ap, "partial_epoch": epoch})
                break

            full = (epoch + 1) % max(self.full_val_every, 1) == 0 or epoch + 1 == self.epochs
            metrics = self.validate(full=full, epoch=epoch)
            ap = float(metrics["AP"])
            if self.monitor is not None:
                self.monitor.scalars("validation", {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, self.global_step)
                if metrics.get("full_metrics"):
                    self.monitor.scalars("validation_full", {k: v for k, v in metrics["full_metrics"].items() if isinstance(v, (int, float))}, self.global_step)
                self.monitor.writer.flush()
            improved = ap > self.best_ap
            if improved:
                self.best_ap = ap
            self._save_epoch(epoch, metrics={"best_ap": self.best_ap, **metrics}, improved=improved)
            last_ckpt_time = time.time()

        summary: dict[str, Any] = {
            "best_ap": self.best_ap, "global_step": self.global_step, "throughput": self.throughput,
            "thermal_pause_s": self.thermal.pause_seconds if self.thermal is not None else 0.0,
            "thermal_pauses": self.thermal.pause_count if self.thermal is not None else 0,
        }
        if stopped_early:
            logger.info("Stopped after %d optimizer steps (max_steps); final validation/export skipped", self.global_step)
            return summary

        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            self._save("best.pt", epoch=max(epoch, 0), metrics={"best_ap": self.best_ap})
        detailed = self.validate_detailed()
        if detailed:
            (self.output_dir / "eval_report.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")
            logger.info("Wrote detailed evaluation report -> %s", self.output_dir / "eval_report.json")
            if self.monitor is not None:
                self.monitor.writer.add_text(
                    "validation/final_report", f"```json\n{json.dumps(detailed, indent=2)[:60000]}\n```", self.global_step
                )
            self._log_detailed(detailed)
        summary["detailed"] = detailed
        summary["export"] = self._auto_export(best_path)
        return summary

    # ------------------------------------------------------------------ epoch
    def _train_epoch(self, epoch: int, last_ckpt_time: float) -> tuple[float, bool]:
        self.model.train()
        num_batches = self.steps_per_epoch
        epoch_start = time.time()
        window_start, window_images = epoch_start, 0
        params = [p for p in self.model.parameters() if p.requires_grad] + list(self.criterion.parameters())
        profiling = self.profile is None and self.profile_steps > 0
        prof_rows: list[dict[str, float]] = []
        grad_norm = torch.zeros(())
        images_seen = 0
        lr = self.schedule.lr_at(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        pause_epoch = 0.0  # seconds spent in thermal pauses this epoch (excluded from img/s and ETA)
        pause_window = 0.0
        if self.thermal is not None:
            self.thermal.reset_epoch_stats()

        iterator = iter(self.train_loader)
        t_prev_end = time.perf_counter()
        for step in range(num_batches):
            paused = self._thermal_wait()  # cheap flag check; blocks only while the guard is hot
            if paused:
                pause_epoch += paused
                pause_window += paused
                t_prev_end = time.perf_counter()
            do_profile = profiling and len(prof_rows) < self.profile_steps
            batch = next(iterator)
            t_data = time.perf_counter()
            images, targets, _metas = batch
            epoch_float = epoch + step / max(num_batches, 1)
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
            image_size = int(images.shape[-1])
            if do_profile:
                self._sync()
            t_h2d = time.perf_counter()

            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                output = self.model(images)
            if do_profile:
                self._sync()
            t_fwd = time.perf_counter()
            self.criterion.set_progress(epoch_float / max(self.epochs, 1))
            losses = self.criterion(output, targets, image_size)
            total = losses["total"]
            if do_profile:
                self._sync()
            t_loss = time.perf_counter()

            self.scaler.scale(total / self.accumulate).backward()
            if do_profile:
                self._sync()
            t_bwd = time.perf_counter()

            stepped = False
            if (step + 1) % self.accumulate == 0 or step + 1 == num_batches:
                lr = self.schedule.apply(self.optimizer, epoch_float)
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)
                if self.use_amp:
                    # GradScaler skips the step on inf/nan gradients and lowers the scale: detect that.
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    stepped = not (self.scaler.get_scale() < scale_before)
                elif bool(torch.isfinite(grad_norm)):
                    self.optimizer.step()
                    stepped = True
                if stepped:
                    self.ema.update(self.model)
                    self.global_step += 1
                else:
                    self._amp_skips += 1
                    logger.warning("Optimizer step skipped (non-finite gradients); AMP scale now %.1f", self.scaler.get_scale())
                self.optimizer.zero_grad(set_to_none=True)
            if do_profile:
                self._sync()
            t_opt = time.perf_counter()
            if do_profile:
                prof_rows.append({
                    "data_wait": t_data - t_prev_end, "h2d": t_h2d - t_data, "forward": t_fwd - t_h2d,
                    "loss": t_loss - t_fwd, "backward": t_bwd - t_loss, "optimizer": t_opt - t_bwd,
                    "size": float(image_size),
                })
                if len(prof_rows) == self.profile_steps:
                    self._finish_profile(prof_rows)
            t_prev_end = t_opt
            images_seen += images.shape[0]
            window_images += images.shape[0]

            # ---- logging
            console = step % self.log_every == 0 or step == num_batches - 1
            tb = self.monitor is not None and (
                self.global_step % self.monitor.log_every == 0 or step == 0 or step == num_batches - 1
            )
            if console or tb:
                now = time.time()
                logged = _losses_to_float(losses)
                elapsed = max(now - epoch_start - pause_epoch, 1e-6)
                window = max(now - window_start - pause_window, 1e-6)
                imgs_per_s = window_images / window if step > 0 else images_seen / elapsed
                window_start, window_images, pause_window = now, 0, 0.0
                eta = (num_batches - step - 1) * elapsed / max(step + 1, 1)
                gnorm = float(grad_norm)
                if console:
                    thermal_status = f" | {self.thermal.status()}" if self.thermal is not None and self.thermal.active else ""
                    logger.info(
                        "E%d/%d S%d/%d | total=%.3f person=%.3f kpt=%.3f present=%.3f visible=%.3f | "
                        "cov=%.3f pos/gt=%.1f | w_o2m=%.2f w_o2o=%.2f | lr=%.5f | gn=%.2f | size=%d | %.1f img/s | ETA %s%s",
                        epoch + 1, self.epochs, step + 1, num_batches,
                        logged["total"], logged.get("person", 0.0), logged.get("kpt", 0.0),
                        logged.get("present", 0.0), logged.get("visible", 0.0),
                        logged.get("coverage", 0.0), logged.get("pos_per_gt", 0.0),
                        logged.get("w_o2m", 0.0), logged.get("w_o2o", 0.0), lr, gnorm, image_size, imgs_per_s,
                        _format_eta(eta), thermal_status,
                    )
                if tb:
                    self.monitor.scalars("loss", {k: v for k, v in logged.items() if not k.startswith(("o2m_", "o2o_"))}, self.global_step)
                    self.monitor.scalars("loss_heads", {k: v for k, v in logged.items() if k.startswith(("o2m_", "o2o_"))}, self.global_step)
                    train_scalars = {
                        "lr": lr, "epoch": epoch_float, "image_size": image_size, "images_per_second": imgs_per_s,
                        "gradient_norm_before_clip": gnorm, "amp_scale": self.scaler.get_scale(),
                        "amp_skips": self._amp_skips, "ema_decay": self.ema.current_decay(), "ema_updates": self.ema.updates,
                    }
                    self.monitor.scalars("training", train_scalars, self.global_step)
                    if self.is_cuda:
                        self.monitor.scalars("cuda", {
                            "allocated_MiB": torch.cuda.memory_allocated() / 2**20,
                            "reserved_MiB": torch.cuda.memory_reserved() / 2**20,
                            "max_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                        }, self.global_step)
                    if self.thermal is not None and self.thermal.active:
                        self.monitor.scalars("sys", self.thermal.scalars(), self.global_step)
                self.throughput = {
                    "images_per_second": imgs_per_s, "epoch_images_per_second": images_seen / elapsed,
                    "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20 if self.is_cuda else 0.0,
                    "coverage": logged.get("coverage", 0.0), "pos_per_gt": logged.get("pos_per_gt", 0.0),
                    "global_step": self.global_step, "amp_skips": self._amp_skips,
                }
            if self.monitor is not None and step == num_batches - 1 and (epoch + 1) % self.monitor.histogram_every == 0:
                self.monitor.histograms({"model": self.model, "criterion": self.criterion}, self.global_step)

            if (time.time() - last_ckpt_time) >= self.ckpt_interval_min * 60:
                self._save("last.pt", epoch=epoch - 1, metrics={"best_ap": self.best_ap, "partial_epoch": epoch})
                last_ckpt_time = time.time()

            if self.max_steps is not None and self.global_step >= self.max_steps:
                logger.info("max_steps=%d reached: %s", self.max_steps, json.dumps(self.throughput))
                (self.output_dir / "throughput.json").write_text(json.dumps(self.throughput, indent=2), encoding="utf-8")
                return last_ckpt_time, False

        elapsed = time.time() - epoch_start - pause_epoch
        thermal_note = ""
        if self.thermal is not None and self.thermal.active:
            peak = self.thermal.max_gpu_temp_epoch
            thermal_note = ", gpu max %s, thermal pause %.0f s this epoch / %.0f s total" % (
                f"{peak:.0f} C" if peak is not None else "n/a", pause_epoch, self.thermal.pause_seconds,
            )
        logger.info(
            "Epoch %d/%d done in %s: %.1f img/s avg, peak VRAM %.0f MiB, amp skips %d%s",
            epoch + 1, self.epochs, _format_eta(elapsed), images_seen / max(elapsed, 1e-6),
            torch.cuda.max_memory_allocated() / 2**20 if self.is_cuda else 0.0, self._amp_skips, thermal_note,
        )
        return last_ckpt_time, True

    def _thermal_wait(self) -> float:
        """Block while the thermal guard is hot; returns the paused seconds (0 almost always)."""

        guard = self.thermal
        if guard is None or not guard.hot:
            return 0.0
        return guard.wait_until_cool(before_pause=self._sync)

    def _sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()

    def _finish_profile(self, rows: list[dict[str, float]]) -> None:
        keys = ("data_wait", "h2d", "forward", "loss", "backward", "optimizer")
        used = rows[_PROFILE_SKIP:] if len(rows) > _PROFILE_SKIP + 2 else rows
        # Medians are robust to the cudnn autotune outliers (first steps of every new input size);
        # means are kept so the outliers remain visible in profile.json.
        medians = {k: float(np.median([r[k] for r in used])) * 1000.0 for k in keys}
        means = {k: float(np.mean([r[k] for r in used])) * 1000.0 for k in keys}
        step_ms = sum(medians.values())
        fractions = {k: medians[k] / step_ms if step_ms > 0 else 0.0 for k in keys}
        per_size = {}
        for size in sorted({int(r["size"]) for r in used}):
            sel = [r for r in used if int(r["size"]) == size]
            per_size[str(size)] = {k: float(np.median([r[k] for r in sel])) * 1000.0 for k in keys} | {"steps": len(sel)}
        self.profile = {
            "steps": len(used), "skipped_warmup_steps": len(rows) - len(used), "batch_size": self.batch_size,
            "sizes": sorted({int(r["size"]) for r in rows}), "median_ms": medians, "mean_ms": means,
            "fractions": fractions, "step_ms": step_ms, "mean_step_ms": sum(means.values()),
            "images_per_second_estimate": self.batch_size / (step_ms / 1000.0) if step_ms > 0 else 0.0,
            "per_size_median_ms": per_size,
            "peak_vram_MiB": torch.cuda.max_memory_allocated() / 2**20 if self.is_cuda else 0.0,
            "per_step": rows,
        }
        (self.output_dir / "profile.json").write_text(json.dumps(self.profile, indent=2), encoding="utf-8")
        breakdown = " ".join(f"{k}={medians[k]:.1f}ms({fractions[k] * 100:.0f}%)" for k in keys)
        logger.info(
            "profile (%d steps, synchronised, medians): step %.1f ms (mean %.1f) -> %s | est. %.1f img/s",
            len(used), step_ms, self.profile["mean_step_ms"], breakdown, self.profile["images_per_second_estimate"],
        )

    # ------------------------------------------------------------------ validation
    def _eval_model(self):
        from ..api import attach_inference_attributes

        model = self.ema.ema
        model.eval()
        attach_inference_attributes(model, config=self.config, device=self.device, network_size=self.eval_size)
        model.channels_last = self.channels_last  # type: ignore[attr-defined]
        return model

    def _predict(self, model, indices: np.ndarray, *, network_size: int):
        from ..api import detect_poses_batch
        from .metrics import ImagePredictions

        predictions = []
        paths = [self.val_paths[i] for i in indices]
        with ThreadPoolExecutor(max_workers=min(8, max(1, self.num_workers))) as pool:
            for start in range(0, len(paths), self.eval_batch):
                self._thermal_wait()
                chunk_paths = paths[start : start + self.eval_batch]
                images = list(pool.map(read_image, chunk_paths))
                valid = [img for img in images if img is not None]
                poses_per_image = detect_poses_batch(
                    valid, model, conf_threshold=self.eval_score_threshold,
                    max_detections=self.eval_max_det, network_size=network_size, oks_nms=False,
                ) if valid else []
                it = iter(poses_per_image)
                for img in images:
                    poses = next(it) if img is not None else []
                    if poses:
                        kpts = np.stack([p.keypoints for p in poses], axis=0)
                        scores = np.array([p.score for p in poses], dtype=np.float32)
                        vis = np.stack([p.visibility for p in poses], axis=0)
                    else:
                        kpts = np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
                        scores = np.zeros((0,), np.float32)
                        vis = np.zeros((0, NUM_KEYPOINTS), np.int64)
                    predictions.append(ImagePredictions(keypoints=kpts, scores=scores, visibility=vis))
        return predictions

    @torch.no_grad()
    def validate(self, *, full: bool = False, epoch: int | None = None, network_size: int | None = None) -> dict[str, Any]:
        """Per-epoch validation (fixed stratified subset); ``full=True`` also scores the whole split."""

        from .metrics import evaluate

        size = int(network_size or self.eval_size)
        model = self._eval_model()
        t0 = time.perf_counter()
        indices = np.arange(len(self.val_paths)) if full else self.val_subset_idx
        if len(indices) == 0:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0, "num_gt": 0, "num_det": 0}
        preds = self._predict(model, indices, network_size=size)
        gts = [self.val_gts[i] for i in indices]
        kwargs = dict(max_det=self.eval_max_det, score_threshold=self.eval_score_threshold)
        if full:
            full_metrics = evaluate(preds, gts, **kwargs)
            pos = {int(i): j for j, i in enumerate(indices)}
            sub = [pos[int(i)] for i in self.val_subset_idx]
            metrics = evaluate([preds[j] for j in sub], [gts[j] for j in sub], **kwargs)
            metrics["full_metrics"] = {k: v for k, v in full_metrics.items() if k != "per_threshold"}
            logger.info(
                "Epoch %s full val (%d imgs @%d): AP=%.4f AP50=%.4f AP75=%.4f AR=%.4f",
                "-" if epoch is None else epoch + 1, len(indices), size,
                full_metrics["AP"], full_metrics["AP50"], full_metrics["AP75"], full_metrics["AR"],
            )
        else:
            metrics = evaluate(preds, gts, **kwargs)
        metrics.pop("per_threshold", None)
        metrics.update(eval_size=size, val_images=int(len(self.val_subset_idx)), val_seconds=time.perf_counter() - t0)
        logger.info(
            "Epoch %s val (%d imgs @%d, %.1fs): AP=%.4f AP50=%.4f AP75=%.4f AR=%.4f (best %.4f)",
            "-" if epoch is None else epoch + 1, len(self.val_subset_idx), size, metrics["val_seconds"],
            metrics["AP"], metrics["AP50"], metrics["AP75"], metrics["AR"], max(self.best_ap, metrics["AP"]),
        )
        return metrics

    @torch.no_grad()
    def validate_detailed(self) -> dict[str, Any]:
        """Final report: full split at every ``eval.sizes`` (AP) + detailed diagnostics at ``eval_size``."""

        from .metrics import evaluate, evaluate_detailed

        if not self.val_paths:
            return {}
        model = self._eval_model()
        indices = np.arange(len(self.val_paths))
        gts = self.val_gts
        report: dict[str, Any] = {"eval_size": self.eval_size, "sizes": {}, "num_images": len(indices)}
        sizes = [int(s) for s in self.config.get("eval", {}).get("sizes", [self.eval_size])]
        if self.eval_size not in sizes:
            sizes.append(self.eval_size)
        for size in sizes:
            preds = self._predict(model, indices, network_size=size)
            oks = evaluate(preds, gts, max_det=self.eval_max_det, score_threshold=self.eval_score_threshold)
            report["sizes"][str(size)] = oks
            logger.info("Final full val @%d: AP=%.4f AP50=%.4f AP75=%.4f AR=%.4f", size, oks["AP"], oks["AP50"], oks["AP75"], oks["AR"])
            if size == self.eval_size:
                try:
                    report["detailed"] = evaluate_detailed(
                        preds, gts, match_threshold=self.eval_match_threshold,
                        conf_threshold=float(self.config.get("decode", {}).get("conf_threshold", 0.25)),
                        max_det=self.eval_max_det,
                    )
                except Exception as exc:  # noqa: BLE001 - the detailed report must not kill a finished run
                    logger.warning("evaluate_detailed failed: %s", exc)
                report["oks"] = oks
        best = self.output_dir / "best.pt"
        if best.exists():
            report["checkpoint_sha256"] = sha256_of_file(best)
        report["data_manifest_sha1"] = self.data_manifest_sha1
        return report

    def _log_detailed(self, report: dict[str, Any]) -> None:
        detailed = report.get("detailed") or {}
        det = detailed.get("detection", {})
        if det:
            logger.info(
                "Detection @conf %.2f/OKS %.2f: precision=%.4f recall=%.4f f1=%.4f duplicates/img=%.4f",
                float(self.config.get("decode", {}).get("conf_threshold", 0.25)), self.eval_match_threshold,
                det.get("precision", 0.0), det.get("recall", 0.0), det.get("f1", 0.0),
                detailed.get("duplicates_per_image", det.get("duplicates_per_image", 0.0)),
            )
        per_landmark = detailed.get("per_landmark", {})
        if isinstance(per_landmark, dict):
            for name in KEYPOINT_NAMES:
                stats = per_landmark.get(name)
                if isinstance(stats, dict):
                    logger.info("  %-14s %s", name, " ".join(f"{k}={v:.4f}" for k, v in stats.items() if isinstance(v, (int, float))))

    # ------------------------------------------------------------------ io
    def _build_checkpoint(self, *, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        return build_checkpoint(
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            config=self.config,
            metrics=metrics,
            criterion_state=self.criterion.state_dict(),
            data_manifest_sha1=self.data_manifest_sha1,
            parent_checkpoint_sha256=self.parent_checkpoint_sha256,
        )

    def _save(self, filename: str, *, epoch: int, metrics: dict[str, Any]) -> Path:
        return save_checkpoint(self._build_checkpoint(epoch=epoch, metrics=metrics), self.output_dir / filename)

    def _save_epoch(self, epoch: int, *, metrics: dict[str, Any], improved: bool) -> Path:
        """End-of-epoch persistence: one checkpoint dict -> ``epochs/epoch_NNNN.pt`` -> copies -> prune.

        With ``keep_epochs == 0`` only ``last.pt`` (and ``best.pt`` when improved) is written, as before.
        Order matters for crash safety: the epoch file is complete before ``last.pt`` is replaced,
        and rotation runs last so an interrupted save never removes the only good state.
        """

        checkpoint = self._build_checkpoint(epoch=epoch, metrics=metrics)
        if self.keep_epochs <= 0:
            if improved:
                save_checkpoint(checkpoint, self.output_dir / "best.pt")
            return save_checkpoint(checkpoint, self.output_dir / "last.pt")
        epoch_path = save_checkpoint(checkpoint, epoch_checkpoint_path(self.output_dir, epoch))
        del checkpoint
        if improved:
            copy_checkpoint(epoch_path, self.output_dir / "best.pt")
        last_path = copy_checkpoint(epoch_path, self.output_dir / "last.pt")
        prune_epoch_checkpoints(self.output_dir, self.keep_epochs, protect=epoch_path)
        return last_path

    # ------------------------------------------------------------------ export
    def _auto_export(self, best_path: Path) -> dict[str, Any] | None:
        export_cfg = self.config.get("export", {})
        if not bool(export_cfg.get("auto_export", True)):
            logger.info("Auto-export disabled in config; skipping")
            return None
        try:
            return export_checkpoint_model(
                model=copy.deepcopy(self.ema.ema).cpu().float(),
                config=self.config,
                out_dir=self.output_dir / "export",
                checkpoint_path=best_path if best_path.exists() else None,
                data_manifest_sha1=self.data_manifest_sha1,
            )
        except Exception as exc:  # noqa: BLE001 - training results must survive export issues
            logger.warning("Auto-export failed (%s). Checkpoints are intact; export manually from %s", exc, best_path)
            return None


# --------------------------------------------------------------------------- export helper


def export_checkpoint_model(
    *,
    model,
    config: dict[str, Any],
    out_dir: Path,
    checkpoint_path: Path | None,
    data_manifest_sha1: str | None = None,
) -> dict[str, Any]:
    """Fuse ``model`` and export ONNX + NCNN for ``export.sizes`` into ``out_dir`` with a JSON manifest."""

    from ..export.onnx_ncnn import convert_onnx_to_ncnn, export_onnx, verify_onnx
    from .checkpoint import keypoint_meta

    export_cfg = config.get("export", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    model.fuse()
    name = f"xtx4-{model.spec.name}"
    sizes = [int(s) for s in export_cfg.get("sizes", [320])]
    opset = int(export_cfg.get("onnx_opset", 17))
    converter = str(export_cfg.get("converter", "pnnx"))
    fp16 = bool(export_cfg.get("fp16", True))
    strides = list(model.strides)
    entries: list[dict[str, Any]] = []
    for size in sizes:
        onnx_path = out_dir / f"{name}-{size}.onnx"
        export_onnx(model, onnx_path, input_size=size, opset=opset)
        diff = verify_onnx(model, onnx_path, input_size=size)
        entry: dict[str, Any] = {
            "size": size, "onnx": onnx_path.name, "onnx_max_abs_diff": float(diff),
            "num_anchors": int(sum((size // s) ** 2 for s in strides)),
        }
        try:
            conv = convert_onnx_to_ncnn(onnx_path, out_dir / f"{name}-{size}", converter=converter, fp16=fp16)
            entry.update(
                ncnn_param=conv.param_path.name, ncnn_bin=conv.bin_path.name,
                input_names=list(conv.input_names), output_names=list(conv.output_names), fp16=conv.fp16,
            )
        except Exception as exc:  # noqa: BLE001 - keep the ONNX even when pnnx is missing
            logger.warning("NCNN conversion failed for %d px (%s); ONNX kept", size, exc)
            entry["ncnn_error"] = str(exc)
        entries.append(entry)
        logger.info("Exported %s @%d (onnx diff %.2e)%s", name, size, diff, "" if "ncnn_param" in entry else " [no ncnn]")

    decode_cfg = config.get("decode", {})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": name,
        "checkpoint_sha256": sha256_of_file(checkpoint_path) if checkpoint_path else None,
        "data_manifest_sha1": data_manifest_sha1,
        "variant_spec": model.spec.to_dict(),
        "keypoints": keypoint_meta(),
        "input": {"name": "in0", "layout": "NCHW", "color": "BGR", "scale": "1/255", "letterbox": "whole", "sizes": sizes},
        "strides": strides,
        "levels": model.spec.levels,
        "anchor_order": "levels ascending stride, row-major (y outer, x inner); centre (i+0.5)*stride",
        "outputs": [
            {"role": "person", "shape": "[1, N, 1]", "layout": "logit"},
            {"role": "pose", "shape": f"[1, N, {model.num_keypoints * 5}]", "layout": "K*5 (dx, dy, log_sigma, present_logit, visible_logit)"},
        ],
        "decode_version": 1,
        "decode": {
            "sigma_floor": float(decode_cfg.get("sigma_floor", 0.05)),
            "sigma_conf_coeff": float(decode_cfg.get("sigma_conf_coeff", 0.0)),
            "conf_threshold": float(decode_cfg.get("conf_threshold", 0.25)),
            "max_detections": int(decode_cfg.get("max_detections", 100)),
        },
        "precision": "fp16" if fp16 else "fp32",
        "exports": entries,
        "torch_version": torch.__version__,
    }
    manifest_path = out_dir / "manifest.json"
    written = False
    try:  # the runtime owner's manifest writer, when it exists
        from ..export.manifest import write_manifest  # type: ignore[attr-defined]
        import inspect

        sig = inspect.signature(write_manifest)
        candidates = {
            "path": manifest_path, "model": model, "config": config, "checkpoint_path": checkpoint_path,
            "exports": entries, "sizes": sizes, "manifest": manifest,
        }
        kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
        if "path" in kwargs:
            write_manifest(**kwargs)
            written = manifest_path.exists()
    except Exception as exc:  # noqa: BLE001 - fall back to the minimal manifest
        logger.info("export.manifest.write_manifest unavailable (%s); writing the minimal manifest", exc)
    if not written:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Export manifest -> %s", manifest_path)
    return manifest
