# XTX4 – binding interface contracts

This file is the single source of truth for interfaces between XTX4 modules. Every
implementer (human or agent) reads it before writing code and updates it in the
same change when an interface has to move. The design rationale lives in
`implementation_plan_fable.md`; this file only fixes *what* the code exposes.

Repository root for all paths below: `C:/git/freesentry/xtx4/`. The XTX3 project
(`C:/git/freesentry/xtx3/`) is **read-only** reference code and the baseline.

---

## 1. Layout, environment, conventions

```
xtx4/
  CONTRACTS.md  implementation_plan_fable.md  implementation_plan.md (Astra, reference)
  EXPERIMENTS.md  RESEARCH.md  README.md
  requirements.txt  init.bat  train.bat  configs/{u,smoke,...}.json
  train.py  scripts/*.py  tests/*.py  bench/  reports/  runs/
  xtx4/
    __init__.py            lazy public API (api.py)
    utils/   keypoints.py geometry.py logging.py tensorboard.py        (ported, stable)
    data/    preprocess.py (ported, stable) annotations.py manifest.py dataset.py augment.py sampler.py cache.py
    models/  head_types.py (stable contract) blocks.py backbone.py neck.py head.py xtx4.py scaling.py
    losses/  targets.py oks.py assign.py rle.py pose_loss.py calibrate.py
    engine/  checkpoint.py (ported) metrics.py optim.py trainer.py
    export/  onnx_ncnn.py deploy.py manifest.py
    runtime/ decode.py ncnn_runner.py pipeline.py      (NO torch import anywhere in runtime/)
    api.py
```

- Python 3.12, torch 2.13 (cu130), numpy, opencv. Interpreter: `xtx4/.venv/Scripts/python.exe`
  (created by `init.bat`; packages per `requirements.txt`). Tests: `.venv/Scripts/python.exe -m pytest tests -q`.
- Style as in XTX3: `from __future__ import annotations`, type hints, keyword-only constructor
  arguments, module docstrings explaining *why*, loggers via `xtx4.utils.logging.get_logger(__name__)`,
  no `print` outside interactive prompts. Deterministic seeds in tests.
- Smoke data for tests (read-only): `C:/git/freesentry/xtx3/.smoke_dataset/{train,test}/annotations.json`
  (COCO-style `images`/`annotations`, 17-point keypoints in **absolute pixels**, 120 train / 40 test
  images). Never write into `xtx3/`.
- Units: network pixels unless stated. Image tensors are BGR, float32, `/255`, NCHW. Letterbox =
  `xtx4.data.preprocess` (identical to XTX3 and to `xtx3/live_test`): aspect preserved, never
  upscaled, centred, black padding, paste offset aligned down to 4, `INTER_AREA` when downscaling.

---

## 2. Keypoints and visibility semantics

`xtx4/utils/keypoints.py` (ported, do not change): order
`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip`;
COCO indices `(0,1,2,3,4,5,6,11,12)`; flip pairs `(1,2),(3,4),(5,6),(7,8)`; OKS sigmas
`[0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.107,0.107]`; `K = NUM_KEYPOINTS = 9`.

Ground-truth visibility channel `v` (float stored, integer-valued):

| v | meaning | localisation loss | present target | visible target |
|---|---|---|---|---|
| -1 | **unknown** (dataset does not annotate this landmark at all, e.g. CrowdPose face points) | no | ignored | ignored |
| 0 | not labeled / absent / outside the view | no | 0 (weight `loss.unlabeled_present_weight`) | ignored |
| 1 | labeled, occluded | yes | 1 | 0 |
| 2 | labeled, visible | yes | 1 | 1 |

"Labeled" means `v > 0`. Predicted visibility class at inference: 0 if `present_prob < present_threshold`,
else 2 if `visible_prob >= visible_threshold` else 1 (`decode.present_threshold` / `decode.visible_threshold`,
defaults 0.5 / 0.5; calibrated per run).

---

## 3. Tensor contracts (`xtx4/models/head_types.py`, already written – do not change signatures)

- `KPT_CHANNELS_PER_POINT = 5`: per keypoint `(dx, dy, log_sigma, present_logit, visible_logit)`.
- `RawPredictions(person (B,N,1) logits, kpt (B,N,K,5), anchors (N,2) px, strides (N,))`.
- `HeadOutput(one2one: RawPredictions, one2many: RawPredictions | None)`; `one2many` is set only when
  `model.training` is True.
- Anchors: `build_anchors` → centres `(i + 0.5) * stride`, levels in **ascending stride order**,
  row-major (y outer, x inner) inside a level. `N = sum((S // stride)^2)` over active levels.
- Decode (`decode_keypoints`): `x = ax + dx * stride`, `y = ay + dy * stride`;
  `sigma_units = softplus(log_sigma) + 0.05`; `sigma_px = sigma_units * stride`;
  `present_prob = sigmoid(present_logit)`; `visible_prob = sigmoid(visible_logit)`;
  `conf = present_prob * exp(-sigma_conf_coeff * sigma_units)` (`decode.sigma_conf_coeff`, default 0).
- `decode_keypoints(..., present_threshold=0.5, visible_threshold=0.5)` additionally returns `visibility (B,N,K)`.
- `decode_detections(raw, conf_threshold, max_detections, sigma_conf_coeff, sigma_floor=0.05,
  present_threshold=0.5, visible_threshold=0.5)` → per image
  `dets (D, 1 + 3K) = [score, K*(x, y, conf)]` sorted by score, `vis (D,K)` int64, `sigma_px (D,K)`.
- Exported graph outputs (see §12): `person [1,N,1]` raw logits, `pose [1,N,K*5]` raw, flattened
  as `kpt.reshape(1, N, K*5)` (keypoint-major: all 5 values of keypoint 0, then keypoint 1, ...).

---

## 4. Model (`xtx4/models/*`)

### 4.1 Builder and API

```python
from xtx4.models import XTX4Model, build_model, VariantSpec, get_variant
model = build_model(config)            # reads config["model"]; returns XTX4Model (nn.Module)
model.spec -> VariantSpec              # resolved architecture spec (frozen dataclass)
model.strides -> list[int]             # e.g. [8,16,32] or [8,32] or [16,32]
model.num_keypoints -> int
model(images) -> HeadOutput            # training: both heads; eval: one2one only
model.predict(images, conf_threshold, max_detections, sigma_conf_coeff) -> (dets, vis, sigma)  # eval-mode helper
model.fuse() -> XTX4Model              # in place: fold BN, reparameterise RepConv, drop one2many; idempotent
model.num_parameters(deploy: bool = False) -> int   # deploy=True counts the fused one2one graph only
model.mac_count(input_size: int) -> float           # GMAC of the fused one2one graph (conv/linear hooks)
```

`build_model(config)` honours `config["model"]`: `variant` (preset name, see 4.2) and optional
overrides `levels`, `act`, `block`, `p2_assist`, `widths`, `depths`, `head_hidden`, `pose_hidden`,
`num_keypoints` (default 9).

### 4.2 Variant presets (`xtx4/models/scaling.py`)

| preset | widths (stem,P2,P3,P4,P5) | depths (s1..s4) | head_hidden / pose_hidden | block | act | default size |
|---|---|---|---|---|---|---|
| `u_a` | 16, 24, 48, 96, 128 | 1,1,1,1 | 48 / 48 | rep | relu | 320 |
| `u_b` (alias `u`) | 16, 32, 64, 128, 192 | 1,2,2,1 | 64 / 64 | rep | relu | 320 |
| `u_c` | 24, 48, 96, 192, 256 | 1,2,2,1 | 64 / 96 | rep | relu | 320 |
| `n` | 24, 48, 96, 192, 256 | 1,2,2,1 | 96 / 96 | rep | relu | 384 |
| `m` | 32, 64, 128, 256, 384 | 1,3,3,1 | 96 / 128 | rep | silu | 384 |
| `l` | 48, 96, 192, 384, 512 | 2,4,4,2 | 128 / 192 | rep | silu | 512 |
| `xtx3u_ctrl` | 24, 48, 96, 144, 144 | 1,1,1,1 | 48 / 64 | dw | silu | 320 |

`levels ∈ {"P3P4P5", "P3P5", "P4P5"}` (default `P3P4P5`), `act ∈ {"relu", "hswish", "silu"}`,
`block ∈ {"rep", "dw"}`, `p2_assist: bool`. All widths are multiples of 8.

### 4.3 Blocks and topology (`block="rep"`)

- `ConvAct`: Conv2d(no bias) + BN + act. `RepConv3x3`: training branches 3×3 + 1×1 (+ BN identity
  when shapes allow), `fuse()` → single 3×3 conv with bias (RepVGG). Downsample = `RepConv3x3` stride 2.
- `CSPRep(C_in, C_out, n)`: 1×1 `C_in → 2h` (`h = C_out // 2`), split, `n` bottlenecks on the second half
  (each: `RepConv3x3 h→h`, `RepConv3x3 h→h`, residual), concat `(2+n)·h` → 1×1 `C_out`. (C2f style.)
- Backbone: stem `ConvAct 3×3/2 3→w0`; stage1 `Rep/2 w0→w1, CSPRep(w1,w1,d1)` → P2 (stride 4);
  stage2 → P3 (8); stage3 → P4 (16); stage4 → P5 (32) + `SPPF(k=5)`.
- Neck (PAN): `td_p4 = CSPRep(w4+w3 → w3, 1)` on `cat(up2(P5), P4)`; `td_p3 = CSPRep(w3+w2 → w2, 1)` on
  `cat(up2(td_p4), P3)`; `bu_p4 = CSPRep(w2+w3 → w3, 1)` on `cat(Rep/2(td_p3), td_p4)`;
  `bu_p5 = CSPRep(w3+w4 → w4, 1)` on `cat(Rep/2(bu_p4), P5)`. Outputs `[N3, N4, N5]`.
  For `levels="P3P5"`: keep td path to N3, bottom-up goes `N3 → Rep/2 → Rep/2 → cat with P5 → bu_p5`
  (skip N4 head only; keep td_p4 for the top-down path). For `"P4P5"`: no td_p3; outputs `[N4, N5]`.
  `p2_assist=True`: `space_to_depth(P2) (4·w1 ch @ P3 res) → 1×1 → w1//2`, concatenated to N3 **only for
  the pose branch input**; the 1×1 is zero-initialised.
- Head (`PoseHead`): per level a `SingleHead`: shared `RepConv3x3 C→head_hidden`; person branch
  `Conv1x1 head_hidden→1` (bias init −4.6); pose branch `RepConv3x3 head_hidden→pose_hidden`,
  `Conv1x1 pose_hidden→K*5`. Two structurally identical `SingleHead` sets: `one2one` and `one2many`
  (separate parameters). Init of the pose 1×1: weights small (std 0.01), bias per keypoint
  `[0, 0, 0.6, 0, 0]` (so `sigma_units ≈ 1.0`, `present_prob = visible_prob = 0.5`).
- `block="dw"` reproduces the XTX3-u topology (PConv downsample in stages 1–2, DWConv elsewhere,
  depthwise bottlenecks, SPPF, DW head towers) as the recipe-control architecture; port from
  `xtx3/xtx3/models/{blocks,backbone,neck}.py` (`xtx4/models/blocks.py` already contains the copied
  XTX3 blocks as a starting point – extend, do not fork).
- Every module implements `fuse()`; after `model.fuse()` the one2one graph contains only
  Conv2d(+bias), activation, Concat, Upsample(nearest), MaxPool2d, (SpaceToDepth = reshape/permute).

---

## 5. Targets contract (dataset → trainer → loss)

Per image (`dict[str, Tensor]`, all float32, network pixels of the rendered view):

| key | shape | content |
|---|---|---|
| `keypoints` | (M, K, 3) | x, y, v with v ∈ {−1, 0, 1, 2} (§2) |
| `area` | (M,) | GT instance area in network px² for OKS; `NaN` when unknown (loss derives it, §6.1) |
| `boxes` | (M, 4) | xyxy network px when the source has boxes, else `NaN` (legacy/eval only; never used by assignment) |
| `iscrowd` | (M,) | bool |

Only persons with ≥ `data.min_labeled_keypoints` labeled points **inside the view** are targets (§8).
Points outside the view get `v = 0`. `collate_fn(batch) -> (images (B,3,S,S) float32, targets: list[dict],
metas: list[SampleMeta])` keeps targets as a list. `xtx4/losses/targets.py` provides

```python
pad_targets(targets: list[dict], device) -> PaddedTargets(kpts (B,M,K,3), area (B,M), mask (B,M) bool, boxes (B,M,4))
```

with `M = max persons in batch` (≥ 1), padded rows masked out.

---

## 6. Assignment contract (`xtx4/losses/assign.py`)

```python
assigner = PoseTAL(sigmas, alpha=0.5, beta=6.0, topk=10, topk2=None,  # topk2=1 → one-to-one (7→1)
                   extent_expand=0.25, extent_min_px=16.0, single_point_box_px=24.0,
                   min_candidates=4, root_prior_weight=0.1, eps=1e-9)
result = assigner.assign(person_prob (B,N), pred_xy (B,N,K,2), anchors (N,2), strides (N,),
                         gt_kpts (B,M,K,3), gt_area (B,M), gt_mask (B,M), root_prior_scale: float)
# AssignResult (all on device):
#   fg_mask       (B,N) bool      positive anchors
#   gt_idx        (B,N) int64     assigned GT index (0 where background)
#   target_score  (B,N) float32   soft person target in [0,1] (0 for background)
#   assigned_oks  (B,N) float32   OKS of the assigned pair (0 for background)
#   stats         dict[str, float]  'gt_total', 'gt_covered', 'pos_per_gt', 'cand_per_gt', 'small_gt'
```

Algorithm (fully batched, FP32, no Python loop over images or GTs; loops over K or levels are fine):
1. Labeled mask `L = v > 0`. Extent box `E` = min/max over labeled xy, expanded by `extent_expand`
   on each side, each side ≥ `extent_min_px`; single labeled point → square `single_point_box_px`.
2. Scale `s² = area` where finite, else `(1.5·w_E)·(1.5·h_E)`, floor `extent_min_px²`.
   Root `r` = mean of labeled shoulders if ≥ 1 shoulder labeled, else mean of labeled torso points,
   else mean of labeled face points.
3. `OKS(a,g) = mean_{k∈L_g} exp(-d²_k / (2 · s_g² · (2σ_k)²))`, `d` between `pred_xy[a,k]` and GT.
4. `align = person_prob^α · OKS^β · 1[anchor centre ∈ E_g]`; plus
   `root_prior_scale · root_prior_weight · exp(-|a − r_g|² / (2·(0.5·s_g)²))` (trainer passes
   `root_prior_scale` linearly decaying 1 → 0 over `assign.root_warmup_epochs`, then 0).
5. Candidate guarantee: if a GT has fewer than `min_candidates` anchors inside `E`, add the
   `min_candidates` anchors nearest to `r_g`.
6. top-k per GT by `align` (k = `topk`); if `topk2` is set, re-select the top `topk2` among them (YOLO26
   7→1). Conflicts (anchor claimed by several GTs) → the GT with the larger `align`.
7. `target_score = align / max_align_g · max_OKS_g` of the assigned GT (YOLOv8 normalisation),
   floored at 0.05 for the guaranteed candidates of small GTs (`max(w_E,h_E) < extent_min_px`).
8. Stats are computed on the batch after conflict resolution.

---

## 7. Loss contract (`xtx4/losses/pose_loss.py`)

```python
criterion = XTX4Loss.from_config(config)      # nn.Module (owns the RLE flow parameters)
criterion.set_progress(t: float)              # t = epoch / epochs in [0,1]; ProgLoss (0.8,0.2)→(0.1,0.9) linear
out = criterion(output: HeadOutput, targets: list[dict], image_size: int)
# out: dict of scalar tensors: 'total' (grad), and detached: 'person','kpt','present','visible',
#      'oks','rle','w_o2m','w_o2o','coverage','pos_per_gt' (+ anything useful for logs)
```

Per head (`_head_loss`): decode in FP32 (`autocast(enabled=False)`), assign (§6, `topk=10` for o2m;
`topk=7, topk2=1` for o2o), then
- person: `loss.person_loss ∈ {"bce","vfl"}` on `target_score`, normalised by `sum(target_score).clamp(1)`;
- kpt (positives, labeled points): `oks_weight·(1 − SOKS)` (`loss.kpt_loss ∈ {"soks","oks"}`) + `rle_weight·RLE`;
  SOKS: `u² = d²/(2 s² (2σ_k)²)`, `SOKS = exp(-u²)` for `u² < 1`, `exp(-(2u − 1))` otherwise (continuous at u=1);
  RLE in `rle_convention ∈ {"reference","xtx3"}`: reference = `log σ − log φ(ε) + log 2σ + |ε|`,
  xtx3 = `log σ + log 2 + |ε| − log φ(ε)`, `ε = (gt − μ)/σ` in **stride units** (σ = `sigma_units`),
  flow = RealNVP from `rle.py`. Normalisation: per positive anchor mean over its labeled points,
  then weighted mean over positives with weight `target_score`;
- present: BCE(`present_logit`, `1[v>0]`) over positives and all K, ignoring `v = −1`, weight
  `loss.unlabeled_present_weight` on `v = 0`; visible: BCE(`visible_logit`, `1[v==2]`) where `v > 0`.
- `total = w_head · (person·λ_person + kpt·λ_kpt + present·λ_present + visible·λ_visible)` summed
  over heads with ProgLoss weights (`w_o2m`, `w_o2o`).
- `xtx4/losses/calibrate.py`: `calibrate_loss_weights(model, criterion, loader, steps)` reports the
  gradient norm each term induces on the head parameters and proposes weights so no term exceeds 3×
  another (writes a JSON; does not change the config by itself).

---

## 8. Data contract (`xtx4/data/*`)

- `annotations.py`: `PersonAnnotation(keypoints_norm (K,3) float32 with v ∈ {−1,0,1,2},
  bbox_xywh_norm (4,) | None, area_norm float | None, iscrowd int)`, `ImageSample(image_path, width,
  height, annotations)`. `scan_split(split_dir, spec: DatasetSpec) -> list[ImageSample]`.
  `DatasetSpec(name, format ∈ {"coco17","xtx9","coco_wholebody","crowdpose","aic","mpii"}, keypoint_map)`
  where `keypoint_map` maps the source index list to the 9 XTX indices; landmarks without a source
  index get `v = −1` for every person of that dataset. Auto-detect absolute vs normalised coordinates
  as XTX3 does, but log the decision per JSON file. COCO `area` (segmentation area) is kept when present,
  else `None`. `num_keypoints == 0` persons are kept as `iscrowd`-like ignore regions only if `iscrowd`;
  otherwise skipped.
- `manifest.py`: `build_manifest(samples, split_name) -> dict` with per-image sha1 of file bytes,
  size, person count, source dataset; `write_manifest(path)`, `check_overlap(manifest_a, manifest_b)`
  (identical hashes across splits → error list).
- `cache.py`: `build_cache(samples, cache_dir, max_side=640)` re-encodes images whose longer side
  exceeds `max_side` as JPEG q95 into `cache_dir` (keeps smaller ones as is, just copies) and returns
  samples with `image_path` rewritten and `width/height` updated; annotations are normalised so they
  stay valid.
- `dataset.py`: `XTX4PoseDataset(samples, sizes, augment_cfg | None, augment: bool,
  min_labeled_keypoints=1, mosaic=True)`; `__getitem__(key: tuple[int, int])` with `key = (sample_idx, S)`
  → `(tensor (3,S,S) float32, targets (§5), SampleMeta(image_path, size S, transform (2,3) network→source px,
  orig_size))`. `__len__ = len(samples)`. Rendering: the augmented source image is letterboxed directly
  at `S` (`preprocess.render_view` equivalent with jittered placement) – never render at one size and
  resize the tensor. Inclusion rule: §5.
- `augment.py`: flip with pair swap; HSV; placement jitter (`scale_min..scale_max`, `translate`);
  mosaic4 that letterboxes each source into its quadrant (no aspect distortion); `upper_body_crop`
  (with probability p choose one person and crop the image below its hips or shoulders, keeping ≥ 1
  labeled point); photometric camera augments (motion blur k≤5, gaussian noise, JPEG 40–95, gamma 0.7–1.4)
  each with its own probability. All geometric ops update keypoints and set `v=0` for points that leave
  the image.
- `sampler.py`: `MultiScaleBatchSampler(num_samples, batch_size, sizes, size_probs, seed, drop_last=True)`
  yields lists of `(idx, S)` with one `S` per batch; `set_epoch(e)`. Weighted sampling by
  per-sample weights (small faces / crowded ×≤2) optional via `weights`.

---

## 9. Metrics contract (`xtx4/engine/metrics.py`)

```python
ImagePredictions(keypoints (D,K,3) [x,y,conf] px, scores (D,), visibility (D,K) | None)
ImageGroundTruth(keypoints (G,K,3) [x,y,v] px (v=−1 treated as 0), areas (G,), iscrowd (G,) bool,
                 image_size (W,H) | None, net_scale float | None)   # optional: needed for edge bucket / network-px buckets
evaluate(preds, gts, *, sigmas=OKS_SIGMAS, max_det=100, score_threshold=0.0) -> dict
#  {'AP','AP50','AP75','AR','per_threshold': {50: ap, 55: ap, ... 95: ap}, 'num_gt', 'num_det'}
evaluate_detailed(preds, gts, *, match_threshold=0.5, conf_threshold=0.25, sigmas=OKS_SIGMAS) -> dict
#  detection P/R/F1 at conf_threshold, duplicates_per_image (pairs with score>=conf and OKS>=0.5),
#  per-landmark mean OKS and NME, visibility confusion + macro-F1, face/torso PCK@0.1 (missed persons
#  count as 0), face-PCK@2px, buckets by face_spread (<8, 8–16, 16–32, >=32 px) and torso spread,
#  crowd buckets (1, 2–4, 5+ persons), edge bucket (any labeled point within 16 px of the border)
coco_eval(preds, gts, image_ids, *, max_det=20) -> dict   # pycocotools cross-check with 9 sigmas
```

Thresholds are integers `range(50, 100, 5)`; AP75 is `per_threshold[75]`. OKS = COCO formula with the
GT `area`; crowd GTs are ignore regions (matched detections neither TP nor FP). OKS matrices are
vectorised (numpy broadcasting). A single perfect detection must yield AP = AP50 = AP75 = AR = 1.

---

## 10. Config schema

`configs/u.json` is the reference; every key there is honoured by exactly one module as listed in §4–§9.
`load_config(path)` deep-merges a user JSON over `configs/u.json`. Checkpoints freeze the merged config.

Crash-resilience keys under `training` (owner: `engine/trainer.py`, `engine/thermal.py`):

- `keep_epochs` (int, default 5): number of rolling per-epoch checkpoints kept in `<run>/epochs/`
  (§11); `0` disables them. CLI: `--keep-epochs N`.
- `thermal` (object, default `{"enabled": true, "interval_s": 10, "gpu_pause_c": 85, "gpu_resume_c": 75,
  "cpu_pause_c": 95, "cpu_resume_c": 85, "pause_on_hw_slowdown": true, "max_pause_min": 60}`):
  `ThermalGuard` samples the GPU temperature / NVIDIA thermal-slowdown flags (pynvml if importable,
  else `nvidia-smi`) and, best effort, the CPU temperature (`psutil.sensors_temperatures()` on Linux,
  WMI `MSAcpi_ThermalZoneTemperature` on Windows) every `interval_s`. Training pauses (once per
  optimizer step, after `torch.cuda.synchronize()`) when any pause threshold is reached or the HW
  slowdown flag is active, and resumes only when every active sensor is at/below its resume threshold
  (hysteresis). Longer pauses than `max_pause_min` only log a warning; the run is never aborted.
  Pause time is excluded from img/s and ETA; TensorBoard `sys/gpu_temp_c`, `sys/cpu_temp_c`,
  `sys/thermal_pause_s`. Inactive (never pauses) without a sensor or on a CPU device.
  CLI: `--thermal-guard` / `--no-thermal-guard`.

---

## 11. Checkpoint format (`xtx4/engine/checkpoint.py`)

Dict with `variant_spec` (dict), `model_state`, `ema_state`, `optimizer_state`, `scaler_state`,
`criterion_state`, `epoch`, `global_step`, `ema_updates`, `rng_state` (python/numpy/torch/cuda),
`config`, `metrics`, `keypoint_meta`, `data_manifest_sha1`, `parent_checkpoint_sha256` (for fine-tune).
`load_model(path, device, fuse=True)` in `api.py` rebuilds from `config["model"]` and loads `ema_state`.

Files written by the trainer into the run directory:

- `last.pt` – state after the last *completed* epoch (`epoch` = its 0-based index), rewritten at
  the end of every epoch and every `training.checkpoint_interval_minutes` mid-epoch (the mid-epoch
  copy still stores the last completed epoch plus `metrics.partial_epoch`, so a resume restarts the
  interrupted epoch).
- `best.pt` – same format, written when the per-epoch subset AP improves.
- `epochs/epoch_NNNN.pt` – rolling full-state checkpoint after every completed epoch, identical
  content to `last.pt` (`last.pt`/`best.pt` are produced by an atomic file copy of it, the dict is built
  once). **Numbering:** `NNNN` is the 1-based epoch number as printed by the log (`Epoch 7/300 done`
  → `epoch_0007.pt`), i.e. `NNNN == checkpoint["epoch"] + 1`. **Rotation:** after the new file is
  completely written, `prune_epoch_checkpoints` keeps only the newest `training.keep_epochs` files by
  *modification time* (not by number): after `--resume epochs/epoch_0098.pt` with stale
  `epoch_0099..0105.pt` still present, the resumed run rewrites `epoch_0099.pt` (fresh mtime) and the
  stale higher-numbered files are the oldest, so they are pruned first. `keep_epochs = 0` disables
  the epoch files.
- Any of the three is a valid `--resume` target and behaves the same: training continues with
  `checkpoint["epoch"] + 1`, `global_step`/EMA/optimizer/scaler/RNG are restored and `best_ap` is
  taken from `metrics.best_ap`. `train.py` derives the run directory from an epoch file
  (`run_dir_of_checkpoint`), and `discover_checkpoints` lists `**/last.pt`, `**/best.pt` and
  `**/epochs/epoch_*.pt`, newest first.

---

## 12. Export and runtime contract

- `xtx4/export/onnx_ncnn.py`: `ExportModel(model)` → outputs `person [1,N,1]`, `pose [1,N,K*5]` (§3);
  `export_onnx(model, path, input_size, opset=17)` (static shape, `dynamo=False`, onnxsim, `onnx.checker`);
  `verify_onnx(model, path, input_size)` → max abs diff (tolerance 1e-3 raw) on a deterministic input;
  `convert_onnx_to_ncnn(onnx_path, prefix, converter="pnnx", fp16=True)` → `<prefix>.ncnn.param/.bin`.
  Input blob `in0`; pnnx names outputs `out0` (person) and `out1` (pose) – the manifest records the real names.
- `xtx4/export/manifest.py`: JSON with `schema_version`, `model_id`, `checkpoint_sha256`,
  `config_sha256`, `variant_spec`, `keypoints` (names, coco indices, flip pairs, sigmas), `input`
  (`layout NCHW`, `color BGR`, `scale 1/255`, `letterbox` description, sizes), `strides`, `levels`,
  `anchor_order`, `outputs` (names, shapes, layout `K*5`), `decode_version` (= 1), `decode`
  (sigma_floor, sigma_conf_coeff, conf_threshold, max_detections, present_threshold, visible_threshold,
  oks_nms, oks_nms_threshold, oks_nms_iou_threshold, oks_nms_scale_factor), `precision`, tool versions.
- `xtx4/runtime/decode.py` (NumPy only): `build_anchors(S, strides)`, `decode_raw(person, pose, S,
  strides, K, conf_threshold, max_detections, sigma_conf_coeff, sigma_floor, present_threshold=0.5,
  visible_threshold=0.5) -> (scores (D,), keypoints (D,K,3), visibility (D,K), sigma_px (D,K))` – filter by
  score **before** decoding keypoints; `to_frame_poses(..., transform)`; optional `oks_nms(poses, threshold,
  sigmas, *, scale_factor=1.5, iou_threshold=None)` (single implementation `oks_nms_indices`, also behind
  `xtx4.api.oks_nms`): OKS scale `max((f·w)(f·h), 16²)` of the candidate's confident-point extent, `f =
  scale_factor`; with `iou_threshold` a pair is also suppressed when the IoU of the expanded extent boxes
  is ≥ it.
- `xtx4/runtime/ncnn_runner.py`: `NcnnPoseRunner(manifest_path | param/bin, size, num_threads=4,
  use_vulkan=False)`; validates blob names and N from a dummy pass; `detect(frame_bgr) -> DetectionResult`
  with timings (preprocess/inference/postprocess ms).

---

## 13. Testing rules

- Every module ships `tests/test_<module>.py`; tests run in < 60 s total on CPU (GPU optional).
- Required fixtures: perfect-detection AP (=1 incl. AP75), duplicate detection (one TP + one FP),
  empty image, crowd ignore, overlapping persons; assignment on tiny person between cell centres,
  two persons in one cell, zero-GT batch, single-point person, FP16-vs-FP32 agreement; dataset:
  head-inside/box-outside crop keeps the person, flip L/R identity, affine round trip, half-pixel resize;
  model: forward shapes at 256/320/384 for all presets and level sets, fuse parity (< 1e-4), export parity.

---

## 14. Ownership (parallel implementation)

| Agent | Owns | Must not touch |
|---|---|---|
| models | `xtx4/models/*` (except `head_types.py`), `xtx4/export/onnx_ncnn.py`, `scripts/profile_model.py`, `scripts/export_random.py`, `tests/test_models.py`, `tests/test_export.py` | losses, data, engine |
| losses | `xtx4/losses/*`, `tests/test_assign.py`, `tests/test_losses.py` | models (import only `head_types`), data, engine |
| metrics | `xtx4/engine/metrics.py`, `scripts/evaluate.py`, `tests/test_metrics.py`, `reports/B0_*.json` | everything else |
| data | `xtx4/data/*` (except `preprocess.py`), `scripts/{build_cache,data_audit}.py`, `tests/test_data.py` | everything else |
| trainer (wave 2) | `xtx4/engine/{trainer,optim}.py`, `train.py`, `train.bat`, `xtx4/api.py`, `tests/test_trainer_smoke.py` | – |
| runtime (wave 2) | `xtx4/runtime/*`, `xtx4/export/{deploy,manifest}.py`, `live_test/`, `tests/test_runtime.py` | – |

If an interface in this file turns out to be impossible, change it here first, explain why in the
change, and keep the other owners' assumptions intact wherever possible.
