# XTX2 Pose

Extremely fast, from-scratch PyTorch human-pose estimation with a **3-level
centre-focused image pyramid**. Three variants — `XTX2-n` (Raspberry Pi via
NCNN), `XTX2-m` and `XTX2-l` (PC CPU/GPU) — detect people and their 17 COCO
keypoints. Inspired by YOLO26-pose (NMS-free dual head, DFL-free boxes, RLE
keypoints, ProgLoss, STAL, MuSGD) with **no dependency** on `ultralytics` or
any pose framework.

## How it works

Every input image is letterboxed onto a virtual **1536×1536 canvas**
(aspect-preserving, never upscaled, black padding) and three **384×384**
levels are cut from it, each with at most one interpolation directly from the
original image:

| Level | Canvas region | Reduction | Field of view |
|-------|---------------|-----------|----------------|
| level_1 | whole 1536 canvas | exact 4× | the entire image |
| level_2 | centred 768×768 | exact 2× | central quarter |
| level_3 | centred 384×384 | none (pure slice) | central 1/16, native resolution |

The three levels run as **one batched forward** `(3, 3, 384, 384)` through the
NMS-free one-to-one head, are decoded per level, mapped back to original-image
pixels via per-level affine transforms, and merged with priority
**level_1 > level_2 > level_3** so no person appears twice. level_3 recovers
small, distant people at the centre of the frame that level_1 cannot resolve.

## Quick start

```bat
:: Windows -- double-click, or run from a console:
train.bat   :: interactive training (variant, dataset path, checkpoints)
infer.bat   :: interactive inference on a single image
```

Both launchers create `.venv` and install `requirements.txt` on first run.

Manual setup:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt
python train.py --variant n --dataset ./dataset
python infer.py --weights runs/xtx2-n/best.pt --image scene.jpg --save out.jpg
```

## Library API

```python
import cv2
from xtx2 import load_model, detect_poses

model = load_model("best.pt", device="cpu")
img = cv2.imread("scene.jpg")
poses = detect_poses(img, model=model)
for p in poses:
    print(p.score, p.keypoints)   # (17,3) [x_pixel, y_pixel, conf]
```

Each `Pose` carries `bbox_xyxy`, `score`, `keypoints` (17×3, original-image
pixels), `keypoints_norm`, `visibility` (per-keypoint class in {0,1,2}) and
`source_level` (1|2|3). `pose.to_array()` flattens to the YOLO26-style
`(56,)` layout `[x1,y1,x2,y2,score, 17*(x,y,conf)]`;
`detect_poses_batch(images)` batches efficiently across images and levels.

## Dataset layout

```
dataset/
  train/
    img0001.jpg
    img0001.json     # COCO-style; one JSON per image OR one JSON for many images
    ...
  test/
    ...
```

Annotations use **normalised** `bbox` `[x, y, w, h]` and a flat `keypoints`
list of 17×3 `[x_norm, y_norm, visibility]` with visibility `0` (absent),
`1` (occluded), `2` (visible). All `*.json` under the split folder are scanned
recursively; malformed records are warned about and skipped.

## Training

`python train.py` (or `train.bat`) asks, in order: resume from checkpoint
(only when checkpoints exist under `./runs`), variant `[n/m/l]`, dataset path
(Enter → `./dataset`), and checkpoint directory. Non-interactive flags:
`--variant --dataset --output --config --resume --device --calibrate`.

Defaults from `config.json`: **50 epochs**, checkpoint every **5 minutes**
plus `last.pt` per epoch and `best.pt` on improved validation OKS mAP.
Training uses the dual head with **ProgLoss** balancing (printed each log
line), **STAL** small-target assignment, a **MuSGD** optimizer
(`training.optimizer: "sgd"` is the fallback), EMA weights, AMP on CUDA, and
warmup + cosine LR. After the last epoch a **merge calibration** grid-searches
the cross-level dedup thresholds on `test/` and embeds them into `best.pt`
and `config.json`.

## NCNN export (Raspberry Pi, XTX2-n)

```bash
python -m xtx2.export.ncnn_export --weights runs/xtx2-n/best.pt --out xtx2_n
```

Pipeline: `fuse()` (Conv+BN fold, RepConv reparam, one-to-many head dropped) →
ONNX (opset 17, raw head tensors `box`/`cls`/`kpt`, input blob `in0`) →
`onnxsim` → NCNN via **pnnx** (preferred) or `onnx2ncnn`, producing
`xtx2_n.ncnn.param` + `xtx2_n.ncnn.bin`.

Converter installation (separate, platform-specific step):

```bash
pip install pnnx          # preferred converter
pip install ncnn          # python runtime for NCNNPoseRunner / benchmarks
# or build ncnn from source for onnx2ncnn: https://github.com/Tencent/ncnn
```

Raw-NCNN inference on the Pi:

```python
from xtx2.export.ncnn_export import NCNNPoseRunner
runner = NCNNPoseRunner("xtx2_n.ncnn.param", "xtx2_n.ncnn.bin", strides=[8, 16, 32])
per_level, orig_size = runner.detect(image)   # decode + merge in host code
```

Decode contract for the raw outputs is documented in
`xtx2/export/ncnn_export.py`.

## Scripts and tests

```bash
python scripts/profile_flops.py            # params/GFLOPs vs the section 6.6 budgets + old ./xtx
python scripts/benchmark.py                # ms/frame + FPS (torch), preprocessing microbenchmark
python scripts/benchmark.py --backend ncnn --param xtx2_n.ncnn.param --bin xtx2_n.ncnn.bin
pytest tests/                              # geometry, annotations, merge, model, STAL/ProgLoss/MuSGD
```

## Project structure

```
xtx2/
  train.py / infer.py / train.bat / infer.bat / config.json
  xtx2/
    api.py                 # Pose, load_model, detect_poses, detect_poses_batch
    models/                # blocks (PConv/RepConv/CSP/SPPF), backbone, PAN neck, dual head
    data/                  # annotations, 1536-canvas preprocessing, dataset, augment
    losses/                # RLE, TAL + STAL + centre prior, CIoU/varifocal, ProgLoss
    engine/                # trainer, MuSGD, checkpoint/EMA, OKS metrics, cross-level merge
    export/                # ONNX + NCNN export, raw-NCNN runner
    utils/                 # keypoint metadata, affine geometry, logging
  scripts/                 # profile_flops.py, benchmark.py
  tests/
```
