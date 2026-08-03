# XTX3 Pose

Efficient, from-scratch PyTorch **head-and-torso** pose estimation. XTX3 is the
specialised successor of XTX2: it detects people and only the **9 landmarks**
that matter for head/torso analysis — nose, eyes, ears, shoulders, hips — which
makes every variant smaller and faster than its 17-keypoint XTX2 counterpart.
Same proven core: NMS-free dual head, DFL-free boxes, RLE keypoints, ProgLoss,
STAL, MuSGD; **no dependency** on `ultralytics` or any pose framework.

Four variants:

| Variant | Params | GFLOPs/pass | Input | Target |
|---------|--------|-------------|-------|--------|
| `XTX3-u` | 0.82M | 0.25 @256 / 0.39 @320 / 0.56 @384 | 256/320/384, whole-frame | **Raspberry Pi 5 real-time (>=25 FPS via NCNN)** |
| `XTX3-n` | 2.65M | 1.51 @384 | 384, whole or pyramid | Raspberry Pi via NCNN |
| `XTX3-m` | 19.68M | 8.85 @384 | 384 | PC CPU/GPU |
| `XTX3-l` | 25.22M | 15.05 @384 | 384, adds P2 level | PC GPU |

## How it works

Two input strategies, selected per variant in `config.json`
(`inference.view_mode`):

- **whole** (default for `u`): the frame is letterboxed into a single `S x S`
  view (aspect-preserving, never upscaled) — one network pass per frame. The
  network is fully convolutional, so the same `u` weights run at 256, 320 or
  384 (trained multi-scale across all three).
- **pyramid** (default for `n`/`m`/`l`): three 384x384 views cut from a virtual
  1536x1536 canvas (whole image at 4x reduction, central quarter at 2x, central
  1/16 native) — exactly the XTX2 geometry, enabling a like-for-like
  comparison. Views run as one batched forward; detections are mapped back via
  per-view affines and deduplicated with priority view 1 > 2 > 3.

Output per detection: 32 floats `[x1, y1, x2, y2, score, 9*(x, y, conf)]` plus
a per-keypoint visibility class {absent, occluded, visible}. No NMS anywhere.

## Quick start

```bat
:: Windows -- double-click, or run from a console:
train.bat   :: interactive training (variant u/n/m/l, dataset path, checkpoints)
infer.bat   :: interactive inference on a single image
```

Both launchers create `.venv` (via `uv`) and install `requirements.txt` plus
the correct CUDA/CPU torch build on first run.

Manual setup:

```bash
uv venv && .venv\Scripts\activate    # or python -m venv .venv
uv pip install -r requirements.txt
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python train.py --variant u --dataset D:\xtxtraining
python infer.py --weights runs/xtx3-u/best.pt --image scene.jpg --save out.jpg
```

## Library API

```python
import cv2
from xtx3 import load_model, detect_poses

model = load_model("best.pt", device="cpu")
img = cv2.imread("scene.jpg")
poses = detect_poses(img, model=model)          # mode/size come from the checkpoint config
for p in poses:
    print(p.score, p.keypoints)                 # (9,3) [x_pixel, y_pixel, conf]
```

Each `Pose` carries `bbox_xyxy`, `score`, `keypoints` (9x3, original-image
pixels), `keypoints_norm`, `visibility` (per-keypoint class in {0,1,2}) and
`source_view` (0 = whole frame, 1|2|3 = pyramid). `pose.to_array()` flattens to
the `(32,)` layout `[x1,y1,x2,y2,score, 9*(x,y,conf)]`;
`detect_poses_batch(images)` batches efficiently across images and views.
`detect_poses(..., view_mode="whole", network_size=256)` overrides the defaults.

## Dataset layout

Same layout and format as XTX2 (e.g. `D:\xtxtraining`) — XTX3 reads 17-point
annotations directly and keeps the 9 retained landmarks deterministically:

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
list of 17x3 (or already 9x3) `[x_norm, y_norm, visibility]` with visibility
`0` (absent), `1` (occluded), `2` (visible). Official COCO
`person_keypoints_*.json` files (absolute pixel coordinates) are also detected
and handled automatically; `scripts/convert_coco.py` converts a raw COCO 2017
download into this layout up front if preferred.

## Training

`python train.py` (or `train.bat`) asks, in order: resume from checkpoint (only
when checkpoints exist under `./runs`), variant `[u/n/m/l]`, dataset path
(Enter -> `./dataset`), and checkpoint directory. Non-interactive flags:
`--variant --dataset --output --config --resume --device --calibrate`.

Defaults from `config.json`: **50 epochs**, checkpoint every **5 minutes** plus
`last.pt` per epoch and `best.pt` on improved validation 9-point OKS mAP.
Training uses the dual head with **ProgLoss** balancing, **STAL** small-target
assignment, **MuSGD** (fallback `training.optimizer: "sgd"`), EMA weights, AMP
on CUDA, and warmup + cosine LR. The `u` variant trains **multi-scale**
(256/320/384 per batch) so one set of weights serves all export sizes.
Validation prints AP/AP50/AP75/AR plus a detailed report (detection P/R,
per-landmark OKS, visibility accuracy, face/torso scale buckets). In pyramid
mode a final **merge calibration** grid-searches the cross-view dedup
thresholds on `test/` and embeds them into `best.pt` and `config.json`.

After training finishes, `best.pt` is **automatically exported to ONNX and
NCNN** (toggle `export.auto_export` in `config.json`); the `u` variant is
exported at all of 256/320/384.

## NCNN export (Raspberry Pi 5)

```bash
python -m xtx3.export.ncnn_export --weights runs/xtx3-u/best.pt --out xtx3_u
```

Pipeline: `fuse()` (Conv+BN fold, RepConv reparam, one-to-many head dropped) ->
ONNX (opset 17, raw head tensors `box`/`cls`/`kpt`, input blob `in0`) ->
`onnxsim` -> numeric verification against PyTorch -> NCNN via **pnnx**
(preferred) or `onnx2ncnn`, producing `xtx3_u.ncnn.param` + `xtx3_u.ncnn.bin`
(per size for `u`: `..._256`, `..._320`, `..._384`).

Converter installation (separate, platform-specific step):

```bash
pip install pnnx          # preferred converter
pip install ncnn          # python runtime for NCNNPoseRunner / benchmarks
# or build ncnn from source for onnx2ncnn: https://github.com/Tencent/ncnn
```

Raw-NCNN inference on the Pi:

```python
from xtx3.export.ncnn_export import NCNNPoseRunner
runner = NCNNPoseRunner("xtx3_u.ncnn.param", "xtx3_u.ncnn.bin", strides=[8, 16, 32])
per_view, orig_size = runner.detect(image, mode="whole", network_size=320)
```

The decode contract for the raw outputs is documented in
`xtx3/export/ncnn_export.py`.

## Scripts and tests

```bash
python scripts/profile_flops.py            # params/GFLOPs vs budgets + XTX2 comparison
python scripts/benchmark.py --variant u --mode whole --size 320
python scripts/benchmark.py --backend ncnn --param xtx3_u.ncnn.param --bin xtx3_u.ncnn.bin
python scripts/convert_coco.py --coco D:\coco2017 --out .\dataset
pytest tests/                              # 91 tests: geometry, annotations, model, losses, merge, metrics
```

## Project structure

```
xtx3/
  train.py / infer.py / train.bat / infer.bat / config.json
  xtx3/
    api.py                 # Pose, load_model, detect_poses, detect_poses_batch
    models/                # blocks (PConv/RepConv/CSP/SPPF), backbone, PAN neck, dual 9-kpt head
    data/                  # annotations (17->9), whole-frame + pyramid preprocessing, dataset, augment
    losses/                # RLE, TAL + STAL + centre prior, CIoU/varifocal, ProgLoss
    engine/                # trainer, MuSGD, checkpoint/EMA, 9-pt OKS metrics, cross-view merge
    export/                # ONNX + NCNN export (multi-size for u), raw-NCNN runner
    utils/                 # 9-landmark metadata, affine geometry, logging
  scripts/                 # profile_flops.py, benchmark.py, convert_coco.py
  tests/
  xtx3_architecture.md     # full architecture description (per-variant diagrams)
  report.md                # measured-results template vs XTX2
```
