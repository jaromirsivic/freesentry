# XTX Pose

Extremely fast, **NMS-free**, **DFL-free** human pose estimation in pure PyTorch, in three
variants (`XTX-n`, `XTX-m`, `XTX-l`). Single class (`person`), 17 COCO keypoints, RLE keypoint
head, and a distinctive **3-scale center-focused crop cascade (A/B/C)** that recovers small,
distant people in the centre of the frame.

Design is *inspired by* Ultralytics YOLO26-pose but has **no dependency** on `ultralytics`,
`mmpose`, `detectron2`, or any pose framework. Everything (network, losses, data pipeline,
training loop, inference, export) is implemented from scratch.

## Install

```bash
pip install -r requirements.txt
# (this repo) the workspace uses uv:
uv pip install -r requirements.txt
```

NCNN export tooling for Raspberry Pi is platform-specific and installed separately:

```bash
pip install pnnx ncnn        # converter (pnnx, preferred) + runtime
# alternatively use onnx2ncnn from the ncnn release tools
```

## Library usage

```python
import cv2
from xtx import load_model, detect_poses

model = load_model("best.pt", variant="n", device="cpu")
img = cv2.imread("scene.jpg")            # HxWx3 uint8 BGR
poses = detect_poses(img, model=model)
for p in poses:
    print(p.score, p.keypoints)          # keypoints: (17, 3) [x_pixel, y_pixel, conf]
```

`detect_poses` returns a list of `Pose` objects; coordinates are in the **original image**
pixel space. Each `Pose` also exposes `keypoints_norm`, `visibility`, `bbox_xyxy`, `score`,
`source_crop` and a `to_array()` helper returning `(56,)` = `[x1,y1,x2,y2,score, 17*(x,y,conf)]`.

## Training

```bash
python train.py                 # interactive: prompts for variant + dataset path
python train.py --variant n --dataset ./dataset      # non-interactive (CI)
python train.py --resume runs/xtx-n/last.pt
```

Dataset layout (COCO-style JSON, normalised bbox/keypoints):

```
dataset/
  train/  img0001.jpg  img0001.json   # one JSON per image, or one big JSON per folder
  test/   ...
```

Checkpoints are written every `checkpoint_interval_minutes` (default 5) plus `last.pt`/`best.pt`.
After training, the merge thresholds are **calibrated** on `test/` and embedded in the checkpoint.

## Inference example

```bash
python infer.py --weights best.pt --image scene.jpg --variant n --save out.jpg --json out.json
```

## Export (XTX-n -> NCNN)

```bash
python -m xtx.export.ncnn_export --weights best.pt --out xtx_n
python scripts/benchmark.py --weights xtx_n --device ncnn   # FPS on Raspberry Pi
```

## How it works

```
image -> normalise to >=1080 square -> center square (S x S, S=1080)
       -> A: full square     -> 384x384  (large/medium people)
       -> B: central 768 -> 384 (2x)     (medium/small central)
       -> C: central 384 native (no resize) (smallest, distant central)
       -> batch (3,3,384,384) -> one forward -> decode per crop
       -> map back to original pixels -> cross-crop merge (priority A > B > C)
```

See `config.json` for all tunable defaults and `tests/` for geometry/merge/forward checks.
