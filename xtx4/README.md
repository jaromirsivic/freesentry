# XTX4 Pose

Box-free detekce **9 landmarků hlavy a torza** (nos, oči, uši, ramena, boky) pro více osob v jednom
snímku. Nástupce XTX3: husté 3×3 RepConv bloky (rychlé v ncnn na Raspberry Pi 5), přiřazování
pozitivních anchorů podle OKS místo IoU boxu (Pose-TAL), duální hlava one-to-many + one-to-one bez NMS,
RLE + Smooth-OKS ztráta, přímý multi-scale render 256/320/384, torch-free runtime pro ncnn.

Dokumenty: `implementation_plan_fable.md` (návrh a zdůvodnění), `CONTRACTS.md` (závazná rozhraní
modulů), `EXPERIMENTS.md` (evidence běhů a baseline B0), `RESEARCH.md`, `reports/` (B0, audit dat,
rozpočty modelu).

## Varianty

| preset | šířky | GMAC @320 | parametry (nasazení) | účel |
|---|---|---|---|---|
| `u_a` | 16/24/48/96/128 | 0,40 | 0,99 M | Pi 5, nejrychlejší |
| `u_b` (= `u`) | 16/32/64/128/192 | 0,77 | 2,03 M | Pi 5, výchozí kandidát |
| `u_c` | 24/48/96/192/256 | 1,54 | 3,94 M | Pi 5 s rezervou / `n` |
| `l` | 48/96/192/384/512 | 7,7 | 18,9 M | GPU, teacher |
| `xtx3u_ctrl` | XTX3-u topologie (DW/PConv, SiLU) | 0,34 | 0,62 M | kontrola receptu |

Úrovně hlavy `P3P4P5` (výchozí), `P3P5`, `P4P5`; aktivace `relu` (výchozí), `hswish`, `silu`.
Kompletní tabulka: `reports/model_budget.md`.

## Rychlý start

```bat
init.bat                      :: vytvoří .venv (uv) s torch cu130, ncnn, pnnx, pycocotools ...
train.bat --config configs/u.json --dataset D:/xtxtraining
```

Výstupy běhu jdou do `run.output_root/run.name` (výchozí `D:/xtx4_runs/<run>`): `train.log`,
TensorBoard `tensorboard/`, `last.pt`, `best.pt`, `profile.json`, `eval_report.json`, `export/`
(ONNX + ncnn + `manifest.json` po skončení).

Užitečné přepínače `train.py`: `--variant u_b`, `--epochs N`, `--batch-size N`, `--resume PATH`,
`--init-weights PATH`, `--calibrate-loss 50`, `--set model.levels=P3P5` (libovolný klíč configu).

## Evaluace, export, benchmark

```bat
.venv\Scripts\python.exe scripts\evaluate.py --model-kind xtx4 --weights D:/xtx4_runs/xtx4-u/best.pt --dataset D:/xtxtraining --split test --sizes 256 320 384 --device cuda --out reports/xtx4u.json
.venv\Scripts\python.exe scripts\export_checkpoint.py --weights D:/xtx4_runs/xtx4-u/best.pt --sizes 256 320 384 --fp16
.venv\Scripts\python.exe scripts\export_random.py --out bench/graphs      :: grafy s náhodnými vahami pro měření latence
python scripts/bench_pi.py --index bench/graphs/index.json --images ~/frames --threads 1 2 3 4 --fp16 on off --out reports/pi_latency.csv   :: na Raspberry Pi 5 (jen numpy+cv2+ncnn+psutil)
```

Živý test z kamery (bez torch): `live_test/start.bat` (viz `live_test/README.md`).

## Testy

```bat
.venv\Scripts\python.exe -m pytest tests -q
```

## Struktura

```
xtx4/
  utils/     keypoints, geometry, logging, tensorboard
  data/      annotations (COCO/XTX/CrowdPose/AIC/MPII), manifest, cache, augment, sampler, dataset, preprocess
  models/    head_types (kontrakt), blocks, backbone, neck, head, xtx4, scaling
  losses/    targets, oks, assign (Pose-TAL), rle, pose_loss, calibrate
  engine/    checkpoint, optim, trainer, metrics
  export/    onnx_ncnn, deploy, manifest
  runtime/   decode, preprocess, ncnn_runner, pipeline   (bez torch)
  api.py     load_model, detect_poses
```
