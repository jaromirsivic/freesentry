# XTX4 Pose

Box-free detekce **9 landmarků hlavy a torza** (nos, oči, uši, ramena, boky) pro více osob v jednom
snímku. Nástupce XTX3: husté 3×3 RepConv bloky (rychlé v ncnn na Raspberry Pi 5), přiřazování
pozitivních anchorů podle OKS místo IoU boxu (Pose-TAL), duální hlava one-to-many + one-to-one bez NMS,
RLE + Smooth-OKS ztráta, přímý multi-scale render 256/320/384, torch-free runtime pro ncnn.

Dokumenty: `implementation_plan_fable.md` (návrh a zdůvodnění), `CONTRACTS.md` (závazná rozhraní
modulů), `EXPERIMENTS.md` (evidence běhů a baseline B0), `RESEARCH.md`, `reports/` (B0, audit dat,
rozpočty modelu).

## Varianty

| preset | šířky | GMAC @320 | parametry (nasazení) | Pi 5 p50 @256/320/384 (3 vl., FP16) | účel |
|---|---|---|---|---|---|
| `u_a` | 16/24/48/96/128 | 0,40 | 0,99 M | 5,0 / 7,6 / 10,9 ms | Pi 5, nejrychlejší |
| `u_b` (= `u`) | 16/32/64/128/192 | 0,77 | 2,03 M | 7,7 / 11,7 / 17,0 ms | Pi 5, **výchozí** (gate G1 splněn) |
| `u_c` | 24/48/96/192/256 | 1,54 | 3,94 M | 13,0 / 19,9 / 28,8 ms | Pi 5 s rezervou (@320 ≤ 25 ms) / `n` |
| `l` | 48/96/192/384/512 | 7,7 | 18,9 M | – | GPU, teacher |
| `xtx3u_ctrl` | XTX3-u topologie (DW/PConv, SiLU) | 0,34 | 0,62 M | 6,7 / 11,3 / 17,8 ms | kontrola receptu |

Úrovně hlavy `P3P4P5` (výchozí), `P3P5`, `P4P5`; aktivace `relu` (výchozí), `hswish`, `silu`.
Kompletní tabulka: `reports/model_budget.md`; latence na Pi 5 (všechny grafy, vlákna 1–4, FP16/FP32, end-to-end):
`reports/pi_latency*.csv` a EXPERIMENTS.md „Gate G1“. Doporučené nasazení: u_b @320, 3 vlákna, FP16, OKS-NMS na hostu
(end-to-end 13 ms ≈ 73 FPS); u_b @384 (20 ms) a u_c @320 (20 ms) se do rozpočtu ≤ 25 ms vejdou také.

## Rychlý start

```bat
init.bat                      :: vytvoří .venv (uv) s torch cu130, ncnn, pnnx, pycocotools ...
train.bat --config configs/u.json --dataset D:/xtxtraining
train_combined.bat            :: u_b a pak u_c, 300 epoch každá; po pádu PC stačí spustit znovu (pokračuje od poslední dokončené epochy)
tensorboard.bat               :: TensorBoard pro všechny běhy v D:/xtx4_runs (nebo tensorboard.bat <běh> [port]); ukončení stop_tensorboard.bat
```

Dvojklik na `train.bat` bez parametrů spustí interaktivní režim: zeptá se na pokračování z checkpointu,
variantu, dataset, **počet epoch** (výchozí 300 z `configs/u.json`) a adresář běhu; Enter = výchozí hodnota.
Počet epoch určuje i rozvrh (kosinový pokles LR k nule, vypnutí mosaic v posledních 15 epochách), proto
ho volte předem a při `--resume` zadejte stejné `--epochs`.

Výstupy běhu jdou do `run.output_root/run.name` (výchozí `D:/xtx4_runs/<run>`): `train.log`,
TensorBoard `tensorboard/`, `last.pt`, `best.pt`, `epochs/epoch_NNNN.pt` (posledních
`training.keep_epochs` = 5 dokončených epoch, číslováno jako v logu `Epoch N/Y`), `profile.json`,
`eval_report.json`, `export/` (ONNX + ncnn + `manifest.json` po skončení).

Užitečné přepínače `train.py`: `--variant u_b`, `--epochs N`, `--batch-size N`, `--resume PATH`
(`last.pt`, `best.pt` nebo `epochs/epoch_0098.pt` – pokračuje se další epochou), `--init-weights PATH`,
`--calibrate-loss 50`, `--keep-epochs N` (0 = bez souborů `epochs/`), `--thermal-guard` /
`--no-thermal-guard`, `--set model.levels=P3P5` (libovolný klíč configu).

### Odolnost proti pádu (restart PC, přehřátí)

- **Rolling checkpointy epoch:** po každé dokončené epoše se uloží plný stav do `epochs/epoch_NNNN.pt`
  (stejný obsah jako `last.pt`, `last.pt` vzniká kopií). Drží se jen nejnovějších `training.keep_epochs`
  souborů (podle času zápisu, mazání až po úplném zápisu nového souboru). Poškozený `last.pt` po
  tvrdém restartu tak nestojí celý běh: `train.py --resume D:/xtx4_runs/xtx4-u/epochs/epoch_0098.pt`.
- **Kombinovaný trénink** `train_combined.bat` (`scripts/train_combined.py`): trénuje u_b a pak u_c, 300 epoch,
  běhy `D:/xtx4_runs/f6_u_b_300ep` a `f6_u_c_300ep`, průběh v `D:/xtx4_runs/f6_combined.log`. Po pádu ho stačí
  spustit znovu: hotovou variantu (soubor `combined_done.json`) přeskočí a přerušenou obnoví z nejnovějšího
  nepoškozeného `epochs/epoch_NNNN.pt` (poškozený zápis pozná podle CRC a vezme starší). Chyba `train.py` se
  2× automaticky zopakuje. Varianty běží záměrně za sebou, ne paralelně (EXPERIMENTS.md, „Paralelní trénink“).
  `train_combined.bat --dry-run` jen vypíše, co by udělal.
- **Teplotní pojistka** (`training.thermal`, výchozí zapnuto): vlákno čte každých 10 s teplotu GPU a
  příznaky thermal throttlingu (`nvidia-smi`, případně `pynvml`), na Linuxu i CPU (`psutil`). Při
  GPU ≥ 85 °C (CPU ≥ 95 °C, nebo aktivním HW thermal slowdown) se trénink po dokončení kroku pozastaví
  (`thermal pause: gpu 86 C >= 85 C`) a pokračuje až po ochlazení na ≤ 75 °C (CPU ≤ 85 °C):
  `thermal resume after 123 s (gpu 74 C)`. Pauza delší než `max_pause_min` (60) jen loguje varování,
  běh se nikdy neukončí. Čas pauzy se nepočítá do img/s ani ETA; v logu kroku je aktuální teplota, na
  konci epochy maximum a součet pauz; TensorBoard `sys/gpu_temp_c`, `sys/thermal_pause_s`.
  Bez senzoru (nebo na CPU) je pojistka neaktivní. Vypnutí: `--no-thermal-guard` nebo
  `--set training.thermal.enabled=false`.

## Evaluace, export, benchmark

```bat
.venv\Scripts\python.exe scripts\evaluate.py --model-kind xtx4 --weights D:/xtx4_runs/xtx4-u/best.pt --dataset D:/xtxtraining --split test --sizes 256 320 384 --device cuda --out reports/xtx4u.json
.venv\Scripts\python.exe scripts\export_checkpoint.py --weights D:/xtx4_runs/xtx4-u/best.pt --sizes 256 320 384 --fp16
.venv\Scripts\python.exe scripts\export_random.py --out bench/graphs      :: grafy s náhodnými vahami pro měření latence
python scripts/bench_pi.py --index bench/graphs/index.json --images ~/frames --threads 1 2 3 4 --fp16 on off --cool-below 70 --out reports/pi_latency.csv   :: na Raspberry Pi 5 (jen numpy+cv2+ncnn+psutil); --cool-below čeká na ochlazení SoC
python scripts/bench_pi_e2e.py --export <export dir> --images ~/frames --sizes 256 320 384 --threads 2 3 4 --nms on off --out reports/pi_latency_e2e.csv   :: celý řetězec snímek -> pózy (NcnnPoseRunner), bez torch
python scripts/summarize_pi_latency.py reports/pi_latency.csv --e2e reports/pi_latency_e2e.csv   :: Markdown tabulky z CSV
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
