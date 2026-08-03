# XTX3 — výsledková zpráva

Šablona pro zaznamenání měřených výsledků. Sekce 1 a 2 jsou změřené na
vývojovém stroji; sekce 3–5 se doplní po plném tréninku na RTX 4070 a po
benchmarku na Raspberry Pi 5.

## 1. Rozpočty modelů (změřeno, `scripts/profile_flops.py`)

| Varianta | Parametry | GFLOPs/pass | Rozpočet | Stav |
|----------|-----------|-------------|----------|------|
| XTX3-u @320 | 0,82 M | 0,389 | ≤1,6 M / ≤0,75 | OK |
| XTX3-u @256 | 0,82 M | 0,249 | – | OK |
| XTX3-u @384 | 0,82 M | 0,561 | – | OK |
| XTX3-n @384 | 2,65 M | 1,511 | 2,0–3,0 M / ≤2,5 | OK |
| XTX3-m @384 | 19,68 M | 8,851 | 16–22 M / ≤24 | OK |
| XTX3-l @384 | 25,22 M | 15,052 | 22–28 M / ≤30 | OK |

Srovnání s XTX2 (stejný backbone, 9bodová vs 17bodová hlava):

| Pár | Parametry | GFLOPs/pass |
|-----|-----------|-------------|
| n: XTX2 → XTX3 | 2,68 M → 2,65 M | 1,525 → 1,511 (rychlejší) |
| m: XTX2 → XTX3 | 19,72 M → 19,68 M | 8,870 → 8,851 (rychlejší) |
| l: XTX2 → XTX3 | 25,26 M → 25,22 M | 15,113 → 15,052 (rychlejší) |

## 2. Testy a desktop benchmark (změřeno)

- `pytest tests/` — **91/91 testů prošlo** (geometrie, anotace 17→9, model,
  ztráty/STAL/ProgLoss/MuSGD, merge, metriky).
- Desktop CPU (PyTorch, orientačně): XTX3-u whole@320 ≈ 36 FPS včetně
  preprocessingu a dekódování; preprocessing whole@320 ≈ 1,3 ms/snímek.

## 3. Přesnost po tréninku (doplnit po tréninku na RTX 4070)

Trénink: `train.bat`, dataset `D:\xtxtraining`, 50 epoch.

| Model | OKS AP | AP50 | AP75 | AR | Det. precision | Det. recall | Vis. acc. |
|-------|--------|------|------|----|----------------|-------------|-----------|
| XTX3-u @320 (whole) | – | – | – | – | – | – | – |
| XTX3-u @256 (whole) | – | – | – | – | – | – | – |
| XTX3-u @384 (whole) | – | – | – | – | – | – | – |
| XTX3-n @384 (pyramid) | – | – | – | – | – | – | – |
| XTX3-m @384 (pyramid) | – | – | – | – | – | – | – |
| XTX3-l @384 (pyramid) | – | – | – | – | – | – | – |

Cíl: XTX3-n ≥ XTX2-n na zachovaných 9 landmarcích (9bodové OKS AP; XTX2-n
referenci lze spočíst vyhodnocením XTX2 predikcí po výběru 9 bodů).

| Reference | 9bodové OKS AP | Poznámka |
|-----------|----------------|----------|
| XTX2-n @384 (pyramid) | – | doplnit ze stávajícího checkpointu XTX2 |

## 4. Rychlost na Raspberry Pi 5 (doplnit po benchmarku na zařízení)

NCNN, 4 vlákna, `scripts/benchmark.py --backend ncnn`:

| Model | Vstup | ms/snímek | FPS | Cíl |
|-------|-------|-----------|-----|-----|
| XTX3-u | 256 whole | – | – | ≥25 FPS |
| XTX3-u | 320 whole | – | – | **≥25 FPS (hlavní cíl)** |
| XTX3-u | 384 whole | – | – | – |
| XTX3-n | 384 pyramid (3 průchody) | – | – | ≥13 FPS (XTX2-n baseline) |
| XTX2-n | 384 pyramid (referenčně) | – | ~13 | baseline |

Výchozí rozlišení varianty `u` pro nasazení se zvolí z tabulek 3 + 4
(nejvyšší přesnost při dodržení ≥25 FPS).

## 5. Export (doplnit po tréninku)

| Model | ONNX verifikace (max. odchylka) | NCNN konverze | Poznámka |
|-------|--------------------------------|---------------|----------|
| XTX3-u 256/320/384 | – | – | auto-export po tréninku |
| XTX3-n 384 | – | – | |
| XTX3-m 384 | – | – | |
| XTX3-l 384 | – | – | |
