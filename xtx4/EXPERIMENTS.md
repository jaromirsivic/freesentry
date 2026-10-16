# XTX4 – evidence experimentů

Každý běh = jeden řádek. `run_id` je název složky v `runs/`. Parent = checkpoint, ze kterého se startovalo (nebo `scratch`). Jedna změna na řádek proti parentu. Hashe: SHA-256 vah (EMA, stejné jako export) a manifestu dat.

| run_id | datum | parent | varianta / změna | config | data manifest | epochy | img/s | peak VRAM | AP9@320 (val) | AP50 | AP75 | dup@0,25 | Pi ms p50 | poznámka |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_xtx3u_256 | 2026-10-11 | xtx3/runs/xtx3-u/best.pt | XTX3-u, nová evaluace, plný test 11 320 obr. | xtx3 | – | 50 (XTX3) | 28 | – | 0,4480 (@256) | 0,6876 | 0,4658 | 0,008 | – | baseline, P/R@0,25 0,707/0,686 |
| **B0_xtx3u_320** | 2026-10-11 | xtx3/runs/xtx3-u/best.pt | XTX3-u, nová evaluace, plný test | xtx3 | – | 50 (XTX3) | 28 | – | **0,4720** | 0,7097 | 0,4970 | 0,008 | – | **hlavní baseline**; P/R@0,25 0,685/0,722; face-PCK@0,1 0,28 (<8 px: 0,08); AR 0,589 |
| B0_xtx3u_384 | 2026-10-11 | xtx3/runs/xtx3-u/best.pt | XTX3-u, nová evaluace, plný test | xtx3 | – | 50 (XTX3) | 28 | – | 0,4711 (@384) | 0,7011 | 0,5033 | 0,011 | – | AR 0,603; recall obličejů <8 px 0,756 vs 0,703 @320 |
| B0_xtx3u_320_raw_o2o | 2026-10-11 | xtx3/runs/xtx3-u/best.pt | XTX3-u bez merge (čistá o2o hlava) | xtx3 | – | 50 (XTX3) | – | – | 0,4252 | 0,6023 | 0,4594 | **3,24** | – | XTX3 „NMS-free“ jen díky OKS/IoU merge v `merge_views`; o2o hlava sama duplicity nepotlačuje |
| B0_xtx3n_whole_384 | 2026-10-11 | xtx3/runs/xtx3-n/best.pt | XTX3-n (trénink pyramid, 27 ep.) vyhodnocena whole | xtx3 | – | 27 (XTX3) | – | – | 0,2908 | 0,5218 | 0,2835 | 0,000 | – | skóre stlačená (R@0,25 jen 0,03) |
| B0_xtx3n_pyramid_384 | 2026-10-11 | xtx3/runs/xtx3-n/best.pt | XTX3-n pyramid (3 průchody) | xtx3 | – | 27 (XTX3) | – | – | 0,2712 | 0,5054 | 0,2527 | 0,000 | – | horší a 3,5× pomalejší než whole (D7 potvrzeno) |
| **f3_ctrl_xtx3u_50ep** | 2026-10-11 | scratch | **F3 kontrola receptu**: XTX3-u topologie (`xtx3u_ctrl`, DW/PConv, SiLU, 0,34 GMAC) + nový recept (box-free Pose-TAL, SOKS+RLE, ProgLoss 0,8→0,1, přímý multi-scale render, mosaic s letterboxem, SGD) | configs/u.json + overrides | reports/manifest_train.json | 50 | 350–375 | 4,3 GB | **0,5316** (plný test) | 0,7106 | 0,5801 | 0,16 (bez NMS) | – | **+6,0 b. AP vs B0** při stejné architektuře; AR 0,710 (+12); P/R@0,25 0,819/0,594 (skóre stlačená, provozní bod volit sweepem); export v D:/xtx4_runs/f3_ctrl_xtx3u_50ep/export |
| **f4_u_b_100ep** | 2026-10-11 | scratch | **XTX4-u kandidát u_b** (16/32/64/128/192, RepConv 3×3, ReLU, P3P4P5, 0,77 GMAC @320, 2,03 M param. v nasazení), nový recept, 100 epoch (screening) | configs/u.json | reports/manifest_train.json | 100 | ~330 | – | **0,5907** (plný test; @256 0,549, @384 0,615) | 0,7542 | 0,6414 | 0,11 (bez NMS) | – | **+11,9 b. AP vs B0**; AR 0,746; matched OKS 0,895 (B0 0,847); face-PCK@0,1 0,40 (B0 0,28); běh přerušen pádem PC v ep. 100 a dokončen z last.pt; export D:/xtx4_runs/f4_u_b_100ep/export |
| f4_u_b_100ep + OKS-NMS 0,5 (host) | 2026-10-11 | – | tytéž váhy, NMS v runtime | – | – | – | – | – | **0,5956** (@256 0,554, @384 0,618) | 0,7794 | 0,6410 | 0,012 | – | F1@f1_max 0,758 vs B0 0,725; F1 při shodném FP/obr. 0,748 vs 0,703 (P/R 0,762/0,734 vs 0,685/0,722) |

## Rozhodnutí (ADR)

- 2026-10-11: architektura u = plné 3×3 RepConv (ne DW), box-free Pose-TAL, duální hlava NMS-free; viz `implementation_plan_fable.md` §5–6.
- 2026-10-11: evaluace zmrazena podle `CONTRACTS.md` §9 (celočíselné prahy, maxDet 100, práh skóre 0,001); historické číslo 0,4589 (200 obrázků) se už nepoužívá jako baseline.
- 2026-10-14: gate G1 (latence Pi 5) splněn – `u_b` @320 11,7 ms p50 (3 vlákna, FP16) zůstává variantou `u`; nasazení 3 vlákna, FP16, OKS-NMS na hostu; rezerva pro @384 nebo `u_c`. Viz „Gate G1“ níže.
- 2026-10-15: varianty se trénují **postupně, ne paralelně** – paralelní běh u_b + u_c nezrychlí trénink a vyčerpá virtuální paměť (viz „Paralelní trénink“ níže). `u_a` se dál nepoužívá.
- 2026-10-14: odolnost tréninku proti pádu PC – rolling checkpointy `epochs/epoch_NNNN.pt` (`training.keep_epochs` = 5) a teplotní pojistka `training.thermal` (výchozí zapnuto, GPU ≥ 85 °C pauza / ≤ 75 °C pokračování; ověřeno reálně na RTX 4070 s prahy 55/50 °C). Viz CONTRACTS §10–11.

## Zjištění z B0 (2026-10-11, `reports/B0_summary.md`)

- Historických 0,4589 (200 obr., práh 0,05, XTX3 evaluátor) je reprodukovatelných; na plném testu novou evaluací je XTX3-u @320 **AP9 0,4720, AP75 0,497** (XTX3 hlásila AP75 0,0 kvůli chybě D1). Rozdíl +0,016 vůči starému evaluátoru jde za 20 GT bez jediného labelovaného bodu, které XTX3 počítala jako missy.
- **XTX3 není NMS-free:** `merge_views` v režimu whole greedy zahazuje detekce s OKS ≥ 0,5 nebo IoU ≥ 0,5 (= NMS). Bez něj dává o2o hlava 3,24 duplicit/obr. při 0,25 a AP klesá na 0,425. XTX4 o2o hlava (topk 7→1, ProgLoss 0,8→0,1) to musí zvládnout sama; metrika dup/img@0,25 je povinná a runtime má OKS-NMS jako pojistku.
- Visibility hlava XTX3 prakticky nikdy nepredikuje „occluded“ (F1 0,04); macro-F1 0,548.
- Face-PCK@0,1 celkem 0,28, u obličejů < 8 px síťově 0,08; recall obličejů < 8 px @320 0,703, @384 0,756.
- pycocotools cross-check (9 sigem, maxDet 20): Δ < 1e-15 na všech metrikách.
- Dataset: 2 byte-identické obrázky sdílené train/test (train `000000341448.jpg` = test `000000000927.jpg`; train `000000410810.jpg` = test `000000373251.jpg`) – vyloučit z testu v manifestu; 10 duplicit uvnitř train, 1 uvnitř test.

## Průběžná diagnostika F3 (kontrola receptu: XTX3-u topologie + nový recept), snapshot epochy 25/50

Prvních 2000 obrázků testu, S=320 (`reports/f3_ctrl_ep25_sweep.json`, `reports/f3_ctrl_ep25_nms05.json`, `reports/B0_xtx3u_whole_320_n2000.json`):

| běh | AP | AP50 | AP75 | AR | F1 @f1_max | F1 @stejné FP/obr. | dup/obr. @fp_match |
|---|---|---|---|---|---|---|---|
| B0 XTX3-u (s merge = NMS) | 0,472 | 0,704 | 0,499 | 0,587 | 0,722 (práh 0,30) | 0,697 | 0,006 |
| F3 ep25 bez NMS | 0,489 | 0,675 | 0,534 | 0,683 | 0,660 (0,25) | 0,650 | 0,452 |
| F3 ep25 + OKS-NMS 0,5 (host) | **0,501** | **0,717** | **0,537** | 0,636 | 0,712 (0,15) | 0,687 | 0,122 |

- o2o hlava v polovině tréninku (w_o2o = 0,25→0,9) ještě nepotlačuje duplicity; OKS-NMS na hostu uzavírá ~80 % rozdílu F1. Zbytek: NMS škáluje OKS podle extent boxu bodů (menší než GT plocha), takže některé duplicity přežijí → zvážit NMS s IoU extent boxů navíc (jako XTX3 merge) nebo větší škálu.
- Skóre nového modelu jsou stlačená (provozní práh ~0,08–0,15 místo 0,25–0,30) – práh se musí volit na val (sweep je v evaluate.py).
- Deficit vůči B0 při shodném FP/obr.: torzo < 16 px recall 0,43 vs 0,52, okraje 0,55 vs 0,59; lepší lokalizace tam, kde detekuje (matched OKS 0,871 vs 0,850; face-PCK@0,1 v 16–32 px 0,52 vs 0,42).
- Viditelnost: s prahy present 0,6 / visible 0,8 macro-F1 0,646 (výchozí 0,5/0,5: 0,541; XTX3: 0,541). Prahy patří do manifestu dekodéru.

## Závěrečné srovnání po screeningu (2026-10-11, plný test 11 320 obr., `reports/*_full_*.json`, `reports/B0_xtx3u_whole_320_sweep.json`)

| model | GMAC @320 | S | AP | AP50 | AP75 | AR | F1 @f1_max (práh) | F1 @FP/obr. XTX3 | dup/obr. @fp_match |
|---|---|---|---|---|---|---|---|---|---|
| XTX3-u (B0, merge = NMS) | 0,39 (DW) | 320 | 0,472 | 0,710 | 0,497 | 0,589 | 0,725 (0,30) | 0,703 | 0,008 |
| F3 kontrola (XTX3-u topologie + recept) | 0,34 (DW) | 320 | 0,532 | 0,711 | 0,580 | 0,710 | – | – | – |
| F3 kontrola + OKS-NMS 0,5 | | 320 | 0,540 | 0,744 | 0,582 | 0,668 | – | – | – |
| **XTX4 u_b, 100 ep.** | 0,77 (plné 3×3) | 256 | 0,549 | 0,720 | 0,595 | 0,710 | | | |
| **XTX4 u_b, 100 ep.** | | 320 | **0,591** | 0,754 | **0,641** | **0,746** | 0,730 (0,25) | 0,718 | 0,284 |
| **XTX4 u_b, 100 ep.** | | 384 | 0,615 | 0,770 | 0,671 | 0,766 | | | |
| **XTX4 u_b + OKS-NMS 0,5** | | 256 | 0,554 | 0,746 | 0,595 | 0,674 | | | |
| **XTX4 u_b + OKS-NMS 0,5** | | 320 | **0,596** | **0,779** | 0,641 | 0,711 | **0,758 (0,15)** | **0,748** | 0,073 |
| **XTX4 u_b + OKS-NMS 0,5** | | 384 | **0,618** | **0,792** | **0,671** | 0,731 | | | |

Při shodném FP/obr. (u_b + NMS vs XTX3-u @320): recall obličejů < 8 px 0,702 vs 0,703, 8–16 px 0,856 vs 0,813, ≥ 32 px 0,889 vs 0,814; crowd 5+ 0,694 vs 0,677; okraj 0,618 vs 0,603; face-PCK@0,1 0,417 vs 0,278; torzo < 16 px 0,472 vs 0,551 (jediný horší bucket), osoby bez obličejových bodů 0,432 vs 0,481.

Nasazovací nastavení v `D:/xtx4_runs/f4_u_b_100ep/export/manifest.json`: práh 0,15, OKS-NMS 0,5, present 0,5 / visible 0,8. Parita ncnn FP16 vs PyTorch: xy p95 0,32–0,44 px, p99 0,7–0,8 px, max ~1,1 px (`parity.json`); rozhodující ΔAP se měří (`--model-kind ncnn`).

Poznámka: PC se během tréninku u_b restartoval (19:52, krok 641/707 poslední epochy); běh byl dokončen z `last.pt`. Doporučuji sledovat teploty při dlouhých bězích.

## Gate G2 – ncnn FP16 export u_b (2026-10-11, prvních 2000 obr. testu, @320, `reports/f4_u_b_{pt,ncnn16}_n2000*.json`)

| | PyTorch (CUDA) | ncnn FP16 (CPU, 4 vlákna) | Δ |
|---|---|---|---|
| bez NMS: AP / AP50 / AP75 / AR | 0,5932 / 0,7547 / 0,6478 / 0,7428 | 0,5932 / 0,7545 / 0,6474 / 0,7428 | ≤ 0,0004 |
| + OKS-NMS 0,5: AP / AP50 / AP75 / AR | 0,5985 / 0,7801 / 0,6474 / 0,7096 | 0,5985 / 0,7802 / 0,6471 / 0,7095 | ≤ 0,0004 |
| F1 @f1_max (NMS) | 0,764 (práh 0,16) | 0,764 (práh 0,15) | 0,000 |

Gate G2 (ΔAP ≤ 0,5 b.) splněn; parita xy p95 0,4 px nemá vliv na přesnost. `evaluate.py --model-kind ncnn --manifest ...` je k dispozici pro další exporty.

## Gate G1 – latence na Raspberry Pi 5 (2026-10-14, `reports/pi_latency.csv`, `reports/pi_latency_e2e.csv`, `reports/pi_sweep.log`)

Zařízení: Raspberry Pi 5 Model B rev 1.1 (8 GB RAM; latence sítě na velikosti RAM nezávisí, RSS u_b < 100 MB),
Debian 13, kernel 6.12, governor `ondemand` 2,4 GHz, **bez aktivního chlazení**, ncnn python 1.0.20260114
(`~/freesentry/.venv`), Vulkan vypnut. Metodika podle plánu §4: `scripts/bench_pi.py` (jen `extractor.input`
+ `extract` obou výstupů), 16 reálných snímků testu letterboxovaných na vstup, 30 warm-up + 200 měřených
iterací, před každou konfigurací čekání na SoC < 70 °C. Grafy: `bench/graphs` (náhodné váhy, pnnx/ncnn
20260526) + natrénovaný export `D:/xtx4_runs/f4_u_b_100ep/export` (`u_b_trained_320`). `fp16=off` =
FP32 aritmetika i úložiště (ncnn `use_fp16_*` = false), `on` = výchozí ncnn na ARMv8.2.

### Network only (120 rows, host freesentry)

| graph | size | GMAC | fp16 | thr 1 p50 / p95 | thr 2 p50 / p95 | thr 3 p50 / p95 | thr 4 p50 / p95 | temp after |
|---|---|---|---|---|---|---|---|---|
| u_a_256 | 256 | 0.25 | on | 8.1 / 8.1 | 5.7 / 5.8 | 5.0 / 5.1 | 5.7 / 12.6 | 70.3 °C |
| u_a_320 | 320 | 0.39 | on | 12.7 / 13.0 | 8.9 / 9.1 | 7.6 / 7.7 | 8.6 / 9.4 | 70.8 °C |
| u_a_384 | 384 | 0.57 | on | 17.7 / 17.7 | 12.7 / 12.9 | 10.9 / 11.0 | 12.5 / 22.4 | 74.7 °C |
| u_b_256 | 256 | 0.49 | on | 12.3 / 12.4 | 8.2 / 8.3 | 7.7 / 7.8 | 8.9 / 9.0 | 73.6 °C |
| u_b_320 | 320 | 0.76 | on | 19.0 / 19.2 | 12.5 / 12.6 | 11.7 / 11.8 | 12.7 / 23.8 | 76.8 °C |
| u_b_384 | 384 | 1.10 | on | 26.2 / 26.3 | 17.2 / 17.4 | 17.0 / 17.2 | 18.1 / 29.0 | 80.7 °C |
| u_b_P3P5_256 | 256 | 0.42 | on | 11.0 / 11.1 | 7.3 / 7.4 | 6.9 / 7.4 | 7.5 / 18.4 | 76.8 °C |
| u_b_P3P5_320 | 320 | 0.66 | on | 17.2 / 17.4 | 11.4 / 11.7 | 10.7 / 10.8 | 12.0 / 22.1 | 79.6 °C |
| u_b_P3P5_384 | 384 | 0.95 | on | 23.8 / 24.0 | 15.8 / 16.0 | 15.7 / 15.9 | 16.8 / 27.5 | 80.1 °C |
| u_b_P4P5_256 | 256 | 0.33 | on | 8.9 / 8.9 | 5.8 / 5.9 | 5.7 / 5.9 | 6.4 / 6.8 | 76.3 °C |
| u_b_P4P5_320 | 320 | 0.52 | on | 13.5 / 13.7 | 8.7 / 8.9 | 8.8 / 8.9 | 9.1 / 20.6 | 77.9 °C |
| u_b_P4P5_384 | 384 | 0.75 | on | 19.2 / 19.4 | 12.6 / 12.7 | 12.5 / 12.6 | 13.0 / 23.4 | 79.0 °C |
| u_b_hswish_256 | 256 | 0.49 | on | 12.5 / 12.6 | 8.3 / 8.4 | 8.0 / 9.3 | 9.1 / 10.1 | 76.8 °C |
| u_b_hswish_320 | 320 | 0.76 | on | 19.4 / 19.6 | 12.7 / 12.9 | 12.0 / 12.3 | 13.2 / 24.0 | 81.2 °C |
| u_b_hswish_384 | 384 | 1.10 | on | 26.9 / 27.1 | 17.6 / 17.8 | 17.6 / 17.7 | 18.8 / 29.4 | 80.7 °C |
| u_b_p2assist_256 | 256 | 0.50 | on | 13.5 / 13.6 | 9.5 / 9.7 | 9.0 / 9.2 | 10.4 / 17.8 | 79.6 °C |
| u_b_p2assist_320 | 320 | 0.78 | on | 20.8 / 21.0 | 14.1 / 14.2 | 13.8 / 14.0 | 14.8 / 26.1 | 80.1 °C |
| u_b_p2assist_384 | 384 | 1.13 | on | 29.4 / 29.6 | 20.5 / 20.6 | 19.7 / 19.8 | 21.6 / 32.5 | 80.7 °C |
| u_b_silu_256 | 256 | 0.49 | on | 14.9 / 14.9 | 9.6 / 9.7 | 9.1 / 9.2 | 9.7 / 20.8 | 76.8 °C |
| u_b_silu_320 | 320 | 0.76 | on | 22.8 / 22.9 | 14.6 / 14.8 | 13.6 / 13.9 | 14.6 / 25.2 | 80.1 °C |
| u_b_silu_384 | 384 | 1.10 | on | 31.9 / 32.1 | 20.5 / 20.6 | 20.1 / 20.2 | 21.2 / 32.0 | 82.9 °C |
| u_b_trained_320 | 320 |  | on | 18.9 / 19.1 | 12.5 / 12.6 | 11.6 / 11.7 | 12.6 / 23.6 | 81.8 °C |
| u_c_256 | 256 | 0.98 | on | 23.0 / 23.2 | 14.6 / 14.7 | 13.0 / 13.2 | 14.0 / 24.6 | 79.6 °C |
| u_c_320 | 320 | 1.54 | on | 35.2 / 35.3 | 22.5 / 22.8 | 19.9 / 20.2 | 22.6 / 25.1 | 82.9 °C |
| u_c_384 | 384 | 2.21 | on | 49.6 / 49.7 | 32.2 / 32.3 | 28.8 / 29.3 | 31.2 / 43.8 | 84.0 °C |
| xtx3u_ctrl_256 | 256 | 0.22 | on | 12.9 / 13.0 | 8.4 / 8.5 | 6.7 / 6.8 | 7.1 / 15.5 | 76.8 °C |
| xtx3u_ctrl_320 | 320 | 0.34 | on | 20.6 / 20.8 | 13.8 / 14.0 | 11.3 / 11.4 | 12.4 / 23.6 | 79.0 °C |
| xtx3u_ctrl_384 | 384 | 0.50 | on | 30.5 / 30.7 | 21.0 / 21.1 | 17.8 / 18.0 | 19.8 / 27.3 | 80.1 °C |
| u_a_320 | 320 | 0.39 | off | - | - | 17.3 / 17.5 | - | 81.2 °C |
| u_b_320 | 320 | 0.76 | off | - | - | 28.3 / 28.9 | - | 84.0 °C |
| u_b_trained_320 | 320 |  | off | 39.4 / 39.6 | 27.6 / 27.8 | 27.1 / 27.3 | 31.0 / 41.5 | 83.4 °C |
| u_c_320 | 320 | 1.54 | off | - | - | 47.8 / 55.7 | - | 84.5 °C |
| xtx3u_ctrl_320 | 320 | 0.34 | off | - | - | 23.5 / 23.9 | - | 81.8 °C |

### End-to-end (16 rows)

| size | thr | fp16 | NMS | total p50 | p95 | pre | infer | post | wall p95 | fps | poses/frame | temp after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 2 | on | off | 10.4 | 10.9 | 1.74 | 8.4 | 0.33 | 10.9 | 96.2 | 2.06 | 75.7 °C |
| 256 | 2 | on | on | 10.7 | 11.3 | 1.74 | 8.4 | 0.56 | 11.3 | 93.8 | 1.82 | 76.3 °C |
| 256 | 3 | on | off | 10.1 | 10.6 | 1.75 | 8.0 | 0.34 | 10.7 | 99.2 | 2.06 | 76.8 °C |
| 256 | 3 | on | on | 10.3 | 10.9 | 1.74 | 8.0 | 0.58 | 10.9 | 97.6 | 1.82 | 79.0 °C |
| 256 | 4 | on | off | 10.6 | 11.1 | 1.74 | 8.5 | 0.34 | 11.1 | 92.4 | 2.06 | 77.9 °C |
| 256 | 4 | on | on | 10.8 | 11.5 | 1.74 | 8.5 | 0.58 | 11.5 | 89.1 | 1.82 | 76.8 °C |
| 320 | 2 | on | off | 13.6 | 14.9 | 0.56 | 12.7 | 0.34 | 15.0 | 70.8 | 2.21 | 76.3 °C |
| 320 | 2 | on | on | 13.9 | 15.2 | 0.56 | 12.7 | 0.58 | 15.2 | 69.7 | 1.83 | 74.7 °C |
| 320 | 3 | on | off | 12.9 | 14.3 | 0.56 | 12.0 | 0.35 | 14.3 | 74.5 | 2.21 | 77.9 °C |
| 320 | 3 | on | on | 13.2 | 14.5 | 0.55 | 12.0 | 0.58 | 14.5 | 73.3 | 1.83 | 79.6 °C |
| 320 | 4 | on | off | 14.5 | 24.7 | 0.58 | 13.0 | 0.35 | 24.7 | 65.5 | 2.21 | 80.7 °C |
| 320 | 4 | on | on | 15.0 | 22.9 | 0.58 | 13.0 | 0.62 | 23.0 | 64.4 | 1.83 | 80.1 °C |
| 384 | 2 | on | off | 19.3 | 20.0 | 1.36 | 17.6 | 0.35 | 20.1 | 51.5 | 2.44 | 78.5 °C |
| 384 | 2 | on | on | 19.6 | 20.1 | 1.38 | 17.6 | 0.59 | 20.2 | 50.8 | 2.19 | 79.0 °C |
| 384 | 3 | on | off | 19.4 | 20.3 | 1.37 | 17.6 | 0.35 | 20.3 | 51.0 | 2.44 | 80.1 °C |
| 384 | 3 | on | on | 19.6 | 20.2 | 1.39 | 17.5 | 0.59 | 20.2 | 50.8 | 2.19 | 80.7 °C |


**Zjištění**

1. **Gate G1 splněn s rezervou 2×.** u_b @320: p50 **11,7 ms** (3 vlákna), 12,5 ms (2 vl.), 19,0 ms (1 vl.);
   natrénovaný export 11,6 ms, tj. grafy s náhodnými vahami jsou reprezentativní. Rozpočet plánu ≤ 25 ms.
2. **Premisa plánu §3.3 potvrzena měřením:** u_b (plné 3×3 RepConv) 0,765 GMAC / 11,7 ms = **65 MMAC/ms**,
   XTX3-u topologie (DW/PConv, SiLU) 0,344 GMAC / 11,3 ms = **30 MMAC/ms** (1 vlákno: 40 vs 17 MMAC/ms).
   Při stejné latenci má u_b 2,2× více výpočtu – a o +6 b. AP (viz závěrečné srovnání výše).
3. **Vlákna:** 3 vlákna jsou optimum; 4 vlákna mají p50 o 5–10 % horší a p95 dvojnásobné (sdílení jádra
   s OS a Wi‑Fi AP). Nasazení: `num_threads=3`.
4. **FP16 aritmetika dává 2,4×** (u_b 320: 11,7 vs 28,3 ms; u_c 19,9 vs 47,8 ms) – výchozí ncnn ponechat,
   přesnost FP16 ověřena gatem G2.
5. **Aktivace:** HardSwish +3 %, SiLU +16 % latence oproti ReLU (u_b 320, 3 vl.) → ReLU zůstává.
   **Úrovně hlavy:** P3P5 −9 %, P4P5 −25 %, P2-assist +18 % (u_b 320). P4P5 je zajímavé jen pokud ablace
   neukáže ztrátu na malých obličejích; P2-assist je drahý.
6. **Rozlišení (3 vl., p50):** u_a 5,0 / 7,6 / 10,9 ms, u_b 7,7 / 11,7 / 17,0 ms, u_c 13,0 / 19,9 / 28,8 ms
   @256 / 320 / 384. V rozpočtu ≤ 25 ms jsou tedy i **u_b @384** (17,0 ms, +2,2 b. AP podle screeningu)
   a **u_c @320** (19,9 ms, kapacita 2× – zisk zatím nezměřen).
7. **End-to-end** (`scripts/bench_pi_e2e.py`, `NcnnPoseRunner.detect`, 3 vl., FP16, manifest práh 0,15):
   u_b @320 **13,2 ms** = letterbox 0,6 + síť 12,0 + decode + OKS-NMS 0,6 ms ≈ **73 FPS**; @384 19,6 ms
   (51 FPS); @256 10,3 ms (98 FPS). OKS-NMS na hostu stojí 0,25 ms – zanedbatelné, duplicity tím řešit lze.
8. **Teplota a stabilita:** bez chladiče SoC při 3–4 vláknech 80–84 °C, firmware hlásil throttling
   (`get_throttled` 0xe0000 = capped/throttled/soft-limit „occurred“); čekání na 70 °C mezi konfiguracemi
   drží čísla srovnatelná. Během poslední e2e konfigurace (384 px, 4 vlákna) se **Pi restartovalo**
   (uptime vynulován, journal není perzistentní → příčina nezjištěna; `usb_max_current_enable=0`, tj.
   zdroj nebyl rozpoznán jako 5 A, EXT5V 5,03 V v klidu – podezření na podpětí při plné zátěži 4 jader +
   Wi‑Fi AP). Chybí proto 2 řádky e2e (384 × 4 vl.). Pro trvalý provoz: chladič/ventilátor, 5 A zdroj,
   3 vlákna.

**Rozhodnutí G1:** `u_b` zůstává výchozí varianta `u` (nasazení @320, 3 vlákna, FP16, OKS-NMS na hostu).
Rezerva rozpočtu se využije buď nasazením @384 (bez tréninku navíc), nebo screeningem `u_c` (100 epoch)
– obojí splňuje ≤ 25 ms. `u_a` není potřeba.

## Paralelní trénink u_b + u_c (2026-10-14/15, notebook i9-13900HX 32 GB RAM, RTX 4070 Laptop 8 GB)

| konfigurace | img/s na běh | celkem img/s | špička commit (limit ~54 GB) | výsledek |
|---|---|---|---|---|
| 1 běh u_c, 6 workerů, `OPENBLAS_NUM_THREADS=1` | 320–350 | 320–350 | 42 GB (+21 GB běh) | stabilní |
| 2 běhy u_b + u_c, 6 workerů každý (14. 10. 23:02) | – | – | vyčerpán | Windows „nedostatek virtuální paměti“ 23:03, chyba 1455 ve workerech, tvrdý restart PC 23:08 |
| 2 běhy u_b + u_c, 3 workery každý (s hlídačem paměti) | 125–165 | ~290 | 56 GB (pagefile se zvětšoval) | doběhl, ale celkem stejně rychlé jako 1 běh |

Paměť jednoho tréninkového procesu: hlavní proces ~10–12 GB (z toho 4–6 GB kernely CUDA/cuDNN v RAM – změřeno
čistým fwd/bwd bez dat; `cudnn.benchmark=False` by ušetřilo 2,4 GB za cenu −13 % rychlosti GPU), každý datový worker
~1,3 GB. Ostatní programy na PC drží ~21–24 GB. Opatření: `train.py` nastavuje `OPENBLAS_NUM_THREADS=1`
(OpenBLAS si jinak v každém procesu předalokuje 0,72 GB; běh tím klesl z +26 na +21 GB). Zkoušený přenos dávek
jako uint8 byl vrácen: halda Windows nafoukla workery na ~4 GB.

**Závěr:** GPU není při jednom běhu úzkým hrdlem (čistý fwd/bwd u_c zvládne 725 img/s, reálně 340 img/s), ale
dva běhy si dělí stejné CPU/datovou cestu, takže dohromady nejsou rychlejší a hrozí pád. Varianty trénovat za sebou.
