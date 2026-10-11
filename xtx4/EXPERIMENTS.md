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
