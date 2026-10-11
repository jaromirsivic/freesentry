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
