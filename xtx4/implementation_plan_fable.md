# XTX4 – implementační plán pro AI agenta (varianta Fable)

Datum rešerše a auditu: 11. 10. 2026. Autor: Claude Fable 5.1. Vstupy: kompletní zdrojový kód, konfigurace, logy a exporty ve složce `xtx3/` a soubor `xtx4/implementation_plan.md` (plán od modelu Astra). Nic jiného z disku nebylo čteno. Webová rešerše proběhla 11. 10. 2026; u každého čísla z webu je uveden zdroj.

Cíl: co nejefektivnější síť pro detekci **9 landmarků hlavy a torza** (nos, oči, uši, ramena, boky) u více osob v jednom snímku. Bounding box se nevyžaduje. Varianta `u` má běžet co nejblíže real-time na Raspberry Pi 5 (4 GB RAM, ncnn, CPU). Trénink na RTX 4070 Laptop 8 GB s AMP FP16. Distribuce ONNX + ncnn.

---

## 0. Shrnutí rozhodnutí (TL;DR)

1. **Navrhnout XTX4-u znovu, ne jen ořezat XTX3-u.** Důvod je změřitelný: na Pi 5 s ncnn jsou sítě z plných 3×3 konvolucí (Winograd) 2–3× efektivnější na jeden MAC než sítě z depthwise/partial konvolucí (viz §3.3, tabulka ncnn z Pi 5; CPUBone, CVPR 2026). XTX3-u je postavená právě z DWConv/PConv s malým počtem kanálů, tedy z toho, co je na tomto CPU nejméně efektivní. Při stejné latenci lze XTX4-u dát 2–3× více výpočtu, a tedy i přesnosti.
2. **Box-free trénink i inference („pose as object“).** Přiřazování pozitivních anchorů řídit metrikou `score^α · OKS^β` místo IoU boxu a box větev úplně zrušit. ER-Pose (03/2026) to na YOLOv8-pose-n ablovala: 50,4 → 54,7 (keypoint assignment) → 57,1 AP (bez box větve). Pro úlohu hlava+torzo je to ještě přirozenější, protože celotělový COCO box má střed v oblasti stehen, daleko od obličeje.
3. **Zachovat duální hlavu (one-to-many + one-to-one) a NMS-free inferenci**, ale s přiřazováním podle YOLO26 (o2o: topk 7 → 1, ProgLoss 0,8/0,2 → 0,1/0,9) a s diagnostikou duplicit. XTX3-u hlásí precision 13 % při prahu 0,05 (3091 FP na 469 TP) – nutno změřit duplicity při provozním prahu 0,25, ne hádat.
4. **Nejprve opravit měření.** AP75 je v XTX3 vždy 0,0 kvůli klíči `0.7500000000000002` z `np.arange` (ověřeno spuštěním numpy, viz §2.4). Validace běží jen na prvních 200 obrázcích testu a podle nich se vybírá `best.pt`. Bez opravy nelze nic porovnávat.
5. **Zrychlit trénink 5–10×.** XTX3-u trénuje 28 obr./s (27 min/epochu, 50 epoch ≈ 22 h) při modelu 0,39 GMAC – GPU je prakticky nevyužitá; bottleneck je Python smyčka assigneru přes obrázky a per-parametrový Newton–Schulz v MuSGD. Dávkový assigner + vektorizované ztráty + cache dat → cíl ≥ 200 obr./s, což umožní 300-epochové rozvrhy (YOLO26n: 150 + 245 epoch; ER-Pose: 600 epoch).
6. **Zrušit pyramidový režim pro n/m/l.** XTX3-n v pyramidě dosáhla po 27 epochách AP 0,25–0,27, zatímco XTX3-u celoobrazově ve stejné epoše 0,38 při 3,2× méně parametrech a 1/12 výpočtu na snímek. Všechny varianty trénovat i nasazovat celoobrazově; pyramida zůstane jen jako volitelný inferenční režim.
7. **Data:** COCO + OCHuman (eval okluzí) + Human-Art + UBody (horní polovina těla) + částečné anotace (CrowdPose/AIC: jen ramena a boky, obličejové body jako „neznámé“) + pseudo-labely z cílové kamery. Opravit pravidlo zařazení osob do výřezu (dnes podle středu celotělového boxu, což zahazuje osoby s hlavou v záběru).
8. **Distilace bez inferenčních nákladů:** in-house teacher XTX4-l a Apache-licencovaný RTMPose-l pro doplnění chybějících obličejových bodů; výstupy teachera cachovat offline (8 GB VRAM).
9. **Export:** fuse → ONNX opset 17 (statický tvar 256/320/384) → pnnx → ncnn FP16 (ncnn na Cortex-A76 počítá ve FP16 implicitně), 4 vlákna, **bez Vulkanu** (na Pi 5 je 10–20× pomalejší než CPU), host decode v NumPy (převzít z `xtx3/live_test`). INT8 jen jako volitelná fáze (v ncnn přináší u hustých sítí ~1,1–1,2×).
10. **Cílová čísla (odhad, ne měření):** XTX4-u @320 ≈ 0,75 GMAC, síťový čas na Pi 5 ≤ 25 ms p50 (end-to-end ≥ 25 FPS s pipeliningem), AP9 ≥ XTX3-u + 8 b. na opravené plné validaci. Zdůvodnění rozsahu je v §11.

---

## 1. Zadání, hranice a hardware

- **Výstup sítě:** pro každou osobu skóre + 9 × (x, y, confidence, viditelnost). Žádný povinný bbox. Pořadí bodů zůstává jako v XTX3 (`xtx3/xtx3/utils/keypoints.py`): nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip; COCO indexy 0,1,2,3,4,5,6,11,12; flip páry (1,2),(3,4),(5,6),(7,8); OKS sigmy 0,026/0,025/0,025/0,035/0,035/0,079/0,079/0,107/0,107.
- **Cílové zařízení varianty u:** Raspberry Pi 5, 4× Cortex-A76 @ 2,4 GHz, 4 GB RAM, 64bit Raspberry Pi OS (Bookworm), ncnn CPU. Cortex-A76 je ARMv8.2-A s FP16 aritmetikou a int8 dot-product.
- **Trénink:** RTX 4070 Laptop 8 GB VRAM, Windows, PyTorch s `torch.amp.autocast(dtype=float16)` + `GradScaler`. Assignment, OKS, flow a redukce ztrát počítat ve FP32 (uvnitř `autocast(enabled=False)`).
- **Distribuce:** ONNX (FP32 reference) + ncnn `.param/.bin` (FP16 váhy přes pnnx).
- **Hranice práce agenta:** nový kód vzniká ve složce `xtx4/`; `xtx3/` se nemění (slouží jako baseline a zdroj kódu k převzetí). Spuštění tréninku, čtení datasetu a přístup k Pi vyžadují explicitní souhlas uživatele – plán je popisuje jako budoucí kroky.

---

## 2. Audit XTX3 – co kód a logy skutečně dokládají

### 2.1 Architektura XTX3-u (ze `xtx3/xtx3/models/*.py`, `scaling.py`)

| Část | Konfigurace | Odhad MMAC @320 (vlastní rozpis) |
|---|---|---|
| Stem | Conv3×3/2, 3→24 @160² | 16,6 |
| Stage 1 (P2, stride 4) | PConv/2 24→48 (3×3 jen na 8 kanálech + maxpool zbytku + 1×1), CSP×1 s DWConv, 48 k @80² | 46 |
| Stage 2 (P3, stride 8) | PConv/2 48→96, CSP×1 DW, 96 k @40² | 45 |
| Stage 3 (P4, stride 16) | DWConv/2 96→144, CSP×1 DW, 144 k @20² | 25 |
| Stage 4 (P5, stride 32) | DWConv/2 144→144, CSP×1 DW, SPPF 5×5, @10² | 14 |
| Neck PAN | td_p4 CSP(288→144), td_p3 CSP(240→96), bu_p4 CSP(240→144), bu_p5 CSP(288→144); vše DW | 146 (z toho td_p3 ≈ 67) |
| Hlava o2o | 3 úrovně × 3 větve (box/cls/kpt), každá 2×DWConv + 1×1; box/cls hidden 48, kpt hidden 64 → 54 kanálů | 72 (box ≈ 19, cls ≈ 19, kpt ≈ 34) |
| **Celkem** | 0,82 M parametrů | ≈ 365 vlastní odhad; `profile_flops.py` hlásí 0,389 GMAC @320 (0,249 @256, 0,561 @384) |

Poznámky:
- `profile_flops.py` a `count_gflops` počítají **MAC** (násobení s akumulací) konvolucí a lineárních vrstev, ne FLOPs. V dokumentaci XTX3 jsou tato čísla označena jako GFLOPs; v XTX4 důsledně používat GMAC.
- Všechny 1×1 konvoluce CSP bloků na P2/P3 tvoří většinu MAC; depthwise 3×3 jsou v MAC zanedbatelné, ale na CPU jsou paměťově vázané (viz §3.3).
- Aktivace SiLU všude (`ConvBNAct`). Na ARM CPU je SiLU dražší než ReLU/HardSwish (ActNAS: YOLOv5n na Cortex-A53: SiLU 938 ms, ReLU 793 ms, HardSwish 824 ms).
- Hlava: 2100 anchorů @320 (1600 + 400 + 100), 59 hodnot na anchor (4 box + 1 cls + 54 kpt). Z 54 kpt kanálů se při inferenci nepoužívá `log_sigma` (9 kanálů) – RLE nejistota se zahazuje místo využití jako confidence.

### 2.2 Trénink a ztráty (z `trainer.py`, `pose_loss.py`, `assignment.py`, `rle.py`, `optim.py`, `config.json`)

- 50 epoch, batch 32, MuSGD lr 0,01 cosine → 1e-4, warmup 3 ep., EMA 0,9999 (warmup 2000 kroků), AMP, multi-scale 256/320/384 pro `u` (přepíná se každých 10 kroků), mosaic 0,5 vypnutý posledních 10 epoch, flip 0,5, scale jitter 0,5, translate 0,1.
- Ztráty: CIoU 7,5 + Varifocal 0,5 + RLE 12 + vis-CE 1,0; TAL o2m topk=10 (α=1, β=6) + STAL (box < 8 px → surrogate 16 px, ≥ 4 anchorů) + centrální prior 0,1; o2o = tentýž assigner s topk=1; ProgLoss w_o2m 1,0→0,25, w_o2o 0,5→1,0.
- **Multi-scale je implementován jako `F.interpolate(bilinear)` dávky vyrenderované na 384** → trénink vidí rozmazané/aliasované 256 a 320, inference používá `INTER_AREA` přímo ze zdroje. Nesoulad trénink/nasazení, nejcitlivější u malých obličejů.
- **Zařazení osoby do výřezu** (`dataset.py: targets_to_network`): osoba je target jen pokud střed **celotělového** boxu leží ve výřezu. Osoba s hlavou a rameny v záběru, ale boxem převážně mimo, se zahodí a její viditelné body se stanou „pozadím“. Pro úlohu hlava+torzo je to systematická chyba labelů.
- **Visibility třída 0** se učí jako „absent“, ale v COCO `v=0` znamená „neanotováno“ (často viditelné body malých osob). Model se tedy učí predikovat „absent“ u skutečně přítomných bodů; vis_acc boků 0,71 tomu odpovídá.
- **MuSGD v XTX3** je čistý Muon pro všechny ≥2D tenzory (bez míchání se SGD) a dělá Newton–Schulz zvlášť pro každý parametr (~150 malých matic × 5 iterací × 3 matmul na krok). Oficiální MuSGD (Ultralytics `optim/muon.py`) míchá Muon a SGD aktualizace (pro n: muon_w 0,53, sgd_w 0,67) a od v8.4.102 dávkuje Newton–Schulz (až 8× rychlejší krok).
- **RLE konvence:** XTX3 = `log σ + log 2 + |ε| − log φ(ε)` (jeden člen log σ). YOLO26 i původní RLE kód = `log σ − log φ(ε) + log 2σ + |ε|` (dva členy log σ). XTX3 varianta je korektně normalizovaná hustota; referenční varianta je empiricky ověřená a σ penalizuje silněji. Oba lze otestovat, ale XTX3 sigmu navíc omezuje na ≥ 1 px (`softplus·stride + 1`).
- **Assigner běží v Python smyčce přes obrázky** (32 obrázků × 2 hlavy = 64 volání na krok, každé s topk, pairwise IoU, STAL smyčkou přes malé GT). Spolu s per-parametrovým Newton–Schulz to vysvětluje 28 obr./s (≈ 1,1 s na krok) při síti, jejíž forward+backward na 4070 trvá řádově desítky ms. **Hypotéza k ověření profilerem**, ale důsledek je jistý: dnešní pipeline neumožní 300-epochové rozvrhy.

### 2.3 Výsledky (z `runs/xtx3-u/train.log`, `eval_report.json`, `runs/xtx3-n*/train.log`)

| Běh | Režim | Epochy | Nejlepší AP9 | AP50 | AP75 | AR | Poznámka |
|---|---|---|---|---|---|---|---|
| xtx3-u | whole, 320, multi-scale | 50/50 | **0,4589** (ep. 50) | 0,7064 | 0,0000 (bug) | 0,5730 | ještě rostlo: ep. 45 0,4565, ep. 49 0,4563 |
| xtx3-u @ ep. 27 | – | – | 0,3779 | 0,6125 | – | 0,5081 | pro srovnání s n |
| xtx3-n | pyramid 384 (3 průchody) | 27/50 (nedokončeno) | 0,2706 (ep. 23) | 0,5183 | 0,0000 | 0,4638 | 2,65 M param., 4,53 GMAC/snímek |

Všechny „Copy“ složky jsou kopie téže historie (shodné řádky logu), ne nezávislé seedy.

Detailní report xtx3-u (prvních 200 obrázků testu, práh 0,05, max 100 detekcí):
- Detekce @OKS≥0,5: precision 0,1317, recall 0,7963 (TP 469, FP 3091, FN 120). Při prahu 0,05 je 87 % výstupů FP – buď duplicity o2o hlavy, nebo nízkoskórové kandidáty, které AP tolik nebolí. Rozlišit lze jen měřením duplicit při 0,25 (viz T1).
- Mean OKS matchnutých bodů: nos 0,760, oči 0,78/0,79, uši 0,84, ramena 0,88, boky 0,87. Obličej je nejslabší; oči a nos mají nejpřísnější sigmy (0,025/0,026).
- Vis accuracy: uši 0,71/0,75, boky 0,71/0,75 – nejhorší; ramena 0,88.
- Face scale (v originálních px): small (<16 px) recall 0,82, medium 0,87, large 0,90. Torso small (<32 px) recall 0,72.
- Dataset podle logu: train 45 279 obr. / 119 735 osob, test 11 320 obr. / 30 078 osob. Test je zároveň selekční sada pro `best.pt`.

### 2.4 Potvrzené defekty měření a tréninku

| # | Defekt | Důkaz | Dopad |
|---|---|---|---|
| D1 | **AP75 vždy 0** | `metrics.py`: `_OKS_THRESHOLDS = np.arange(0.5, 1.0, 0.05)`; `per_threshold_ap.get(0.75)`; spuštění numpy dnes: prahy jsou `0.5, 0.55, 0.6000000000000001, …, 0.7500000000000002, …`; `0.75 in dict → False` | AP75 nelze použít; `AP` (průměr přes 10 prahů) je správně |
| D2 | Validace na 200 obrázcích (`validation.max_images: 200`) a na ní se vybírá `best.pt` | `trainer.py: _ground_truths(samples[:max_val_images])` | šum ±2–3 AP; selekce na „testu“ |
| D3 | Multi-scale přes bilineární `F.interpolate` z 384 | `trainer.py: _train_epoch` | nesoulad s nasazením (INTER_AREA), hlavně malé obličeje |
| D4 | Zařazení osob podle středu celotělového boxu | `dataset.py: targets_to_network` | chybějící targety u ořezaných osob |
| D5 | `v=0` trénováno jako „absent“ | `pose_loss.py` vis CE přes `vis_gt ∈ {0,1,2}` | label noise na viditelnosti |
| D6 | Propustnost tréninku 28 obr./s | log `img/s`; smyčky v `_head_loss`/`assign`, per-param NS | 22 h / 50 epoch |
| D7 | Pyramidový režim n/m/l | log xtx3-n | 3× dražší a horší než whole |
| D8 | Prahy evaluace: eval na prahu 0,05 místo ≈0,001 | `trainer.py: _predict_images(conf_threshold=0.05)` | mírné podhodnocení AP |
| D9 | O2O assigner topk=1 jednofázově, centrální prior obrazu 0,1 | `assignment.py` | odlišné od YOLO26 (7→1), prior nemá oporu pro vícero osob u okrajů |

Co funguje a převezme se: geometrie letterboxu a affine `T_k` (`data/preprocess.py`, bit-přesná parita s live testem), metadata 9 bodů, EMA/checkpointing, export pipeline (fuse → ONNX → onnxsim → verifikace → pnnx), NumPy dekodér a ncnn runner z `live_test/livetest/{decode,runner,preprocess}.py` (má filtr skóre před dekódováním, cache anchorů, validaci blobů).

---

## 3. Rešerše (stav k 11. 10. 2026) a co z ní plyne

### 3.1 Detektory a pose modely

| Zdroj | Co dokládá | Význam pro XTX4 |
|---|---|---|
| **YOLO26** (Ultralytics, vydáno 01/2026; článek arXiv 2606.03748; docs `models/yolo26`, `tasks/pose`) | NMS-free duální hlava: o2m TAL topk=10, o2o TAL topk=7 → topk2=1; ProgLoss α(t) lineárně (0,8; 0,2) → (0,1; 0,9) po epochách; STAL: GT s w nebo h < 8 px se **jen pro výběr kandidátů** zvětší na 16 px; DFL odstraněn; RLE: σ ∈ (0,1)² přes sigmoid, loss `log σ − log φ(ε) + log 2σ + |ε|`; **OKS + RLE kombinace** je lepší než každá zvlášť (26s-pose: OKS 61,5 / RLE 61,9 / obě 63,0 AP); MuSGD míchá Muon a SGD; yolo26n-pose: 57,2 AP e2e @640, 2,9 M param., 7,6 GFLOPs (≈3,8 GMAC), rozdíl o2m+NMS vs o2o jen 0,8 AP. Recept n: Objects365 150 ep. + COCO 245 ep., batch 128, lr0 0,0054, mosaic 0,91, close_mosaic 10, mixup 0,01, copy_paste 0,08, scale 0,5. | Převzít assigner o2o 7→1, ProgLoss schedule, STAL 16 px, kombinaci OKS+RLE, dlouhý rozvrh. Licence AGPL-3.0 – kód nepřebírat, jen postupy a jako referenční benchmark. |
| **YOLO27** (docs `models/yolo27`, stav „coming soon“, váhy nedostupné) | N/S predikují jen ze **dvou úrovní** (jemná + hrubá, prostřední vynechána), rozšířená raná vysokorozlišovací fáze; trénink navíc s pomocnou „foreground-alignment“ větví → rozdíl o2m/o2o 0,4 AP; YOLO27n předběžně 42,3 mAP, 3,0 M, 7,2 GFLOPs. | Ablace dvouúrovňové hlavy {P3,P5} pro u; při vydání YOLO27 zopakovat referenční benchmark. |
| **ER-Pose** (arXiv 2603.08681, 03/2026, kód zatím nezveřejněn) | Box-free YOLOv8-pose: přiřazování `conf^0,5 · OKS^6`, Smooth-OKS ztráta, viditelnost i confidence BCE, váhy pose:vis:conf = 15:1:1, duální hlava (o2m + o2o) NMS-free; **n varianta používá jen {P4,P5}**. COCO test-dev: ER-Pose-n 56,0 AP (YOLOv8-pose-n 49,1), 2,53 M param.; ablace (val): box+IoU assignment 50,4 → box+kpt assignment 54,7 → bez boxu 57,1; SAH-only 53,1 / MAH+OKS-NMS 56,6 / MAH+SAH 57,1; OCHuman 37,1 vs 30,7; AdamW wd 0,05, lr 0,01→1e-4, 600 epoch. | Hlavní opora pro box-free Pose-TAL a pro zachování duální hlavy. |
| **RTMO** (CVPR 2024; mmpose `projects/rtmo`) | Jednostupňový YOLOX-styl s dynamickou koordinátovou klasifikací; RTMO-t 416 (body7) 57,4 AP; RTMO-s 640 67,7 AP, 8,9 ms V100 ORT; ONNX ano, ncnn neuvedeno. | Referenční bod přesnosti; coordinate classifier je pro ncnn/Pi rizikový, nepřebírat. |
| **RTMPose / DWPose** (arXiv 2303.07399; 2307.15880) | SimCC hlavy (sub-pixel bez heatmap), DWPose dvoustupňová distilace (RTMPose-l 64,8 → 66,5 whole-body AP), přínos UBody dat. | RTMPose-l (Apache-2.0) jako teacher pro obličejové body; postup distilace. |
| **CPUBone** (CVPR 2026, arXiv 2603.26425, Apache-2.0, `github.com/altair199797/CPUBone`, timm váhy) | Měřeno **přímo na Pi 5** (ORT, batch 1): CPUBone-B0 519 MMAC → 24,2 ms (77,6 % top-1); T0 269 MMAC → 12,5 ms; S0 359 MMAC → 15,5 ms; RepViT-M0.9 816 MMAC → 40,8 ms; FasterNet-T1 (PConv) 850 MMAC → 32,8 ms; GhostNetV2 183 MMAC → 35,2 ms; EfficientFormerV2-S0 400 MMAC → 36,3 ms. **Depthwise konvoluce má na ARM CPU ~40× nižší propustnost MAC/s než plná nebo g=2 konvoluce** (1,1–1,3 vs ~40 MMAC/ms při 14×14); fused MBConv (plná 3×3 expanze) má o 70 % vyšší MAC/s pod 256 kanály; groups=2 je téměř zdarma. | Potvrzuje: pro Pi 5 CPU stavět z plných (nebo g=2) 3×3 konvolucí, ne z depthwise. |
| **MobileNetV4, RepViT, StarNet, LSNet, MobileOne** (2023–2025) | Reparametrizace (RepVGG-styl) je zdarma při nasazení; large-kernel DW a attention jsou cíleny na NPU/GPU; MobileOne: latence v efektivním režimu nekoreluje s FLOPs. | Použít reparam 3×3; nepoužívat attention ani large-kernel DW v u. |

### 3.2 Datové sady použitelné pro 9 bodů

| Sada | Rozsah | Body | Licence (ověřit) | Použití |
|---|---|---|---|---|
| COCO 2017 keypoints | ~150 k osob s ≥1 bodem | 17 → 9 | CC BY 4.0 | hlavní trénink; val2017 jako nedotčený holdout |
| OCHuman | 4 731 obr. / 8 110 silně okludovaných osob | 17 | výzkum | **evaluace** robustnosti vůči okluzi (ER-Pose protokol) |
| Human-Art | 50 k obr. / 123 k osob, 21 bodů (17 = COCO) | 17 → 9 | výzkum/nekomerční | doplněk tréninku (ověřit licenci pro produkt) |
| UBody | >1 M snímků, 15 scénářů, COCO-WholeBody 133 bodů, zaměřeno na horní polovinu těla | 133 → 9 | výzkum | podvzorek 20–40 k snímků; přesně náš use-case |
| CrowdPose, AI Challenger, MPII | 20 k / 300 k / 25 k obr. | 14/14/16, **bez nosu/očí/uší** | výzkum | částečné labely: ramena+boky, obličejové body jako „neznámé“ (ignorovat v loss) |
| Vlastní záběry z cílové kamery | dle dostupnosti | – | vlastní | pseudo-labely teacherem; největší očekávaný přínos pro nasazení |

### 3.3 Raspberry Pi 5 + ncnn: co je změřeno

Výchozí `ncnn::Option` (ověřeno v `src/option.cpp`): `use_fp16_packed = use_fp16_storage = use_fp16_arithmetic = true`, `use_int8_inference = true`, `use_winograd_convolution = true` (23/43/63), `use_sgemm_convolution = true`, `use_packing_layout = true`, `use_vulkan_compute = false`, `num_threads = fyzické big jádra`. `benchncnn` tyto flagy nastavuje natvrdo. **Na Cortex-A76 tedy ncnn počítá konvoluce ve FP16 implicitně**; níže uvedená čísla jsou FP16 čísla.

benchncnn na Pi 5, 4 vlákna, Bookworm 64bit (Q-engineering, `qengineering.eu/ncnn_rpi5.html`; hodnoty ms, sloupec CPU):

| Model (vstup) | ≈ MMAC | ms CPU | ≈ MMAC/ms | Typ |
|---|---|---|---|---|
| shufflenet_v2 (224) | 146 | 3,39 | 43 | DW + shuffle |
| mobilenet_v3 (224) | ~220 | 7,35 | ~30 | DW + SE + h-swish |
| mobilenet_v2 (224) | 300 | 10,52 | 29 | DW |
| efficientnet_b0 (224) | 390 | 12,33 | 32 | DW + SE |
| nanodet_m (320) | ~360–720 | 11,18 | 32–64 | ShuffleNetV2 + PAN |
| resnet18 (224) | 1 820 | 19,96 | **91** | plné 3×3 (Winograd) |
| yolov4-tiny (416) | ~3 400 | 46,29 | **73** | plné 3×3 |
| resnet50 (224) | 4 100 | 46,21 | 89 | plné + 1×1 |
| mobilenet v1 (224) | 569 | 33,88 | 17 | DW + široké 1×1 |
| YOLO26n det (640, Ultralytics guide, FP32) | 2 750 | 67,03 | 41 | CSP, SiLU |
| resnet18_int8 / resnet50_int8 | – | 16,74 / 40,21 | – | INT8 zisk 1,15–1,2× |
| **Vulkan (VideoCore VII)** | – | nanodet_m 205 ms, resnet18 349 ms | – | **10–20× pomalejší než CPU, nepoužívat** |

Závěry pro návrh:
1. Plné 3×3 konvoluce s Winogradem dosahují na Pi 5 v ncnn 70–90 MMAC/ms; depthwise sítě 17–32 MMAC/ms. **Stejná latence = 2–3× více MAC pro plné konvoluce.**
2. YOLO26n @640 běží v ncnn 67 ms ⇒ síť velikosti YOLO26n při 320 (≈0,69 GMAC det, ≈0,94 GMAC pose) by měla stát zhruba 17–25 ms. To je přesně rozpočet pro XTX4-u.
3. FP16 je výchozí; INT8 dává v ncnn u hustých sítí jen ~1,1–1,2× (FP16 už je rychlý). INT8 odložit.
4. Vulkan na Pi 5 vyloučit; `live_test` má volbu `use_vulkan` – nechat `false`.
5. Všechny hodnoty jsou z cizích měření; agent je musí reprodukovat na skutečném Pi (T2). Do té doby jsou to plánovací odhady.

---

## 4. Rozpočet latence a protokol měření na Pi 5

**Rozpočet pro ≥ 25 FPS:** snímek = 40 ms. S pipeliningem (vlákno kamery + vlákno sítě + vlákno kreslení) je propustnost dána nejdelším stupněm, ne součtem. Cíl: síť (ncnn `extract`) ≤ 25 ms p50 a ≤ 33 ms p95 při 320, 4 vlákna; předzpracování (letterbox INTER_AREA z 720p) ≈ 1–2 ms; dekódování NumPy < 1 ms při filtru skóre před dekódem; capture+MJPG dekód 5–8 ms v jiném vláknu.

**Model efektivity (plánovací):** husté 3×3 sítě 50–70 MMAC/ms (konzervativně pod resnet18, protože malé kanály a PAN mají horší využití), DW sítě 20–30 MMAC/ms. Odtud:
- XTX3-u @320: 0,389 GMAC DW ⇒ odhad 13–20 ms.
- XTX4-u-A @320: ≈0,40 GMAC plné ⇒ 6–9 ms.
- XTX4-u-B @320: ≈0,77 GMAC plné ⇒ 12–17 ms. **Výchozí kandidát.**
- XTX4-u-C @320: ≈1,5 GMAC plné ⇒ 22–32 ms (horní mez Pareta).

**Protokol (T2):** `benchncnn`-ekvivalent se skutečnými snímky (0/1/4/8 osob), 100 warm-up + ≥ 1 000 snímků, p50/p90/p95/p99, 1/2/3/4 vlákna, `fp16_arithmetic` on/off, teplota a `vcgencmd get_throttled` před/po, governor, verze ncnn/pnnx, RSS. End-to-end zvlášť: frame-in-memory → Pose9 a camera → display. Žádný přepočet z FLOPs se nevydává za FPS.

**Latency-first výběr architektury:** kandidáty A/B/C a ablace (aktivace, úrovně hlavy, P2-assist) exportovat **s náhodnými vahami** do ncnn a změřit na Pi **před** tréninkem. Trénuje se jen to, co se vejde do rozpočtu. Stojí to hodiny, ne dny.

---

## 5. Návrh XTX4

### 5.1 Principy

1. **Plné 3×3 konvoluce všude ve stem/backbone/neck/hlavě u varianty u.** Trénovat jako RepConv (3×3 + 1×1 + identita), nasadit jako jednu 3×3 (zdarma). Depthwise jen pokud měření na Pi ukáže opak.
2. **Kanály násobky 8 (ideálně 16)** kvůli pack8/pack4 layoutu ncnn; stride-1 3×3 kvůli Winogradu.
3. **Aktivace ReLU** jako výchozí (ncnn ji fúzuje do konvoluce; bez `exp`), HardSwish jako ablace; SiLU jen pokud by ztráta přesnosti překročila 0,5 AP a latence to dovolila.
4. **Žádná box větev.** Osoba = skóre + 9 bodů. Trénink i inference box-free (§6).
5. **Duální hlava (o2m + o2o), NMS-free inference**, oddělené věže jako v XTX3/YOLO26. Levná pojistka: host-side OKS-NMS přepínatelná v runtime (pro případ, že by duplicity přetrvaly).
6. **Tři úrovně {P3,P4,P5} jako výchozí, dvě úrovně {P3,P5} jako hlavní ablace** (YOLO27 N/S, ER-Pose-n). P2 hlavu neplánovat (4× více anchorů); místo ní levný „P2-assist“ pro keypoint větev (§5.4).
7. Žádná attention, žádné large-kernel DW, žádné dynamické operátory (TopK/NMS/GridSample) v grafu. Graf = Conv/ReLU/Concat/Upsample(nearest)/MaxPool/Reshape.
8. Plně konvoluční; jedna sada vah pro 256/320/384 (multi-scale trénink s přímým renderem).

### 5.2 XTX4-u – specifikace (kandidát B, výchozí)

Vstup 3×S×S, S ∈ {256, 320, 384}; BGR/255 jako XTX3 (zachovat paritu s `live_test`). Blok `CSPRep(C, n)`: 1×1 C→2h (h=C/2), n× bottleneck [RepConv3×3 h→h, RepConv3×3 h→h] s reziduem, concat (2+n)·h → 1×1 → C. Downsample = RepConv3×3 stride 2.

| Část | Výstup @320 | Vrstvy | Odhad MMAC |
|---|---|---|---|
| Stem | 16 × 160² | Conv3×3/2 3→16 | 11 |
| Stage 1 → P2 | 32 × 80² | Rep3×3/2 16→32, CSPRep(32, 1) | 75 |
| Stage 2 → P3 | 64 × 40² | Rep3×3/2 32→64, CSPRep(64, 2) | 108 |
| Stage 3 → P4 | 128 × 20² | Rep3×3/2 64→128, CSPRep(128, 2) | 108 |
| Stage 4 → P5 | 192 × 10² | Rep3×3/2 128→192, CSPRep(192, 1), SPPF(5) | 57 |
| Neck td_p4 | 128 × 20² | up(P5) ⊕ P4 → CSPRep(128, 1), vstup 320 k | 56 |
| Neck td_p3 → N3 | 64 × 40² | up(td_p4) ⊕ P3 → CSPRep(64, 1), vstup 192 k | 59 |
| Neck bu_p4 → N4 | 128 × 20² | Rep3×3/2 64→64 ⊕ td_p4 → CSPRep(128, 1) | 64 |
| Neck bu_p5 → N5 | 192 × 10² | Rep3×3/2 128→128 ⊕ P5 → CSPRep(192, 1) | 43 |
| Hlava (o2o, 3 úrovně) | – | per úroveň: sdílená Rep3×3 C→64; person: 1×1 64→1; pose: Rep3×3 64→64, 1×1 64→45 | 183 (P3 123, P4 45, P5 15) |
| **Celkem** | 2 100 anchorů @320 | ≈ 1,6–2,0 M parametrů (nasazení) | **≈ 765 MMAC @320**, ≈ 490 @256, ≈ 1 100 @384 |

Kandidát **A** (úzký): šířky 16/24/48/96/128, hloubky 1/1/1/1, hlava hidden 48 → ≈ 395 MMAC @320 (stejné MAC jako XTX3-u, ale plné konvoluce). Kandidát **C** (široký): 24/48/96/192/256, hloubky 1/2/2/1, hlava 64/96 → ≈ 1,5 GMAC. Všechny tři exportovat a změřit (T2); vybrat největší, který splní ≤ 25 ms p50.

Výstup na anchor (46 hodnot): `person_logit` (1) + 9 × (`dx`, `dy`, `log_sigma`, `present_logit`, `visible_logit`). Dekódování: `x = ax + dx·stride`, `y = ay + dy·stride`; `σ_px = softplus(log_sigma)·stride` (nebo sigmoid·stride, viz §6.3); `conf_k = sigmoid(present)·g(σ)` (§6.4); viditelnost `visible = sigmoid(visible_logit) > 0,5` jen pro body s `conf_k > práh`. Export tensory: `person [1,N,1]`, `pose [1,N,45]`.

Při jiné kombinaci úrovní se mění jen N: {P3,P4,P5}: 2 100 @320; {P3,P5}: 1 700; {P4,P5}: 500.

### 5.3 Ostatní varianty

| Varianta | Role | Návrh | Vstup |
|---|---|---|---|
| u | Pi 5 real-time | §5.2 (A/B/C podle měření) | 320 (256/384) |
| n | edge s rezervou (Pi s vyšší latencí, Jetson, telefon) | = kandidát C nebo B s hloubkami 1/2/2/1 a hlavou 96 | 384 |
| l | GPU, **teacher** pro distilaci | 48/96/192/384/512, hloubky 2/4/4/2, RepConv, SiLU povolena, hlava 128/192, 3 úrovně | 512 |
| m | volitelné (GPU střední) | 32/64/128/256/384, 1/3/3/1 | 384 |

Všechny celoobrazově. Pyramidový režim zůstává v API jako inferenční volba (`view_mode="pyramid"`) bez vlastního tréninku.

### 5.4 Volitelné prvky pro malé obličeje (ablace, ne výchozí)

1. **P2-assist pro pose větev:** `space_to_depth(P2: 32×80²) → 128×40²`, 1×1 → 32, concat s N3 (64) jen pro pose větev na P3; poslední 1×1 nulově inicializovaná (start = baseline). Cena ≈ +8 MMAC. Testovat na metrice small-face PCK (§9).
2. **Dvouúrovňová hlava {P3,P5}** s pevným škálováním fúzovaných rysů (YOLO27): ušetří P4 hlavu (≈45 MMAC) a část necku.
3. **Vstup 384 místo 320** se stejnými vahami: 1,44× výpočtu; rozhodne AP + small-face PCK + latence.
4. **Druhý stupeň „face refine“** (volitelný mód): pro každou detekovanou osobu s `face_spread_net < 12 px` vyříznout ROI 48×48 z původního snímku a spustit mini-síť (≈15 MMAC) pro 5 obličejových bodů. Jen pokud 1–3 nejsou dost a kamera má rezervu. Není v hlavní cestě.

### 5.5 Co se přebírá z XTX3 beze změny funkce

`utils/keypoints.py`, `data/preprocess.py` (geometrie; rozšířit o přímý render na cílové S), `engine/checkpoint.py` (EMA – upravit na `torch._foreach_*`), `export/ncnn_export.py` (pipeline; změnit výstupní kontrakt), `live_test/livetest/{decode,preprocess,runner,merge}.py` (runtime), `engine/merge.py` jen pro pyramidový režim.

---

## 6. Přiřazování, ztráty a dekódování (box-free)

### 6.1 Geometrie targetu bez boxu (jen trénink)

Pro každou GT osobu s labelovanými body `L = {k : v_k > 0}`:
- **Extent box** `E` = obálka bodů v `L` rozšířená o 25 % na každou stranu; minimální strana 16 px (= 2 × stride 8, YOLO26 STAL); pokud `|L| = 1`, čtverec 24 px kolem bodu.
- **Měřítko pro OKS** `s² = area` z COCO boxu, je-li k dispozici (trénink ho smí použít; nasazení ne). Bez boxu: `s² = (1,5·w_E)·(1,5·h_E)` s minimem 16².
- **Kořen osoby** `r` (jen pro tie-break): střed ramen, jinak medián torzových bodů, jinak medián obličejových.
- **Zařazení do výřezu (oprava D4):** osoba je target, pokud má ≥ 1 labelovaný bod uvnitř výřezu; body mimo výřez dostanou `v = 0` s příznakem „mimo obraz“ (present target 0). Osoby s 0 body uvnitř nejsou target ani negativ (ignorovat jejich oblast? ne – pozadí zůstává pozadím, jen se neučí „absent“).

### 6.2 Pose-TAL (task-aligned assignment podle OKS)

Pro každou dvojici (GT g, anchor a):
```
OKS(a,g) = mean_{k∈L_g} exp( −d²_k / (2 · s_g² · (2σ_k)²) )     # σ_k = COCO sigmy 9 bodů
align(a,g) = p_a^α · OKS(a,g)^β · 1[anchor centre ∈ E_g]       # α = 0,5, β = 6 (ER-Pose); ablace α = 1
```
- Kandidáti: anchor centra uvnitř `E_g` (STAL: `E` už má min 16 px; navíc zaručit ≥ 4 kandidáty z nejbližších anchorů ke kořeni `r`, pokud `E` neobsáhne žádný).
- **Warm-up ties:** prvních ~3 epochy jsou OKS ≈ 0 pro všechny anchory; přidat člen `exp(−|a − r|² / (2·(0,5·s)²))` s vahou 0,1, aby se pozitivy soustředily kolem kořene a ne náhodně.
- **o2m:** topk = 10, soft target skóre = `align / max_align_g · max_OKS_g` (jako XTX3), konflikty → GT s vyšším align.
- **o2o:** topk = 7, pak topk2 = 1 podle téhož metriky (YOLO26). Jeden anchor nikdy pro dvě GT.
- **Žádný prior středu obrazu.** (XTX3 `center_prior_weight 0,1` zrušit; ablace A5.)
- Implementace **dávková**: GT paddované na `M_max` v dávce, tenzory `(B, M, N)`; OKS přes broadcast `(B, M, N, K)` ve FP32. Pro B=64, M=32, N=2100, K=9 → 38 M prvků (150 MB fp32) – OK; při větších M použít chunkování.
- Logovat: pokrytí GT (podíl GT s ≥ 1 pozitivem po konfliktech, cíl ≥ 99,5 %), počet pozitiv/GT, podíl anchorů s align < eps, rozložení podle stride.

### 6.3 Ztráty

```
L = w_o2m(t)·L_head(o2m) + w_o2o(t)·L_head(o2o)              # ProgLoss (0,8;0,2) → (0,1;0,9) lineárně po epochách
L_head = λ_person·L_person + λ_kpt·L_kpt + λ_pres·L_present + λ_vis·L_visible
```
- **L_person:** BCE s logity na soft target (YOLO26) jako výchozí; Varifocal (α 0,75, γ 2, jako XTX3) jako ablace. Normalizace součtem target skóre.
- **L_kpt (pozitivní anchory, body s v>0):** `L_oks + w_rle·L_rle`.
  - `L_oks = 1 − SOKS(u)` se `u = d/(s·2σ_k)`; Smooth-OKS (ER-Pose: kvadratická pro u² < 1, lineární v exponentu dál) jako výchozí, čistá OKS jako ablace. Jmenovatel přesně jako evaluátor (`2·s²·(2σ_k)²`), ne `2σ²·area` jako XTX3 `OKSL1Loss`.
  - `L_rle` v **referenční konvenci** (YOLO26/původní RLE): `log σ − log φ(ε) + log 2σ + |ε|`, `ε = (gt − μ)/σ`, σ v jednotkách stride přes `softplus(·) + 0,05` (nebo sigmoid·stride; obě otestovat). RealNVP flow z XTX3 (`rle.py`) zachovat; flow i OKS počítat ve FP32.
  - Poměr `w_oks : w_rle` – začít podle YOLO26 (24 : 1 v jejich normalizaci) a **kalibrovat gradientovými normami** na prvních 200 krocích, aby žádný člen nedominoval o více než 3×; výsledek zapsat do configu.
  - Normalizace: **po osobách** – pro každý pozitivní anchor průměr přes jeho labelované body, pak vážený průměr přes anchory vahou target skóre. (XTX3 průměruje přes všechny body globálně, což zvýhodňuje osoby s více pozitivy.)
- **L_present:** BCE(`present_logit`, `1[v>0]`) pro všech 9 bodů pozitivních anchorů; u datasetů s částečnými labely (CrowdPose/AIC) obličejové body **ignorovat** (maska −1). Ablace D5: váha 0,3 pro `v = 0` u osob, které mají ≥ 5 labelovaných bodů (pravděpodobně „neanotováno“, ne „absent“).
- **L_visible:** BCE(`visible_logit`, `1[v = 2]`) jen kde `v > 0`.
- Váhy start: `λ_person 1,0 · λ_kpt 15 · λ_pres 1,0 · λ_vis 0,5` (ER-Pose 15:1:1); přeškálovat po kalibraci.
- Žádná box ztráta. Teacher ztráty viz §8.4.

### 6.4 Confidence bodu z RLE nejistoty

XTX3 používá `conf = P(occluded) + P(visible)`, což je pravděpodobnost přítomnosti, ne kvality lokalizace. RLE dává σ zdarma. Výchozí: `conf_k = sigmoid(present_k) · exp(−c · σ_k / stride)` s `c` vybraným na validaci (maximalizace AP9); ablace: `conf_k = sigmoid(present_k)`. Pokud `log_sigma` nepomůže, 9 kanálů se při exportu odřízne (Astra T3 postup).

### 6.5 Duplicity a pojistka

Metrika „duplicates@0,25“: dvojice detekcí se skóre ≥ 0,25 a vzájemným OKS ≥ 0,5. Cíl < 0,02 na obrázek. Runtime má přepínač OKS-NMS (host, NumPy, O(D²) při D ≤ 50 = zanedbatelné), výchozí vypnuto; zapne se jen pokud o2o po tréninku nesplní cíl.

---

## 7. Data, augmentace, rozdělení

### 7.1 Rozdělení a holdout

- Zjistit původ XTX3 datasetu (`D:\xtxtraining`, 45 279 / 11 320 obr.) – podle počtů nejde o surové COCO. Agent vytvoří manifest (hashe souborů, zdroj, split) a zkontroluje přesahy mezi splity podle obsahu (perceptual hash), ne jen názvů.
- **Val** (selekce): 2 000 obrázků stratifikovaných podle počtu osob a velikosti obličeje, vyčleněných z trénovacích zdrojů; **plný val** = dosavadní XTX3 test (11 320 obr.) pro srovnání s historií; **holdout** = COCO val2017 person-keypoints (5 000 obr.) použitý **jednou** na konci, plus OCHuman val pro okluze.
- Každý report nese hash checkpointu, hash manifestu, S, práh, maxDet.

### 7.2 Doplňkové sady

Pořadí přidávání (každé jako samostatná ablace, viz §11): Human-Art (17 bodů) → UBody podvzorek (20–40 k snímků, deduplikace sousedních snímků videa) → CrowdPose+AIC jako částečné labely → pseudo-labely z cílové kamery. Smíchávání po dávkách s pevnými poměry (např. COCO 60 %, UBody 15 %, Human-Art 10 %, částečné 5 %, pseudo 10 %). Licenci každé sady ověřit vzhledem k zamýšlenému (komerčnímu) nasazení dříve, než se z ní trénuje finální model.

### 7.3 Augmentace (cílená na kameru a horní polovinu těla)

- **Přímý render na cílové S:** sampler vrací `(idx, S)` a dataset renderuje letterbox rovnou na S z originálu (INTER_AREA při zmenšení). Nikdy `F.interpolate` dávky. Multi-scale poměr 256/320/384 = 0,2/0,5/0,3.
- Mosaic 4 s **letterboxem každého zdroje** (bez deformace poměru stran), p = 0,5 do 70 % epoch, pak 0 (close-mosaic ≥ 15 epoch).
- Scale jitter 0,5–1,5 (×), translate 0,1, flip 0,5 s prohozením párů, HSV (0,015/0,7/0,4).
- **Upper-body crop:** s p = 0,3 náhodný výřez, který odřízne obraz pod boky nebo pod rameny (simuluje blízké záběry); labely mimo výřez → `v = 0` mimo obraz.
- Fotometrika kamery: mírný motion blur (k ≤ 5), gaussovský šum, JPEG 40–95, gamma 0,7–1,4. Nic, co zničí 4px oči (žádný silný blur, žádné cutout na obličeji).
- Copy-paste malých osob (YOLO26n: 0,08) jako ablace.
- Sampling: obrázky se small-face (< 16 px síťově) a ≥ 3 osobami převzorkovat max 2×.

---

## 8. Trénink na RTX 4070 Laptop 8 GB

### 8.1 Propustnost (podmínka všeho ostatního)

Cíl ≥ 200 obr./s při 320 pro u (XTX3: 28). Kroky:
1. Dávkový Pose-TAL a vektorizované ztráty (§6.2) – žádná smyčka přes obrázky v Pythonu.
2. Optimalizátor: výchozí **SGD + Nesterov** (momentum 0,937, lr0 0,01, cosine → 1e-4, warmup 3 ep., wd 5e-4 bez decay na BN/bias). MuSGD jen v referenční podobě (míchání Muon+SGD, dávkový Newton–Schulz přes `torch.bmm`); jedna kontrolní ablace.
3. EMA přes `torch._foreach_mul_/_foreach_add_` (decay 0,9998, warmup 2000 kroků).
4. Data: jednorázová cache obrázků zmenšených na delší stranu ≤ 640 px (JPEG q95 na NVMe); `cv2.setNumThreads(0)` ve workerech; `persistent_workers`, `pin_memory`, `prefetch_factor 4`; num_workers sweep 4/6/8 (Windows spawn).
5. `channels_last`, AMP FP16, `torch.backends.cudnn.benchmark = True`; volitelně `torch.compile` modelu (ne ztrát).
6. Validace po epoše na 2 000 obr. s vektorizovanou OKS maticí (NumPy broadcast, ne dvojitá smyčka jako v `metrics.py`); plný val každých 10 epoch a na konci.
7. Profilovat `torch.profiler` prvních 50 kroků a zapsat podíl dataloader/forward/backward/loss/optimizer; bez tohoto důkazu se další optimalizace nedělají.

### 8.2 Rozvrhy

| Běh | Epochy | Batch @320 | Odhad času při 200 obr./s a 60 k obr. | Účel |
|---|---|---|---|---|
| Smoke | 100 kroků | 64 | minuty | finite loss, VRAM, pokrytí GT |
| Kontrola receptu: XTX3-u architektura + nový recept | 50 | 64 | ~4 h | izoluje přínos receptu od architektury |
| Screening kandidátů u-A/u-B/u-C | 100 | 64 | ~8 h každý | výběr architektury |
| Ablace (jedna změna) | 100 | 64 | ~8 h každá | §11 |
| Finální u | 300 | 64 | ~25 h | release kandidát |
| Teacher l @512 | 200 | 16 (akum. 4) | ~60 h | distilace |
| Distilační fine-tune u | 60 | 64 | ~5 h | release |

VRAM: u-B batch 64 @384 odhad 4–5 GB; l batch 16 @512 odhad 6–7 GB. Při OOM batch 32 + akumulace 2 (BN statistiky se tím mění; uvést).

### 8.3 Stabilita AMP

- Ztráty, assignment, flow ve FP32; `GradScaler`; clip grad-norm 10 přes **model i criterion** (XTX3 clipuje jen model).
- EMA a počitadlo kroků aktualizovat jen po skutečně provedeném `optimizer.step()` (ne při overflow skipu).
- Logovat AMP scale, podíl skipů, grad-norm, kvantily σ, pokrytí assignmentu, img/s, peak VRAM.
- Seed: Python/NumPy/torch/CUDA/DataLoader generator; resume obnovuje model, EMA (`updates`), optimizer, scaler, RNG, epochu; checkpoint uvnitř epochy je „restart epochy“, ne přesné pokračování (zdokumentovat).

### 8.4 Distilace a pseudo-labely (bez inferenčních nákladů)

1. **Teacher A – in-house XTX4-l** (žádný licenční problém): po natrénování spustit na celém tréninku při 512 a uložit pro každou GT osobu 9 souřadnic + σ + present (cache `.npz` na obrázek, s hashem obrázku a affine). Student dostane navíc `L_distill = 0,3 · OKS(student, teacher)` jen pro body s GT `v = 0` a teacher confidence > 0,6 (GT má vždy přednost).
2. **Teacher B – RTMPose-l (Apache-2.0, top-down na GT boxech)** pro obličejové body; stejná cache; použít tam, kde se teacher A a B shodnou (OKS > 0,8) nebo jen B u obličeje.
3. **Pseudo-labely z cílové kamery:** nahrávky bez anotací → teacher A (+B) → filtr: person score > 0,7, ≥ 4 body s conf > 0,6, shoda flip-TTA (OKS > 0,85). 10 % dávky. Doménová adaptace je pravděpodobně největší zisk pro skutečné nasazení.
4. Feature distilace (CWD) až jako poslední ablace, pokud 1–3 nestačí.
5. YOLO26-pose (AGPL) pouze jako externí **referenční měření**, ne jako teacher pro produkční váhy, dokud právní posouzení neřekne jinak.

---

## 9. Evaluace (zmrazit před první ablací)

1. **AP75 oprava:** prahy jako celočíselné procenta `range(50, 100, 5)` nebo indexy; test: jedna perfektní detekce → AP = AP50 = AP75 = AR = 1.
2. **AP9_project:** OKS s 9 sigmami, prahy 50:5:95, 101bodová interpolace, maxDet 100, práh skóre 0,001; **AP9_coco20:** totéž s maxDet 20 pro srovnání s pycocotools (`kpt_oks_sigmas` = 9 sigem, `num_keypoints` = 9, `iscrowd` respektován). Obě varianty pojmenované, nikdy smíchané.
3. Plocha GT pro OKS: COCO `area` pokud existuje, jinak plocha boxu (XTX3 legacy) – uvést v reportu.
4. Reportovat při S = 256/320/384 pro tytéž váhy; vybrat nasazovací S na val.
5. **Doplňkové metriky:** Face-PCK@0,1·face_spread (nenalezená osoba = 0), Face-PCK@2px síťově, Torso-PCK@0,1, NME obličeje (median/p90), bucket `face_spread_net` < 8 / 8–16 / 16–32 / ≥ 32 px, precision/recall @0,25, **duplicates@0,25**, confusion viditelnosti (macro-F1), crowd buckety (1 / 2–4 / 5+ osob), okraje (osoby s body do 16 px od okraje).
6. Holdout (COCO val2017 + OCHuman) jednou na konci na zmrazeném S a prahu.
7. Report obsahuje hash vah (EMA = export), manifest, verze, rozlišení, práh, maxDet.

Baseline pro všechna srovnání: **B0 = XTX3-u best.pt přeměřený touto evaluací** na plném valu při 256/320/384 (bez tréninku). Všechny zisky se vztahují k B0, ne k číslu 0,4589.

---

## 10. Export, runtime, Pi

### 10.1 Export

1. Načíst EMA váhy (hash), deployment kopie, `fuse()` (BN, RepConv), zahodit o2m; nový výstupní kontrakt `person [1,N,1]`, `pose [1,N,45]`; žádný box.
2. ONNX opset 17, statický `[1,3,S,S]` pro S ∈ {256,320,384}, `dynamo=False`; `onnx.checker`; onnxsim; verifikace vs PyTorch (raw atol 1e-3, xy ≤ 0,05 px síťově, skóre ≤ 1e-4) na reálných snímcích i konstantním/extrémním vstupu.
3. pnnx → ncnn: `fp16=1` (nasazení) i `fp16=0` (reference); zaznamenat verze; kontrola, že graf obsahuje jen Convolution/ReLU/Concat/Interp/Pooling/Reshape/Permute.
4. Manifest JSON: schema_version, model_id, hashe, names, strides, N, layout `9×5`, sigmy, barevný kontrakt BGR/255, affine konvence, decode_version, prahy, verze pnnx/ncnn.
5. Parita ncnn FP16 vs PyTorch: p95 xy chyby ≤ 0,25 px síťově, p99 ≤ 1 px, ΔAP9 ≤ 0,5 b.; porovnávat podle anchor id před filtrem.

### 10.2 Runtime na Pi

- ncnn, CPU, `num_threads` 4 (ablace 3 – nechat jádro kameře), `use_fp16_storage/arithmetic = true` (výchozí), `use_vulkan_compute = false`, `lightmode = true`.
- Dekodér: NumPy z `live_test/livetest/decode.py` upravený na kontrakt 1 + 45: sigmoid person → filtr prahu → top-k → dekód jen vybraných → affine zpět. Balíček bez závislosti na torch; test importu v čistém procesu.
- Pipelining: vlákno capture (fronta hloubky 1, zahazovat staré), vlákno inference, vlákno display; logovat stáří snímku a dropnuté snímky.
- Paměť: cíl RSS ≤ 500 MB (model < 10 MB).
- INT8 (ncnn2table KL, ≥ 1 000 kalibračních snímků s malými obličeji a negativy; hlava a poslední 1×1 ve FP16) jen pokud FP16 nesplní rozpočet; očekávaný zisk 1,1–1,3×.

---

## 11. Experimentální plán, gates a očekávané zisky

### 11.1 Fáze

| Fáze | Trvání (odhad) | Obsah | Gate |
|---|---|---|---|
| F0 Základna | 1–2 dny | scaffold `xtx4/`, přenos geometrie/metadat, oprava evaluace, B0 přeměření XTX3-u (256/320/384) a XTX3-n **celoobrazově** (view 1 pyramidy = whole), profiler tréninku XTX3 | G0: perfektní fixture AP75 = 1; B0 report s hashem |
| F1 Latence | 1–2 dny (vyžaduje Pi) | export XTX3-u a kandidátů A/B/C + ablací (ReLU/SiLU/HardSwish, {P3,P5}, P2-assist, DW vs plné bloky) s náhodnými vahami, benchmark na Pi podle §4 | G1: tabulka ms p50/p95 pro každý graf; výběr ≤ 25 ms |
| F2 Pipeline | 2–3 dny | dávkový Pose-TAL, ztráty, přímý render, dataset s částečnými labely, throughput ≥ 200 obr./s, smoke | G2: pokrytí GT ≥ 99,5 %, finite loss, img/s doloženo |
| F3 Kontrola receptu | ~4 h GPU | XTX3-u architektura + nový recept, 50 epoch | G3: AP9 ≥ B0 + 2 b. (jinak hledat chybu v receptu, ne v architektuře) |
| F4 Screening | ~1 den GPU | u-A/u-B/u-C 100 epoch | G4: výběr podle AP9 při latenci z F1 |
| F5 Ablace | 3–4 dny GPU | A1–A12 níže, 100 epoch, jedna změna | zapsat do `ablation.csv` |
| F6 Finále | ~2 dny GPU | vítěz 300 epoch + teacher l (může běžet dřív, paralelně není kvůli 8 GB) | G5: AP9 ≥ B0 + 8 b. na plném valu |
| F7 Distilace | ~1 den | cache teacherů, pseudo-labely, fine-tune 60 ep. | G6: další ≥ +1 b. bez změny latence |
| F8 Release | 1–2 dny | export, parita, 30min Pi běh, holdout, report, model card | G7: p50 ≤ 25 ms, duplicates@0,25 < 0,02, bez throttlingu/OOM |

Celkem ≈ 100–130 GPU-hodin na 4070 a 2–3 týdny kalendářně. Bez Pi lze dělat F0, F2–F7; F1 a F8 čekají.

### 11.2 Ablační matice (jedna změna, 100 epoch, stejná data a seed)

| ID | Změna | Proti | Rozhoduje |
|---|---|---|---|
| A1 | Pose-TAL (OKS) místo IoU-TAL s pomocným boxem | stejná síť s box větví | AP9, pokrytí, duplicity |
| A2 | α 0,5 vs 1,0 v align | A1 | AP9 |
| A3 | o2o topk 7→1 vs 1 | A1 | duplicates@0,25, AP9 |
| A4 | Smooth-OKS+RLE vs OKS vs RLE | – | AP75, face NME |
| A5 | prior středu obrazu 0 vs 0,1 | – | AP9 střed/okraj zvlášť |
| A6 | conf z σ vs present-only | – | AP9 (pořadí) |
| A7 | ReLU vs HardSwish vs SiLU | – | AP9 a ms (z F1) |
| A8 | {P3,P4,P5} vs {P3,P5} vs {P4,P5} | – | AP9, small-face PCK, ms |
| A9 | P2-assist | – | small-face PCK, ms |
| A10 | přímý render vs interpolate | – | small-face PCK |
| A11 | `v=0` váha 0,3 vs 1,0 v present | – | vis macro-F1, AP9 |
| A12 | +Human-Art / +UBody / +částečné / +pseudo | postupně | AP9, holdout, OCHuman |
| A13 | SGD vs referenční MuSGD | – | AP9 @100 ep |
| A14 | multi-scale 256/320/384 vs pevné 320 | – | AP9 při 320 a 256 |

### 11.3 Očekávané přínosy (odhady s oporou, k potvrzení)

| Zdroj zisku | Odhad AP9 | Opora |
|---|---|---|
| Box-free Pose-TAL + Smooth-OKS | +3 až +6 | ER-Pose ablace 50,4 → 57,1 na YOLOv8-pose-n |
| 2–3× více výpočtu při stejné latenci (plné konvoluce) | +3 až +5 | ncnn Pi 5 efektivita (§3.3); YOLO26 škálování n→s |
| 300 epoch místo 50, správný multi-scale, mosaic s letterboxem | +3 až +5 | XTX3-u křivka ještě rostla; YOLO26n 245+150 ep. |
| Data + distilace + pseudo-labely | +1 až +3 (víc v cílové doméně) | DWPose +1,7; UBody +0,6 |
| OKS+RLE kombinace, confidence ze σ | +0,5 až +1,5 | YOLO26 Table 9 |

Celkem řádově +8 až +15 b. AP9 oproti B0 při nižší nebo stejné latenci. Pokud F3 (kontrola receptu) nedá ≥ +2 b., je chyba v implementaci receptu a nemá smysl trénovat nové architektury.

---

## 12. Úkoly pro agenta

Konvence: všechny cesty relativní k `C:/git/freesentry/`. Každý úkol končí důkazem (test/log/report) a zápisem do `xtx4/EXPERIMENTS.md` (run_id, parent, změna, config hash, data manifest hash, seed, img/s, VRAM, výsledek). Nikdy nepřepisovat `best.pt` mezi nepříbuznými běhy. Python prostředí přes `uv` (stejně jako XTX3 launchery).

**T0 – Rešerše a scaffold** (bez GPU)
- Projít §3 zdroje; ověřit, zda vyšel YOLO27 nebo kód ER-Pose; zapsat do `xtx4/RESEARCH.md` s daty přístupu.
- `xtx4/pyproject.toml` (torch/cu13x zvlášť jako v XTX3), `xtx4/xtx4/{utils,data,models,losses,engine,export,runtime}/`, `xtx4/configs/u.json`, `xtx4/tests/`.
- Přenést `keypoints.py`, `preprocess.py` (+ přímý render na S), `geometry.py`, `checkpoint.py`, `logging.py`, `tensorboard.py`.
- Důkaz: `pytest xtx4/tests/test_geometry.py` – parita letterboxu s `xtx3/live_test/tests/test_preprocess_parity.py`.

**T1 – Evaluace a B0**
- `xtx4/xtx4/engine/metrics.py`: oprava AP75, vektorizovaná OKS matice, AP9_project a AP9_coco20, doplňkové metriky §9, duplicates@0,25, pycocotools cross-check.
- `xtx4/scripts/evaluate.py --weights xtx3/runs/xtx3-u/best.pt --sizes 256 320 384 --split full`; totéž pro `xtx3/runs/xtx3-n/best.pt --view-mode whole`.
- Důkaz: `xtx4/reports/B0_xtx3u.json`, testy perfektní/duplicitní/prázdný/crowd fixture.

**T2 – Latence na Pi** (vyžaduje přístup k Pi)
- `xtx4/xtx4/models/xtx4.py` s konfigurovatelnými šířkami/hloubkami/aktivací/úrovněmi; `xtx4/scripts/export_random.py` → ncnn grafy pro XTX3-u, u-A/B/C a ablace F1.
- `xtx4/scripts/bench_pi.py` (protokol §4) + `xtx4/reports/pi_latency.csv`.
- Důkaz: tabulka ms p50/p95 × vlákna × fp16 on/off; volba kandidáta.

**T3 – Datová vrstva**
- `xtx4/xtx4/data/{annotations,dataset,augment,sampler}.py`: manifest a hash splitů, COCO + částečné labely (maska „neznámý bod“), pravidlo zařazení podle bodů, sampler `(idx, S)`, mosaic s letterboxem, upper-body crop, kamerová fotometrika, cache ≤ 640 px.
- Důkaz: `tests/test_targets.py` (ořez s hlavou uvnitř a boxem venku; chybějící boky; flip L/R; affine zpět; půlpixelový resize fixture), vizuální dump 50 vzorků.

**T4 – Pose-TAL a ztráty**
- `xtx4/xtx4/losses/{assign.py,pose_loss.py,rle.py}` dávkově podle §6; diagnostika pokrytí; kalibrace vah gradientovými normami.
- Důkaz: `tests/test_assign.py` (tiny osoba mezi buňkami, dvě osoby v jedné buňce, zero-GT, jednobodová osoba, FP16 vs FP32 shoda), `tests/test_losses.py` (analytické OKS, RLE s identity flow, SOKS spojitost v u = 1).

**T5 – Trainer**
- `xtx4/xtx4/engine/trainer.py`: AMP hranice, foreach EMA, akumulace, clip model+criterion, seed/resume, ProgLoss po epochách, validace 2 000 obr. + plný val po 10 ep., TensorBoard jako XTX3.
- Důkaz: profiler report (podíly), `img/s ≥ 200` na XTX3-u architektuře, smoke 100 kroků @384, resume fixture.

**T6 – Kontrola receptu a screening** (F3, F4) – podle §11.1; výstup `ablation.csv`.

**T7 – Ablace** (F5) – A1–A14 podle priority a dostupného času; každá vlastní run_id.

**T8 – Finální u + teacher l** (F6).

**T9 – Distilace a pseudo-labely** (F7): `xtx4/scripts/cache_teacher.py`, `xtx4/scripts/pseudo_label.py`, kontrola geometrie cache při augmentacích (flip/mosaic/crop).

**T10 – Export a runtime** (`xtx4/xtx4/export/{deploy.py,manifest.py}`, `xtx4/xtx4/runtime/{decode,ncnn_runner,pipeline}.py`, `xtx4/live_test/` jako fork XTX3 live testu na nový kontrakt): parita §10.1, runtime bez torch, OKS-NMS přepínač.

**T11 – Release** (F8): `xtx4/release/<model_id>/` (pt, onnx, ncnn param/bin, manifest, benchmark CSV, accuracy JSON, model card), `xtx4/report.md`.

**T12 – Volitelné:** INT8 kalibrace; face-refine druhý stupeň; pyramidový inferenční režim pro vzdálené osoby.

---

## 13. Rizika a mitigace

| Riziko | Mitigace |
|---|---|
| Pi nedostupné pro F1 | použít libovolné ARMv8.2 zařízení s ncnn jako proxy (poměry mezi grafy drží lépe než absolutní čísla); finální čísla označit „pending“ |
| Plné konvoluce nejsou v ncnn tolik rychlejší, než ukazují cizí benchmarky | F1 to odhalí před tréninkem; pak kandidát A (0,4 GMAC) nebo smíšený design (plné 3×3 na P3+, DW jen v stem) |
| o2o hlava dál produkuje duplicity | topk 7→1, ProgLoss (0,8→0,1), pojistka OKS-NMS na hostu (zanedbatelná cena) |
| RLE flow nestabilní v AMP | FP32 pro flow, clamp ε ±100, fallback OKS-only (A4) |
| Box-free assignment konverguje pomalu na začátku | warm-up tie-break kolem kořene osoby (§6.2), STAL min 4 anchory |
| Licence doplňkových datasetů / teacherů | před finálním tréninkem seznam zdrojů a podmínek; produkční model jen z povolených dat |
| 8 GB VRAM na teacher l | batch 16 + akumulace; teacher cachovat offline, nikdy teacher+student současně na GPU |
| Windows dataloader (spawn) | cache ≤ 640 px, sweep workerů, `persistent_workers` |
| Přetrénování na val při mnoha ablacích | holdout jednou na konci; ablace s pevným prahem a maxDet |

---

## 14. Seznam zdrojů k prostudování (agent začne tady)

- YOLO26 článek: https://arxiv.org/abs/2606.03748 ; docs: https://docs.ultralytics.com/models/yolo26/ , https://docs.ultralytics.com/tasks/pose/ , ztráty: https://docs.ultralytics.com/reference/utils/loss/ , MuSGD: https://docs.ultralytics.com/reference/optim/muon , Pi guide: https://docs.ultralytics.com/guides/raspberry-pi/
- YOLO27 preview: https://docs.ultralytics.com/models/yolo27/
- ER-Pose: https://arxiv.org/abs/2603.08681
- CPUBone: https://arxiv.org/abs/2603.26425 , kód: https://github.com/altair199797/CPUBone
- RTMO: https://arxiv.org/abs/2312.07526 , https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo
- RTMPose: https://arxiv.org/abs/2303.07399 ; DWPose: https://arxiv.org/abs/2307.15880
- SimCC: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4591_ECCV_2022_paper.php
- MobileNetV4: https://arxiv.org/abs/2404.10518 ; RepViT / MobileOne (CVPR 2023/24); StarNet: https://arxiv.org/abs/2403.19967 ; LSNet: https://arxiv.org/abs/2503.23135
- ncnn benchmark na Pi 5: https://qengineering.eu/ncnn_rpi5.html ; ncnn Option: https://github.com/Tencent/ncnn/blob/master/src/option.cpp ; ncnn int8: https://github.com/Tencent/ncnn/wiki/quantized-int8-inference ; pnnx: https://github.com/Tencent/ncnn/tree/master/tools/pnnx
- INT8 na A76: https://arxiv.org/abs/2609.16085
- Aktivace na ARM (ActNAS): https://arxiv.org/abs/2410.10887
- Datasety: COCO https://cocodataset.org ; OCHuman https://github.com/liruilong940607/OCHumanApi ; Human-Art https://arxiv.org/abs/2303.02760 ; UBody (OSX, CVPR 2023) https://arxiv.org/abs/2303.16160 ; COCO-WholeBody https://arxiv.org/abs/2007.11858
- Luxonis pose-aware TAL (OKS v alignment metrice, otevřený kód): https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-train/api-reference
- pycocotools COCOeval: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

---

## 15. Kde se tento plán liší od `implementation_plan.md` (Astra) a proč

Shoda: oprava AP75 a validace, oddělení val/holdout, hash vah u reportů, parita exportu, rozlišení FP16 storage/arithmetic, NumPy runtime z live_testu, box jako nepovinný výstup, gates a reprodukovatelnost. To vše zde přebírám.

Rozdíly:
1. **Architektura:** Astra zachovává XTX3-u (DW/PConv) a mění ji jen „podle naměřené rezervy“. Tento plán z cizích měření na Pi 5 (ncnn tabulka, CPUBone) odvozuje, že DW design je na tomto CPU 2–3× méně efektivní na MAC, a navrhuje konkrétní hustou síť (u-A/B/C, §5.2) s výběrem podle změřené latence **před** tréninkem (F1). To je přímá odpověď na zadání „navrhnout od nuly, pokud to dává smysl“.
2. **Box-free jako hlavní cesta:** Astra nechává box jako pomocný trénink a box-free řadí do podmíněné kapitoly 8.2. Zde je Pose-TAL výchozí, s oporou v ablacích ER-Pose (+6,7 AP na n) a s konkrétní geometrií (extent box, měřítko, kořen, warm-up), a box větev zmizí i z tréninku.
3. **Propustnost tréninku** jako první technický úkol (28 → ≥ 200 obr./s). Astra ji neřeší; bez ní nejsou dlouhé rozvrhy (YOLO26n 395 ep., ER-Pose 600 ep.) realizovatelné na jedné laptopové GPU.
4. **Pyramidový režim** zrušit pro trénink (důkaz z logů n vs u), Astra ho neposuzuje.
5. **Konkrétní receptura** (ztráty, váhy, assigner, augmentace, rozvrhy, distilace, pseudo-labely z cílové kamery) a konkrétní data s licenčními poznámkami; Astra zůstává u obecných pravidel.
6. **ncnn specifika:** Vulkan na Pi 5 vyloučen měřením, FP16 je výchozí (ne „ablace“), INT8 má doložený malý zisk, aktivace ReLU vs SiLU.
7. **Dvouúrovňová hlava a P2-assist** jako ablace pro malé obličeje s oporou v YOLO27/ER-Pose.
8. Astra má správně, že řada mých čísel jsou odhady; proto má každý krok gate s měřením a plán začíná B0 a F1, ne tréninkem.
