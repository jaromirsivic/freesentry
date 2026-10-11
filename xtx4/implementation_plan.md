# XTX4: implementační plán devítibodové multi-person pose

Datum rešerše: 11. 10. 2026. Stav: plán k následné realizaci, žádný nový model nebyl implementován ani trénován. Termín této výzkumné práce je podle uživatele neomezený.

## 1. Rozhodnutí a rozsah

Doporučená cesta je převzít kompatibilní XTX3-u, opravit měření a nejdříve využít změny, které zvyšují kvalitu tréninku nebo zmenšují export bez zdražení inference. Nezačínat kompletním přepisem na transformer, plošnou P2 hlavou ani novým box-free matcherem. V prvním nasaditelném kandidátovi ponechat box jako pomocný tréninkový úkol, ale při whole-frame one-to-one inferenci odstranit celý box tower a nepoužívané sigma kanály. Přesnost potom zvyšovat kontrolovanými změnami assignmentu, loss, dat a distilací. Architekturou utrácet jen skutečně naměřenou rezervu.

Současně zařadit jeden vnější referenční model, pretrained YOLO26n-pose, redukovaný nebo při evaluaci vybraný na stejných 9 bodech. Před přetrénováním provést jeho export a benchmark při 256, 320 a 384. Pokud překoná XTX3 v reálném accuracy/latency měření, není důvod udržovat vlastní backbone jen kvůli kontinuitě názvu. Žádné převzetí vah jiné architektury pomocí pouhého strict=False nevydávat za korektní transfer.

Pořadí bodů je závazné:

| Index XTX4 | Bod | COCO index | Flip partner |
|---|---|---|---|
| 0 | nos | 0 | 0 |
| 1 | levé oko | 1 | 2 |
| 2 | pravé oko | 2 | 1 |
| 3 | levé ucho | 3 | 4 |
| 4 | pravé ucho | 4 | 3 |
| 5 | levé rameno | 5 | 6 |
| 6 | pravé rameno | 6 | 5 |
| 7 | levý bok | 11 | 8 |
| 8 | pravý bok | 12 | 7 |

Cílem je více osob v jednom snímku, přesné malé body obličeje, zachování schopnosti detekovat zakryté a částečně oříznuté osoby a malá koncová latence. Bounding box uživatel nevyžaduje. Varianta u má mířit co nejblíže real-time na Raspberry Pi 5 se 4 GB RAM. Hodnota 25 FPS, tedy přibližně 40 ms na výsledek, je převzatá aspirace z XTX3 README/reportu. Není nově potvrzeným tvrdým požadavkem uživatele. Navržené akceptační prahy v tomto plánu jsou inženýrské rozhodovací meze, nikoli již naměřené vlastnosti.

Tréninkový hardware je NVIDIA GeForce RTX 4070 Laptop GPU s 8 GB VRAM. Výchozí trénink používá AMP FP16. Distribuují se ONNX a ncnn. Pro Pi je výchozí ncnn CPU, další backend je pouze měřená ablace.

### Oprávnění a hranice této práce

Při přípravě plánu se četly pouze místní soubory uvnitř C:/git/freesentry/xtx3. Webová rešerše byla povolena. Nečetly se jiné místní projekty, externí dataset, lokální skills ani git metadata. Nebyl spuštěn training, inference ani skript, který by mohl načíst data mimo povolenou složku. Zejména původní profiler a benchmark obsahují odkazy na XTX2, a proto nebyly spuštěny.

Jediný povolený zápis je tento nový soubor C:/git/freesentry/xtx4/implementation_plan.md. Realizace níže je budoucí práce. Před ní musí být jasně autorizována implementace v xtx4, čtení konkrétního datasetu a přístup k Pi. Dnešní souhlas s výzkumem tyto přístupy automaticky neuděluje. Při realizaci pracovat v aktivní feature větvi, nevytvářet worktree bez pokynu, zjistit stav cílových souborů a zachovat cizí změny. Python balíčky spravovat pouze přes uv. Nevytvářet další agenty bez příslušné autorizace.

## 2. Co skutečně dokládá XTX3

### 2.1 Architektura a běh

Zdroje: [head.py](C:/git/freesentry/xtx3/xtx3/models/head.py), [xtx3.py](C:/git/freesentry/xtx3/xtx3/models/xtx3.py), [neck.py](C:/git/freesentry/xtx3/xtx3/models/neck.py), [blocks.py](C:/git/freesentry/xtx3/xtx3/models/blocks.py), [config.json](C:/git/freesentry/xtx3/config.json).

- U používá CSP backbone, částečné konvoluce v raných stupních, depthwise bloky, PAN/FPN s P3/P4/P5 a stridy 8/16/32.
- Každá z obou tréninkových hlav má oddělené box, person a keypoint towers. One-to-many se při exportu zahazuje, inference dnes používá one-to-one.
- Každý bod má dx, dy, log_sigma a tři visibility logity. Sigma je v inference výstupu přenášena, ale dekodér ji pro souřadnice ani confidence nepoužívá.
- Confidence bodu znamená P(v=1)+P(v=2), tedy pravděpodobnost anotovaného přítomného bodu. Není to kalibrovaná pravděpodobnost přesné lokalizace.
- U má whole-frame režim, výchozí inference 320 a multiscale training 256/320/384. N/m/l standardně používají tři pyramidové pohledy, což není férová přímá baseline pro jeden průchod u.
- NMS-free je návrh tréninku a dekódování. Samotné označení nezaručuje, že síť opravdu nevytváří duplicity.

[report.md](C:/git/freesentry/xtx3/report.md) uvádí pro u 0,82 M parametrů a 0,249/0,389/0,561 „GFLOPs“ při 256/320/384. [count_gflops](C:/git/freesentry/xtx3/xtx3/engine/trainer.py:150) a [profiler](C:/git/freesentry/xtx3/scripts/profile_flops.py) ve skutečnosti počítají násobení s akumulací konvolucí a lineárních vrstev. Přeznačit na GMAC. Při konvenci násobení plus součet jako dvě operace je konvoluční FLOP přibližně dvojnásobný. Operace aktivací, resize, paměťové přesuny a host postprocessing v tomto čísle nejsou. Počet parametrů dále rozlišit na training včetně o2m a criterion, eval a skutečný fused/pruned deployment. Hodnota 0,82 M není důkaz velikosti finálního ořezaného grafu.

V reportu je historický desktop CPU údaj přibližně 36 FPS, ale ne úplný reprodukovatelný hardware/protokol. Raspberry Pi tabulka pro XTX3-u zůstává prázdná. Existující soubory ONNX/ncnn dokládají vznik artefaktů, nikoli jejich kvalitu na Pi nebo splnění 25 FPS.

### 2.2 Logy a výsledky

Zdroje: [train.log u](C:/git/freesentry/xtx3/runs/xtx3-u/train.log), [eval_report.json u](C:/git/freesentry/xtx3/runs/xtx3-u/eval_report.json), [train.log n](C:/git/freesentry/xtx3/runs/xtx3-n/train.log).

| Běh | Poslední zalogovaná epocha | Nejlepší zalogované AP | Poznámka |
|---|---:|---:|---|
| xtx3-u | 50 | 0,4589 | 47 val záznamů, více resume |
| xtx3-u - Copy | 20 | 0,2702 | historie části stejné práce |
| xtx3-u - Copy (2) | 37 | 0,4112 | historie části stejné práce |
| xtx3-u - Copy (3) | 50 | 0,4589 | train.log má stejný SHA256 jako xtx3-u |
| xtx3-n | 27 | 0,2706 | nedokončený, jiný view režim |
| xtx3-n - Copy | 23 | 0,2706 | nelze považovat za nezávislý seed |
| xtx3-n - Copy (2) | 24 | 0,2706 | nelze považovat za nezávislý seed |

Shodný hash obou kompletních u logů: 49686DE37199AFF86BB4918C8F466A149E5E1FA2C16E4457797F988F9911F6EA. Kopie nejsou sedm nezávislých experimentů. U log hlásí train 45 279 snímků / 119 735 osob a test 11 320 snímků / 30 078 osob. Jde o obsah logu, dataset nebyl otevřen ani nezávisle auditován.

Detailní report u uvádí AP9=0,458940, AP50=0,706401, AP75=0, AR=0,573005. Při confidence 0,05 a max_detections=100 má 469 TP, 3091 FP a 120 FN, precision 13,17 %, recall 79,63 %. To není provozní precision při výchozím prahu 0,25. Je nutné změřit PR křivku a duplicity při totožném operating point. Nelze z tohoto samotného čísla vyvodit, že deployment vytváří stejně mnoho falešných osob.

Matched-only mean OKS je pro nos 0,760, oči 0,783/0,794, uši 0,840/0,843 a ramena 0,878/0,881. Tyto hodnoty zvýhodňují úspěšně nalezené osoby a různé sigmy bodů, nejsou měřítkem všech chyb obličeje. Původní small-face bucket má 171 GT a recall 0,825, ale vzniká z rozměru v originálních pixelech, nikoli v síťovém vstupu. Nelze jej použít jako důkaz zvládnutých čtyřpixelových obličejů při 320.

### 2.3 Priority auditu, které ovlivňují návrh

| Priorita | Zjištění | Důsledek pro realizaci |
|---|---|---|
| Blokuje srovnání | [metrics.py](C:/git/freesentry/xtx3/xtx3/engine/metrics.py) vytváří prahy np.arange a AP75 vyhledává přesný float klíč 0.75 | Opravit celočíselnými procenty nebo indexy, potvrdit perfektním fixture. Nulové AP75 není důkaz špatné lokalizace |
| Blokuje srovnání | [trainer.py](C:/git/freesentry/xtx3/xtx3/engine/trainer.py) vybírá jen prvních max_images=200 a pro training používá split nazvaný test | Oddělit validation a nedotčený test. Dosavadní test je již development sada |
| Blokuje srovnání | Final detailed hodnotí aktuální EMA, export načítá best.pt | Každý finální report svázat s konkrétním hashem exportovaných best EMA vah |
| Blokuje runtime gate | [benchmark.py](C:/git/freesentry/xtx3/scripts/benchmark.py) neměří torch a ncnn přes totožný finální výstup | Změřit obě pipeline až po zpětné transformaci, filtrování a vytvoření poses |
| Velký levný přínos | [ncnn_export.py](C:/git/freesentry/xtx3/xtx3/export/ncnn_export.py) používá torch host decode a dekóduje všechny body před filtrem | Převzít existující NumPy decode z live_test, nesestavovat třetí implementaci |
| Tréninková hypotéza | [assignment.py](C:/git/freesentry/xtx3/xtx3/losses/assignment.py) filtruje score^alpha*IoU^beta přes eps, top1 nemá fallback a kolize mohou připravit GT o pozitivum | Instrumentovat GT coverage a opravit na reprodukovatelném selhání, hlavně pro malé a překrývající se osoby |
| Tréninková hypotéza | [rle.py](C:/git/freesentry/xtx3/xtx3/losses/rle.py) není formulačně totožný s referenčním RLE | Neprohlašovat dvojí Laplace za samostatně dokázanou chybu. Porovnat přesnou hustotu, log_sigma a redukci, viz úkol T5 |
| Datová chyba pro cílové chování | [dataset.py](C:/git/freesentry/xtx3/xtx3/data/dataset.py) zahodí osobu, když centrum full-person boxu leží mimo crop | Horní část těla může být uvnitř a přesto nedostat target. Ověřit cíleným crop fixture |
| Omezení metrik | [annotations.py](C:/git/freesentry/xtx3/xtx3/data/annotations.py) neuchovává původní annotation area, evaluator nemá crowd ignore | Zachovat legacy AP9bbox a přidat jasně označenou COCO-kompatibilní AP9 evaluaci, nezaměnit je |

Další cílené trainer opravy patří do jednoho úkolu, nikoli do obecného přepisu infrastruktury: seed dnes řídí hlavně sampler, chybí plný RNG state a ema.updates při resume, časový checkpoint uvnitř epochy pokračuje na další epochě, criterion flow se neclipuje spolu s modelem a AMP/pinned-memory kontroly device=='cuda' nezahrnují cuda:0. Tyto věci jsou relevantní zejména tehdy, když se skutečně používá resume a dlouhé běhy.

## 3. Webová rešerše a výběr současných kandidátů

Všechny odkazy v této kapitole byly ověřovány 11. 10. 2026. Datum přístupu není datum vydání. Živé větve main/master a docs mohou být změněny, proto implementační agent před experimentem uloží konkrétní verzi balíčku, commit dostupný z webu a hash checkpointu. Rešerše není tvrzením úplnosti všech publikací.

| Kandidát | Doložený stav a datum | Význam pro XTX4 |
|---|---|---|
| YOLO26 | Vydaná rodina, oficiální článek arXiv v1 z 2. 6. 2026, docs označují release leden 2026 | Nejdůležitější vnější baseline a zdroj training-only postupů |
| YOLO27 | Oficiální docs při přístupu stále uvádějí coming soon, bez veřejných vah, konfigurací a implementace | Neplánovat na něm závislou implementaci. Po vydání případně zopakovat krátký benchmark |
| RTMO-t / RTMO-s | Oficiální model zoo od prosince 2023, CVPR 2024 | Alternativa jednostupňového multi-person odhadu, RTMO-t má veřejný ONNX při 416 |
| RTMPose-t/s, RTMW, DWPose | RTMPose od března 2023, DWPose ICCV workshop 2023 | Především teacher nebo top-down kontrolní reference, která musí započítat detektor a všechny osoby |
| DETRPose | Primární arXiv 2506.13027, první verze 16. 6. 2025 | Dokládá novější end-to-end transformer směr, pro Pi-u pouze podmíněný kandidát |
| RF-DETR Keypoint Preview | Dostupný preview v primárních Roboflow zdrojích při přístupu, přesné datum vydání v otevřeném zdroji nezjištěno | Současná kvalitativní reference/teacher, ne výchozí model u |

### 3.1 Co převzít z YOLO26

Oficiální článek popisuje DFL-free regresi, duální assignment, progressive loss, STAL, MuSGD a kombinaci OKS s RLE. Pose ablace ukazuje, že kombinace loss může být výhodnější než samotné RLE. To je důvod k lokální ablaci, nikoli důkaz optimálních vah pro 9 bodů a miniaturní u. [Oficiální článek YOLO26](https://arxiv.org/html/2606.03748v1).

Aktuální dokumentace rozlišuje dense větev s NMS pro výchozí inference/validation a volitelnou one-to-one větev přes nms=False. Článek a živé API se mohou v defaultu lišit. V manifestu explicitně zapsat zvolenou větev, nepřebírat default nepřipnuté verze. [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26).

Oficiální pose tabulka uvádí pro YOLO26n-pose při 640 hodnotu 57,2 AP na standardních 17 bodech, 2,9 M parametrů, 7,6 GFLOPs a 40,3 ms CPU ONNX. CPU v této tabulce není Raspberry Pi. Tyto výsledky nesrovnávat s lokálním AP9=45,89 jako přímý nárůst přesnosti. [YOLO26 pose model zoo](https://docs.ultralytics.com/tasks/pose).

YOLO27 má při přístupu zveřejněné předběžné tabulky, ale současně výslovně říká, že modely ještě nejsou k dispozici. Jeho uváděná rychlost na RTX PRO 6000 není důkaz vhodnosti pro Pi. Zablokovaný download nebo absentní release se nesmí nahradit neověřeným komunitním checkpointem. [Oficiální YOLO27 preview](https://docs.ultralytics.com/models/yolo27).

### 3.2 Ostatní metody

RTMO používá dynamickou klasifikaci souřadnic ve společné jednostupňové síti. Oficiální RTMO-t je 416x416, trénovaný na body7 a má ONNX. Body7 označuje kombinaci sedmi datasetů, nikoli sedm keypointů. Uváděné latence RTMO-s/m/l jsou na V100 s ONNX Runtime. Pro XTX4 je užitečné srovnání reprezentace souřadnic, ale převzetí dynamického classifieru znamená větší riziko ncnn exportu. Podpora ncnn pro RTMPose sama nedokazuje podporu konkrétní RTMO cesty. [Primární RTMO model zoo a deployment](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo), [RTMO paper](https://arxiv.org/abs/2312.07526).

RTMPose má doložené ONNX/ncnn nasazení. Jeho pipeline tabulka však používá detekční interval pěti snímků a top-down zpracování osob. Měřit T_detector + počet_osob * T_pose + crop/decode a zvlášť nově příchozí osoby. Z tohoto důvodu zůstává model s jedním průchodem pro u výchozí. DWPose podporuje smysl distilace jako směru, nepředepisuje úspěch pro naši devítibodovou síť. [RTMPose primární README](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [DWPose](https://github.com/IDEA-Research/DWPose).

DETRPose uvádí end-to-end multi-person transformer a rychlejší konvergenci. Kvalitu a export musí agent prověřit samostatně, bez předpokladu, že nižší počet epoch znamená nízké lokální náklady. [DETRPose, primární arXiv](https://arxiv.org/abs/2506.13027).

RF-DETR Keypoint Preview má v primární dokumentaci 40,7 M parametrů, COCO17 AP 71,8 a 9,7 ms na NVIDIA T4 s TensorRT FP16 při 576x576. Je to dostupné preview s proměnlivým API. Jeho velikost a GPU benchmark jej neopravňují k označení jako real-time Pi model. Vhodnější je případná offline teacher inference. [Roboflow RF-DETR keypoint documentation](https://github.com/roboflow/rf-detr/blob/develop/docs/learn/run/keypoints.md). Nespouštět těžkého teachera a studenta současně jako povinný předpoklad na 8GB GPU.

Heatmap/associative-embedding nebo OpenPose-like bottom-up přepis může odstranit box, ale přidá grouping, tuning kolizí a husté výstupy. Na devíti bodech není automaticky rychlejší. Ponechat mimo hlavní plán, pokud levnější přímá regrese projde gates.

### 3.3 Co web skutečně říká o Pi a FP16

Oficiální Ultralytics Pi guide uvádí pro detekční YOLO26n při 640 ncnn inference 67,03 ms ve své tabulce z verze 8.4.108. Výslovně nezahrnuje pre/postprocessing. Jde o detekci, jiné váhy a jiný vstup, nikoli o XTX4 pose. Údaj pouze potvrzuje potřebu měřit skutečný hardware, ne slib 25 FPS přepočtem FLOPs. [Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi).

Pi 5 používá Cortex-A76. ARM uvádí pro A76 IEEE FP16 a INT8 dot product, takže kategorické tvrzení „FP16 je na CPU Pi nemožné“ je nesprávné. Podmínkou je odpovídající build a kernel. [Raspberry Pi 5 specifications](https://www.raspberrypi.com/products/raspberry-pi-5/), [ARM Cortex-A comparison table](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Cortex-A%20R%20M%20datasheets/Arm%20Cortex-A%20Comparison%20Table_v4.pdf).

ncnn odděluje FP16 storage a arithmetic. ARM convolution obsahuje obě cesty, volené podle runtime voleb a dostupné podpory. Menší binární soubor s FP16 vahami proto sám nedokazuje výpočet v FP16 ani zrychlení. [ncnn Option](https://github.com/Tencent/ncnn/blob/master/src/option.h), [ARM convolution dispatch](https://github.com/Tencent/ncnn/blob/master/src/layer/arm/convolution_arm.cpp), [ncnn layer support](https://github.com/Tencent/ncnn/wiki/layer-support-behavior).

ONNX uchovávat jako přenosnou FP32 referenci. Dokumentace ONNX Runtime upozorňuje na omezení FP16 CPU cesty. Nelze přenést očekávaný přínos ncnn FP16 na CPUExecutionProvider pouhou konverzí ONNX tensorů. [ONNX Runtime float16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

## 4. Navržený model a rozhraní

### 4.1 Výchozí nasaditelná varianta

Zachovat backbone/PAN/stridy XTX3-u, o2m/o2o trénink a prvotně současnou 3-class visibility. Vytvořit samostatný deployment wrapper, který volá pouze person a keypoint towers o2o. U poslední 1x1 keypoint projekce při exportu vybrat pro každý bod kanály dx, dy, vis0, vis1, vis2. Vyjmout log_sigma fyzicky z vah a biasu této projekce. Nechat celý tréninkový model a checkpoint nedotčené.

Exportovaný v1 kontrakt:

- Vstup: float32 NCHW, batch 1, shape [1,3,S,S], S v {256,320,384}.
- Barvy pro XTX3 transfer: BGR a dělení 255, nulový padding a stávající letterbox, přesná affine transformace v metadatech.
- Výstup person_logits: [1,N,1].
- Výstup pose_raw: [1,N,45], pro každý bod pět hodnot dx,dy,vis0,vis1,vis2.
- N je 1344 při 256, 2100 při 320 a 3024 při 384 pro stridy [8,16,32].
- Host používá FP32 sigmoid/softmax, score filtr a stabilní pořadí podle score, při shodě anchor index. Poté dekóduje pouze vybrané kandidáty a vrátí body v originálních pixelech.
- Veřejné Pose9 obsahuje person_score, keypoints [9,3], visibility [9] a volitelnou technickou source_view. Kompaktní pole může být [D,28], tedy score + 9*(x,y,confidence). Žádný povinný bbox.
- Výstupy neoznačovat automaticky za trvalé identity. Multi-person tracking a časové vyhlazování nejsou součástí základního měření.

Při whole-frame režimu nemá cross-view merger co slučovat. Box nepotřebujeme k decode xy ani k výpočtu score. Při pozdější volbě o2m+NMS musí konkrétní model profil box znovu poskytovat, nebo mít ověřenou pose-NMS. Nevracet smyšlený bbox a nepoužívat neověřený min/max bodů jako drop-in náhradu full-person boxu.

Statický odhad box tower u při 320 z rozměrů a depthwise konvolucí je přibližně 19,0 MMAC, asi 4,9 % deklarovaných 0,389 GMAC. Nezahrnuje aktivace, BN a paměťový provoz. Vyřazení devíti sigma projekcí je dalších přibližně 1,21 MMAC. Hustý transport klesne z 59 na 46 hodnot na anchor, asi o 22 %. Celková latence se nemusí zmenšit úměrně. Parametry ubudou v deployment grafu, nikoli v tréninkovém modelu.

Podmínka úspěchu je, že ONNX a ncnn skutečně neobsahují box tower ani sigma kanály. Pouhé ignorování vráceného box tensoru na hostu úsporu sítě nedokládá. Perfektně zachované poses při menší latenci jsou samostatný úspěšný výsledek, nemusí zároveň získat +2 AP.

### 4.2 Přesnost malého obličeje při omezené latenci

Rozlišení má přednost před přidáváním exotických operátorů. Změřit 256/320/384 na stejných vahách. Zvýšení 320 na 384 znamená 1,44x více vstupních pixelů, nikoli bezplatné zlepšení. Přínos rozhodne společné AP, face-PCK a E2E latency.

Pokud training-only změny nestačí a Pi má rezervu, zkoušet právě jednu z těchto architektur:

1. Kpt hidden cap 48/64/80 při ostatní topologii beze změny.
2. Lehká P2-to-P3 fúze pouze do keypoint větve. P2 má u 48 kanálů, použít 1x1 na 24, depthwise 3x3 stride2, 1x1 na 96 a přičíst k P3 kpt feature. Poslední projekci inicializovat nulově, aby začátek odpovídal baseline. Class/box a zbytek necku ponechat pro tuto ablaci stejné.
3. Teprve pokud face chyba prokazatelně vzniká prostorovým rozlišením, otestovat úzkou P2 predikční hlavu. Při 320 přidává 6400 kandidátů k původním 2100, dohromady čtyřnásobný počet, což významně zasáhne head, transport a decode.
4. PConv/DWConv nahrazovat pouze po per-layer měření. Nízké MAC neznamenají nízkou latenci. Partial split/concat a depthwise mohou být memory-bound. ReLU/Rep bloky nebo jiné šířky netestovat současně s P2, jinak nebude jasný původ změny.

Nehodnotit malý obličej jen podle velikosti osoby. STAL reagující na min side person boxu neřeší automaticky pět titěrných face bodů velké osoby.

## 5. Metriky, datasety a rozhodovací gates

### 5.1 Zmrazit evaluaci před první ablací

1. Po budoucím povolení datasetu sestavit manifest zdrojových obrázků, anotací a splitů. Zaznamenat hashes, původ, skupinu kamery/sekvence a počty osob. Kontrolovat duplicity mezi splity podle obsahu i sekvence, ne jen názvu souboru.
2. Stávající složku test označit jako legacy development, protože se podle ní vybíral checkpoint. Netvrdit, že nové náhodné rozdělení jejích snímků vrací skutečně nedotčený test. Získat oddělený holdout z nepoužité sekvence/kamery nebo explicitně uvést chybějící důkaz generalizace.
3. Validation rozdělit na rychlý stratifikovaný monitorovací subset a plný finální val. Nikdy automaticky prvních 200 řádků. Rychlý subset neslouží jako jediný důkaz zlepšení.
4. Na celé dostupné finální eval sadě znovu vyhodnotit současný best.pt se stejnou EMA jako export při 256/320/384. K tomu není nutný nový training. Uchovat legacy threshold 0,05 jako reprodukční údaj, hlavní AP sbírat s dostatečně nízkým prahem, výchozí 0,001, a ověřit, že další snížení už AP nemění.
5. Zvolit jednoznačné AP9_project s maxDet=100, prahy 50:5:95, COCO subset sigmami a identickou GT area mezi kandidáty. Současně uvést COCO-style AP9_max20 pro konvenci oficiálního keypoint evaluátoru. Hodnoty s jiným maxDet se nesmějí mísit.
6. Zachovat AP9_bbox_legacy s původní full-person bbox area pro reprodukci. Pokud jsou dostupné původní COCO annotation area a crowd/ignore, použít je v nové standardizované evaluaci, jasně pojmenovat variantu metriky a přepočítat i baseline. Predikovaný box nebo nový torso box nikdy nesmí měnit normalizaci referenčního GT.
7. Novou evaluaci porovnat s pycocotools COCOeval s devítiprvkovým kpt_oks_sigmas, správným num_keypoints, area, iscrowd/ignore a dohodnutým maxDets. COCO9 je vlastní subset protokol, není číselně stejný jako oficiální COCO17 AP. [Primární COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
8. Konfidenční operating point vybrat pouze na val, například maximem F1 při recall nejméně 0,80. Pokud žádný takový bod neexistuje, uvést kompromis a kompletní křivku, práh potichu nesnižovat. Test pak použije zmrazený práh i maxDet.
9. Reportovat také precision/recall při shodném pevném prahu 0,25 a recall při shodné FP/image. Každý experiment nesmí získat výhodu jen jiným prahováním.

AP75 vybírat indexem 5 v pevném seznamu deseti prahů nebo prahy jako integer 50,55,...,95. Perfektní jediná detekce musí dát AP=AP50=AP75=AR=1. Současný float lookup je první věc, kterou test chrání.

### 5.2 Definice kvality obličeje a torza

Pro každý GT s alespoň dvěma anotovanými face body v>0 definovat face_spread jako maximální párovou eukleidovskou vzdálenost těchto bodů v původním obraze. Pro stejný GT spočítat face_spread_net po přesném validačním preprocessing v daném S. Nezaměňovat S=320 a S=384 ani originální rozlišení kamer.

Reportovat obě klasifikace:

- Historické original-pixel face buckets <16, 16 až <48, >=48 pro reprodukci XTX3.
- Pro nový experiment síťové buckets <4, 4 až <8, 8 až <16, >=16 pixelů. Hlavní small-face skupina je face_spread_net<16. GT s méně než dvěma face body nebo nulovým spread má explicitní unknown-scale skupinu.

Person matching pro detailní diagnostiku provést deterministicky podle score a OKS9>=0,5. Vedle matched-only chyb uvádět detekčně podmíněnou metriku zahrnující všechna GT:

- Face-PCK@0.1: u každého anotovaného face bodu je úspěch, pokud nalezená přiřazená osoba má vzdálenost bodu nejvýše 0,1*face_spread. Nenalezená osoba má u všech svých anotovaných bodů úspěch 0. Nezvyšovat toleranci skrytým floor podle výsledku modelu.
- Face-PCK@2px-net: doplňkově chyba <=2 pixelů po transformaci do daného síťového vstupu. Slouží k porozumění kvantizaci a šumu velmi malých obličejů.
- Face NME: vzdálenost/face_spread na matched bodech, median/p90/p95. Vždy vedle coverage, nikoli místo metriky započítávající missed osoby.
- Torso-PCK@0.1 obdobně vůči maximální párové vzdálenosti anotovaných ramen/boků, při méně než dvou bodech unknown-scale. Nepředpokládat viditelné boky.
- Rozdělení GT bodů v=1 a v=2 hodnotit samostatně. v=0 je neanotovaný bod bez lokalizačního targetu, nesmí se přičítat jako správná nula.
- U visibility uvést confusion matrix, macro-F1 a četnost tříd. Samotná accuracy může skrýt většinové absent body. Nepoužívat predikovanou confidence ke skrytí špatných bodů z PCK.
- Crowd skupiny určit počtem GT osob 0,1,2 až 4,5 až 8,9+ a navíc překryvem GT boxes. Reportovat duplicity, person swaps, edge/truncation a occlusion. Jedna nalezená osoba nesmí být TP pro více GT.

Pro hlavní small-face gate musí sada obsahovat nejméně 200 způsobilých GT osob a 500 anotovaných face bodů. Je to minimální screeningová mez, ne záruka úzkého intervalu. Pokud chybí podpora, uvést nedostatek a rozšířit validaci, nelze tvrdit statisticky potvrzený přínos. U jednotlivých jemných bucketů pod 100 osob ponechat výsledek jako diagnostický. Bootstrap provádět po snímcích, u videa po sekvencích, nikoli po silně korelovaných bodech.

### 5.3 Gates podle typu změny

Všechny rozdíly AP/PCK níže jsou absolutní procentní body. Například 0,4589 na 0,4789 je +2,0 pp AP.

| Gate | Akceptační podmínka | Co dělat při neúspěchu |
|---|---|---|
| G0 metriky | Perfektní fixture AP75=1, nezávislá COCO9 reference v dohodnuté konvenci, dokumentované area/crowd/maxDet, report má checkpoint hash | Neprovádět accuracy rozhodnutí |
| G1 runtime bez změny vah | V PyTorch FP32 odpovídají score/xy/vis před a po pruning, bez ztráty osob. Seřazení je deterministické. ONNX/ncnn passes G2 | Odhalit dekódování, barvy, blob names nebo chybné řezy, nepřetrénovávat kvůli exportní chybě |
| G2 export | FP32 absolutní ΔAP<=0,2 pp, FP16 ztráta AP<=0,5 pp a small-face PCK<=1 pp, žádné NaN/Inf. Viz numerické tolerance v kapitole 9 | Artefakt označit failed, ne release |
| G3 hlavní quality kandidát | Cíl +2 pp AP9_project a +3 pp small-face Face-PCK@0.1 proti opravené baseline. Paired 95% bootstrap CI pro ΔAP a hlavní ΔPCK nezahrnuje nulu | Výsledek pouze kandidát/Pareto bod, žádné tvrzení o prokázaném zlepšení |
| G4 ochrana multi-person | Recall crowd/occluded/truncated neklesne >1 pp na dostatečně obsazených skupinách, duplicate FP/image nevzroste >10 % | Prověřit matching, pose-NMS a data, nepřijmout zlepšení jen na snadných osobách |
| G5 latence | Final p50 i p95 E2E processing <=1,10x optimalizovaná kompatibilní baseline B1 na stejném Pi profilu. Pokud B1 splňuje 40 ms, kandidát má tento rozpočet zachovat | Zamítnout pro u nebo ponechat jako jiný profil s přiznaným kompromisem |
| G6 provoz | Batch 1, žádný swap/OOM, žádný růst RSS >5 % po ustálení v 30min běhu, bez thermal/power throttling při validním benchmarku | Opravit runtime nebo test zopakovat se správným chlazením |
| G7 reprodukce | Finální top dva kandidáti nejméně ve 3 seedech, porovnání stejného budgetu a dat, plus nezávislý holdout pokud dostupný | Uvést nejistotu, nepovažovat jediný šťastný seed za důkaz |

B0 je původní best checkpoint přeměřený společnou opravenou pipeline. B1 je stejný checkpoint po přijatých runtime/export optimalizacích. Architektura se porovnává proti B1, aby dražší síť nemohla skrýt náklad za opravu pomalého dekodéru. Runtime B1 nepotřebuje splnit G3, jeho cílem je parity a zrychlení.

Pokud žádný model nesplní současně přesnost a latenci, nevolit vítěze manipulací definic. Vrátit Pareto tabulku a konkrétní limit. Bez přístupu k fyzickému Pi jsou G5/G6 pending. Desktop GPU benchmark je nenahrazuje.

## 6. Benchmark na Raspberry Pi 5 4 GB

Benchmark spustit až po budoucím povolení přístupu k zařízení. Neměnit frekvence nebo governor systému bez odpovídajícího oprávnění. Zaznamenat skutečný stav.

### 6.1 Reprodukovatelné prostředí

Uložit: board/RAM, 64bit OS a kernel, CPU frekvenci a governor, napájení, typ chlazení, teplotu, throttling flags před a po, ncnn release/commit a build options, compiler, Python/NumPy/OpenCV verze, počet vláken runtime i OpenCV, backend, vulkan flag, storage/arithmetic flags, hash modelu a konfigurace.

Základní matice je S={256,320,384}, threads={1,2,3,4}, CPU FP32. Nejdříve krátký sweep všech kombinací, poté tři opakované nezávislé běhy finalistů. U vítězného thread profilu přidat FP16 storage a FP16 storage+arithmetic. Vulkan zkoušet pouze jako další změřenou možnost s celými transfery a s explicitním hlášením fallbacku. Pi nemá v základním zadání externí NPU.

### 6.2 Přesné časové úseky

V monotónních timestamps měřit:

1. T_capture/decode a stáří snímku od camera timestamp.
2. T_preprocess včetně resize, barevného kontraktu, paddingu, normalizace a layout.
3. T_input_copy, T_network a T_output_extract/copy odděleně, pokud je API umí rozlišit.
4. T_filter/decode v FP32, T_inverse_affine a případný merger/NMS.
5. T_result: vytvoření finálního Pose9 výstupu.
6. T_draw/display uvést zvlášť, ale pro live demo také celkový camera-to-result a camera-to-display čas.

Primární E2E processing je hotový frame v paměti -> finální Pose9. Camera-to-result přidává capture a queue. Oba časy jsou nutné. Inference-only FPS, processing throughput a camera FPS nejsou zaměnitelné. Výchozí fronta drží maximálně jeden poslední snímek. Vynechané frames a stáří snímku logovat, neskrývat frontu vyšším průtokem.

Použít reprezentativní přednačtené snímky s 0/1/4/8+ osobami a video s malými obličeji, pohybem, překryvy a změnou osvětlení. Náhodný šum slouží pouze k per-layer screening testu. Na šumu nelze odhadovat náklady score filtrování a postprocessingu natrénovaného modelu.

Pro krátké měření warmup alespoň 100 iterací, poté alespoň 1000 snímků a nejméně 60 sekund. Pro finalistu 30min kontinuální běh, p50/p90/p95/p99, mean, FPS, peak RSS, počet zpracovaných/dropped frames, teplota a throttling. Tři opakování po stabilizaci, uvést rozptyl a přesný soubor vstupů. Initial model load a cold-start latenci měřit zvlášť. Nezahrnovat disk I/O do model-only času.

Na 4GB Pi je pracovní cíl runner RSS <=1 GiB a ponechání nejméně 1 GiB dostupné RAM při testu. Tvrdé provozní omezení je žádný swap/OOM a stabilní paměť. Pokud knihovny překročí pracovní cíl, výslovně reportovat skutečnou rezervu, ne tvrdit univerzální paměťový limit z počtu parametrů.

Na RTX používat CUDA events nebo synchronizaci kolem měřeného úseku. GPU čas pouze pomáhá plánovat trénink, nepřevádí se na Pi FPS.

## 7. Trénink, transfer a ablace

### 7.1 Společné podmínky na RTX 4070 Laptop 8 GB

- Použít torch.amp.autocast pro CUDA FP16 a GradScaler, master parametry/optimizer state ponechat FP32. AMP neznamená model.half() pro celý training.
- Device normalizovat přes torch.device a testovat device.type=='cuda'. To pokryje cuda i cuda:0.
- Assignment, IoU mocniny, výpočet OKS, flow.log_prob, log_sigma a citlivé redukce provést s vypnutým autocast a vstupy FP32. Není nutné nutit celou síť do FP32.
- Nejprve 100 reálných kroků při nejvyšším S=384, včetně maximální typické hustoty osob. Výchozí batch 16, při OOM batch 8. Effective batch 32 zajistit akumulací 2 nebo 4, bez tvrzení, že to dává totožné BatchNorm statistiky jako fyzický batch 32.
- Cíl peak allocated VRAM <=7,0 GiB. Současně reportovat reserved i celkovou GPU spotřebu. Laptop TGP, napájení, teplota a throttling mají vliv na čas.
- Při akumulaci dělit loss počtem mikrobatchů, správně obsloužit poslední neúplnou skupinu, unscale a clip udělat až před optimizer step. Clip zahrnuje všechny trénované model i criterion parametry.
- EMA a počitadlo optimizer steps aktualizovat jen při skutečné provedené aktualizaci, nikoli při GradScaler overflow skip. Oddělit micro_step a optimizer_step.
- Num_workers zvolit krátkým sweepem 0/2/4 podle Windows datového pipeline, nikoli dogmaticky 8. Bez dalšího důvodu nezapínat těžké histogramy každé vrstvy v každé epoše.
- Zaznamenat AMP scale, skip ratio, gradient norm po unscale, flow/sigma kvantily, coverage assignmentu, step time a peak VRAM.
- Požadovat finite forward/backward a nízký počet skip po warmup. Při opakovaných NaN nejprve přesný failing batch v izolaci, ne automatické vypnutí všech optimalizací.

Postup akumulace a unscale odpovídá primární PyTorch dokumentaci. Konkrétní verzi PyTorch/CUDA připnout, neaktualizovat během ablační řady. [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html).

### 7.2 Transfer vah

První training experiment inicializovat z existujícího XTX3-u best EMA, protože tvarově odpovídá backbone, necku i hlavám. Oddělit fine-tune od přesného resume:

- Fine-tune: převzít vahový stav, explicitně reset optimizer/scaler/schedule a uložit parent checkpoint SHA256.
- Resume: obnovit model, criterion, optimizer, scaler, RNG, ema.updates a pozici epochy. Neoznačovat fresh optimizer run za resume.
- Pokud se neimplementuje přesné obnovení uprostřed epochy, poskytovat spolehlivý epoch-boundary checkpoint a u záchranného checkpointu otevřeně uvést, že jde o restart části epochy. Nikdy potichu nepřeskočit zbývající batch.
- Změna kpt tower šířky: převzít pouze prokazatelně kompatibilní vrstvy. Novou větev inicializovat a logovat loaded/missing/unexpected klíče i podíl parametrů, ne skrývat neúplné načtení.
- Změna na P2-to-P3 reziduální fúzi dovoluje nulovou poslední projekci a zachování baseline na začátku.

U YOLO26 reference je nejlevnější první měření ponechat pretrained 17bodovou síť a vybírat devítibodový subset až ve výstupu. To dává kvalitu reference, ale nesmí se nazývat devítibodově zrychlenou sítí. Poté, projde-li předběžným Pi gate, nastavit skutečné K=9 a mapovat finální projekce podle zdokumentovaného layoutu vybrané verze. Nelze předpokládat, že má YOLO26 stejný K*6 layout jako XTX3.

Při převzetí pretrained RGB modelu musí manifest určit RGB preprocessing. XTX3 dnes používá BGR. Buď změnit runtime a training u daného profilu společně, nebo explicitně prohodit vstupní kanály první konvoluce a ověřit numerickou paritu. Nezavádět náhodnou BGR/RGB změnu spolu s jinou ablací.

Vnější modely mají vlastní licenční podmínky. U přebíraného kódu, checkpointu a distilačních dat uložit přesný původ a podmínky distribuce. Pro rozhodnutí o produktu nepředpokládat zaměnitelnost AGPL a Apache modelů.

### 7.3 Assignment a loss před kapacitním růstem

Nejdříve zachytit diagnostiku na prvních 100 tréninkových batchech a ve stejné malé množině každou epochu:

- počet kandidátů a konečných positives/GT v o2m/o2o,
- podíl eligible GT bez pozitiv po conflict resolution, ne pouze před ní,
- breakdown min-side person boxu, face_spread_net, crowd a stride,
- rozdělení target_scores, počet hodnot pod eps a rozdíl FP32 proti AMP,
- podíl GT s jedním bodem, absent boky a uříznutým full-person boxem.

Vyřadit globální prior směrem ke středu celého obrazu jako výchozí novou ablaci. Jeho ospravedlnění „kamera mívá osobu ve středu“ není garance pro multi-person okraje. Pokud pomáhá na konkrétní doméně, musí to ukázat rovnoměrné center/edge vyhodnocení.

Pro one-to-one opravit candidate selection tak, aby validity nezávisela jen na tom, zda rané score*IoU^6 překročilo eps. Využít malý geometricky přípustný candidate pool, výchozí topk7, vyřešit konflikty mezi GT a teprve pak vybrat nejvýše jedno pozitivum na GT. Při neobsazeném GT zkusit další volný geometrický kandidát v deterministickém pořadí. Jednomu anchoru nikdy nepřiřadit dvě osoby. Reálné kolize logovat. Geometricky vynucené positives musí mít také nenulový použitelný klasifikační target, jinak pouhý fg_mask neopraví chybějící signál. Dočasný warmup floor soft targetu explicitně konfigurovat a logovat, jeho vliv neplést s vyšší kalibrací score. Pokud lokální fallback zůstane pro crowd nedostatečný, teprve potom zvážit jednorázový bipartitní matcher nad omezenou množinou kandidátů v trainingu.

Současná STAL implementace pouze před konflikty přidává positives malým GT a top1 fallback nepoužívá. Netvrdit „garantuje 4 positives každému GT“ bez post-conflict kontroly. Referenční Ultralytics TAL má samostatný candidate topk a topk2, rozlišení je podstatné. [Primární TAL reference](https://docs.ultralytics.com/reference/utils/tal).

Pro nové loss připravit tři úzce definované varianty se stejnou datovou pipeline:

- L0: dosavadní RLE, ponechaný jako kontrola pro navazující fine-tune, s FP32 numerikou.
- L1: čistý maskovaný OKS plus malý normalizovaný SmoothL1 pro ranou fázi. OKS denominator přesně 8*sigma_k^2*A jako evaluator. Původní OKSL1Loss používá 2*sigma_k^2*A a nenormalizované pixelové L1, takže není ekvivalent.
- L2: OKS + ověřený referenční RLE. RLE váhu začít malou a zvolit podle gradientové škály, ne slepě opsat loss.kpt=12 z jiného reduction.

Implementovat zvolenou RLE konvenci matematicky explicitně: error=(gt-mu)/sigma, joint flow NLL=sum(log sigma)-log p_flow(error), reziduální term sum(log(2*sigma)+abs(error)). Vysvětlit, zda log p_flow je společná 2D hustota nebo per-coordinate člen a kolikrát se při reduction přičítá. Proti konkrétní reference ověřit value i gradients, včetně identity flow. Místní kód má jediný sum(log sigma) a flow s Laplace base. Oficiální původní a MMPose implementace pracují s reziduálním Q termem obsahujícím další sigma, ale liší se normalizací a rozměrovou redukcí. Neopravovat to jediným odhadnutým koeficientem bez fixture. [Původní RLE criterion](https://github.com/Jeff-sjtu/res-loglikelihood-regression/blob/master/rlepose/models/criterion.py), [MMPose RLE](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py), [Ultralytics RLE a PoseLoss26](https://docs.ultralytics.com/reference/utils/loss).

Pro fair ablace držet stejné souřadnicové jednotky. Změnu sigma floor a isotropic vs anisotropic sigma testovat až samostatně. Dosavadní floor +1 pixel nezakazuje subpixelovou predikci mu, pouze omezuje modelovanou nejistotu. Nesmí být označen za důkaz kvantizace souřadnic.

Loss normalizovat po GT osobách a po počtu jejích anotovaných bodů, pak přes osoby. Současné globální průměrování per-anchor/per-point může zvýhodnit vícekrát přiřazenou osobu. Změnu reduction oddělit od typu loss a zaznamenat gradient magnitudes. Pro face prior začít váhou 1,0, případnou 1,5 pro nos/oči/uši zkoušet zvlášť, nepřekročit crowd/torso ochranný gate.

Současný místní MuSGD není stejný algoritmus jako oficiální MuSGD, který míchá SGD a Muon aktualizace na multidim vahách. Pro první opravenou fine-tune baseline použít dobře definované SGD s momentum, poté pouze jednu kontrolovanou optimizer ablaci, pokud convergence odůvodní náklad. Netvrdit, že název lokální třídy znamená reprodukci YOLO26. [Primární MuSGD reference](https://docs.ultralytics.com/reference/optim/muon).

### 7.4 Curriculum a data

Na existujících kompatibilních vahách je výchozí plán 60 epoch fine-tune. Je to startovní rozpočet k validaci, ne garance výsledku. Nejprve smoke a krátký screening, ne šest plných běhů současně.

| Fáze | Epochy | Postup |
|---|---:|---|
| Sanity | mimo hlavní běh | Overfit 16 až 32 různých snímků bez augmentace, sledovat body, positives a prázdné obrázky |
| Adaptace | 0 až 2 | Warmup LR, pevné 320, mírná augmentace, žádný mosaic, o2m i o2o nenulové |
| Hlavní | 3 až 44 | Multiscale 256/320/384 s poměrem 0,2/0,5/0,3, flip 0,5, realistické světlo/blur, řízený mosaic nanejvýš 0,2 |
| Dolaďování | 45 až 59 | Mosaic 0, omezený translate/scale, poměr rozlišení 0,1/0,6/0,3, větší důraz na inference o2o |
| Final | po výběru best | Přeměřit celý val ve všech třech velikostech, zmrazit deployment size/práh a teprve potom holdout |

Při scale změnách upřednostnit jeden geometricky přesný resize ze zdroje do finálního rozměru. Dnes se obrázek renderuje do 384 a následně v traineru resizuje na menší S, což může jinak rozmazat malé face body než přímý deployment resize. Nejdříve parity fixture a ablace přímého renderu, ne skryté přepnutí celé datové pipeline. Velikost batch předávat workerům explicitně jako součást sampled indexu nebo batch zadání. Nespoléhat na změnu dataset.network_size pouze v hlavním procesu při aktivních workerech.

Dosavadní mosaic neudržuje poměr stran zdrojů, protože každý zdroj deformuje do čtverce. Testovat letterbox/affine variantu s konzistentními labely a menší pravděpodobnost mosaic. Přidávat jen augmentace odpovídající kameře: mírný motion blur, noise, komprese, expozice a částečné zakrytí. Silné transformace, které mažou několika pixelové oči, nejsou automaticky užitečné.

Sampling může mírně zvýšit podíl small-face a crowd snímků, nejvýše 2x na startu. V každé epoše zachovat reprezentativní negativní snímky a běžné osoby. Nevybírat jen osoby s oběma boky. Osoby s jedním anotovaným bodem držet jako explicitní skupinu a nevyvozovat z nich chybějící GT souřadnice.

Výběr targetu při cropu odvodit od přítomnosti relevantních anotovaných bodů a viditelné horní části, nikoli výhradně od centra full-person boxu. Pro pomocný box zachovat korektní clipped training target s minimálním rozměrem. Auditovat crowd annotation a ignored osoby, aby se neanotované osoby nezačaly používat jako snadné negativy.

Při plném scratch přepisu neporovnávat 50 scratch epoch proti rozsáhle pretrained modelu jako srovnání architektur. Použít odpovídající delší schedule, rozumný výchozí rozsah 180 až 300 epoch, nebo explicitně hodnotit praktický training budget místo architekturní čistoty.

### 7.5 Distilace bez inference nákladu

Teacher je pouze training/offline prostředek. Preferovat dostupný kvalitní pretrained model, například YOLO26s/m-pose nebo RTMPose-m s GT crops. RF-DETR preview je alternativní teacher, pokud přínos ospravedlní přípravu. Nezačínat vlastním tréninkem modelu l.

Na 8 GB použít frozen teacher eval/no_grad AMP v mikrobatchech nebo offline cache. Pokud teacher a student společně neprojdou paměťovým testem při 384, nevyžadovat jejich současné umístění na GPU.

První jednoduchá distilace pracuje s devíti teacher souřadnicemi a kvalitou přiřazení k GT osobám. Teacher výstup mapovat zpět do původního obrazu, uložit parent image hash, teacher hash, preprocessing a odpovídající GT id. Při crop/flip/scale student vstupu transformovat cache stejně jako labely, včetně levé/pravé sémantiky. U mosaic přemapovat každou zdrojovou osobu a maskovat body mimo crop, nebo pro distilační část mosaic vypnout.

Ground truth má přednost. Nevytvářet automaticky „pravdivé“ labely u v=0. Teacher chyby u malých obličejů filtrovat důvěrou a shodou s anotovanou geometrií. Výchozí distilační váhu postupně zvýšit během prvních 5 epoch, pilotně tak, aby její gradientová norma nebyla větší než hlavní GT loss. Pseudo-labely dalších osob mimo anotaci jsou samostatný datový experiment, ne skrytá součást baseline.

Feature distillation mezi nesourodými backbony nezařazovat jako první krok. Jedna malá training-only projekce dává smysl pouze tehdy, když souřadnicová distilace prokazatelně nestačí a export ji odstraní. Žádný teacher nebo flow nesmí zůstat v deployment grafu.

## 8. Konkrétní realizační úkoly pro AI agenta

Níže uvedené nové cesty jsou návrhy pro budoucí implementaci. Při této práci nebyly vytvořeny ani čteny. Portovat jen potřebné části XTX3, nekopírovat runs, .venv, exporty a celý projekt jako nerozlišený celek. Každý úkol má vlastní důkaz dokončení a navazuje jen na potřebné předchůdce.

### T0: Založit reprodukovatelnou pracovní základnu

Závislost: nové oprávnění k implementaci, datasetům a případně Pi.

Cesty: C:/git/freesentry/xtx4/pyproject.toml, C:/git/freesentry/xtx4/configs/u_base.json, C:/git/freesentry/xtx4/xtx4/utils/keypoints.py, C:/git/freesentry/xtx4/scripts/experiment_manifest.py.

Převzít devítibodová metadata, relevantní model a geometrii. Zjistit stav cílových souborů před editací a pracovat v aktivní větvi. Připnout závislosti přes uv lock nebo uv pip compile ve zvoleném jednoduchém existujícím stylu. Oddělit training/export dependencies od lehkého Pi runtime. Manifest obsahuje model/config/source/checkpoint/data hashes, seed, verze, hardware, preprocessing, output schema a výběr head. Nepotřebuje novou databázi ani registry server.

Výstup: u_base dokáže přísně načíst XTX3-u checkpoint a identifikovat vybrané EMA váhy. Test: stejný model před a po portu má na deterministickém vstupu stejné původní raw výstupy, při budoucím povolení také na reálných snímcích. Gate: žádné nevysvětlené missing/unexpected váhy, zachované body i pořadí.

### T1: Opravit a zmrazit evaluaci

Závislost: T0.

Cesty: C:/git/freesentry/xtx4/xtx4/engine/metrics.py, C:/git/freesentry/xtx4/xtx4/data/annotations.py, C:/git/freesentry/xtx4/scripts/evaluate.py, C:/git/freesentry/xtx4/tests/test_metrics.py.

Opravit AP75, přidat checkpoint hash a split manifest, area/crowd konvenci, úplnou finální evaluaci a face metrics z kapitoly5. Ponechat legacy evaluaci pro srovnání. Vyhodnotit existující best EMA při všech třech S bez retrénování. Report musí rozlišovat 200-image historický subset, plný legacy dev a případný skutečný holdout.

Výstup: baseline_accuracy.json plus per-image predictions a per-image metrics v novém run directory. Test: perfektní případ, prázdný snímek, duplicita, dvě překrývající se osoby, v=0 a crowd. Gate: G0. Bez splnění se žádná ablace neoznačí za zlepšení.

### T2: Sjednotit skutečný runtime a změřit B0

Závislost: T0, accuracy část na T1.

Cesty: C:/git/freesentry/xtx4/xtx4/runtime/preprocess.py, C:/git/freesentry/xtx4/xtx4/runtime/decode.py, C:/git/freesentry/xtx4/xtx4/runtime/ncnn_runner.py, C:/git/freesentry/xtx4/scripts/benchmark.py.

Vycházet z [existujícího live_test NumPy dekodéru](C:/git/freesentry/xtx3/live_test/livetest/decode.py) a [runneru](C:/git/freesentry/xtx3/live_test/livetest/runner.py), které již mají score filtr před decode, cached anchors, stabilní softmax, vlákna, kontrolu load/extract a mapování blob names. Neportovat pomalejší torch dekodér jako runtime základ.

Nový export manifest má být autoritou pro input/output names, layout, strides a precision. Staré pnnx artefakty mohou mít out0/out1/out2, zatímco export runner očekává box/cls/kpt. Pro legacy import explicitně zaznamenat převod. Nové artefakty nesmějí tiše hádat jména nebo stridy z počtu hodnot.

Nastavit runtime flags před load. Kontrolovat návratové kódy input, load a extract, rozměry a dtype. NumPy návrat z ncnn kopírovat jen tam, kde je to nutné kvůli životnosti bufferu. Runtime package nesmí při importu natahovat torch.

Výstup: jednotné Pose9 výsledky a timing schema pro PyTorch referenci, ONNX a ncnn. Test: reálná parity plus ověření importu runtime bez torch. Gate: G1/G2 pro kompatibilní plný export. Změřit B0 na Pi, nebo označit hardware gate pending a pokračovat nezávislými kroky.

### T3: Zmenšit deployment bez retrénování

Závislost: T1,T2.

Cesty: C:/git/freesentry/xtx4/xtx4/models/head.py, C:/git/freesentry/xtx4/xtx4/export/deploy.py, C:/git/freesentry/xtx4/xtx4/export/manifest.py, C:/git/freesentry/xtx4/tests/test_export_parity.py.

Vytvořit export kopii modelu, fuse a oříznout box tower i sigma projekce podle kapitoly4. Zachovat o2m/box/flow v training modelu. Dokumentovat novou dvoutensorovou raw ABI, nepromíchávat ji se starým K*6 formátem. Zkontrolovat skutečné grafy po ONNX simplification i ncnn conversion.

Výstup: B1 při 256/320/384, FP32 ONNX a ncnn, graf/param/MAC/output-bytes tabulka. Test: score/xy/vis parity před/po pruning, reálné threshold a tie cases. Gate: G1,G2 a žádné zhoršení runtime. Pokud úspora není významná, ponechat jednoduchou variantu a uvést měření, nepřidávat další kompresní formát.

### T4: Zajistit numericky spolehlivý training

Závislost: T0,T1. T2/T3 nemusí čekání na Pi blokovat.

Cesty: C:/git/freesentry/xtx4/xtx4/engine/trainer.py, C:/git/freesentry/xtx4/xtx4/engine/checkpoint.py, C:/git/freesentry/xtx4/tests/test_training_state.py.

Implementovat omezený soubor oprav z kapitoly7: device.type, AMP boundaries, all-parameter clip, gradient accumulation a EMA jen při úspěšném optimizer step, seed všech RNG, checkpoint state. U finální evaluace načíst stejné best EMA jako export. Trénovat nic déle než smoke, dokud se testy a VRAM probing nevyjasní.

Výstup: 100-step AMP smoke při 384, doložená peak VRAM a malý deterministic resume fixture. Gate: finite loss/grad, žádné systematic assignment underflow a schopnost reprodukovat navazující kroky v deklarovaném režimu resume. Nekonstruovat obecný trainer framework.

### T5: Ověřit a zlepšit assignment/loss

Závislost: T1,T4.

Cesty: C:/git/freesentry/xtx4/xtx4/losses/assignment.py, C:/git/freesentry/xtx4/xtx4/losses/pose_loss.py, C:/git/freesentry/xtx4/xtx4/losses/rle.py, C:/git/freesentry/xtx4/tests/test_assignment.py, C:/git/freesentry/xtx4/tests/test_losses.py.

Nejprve instrumentovat současné chování. Poté opravit konkrétně prokázaný zero-positive/kolizní problém a vypnutí global center prior posoudit samostatně. Přidat verified OKS a verified RLE konvenci, testovat L0/L1/L2 po jedné změně. Názvy variant musí identifikovat formula/reduction version.

Výstup: assignment coverage report, loss reference fixtures, krátké kontrolované runy. Test: tiny box mezi cell centry, okraj, dva blízké GT bojující o anchor, nulové rané scores, zero-GT, jeden označený bod a mixed precision. Gate: každé validní nekolizní GT ve fixture dostane pozitivum, žádný anchor dvě GT, o2o nejvýše jeden anchor/GT. Na reálném screening subsetu cílit >=99,5 % eligible GT coverage a každou ztrátu vysvětlit. Nová asymptotika nesmí zbytečně zpomalit training.

### T6: Opravit cílové datové chyby a curriculum

Závislost: T1,T4. Stejnou data variantu použít pro párové loss experimenty.

Cesty: C:/git/freesentry/xtx4/xtx4/data/dataset.py, C:/git/freesentry/xtx4/xtx4/data/augment.py, C:/git/freesentry/xtx4/configs/u_finetune.json, C:/git/freesentry/xtx4/tests/test_targets.py.

Opravit relevant-point crop inclusion, zachovat v=0/1/2, definovat format/schema anotací místo heuristiky „hodnota>1,5 znamená pixels“. Legacy auto-detekci ponechat jen v explicitním importu s reportem. Auditovat osoby ze více JSON bez zdvojení, souřadnice, rozměry a chybné anotace. Přidat přímý finální resize a realistickou augmentaci jako oddělené ablace.

Výstup: datový audit, ukázky transformací s body a počty, verze curriculum. Test: crop zachovávající hlavu při bbox centru mimo obraz, chybějící hips, flip levá/pravá, mosaic a zpětná affine. Gate: vizuálně i numericky konzistentní labels, žádné proniknutí validation do train.

### T7: Provést omezenou ablační řadu a distilaci

Závislost: T1 až T6.

Cesty: C:/git/freesentry/xtx4/scripts/train.py, C:/git/freesentry/xtx4/scripts/cache_teacher.py, C:/git/freesentry/xtx4/configs/ablations/, C:/git/freesentry/xtx4/runs/<run_id>/.

Provést níže uvedené experimenty sekvenčně podle priority. Run directory je vždy nové. Existující výsledky se nepřepisují. Smoke rozpočet je stejný počet optimizer steps, screening 5 až10 epoch slouží k odhalení nestability a velmi špatných kandidátů. Malý rozdíl v rané AP není spolehlivý důvod zavrhnout pomaleji konvergující metodu. Úspěšné dvě varianty dostanou společný 60epoch fine-tune rozpočet.

Distilaci přidat až proti stabilní nejlepší variantě bez teachera. Offline cache musí projít geometrickou paritou a VRAM testem. Výstup: ablation.csv s nezměněnými raw reports. Gate: nejméně kandidát pro G3/G4 a zlepšení nemůže být vysvětleno rozdílným rozlišením, test leakage nebo jiným prahem.

### T8: Pouze podle potřeby změnit architekturu

Závislost: T3,T7 a dostupný Pi benchmark.

Cesty: C:/git/freesentry/xtx4/xtx4/models/head.py, C:/git/freesentry/xtx4/xtx4/models/neck.py, C:/git/freesentry/xtx4/configs/ablations/.

Nejdříve hardware screening kpt šířky a P2-to-P3 varianty na skutečném ncnn grafu, pak trénovat nejvýše dvě smysluplné architektury. Plná P2 hlava, skutečně box-free training a transformer mají stop gate před dlouhým tréninkem. Vnější YOLO26n reference sem vstupuje s doloženou přesností a Pi exportem, ne s papírovým odhadem.

Výstup: Pareto tabulka kvality/latence/RAM a odůvodnění zvoleného profilu. Gate: G3,G4,G5. Pokud B1+training-only změny splňují cíl, T8 dokončit rozhodnutím „není potřeba“, bez povinného redesignu.

### T9: Potvrdit finalistu a vydat artefakty

Závislost: T7 a podle potřeby T8.

Cesty: C:/git/freesentry/xtx4/scripts/export.py, C:/git/freesentry/xtx4/release/<model_id>/, C:/git/freesentry/xtx4/report.md.

Top kandidáty přeměřit ve 3 seedech, vyhodnotit stejný best checkpoint plně při všech 3 rozměrech. Vybrat model/size/práh na val. Teprve potom jednou použít nedotčený holdout. Exportovat a otestovat final precision režimy. Pro release provést 30min Pi běh a replay stejného videa přes všechny distribuované backendy.

Výstup: checkpoint, ONNX, ncnn param/bin, manifest, přesný decode contract, benchmark JSON/CSV, accuracy report a stručná model card. Neschválené ablace distribuovat pouze jako experimentální. Gate: G0 až G7 podle typu změny, bez Pi nevydávat tvrzení o real-time. Otevřené problémy vyjmenovat. Nakonec jeden souvislý diff review a cílené opravy, uklidit pouze vlastní testové procesy.

### 8.1 Omezená experimentální matice

| ID | Jediná měněná věc | Referenční srovnání | Rozhodnutí |
|---|---|---|---|
| A0 | Pouze opravená evaluace | Historická best EMA | Nová baseline bez trainingu |
| A1 | NumPy runtime/threads | A0 stejné váhy | Přijmout při parity a prokazatelné úspoře |
| A2 | Box+sigma deployment pruning | A1 stejné váhy | B1, parity gate |
| A3 | FP16 storage/arithmetic | B1 FP32, stejný graf | Vybrat přesnost/latence, žádný retraining |
| A4 | FP32 assignment + konkrétní coverage oprava | Párová fine-tune control se stejným optimizerem | Coverage, AP/crowd |
| A5 | Center prior 0 proti 0,1 | A4 | Zvlášť center a edge |
| A6 | L1 proti L0, pak L2 proti vítězi | Stejné reduction/schedule/data | AP75, face PCK a sigma diagnostika |
| A7 | Data inclusion/resize/curriculum | Vítěz A6 | Jednotlivé opravy oddělit, ne souhrn magic recipe |
| A8 | Teacher coordinate distillation | Vítěz A7 | Přínos bez inference nákladu |
| A9 | Pretrained YOLO26n-pose9 profil | B1 a nejlepší custom model | Přijmout jen po stejných eval/Pi gates |
| A10 | Kpt cap 48/64/80 nebo P2 feature fusion | Vítěz custom bez této změny | Nejdříve hardwarový screening |
| A11 | P2 head nebo true box-free | Jen pokud selžou jednodušší možnosti | Samostatné ADR a podmíněná realizace |

Nevytvářet kartézský součin loss x optimizer x input x backbone x augmentation. Rozlišení jsou při inference nejprve sweep stejného checkpointu. Jako jeden krátký kontrolní experiment ověřit SGD versus opravený a připnutý MuSGD pouze tehdy, když je convergence problém.

### 8.2 Podmíněná cesta skutečně bez boxů při trainingu

Tato cesta není nutná pro odstranění boxů z deploymentu. Přidává riziko a má smysl až při doloženém škodlivém vlivu box supervision nebo zásadní úspoře, kterou T3 neposkytne.

Definovat root deterministicky ze značených upper-body bodů: midpoint obou ramen, jinak medián dostupných torso bodů, při absenci torza medián dostupných face bodů. Nikdy zahrnovat v=0 souřadnice. Změnu reference při missing joints chápat jako nový target design, nikoli stejnou úlohu.

Použít root-centered candidate region a pose-aware cost složený z person score, maskovaného normalizovaného keypoint error a root distance. Pro training-only normalizaci a evaluation lze stále použít původní GT area, i když síť box nepředpovídá. Jedinečnost přiřazení musí řešit překryvy a identické root cell. Osoba s jediným označeným bodem má malou jistotu scale a vyžaduje explicitně definovanou fallback normalizaci, nikoli domyšlenou torso velikost.

Důležité tests: absent hips, jedna polovina ramen, jediný nos, dvě osoby ve stejné cell, úplný occlusion, osoba přes okraj. Ověřit crowded recall, duplicate FP a výběr confidence. Neúspěch této ablace neblokuje vydání jednoduššího training-only box řešení.

## 9. Exportní kontrakt a precision

### 9.1 Spolehlivá exportní cesta

1. Načíst přesně zvolený best EMA checkpoint na CPU FP32, uložit hash. Přepnout eval, vytvořit samostatnou deployment kopii a fuse Conv+BN/Rep bloky.
2. Odstranit o2m, box tower a nevyužité sigma projekce pro zvolený slim profil. Flow a teacher se nikdy neexportují.
3. ONNX exportovat s batch 1 a statickými H/W pro každé S. Opset17 je rozumný výchozí bod kompatibilní se současnou cestou, ale připnutý converter musí projít skutečným testem. Neměnit opset jen kvůli nejnovějšímu číslu.
4. Host decode nechat mimo graf. Nepřidávat do základní ncnn cesty dynamické TopK, NMS, GridSample, ROIAlign, deformable ops, custom operators ani datově závislé Python smyčky.
5. Pokud reshape/transpose/split/concat nebo Resize překážejí converteru, zjednodušit wrapper a přesně ověřit flatten order. Neobcházet problém tichým přeuspořádáním dat.
6. onnx.checker a shape checks jsou povinné. Simplification je volitelná optimalizace, po ní se znovu měří parity. Její selhání může ponechat validní nezjednodušený ONNX, ale tento stav musí být v manifestu.
7. Provést pnnx konverzi s explicitní precision volbou. PNNX README uvádí default fp16=1 pro uložení ncnn vah/ONNX. Referenční FP32 varianta proto musí výslovně použít fp16=0. [Primární PNNX README](https://github.com/Tencent/ncnn/tree/master/tools/pnnx).
8. Pokud připnutý pnnx neumí přímý ONNX vstup pro zvolený graf, použít podporovanou TorchScript -> pnnx cestu ze stejného deployment wrapperu. ONNX zůstane samostatně distribuovanou a ověřenou větví. Není nutné násilně trvat na ONNX jako jediném meziformátu.
9. Nový manifest ukládá skutečná input/output jména vyrobeného grafu. ncnn runner validuje tvar a layout při startupu. Nevymýšlí je podle názvu souboru.
10. Chybějící onnxruntime/ncnn/converter nebo nenulový return code znamená příslušný export gate failed/pending. Původní best-effort warning nestačí k tvrzení „ONNX a ncnn ověřeny“.

### 9.2 Precision profily

| Profil | Váhy/graf | ncnn runtime | Host decode | Účel |
|---|---|---|---|---|
| ref_fp32 | FP32 ONNX a ncnn, pnnx fp16=0 | fp16_storage=false, fp16_arithmetic=false, bf16=false, int8=false | FP32 | Numerická reference |
| fp16_storage | Explicitně označené FP16 uložení vah | fp16_storage=true, fp16_arithmetic=false, další flags připnuté | FP32 | Paměť/bandwidth ablace |
| fp16_compute | FP16 profil podle ověřeného buildu | fp16_storage=true, fp16_arithmetic=true, podpora CPU ověřena | FP32 | Hlavní kandidát zrychlení |
| int8_experiment | Kalibrovaný deployment profil | Pouze podporované int8 kernels | FP32 | Volitelné až po dokončení FP32/FP16 |

Také use_fp16_packed, use_packing_layout, lightmode a případné BF16 fallback volby zaznamenat. Pro smysluplné srovnání měnit jednu precision vlastnost najednou a ověřit skutečnou aktivní cestu. Žádné nastavení dtype se nesmí vydávat za důkaz, že všechny vrstvy počítají stejnou přesností.

INT8 nezařazovat jako předpoklad. Pokud bude třeba, kalibraci tvoří několik set až přibližně 1000 reprezentativních train/kalibračních snímků se small-face, occlusion a negatives. Holdout se nepoužívá. QAT až tehdy, když PTQ naruší localization a zisk latence je dostatečný. Nechat finální coordinate/presence projekce ve FP16/FP32, pokud smíšená cesta prokazatelně pomůže a runtime ji podporuje. Žádná garance rychlosti jen z bitové šířky.

### 9.3 Numerická a funkční parity

Testovat všechny tři S, reprezentativní reálné snímky, tmavý/světlý/konstantní vstup, extrémní logity a prázdné výsledky. Random input je doplňkový sanity test, nikoli celá validace.

- Před exportem, stejné FP32 váhy a pruning: očekávat raw person/xy/vis rozdíl blízký strojové numerice. Výchozí allclose atol 1e-6, rtol 1e-5 pro samotný řez projekcí, fuse porovnat zvlášť.
- PyTorch fused -> ONNX/ncnn FP32: orientační raw atol 1e-3, zároveň max rozdíl vybraných xy<=0,05 network pixelu a score/conf<=1e-4. Pokud oprávněné rozdíly kernelů tuto mez překročí, vyhodnotit skutečnou chybu a uvést konkrétní toleranci, nikdy ji potichu nezvednout.
- FP16: sledovat p50/p95/p99 xy chyby, výchozí p95<=0,25 network pixelu a p99<=1 pixel. Všechny outliery>2 pixely na vybraných osobách prohlédnout. Rozhodující je také accuracy G2, ne pouze shoda tisíců nízkoskórových anchorů.
- Porovnávat raw výstup podle stejného anchor id před filtrem, aby matching nezamaskoval chybu pořadí. Finální poses potom porovnat deterministicky. Shodné score, prahová hranice a maxDet truncation mají vlastní fixtures.
- Při low precision mohou nepatrné rozdíly překlopit kandidáta těsně u threshold. Tyto případy reportovat, neskrývat je volnější matching tolerancí. Plná AP a provozní recall určí praktický dopad.
- Fúzi a pruned export ověřit i po opakovaném volání. Původní tréninkový model nesmí být exportem zmutovaný.

Manifest pro release obsahuje schema_version, model_id, checkpoint_sha256, config_sha256, keypoint names/COCO mapping/flip/sigmas, input color/range/padding/affine convention, size/strides/anchor order, output names/layout, decode version, maxDet/threshold, precision, converter/runtime versions a původ vah. Malý JSON stačí.

## 10. Smysluplné testy a reprodukovatelnost

Testy mají chránit reálné chování, nikoli jen duplikovat implementační výrazy. Nepřidávat testy, které pouze kontrolují počet vrstev nebo existenci config klíče.

| Test | Nezávislý očekávaný výsledek | Proč je důležitý |
|---|---|---|
| Perfektní AP fixture | Všechny prahy včetně AP75 jsou 1 | Zachytí nynější float-key problém |
| Jeden bod s analytickým OKS | Například sigma 0,1, area 100, vzdálenost 1 dává exp(-1/8) | Odhalí čtyřnásobný denominator rozdíl |
| Duplicate prediction | Jedno GT se matchuje jednou, druhá detekce je FP | Ověřuje skutečnou multi-person unikátnost |
| Overlapping people | Dva GT dostanou různé predikce bez identity swapu v přiřazení | Chrání crowd kvalitu |
| Tiny assignment init | Slabé scores a box mezi mřížkami stále mají platné geometrické pozitivum | Chrání trénink nejmenších targetů |
| Post-conflict coverage | Dvě GT soutěžící o první anchor nezmizí kvůli prostému top1 | Chrání výstup o2o |
| Missing/cropped joints | Hlavu uvnitř cropu zachovat i s bbox centrem venku, absent body nevstupují do loss | Chrání hlavní upper-body použití |
| Flip + affine | Ručně určené body po flipu a zpětné transformaci mají správnou levou/pravou identitu | Chrání geometrii |
| Resize půl pixelu | Pevně zvolený syntetický obraz a body odpovídají deklarované OpenCV/síťové konvenci | Oči nesmějí systematicky ujet o pixel |
| RLE value/gradient | Identity flow a zmrazená referenční implementace dají stejnou konvenci | Chrání pravděpodobnostní loss |
| Zero-GT batch | Finite loss, žádné NaN, background cls se trénuje | Chrání běžný prázdný frame |
| AMP accumulation | Na malém deterministickém modelu odpovídá referenční effective batch v rámci tolerance, bez BN rozdílu v test fixture | Chrání škálování loss a clipping |
| Resume | Další kroky v malém deterministic fixture odpovídají nepřerušené cestě | Chrání skutečně používané dlouhé runy |
| Raw/decoded export parity | Všechny S a backendy zachovají vybrané body/skóre/visibility | Chrání distribuci |
| Runtime bez torch | Samostatný proces importuje a spustí ncnn runner bez torch dependency | Chrání paměť a startup na Pi |
| Load/extract failure | Chybné param/bin/blob jméno vytvoří srozumitelnou chybu, ne prázdné „platné“ poses | Chrání nasazení |

Pro každý experiment uložit nové run_id, parent run, jedinou změnu, plný config, init weights hash, dataset/split manifest hash, seed, effective batch, steps, wall time, peak VRAM, optimizer/loss verzi, zvolené EMA weights a export hashes. Nepřepisovat best.pt mezi nepříbuznými variantami. Resume musí číst jen explicitně zvolený run.

Seed ovládá Python random, NumPy, torch CPU/CUDA, DataLoader generátor a workery. Deterministický režim použít pro malé testy a kontrolní běhy. U výkonnostních běhů uvést případný nedeterminismus a skutečné backend volby. Tři finální seedy uvádět samostatně i průměr/rozptyl, image bootstrap není náhradou variability mezi training seedy.

Porovnávat stejné tréninkové budgety a stejné checkpoint-selection pravidlo. Výběr best provést na primární val metrice se secondary face a latency constraints, bez opakovaného ladění na holdoutu. Pokud se primary výběr změní uprostřed řady, označit novou experimentální sérii.

## 11. Důkazy a otevřené otázky

### Co je již doloženo

- XTX3 skutečně implementuje 9 bodů, duální o2m/o2o hlavu, whole-frame u a export při 256/320/384.
- Reportovaný běh u dosáhl historického AP 0,45894 na omezeném eval subsetu, nikoli prokazatelně na celé test sadě.
- AP75 má konkrétní chybnou cestu přes float-key lookup.
- Dekodér sigma nevyužívá a box/score/kpt towers jsou oddělené.
- Existuje vhodnější NumPy ncnn runtime v live_test, který lze převzít.
- Pi výsledky XTX3-u v reportu nejsou vyplněné.
- YOLO26 má vydané primární zdroje a modely, YOLO27 byl při ověření stále preview bez veřejného release.
- ARM/ncnn poskytují relevantní FP16 CPU mechanismy, samotný FP16 soubor však neurčuje aritmetiku.

### Co zůstává hypotézou nebo čeká na hardware/data

- Kolik přesnosti přidá opravený assignment, verified RLE+OKS, přímý resize a distilace.
- Jaký je skutečný podíl zero-positive GT, duplicity při 0,25 a chyba na drobných face bodech v síťových pixelech.
- Zda removal box tower přinese na Pi měřitelně přibližně 5 % nebo odlišnou úsporu.
- Zda je nejlepší320 nebo384, kolik vláken a jaká precision na konkrétním Pi.
- Zda pretrained YOLO26n porazí vlastní u v devítibodovém accuracy/latency kompromisu.
- Zda existují neporušená holdout data z cílových kamer a jaký počet osob má být běžným provozním zatížením.
- Zda skutečně box-free training přinese něco navíc oproti pomocnému boxu odstraněnému při exportu.

Chybějící Pi neblokuje přípravu kódu, exportní parity ani tréninkové zkoušky po jejich autorizaci. Blokuje závěr o real-time nasazení. Chybějící nezávislý holdout neblokuje development ablace, ale blokuje tvrzení o ověřené generalizaci.

## 12. Předávací checklist

Realizace je úspěšně uzavřena, když existuje konkrétní zvolený model se zdokumentovaným kompromisem a následujícími důkazy:

- [ ] Závazných 9 bodů a jejich levá/pravá sémantika jsou shodné ve všech formátech.
- [ ] Opravený baseline report patří stejným EMA vahám jako baseline export.
- [ ] AP75, crowd/area/maxDet a small-face metriky mají nezávislé fixtures.
- [ ] Val a holdout mají poctivě popsaný původ bez tvrzení o nově „nedotčeném“ historicky používaném testu.
- [ ] B0 a B1 jsou přeměřeny na totožném vstupu a finální Pose9 výstupu.
- [ ] Bounding box je buď fyzicky odstraněn z deployment grafu, nebo jeho ponechání odůvodňuje naměřená výhoda.
- [ ] AMP FP16 training na 8GB GPU má finite gradients, doloženou peak VRAM a použitelný checkpoint/resume.
- [ ] Jednotlivé ablace mají stejné podmínky a zdroj každého nárůstu je dohledatelný.
- [ ] Multi-person crowd, occlusion a truncation se nezhoršily skrytím chybných osob/bodů.
- [ ] ONNX a ncnn oba prošly skutečnou numerickou i accuracy validací.
- [ ] Pi FP16 storage a arithmetic se reportují odděleně, host decode zůstává FP32.
- [ ] Finální Pi benchmark obsahuje p50/p95/p99, camera age, dropped frames, RSS, teplotu a throttling.
- [ ] Vydaný profil splňuje deklarované gates, nebo jsou nesplněné body explicitně označené jako pending/limit.
- [ ] Výsledný report obsahuje absolutní cesty artefaktů, hashes, verze, data/protokol a přesný bod případného pokračování.

Tento dokument předává proveditelnou cestu od současných vah k ověřenému devítibodovému modelu. Neobsahuje tvrzení, že nové přesnosti nebo real-time Pi provozu už bylo dosaženo.
