# XTX4 test app

Testovací aplikace pro natrénované sítě XTX4 (zatím pro Windows). Vstupem je kamera,
video nebo obrázek. Obraz projde pose sítí XTX4 a výsledná 9bodová kostra se vykreslí
do živého obrazu. Kostra má body nos, oči, uši, ramena a boky. XTX4 nepředpovídá
boxy.

Okno má dvě části:

- **Vlevo** je video s vykreslenými pózami.
- **Vpravo** je panel se všemi volbami modelu, dekódování, zdroje a zobrazení.
  Každá změna se projeví **hned za běhu**.

Aplikace **nepotřebuje torch**. Letterbox, dekódování, OKS-NMS a vláknová pipeline se
importují z `xtx4.runtime` o adresář výš. Složka `live_test/` proto musí zůstat uvnitř
repozitáře `xtx4`, případně nastavte `XTX4_ROOT`.

## Spuštění

```
init.bat      (jednou: uv + .venv s Pythonem 3.12 + requirements.txt)
start.bat     (spustí init automaticky, pokud .venv chybí)
```

Při startu se obnoví nastavení z minulé relace (`.last_settings.json`).
S `--fresh` aplikace začne z výchozího nastavení v `config.json`.

## Modely

Aplikace načítá modely z nasazovacích exportů v `../models/<varianta>/`:

| síť v aplikaci | soubory | váhy ncnn | poznámka |
|---|---|---|---|
| `u_b` | `models/u_b/deploy/` | FP16 | síť pro Pi 5; AP 0,582 / 0,625 / 0,647 @256 / 320 / 384 |
| `u_c` | `models/u_c/deploy/` | FP16 | přesnější varianta; AP 0,640 / 0,679 / 0,700 |
| `u_b-fp32w`, `u_c-fp32w` | `models/*/deploy_fp32/` | FP32 | reference přesnosti (ncnn s FP32 vahami) |
| `u_b-random` | `../bench/graphs/` | FP16 | náhodné váhy, jen na měření latence |

Každý export obsahuje grafy ncnn a ONNX pro 256, 320 a 384, dále `manifest.json`
(dekódovací kontrakt) a `parity.json` (shoda s PyTorch). Vznikly příkazem:

```
..\.venv\Scripts\python.exe ..\scripts\export_checkpoint.py --weights ..\models\u_b\best.pt --out ..\models\u_b\deploy --prefix xtx4-u_b
..\.venv\Scripts\python.exe ..\scripts\export_checkpoint.py --weights ..\models\u_b\best.pt --out ..\models\u_b\deploy_fp32 --prefix xtx4-u_b --fp32
```

Další síť přidáte záznamem do `networks` v `config.json`. Stačí `id`, `label`,
`manifest` a případně `description` a `default_size`.

## Co jde nastavit (pravý panel)

| záložka | volby |
|---|---|
| **Model** | síť; backend **ncnn** (nasazení) nebo **ONNX Runtime FP32** (reference); vstupní velikost 256 / 320 / 384; režim **whole** (1 průchod) nebo **pyramid** (3 průchody, ablace z XTX3); počet CPU vláken; FP16 úložiště a aritmetika v ncnn; Vulkan GPU. Pod tím je info o modelu: varianta, přesnost, checkpoint, parita, počet anchorů a soubory. |
| **Detection** | práh detekce; max. počet osob; prahy present a visible (nepřítomný / zakrytý / viditelný bod); práh kreslení bodů; **OKS-NMS** (zapnutí, práh OKS, scale faktor, conf bodů pro NMS, volitelně IoU extent boxů); pokročilé dekódování (`sigma_conf_coeff`, `sigma_floor`); tlačítko „Reset to manifest defaults“. |
| **Source** | kamera (výběr, Media Foundation nebo DirectShow, rozlišení, FPS, MJPG); video soubor nebo RTSP/HTTP URL (smyčka v reálném čase); obrázek (posílá se opakovaně, takže každá změna je hned vidět); zrcadlení. Změny zdroje se projeví až tlačítkem **Open source**. |
| **Display** | čáry kostry, body, zakryté body, skóre, čísla osob, jména bodů, **sigma kruhy** (nejistota polohy z RLE hlavy), **extent boxy NMS**, barvy podle strany těla nebo podle osoby, velikost bodů a čar, HUD, **vstupní pohled sítě** (letterbox s body v pixelech sítě). |
| **Poses** | tabulka osob (skóre, pohled, počet viditelných / zakrytých / nepřítomných bodů, anchor) a pro vybranou osobu všech 9 bodů (x, y, conf, viditelnost, sigma v px). Osobu vyberete kliknutím do tabulky nebo do videa. |
| **Stats** | p50 / p95 / mean / max časů preprocess / síť / postprocess / celkem, čekání ve frontě, stáří snímku, FPS zdroje a inference, zahozené snímky, RSS a CPU procesu; reset a kopírování. |
| **Help** | klávesové zkratky. |

Tlačítka dole:

- **Pause / Step**: zastaví zdroj a pustí jeden snímek. Během pauzy se zmrazený
  snímek po každé změně nastavení detekuje znovu, takže jde ladit prahy na jednom
  záběru.
- **Screenshot**: uloží PNG a JSON se všemi pózami a nastavením do `screenshots/`.
- **Record**: nahrává anotované MP4 do `recordings/`.

### Předvolby

Předvolba je částečné nastavení. Vestavěné předvolby jsou v `config.json` (`presets`):

| předvolba | obsah |
|---|---|
| Pi 5 deployment | u_b @320, ncnn FP16, 3 vlákna, práh 0,15 + OKS-NMS 0,5, visible 0,8 (nasazovací nastavení) |
| Best accuracy | u_c @384 + NMS |
| Fastest | u_b @256 |
| Raw network output | dekódování z manifestu, bez NMS (síť, velikost a backend zůstanou) |
| FP32 reference (ONNX Runtime) | přepne jen backend |
| ncnn pure FP32 | u_b s FP32 vahami a FP32 aritmetikou |
| Crowd / small people | u_c @384, práh 0,10, 100 osob, NMS scale 2,0 + IoU 0,5 |
| Pyramid mode | 3 pohledy |
| Debug overlay / Clean overlay | všechny ladicí vrstvy / jen kostry |

Vlastní předvolbu uložíte tlačítkem **Save current as…** do `user_presets.json`.
Uloží se všechno kromě zdroje. Klíč `"decode": "manifest"` v předvolbě nejdřív vrátí
dekódovací hodnoty na hodnoty z manifestu.

### Klávesy

`mezerník` pauza · `.` další snímek · `+`/`-` práh detekce · `[`/`]` práh kreslení ·
`1`–`3` velikost vstupu · `b` ncnn ↔ ONNX Runtime · `w` whole ↔ pyramid · `n` OKS-NMS ·
`m` zrcadlení · `d` vstup sítě · `e` extent boxy · `g` sigma · `l` jména bodů ·
`c` skóre · `i` HUD · `s` screenshot · `r` nahrávání · `Tab` skrýt panel ·
`F11`/`f` celá obrazovka (`Esc` zpět) · `q` konec

## Příkazová řádka

```
start.bat --preset "Pi 5"                                  # start s předvolbou (stačí začátek názvu)
start.bat --network u_c --size 384 --backend onnxruntime   # přímá volba
start.bat --camera 1 | --camera video.mp4 | --image foto.jpg
start.bat --image foto.jpg --save out.png                  # anotace jednoho obrázku bez okna (+ out.png.json)
start.bat --headless --frames 500 --camera 0 --preset "Pi 5"   # benchmark pipeline bez okna
start.bat --list-networks | --list-presets | --list-cameras
```

Další přepínače:

- volba modelu a výpočtu: `--mode whole|pyramid`, `--threads N`, `--fp16`/`--fp32`,
  `--vulkan`/`--cpu`
- zdroj a obraz: `--resolution 1280x720`, `--mirror`/`--no-mirror`
- detekce: `--conf 0.2`, `--oks-nms`/`--no-oks-nms`
- ostatní: `--fresh`, `--config jiny.json`, `--json stats.json` (headless), `-v`

## Poznámky k přesnosti a rychlosti

- **ncnn FP16 a FP32:** parita FP16 exportu proti PyTorch je xy p95 0,25–0,67 px.
  Je to nad formální tolerancí 0,25 px exportního skriptu, ale gate G2 ukázal ΔAP
  ≤ 0,0004. FP32 export (`*-fp32w`) se s PyTorch shoduje na ~0,001 px. Rozdíl uvidíte
  přímo, když přepnete síť nebo backend (klávesa `b`).
- **ONNX Runtime je připnutý na 1.20.1.** Novější verze tráví při `import onnxruntime`
  na Windows 4–15 s (zjišťování zařízení s drženým GIL), takže by přepnutí backendu
  zamrazilo okno.
- **Zobrazení:** video se kreslí přímo přes GDI (`SetDIBitsToDevice`). Přenos přes Tk
  `PhotoImage` stál 35–50 ms na snímek. Na jiných systémech se použije Pillow `ImageTk`.

## Řešení potíží

- **Pomalé otevírání kamery nebo nefunkční rozlišení:** v záložce Source přepněte na
  DirectShow.
- **Nízké FPS kamery ve vysokém rozlišení:** nechte zapnuté MJPG.
- **`ModuleNotFoundError: xtx4`:** `live_test/` není uvnitř repozitáře xtx4. Nastavte
  `XTX4_ROOT`.
- **Model files missing:** chybí `models/<v>/deploy`. Vyexportujte ho příkazem výše.

Testy jsou v hlavním repozitáři: `tests/test_live_app.py` (nastavení, předvolby,
engine, GUI) a `tests/test_runtime.py` (runtime, ONNX runner, headless běh).
