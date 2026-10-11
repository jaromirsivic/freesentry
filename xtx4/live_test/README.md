# XTX4 live test

Rychlý živý test exportovaných pose sítí XTX4: obraz z kamery → ncnn síť → vykreslení
9bodové kostry (nos, oči, uši, ramena, boky) přímo do živého obrazu. Bez boxů –
XTX4 je box-free. Funguje na Windows, Linuxu, macOS i na Raspberry Pi 5 a
**nepotřebuje torch**: jen `ncnn`, `numpy`, `opencv-python`, `cv2-enumerate-cameras`
a `psutil`.

Dekódování, letterbox a vláknová pipeline se **nevendorují** – importují se z balíčku
`xtx4.runtime` o adresář výš (`xtx4/runtime/`, bez závislosti na torch). Složka
`live_test/` proto musí zůstat uvnitř repozitáře `xtx4` (na Pi stačí zkopírovat celé
repo, nebo nastavit proměnnou prostředí `XTX4_ROOT` na kořen repozitáře).

## Spuštění

| | Windows | Linux / macOS / Pi |
|---|---|---|
| Inicializace (jednou) | `init.bat` | `./init.sh` |
| Start | `start.bat` | `./start.sh` |

- `init.*` nainstaluje [uv](https://docs.astral.sh/uv/) z astral.sh, pokud chybí,
  vytvoří `.venv` s Pythonem 3.12 a nainstaluje `requirements.txt`. Na konci ověří,
  že jde importovat `xtx4.runtime` bez torch.
- `start.*` spustí `init` automaticky, pokud `.venv` ještě neexistuje.
  Argumenty se předávají do `main.py`.

Po startu se otevře dialog: **síť**, **velikost vstupu**, **kamera** (nebo video
soubor / RTSP-HTTP URL), rozlišení kamery, práh detekce, zrcadlení a Vulkan GPU.
Poslední volba se pamatuje (`.last_selection.json`). Bez tkinteru se použije
textové menu v konzoli.

### Příkazová řádka

```
start.bat --network xtx4-u --size 320 --camera 0 --no-dialog
start.bat --camera video.mp4 --no-dialog            # video ve smyčce
start.bat --image foto.jpg --save vystup.png        # jeden obrázek
start.bat --headless --frames 500 --camera 0        # benchmark pipeline bez okna
start.bat --headless --frames 300 --image foto.jpg  # benchmark na statickém snímku (simulace 30 fps kamery)
start.bat --list-cameras
start.bat --console                                 # textové menu místo dialogu
```

Další přepínače: `--resolution 1920x1080`, `--conf 0.4`, `--threads 3`,
`--fp16` / `--fp32`, `--vulkan` / `--cpu`, `--mirror` / `--no-mirror`, `--oks-nms`,
`--config jiny.json`, `--json stats.json` (headless), `--warmup 20`, `-v`.

`--headless` spustí stejnou vláknovou pipeline jako okno (vlákno kamery → fronta
hloubky 1 „nejnovější snímek“ → vlákno inference) a vypíše statistiku: zachycené /
zpracované / zahozené snímky, FPS kamery a inference a p50/p95 časů
preprocess / inference (ncnn `extract`) / postprocess / celkem / čekání ve frontě /
stáří snímku (od zachycení po hotový výsledek), plus RSS procesu.

## Ovládání v okně

| Klávesa | Akce |
|---|---|
| `q` / `Esc` | konec (nebo zavřít okno křížkem) |
| `mezerník` | pauza / pokračovat |
| `+` / `-` | práh detekce ±0,05 |
| `[` / `]` | práh keypointů ±0,05 |
| `1`…`9` | velikost vstupu podle pořadí v `sizes` (typicky 256 / 320 / 384) |
| `n` | OKS-NMS (potlačení duplicit; výchozí vypnuto = surový one-to-one výstup) |
| `c` / `l` / `i` | skóre osob / jména bodů / info panel |
| `m` | zrcadlení |
| `d` | debug: vstupní pohled sítě (letterbox) s body v síťových pixelech |
| `f` | celá obrazovka |
| `s` | screenshot do `screenshots/` |
| `h` | nápověda |

Body: plné kolečko = viditelný, prázdné = zakrytý. Nepřítomné body (present < 0,5)
a body pod prahem keypointů se nekreslí. Zelená = levá strana osoby, oranžová =
pravá, žlutá = středové spojnice. Nad osobou je její skóre.

Info panel: síť, velikost, CPU vlákna / Vulkan, fp16; FPS kamery, inference a
zobrazení, počet zahozených snímků; klouzavé průměry preprocess / net / post v ms
a stáří snímku; počet osob a kandidátů nad prahem.

## Napojení na natrénovaný běh (`D:/xtx4_runs/<run>/export/`)

1. Exportujte checkpoint (v hlavním venv XTX4 s torch):

   ```
   ..\.venv\Scripts\python.exe ..\scripts\export_checkpoint.py --weights D:/xtx4_runs/<run>/best.pt
   ```

   Vznikne `D:/xtx4_runs/<run>/export/` s `xtx4_{256,320,384}.onnx`,
   `xtx4_{size}.ncnn.param/.bin`, `manifest.json` a `parity.json`
   (parita PyTorch × ONNX Runtime × ncnn).

2. V `config.json` přidejte (nebo upravte) záznam sítě – stačí cesta k manifestu,
   velikosti, strides, názvy blobů, dekódovací parametry i metadata bodů se
   načtou z něj:

   ```json
   {
     "id": "xtx4-u",
     "label": "XTX4-u (run xtx4-u)",
     "manifest": "D:/xtx4_runs/xtx4-u/export/manifest.json",
     "default_size": 320
   }
   ```

   Alternativně bez manifestu (např. grafy s náhodnými vahami z `bench/graphs`):

   ```json
   {
     "id": "u_b-random",
     "param": "../bench/graphs/u_b_{size}.ncnn.param",
     "bin": "../bench/graphs/u_b_{size}.ncnn.bin",
     "sizes": [256, 320, 384],
     "strides": [8, 16, 32],
     "output_blobs": { "person": "out0", "pose": "out1" }
   }
   ```

- Relativní cesty jsou vůči `config.json`, `{size}` se nahradí zvolenou velikostí.
  `param`/`bin` mají přednost před soubory z manifestu.
- `strides`: `[8,16,32]` (P3P4P5), `[8,32]` (P3P5) nebo `[16,32]` (P4P5). Když
  chybí nebo nesedí s počtem anchorů, které síť vrací, odvodí se z něj (s varováním).
- Výstupy grafu: `out0` = person `(N,1)`, `out1` = pose `(N,45)` (9 bodů × 5).
  Pokud nakonfigurované názvy v `.param` nejsou a graf má právě dva výstupy, použijí
  se v pořadí grafu.
- Volitelně `"conf_threshold"` – výchozí práh pro tuto síť; `"mode": "pyramid"` pro
  3pohledový režim (jen pro ablace, výchozí `whole`).
- Síť, ke které chybí soubory nebo manifest, se v dialogu zobrazí jako „model files missing“.

Ostatní sekce: `camera` (rozlišení, FPS, MJPG, `windows_backend`: `msmf`/`dshow`),
`inference` (prahy, `max_detections`, `num_threads`, `use_vulkan`, `fp16`, `oks_nms`), `display`.

Dekódovací a NMS parametry v `inference` – hodnota `"manifest"` znamená „vzít
z bloku `decode` v manifestu sítě“ (bez manifestu výchozí hodnoty), číslo ji přebije:

| klíč | význam | výchozí |
|---|---|---|
| `present_threshold` | bod je nepřítomný (nekreslí se), když `present_prob < práh` | 0,5 |
| `visible_threshold` | přítomný bod je viditelný (plné kolečko), když `visible_prob >= práh`, jinak zakrytý | 0,5 |
| `oks_nms_threshold` | OKS-NMS: duplicita, když OKS ≥ práh | 0,5 |
| `oks_nms_scale_factor` | zvětšení obálky jistých bodů pro měřítko OKS (1,5 ≈ box hlava+trup; metrika používá větší box celé osoby, proto vyšší hodnota potlačí víc) | 1,5 |
| `oks_nms_iou_threshold` | navíc potlačit, když IoU zvětšených obálek ≥ práh (`null` = vypnuto; XTX3 používal 0,5) | `null` |

## Raspberry Pi 5

- Doporučeno: `num_threads` 4 (případně 3, ať jedno jádro zbude kameře), `fp16: true`,
  Vulkan vypnutý, rozlišení kamery 1280x720 MJPG.
- Latence samotné sítě měří `scripts/bench_pi.py`; celou pipeline včetně kamery
  `./start.sh --headless --frames 1000 --camera 0 --network xtx4-u --size 320`.

## Testy

Testy runtime jsou v hlavním repozitáři (`tests/test_runtime.py`): NumPy dekódování
proti torch `decode_detections`, letterbox bit po bitu proti `xtx4.data.preprocess`,
ncnn proti ONNX Runtime, OKS-NMS, manifest, export a headless běh tohoto live testu
bez torch.

```
..\.venv\Scripts\python.exe -m pytest ..\tests\test_runtime.py -q
```

## Řešení potíží

- **macOS:** při prvním spuštění povolte kameru pro terminál
  (Nastavení systému → Soukromí a zabezpečení → Kamera).
- **Linux / Pi:** uživatel musí být ve skupině `video` (`sudo usermod -aG video $USER`).
  Na minimálních systémech OpenCV potřebuje `sudo apt install libgl1 libglib2.0-0`.
- **Windows:** pomalé otevírání kamery nebo nefunkční rozlišení → v configu
  `"windows_backend": "dshow"`.
- **Nízké FPS kamery ve vysokém rozlišení:** nechte `"mjpg": true`.
- **`ModuleNotFoundError: xtx4`:** `live_test/` není uvnitř repozitáře xtx4 – nastavte
  `XTX4_ROOT`.
