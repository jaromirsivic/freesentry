# XTX3 live test

Rychlý živý test natrénovaných pose sítí XTX3: obraz z kamery → NCNN síť → vykreslení
kostry (nos, oči, uši, ramena, boky) přímo do živého obrazu. Funguje na Windows, Linuxu
i macOS a nepotřebuje torch – jen `ncnn`, `numpy`, `opencv-python` a `cv2-enumerate-cameras`.

## Spuštění

| | Windows | Linux / macOS |
|---|---|---|
| Inicializace (jednou) | `init.bat` | `./init.sh` |
| Start | `start.bat` | `./start.sh` |

- `init.*` nainstaluje [uv](https://docs.astral.sh/uv/) z astral.sh, pokud chybí,
  vytvoří `.venv` s Pythonem 3.12 (uv si ho případně sám stáhne) a nainstaluje
  `requirements.txt`. Při opakovaném spuštění se zeptá, zda prostředí vytvořit znovu.
- `start.*` spustí `init` automaticky, pokud `.venv` ještě neexistuje.
  Argumenty se předávají do `main.py`.

Po startu se otevře dialog: **síť**, **velikost vstupu**, **kamera** (nebo video
soubor / RTSP-HTTP URL), rozlišení kamery, práh detekce, zrcadlení a Vulkan GPU.
Poslední volba se pamatuje (`.last_selection.json`). Bez tkinteru se použije
textové menu v konzoli.

### Příkazová řádka

```
start.bat --network xtx3-u --size 320 --camera 0 --no-dialog
start.bat --camera video.mp4 --no-dialog        # video ve smyčce
start.bat --image foto.jpg --save vystup.png    # jeden obrázek
start.bat --list-cameras
start.bat --console                             # textové menu místo dialogu
```

Další přepínače: `--resolution 1920x1080`, `--conf 0.4`, `--vulkan` / `--cpu`,
`--mirror` / `--no-mirror`, `--dedup`, `--config jiny.json`, `-v`.

## Ovládání v okně

| Klávesa | Akce |
|---|---|
| `q` / `Esc` | konec (nebo zavřít okno křížkem) |
| `mezerník` | pauza / pokračovat |
| `+` / `-` | práh detekce ±0,05 |
| `[` / `]` | práh keypointů ±0,05 |
| `1`…`9` | velikost vstupu podle pořadí v `sizes` (u xtx3-u: 256 / 320 / 384) |
| `b` / `l` / `i` | bounding boxy / jména bodů / info panel |
| `n` | potlačení duplicitních detekcí (výchozí vypnuto = surový výstup sítě) |
| `m` | zrcadlení |
| `d` | debug: vstupní pohled(y) sítě (letterbox / 3 pohledy pyramidy) |
| `f` | celá obrazovka |
| `s` | screenshot do `screenshots/` |
| `h` | nápověda |

Body: plné kolečko = viditelný, prázdné = zakrytý / nejistý. Zelená = levá strana
osoby, oranžová = pravá, žlutá = středové spojnice.

## Přidání další sítě (`config.json`)

Do pole `networks` přidejte záznam. Cesty jsou relativní k `config.json`,
`{size}` se nahradí zvolenou velikostí:

```json
{
  "id": "xtx3-n",
  "label": "XTX3-n (2.65 M params, pyramid)",
  "format": "ncnn",
  "param": "../runs/xtx3-n/export/xtx3_n_{size}.ncnn.param",
  "bin": "../runs/xtx3-n/export/xtx3_n_{size}.ncnn.bin",
  "sizes": [384],
  "default_size": 384,
  "mode": "pyramid",
  "strides": [8, 16, 32],
  "input_blob": "in0",
  "output_blobs": { "box": "out0", "cls": "out1", "kpt": "out2" },
  "skeleton_preset": "xtx3_9"
}
```

- `mode`: `whole` (u) nebo `pyramid` (n/m/l – 3 průchody sítí na snímek a sloučení
  s prioritou pohled 1 > 2 > 3, stejně jako v `xtx3/api.py`).
- `strides`: `[8,16,32]`, pro variantu l (s P2) `[4,8,16,32]`. Při nesouladu
  s počtem anchorů, které síť vrací, aplikace strides odvodí sama a vypíše varování.
- `output_blobs`: pnnx export pojmenuje výstupy `out0..2`. Pokud v síti nejsou,
  zkusí se `box`/`cls`/`kpt` (export přes onnx2ncnn).
- Volitelně `"conf_threshold"` – výchozí práh pro tuto síť.
- Síť, ke které chybí soubory, se v dialogu zobrazí jako „model files missing“.

Ostatní sekce: `camera` (rozlišení, FPS, MJPG, `windows_backend`: `msmf`/`dshow`),
`inference` (prahy, `num_threads`, `use_vulkan`, `async_inference` pro pomalé sítě,
`dedup`), `display`.

## Testy

```
uv run --no-project --python .venv\Scripts\python.exe --with pytest --with onnxruntime python -m pytest tests
..\.venv\Scripts\python.exe -m pytest tests     # parita s xtx3 (torch) z venv xtx3
```

- `test_preprocess_parity.py` – letterbox / pyramida bit po bitu shodná s `xtx3.data.preprocess`.
- `test_decode_parity.py` – dekódování shodné s `xtx3.export.ncnn_export.decode_raw_outputs`.
- `test_runner_parity.py` – NCNN proti ONNX Runtime (napojení blobů, BGR, /255).
  pnnx ukládá váhy NCNN ve fp16, proto se surové výstupy liší v řádu 1e-3
  (výjimečně až ~0,3).

## Řešení potíží

- **macOS:** při prvním spuštění povolte kameru pro terminál
  (Nastavení systému → Soukromí a zabezpečení → Kamera).
- **Linux:** uživatel musí být ve skupině `video` (`sudo usermod -aG video $USER`).
  Na minimálních systémech OpenCV potřebuje `sudo apt install libgl1 libglib2.0-0`.
- **Windows:** pomalé otevírání kamery nebo nefunkční rozlišení → v configu
  `"windows_backend": "dshow"`.
- **Nízké FPS kamery ve vysokém rozlišení:** nechte `"mjpg": true`.
