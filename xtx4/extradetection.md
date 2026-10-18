# XTX4 – doplňková detekce orgánů (mozek, hruď)

Implementační plán. Cílem je přidat na úplný konec sítě XTX4 netrénovanou vrstvu s pevnými parametry.
Ta pro každou detekovanou pózu spočítá **mozek** a **hruď**, každý jako kruh `(x, y, radius)` a s
konfidencí. Stávající výstupy (`person`, `pose`) zůstávají beze změny, orgány se k nim jen přidávají.
Live View pak umožní přepínat zobrazení mezi klíčovými body, orgány a obojím.

Srdce, játra ani břicho (abdomen) se nepočítají.

---

## 1. Zdroj pravdy: výpočet v serveru FreeSentry

Výpočty se přebírají beze změny z `server/cameraai.py`:
`_get_brain_position` (ř. 232–307) a `_get_chest_and_abdomen_position` (ř. 309–347, z ní jen `chest`).
Pomocné funkce:

```
calculate_point(a, b, f) = a + f·(b − a) = (1 − f)·a + f·b
calculate_distance(a, b) = ‖a − b‖₂
```

Server počítá se 17 body COCO. XTX4 má 9 bodů (`nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_hip, right_hip`), které obsahují všechny body potřebné pro mozek
i hruď. Značení: `N` = nos, `LEy/REy` = oči, `LEa/REa` = uši, `LS/RS` = ramena, `LH/RH` = boky,
`c_k` = konfidence bodu k.

### 1.1 Mozek – po rozepsání řetězce `calculate_point` do lineární kombinace

**Plná větev** (výchozí):

```
nose_left_ear      = −N + 2·LEa
nose_left_ear2     = 0.5·N + 0.5·LEa
nose_left_eye      = −0.5·N − 0.5·LEa + 2·LEy
nose_left_ear_eye  = −0.75·N + 0.75·LEa + LEy           (pravá strana symetricky)
brain_full = −0.75·N + 0.5·LEy + 0.5·REy + 0.375·LEa + 0.375·REa
```

Součet vah je **1** (−0.75 + 0.5 + 0.5 + 0.375 + 0.375). Jde tedy o afinní kombinaci, což je podstatné
pro bod 3.2.

```
r_full    = 0.59 · max(‖B − N‖, ‖B − LEa‖, ‖B − REa‖)
conf_full = min5 + 0.8·(avg5 − min5) = 0.2·min5 + 0.8·avg5      (min/avg přes N, LEy, REy, LEa, REa)
```

**Ušní větev**, použije se právě tehdy, když `NOT all5_ok AND c_LEa ≥ 0.01 AND c_REa ≥ 0.01`:

```
brain_ear = 0.5·LEa + 0.5·REa
r_ear     = 0.59 · ‖brain_ear − LEa‖ = 0.295 · ‖LEa − REa‖
conf_ear  = (c_LEa + c_REa) / 2
```

`all5_ok` platí, když mají všech pět bodů hlavy `c ≥ 0.01`. Práh 0.01 je výchozí `confidence_threshold`
funkce v serveru.

### 1.2 Hruď

```
chest = 0.8·mid(LS,RS) + 0.2·mid(LH,RH) = 0.4·LS + 0.4·RS + 0.1·LH + 0.1·RH      (součet vah = 1)
d1 = ‖LS − RS‖, d2 = ‖LH − RH‖, d3 = ‖LS − LH‖, d4 = ‖RS − RH‖
r_chest    = max(0.21·(d1 + d2), 0.1·(d3 + d4))
conf_chest = clamp((2·c_LS + 2·c_RS + c_LH + c_RH + min(c_LS, c_RS, c_LH, c_RH)) / 7, 0, 1)
```

**Odchylka od serveru (rozhodnutí O2):** server dělí pěti, takže `conf_chest` tam může dosáhnout až
1.4 (ramena se počítají dvakrát). XTX4 dělí **sedmi**, tedy součtem vah 2 + 2 + 1 + 1 + 1. Výsledek je
skutečný vážený průměr a pro `c_k ∈ [0, 1]` leží vždy v intervalu **[0, 1]**. Na rozdíl od pouhého
oříznutí se tím neztratí rozlišení mezi dobře a velmi dobře viditelným trupem. Závěrečný
`clamp(…, 0, 1)` jen hlídá zaokrouhlení ve FP16.

Platí přesně `conf_chest_xtx4 = conf_chest_server · 5/7`, protože server hodnotu neořezává. Serverový
práh 0.20 proto odpovídá **0.143** (`0.20 · 5/7`). Tato hodnota je výchozí v Live View, aby se hruď
kreslila za stejných podmínek jako ve FreeSentry.

Stejný ochranný `clamp(…, 0, 1)` dostane i `conf_brain`. Ta je konvexní kombinací konfidencí, takže
v [0, 1] leží už ze své podstaty.

### 1.3 Grafický styl (server `aidraw.draw_aicircle`, 1:1)

```
circle_outline_size = clamp(int(max(W, H) / 640), 1, 20)
r   = int(radius · sizeMultiplier)
cb  = 1 − |conf − 0.5|·2
color (BGR) = (int(cb·255), 0, int(conf·255))
confidence_achieved AND r > minimumRadius:
    r > 8·os : bílý kruh  r−3·os, tloušťka 8·os
               barevný    r−4·os, tloušťka 3·os
    jinak    : vyplněný bílý kruh r·os
               barevný obrys r·os, tloušťka 4·os
jinak:       tenký barevný kruh r·os, tloušťka 2·os
```

`os = circle_outline_size`. Násobení `r·os` v malých větvích je zvláštnost serveru. Přebírá se beze
změny, aby kruhy vypadaly stejně (rozhodnutí O4).

Výchozí hodnoty nastavení podle `server/wwwroot/src/assets/settings_default.json`:

| orgán | enabled | sizeMultiplier | confidenceThreshold | minimumRadius |
|---|---|---|---|---|
| brain | true | 1.0 | 0.25 | 1 |
| chest | true | 1.0 | 0.143 (server 0.20 · 5/7, viz 1.2) | 1 |

---

## 2. Cílový stav (co síť vrací)

| výstup | tvar | obsah | změna |
|---|---|---|---|
| `person` (`out0`) | `[1, N, 1]` | logit osoby | beze změny |
| `pose` (`out1`) | `[1, N, 9·5]` | `(dx, dy, log_sigma, present_logit, visible_logit)` | beze změny |
| **`organs` (`out2`)** | **`[1, N, 2·4]`** | pro `brain`, pak `chest`: **`(dx, dy, r_units, conf)`** | **nový** |

- `dx, dy` mají stejnou konvenci jako u klíčových bodů: `x = ax + dx·stride`, `y = ay + dy·stride`.
- `r_units` je poloměr v jednotkách kroku mřížky (stride): `r_px = r_units · stride`.
- `conf` je přímo pravděpodobnost, ne logit. Počítá se ze `sigmoid(present_logit)` podle vzorců z bodu 1.
- Výstup existuje pro všech `N` anchorů. Host z něj vybere řádky stejných anchorů, které prošly
  prahem osoby (`select_candidates`), takže každá vrácená póza nese i svůj mozek a hruď.

Ve frame pixelech je pak `Pose` rozšířená o pole `organs: np.ndarray | None` tvaru `(2, 4)` s hodnotami
`[x, y, radius_px, conf]` (řádek 0 = brain, 1 = chest). Klíčové body se nijak nemění.

---

## 3. Návrhová rozhodnutí

### 3.1 Vrstva bez trénování, bez vah ve state_dict
- Nový modul `xtx4/models/organs.py` → `OrganLayer(nn.Module)`. Konstanty z bodu 1 jsou v něm
  zaregistrované jako buffery s `persistent=False`, takže **formát checkpointu se nemění**
  a `load_state_dict(strict=True)` funguje se stávajícími `best.pt`.
- Žádné `nn.Linear` ani `nn.Conv`: jen funkcionální operace (`mul`, `add`, `sqrt`, `maximum`,
  `minimum`, `clamp`, `sigmoid`). `num_parameters()` a `mac_count()` (hooky na conv/linear) tak zůstanou
  beze změny.
- Trénink a `HeadOutput` zůstávají beze změny, takže se nemusí sahat do `head_types.py`,
  který je podle CONTRACTS §3 zamčený. Vrstva se volá jen v exportní cestě a v `predict`.

### 3.2 Počítání v relativních jednotkách anchoru (bez konstant anchorů v grafu a bezpečné pro FP16)
Obě středové kombinace mají součet vah 1, proto platí

```
Σ w_k · (a + s·d_k) = a + s · Σ w_k·d_k
‖P_i − P_j‖ = s · ‖d_i − d_j‖
```

Celý výpočet lze tedy provést přímo nad surovými `(dx, dy)` z výstupu `pose`. Graf nepotřebuje
souřadnice anchorů ani stride a výsledek se dekóduje stejně jako klíčové body. Výhody:

1. **FP16 (ncnn na Pi)**: v pixelech by `dx² + dy²` pro vzdálenost ~384 px přeteklo
   (384² = 147 456 > 65 504 = maximum FP16). V jednotkách stride jsou hodnoty v řádu jednotek až desítek.
2. Graf zůstává nezávislý na vstupní velikosti stejně jako dnes a stejná vrstva funguje pro 256, 320 i 384.
3. Na hostu se dekóduje stejně jako `pose` (`decode_selected`), takže se nic nového neodvozuje.

### 3.3 Větvení mozku bez `if` (bez `Where` v ncnn)
ncnn/pnnx nemusí spolehlivě převést `torch.where`. Hradlo se proto skládá jen z aritmetiky:

```
g(c)  = clamp((c − 0.01) · 1e4, 0, 1)          # schod; 1e4 je bezpečné pro FP16
m     = g(c_LEa) · g(c_REa) · (1 − g(c_N)·g(c_LEy)·g(c_REy))
brain = m·brain_ear + (1 − m)·brain_full       # stejně r a conf
```

Výsledek je přesně shodný se serverem, s výjimkou pásma `|c − 0.01| < 1e-4`, kde hradlo přechází
lineárně. Prakticky se to neprojeví a test to pokrývá (bod 6).

### 3.4 Vektorizace kvůli latenci v ncnn
V ncnn je každá elementwise operace samostatná vrstva nad tenzorem velikosti `N`. Proto se počítá
vektorově:
- `P = pose.reshape(1, N, 9, 5)`, `D = P[..., 0:2]`, `C = sigmoid(P[..., 3])`,
- středy `brain_full`, `brain_ear`, `chest` jako jeden součin `D` s konstantní maticí vah `9×3`
  (přes `mul` + `sum`, nebo `matmul` s bufferem, podle toho, co pnnx převede na podporované vrstvy),
- vzdálenosti jako jeden blok `[N, k, 2]` → `sum(sq)` → `sqrt`.

Cíl: přidané vrstvy zaberou < 0.3 ms na Pi 5 při 320 px (tedy < 3 % z 11.7 ms sítě u_b). Ověřuje se
v bodě 7.

### 3.5 Konfidence bodů
Pro vzorce se bere `c_k = sigmoid(present_logit_k)`. Hostový parametr `sigma_conf_coeff` (výchozí 0,
v presetu pro Pi také 0) do grafu nevstupuje. Při nenulové hodnotě by se konfidence orgánu lišila od
`conf` zobrazovaného klíčového bodu. Bude to zdokumentované v manifestu.

### 3.6 Zrcadlení a pyramid
- Oba vzorce jsou levo-pravě symetrické, takže zrcadlení (`mirror`) ani prohození L/R výsledek nezmění.
- V režimu pyramid se orgány převádějí afinní transformací každého pohledu (`to_frame_poses`) a
  `merge_views` / OKS-NMS je přenese spolu s pózou. Výběr pózy se kvůli orgánům nemění.

### 3.7 Zpětná kompatibilita
Starý graf bez `out2` (staré manifesty, bench graf `u_b-random` s explicitními `output_blobs`)
**musí dál fungovat**. Runner v takovém případě dopočítá orgány na hostu NumPy referencí
(`xtx4/utils/organs.py`) ze zvolených `(dx, dy, present)`. Je to stejný kód, proti kterému se testuje
vrstva v grafu. Live View zobrazí, odkud orgány pocházejí (`graph` / `host`).

---

## 4. Změny po souborech

### 4.1 Konstanty a reference (jediný zdroj pravdy)
**`xtx4/utils/organs.py`** (nový, jen NumPy):
- `ORGAN_NAMES = ("brain", "chest")`, `ORGAN_CHANNELS = ("dx", "dy", "r_units", "conf")`,
  `ORGAN_CHANNELS_PER_ORGAN = 4`, `ORGAN_VERSION = 1`.
- Váhy: `BRAIN_FULL_WEIGHTS`, `BRAIN_EAR_WEIGHTS`, `CHEST_WEIGHTS` (vektory délky 9 v pořadí
  `KEYPOINT_NAMES`), `BRAIN_RADIUS_FACTOR = 0.59`, `BRAIN_EAR_THRESHOLD = 0.01`,
  `BRAIN_CONF_MIN_WEIGHT = 0.2`, `CHEST_RADIUS_LR = 0.21`, `CHEST_RADIUS_TB = 0.1`,
  `CHEST_CONF_WEIGHTS = (2, 2, 1, 1, 1)` (LS, RS, LH, RH, min), `CHEST_CONF_DIVISOR = 7.0`,
  `GATE_GAIN = 1e4`.
- `compute_organs(xy (…, 9, 2), conf (…, 9), *, hard_gate: bool = True) -> (…, 2, 4)`. Jde o referenční
  implementaci z bodu 1, která pracuje v libovolných jednotkách (relativních i pixelových).
- `organs_to_dict(organs)` vrací serverový tvar `{"brain": {"point", "radius", "confidence"}, …}`
  pro případnou integraci do FreeSentry serveru.
- Docstring odkazuje na `server/cameraai.py` a obsahuje odvození vah.

### 4.2 Vrstva v síti
**`xtx4/models/organs.py`** (nový): `OrganLayer(num_keypoints=9)`, `forward(pose_flat [B,N,K*5]) ->
organs [B,N,8]`. Konstanty bere z `xtx4/utils/organs.py`, používá měkké hradlo z bodu 3.3 a vektorizaci
z bodu 3.4.

**`xtx4/models/xtx4.py`**: `self.organs = OrganLayer(...)` (vždy, bez parametrů). `predict(...)` vrací
navíc orgány vybraných detekcí (nový návratový prvek za klíčovým argumentem `with_organs=False`, aby se
nerozbila stávající signatura).

**`xtx4/models/__init__.py`**: export `OrganLayer`.

### 4.3 Export
**`xtx4/export/onnx_ncnn.py`**:
- `OUTPUT_NAMES = ("person", "pose", "organs")`.
- `ExportModel(model, with_organs=True)`: `forward` vrací `(person, pose, organs)`; s `with_organs=False`
  zůstává dnešní dvojice (kvůli porovnávání).
- `verify_onnx` porovnává i `organs` (tolerance 1e-3).
- Kontrola typů vrstev v `.param` (už existuje histogram v `ř. 187`): allowlist pro organovou část
  (`BinaryOp`, `UnaryOp`, `Sigmoid`, `Clip`, `Slice`/`Crop`, `Reshape`, `Concat`, `Reduction`, `MatMul`/`Gemm`).
  Při výskytu neznámého typu nebo `pnnx.*` export selže.

**`xtx4/export/manifest.py`**:
- `schema_version` +1, `outputs.names.organs`, `outputs.shapes.organs = [1, N, 8]`,
  `outputs.blobs.organs` (`out2`), popis layoutu.
- Nový blok `"organs"`: `version`, `names`, `channels`, `units` (`dx,dy,r` ve stride, `conf`
  jako pravděpodobnost), `source: "server/cameraai.py"`, všechny konstanty z 4.1 (kvůli auditu, runtime
  je nečte).
- `decode_version` zůstává 1, protože dekódování `pose` se nemění.

**`xtx4/export/deploy.py`**: `blobs` s `organs`; parity reportuje i `organs`
(PyTorch ↔ ONNX ↔ ncnn FP32 ↔ ncnn FP16).

**`CONTRACTS.md`** §3 a §12: nový výstup, layout, jednotky a závazek, že `person` a `pose` se nemění.
Upravit **jako první** (podle §14 se kontrakty mění jako první).

### 4.4 Runtime
**`xtx4/runtime/decode.py`**:
- `Pose.organs: np.ndarray | None = None` (`(2, 4)`, `[x, y, r_px, conf]`).
- `decode_organs_selected(organs_raw (N,8), keep, network_size, strides) -> (D, 2, 4)` v síťových
  pixelech: `x = ax + dx·s`, `r = r_units·s`.
- `organs_from_pose_selected(pose, keep, …)` je hostová záloha pro graf bez `organs`; volá
  `xtx4.utils.organs.compute_organs` nad relativními `(dx, dy)`.
- `to_frame_poses(..., organs=None)`: střed přes `lin`/`offset`, poloměr × `sigma_scale` (stejné
  měřítko jako u sigmy).

**`xtx4/runtime/ncnn_runner.py`** a **`onnx_runner.py`**:
- `_resolve_blobs` přijme 2 nebo 3 výstupy, `forward` vrací `(person, pose, organs | None)`.
- `detect(...)` dekóduje orgány pro `keep` a předá je do `to_frame_poses`.
- `self.organ_source = "graph" | "host"` pro HUD; `DetectionResult` beze změny tvaru, data nese `Pose`.
- `Timings`: orgány na hostu se započítají do `postprocess_ms`.

**`xtx4/runtime/__init__.py`**, **`xtx4/api.py`**: re-export nových funkcí, `Pose.organs` ve výstupu
API (dict / JSON).

### 4.5 Live View (`live_test/`)
**`livetest/draw.py`**:
- `draw_organs(frame, poses, opts)`: port `server/aidraw.draw_aicircle` 1:1 (bod 1.3).
- `DrawOptions`: `overlay_mode: str = "organs"` (`"keypoints" | "organs" | "both"`, výchozí podle O3),
  `brain_enabled`, `brain_size_multiplier`, `brain_conf_threshold`, `brain_min_radius`, totéž pro `chest`.
- `draw_poses` respektuje `overlay_mode`: v režimu `organs` se kostra a body nekreslí. V režimu `both`
  se nejdřív kreslí orgány a přes ně kostra, aby bílé okraje kruhů nepřekryly body.
- Skóre a číslo osoby se umístí nad nejvyšší viditelný bod; v režimu `organs` nad horní okraj kruhu
  mozku, případně hrudi.

**`livetest/settings.py`**: nová pole podle `DrawOptions`, `overlay_mode` v `_CHOICES`, rozsahy
(`size_multiplier` 0.1–5, `conf_threshold` 0–1, `min_radius` 0–200), zařazení do `DISPLAY_KEYS`.

**`livetest/gui.py`**:
- Záložka Display → skupina **Overlay**: přepínač „Show: Keypoints / Organs / Both“ (radio).
- Nová skupina **Organs (FreeSentry style)**: pro brain a chest checkbox *enabled*, slider
  *size multiplier*, slider *confidence threshold*, spin *minimum radius*. Tooltip odkazuje na server
  `ai_setup.organs`.
- Klávesa **`o`** cyklicky přepíná `keypoints → organs → both` (zatím volná) a zobrazí flash zprávu.
- Tabulka bodů (`_build_poses_tab`, sloupce `keypoint, x, y, conf, visibility, sigma px`) dostane navíc
  řádky `brain` / `chest`, kde sloupec *sigma px* zobrazí radius. Popisek sloupce se změní na „sigma / r px“.
- HUD: `organs: graph|host`.

**`live_test/config.json`**: `display.overlay_mode = "organs"` (rozhodnutí O3: po spuštění se ukazují
orgány), `display.organs` s výchozími hodnotami z tabulky v bodě 1.3. Presety:
- nový „Organs (FreeSentry style)“: `overlay_mode: organs`, `show_scores: false`,
- nový „Keypoints + organs“: `overlay_mode: both`,
- nový „Keypoints only“: `overlay_mode: keypoints` (návrat k dosavadnímu zobrazení),
- stávající presety zaměřené na body dostanou `overlay_mode` explicitně, protože presety jsou jen
  částečná nastavení a jinak by zůstal aktivní režim `organs`: „Debug overlay“ → `both`,
  „Clean overlay“ → `keypoints`.

**`livetest/config.py`**: načtení `display.organs` a volitelných `output_blobs.organs`.

**`live_test/README.md`**: popis přepínače, klávesy `o` a nastavení orgánů.

### 4.6 Modely v repozitáři
Přeexportovat `models/u_b/{deploy,deploy_fp32}` a `models/u_c/{deploy,deploy_fp32}` pomocí
`scripts/export_checkpoint.py` (z `best.pt`, bez tréninku) a aktualizovat `parity.json` a
`reports/pi_latency_e2e.csv`. Bench grafy `bench/graphs` (náhodné váhy) není nutné měnit, protože
fungují přes hostovou zálohu.

---

## 5. Pořadí práce (fáze)

| # | fáze | výstup | závisí na |
|---|---|---|---|
| F0 | Kontrakty | CONTRACTS.md §3, §12 | – |
| F1 | Reference + testy proti serveru | `xtx4/utils/organs.py`, `tests/test_organs.py` | F0 |
| F2 | `OrganLayer` + parita s referencí | `xtx4/models/organs.py`, `xtx4.py` | F1 |
| F3 | Export ONNX/ncnn + manifest + allowlist vrstev | `onnx_ncnn.py`, `manifest.py`, `deploy.py` | F2 |
| F4 | Runtime dekód + hostová záloha | `decode.py`, runnery, `api.py` | F1 (F3 jen pro cestu „graph“) |
| F5 | Live View: kreslení, nastavení, GUI | `live_test/*` | F4 |
| F6 | Přeexport modelů, parita, latence na Pi | `models/*`, `reports/*` | F3, F4 |

F4 (hostová záloha) a F5 lze rozpracovat souběžně s F2–F3, protože stačí reference z F1.

---

## 6. Testy

**`tests/test_organs.py`** (nový):
1. **Zlatá shoda se serverem**: do testu se doslovně zkopírují `calculate_point`, `calculate_distance`,
   `_get_brain_position` a `chest` část z `server/cameraai.py` (bez importu serveru). Na 1000 náhodných
   pózách se porovnají s `compute_organs`; tolerance 1 px na střed, 1 px na radius (server ořezává
   na `int`) a 1e-6 na konfidenci. Konfidence hrudi se porovnává se serverovou hodnotou vynásobenou
   `5/7` (záměrná odchylka O2), konfidence mozku přímo.
2. **Větve mozku**: všech 5 bodů OK → plná; ucho pod prahem → plná; oči/nos pod prahem a uši OK → ušní;
   hodnoty přesně na prahu 0.01 a ±1e-3.
3. **Afinní invariance**: posun a škálování vstupu → orgány se posunou a škálují stejně (ověřuje bod 3.2).
4. **L/R symetrie**: prohození levých a pravých bodů nemění výsledek.
5. `conf_chest` i `conf_brain` ∈ [0, 1] pro všechny vstupy včetně krajních (všechny `c = 0`, všechny
   `c = 1`, hodnoty těsně nad 1 z FP16).

**`tests/test_models.py`**: `OrganLayer` vs `compute_organs` (< 1e-5 FP32); `state_dict` beze změny
klíčů; `num_parameters`/`mac_count` beze změny; tvary pro 256/320/384.

**`tests/test_export.py`**: ONNX má 3 výstupy, parita `organs` ≤ 1e-3; ncnn `.param` obsahuje jen
povolené vrstvy; FP16 parita ncnn ≤ 0.02 stride jednotky / 0.01 conf; `ExportModel(with_organs=False)`
vytvoří bitově stejný `person`/`pose` jako dnes.

**`tests/test_runtime.py`**: `decode_organs_selected` + `to_frame_poses` s letterbox transformací
(střed, poloměr); cesta „graph“ == cesta „host“ (< 0.05 px); starý manifest bez `organs` → `organ_source
== "host"`; pyramid a OKS-NMS orgány zachovají.

**`tests/test_live_app.py`**: `draw_organs` pro obě velikostní větve, `confidence_achieved` true/false,
`overlay_mode` všech tří hodnot (pixelová kontrola, že v režimu `organs` nejsou na snímku barvy kostry);
`coerce` nových nastavení; preset „Organs“ se aplikuje; výchozí nastavení z `config.json` je
`overlay_mode == "organs"`. Stávající testy, které kontrolují vykreslení kostry nebo bodů, si nastaví
`overlay_mode="keypoints"` explicitně, protože výchozí režim se mění.

---

## 7. Akceptační kritéria

1. `person` a `pose` jsou po přeexportu bitově shodné s dnešními exporty (stejný checkpoint, FP32).
2. Mozek a hruď z grafu odpovídají serverovému výpočtu nad stejnými body (test 6.1).
3. ncnn FP16 na Pi 5 (u_b @320, 3 vlákna): přírůstek latence sítě **< 0.3 ms**, v `bench_pi_e2e.py`
   end-to-end bez měřitelného zhoršení FPS.
4. Live View: přepínač Keypoints / Organs / Both a klávesa `o` fungují. Kruhy vypadají jako ve FreeSentry
   (bílý okraj, barva podle konfidence, tenký kruh pod prahem).
5. Starý manifest a bench graf fungují (orgány z hostu).
6. Všechny testy projdou za < 60 s na CPU (CONTRACTS §13).

---

## 8. Rizika

| riziko | dopad | opatření |
|---|---|---|
| pnnx převede část operací na `pnnx.*` / nepodporovanou vrstvu | export nepoběží v ncnn | vektorizovaná varianta z 3.4 jen se základními op; allowlist test; záloha: `matmul` → `mul+sum` |
| FP16 přetečení/nepřesnost | špatné poloměry na Pi | relativní stride jednotky (3.2), hradlo s gain 1e4 |
| Latence mnoha malých vrstev nad N=3024 (384 px) | pomalejší síť | měření v F6; případně spojit do menšího počtu vrstev |
| Ucho z profilu má `present` nízké, ale ≥ 0.01 → plná větev použije souřadnici „nepřítomného“ ucha | mozek posunutý u profilu | vědomě převzato 1:1 ze serveru (rozhodnutí O1) |
| Hruď nemá záložní větev (server také ne) | při zakrytých bocích nespolehlivý radius | převzato 1:1; konfidence to sníží |

---

## 9. Rozhodnuto

- **O1 – ano:** práh 0.01 pro volbu ušní větve mozku zůstává shodný se serverem, i když present
  pravděpodobnosti XTX4 jsou kalibrované jinak. Případná pozdější změna znamená jinou konstantu
  a přeexport.
- **O2 – normalizovat:** `conf_chest` je vážený průměr dělený 7 a výsledek je oříznutý na [0, 1]
  (bod 1.2). Výchozí práh hrudi je 0.143, což odpovídá serverovým 0.20.
- **O3 – `organs`:** Live View po spuštění zobrazuje orgány. Klíčové body se zapnou přepínačem,
  klávesou `o` nebo presetem „Keypoints only“ / „Keypoints + organs“ (bod 4.5).
- **O4 – zatím jako ve FreeSentry:** zvláštnost serveru `r·os` u malých kruhů se přebírá beze změny
  (bod 1.3). Opravu lze provést později jednou změnou v `draw_organs`.
- **O5 – mimo rozsah:** napojení XTX4 přímo do `server/` místo YOLO není součástí tohoto plánu.
  `organs_to_dict` jen připravuje data ve tvaru, který server používá.

Všechny otázky jsou rozhodnuté.

---

## 11. Stav implementace (2026-10-17)

Fáze F0–F6 jsou hotové. Odchylky od návrhu a zjištění:

- **Tvar výpočtu v grafu (3.4):** pnnx/ncnn chybně převádí redukce přes poslední osu 4D tenzorů (výstup
  měl 13 sloupců místo 8). `OrganLayer` proto pracuje jen se 3D tenzory: souřadnice x a y počítá zvlášť
  (dva `MatMul` s konstantní maticí `9×11`, která obsahuje středy i rozdílové vektory pro poloměry).
  V ncnn grafu jsou jen vrstvy `Gemm`, `BinaryOp`, `UnaryOp`, `Reduction`, `Clip`, `Eltwise`, `Crop`,
  `Slice`, `Squeeze`, `Concat`. Export nově selže, pokud pnnx nechá v grafu operátor `pnnx.*`.
- **Hradlo větve mozku (3.3):** `g(c) = clamp((c − 0.01)·1e4 + 1, 0, 1)`, takže pro `c ≥ 0.01` platí
  přesně jako `>=` v serveru a přechod leží v pásmu 1e-4 pod prahem. V tomto pásmu strmost hradla
  zesiluje zaokrouhlení float32. Kontroly parity (ONNX, deploy) proto kotvy v pásmu vynechávají
  (`xtx4/utils/organs.py::gate_ramp_rows`); v reálných exportech jich bylo 0.
- **Parita přeexportovaných modelů** (`models/*/deploy*/parity.json`): orgány graf ↔ hostová reference
  0,0000 px, ONNX ≤ 0,0004 px, ncnn FP16 p99 ≤ 0,66 px (u_c @320 má jeden extrém 4,8 px – pravděpodobně
  přepnutá větev mozku kvůli FP16 konfidenci u prahu 0,01). Klíčové body ve FP16 nesplňují toleranci
  `ncnn_xy_p95_px` stejně jako před změnou (čísla jsou shodná na tisíciny), proto byl export spuštěn
  s `--no-strict`. Původní nasazené složky jsou zálohované v `D:/xtx4_runs/_deploy_backup_20261017/`.
- **Latence (7.3):** na desktopu (u_b @320, 3 vlákna, FP16) přidá vyhodnocení orgánového výstupu
  +0,14 až +0,45 ms (u_c +0,15 ms). **Na Raspberry Pi 5 zatím neměřeno** – akceptační kritérium
  < 0,3 ms je třeba ověřit skriptem `scripts/bench_pi_e2e.py`. Runner počítá orgány jen tehdy, když
  je o ně požádáno (`compute_organs`), takže `forward()` bez orgánů ncnn orgánové vrstvy vůbec nespouští.
- **API:** `xtx4.api.Pose.organs` a `to_dict()["organs"]` ve tvaru serveru; `model.predict(..., with_organs=True)`.
