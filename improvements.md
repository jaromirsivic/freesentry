# Doporučení po revizi serveru

## Rozsah a závěr

Prošel jsem staticky všech 39 souborů `server/*.py` (včetně pomocných modulů a benchmarku). Nezasahoval jsem do aplikačního kódu. Největší rizika nejsou drobné mikrooptimalizace, ale bezpečný provoz motorů, neautorizované administrační API a zbytečná práce ve video cestě. Na Raspberry Pi by jejich odstranění přineslo podstatně větší efekt než ladění jednotlivých výrazů v Pythonu.

Nejdřív bych řešil body P0. Potom bych změřil skutečný FPS/latenci a implementoval P1 v pořadí: jeden výstupní obraz místo tří, odstranění 5ms pollingu, omezení kopírování a validace konfigurace.

Priorita:

- **P0** – bezpečnost, bezpečný stav hardwaru nebo prokazatelná funkční chyba.
- **P1** – výrazný dopad na FPS, odezvu, RAM nebo stabilitu na Raspberry Pi.
- **P2** – návrh, údržba a čitelnost.

## P0 – bezpečnost a bezpečný stav hardwaru

1. **Uzamknout celé administrativní API.** `server/main.py:232-236` povoluje všechny CORS originy a API nemá žádnou autentizaci ani autorizaci. Přitom endpointy řídí motory, GPIO, Wi-Fi, datum, časové pásmo, restart systému a zapisují kompletní konfiguraci (`restapimotors.py`, `restapimanualcontrol.py`, `restapiosmanagement.py`, `restapisettings.py`). CORS není síťová ochrana a samostatně nechrání před přímým HTTP požadavkem. API má být dostupné jen z lokální sítě/VPN, s autentizací a rolemi (alespoň viewer/operator/admin), CSRF ochranou pro cookie session a explicitním allowlistem originů. Neprivilegovaný webový proces by neměl přímo disponovat oprávněním k restartu, Wi-Fi ani GPIO.

2. **Odstranit `exec()` nad hodnotou z konfigurace.** `server/main.py:62-84` provádí `exec(script)`, přičemž `/api/settings` přijímá a vrací kompletní nastavení. V kombinaci s bodem 1 je to vzdálené spuštění libovolného Pythonu při příštím startu služby; i po zavedení přihlášení zůstává příliš silné oprávnění. Nahradit ho výhradně předem registrovanými akcemi nebo podepsanými, neměnnými skripty v aplikaci. Pokud je rozšiřitelnost nutná, spouštět izolovaný proces s minimálními právy a se striktním rozhraním, nikdy ne `exec` v procesu serveru.

3. **Zavést dead-man watchdog pro každý motor.** `server/restapimanualcontrol.py:214-310` nastavuje cílovou rychlost, ale při odpojení klienta poslední rychlost zůstává aktivní. To je nebezpečné i bez útoku. Každý příkaz z ručního ovládání má nést čas/sekvenci a po krátkém TTL (např. 150–300 ms) musí centrální motorový scheduler nastavit všechny ručně řízené motory na nulu. Musí existovat nezávislý hardwarový emergency stop a koncové spínače/proudová ochrana; softwarově odhadovaná `position` v `motorlinearactuator.py` nestačí jako bezpečnostní limit.

4. **Při resetu či změně konfigurace prokazatelně vypnout staré výstupy.** `MotorsController._initialize_motors()` v `server/motorscontroller.py:118-175` zahodí slovníky motorů, aniž by předtím všem dříve používaným pinům nastavil nulu a bezpečně je uvolnil. `J8.reset()` navíc při již inicializovaném J8 nic neudělá (`server/j8.py:79-91`). Změna nebo odstranění motoru tak může nechat předchozí PWM v aktivním stavu. Implementovat transakční reset: zastavit scheduler, potvrdit nulové PWM, vypnout enable piny, zavřít/uvolnit staré objekty, teprve potom vytvořit nové. Při jakékoli chybě zůstat ve fail-safe stavu, ne v polovičně rekonfigurovaném stavu.

5. **Nedovolit ruční override libovolného fyzického pinu.** `start_manual_override()` v `server/motorscontroller.py:205-236` ověřuje jen rozsah indexu J8. Uživatel tak může přes API zkusit řídit I2C/SPI/UART či jiné nebezpečné piny uvedené v `J8`. Povolit jen explicitní diagnostický allowlist motorových PWM pinů, vyžadovat roli servisního uživatele, časový limit override a vždy dostupné automatické vypnutí.

6. **Opravit zacházení s PWM frekvencí 0 a hodnotami NaN/inf.** `MotorLinearActuator` nastavuje enable pinům `pwm_frequency = 0` (`server/motorlinearactuator.py:14-23`), ale hardware-PWM větev v `PWMWrapper.frequency` dělí frekvencí (`server/pin.py:120-128`). To může skončit dělením nulou nebo neplatným nastavením ovladače. Pro enable pin použít samostatný digitální výstup, nikoli „PWM s frekvencí 0“. Na hranicích API i v `Pin.value` ověřit konečnost a interval hodnot; Pydantic modely nyní rychlosti neomezují.

7. **Oddělit systémová oprávnění od webového procesu.** Volání `sudo nmcli`, `timedatectl`, `reboot` a změny času jsou dostupné z aplikačního procesu (`main.py:89-147`, `osmanagement.py`). Použít úzký privilegovaný helper/service s pevně definovanými příkazy a politikou oprávnění. Pro `nmcli` přidat timeout a `sudo -n`, aby server nečekal na heslo. Wi-Fi připojení upravovat idempotentně (existující profil změnit/aktivovat, ne při každém startu přidávat nový); heslo neposílat v argumentech procesu, kde je lokálně viditelné.

8. **Validovat časové pásmo proti skutečnému allowlistu.** Linuxová fallback větev v `server/osmanagement.py:132-151` skládá cestu z uživatelského `timezone_name` a kontroluje pouze existenci. Použít `zoneinfo.available_timezones()` nebo pevný seznam, odmítnout cestovní segmenty a nepředávat cestu do `ln`. API současně nesmí vracet Wi-Fi hesla a interní cesty: `/api/settings` a `/api/system/platform-info` dnes zveřejňují citlivé nastavení, uživatele i home directory.

9. **Chyby nevracet klientovi v plném znění.** Většina routerů předává `detail=str(e)` a současně používá `print`. Detail může odhalit cesty, konfiguraci a stav hardwaru. Klientovi vracet stabilní chybový kód a obecnou zprávu, serverově logovat korelační ID a traceback přes `logger.exception` (bez hesel a tokenů).

## P0 – nalezené funkční chyby

10. **AI engagement zahazuje `aiSetup` z neměnného snapshotu.** `get_settings_sync()` vrací `FrozenDict`, tedy `Mapping`, ne vestavěný `dict` (`server/settingscontroller.py:226-238`). `server/camera.py:587-615` pak podmiňuje použití nastavení `isinstance(settings, dict)`, takže do `AIAgent.engage()` a `build_engagement_snapshot()` předá `{}`. Konfigurované prahy, orgány, reticle a strategie se v hlavním procesu ignorují. Opravit typy na `Mapping[str, Any]` a tam, kde je nutná mutace/pickling, explicitně vytvořit obyčejný `dict`; stejný anti-pattern je i v `aiagent.py:643-674`.

11. **Exit strategie neumí přejít do ukončeného stavu a nemusí zastavit své motory.** Stavový automat nastaví `EXIT_STRATEGY_UNDER_EXECUTION` (`server/aiagent.py:456-479`), ale nikde nepřejde na `EXIT_STRATEGY_EXECUTED`. Větev, která má exit motory vypnout, proto není dosažitelná (`server/aiagent.py:606-609`); `exitStrategyDuration` se vůbec nepoužívá. Doplnit jednoznačný přechod podle monotónního času, testy pro start/stop a fail-safe stop při chybě inference či ztrátě kamery.

12. **Nesprávný zdroj času může rozbít pohyb motorů.** `server/motorlinearactuator.py` používá `time.time()` pro výpočet `delta`, zatímco server nabízí změnu systémového času. Skok hodin zpět zastaví integraci, skok vpřed vytvoří nesmyslně velký krok odhadované pozice. Pro všechny intervaly, timeouty a regulaci používat `time.monotonic()`/`perf_counter()`; wall-clock čas ponechat jen pro audit a UI.

13. **Nevalidované camera nastavení může ukončit worker.** `CameraUpdateRequest` nemá limity pro rozměry, FPS, crop, rotaci ani stretch (`server/restapicameras.py:20-58`). Pokud je `stretch_enabled=True` a rozměr je 0 či extrémní, `_crop_and_resize()` zavolá `cv2.resize` mimo lokální `try` blok (`server/cameraworker.py:787-815`) a vnější catch ukončí celý worker. Použít striktní Pydantic model: konečné hodnoty, povolené rotace, crop 0..1 s nenulovou výslednou oblastí, maximální rozměr/pixely a FPS podporované zařízením. Neplatná konfigurace má vrátit 422 a nikdy neshodit pipeline.

## P1 – video, AI a výkon na Raspberry Pi

14. **AI stream zbytečně vytváří tři JPEGy na každý snímek.** `Camera._touch_access()` pro režim 3 aktivuje raw, masked i AI (`server/camera.py:241-262`). Worker pak v jednom cyklu JPEG-enkóduje raw, masked a AI (`server/cameraworker.py:596-768`), přestože dokumentace režimu AI fallback raw/masked nepoužívá. Pro AI režim požadovat jen AI výstup (masku stále vytvořit jako vstup pro AI, ale neenkódovat ji); fallback definovat explicitně. To je pravděpodobně největší okamžitá úspora CPU.

15. **Nahradit 5ms polling a opakované IPC příkazy eventově řízeným mechanismem.** HTTP generátor běží po 5 ms (`restapicameras.py:574-688`), každým průchodem volá synchronní `get_stream_frame`, které posílá `update_demand`; proxy thread totéž posílá po 5 ms (`camera.py:485-529`, `camerascontroller.py:467-492`). To produkuje stovky Pipe zpráv za sekundu na stream, spotřebovává CPU a může vytlačit užitečné příkazy. Demand/quality posílat jen při změně a periodické expiraci řešit jedním časovačem. Čekání na nový snímek propojit s událostí/condition a uklidit při odpojení klienta.

16. **Neblokovat asyncio event loop synchronním čekáním na JPEG.** `generate_camera_frames()` je async generator, ale `camera.get_stream_frame()` v něm může dělat spin-wait a až 30ms blokující čekání (`camera.py:400-443`, `restapicameras.py:590`). Jeden pomalý stream tedy zdržuje ostatní API a streamy. Ideální je asynchronní odběr z události; přechodově lze blokující část bezpečně přesunout do dedikovaného vlákna s omezeným počtem workerů. Počet souběžných MJPEG klientů musí být omezen a sdílet jeden producer.

17. **Nevytvářet plný černý obraz pro každou AI detekci.** `server/camera.py:610` alokuje `np.zeros((height, width, 3))`, pouze aby `AIAgent` znal rozměr obrázku. U 1920×1080 jde o ~6 MB na každý pose snímek navíc v hlavním procesu. Změnit rozhraní `engage`/`_get_reticle_position` tak, aby přijalo `image_size: tuple[int, int]`, případně malý objekt se `shape`. Tím odpadne alokace, nulování paměti a tlak na GC.

18. **Omezit kopírování snímků a paměť sdílené transportní vrstvy.** Tři triple-buffery po 2 MiB v `server/cameraframetransport.py:75-76` vyžadují nejméně 18 MiB shared memory ještě před dalšími kopiemi `buf -> bytes -> shared memory -> bytes -> ASGI`. Na Pi je to citelné. Alokovat jen aktivně požadované režimy, velikost odvodit z reálně povoleného rozlišení/quality a v případě překročení snímek řízeně zmenšit či zahodit, ne jen dokola logovat chybu. Pro USB kamery, které dodávají MJPEG, zvažte přeposílání původního JPEG nebo hardwarový encoder/libcamera místo reenkódování každého frame.

19. **Pro AI nastavit rozpočet FPS a rozlišení, ne „co nejrychlejší“ smyčku.** Worker má pevné `EPSILON_DELAY = 5 ms` a inference provádí při každém dostupném snímku. Na Pi má AI běžet s konfigurovatelným maximem (např. 5–10 FPS), na zmenšeném inference vstupu, s opětovným použitím posledního výsledku/overlay mezi inference snímky. Samostatně měřit capture, preprocessing, inference a každý JPEG encode. `benchmark_ncnn_pose.py` je dobrý začátek, ale sám správně uvádí, že neměří postprocess ani celý produkční řetězec.

20. **Nenačítat a nezkoušet všechny kamery při startu.** `CamerasController.reset()` vytváří 14 proxy objektů (`server/camerascontroller.py:129-183`); konstruktor `Camera` hned volá `get_supported_resolutions()` a `get_properties()` (`camera.py:93-113`). Pro osm CV2 indexů to otevírá zařízení a opakovaně zkouší dlouhý seznam až 8K rozlišení (`cameradevice.py:73-110`). Na Pi to prodlužuje start, může kolidovat s kamerou a nemá smysl bez zájmu uživatele. Capability probing provádět lazy po výběru zařízení, cacheovat výsledek a omezit kandidáty podle skutečné enumerace `/dev/video*`.

21. **Odstranit plné deep-copy nastavení z horkých cest.** `get_settings()` vždy kopíruje celý strom a „async“ funkce přitom provádějí synchronní lock/IO (`settingscontroller.py:115-141`, `240-278`). Ruční ovládání ho volá v každém periodickém HTTP požadavku (`restapimanualcontrol.py:232`). Horké cesty mají číst pouze potřebný neměnný snapshot/konkrétní sekci, nikoli celý JSON. Zápisy ponechat serializované a mimo event loop. Verzi konfigurace šířit atomicky, aby se změny do workeru posílaly jen jednou.

22. **Motor scheduler spouštět podle deadline, ne 1000× za sekundu.** `MotorsController.run()` bere zámek a iteruje motory každou 1 ms, zatímco samotný motor nic nedělá před `motor_frame = 10 ms` (`motorscontroller.py:289-307`, `motorlinearactuator.py:101-105`). Naplánovat probuzení na nejbližší potřebný okamžik (typicky 100 Hz), použít `Event.wait(timeout)` a integraci podle monotónního `dt`. To sníží probouzení i contention, zlepší stabilitu rampy při zatížení a umožní přesně měřit jitter.

23. **Throttle a strukturovat logování v opakovaných smyčkách.** V capture, encode, inference, streamu i motorové smyčce jsou `print` v širokých `except Exception`. Při odpojené kameře či chybné konfiguraci se mohou logovat stovky řádků za sekundu, což samo sníží FPS a zaplní disk. Použít `logging`, stavové/rate-limited hlášení („chyba se změnila“, „jednou za 30 s“) a čítače metrik.

24. **Lazy importy a balíčky upravit pro ARM.** Server při diagnostice importuje `YOLOModels`, jehož modul bezpodmínečně importuje `ultralytics` a zkouší `torch` (`yolomodels.py:7-12`). Na Raspberry Pi rozdělit backendy na extras, lazy-loadovat až při prvním AI použití a připravit konkrétní ARM/NCNN build. Zvažte `opencv-python-headless` místo GUI varianty, pinujte kompatibilní verze pro cílovou architekturu a ověřte, zda `requires-python = ">=3.14"` je na zamýšleném Raspberry Pi OS reálně distribuovatelný.

## P1/P2 – návrh a stabilita

25. **Zpřesnit hranici „jedna aktivní kamera“.** `CamerasController` má záměrně jen jeden worker a jednu `_active_camera` (`server/camerascontroller.py:375-423`). Dva klienti, kteří chtějí současně různé kamery, si ji navzájem přepnou a zneplatní stream token. Pokud je to produktové omezení, API/UI má vracet jasný stav „camera busy“ a vlastníka; pokud ne, architektura potřebuje worker na kameru nebo plánovač sdílených capture relací. Současné chování je závod závislý na pořadí HTTP požadavků.

26. **Zjednodušit životní cyklus threadů a singletonů.** `Camera` volá přímo `threading.Thread.__init__()` pro opakované spuštění (`camera.py:305-320`), `CamerasController` i `J8` jsou globální singletony a `J8.__init__()` znovu resetuje instanci při každém `J8()` (`j8.py:8-34`). To je křehké při reloadu, testech i více FastAPI aplikacích. Použít kompoziční worker objekty s jednorázovým lifecycle, vlastnictví uložit do `app.state` a explicitně injektovat závislosti. `J8` nemá dědit z `list`, když skutečná data drží v `_pins`.

27. **Odlišit konfiguraci, runtime stav a příkazy.** Routery někdy ukládají raw `dict`, jindy rovnou resetují controller; například obecné `/api/settings` runtime vůbec nerekonfiguruje, zatímco jednotlivé endpointy ano. Vytvořit jednotný `Settings` Pydantic model, explicitní `ApplySettings` příkaz a immutable versionovaný snapshot. Validace dnes pokrývá hlavně histogram a dvě datumová pole (`settingsschema.py`), zatímco AI, kamery, Wi-Fi a ovládání motorů přijímají prakticky libovolné dictionary.

28. **Použít datové modely konzistentně.** `restapiaisetup.py` definuje řadu Pydantic tříd, ale `SaveAISetupRequest.aiSetup` je zase nevalidovaný `dict`; podobně jsou raw dictionary ve většině routerů. Nahradit je striktními modely s `extra="forbid"`, `Field(ge=..., le=...)`, `default_factory` pro listy/slovníky a dokumentovanými jednotkami. Zvlášť validovat konečnost čísel, duplicitní názvy motorů/piny a konflikty přiřazení kamer.

29. **Chyby workeru nepohlcovat, ale předávat jako stav.** Široké `except Exception: pass` maskují poruchy v `camera.py`, `camerascontroller.py`, `cameraworker.py`, `j8.py` i routerech. Worker po deseti restartech přestane startovat (`_MAX_RESTART_ATTEMPTS`) bez dobře viditelného stavu pro UI. Definovat typované doménové chyby, stavovou diagnostiku a operátorovi vystavit poslední příčinu, počet restartů, aktivní konfiguraci a akci „bezpečně restartovat“.

30. **Zpřesnit práci s datem a timezone.** `compare_datetimes()` v `isodatetime.py` při kombinaci aware a naive hodnot zahodí timezone informaci místo jednoznačné normalizace. Vyžadovat timezone v uložených ISO časech (nejlépe UTC), převést na UTC a používat monotónní čas pro intervaly. `get_timezone_offset_minutes()` má počítat skutečný `utcoffset()`, ne rozdíl dvou nezávisle načtených časů.

31. **Vyčistit nedokončený a mrtvý kód.** `common.py` drží zakomentovaný multiprocessing timeout kód a nepoužité importy; `cameramotion.py` má kvalitní, ale produkčně nepoužitý subsystém; `Frame.copy()` už v nové JPEG architektuře není horká cesta. Buď jasně integrovat a měřit, nebo odstranit/oddělit za feature flag. Menší a jednoúčelové moduly budou na Pi i pro údržbu předvídatelnější.

32. **Rozdělit velké moduly podle odpovědnosti.** `cameraworker.py` míchá IPC protokol, životní cyklus zařízení, capture, transformace, inference i encoding; `restapicameras.py` míchá CRUD, stream a watchdog. Rozdělit na malé testovatelné komponenty (command/state machine, frame pipeline, encoder, stream relay, watchdog). Přínos je hlavně čitelnost a možnost profilovat či testovat každou část izolovaně.

33. **Doplnit provozní limity HTTP.** Validovat `mode` a `quality` ve stream endpointu (nyní se quality skrytě snižuje o 10), nastavit limit počtu stream klientů, timeout neaktivních klientů, maximální velikost requestu a rate limit pro motorové/administrativní operace. Po odpojení okamžitě snížit demand a případně zastavit kameru/inference.

## Doporučené pořadí implementace

1. Uzamknout síťový přístup a role, zrušit `exec`, zúžit systémová oprávnění.
2. Implementovat motorový dead-man timeout, nouzové vypnutí a bezpečný reset pinů; přidat integrační testy na to, že po každé chybě jsou PWM/enable piny na nule.
3. Opravit předávání `FrozenDict` do AI a dokončit exit strategii; nejdřív unit testy těchto dvou konkrétních regresí.
4. Zavést plnou validaci konfigurace a atomické „uložit + aplikovat“.
5. Změnit video pipeline tak, aby AI režim produkoval jen nutné výstupy, a posílal demand pouze při změně.
6. Nahradit polling eventy, odstranit blokující práci z asyncio event loop a plnou alokaci placeholder snímku.
7. Teprve potom profilovat konkrétní Pi konfiguraci a ladit backend modelu, rozlišení, FPS a encoder.

## Měření a ověření po úpravách

- Měřit end-to-end p50/p95 latency snímku, capture FPS, inference FPS, encode čas pro raw/mask/AI, CPU per proces, RSS a `/dev/shm`.
- Testovat alespoň 1, 2 a limitní počet současných stream klientů; scénáře přepnutí kamery, odpojení klienta, unplug USB kamery, chybné nastavení a opakovaný restart workeru.
- Hardware-in-the-loop testy: ztráta klienta, výjimka ve workeru, změna systémového času, reset konfigurace a vypnutí procesu musí vždy ukončit pohyb motorů.
- Přidat `ruff`, formatter, `pyright`/mypy, unit testy stavových automatů a smluvní API testy do CI. Současné testy jsem v tomto prostředí nespustil: projektový interpreter selhal na oprávnění (`uv trampoline ... permission denied`) a `ruff` zde není nainstalovaný.
