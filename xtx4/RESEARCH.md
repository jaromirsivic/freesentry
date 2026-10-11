# XTX4 – rešeršní poznámky (stav k 11. 10. 2026)

Plný rozbor a zdroje jsou v `implementation_plan_fable.md` §3 a §14. Tento soubor eviduje pozdější ověření.

| Datum | Co ověřit | Stav |
|---|---|---|
| 2026-10-11 | YOLO27 vydání (docs říkají „coming soon“) | nevydáno; po vydání zopakovat referenční benchmark na Pi |
| 2026-10-11 | ER-Pose kód (arXiv 2603.08681) | nezveřejněn; implementace Pose-TAL podle článku + YOLO26 top-k 7→1 |
| 2026-10-11 | CPUBone kód (Apache-2.0) | dostupný: https://github.com/altair199797/CPUBone (jen klasifikace; princip hustých/grouped konvolucí převzat) |
| 2026-10-11 | ncnn výchozí FP16 na ARMv8.2 | potvrzeno v `src/option.cpp`: fp16 storage/arithmetic = true; benchncnn totéž |
| 2026-10-11 | Vulkan na Pi 5 | 10–20× pomalejší než CPU (Q-engineering), nepoužívat |
