# XTX4 – rešeršní poznámky (stav k 11. 10. 2026)

Plný rozbor a zdroje jsou v `implementation_plan_fable.md` §3 a §14. Tento soubor eviduje pozdější ověření.

| Datum | Co ověřit | Stav |
|---|---|---|
| 2026-10-11 | YOLO27 vydání (docs říkají „coming soon“) | nevydáno; po vydání zopakovat referenční benchmark na Pi |
| 2026-10-11 | ER-Pose kód (arXiv 2603.08681) | nezveřejněn; implementace Pose-TAL podle článku + YOLO26 top-k 7→1 |
| 2026-10-11 | CPUBone kód (Apache-2.0) | dostupný: https://github.com/altair199797/CPUBone (jen klasifikace; princip hustých/grouped konvolucí převzat) |
| 2026-10-11 | ncnn výchozí FP16 na ARMv8.2 | potvrzeno v `src/option.cpp`: fp16 storage/arithmetic = true; benchncnn totéž |
| 2026-10-11 | Vulkan na Pi 5 | 10–20× pomalejší než CPU (Q-engineering), nepoužívat |
| 2026-10-14 | Efektivita plných 3×3 vs DW sítí v ncnn na Pi 5 (plán §3.3: 70–90 vs 17–32 MMAC/ms) | změřeno vlastními grafy: u_b 65 MMAC/ms (3 vl.) / 40 (1 vl.), XTX3-u topologie 30 / 17 – poměr 2,2× potvrzen (`reports/pi_latency.csv`) |
| 2026-10-14 | FP16 aritmetika ncnn na Cortex-A76 | 2,4× rychlejší než FP32 (u_b 320: 11,7 vs 28,3 ms); 4 vlákna horší než 3 (p95 2×) |
| 2026-10-14 | Pi 5 bez chlazení pod trvalou zátěží | 80–84 °C, throttling hlášen; restart zařízení při 4 vláknech (zdroj nerozpoznán jako 5 A) – pro nasazení chladič + 5 A zdroj + 3 vlákna |
