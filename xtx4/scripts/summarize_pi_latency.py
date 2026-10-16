"""Summarise ``bench_pi.py`` / ``bench_pi_e2e.py`` CSVs into Markdown tables.

    python scripts/summarize_pi_latency.py reports/pi_latency.csv --e2e reports/pi_latency_e2e.csv

The network-only table lists p50/p95 (ms) per graph and thread count (FP16 rows
unless ``--fp16 off`` rows are present, which are shown separately). The
end-to-end table lists the pipeline split (pre / infer / post) per size, thread
count and NMS setting.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def network_table(rows: list[dict[str, str]]) -> str:
    threads = sorted({int(r["threads"]) for r in rows})
    by_key: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_key[(r["graph"], r["fp16"])][int(r["threads"])] = r
    lines = ["| graph | size | GMAC | fp16 | " + " | ".join(f"thr {t} p50 / p95" for t in threads) + " | temp after |",
             "|---|---|---|---|" + "---|" * len(threads) + "---|"]
    for (graph, fp16), per_thread in sorted(by_key.items(), key=lambda kv: (kv[0][1] != "on", kv[0][0])):
        any_row = next(iter(per_thread.values()))
        gmac = any_row.get("gmac") or ""
        try:
            gmac = f"{float(gmac):.2f}"
        except ValueError:
            pass
        cells = []
        for t in threads:
            r = per_thread.get(t)
            cells.append(f"{float(r['p50_ms']):.1f} / {float(r['p95_ms']):.1f}" if r else "-")
        temps = [per_thread[t].get("temp_after", "") for t in threads if t in per_thread]
        lines.append(f"| {graph} | {any_row['size']} | {gmac} | {fp16} | " + " | ".join(cells) + f" | {temps[-1] if temps else ''} |")
    return "\n".join(lines)


def e2e_table(rows: list[dict[str, str]]) -> str:
    lines = ["| size | thr | fp16 | NMS | total p50 | p95 | pre | infer | post | wall p95 | fps | poses/frame | temp after |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (int(r["size"]), int(r["threads"]), r["fp16"], r["nms"])):
        lines.append(
            f"| {r['size']} | {r['threads']} | {r['fp16']} | {r['nms']} | {float(r['total_p50']):.1f} | {float(r['total_p95']):.1f} | "
            f"{float(r['pre_p50']):.2f} | {float(r['infer_p50']):.1f} | {float(r['post_p50']):.2f} | {float(r['wall_p95']):.1f} | "
            f"{float(r['fps_mean']):.1f} | {float(r['poses_per_frame']):.2f} | {r.get('temp_after', '')} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("network_csv", nargs="?", default=None)
    parser.add_argument("--e2e", default=None)
    args = parser.parse_args()
    if args.network_csv:
        rows = _rows(Path(args.network_csv))
        print(f"### Network only ({len(rows)} rows, host {rows[0]['host'] if rows else '?'})\n")
        print(network_table(rows))
        print()
    if args.e2e:
        rows = _rows(Path(args.e2e))
        print(f"### End-to-end ({len(rows)} rows)\n")
        print(e2e_table(rows))


if __name__ == "__main__":
    main()
