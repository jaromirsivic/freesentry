#!/usr/bin/env bash
# Waits for the training chain (F3 control + u_b screening) and evaluates both best.pt
# on the full test split at 256/320/384 with the threshold sweep, with and without host OKS-NMS 0.5.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
until [ -f /d/xtx4_runs/chain_done.flag ]; do sleep 60; done
echo "chain done at $(date +%H:%M:%S)"
for run in f3_ctrl_xtx3u_50ep f4_u_b_100ep; do
  w=/d/xtx4_runs/$run/best.pt
  [ -f "$w" ] || { echo "missing $w"; continue; }
  for nms in none 0.5; do
    extra=""; tag="nonms"
    if [ "$nms" != "none" ]; then extra="--oks-nms $nms"; tag="nms05"; fi
    $PY scripts/evaluate.py --model-kind xtx4 --weights "$w" --dataset D:/xtxtraining --split test --sizes 256 320 384 --device cuda $extra --out "reports/${run}_full_${tag}.json" --no-summary 2>&1 | grep -E "AP=|Report written|Error|Traceback"
  done
done
echo "post-chain evaluation finished at $(date +%H:%M:%S)"
