#!/usr/bin/env bash
# XTX3 live test - launcher (Linux / macOS). Extra arguments are passed to main.py, e.g.:
#   ./start.sh --network xtx3-u --size 320 --camera 0 --no-dialog
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
    echo "The Python environment is not initialised yet - running init.sh ..."
    bash ./init.sh --auto
fi

exec .venv/bin/python main.py "$@"
