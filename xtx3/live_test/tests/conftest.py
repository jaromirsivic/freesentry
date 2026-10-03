"""Make ``livetest`` (and, for parity tests, the training package ``xtx3``) importable."""

from __future__ import annotations

import sys
from pathlib import Path

LIVE_TEST_DIR = Path(__file__).resolve().parent.parent
XTX3_ROOT = LIVE_TEST_DIR.parent
EXPORT_DIR = XTX3_ROOT / "runs" / "xtx3-u" / "export"

for path in (LIVE_TEST_DIR, XTX3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
