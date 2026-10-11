"""XTX4 live test: camera -> ncnn pose network -> 9-point skeleton overlay.

Torch-free: preprocessing, decode and the threaded pipeline come from
``xtx4.runtime`` (one directory up, in the xtx4 repository); nothing is
vendored. Importing this package puts the repository root on ``sys.path``
(override with the ``XTX4_ROOT`` environment variable).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "2.0.0"

LIVE_TEST_DIR = Path(__file__).resolve().parent.parent
XTX4_ROOT = Path(os.environ.get("XTX4_ROOT", LIVE_TEST_DIR.parent)).resolve()

if (XTX4_ROOT / "xtx4" / "runtime").is_dir() and str(XTX4_ROOT) not in sys.path:
    sys.path.insert(0, str(XTX4_ROOT))
