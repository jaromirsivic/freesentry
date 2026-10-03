"""XTX3 live test: camera -> NCNN pose network -> skeleton overlay.

Self-contained runtime (ncnn + numpy + opencv only, no torch). The geometry and
decode code mirror ``xtx3.data.preprocess`` / ``xtx3.models.head`` and are kept in
sync by the parity tests under ``tests/``.
"""

__version__ = "1.0.0"
