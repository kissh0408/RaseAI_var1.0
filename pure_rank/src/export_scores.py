"""Shim: score export moved to pure_rank/analysis/export_scores.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "export_scores.py"

if __name__ == "__main__":
    sys.argv[0] = str(_ANALYSIS)
    runpy.run_path(str(_ANALYSIS), run_name="__main__")
else:
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_scores", _ANALYSIS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    export_scores = mod.export_scores
