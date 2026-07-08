#!/usr/bin/env python3
"""End-to-end Benter pipeline: export scores -> fit fusion -> backtest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "pure_rank/src/export_scores.py", "--split", "all"])
    run([py, "evaluation/run_market_baseline.py"])
    run([py, "prob_fusion/src/run_fit.py"])
    run([py, "betting/src/run_backtest.py"])
    print("Done. See evaluation/reports/ for gate results.")


if __name__ == "__main__":
    main()
