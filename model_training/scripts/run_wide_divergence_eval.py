#!/usr/bin/env python3
"""Run L1/L2 wide divergence comparison on evaluation.csv (Step 3 eval gate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.src.combo_backtest import compare_l1_l2_wide_divergence


def main() -> None:
    parser = argparse.ArgumentParser(description="L1/L2 wide divergence backtest")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=ROOT / "model_training" / "output" / "evaluation.csv",
    )
    parser.add_argument(
        "--odds-dir",
        type=Path,
        default=ROOT / "common" / "data" / "output" / "odds",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.eval_csv.exists():
        raise FileNotFoundError(f"evaluation.csv not found: {args.eval_csv}")

    import pandas as pd

    eval_df = pd.read_csv(args.eval_csv)
    result = compare_l1_l2_wide_divergence(eval_df, args.odds_dir)
    out_path = args.output or (ROOT / "pure_rank" / "data" / "02_features" / "wide_divergence_l1_l2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
