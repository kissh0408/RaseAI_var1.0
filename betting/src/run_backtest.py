"""CLI: run betting walk-forward backtest (on-the-fly fusion, win only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from betting.src.backtest import load_scored_odds_frame, run_walkforward_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Benter betting backtest")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    args = parser.parse_args()
    df = load_scored_odds_frame(args.scores, args.features)
    report = run_walkforward_backtest(df)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
