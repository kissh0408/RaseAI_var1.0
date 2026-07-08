"""CLI: compute and save market baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from evaluation.market_baseline import compute_and_save_market_baseline, load_hr_win_lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute market baseline metrics")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    parser.add_argument(
        "--hr",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "01_preprocessed" / "HR_preprocessed.parquet",
    )
    args = parser.parse_args()
    df = pd.read_parquet(args.features)
    if "horse_num" not in df.columns and "horse_number" in df.columns:
        df["horse_num"] = df["horse_number"]
    hr_lookup = load_hr_win_lookup(args.hr)
    report = compute_and_save_market_baseline(df, hr_win_lookup=hr_lookup)
    print(report)


if __name__ == "__main__":
    main()
