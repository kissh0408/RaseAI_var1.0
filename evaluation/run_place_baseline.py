"""CLI: compare model-top1 vs favorite place ROI on OOS TEST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.market_baseline import build_win_odds_lookup_from_df
from evaluation.odds_loader import attach_odds_from_se_parquet
from evaluation.place_baseline import compute_place_baseline_oos, write_place_baseline_report
from evaluation.place_payout_loader import build_place_payout_lookup


def load_scores_with_dates(scores_path: Path, features_path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_num" not in scores.columns and "horse_number" in scores.columns:
        scores["horse_num"] = scores["horse_number"]
    if "race_date" in scores.columns:
        return scores
    features = pd.read_parquet(features_path, columns=["race_id", "horse_num", "race_date", "finish_rank"])
    features["race_id"] = features["race_id"].astype(str)
    return scores.merge(features, on=["race_id", "horse_num"], how="inner")


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS place ROI baseline")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet",
    )
    parser.add_argument(
        "--hr-parquet",
        type=Path,
        default=ROOT / "pure_rank" / "data" / "01_preprocessed" / "HR_preprocessed.parquet",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        default=ROOT / "common" / "data" / "output" / "race_hr",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "evaluation" / "reports" / "place_baseline_oos.json",
    )
    args = parser.parse_args()

    df = load_scores_with_dates(args.scores, args.features)
    if "odds" not in df.columns:
        df = attach_odds_from_se_parquet(df)
    win_odds_lookup = build_win_odds_lookup_from_df(df)
    place_lookup = build_place_payout_lookup(hr_parquet=args.hr_parquet, hr_dir=args.hr_dir)
    report = compute_place_baseline_oos(
        df,
        win_odds_lookup=win_odds_lookup,
        place_lookup=place_lookup,
    )
    write_place_baseline_report(report, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
