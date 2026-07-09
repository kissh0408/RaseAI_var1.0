"""market_leak_diagnostic: 本番特徴量に確定オッズ由来の市場情報列を追加する。

本番 pure_rank/data/02_features/features_v39_course_slim.parquet は変更しない。
出力は pure_rank/experiments/market_leak_diagnostic/data/ 配下のみ。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
BASE_FEATURES = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
ODDS_DIR = ROOT / "common" / "data" / "output" / "odds"
OUT_PATH = EXP_DIR / "data" / "features_market_leak.parquet"


def _load_win_odds(years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = ODDS_DIR / f"WinOdds_{year}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, dtype={"race_id": str})
        frames.append(df[["race_id", "horse_num", "odds", "popularity"]])
    if not frames:
        raise FileNotFoundError(f"WinOdds_*.csv が見つかりません: {ODDS_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["race_id", "horse_num"], keep="last")
    return out


def _attach_market_log_odds(df: pd.DataFrame, odds_col: str = "exp_win_odds") -> pd.Series:
    """proportional 法: q_i = (1/odds_i) / sum(1/odds_j)（prob_fusion と同一式）。"""

    def _per_race(group: pd.Series) -> pd.Series:
        odds = group.to_numpy(dtype=float)
        raw = np.where(odds > 1.0, 1.0 / odds, 0.0)
        total = raw.sum()
        if total <= 0:
            q = np.full(len(odds), 1.0 / len(odds))
        else:
            q = raw / total
        return pd.Series(np.log(np.clip(q, 1e-12, None)), index=group.index)

    return df.groupby("race_id", sort=False)[odds_col].transform(_per_race)


def build_experiment_features() -> Path:
    print(f"Loading base features: {BASE_FEATURES}")
    base = pd.read_parquet(BASE_FEATURES)
    base["race_id"] = base["race_id"].astype(str)
    print(f"  rows={len(base):,}")

    print(f"Loading WinOdds CSVs from: {ODDS_DIR}")
    odds = _load_win_odds(range(2015, 2027))
    odds["race_id"] = odds["race_id"].astype(str)
    print(f"  odds rows={len(odds):,}")

    merged = base.merge(odds, on=["race_id", "horse_num"], how="left")
    n_missing = merged["odds"].isna().sum()
    print(f"  missing odds after merge: {n_missing:,} / {len(merged):,}")

    merged["exp_win_odds"] = merged["odds"].astype(float)
    merged["exp_ln_odds"] = np.log(merged["exp_win_odds"].clip(lower=1.01))
    merged["exp_popularity"] = merged["popularity"].astype(float)
    merged["exp_market_log_odds"] = _attach_market_log_odds(merged, odds_col="exp_win_odds")
    merged.loc[merged["exp_win_odds"].isna(), "exp_market_log_odds"] = np.nan
    merged = merged.drop(columns=["odds", "popularity"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH} ({len(merged):,} rows, {merged.shape[1]} cols)")
    return OUT_PATH


if __name__ == "__main__":
    build_experiment_features()
