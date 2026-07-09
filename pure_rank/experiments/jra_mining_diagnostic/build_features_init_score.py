"""var2.0.0 の init_score(base_margin) トリックをこちらでも再現する診断実験。

jra_tm_log_odds を特徴量としてではなく LightGBM の init_score として与え、
モデルには「JRA公式マイニング予想からの残差」だけを学習させる。
本番 pure_rank/data/02_features/features_v39_course_slim.parquet は変更しない。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
BASE_FEATURES = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
CANDIDATE_PATH = EXP_DIR / "data" / "jra_tm_candidate.parquet"
OUT_PATH = EXP_DIR / "data" / "features_jra_init_score.parquet"


def build_features_init_score() -> Path:
    print(f"Loading base features: {BASE_FEATURES}")
    base = pd.read_parquet(BASE_FEATURES)
    base["race_id"] = base["race_id"].astype(str)
    print(f"  rows={len(base):,}")

    cand = pd.read_parquet(CANDIDATE_PATH)
    cand["race_id"] = cand["race_id"].astype(str)

    merged = base.merge(
        cand[["race_id", "horse_num", "jra_tm_log_odds"]], on=["race_id", "horse_num"], how="left"
    )
    n_missing = merged["jra_tm_log_odds"].isna().sum()
    print(f"  missing jra_tm_log_odds: {n_missing:,} / {len(merged):,} ({n_missing / len(merged):.1%})")
    # TMデータ欠損レースは中立値（0 = base_marginなし）で埋める。var2.0.0 も base_margin
    # 欠損時に同等の中立フォールバックを想定（明示コードは無いためここでは0を採用）。
    merged["jra_tm_log_odds"] = merged["jra_tm_log_odds"].fillna(0.0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH} ({len(merged):,} rows, {merged.shape[1]} cols)")
    return OUT_PATH


if __name__ == "__main__":
    build_features_init_score()
