"""alpha_segments: build_dataset.py

fold2 OOS スコア（`pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet`）
に、セグメント判定に必要な列（features の hist_last_rank/horse_count/
track_condition_code/course_code、RA の race_condition_code）を race_id
(+horse_num) で inner merge し、`attach_odds_from_se_parquet` ->
`attach_market_q` で L2 統合変数 ln_market_q を付与する。

TEST 期間（2025+）の行もこの dataset には含めてよい（仕様書 §7 Stage 1-1）。
Stage 1/2 のスクリプト側で 2024-12-31 以前にフィルタする。

出力:
    pure_rank/experiments/alpha_segments/data/gate_dataset.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.odds_loader import attach_odds_from_se_parquet  # noqa: E402
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402

SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
RA_PATH = ROOT / "pure_rank" / "data" / "01_preprocessed" / "RA_preprocessed.parquet"
DATA_DIR = EXP_DIR / "data"
OUT_PATH = DATA_DIR / "gate_dataset.parquet"

FEATURE_COLS = [
    "race_id",
    "horse_num",
    "hist_last_rank",
    "horse_count",
    "track_condition_code",
    "course_code",
]
RA_COLS = ["race_id", "race_condition_code"]


def _prep_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    if "horse_num" in out.columns:
        out["horse_num"] = pd.to_numeric(out["horse_num"], errors="coerce").astype("Int64")
    return out


def build_alpha_segments_dataset() -> Path:
    scores = pd.read_parquet(SCORES_PATH)
    scores = _prep_id_cols(scores)
    if "horse_num" not in scores.columns and "horse_number" in scores.columns:
        scores["horse_num"] = pd.to_numeric(scores["horse_number"], errors="coerce").astype("Int64")
    n_scores_rows = len(scores)
    n_scores_races = scores["race_id"].nunique()
    print(f"scores: rows={n_scores_rows:,}, races={n_scores_races:,}")

    # race_date / finish_rank / course_code are already present in `scores`
    # (authoritative source for this experiment); only pull the segment
    # columns not already in scores to avoid duplicate-column suffixing.
    features = pd.read_parquet(FEATURES_PATH, columns=FEATURE_COLS)
    features = _prep_id_cols(features)
    features = features.drop(columns=[c for c in ("course_code",) if c in scores.columns])
    print(f"features: rows={len(features):,}, races={features['race_id'].nunique():,}")

    ra = pd.read_parquet(RA_PATH, columns=RA_COLS)
    ra = _prep_id_cols(ra)
    print(f"RA: rows={len(ra):,}, races={ra['race_id'].nunique():,}")

    merged = scores.merge(features, on=["race_id", "horse_num"], how="inner")
    n_after_feat = len(merged)
    print(
        f"after merge with features: rows={n_after_feat:,} "
        f"(dropped {n_scores_rows - n_after_feat:,} of {n_scores_rows:,}), "
        f"races={merged['race_id'].nunique():,}"
    )

    merged = merged.merge(ra, on="race_id", how="inner")
    n_after_ra = len(merged)
    print(
        f"after merge with RA (race_condition_code): rows={n_after_ra:,} "
        f"(dropped {n_after_feat - n_after_ra:,}), races={merged['race_id'].nunique():,}"
    )

    n_before_odds = len(merged)
    merged = attach_odds_from_se_parquet(merged)
    n_odds_missing = merged["odds"].isna().sum()
    print(
        f"odds attached: rows={len(merged):,}, missing_odds={n_odds_missing:,} "
        f"({n_odds_missing / max(len(merged), 1):.4%})"
    )
    merged = merged.dropna(subset=["odds"])
    print(f"after dropping missing-odds rows: rows={len(merged):,} (dropped {n_before_odds - len(merged):,})")

    merged = attach_market_q(merged)

    n_final_rows = len(merged)
    n_final_races = merged["race_id"].nunique()
    print(f"final gate_dataset: rows={n_final_rows:,}, races={n_final_races:,}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build_alpha_segments_dataset()
