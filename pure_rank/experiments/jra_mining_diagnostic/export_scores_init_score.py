"""init_score トリックの合成スコアを OOS エクスポートする。

var2.0.0 の strategy/src/inference_common.py と同じ手順:
  combined_score = booster.predict(X) + base_margin（jra_tm_log_odds）
Booster.predict() は init_score を自動で足し戻さないため、ここで明示的に加算する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, load_config  # noqa: E402
from score_utils import attach_pure_score_z  # noqa: E402

from train_fold2_init_score import BASE_MARGIN_COL, FEATURES_PATH, FOLD, MODELS_DIR  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
OUT_PATH = EXP_DIR / "scores" / "scores_jra_init_score_fold2_oos.parquet"
EXPECTED_SEEDS = 5


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def export_init_score_scores() -> Path:
    cfg = load_config()
    print(f"Loading: {FEATURES_PATH}")
    df = _apply_filters(pd.read_parquet(FEATURES_PATH), cfg)
    print(f"  rows={len(df):,}, races={df['race_id'].nunique():,}")

    feature_cols = [c for c in get_feature_cols(df, cfg) if c != BASE_MARGIN_COL]

    model_paths = sorted(MODELS_DIR.glob(f"lambdarank_fold{FOLD}_seed*.txt"))
    if len(model_paths) != EXPECTED_SEEDS:
        raise ValueError(
            f"fold{FOLD} のモデル数が {len(model_paths)} 本（期待 {EXPECTED_SEEDS} 本）: {MODELS_DIR}\n"
            "先に train_fold2_init_score.py を実行してください。"
        )
    models = [lgb.Booster(model_file=str(p)) for p in model_paths]
    print(f"fold{FOLD} のみ {len(models)} モデルでスコアリング（init_score トリック版）")

    df = df.copy()
    import numpy as np

    raw_preds = np.array([m.predict(df[feature_cols]) for m in models])
    residual_score = raw_preds.mean(axis=0)
    # var2.0.0 と同じ: 予測時に base_margin を明示的に足し戻す
    df["pure_score"] = residual_score + df[BASE_MARGIN_COL].to_numpy()
    df = attach_pure_score_z(df, score_col="pure_score", race_id_col="race_id", out_col="pure_score_z")

    out_cols = [
        "race_id", "race_date", "ketto_num", "horse_num", "horse_number",
        "course_code", "finish_rank", "pure_score", "pure_score_z",
    ]
    if "horse_number" not in df.columns:
        df["horse_number"] = df["horse_num"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].sort_values(["race_date", "race_id", "horse_num"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH} ({len(out_df):,} rows)")
    return OUT_PATH


if __name__ == "__main__":
    export_init_score_scores()
