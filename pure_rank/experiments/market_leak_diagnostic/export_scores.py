"""market_leak_diagnostic: fold2(市場情報つき) 5 シードモデルの OOS スコアを書き出す。

本番 pure_rank/data/03_scores/ には一切保存しない。
出力フォーマットは本番 export_scores.py --fold 2 --split all と同一列構成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, load_config  # noqa: E402
from evaluate import ensemble_predict  # noqa: E402
from score_utils import attach_pure_score_z  # noqa: E402

from train_fold2 import FEATURES_PATH, FOLD, MARKET_COLS, MODELS_DIR  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
OUT_PATH = EXP_DIR / "scores" / "scores_market_leak_fold2_oos.parquet"
EXPECTED_SEEDS = 5


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def export_experiment_scores() -> Path:
    cfg = load_config()
    print(f"Loading: {FEATURES_PATH}")
    df = _apply_filters(pd.read_parquet(FEATURES_PATH), cfg)
    print(f"  rows={len(df):,}, races={df['race_id'].nunique():,}")

    base_cols = [c for c in get_feature_cols(df, cfg) if c not in MARKET_COLS]
    feature_cols = base_cols + MARKET_COLS

    model_paths = sorted(MODELS_DIR.glob(f"lambdarank_fold{FOLD}_seed*.txt"))
    if len(model_paths) != EXPECTED_SEEDS:
        raise ValueError(
            f"fold{FOLD} のモデル数が {len(model_paths)} 本（期待 {EXPECTED_SEEDS} 本）: {MODELS_DIR}\n"
            "先に train_fold2.py を実行してください。"
        )
    models = [lgb.Booster(model_file=str(p)) for p in model_paths]
    print(f"fold{FOLD} のみ {len(models)} モデルでスコアリング（実験版 OOS）")

    df = df.copy()
    df["pure_score"] = ensemble_predict(models, df[feature_cols])
    df = attach_pure_score_z(df, score_col="pure_score", race_id_col="race_id", out_col="pure_score_z")

    out_cols = [
        "race_id", "race_date", "ketto_num", "horse_num", "horse_number",
        "course_code", "finish_rank", "pure_score", "pure_score_z",
    ]
    if "horse_number" not in df.columns:
        df["horse_number"] = df["horse_num"]
        out_cols = [c for c in out_cols]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].sort_values(["race_date", "race_id", "horse_num"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH} ({len(out_df):,} rows)")
    return OUT_PATH


if __name__ == "__main__":
    export_experiment_scores()
