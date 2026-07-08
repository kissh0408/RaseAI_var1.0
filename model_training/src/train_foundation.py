"""基礎モデル（オッズ・直前情報 非依存）の年次ウォークフォワード学習。

指示書フェーズ1: 市場の群衆心理（当日オッズ・馬体重変化）を含まない
純粋な能力評価モデルを学習し、各馬の基礎予測勝率 P_fund を出力する。

リーク防止（最重要）:
  各年Yの P_fund は「Y未満のデータのみで学習したモデル」によるOOS予測。
  残差モデルの学習行に与える特徴量もアウトオブサンプルであることを保証する。

出力: model_training/data/02_features/foundation_pred.parquet
      (race_id, horse_id, p_fund=レース内正規化勝率, p_fund_log_odds)

実行: python model_training/src/train_foundation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import FEATURES_DIR, load_config
from train import build_base_params, get_feature_cols, load_merged_features

# 市場・直前情報の特徴量（基礎モデルから除外）
MARKET_LATE_FEATURES = {
    "market_prob",
    "weight_diff",
    "weight_diff_trend",
    "weight_relative_z",
}

START_YEAR = 2017  # これ以前は学習データ不足のためP_fund=NaN


def run_foundation_walkforward() -> None:
    cfg = load_config()
    t_cfg = cfg["training"]
    df = load_merged_features()
    df = df[df["finish_rank"] > 0].copy()
    df = df[~df["grade_code"].isin(t_cfg["exclude_grade_codes"])].copy()
    df["year"] = df["race_date"].dt.year

    feature_cols = [c for c in get_feature_cols(cfg) if c not in MARKET_LATE_FEATURES]
    print(f"基礎モデル特徴量: {len(feature_cols)}個（市場・直前系{len(MARKET_LATE_FEATURES)}個を除外）")

    params = build_base_params(t_cfg, verbose=-1)
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = 300  # early stopping なし（バリデーション分割を単純化）のため固定

    preds = []
    years = sorted(df["year"].unique())
    for year in [y for y in years if y >= START_YEAR]:
        train_df = df[df["year"] < year]
        test_df = df[df["year"] == year]
        if len(train_df) < 50_000 or len(test_df) == 0:
            continue
        avail = [c for c in feature_cols if c in df.columns]
        dtrain = lgb.Dataset(train_df[avail], label=(train_df["finish_rank"] == 1).astype(int))
        model = lgb.train({k: v for k, v in params.items() if k != "n_estimators"},
                          dtrain, num_boost_round=params["n_estimators"])
        raw = model.predict(test_df[avail].values)
        out = test_df[["race_id", "horse_id"]].copy()
        out["p_raw"] = raw
        # レース内正規化
        out["p_fund"] = out.groupby(test_df["race_id"].values)["p_raw"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )
        preds.append(out)
        print(f"  {year}: train={len(train_df):,} test={len(test_df):,} 完了")

    result = pd.concat(preds, ignore_index=True)
    p = result["p_fund"].clip(1e-6, 1 - 1e-6)
    result["p_fund_log_odds"] = np.log(p / (1 - p))
    out_path = FEATURES_DIR / "foundation_pred.parquet"
    result[["race_id", "horse_id", "p_fund", "p_fund_log_odds"]].to_parquet(out_path, index=False)
    print(f"保存: {out_path} ({len(result):,}行, {result['p_fund'].isna().mean():.1%} NaN)")


if __name__ == "__main__":
    run_foundation_walkforward()
