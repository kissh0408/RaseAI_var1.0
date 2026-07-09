"""market_leak_diagnostic: fold2 のみ 5 シード学習（市場情報つき特徴量）。

本番 pure_rank/models/ には一切保存しない。
本番プロトコル（pure_rank/src/train.py の fold2 = 2023 valid）と同一の
train/valid 分割・ハイパーパラメータを使い、追加した exp_* 列だけが差分。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import FORBIDDEN_COLS, get_feature_cols, get_group_sizes, load_config  # noqa: E402
from train import get_fold_split, train_lambdarank  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = EXP_DIR / "data" / "features_market_leak.parquet"
MODELS_DIR = EXP_DIR / "models"

MARKET_COLS = ["exp_win_odds", "exp_ln_odds", "exp_popularity", "exp_market_log_odds"]
FOLD = 2


def main() -> None:
    assert not (FORBIDDEN_COLS & set(MARKET_COLS)), (
        "MARKET_COLS が本番 FORBIDDEN_COLS と衝突しています。列名を変更してください。"
    )

    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    base_cols = [c for c in get_feature_cols(df, cfg) if c not in MARKET_COLS]
    feature_cols = base_cols + MARKET_COLS
    cat_features = feat_cfg["categorical"]
    print(f"  Feature cols: {len(feature_cols)} (含む市場列: {MARKET_COLS})")

    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = training_cfg["seeds"]  # [42, 43, 44, 45, 46]
    print(f"\nTraining fold={FOLD}, seeds={seeds}")

    train_df, valid_df = get_fold_split(df_train_pool, FOLD, training_cfg["fold_valid_years"])
    print(f"  Train: {len(train_df):,} rows, {train_df['race_id'].nunique():,} races "
          f"({train_df['race_date'].min().date()} - {train_df['race_date'].max().date()})")
    print(f"  Valid: {len(valid_df):,} rows, {valid_df['race_id'].nunique():,} races "
          f"({valid_df['race_date'].min().date()} - {valid_df['race_date'].max().date()})")

    y_train = train_df[feat_cfg["lr_label"]]
    y_valid = valid_df[feat_cfg["lr_label"]]
    group_train = get_group_sizes(train_df)
    group_valid = get_group_sizes(valid_df)

    for seed in seeds:
        model_path = MODELS_DIR / f"lambdarank_fold{FOLD}_seed{seed}.txt"
        print(f"\n--- Fold {FOLD} / Seed {seed} ---")
        model = train_lambdarank(
            X_train=train_df,
            y_train=y_train,
            group_train=group_train,
            X_valid=valid_df,
            y_valid=y_valid,
            group_valid=group_valid,
            feature_cols=feature_cols,
            cat_features=cat_features,
            params_cfg=params_cfg,
            training_cfg=training_cfg,
            seed=seed,
        )
        model.save_model(str(model_path))
        print(f"  Saved: {model_path}")

    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    print("\nTop 20 features by gain (last seed):")
    print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
