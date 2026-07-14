"""cushion_oos_diagnostic: fold2 のみ 5 シード学習（v51_cushion 特徴量、A4/A5削除後4列版）。

本番 pure_rank/models/ には一切保存しない。
本番プロトコル（pure_rank/src/train.py の fold2 = 2023 valid）と同一の
train/valid 分割・ハイパーパラメータを使い、features_v51_cushion.parquet
（v39_course_slim の132列 + cushion_value/moisture_pct/cushion_diff_track_avg/
last3f_rank_x_cushion の4列 = 136列）を使用する点だけが差分。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, get_group_sizes, load_config  # noqa: E402
from train import get_fold_split, train_lambdarank  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v51_cushion.parquet"
MODELS_DIR = EXP_DIR / "models"
FOLD = 2


def main() -> None:
    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    feature_cols = get_feature_cols(df, cfg)
    cat_features = feat_cfg["categorical"]
    print(f"  Feature cols: {len(feature_cols)}")

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

    model = None
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
    print("\nNew v51_cushion feature ranks (last seed, gain importance):")
    rank_series = importance.rank(ascending=False)
    for col in ["cushion_value", "moisture_pct", "cushion_diff_track_avg", "last3f_rank_x_cushion"]:
        if col in importance.index:
            print(f"  {col}: gain={importance[col]:.2f}, rank={int(rank_series[col])}/{len(importance)}")


if __name__ == "__main__":
    main()
