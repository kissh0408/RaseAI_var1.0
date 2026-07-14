"""track_bias_daily: fold2のみ5シード学習（v39_course_slim + post_bias_today + pace_bias_x_style）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, get_group_sizes, load_config  # noqa: E402
from train import get_fold_split, train_lambdarank  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
BIAS_PATH = EXP_DIR / "data" / "track_bias_features.parquet"
MODELS_DIR = EXP_DIR / "models"
FOLD = 2
NEW_COLS = ["post_bias_today", "pace_bias_x_style"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[2])
    args = parser.parse_args()

    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    bias = pd.read_parquet(BIAS_PATH)
    df = df.merge(bias, on=["race_id", "horse_num"], how="left")

    feature_cols = get_feature_cols(df, cfg)
    for c in NEW_COLS:
        assert c in feature_cols, f"{c} が特徴量列から漏れています"
    cat_features = feat_cfg["categorical"]
    print(f"  rows={len(df):,}, feature_cols={len(feature_cols)} (v39=106 + 新規{len(NEW_COLS)})")

    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = training_cfg["seeds"]

    for fold in args.folds:
        train_df, valid_df = get_fold_split(df_train_pool, fold, training_cfg["fold_valid_years"])
        print(f"\n=== fold={fold}: Train {len(train_df):,} / Valid {len(valid_df):,} ===")

        y_train = train_df[feat_cfg["lr_label"]]
        y_valid = valid_df[feat_cfg["lr_label"]]
        group_train = get_group_sizes(train_df)
        group_valid = get_group_sizes(valid_df)

        model = None
        for seed in seeds:
            print(f"\n--- track_bias / fold{fold} / seed {seed} ---")
            model = train_lambdarank(
                X_train=train_df, y_train=y_train, group_train=group_train,
                X_valid=valid_df, y_valid=y_valid, group_valid=group_valid,
                feature_cols=feature_cols, cat_features=cat_features,
                params_cfg=params_cfg, training_cfg=training_cfg, seed=seed,
            )
            model_path = MODELS_DIR / f"lambdarank_fold{fold}_seed{seed}.txt"
            model.save_model(str(model_path))
            print(f"  Saved: {model_path} (best_iteration={model.best_iteration})")

        if fold == 2 and model is not None:
            importance = pd.Series(
                model.feature_importance(importance_type="gain"), index=feature_cols,
            ).sort_values(ascending=False)
            rank_series = importance.rank(ascending=False)
            print("\n新規特徴量の重要度 (fold2最終seed, gain):")
            for c in NEW_COLS:
                if c in importance.index:
                    print(f"  {c}: gain={importance[c]:.2f}, rank={int(rank_series[c])}/{len(importance)}")


if __name__ == "__main__":
    main()
