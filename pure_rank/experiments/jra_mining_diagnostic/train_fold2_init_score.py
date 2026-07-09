"""market_leak_diagnostic 同様、fold2 のみ 5 シード学習だが、今回は
var2.0.0 と同じ「init_score(base_margin) 残差学習」トリックを再現する。

jra_tm_log_odds は特徴量には含めず、lgb.Dataset の init_score として渡す。
モデルはこの base_margin からの残差だけを学習する。
本番 pure_rank/models/ には一切保存しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, get_group_sizes, load_config  # noqa: E402
from train import get_fold_split  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = EXP_DIR / "data" / "features_jra_init_score.parquet"
MODELS_DIR = EXP_DIR / "models"

BASE_MARGIN_COL = "jra_tm_log_odds"
FOLD = 2


def train_with_init_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    base_margin_train,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    group_valid: list[int],
    base_margin_valid,
    feature_cols: list[str],
    cat_features: list[str],
    params_cfg: dict,
    training_cfg: dict,
    seed: int,
) -> lgb.Booster:
    valid_cat = [c for c in cat_features if c in feature_cols]
    params = {
        "objective": params_cfg["objective"],
        "metric": params_cfg["metric"],
        "ndcg_eval_at": params_cfg["ndcg_eval_at"],
        "label_gain": params_cfg["label_gain"],
        "num_leaves": params_cfg["num_leaves"],
        "min_child_samples": params_cfg["min_child_samples"],
        "reg_alpha": params_cfg["reg_alpha"],
        "reg_lambda": params_cfg["reg_lambda"],
        "learning_rate": params_cfg["learning_rate"],
        "seed": seed,
        "verbose": -1,
    }
    lgb_train = lgb.Dataset(
        X_train[feature_cols],
        label=y_train,
        group=group_train,
        categorical_feature=valid_cat,
        init_score=base_margin_train,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid[feature_cols],
        label=y_valid,
        group=group_valid,
        categorical_feature=valid_cat,
        init_score=base_margin_valid,
        reference=lgb_train,
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=params_cfg["n_estimators"],
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(training_cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(training_cfg["log_eval_period"]),
        ],
    )
    return model


def main() -> None:
    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    feature_cols = [c for c in get_feature_cols(df, cfg) if c != BASE_MARGIN_COL]
    cat_features = feat_cfg["categorical"]
    print(f"  Feature cols: {len(feature_cols)} (base_margin列 {BASE_MARGIN_COL} は特徴量から除外)")

    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = training_cfg["seeds"]
    print(f"\nTraining fold={FOLD} (init_score trick), seeds={seeds}")

    train_df, valid_df = get_fold_split(df_train_pool, FOLD, training_cfg["fold_valid_years"])
    print(f"  Train: {len(train_df):,} rows, {train_df['race_id'].nunique():,} races "
          f"({train_df['race_date'].min().date()} - {train_df['race_date'].max().date()})")
    print(f"  Valid: {len(valid_df):,} rows, {valid_df['race_id'].nunique():,} races "
          f"({valid_df['race_date'].min().date()} - {valid_df['race_date'].max().date()})")

    y_train = train_df[feat_cfg["lr_label"]]
    y_valid = valid_df[feat_cfg["lr_label"]]
    group_train = get_group_sizes(train_df)
    group_valid = get_group_sizes(valid_df)
    bm_train = train_df[BASE_MARGIN_COL].to_numpy()
    bm_valid = valid_df[BASE_MARGIN_COL].to_numpy()

    for seed in seeds:
        model_path = MODELS_DIR / f"lambdarank_fold{FOLD}_seed{seed}.txt"
        print(f"\n--- Fold {FOLD} / Seed {seed} (init_score trick) ---")
        model = train_with_init_score(
            X_train=train_df,
            y_train=y_train,
            group_train=group_train,
            base_margin_train=bm_train,
            X_valid=valid_df,
            y_valid=y_valid,
            group_valid=group_valid,
            base_margin_valid=bm_valid,
            feature_cols=feature_cols,
            cat_features=cat_features,
            params_cfg=params_cfg,
            training_cfg=training_cfg,
            seed=seed,
        )
        model.save_model(str(model_path))
        print(f"  Saved: {model_path}")


if __name__ == "__main__":
    main()
