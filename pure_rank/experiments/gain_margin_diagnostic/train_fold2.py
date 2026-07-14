"""gain_margin_diagnostic: fold2 のみ5シード学習（着差反映label_gain版）。

特徴量は v39_course_slim のまま変更なし。学習ラベルのみ lr_label_margin
（8段階、僅差2着に部分点gain=50）に差し替える。
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

from build_margin_label import LABEL_GAIN_MARGIN  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
MARGIN_LABEL_PATH = EXP_DIR / "data" / "lr_label_margin.parquet"
MODELS_DIR = EXP_DIR / "models"
FOLD = 2

BAGGING_PARAMS = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}


def train_lambdarank_margin(
    X_train, y_train, group_train, X_valid, y_valid, group_valid,
    feature_cols, cat_features, params_cfg, training_cfg, seed,
    extra_params: dict | None = None,
) -> lgb.Booster:
    valid_cat = [c for c in cat_features if c in feature_cols]
    params = {
        "objective": params_cfg["objective"],
        "metric": params_cfg["metric"],
        "ndcg_eval_at": params_cfg["ndcg_eval_at"],
        "label_gain": LABEL_GAIN_MARGIN,  # 8段階（僅差2着=50）
        "num_leaves": params_cfg["num_leaves"],
        "min_child_samples": params_cfg["min_child_samples"],
        "reg_alpha": params_cfg["reg_alpha"],
        "reg_lambda": params_cfg["reg_lambda"],
        "learning_rate": params_cfg["learning_rate"],
        "seed": seed,
        "verbose": -1,
    }
    if extra_params:
        params.update(extra_params)
    lgb_train = lgb.Dataset(
        X_train[feature_cols], label=y_train, group=group_train,
        categorical_feature=valid_cat, free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid[feature_cols], label=y_valid, group=group_valid,
        categorical_feature=valid_cat, reference=lgb_train, free_raw_data=False,
    )
    model = lgb.train(
        params, lgb_train, num_boost_round=params_cfg["n_estimators"],
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(training_cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(training_cfg["log_eval_period"]),
        ],
    )
    return model


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", action="store_true", help="bagging正則化も併用する")
    parser.add_argument("--folds", nargs="+", type=int, default=[2], help="学習するfold番号（1,2,3）")
    args = parser.parse_args()
    extra_params = BAGGING_PARAMS if args.combo else None
    models_dir = EXP_DIR / "models" / ("combo" if args.combo else "margin_only")

    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    # feature_cols は margin ラベルをマージする前の列集合で確定させる。
    # 先にマージすると lr_label_margin 自体が特徴量列に紛れ込みリークする。
    feature_cols = get_feature_cols(df, cfg)
    cat_features = feat_cfg["categorical"]

    margin_label = pd.read_parquet(MARGIN_LABEL_PATH)
    df = df.merge(margin_label, on=["race_id", "horse_num"], how="left")
    assert df["lr_label_margin"].isna().sum() == 0, "margin label に欠損があります"
    assert "lr_label_margin" not in feature_cols, "ラベル列がリークしています"
    print(f"  rows={len(df):,}, feature_cols={len(feature_cols)}")

    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

    models_dir.mkdir(parents=True, exist_ok=True)
    seeds = training_cfg["seeds"]

    for fold in args.folds:
        train_df, valid_df = get_fold_split(df_train_pool, fold, training_cfg["fold_valid_years"])
        print(f"\n=== fold={fold}: Train {len(train_df):,} rows / Valid {len(valid_df):,} rows ===")

        y_train = train_df["lr_label_margin"]
        y_valid = valid_df["lr_label_margin"]
        group_train = get_group_sizes(train_df)
        group_valid = get_group_sizes(valid_df)

        for seed in seeds:
            print(f"\n--- gain_margin ({'combo' if args.combo else 'margin_only'}) / fold{fold} / seed {seed} ---")
            model = train_lambdarank_margin(
                X_train=train_df, y_train=y_train, group_train=group_train,
                X_valid=valid_df, y_valid=y_valid, group_valid=group_valid,
                feature_cols=feature_cols, cat_features=cat_features,
                params_cfg=params_cfg, training_cfg=training_cfg, seed=seed,
                extra_params=extra_params,
            )
            model_path = models_dir / f"lambdarank_fold{fold}_seed{seed}.txt"
            model.save_model(str(model_path))
            print(f"  Saved: {model_path} (best_iteration={model.best_iteration})")


if __name__ == "__main__":
    main()
