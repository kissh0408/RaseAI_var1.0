"""place_direct: train_fold2.py

複勝(top3)確率を binary 分類で直接予測する LightGBM モデルを 5 シード学習する。

- train: race_date < 2023-01-01
- early stopping: 2023 年（弱汚染として明記。仕様書 §3.5）
- 2024/2025+ は学習・モデル選択に一切使わない
- パラメータは config.json（仕様書 §3.2。本番 train_config.json の値をそのまま継承、
  objective/metric のみ binary 用に変更）
- init_score は使わない
- categorical_feature は本番 train_config.json の features.categorical をそのまま使う

モデル保存先: pure_rank/experiments/place_direct/models/place_direct_seed{42..46}.txt
（本番 pure_rank/models/ には一切保存しない）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import load_config as load_prod_config  # noqa: E402

from place_lib import get_experiment_feature_cols  # noqa: E402

DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
CONFIG_PATH = EXP_DIR / "config.json"


def load_experiment_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def train_place_direct(
    train_df: pd.DataFrame,
    es_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    params_cfg: dict,
    training_cfg: dict,
    seed: int,
) -> lgb.Booster:
    """binary 分類で target_place を直接学習する。init_score は使わない。"""
    valid_cat = [c for c in cat_features if c in feature_cols]

    params = {
        "objective": params_cfg["objective"],
        "metric": params_cfg["metric"],
        "num_leaves": params_cfg["num_leaves"],
        "min_child_samples": params_cfg["min_child_samples"],
        "reg_alpha": params_cfg["reg_alpha"],
        "reg_lambda": params_cfg["reg_lambda"],
        "learning_rate": params_cfg["learning_rate"],
        "seed": seed,
        "verbose": -1,
    }

    lgb_train = lgb.Dataset(
        train_df[feature_cols],
        label=train_df["target_place"],
        categorical_feature=valid_cat,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        es_df[feature_cols],
        label=es_df["target_place"],
        categorical_feature=valid_cat,
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
    exp_cfg = load_experiment_config()
    prod_cfg = load_prod_config()
    params_cfg = exp_cfg["model"]
    training_cfg = exp_cfg["training"]
    cat_features = prod_cfg["features"]["categorical"]

    print(f"Loading: {DATA_DIR / 'train_fold2.parquet'}")
    train_df = pd.read_parquet(DATA_DIR / "train_fold2.parquet")
    es_df = pd.read_parquet(DATA_DIR / "es_fold2_2023.parquet")
    print(f"  train rows={len(train_df):,}, es rows={len(es_df):,}")

    # リーク防止再確認（学習直前）
    assert train_df["race_date"].max() < pd.Timestamp("2023-01-01")
    assert set(es_df["race_date"].dt.year.unique().tolist()) == {2023}

    feature_cols = get_experiment_feature_cols(train_df, prod_cfg)
    assert "target_place" not in feature_cols
    print(f"  Feature cols: {len(feature_cols)}")
    print(f"  Cat features: {cat_features}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = training_cfg["seeds"]
    print(f"\nTraining place_direct binary model, seeds={seeds}")

    best_iters = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        model = train_place_direct(
            train_df, es_df, feature_cols, cat_features, params_cfg, training_cfg, seed
        )
        model_path = MODELS_DIR / f"place_direct_seed{seed}.txt"
        model.save_model(str(model_path))
        best_iters[seed] = model.best_iteration
        print(f"  Saved: {model_path} (best_iteration={model.best_iteration})")

    print("\nBest iterations per seed:", best_iters)

    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    print("\nTop 20 features by gain (last seed):")
    print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
