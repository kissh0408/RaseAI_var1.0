"""
train.py — RaceAI_var1.0 LambdaRank 学習スクリプト

使用方法:
    python pure_rank/src/train.py              # seed=42 のみ、fold 3 のみ
    python pure_rank/src/train.py --ensemble   # 5 seeds × 3 folds = 15 モデル

モデル保存先:
    pure_rank/models/lambdarank_fold{1,2,3}_seed{42-46}.txt

禁止事項:
- init_score に市場オッズ由来の値を使わない
- categorical_feature を lgb.Dataset に未指定にしない
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── 特徴量列の選択 ────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """学習に使う特徴量列を返す。

    ID 列・ラベル列・禁止列を除外する。
    """
    id_cols = set(cfg["features"]["id_cols"])
    forbidden = {
        # 市場情報（絶対禁止）
        "odds", "popularity", "win_odds", "place_odds",
        "quinella_odds", "market_prob", "market_log_odds",
        "init_score", "ninki",
        # 一時作業列
        "_time_dev",
        # RA / SE のメタ列（特徴量として不要）
        "year", "month_day", "kai", "nichi", "race_num",
        "horse_num", "registered_count", "finish_count",
        "race_type_code", "weight_type", "race_condition_code",
        "race_level", "race_age_type", "course_kubun",
        "track_code",
        "obstacle_mile_time_sec",
        "dead_heat_flag", "dead_heat_count",
        "breed_code", "region_code",
        # 血統 ID（文字列。特徴量としては派生した win_rate 系を使う）
        "sire_id", "bms_id",
        # ─── レース後にしか判明しない後出し情報（特徴量にしてはならない） ───
        # 走破タイム・上がり3F（結果。hist_ 系経由で過去走データは使用可）
        "racetime", "time_3f_after",
        # コーナー通過順（レース中の位置情報。結果）
        "corner_1", "corner_2", "corner_3", "corner_4",
        # 脚質判定（レース後判定）
        "running_style_code",
        # 異常区分（レース後確定）
        "abnormal_code",
        # 賞金（レース後確定。hist_ 系経由で過去走データは使用可）
        "hon_shokin", "fuka_shokin",
        # 生ラベル（全てレース後確定）
        "finish_rank", "is_win", "is_place", "lr_label",
    }
    exclude = id_cols | forbidden

    # 残った数値・カテゴリ列を特徴量とする
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype not in ["object", "string"]
    ]
    return feature_cols


def get_group_sizes(df: pd.DataFrame, race_id_col: str = "race_id") -> list[int]:
    """LightGBM LambdaRank 用 group 配列（レースごとの頭数リスト）を返す。"""
    return df.groupby(race_id_col, sort=False).size().tolist()


# ─── 時系列 Fold 定義 ─────────────────────────────────────────────────────────

# 3-fold 時系列 CV の valid 期間
# Fold 1: 〜2021, valid=2022  Fold 2: 〜2022, valid=2023  Fold 3: 〜2023, valid=2024
FOLD_VALID_RANGES = [
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]


def get_fold_split(
    df: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fold 番号（1-indexed）に対応する train / valid を返す。

    Parameters
    ----------
    df : 全学習対象データ（テスト期間を含まない）
    fold : 1, 2, 3

    Returns
    -------
    (train_df, valid_df)
    """
    valid_start, valid_end = FOLD_VALID_RANGES[fold - 1]
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)

    train_df = df[df["race_date"] < valid_start_ts].copy()
    valid_df = df[
        (df["race_date"] >= valid_start_ts) & (df["race_date"] <= valid_end_ts)
    ].copy()
    return train_df, valid_df


# ─── LambdaRank 学習 ───────────────────────────────────────────────────────────

def train_lambdarank(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    group_valid: list[int],
    feature_cols: list[str],
    cat_features: list[str],
    params_cfg: dict,
    training_cfg: dict,
    seed: int,
) -> lgb.Booster:
    """LambdaRank モデルを学習して返す。

    Parameters
    ----------
    init_score は使わない（RaceAI_var2.0.0 との根本的な違い）
    categorical_feature を lgb.Dataset に必ず指定する
    """
    # cat_features のうち実際に feature_cols に含まれるものだけ指定
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

    # init_score は使わない（市場オッズ由来の残差学習は禁止）
    lgb_train = lgb.Dataset(
        X_train[feature_cols],
        label=y_train,
        group=group_train,
        categorical_feature=valid_cat,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid[feature_cols],
        label=y_valid,
        group=group_valid,
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


# ─── メイン ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RaceAI_var1.0 LambdaRank Training")
    parser.add_argument(
        "--ensemble", action="store_true",
        help="5 seeds × 3 folds の全モデルを学習する（省略時は seed=42 + fold 3 のみ）"
    )
    args = parser.parse_args()

    cfg = load_config()
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    version = cfg["data"]["features_version"]
    feat_path = PROJECT_ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print(f"Loading features: {feat_path}")
    df = pd.read_parquet(feat_path)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    # 特徴量列を決定
    feature_cols = get_feature_cols(df, cfg)
    cat_features = feat_cfg["categorical"]
    print(f"  Feature cols: {len(feature_cols)}")
    print(f"  Cat features: {cat_features}")

    # テスト期間を除外（学習・バリデーション用のみ）
    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()
    df_test = df[df["race_date"] > valid_end_ts].copy()
    print(f"  Train pool: {len(df_train_pool):,} rows | Test: {len(df_test):,} rows")

    # 学習対象のシード・フォールドを決定
    if args.ensemble:
        seeds = training_cfg["seeds"]
        folds = list(range(1, training_cfg["folds"] + 1))
    else:
        seeds = [training_cfg["seeds"][0]]  # 42 のみ
        folds = [training_cfg["folds"]]     # fold 3 のみ

    print(f"\nTraining: seeds={seeds}, folds={folds}")
    print(f"Total models: {len(seeds) * len(folds)}")

    trained_models = []
    for seed in seeds:
        for fold in folds:
            model_path = models_dir / f"lambdarank_fold{fold}_seed{seed}.txt"

            print(f"\n--- Fold {fold} / Seed {seed} ---")
            train_df, valid_df = get_fold_split(df_train_pool, fold)
            print(f"  Train: {len(train_df):,} rows, {train_df['race_id'].nunique():,} races "
                  f"({train_df['race_date'].min().date()} - {train_df['race_date'].max().date()})")
            print(f"  Valid: {len(valid_df):,} rows, {valid_df['race_id'].nunique():,} races "
                  f"({valid_df['race_date'].min().date()} - {valid_df['race_date'].max().date()})")

            if len(valid_df) == 0:
                print("  [SKIP] valid_df が空です")
                continue

            # LambdaRank ラベル（lr_label）
            y_train = train_df[feat_cfg["lr_label"]]
            y_valid = valid_df[feat_cfg["lr_label"]]

            # group 配列（レースごとの頭数）
            # 注意: race_id の順序を保持するため sort=False
            group_train = get_group_sizes(train_df)
            group_valid = get_group_sizes(valid_df)

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
            trained_models.append(model)

    print(f"\n[train] Done. {len(trained_models)} models trained.")

    # 特徴量重要度サマリー（最後のモデル）
    if trained_models:
        last_model = trained_models[-1]
        importance = pd.Series(
            last_model.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=False)
        print("\nTop 20 features by gain:")
        print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
