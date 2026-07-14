"""train_config_tuning: 共通ヘルパー。

本番 pure_rank/src/ の関数（get_fold_split, get_feature_cols, get_group_sizes,
ensemble_predict, compute_metrics）を再利用し、学習ロジックのみローカルに
複製する（train_lambdarank は params のキーをホワイトリストしているため、
lambdarank_truncation_level 等の未対応パラメータを渡せない。本番コードは
変更せず、実験側でパラメータを拡張したバージョンを持つ）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, get_group_sizes, load_config  # noqa: E402
from train import get_fold_split  # noqa: E402
from evaluate import ensemble_predict, compute_metrics  # noqa: E402
from score_utils import attach_pure_score_z  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
FOLD = 2
BASELINE_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"


def train_lambdarank_ext(
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
    extra_params: dict | None = None,
) -> lgb.Booster:
    """train.py::train_lambdarank と同一ロジック + extra_params (実験用パラメータ)。"""
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


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def load_fold2_train_valid(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = get_feature_cols(df, cfg)
    cat_features = cfg["features"]["categorical"]

    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()
    train_df, valid_df = get_fold_split(df_train_pool, FOLD, cfg["training"]["fold_valid_years"])
    return train_df, valid_df, feature_cols, cat_features


def export_scores_for_models(models: list[lgb.Booster], out_path: Path, cfg: dict) -> pd.DataFrame:
    df = _apply_filters(pd.read_parquet(FEATURES_PATH), cfg)
    df = df[df["race_date"] >= pd.Timestamp("2023-01-01")]
    feature_cols = get_feature_cols(df, cfg)

    df = df.copy()
    df["pure_score"] = ensemble_predict(models, df[feature_cols])
    df = attach_pure_score_z(df, score_col="pure_score", race_id_col="race_id", out_col="pure_score_z")

    out_cols = ["race_id", "race_date", "ketto_num", "horse_num", "finish_rank",
                "lr_label", "pure_score", "pure_score_z"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].sort_values(["race_date", "race_id", "horse_num"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False, compression="snappy")
    return out_df


def evaluate_scores_path(scores_path: Path) -> dict:
    """scores parquet (pure_score_z, finish_rank, lr_label 列を含む) から指標を計算する。"""
    df = pd.read_parquet(scores_path)
    if "lr_label" not in df.columns:
        # 既存の本番 fold2 OOS ファイルは lr_label を含まないので features から補完する
        feat = pd.read_parquet(FEATURES_PATH, columns=["race_id", "horse_num", "lr_label"])
        feat["race_id"] = feat["race_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)
        df = df.merge(feat, on=["race_id", "horse_num"], how="left")
    return compute_metrics(df, df["pure_score_z"].values)
