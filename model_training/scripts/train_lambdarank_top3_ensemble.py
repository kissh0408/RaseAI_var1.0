"""NDCG 最適化 Lambdarank（1-3着 relevance）を champion 特徴量で学習。

ワイド軸 KPI（top3 順位付け）向上を目的とする。出力: lgbm_lambdarank_top3_fold{N}_seed{S}.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import MODELS_DIR  # noqa: E402
from train import (  # noqa: E402
    _load_binary_training_features,
    _relevance_from_finish_rank,
    get_feature_cols,
    load_train_config,
    TRAIN_CONFIG_PATH,
)

SEEDS = [42, 43, 44, 45, 46]


def _race_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Lambdarank 用に race_id 単位でソートし group サイズを返す。"""
    out = df.sort_values(["race_id", "horse_num"], kind="mergesort").reset_index(drop=True)
    sizes = out.groupby("race_id", sort=False).size().tolist()
    return out, sizes


def _train_one_fold(
    df: pd.DataFrame,
    fold_cfg: dict,
    feature_cols: list[str],
    t_cfg: dict,
    seed: int,
) -> tuple[lgb.Booster, dict]:
    fold_n = int(fold_cfg["fold"])
    train_end = pd.Timestamp(fold_cfg["train_end"])
    valid_start = pd.Timestamp(fold_cfg["valid_start"])
    valid_end = pd.Timestamp(fold_cfg["valid_end"])

    train_df = df[df["race_date"] <= train_end].copy()
    valid_df = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)].copy()

    for part in (train_df, valid_df):
        part["relevance"] = _relevance_from_finish_rank(
            pd.to_numeric(part["finish_rank"], errors="coerce")
        )

    cols = [c for c in feature_cols if c in train_df.columns]
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "learning_rate": min(float(t_cfg["learning_rate"]), 0.05),
        "num_leaves": int(t_cfg["num_leaves"]),
        "min_child_samples": int(t_cfg["min_child_samples"]),
        "subsample": float(t_cfg["subsample"]),
        "colsample_bytree": float(t_cfg["colsample_bytree"]),
        "reg_alpha": float(t_cfg["reg_alpha"]),
        "reg_lambda": float(t_cfg["reg_lambda"]),
        "verbose": -1,
        "seed": seed,
        "force_row_wise": True,
    }

    train_df, train_groups = _race_groups(train_df)
    valid_df, valid_groups = _race_groups(valid_df)

    ds_train = lgb.Dataset(
        train_df[cols],
        label=train_df["relevance"],
        group=train_groups,
        free_raw_data=False,
    )
    ds_valid = lgb.Dataset(
        valid_df[cols],
        label=valid_df["relevance"],
        group=valid_groups,
        reference=ds_train,
        free_raw_data=False,
    )

    booster = lgb.train(
        params,
        ds_train,
        num_boost_round=int(t_cfg.get("n_estimators", 1000)),
        valid_sets=[ds_valid],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=max(int(t_cfg.get("early_stopping_rounds", 50)), 100),
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    meta = {
        "fold": fold_n,
        "objective": "lambdarank_top3_relevance",
        "seed": seed,
        "feature_cols": cols,
        "best_iteration": int(booster.best_iteration or 0),
        "params": params,
    }
    return booster, meta


def main() -> None:
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    t_cfg = cfg["training"]
    df = _load_binary_training_features()
    feature_cols = get_feature_cols(cfg)

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        for fold_cfg in t_cfg["walkforward_folds"]:
            fold = int(fold_cfg["fold"])
            booster, meta = _train_one_fold(df, fold_cfg, feature_cols, t_cfg, seed)
            out_txt = MODELS_DIR / f"lgbm_lambdarank_top3_fold{fold}_seed{seed}.txt"
            booster.save_model(str(out_txt))
            meta_path = out_txt.with_name(out_txt.stem + "_meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  saved {out_txt.name} iter={meta['best_iteration']}")


if __name__ == "__main__":
    main()
