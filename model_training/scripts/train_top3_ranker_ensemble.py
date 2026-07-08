"""3着以内(target=top3) binary 残差モデルを champion と同設定で学習する。

出力: lgbm_top3_fold{N}_seed{S}.txt
評価: combo_rank_hit_rates.py --prob-source top3
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
    build_base_params,
    get_feature_cols,
    load_train_config,
    TRAIN_CONFIG_PATH,
)

SEEDS = [42, 43, 44, 45, 46]


def _place_log_odds(df: pd.DataFrame) -> np.ndarray:
    """複勝相当の粗い事前 log-odds: log(3 / field_size)。"""
    n = pd.to_numeric(df["horse_count"], errors="coerce").fillna(16).clip(lower=3)
    p = 3.0 / n
    return np.log(p / (1.0 - p)).astype(float).to_numpy()


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
        fr = pd.to_numeric(part["finish_rank"], errors="coerce")
        part["is_top3"] = ((fr >= 1) & (fr <= 3)).astype(int)

    cols = [c for c in feature_cols if c in train_df.columns]
    params = build_base_params(t_cfg)
    params["seed"] = seed

    train_margin = _place_log_odds(train_df)
    valid_margin = _place_log_odds(valid_df)

    ds_train = lgb.Dataset(
        train_df[cols], label=train_df["is_top3"], init_score=train_margin, free_raw_data=False
    )
    ds_valid = lgb.Dataset(
        valid_df[cols],
        label=valid_df["is_top3"],
        init_score=valid_margin,
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
                stopping_rounds=int(t_cfg.get("early_stopping_rounds", 50)), verbose=False
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    meta = {
        "fold": fold_n,
        "objective": "top3_binary",
        "seed": seed,
        "feature_cols": cols,
        "best_iteration": int(booster.best_iteration or 0),
    }
    return booster, meta


def main() -> None:
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    t_cfg = cfg["training"]
    df = _load_binary_training_features()
    feature_cols = get_feature_cols(cfg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f"\n=== top3 ranker seed={seed} ===")
        t_cfg["seed"] = seed
        for fold_cfg in t_cfg["walkforward_folds"]:
            model, meta = _train_one_fold(df, fold_cfg, feature_cols, t_cfg, seed)
            fold_n = int(fold_cfg["fold"])
            path = MODELS_DIR / f"lgbm_top3_fold{fold_n}_seed{seed}.txt"
            meta_path = MODELS_DIR / f"lgbm_top3_fold{fold_n}_seed{seed}_meta.json"
            model.save_model(str(path))
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  saved {path.name}")


if __name__ == "__main__":
    main()
