"""戦略なし: model_prob 上位3頭の3着以内率（baseline vs latent 比較）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))
sys.path.insert(0, str(ROOT / "strategy" / "src"))

from evaluation import calculate_ranking_metrics
from inference_common import _load_booster_crlf_safe, compute_market_log_odds, predict_model_probs
from pipeline_common import FEATURES_DIR, MODELS_DIR, load_config


def load_ensemble_from(dir_path: Path, fold: int) -> list:
    paths = sorted(dir_path.glob(f"lgbm_binary_fold{fold}_seed*.txt"))
    if not paths:
        p = dir_path / f"lgbm_binary_fold{fold}.txt"
        paths = [p] if p.exists() else []
    return [_load_booster_crlf_safe(p) for p in paths]


def load_features(name: str) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_DIR / name)
    if "race_date" not in df.columns:
        df["race_date"] = pd.to_datetime(df["date"])
    else:
        df["race_date"] = pd.to_datetime(df["race_date"])
    if "market_log_odds" not in df.columns:
        df = compute_market_log_odds(df, odds_col="odds")
    df["race_id"] = df["race_id"].astype(str)
    return df


def main() -> None:
    cfg = load_config()
    folds = cfg["training"]["walkforward_folds"]
    base_margin = cfg["training"].get("base_margin_col", "market_log_odds")

    variants = [
        ("baseline_v6", "features_v6.parquet", MODELS_DIR / "backup_baseline_v6"),
        ("v6_latent", "features_v6_latent.parquet", MODELS_DIR / "backup_latent_20260616"),
        (
            "refined_moisture",
            "features_v6_refined_moisture.parquet",
            MODELS_DIR / "ablation_latent/models_refined_moisture",
        ),
        (
            "v6_training",
            "features_v6_ablation_training.parquet",
            MODELS_DIR / "ablation_latent/models_v6_training",
        ),
    ]

    rows: list[dict] = []
    for vname, parquet, model_dir in variants:
        if not model_dir.exists() or not (FEATURES_DIR / parquet).exists():
            print(f"SKIP {vname}: missing data or models")
            continue
        df_all = load_features(parquet)
        for fold_cfg in folds:
            fold = fold_cfg["fold"]
            test_df = df_all[
                (df_all["race_date"] >= pd.Timestamp(fold_cfg["test_start"]))
                & (df_all["race_date"] <= pd.Timestamp(fold_cfg["test_end"]))
            ].copy()
            models = load_ensemble_from(model_dir, fold)
            if not models:
                print(f"SKIP {vname} F{fold}: no models")
                continue
            feat_cols = list(models[0].feature_name())
            test_df["model_prob"] = predict_model_probs(models, test_df, feat_cols, base_margin).values
            m = calculate_ranking_metrics(test_df)
            row = {"variant": vname, "fold": fold, **m}
            rows.append(row)
            print(
                f"{vname} F{fold}: overlap={m['top3_overlap_rate']:.1%} "
                f"top1_win={m['top1_win_rate']:.1%} box3={m['top3_box_rate']:.1%} "
                f"races={m['n_races']}"
            )

    out = MODELS_DIR / "ablation_latent/top3_ranking_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
