"""v48_agari_turn vs v49_six_lap: 同一条件で学習・評価・比較する。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pure_rank" / "src"
sys.path.insert(0, str(SRC))

from common import get_feature_cols, get_group_sizes, load_config  # noqa: E402
from evaluate import check_leakage_threshold, compute_metrics, ensemble_predict  # noqa: E402
from train import get_fold_split, train_lambdarank  # noqa: E402

VERSIONS = ["v48_agari_turn", "v49_six_lap"]
BASELINE_V39 = {
    "top1_rate": 0.3024,
    "top3_rate": 0.6176,
    "ndcg_at_3": 0.5359,
    "spearman": 0.5048,
    "label": "v39_course_slim (本番)",
}


def train_version(version: str, cfg: dict) -> Path:
    params_cfg = cfg["model"]
    training_cfg = cfg["training"]
    feat_cfg = cfg["features"]

    feat_path = ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    models_dir = ROOT / cfg["data"]["models_dir"] / "compare" / version
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nTraining {version}\n{'=' * 60}")
    df = pd.read_parquet(feat_path)
    feature_cols = get_feature_cols(df, cfg)
    cat_features = feat_cfg["categorical"]

    valid_end_ts = pd.Timestamp(training_cfg["valid_end"])
    df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

    seeds = training_cfg["seeds"]
    folds = list(range(1, training_cfg["folds"] + 1))
    trained = 0

    for seed in seeds:
        for fold in folds:
            out = models_dir / f"lambdarank_fold{fold}_seed{seed}.txt"
            train_df, valid_df = get_fold_split(
                df_train_pool, fold, training_cfg["fold_valid_years"]
            )
            if valid_df.empty:
                continue
            model = train_lambdarank(
                X_train=train_df,
                y_train=train_df[feat_cfg["lr_label"]],
                group_train=get_group_sizes(train_df),
                X_valid=valid_df,
                y_valid=valid_df[feat_cfg["lr_label"]],
                group_valid=get_group_sizes(valid_df),
                feature_cols=feature_cols,
                cat_features=cat_features,
                params_cfg=params_cfg,
                training_cfg=training_cfg,
                seed=seed,
            )
            model.save_model(str(out))
            trained += 1
            print(f"  saved {out.name} ({trained}/{len(seeds) * len(folds)})")

    return models_dir


def evaluate_version(version: str, cfg: dict, models_dir: Path) -> dict:
    feat_path = ROOT / cfg["data"]["features_dir"] / f"features_{version}.parquet"
    df = pd.read_parquet(feat_path)
    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end_ts].copy()
    feature_cols = get_feature_cols(df_test, cfg)

    model_files = sorted(models_dir.glob("lambdarank_fold*_seed*.txt"))
    if not model_files:
        raise FileNotFoundError(f"No models in {models_dir}")
    models = [lgb.Booster(model_file=str(p)) for p in model_files]
    preds = ensemble_predict(models, df_test[feature_cols])
    metrics = compute_metrics(df_test, preds)
    check_leakage_threshold(metrics)
    metrics["version"] = version
    metrics["n_features"] = len(feature_cols)
    metrics["n_models"] = len(models)
    return metrics


def main() -> None:
    cfg = load_config()
    results: dict[str, dict] = {}

    for version in VERSIONS:
        models_dir = train_version(version, cfg)
        results[version] = evaluate_version(version, cfg, models_dir)

    out_dir = ROOT / "pure_rank" / "data" / "02_features"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": results,
        "baseline_v39": BASELINE_V39,
        "delta_v49_vs_v48": {
            k: results["v49_six_lap"][k] - results["v48_agari_turn"][k]
            for k in ("top1_rate", "top3_rate", "ndcg_at_3", "spearman")
        },
    }
    report_path = out_dir / "compare_v48_v49_results.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY (2025+ test)")
    print("=" * 60)
    header = f"{'version':<20} {'Top-1':>8} {'Top-3':>8} {'NDCG@3':>8} {'Spearman':>9} {'#feat':>6}"
    print(header)
    print("-" * len(header))
    rows = [("v39 (baseline)", BASELINE_V39)]
    for ver, m in results.items():
        rows.append((ver, m))
    for name, m in rows:
        print(
            f"{name:<20} {m['top1_rate']*100:7.2f}% {m['top3_rate']*100:7.2f}% "
            f"{m['ndcg_at_3']:8.4f} {m['spearman']:9.4f} "
            f"{m.get('n_features', '-'):>6}"
        )
    d = report["delta_v49_vs_v48"]
    print("-" * len(header))
    print(
        f"{'v49 - v48':<20} {d['top1_rate']*100:+6.2f}pp {d['top3_rate']*100:+6.2f}pp "
        f"{d['ndcg_at_3']:+8.4f} {d['spearman']:+9.4f}"
    )
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
