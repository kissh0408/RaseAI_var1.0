"""train_config_tuning: variantのfold2モデルをスコアリングし、ベースラインと比較する。

使用法:
    python evaluate_variant.py --variant truncation
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb

from lib import (
    EXP_DIR, FOLD, BASELINE_SCORES_PATH, load_config,
    export_scores_for_models, evaluate_scores_path,
)

REPORTS_DIR = EXP_DIR / "reports"
SCORES_DIR = EXP_DIR / "scores"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    cfg = load_config()
    models_dir = EXP_DIR / "models" / args.variant
    model_paths = sorted(models_dir.glob(f"lambdarank_fold{FOLD}_seed*.txt"))
    if not model_paths:
        raise ValueError(f"モデルが見つかりません: {models_dir}。先に train_variant.py を実行してください。")
    models = [lgb.Booster(model_file=str(p)) for p in model_paths]
    print(f"variant={args.variant}: {len(models)}モデルでスコアリング")

    scores_path = SCORES_DIR / f"scores_{args.variant}_fold2_oos.parquet"
    export_scores_for_models(models, scores_path, cfg)
    print(f"Saved: {scores_path}")

    metrics = evaluate_scores_path(scores_path)
    baseline_metrics = evaluate_scores_path(BASELINE_SCORES_PATH)

    print("\n=== 比較 (fold2 OOS, race_date>=2023-01-01) ===")
    print(f"{'指標':12s} {'baseline(v39)':>15s} {args.variant:>15s} {'差分':>10s}")
    for key in ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]:
        b, v = baseline_metrics[key], metrics[key]
        print(f"{key:12s} {b:15.4f} {v:15.4f} {v - b:+10.4f}")
    print(f"{'n_races':12s} {baseline_metrics['n_races']:15d} {metrics['n_races']:15d}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "variant": args.variant,
        "baseline": baseline_metrics,
        "variant_metrics": metrics,
        "delta": {k: metrics[k] - baseline_metrics[k] for k in
                  ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]},
    }
    report_path = REPORTS_DIR / f"comparison_{args.variant}.json"
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
