"""gain_margin_diagnostic: combo(15モデル)をTEST期間で1回だけ正式評価する（Rule 3）。

本番 pure_rank/src/evaluate.py::main() と同一プロトコル
（df_test = race_date > valid_end, 全モデルをensemble_predict）で、
本番モデル群ではなく combo (margin gain + bagging, 15モデル) を評価する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, load_config  # noqa: E402
from evaluate import load_models, ensemble_predict, compute_metrics  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
MODELS_DIR = EXP_DIR / "models" / "combo"

# CLAUDE.md記載の現行正式ベースライン（2025-01-05〜2026-05-24, 4,775レース、evaluator合格値）
OFFICIAL_BASELINE = {
    "top1_rate": 0.3024,
    "top3_rate": 0.6176,
    "ndcg_at_3": 0.5359,
    "spearman": 0.5048,
}


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(FEATURES_PATH)

    valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end_ts].copy()
    print(f"TEST set: {len(df_test):,} rows, {df_test['race_id'].nunique():,} races "
          f"({df_test['race_date'].min().date()} - {df_test['race_date'].max().date()})")

    feature_cols = get_feature_cols(df_test, cfg)
    print(f"Feature cols: {len(feature_cols)}")

    models = load_models(MODELS_DIR)
    print(f"{len(models)} models loaded from {MODELS_DIR}")

    preds = ensemble_predict(models, df_test[feature_cols])
    metrics = compute_metrics(df_test, preds)

    print("\n=== TEST期間 正式評価（Rule 3: 1回限り） ===")
    print(f"{'指標':12s} {'公式baseline(v39)':>18s} {'combo(15モデル)':>18s} {'差分':>10s}")
    for key in ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]:
        b, v = OFFICIAL_BASELINE[key], metrics[key]
        print(f"{key:12s} {b:18.4f} {v:18.4f} {v - b:+10.4f}")
    print(f"{'n_races':12s} {'4775 (参考)':>18s} {metrics['n_races']:18d}")

    report = {
        "variant": "gain_margin_combo_test_official",
        "official_baseline_v39": OFFICIAL_BASELINE,
        "combo_metrics": metrics,
        "delta": {k: metrics[k] - OFFICIAL_BASELINE[k] for k in
                  ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]},
    }
    report_path = EXP_DIR / "reports" / "test_official_combo.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
