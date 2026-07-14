"""gain_margin_diagnostic: fold2 5シードモデルのOOSスコアを書き出し、標準lr_labelで評価する。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import get_feature_cols, load_config  # noqa: E402
from evaluate import ensemble_predict, compute_metrics  # noqa: E402
from score_utils import attach_pure_score_z  # noqa: E402

from train_fold2 import FEATURES_PATH, EXP_DIR as _EXP_DIR, FOLD  # noqa: E402

EXP_DIR = _EXP_DIR
BASELINE_SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def evaluate_scores_path(scores_path: Path) -> dict:
    df = pd.read_parquet(scores_path)
    if "lr_label" not in df.columns:
        feat = pd.read_parquet(FEATURES_PATH, columns=["race_id", "horse_num", "lr_label"])
        feat["race_id"] = feat["race_id"].astype(str)
        df["race_id"] = df["race_id"].astype(str)
        df = df.merge(feat, on=["race_id", "horse_num"], how="left")
    return compute_metrics(df, df["pure_score_z"].values)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="margin_only", choices=["margin_only", "combo"])
    args = parser.parse_args()

    models_dir = EXP_DIR / "models" / args.variant
    scores_path = EXP_DIR / "scores" / f"scores_gain_{args.variant}_fold2_oos.parquet"

    cfg = load_config()
    df = _apply_filters(pd.read_parquet(FEATURES_PATH), cfg)
    df = df[df["race_date"] >= pd.Timestamp("2023-01-01")]
    feature_cols = get_feature_cols(df, cfg)

    model_paths = sorted(models_dir.glob(f"lambdarank_fold{FOLD}_seed*.txt"))
    if len(model_paths) != 5:
        raise ValueError(f"モデル数が {len(model_paths)}（期待5）: {models_dir}。先に train_fold2.py を実行してください。")
    models = [lgb.Booster(model_file=str(p)) for p in model_paths]
    print(f"{len(models)}モデルでスコアリング")

    df = df.copy()
    df["pure_score"] = ensemble_predict(models, df[feature_cols])
    df = attach_pure_score_z(df, score_col="pure_score", race_id_col="race_id", out_col="pure_score_z")

    out_cols = ["race_id", "race_date", "ketto_num", "horse_num", "finish_rank", "lr_label", "pure_score", "pure_score_z"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].sort_values(["race_date", "race_id", "horse_num"])
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(scores_path, index=False, compression="snappy")
    print(f"Saved: {scores_path}")

    metrics = evaluate_scores_path(scores_path)
    baseline = evaluate_scores_path(BASELINE_SCORES_PATH)

    print(f"\n=== 比較 (fold2 OOS, race_date>=2023-01-01, 評価は標準lr_label, variant={args.variant}) ===")
    print(f"{'指標':12s} {'baseline(v39)':>15s} {args.variant:>15s} {'差分':>10s}")
    for key in ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]:
        b, v = baseline[key], metrics[key]
        print(f"{key:12s} {b:15.4f} {v:15.4f} {v - b:+10.4f}")
    print(f"{'n_races':12s} {baseline['n_races']:15d} {metrics['n_races']:15d}")

    report = {
        "variant": f"gain_{args.variant}",
        "baseline": baseline,
        "variant_metrics": metrics,
        "delta": {k: metrics[k] - baseline[k] for k in ["top1_rate", "top3_rate", "ndcg_at_3", "spearman"]},
    }
    report_path = EXP_DIR / "reports" / f"comparison_gain_{args.variant}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
