"""
analyze_errors.py — 条件別エラー分析スクリプト

テストセット（2025-01-01以降）でモデルが弱い条件を特定する。
モデルや特徴量は変更しない（分析のみ）。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_config, load_models, ensemble_predict, get_feature_cols

CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


SURFACE_LABELS = {"1": "芝", "2": "ダート", "5": "その他"}
TRACK_LABELS = {"0": "不明", "1": "良", "2": "稍重", "3": "重", "4": "不良"}
DIST_LABELS = {"0": "短距離(~1400m)", "1": "マイル(1401-1800m)",
               "2": "中距離(1801-2200m)", "3": "長距離(2201m+)"}


def main() -> None:
    cfg = load_config()

    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    version = cfg["data"]["features_version"]

    feat_path = feat_dir / f"features_{version}.parquet"
    print(f"Loading features: {feat_path.name}")
    df = pd.read_parquet(feat_path)

    # フィルタ
    df = df[
        (~df["grade_code"].isin(cfg["filters"]["exclude_grade_codes"])) &
        (~df["abnormal_code"].isin(cfg["filters"]["exclude_abnormal_codes"])) &
        (df["horse_count"] >= cfg["filters"]["min_horse_count"]) &
        (df["finish_rank"] > 0)
    ]

    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    df_test = df[df["race_date"] > valid_end].copy()
    print(f"Test samples: {len(df_test):,} rows / {df_test['race_id'].nunique():,} races")

    feat_cols = get_feature_cols(df_test, cfg)

    # モデル読み込みと予測
    print("Loading models...")
    models = load_models(models_dir)
    X = df_test[feat_cols]
    df_test = df_test.copy()
    df_test["pred_score"] = ensemble_predict(models, X)

    # レース単位の集計
    race_results = []
    for race_id, grp in df_test.groupby("race_id"):
        pred_best = grp.loc[grp["pred_score"].idxmax()]
        is_hit = int(pred_best["finish_rank"] == 1)
        meta = grp.iloc[0]

        # 頭数バケット
        hc = int(meta["horse_count"])
        if hc <= 8:
            hc_bucket = "5-8"
        elif hc <= 12:
            hc_bucket = "9-12"
        elif hc <= 16:
            hc_bucket = "13-16"
        else:
            hc_bucket = "17+"

        # grade_code を 2-4 を統合
        gc = int(meta["grade_code"])
        gc_key = str(gc) if gc not in [2, 3, 4] else "2-4"

        race_results.append({
            "race_id": race_id,
            "is_hit": is_hit,
            "surface_code": str(int(meta["surface_code"])),
            "track_condition_code": str(int(meta["track_condition_code"])),
            "distance_category": str(int(meta["distance_category"])),
            "horse_count_bucket": hc_bucket,
            "grade_code": gc_key,
            "course_code": str(int(meta["course_code"])),
        })

    df_race = pd.DataFrame(race_results)
    n_races = len(df_race)
    overall_top1 = df_race["is_hit"].mean()

    print(f"\n=== Error Analysis: {version} ===")
    print(f"Overall Top-1: {overall_top1:.1%}  ({n_races:,} races)")

    result = {
        "model_version": version,
        "overall": {"top1_rate": round(overall_top1, 6), "n_races": n_races},
        "by_surface_code": {},
        "by_track_condition_code": {},
        "by_distance_category": {},
        "by_horse_count_bucket": {},
        "by_grade_code": {},
        "by_course_code": {},
        "worst_conditions": [],
    }

    def analyze_axis(col: str, labels: dict | None = None) -> dict:
        out = {}
        print(f"\n--- by {col} ---")
        for val, grp in df_race.groupby(col, sort=True):
            rate = grp["is_hit"].mean()
            n = len(grp)
            gap = rate - overall_top1
            label = (labels or {}).get(str(val), str(val))
            warning = "n<100" if n < 100 else None
            entry = {"top1_rate": round(rate, 6), "n_races": n,
                     "label": label, "gap_vs_overall": round(gap, 6)}
            if warning:
                entry["warning"] = warning
            out[str(val)] = entry
            gap_str = f"{gap:+.1%}"
            warn_str = "  [WARN n<100]" if warning else ""
            print(f"  {label:12s}({val}): {rate:.1%}  ({n:5d} races)  [diff: {gap_str}]{warn_str}")
        return out

    result["by_surface_code"] = analyze_axis("surface_code", SURFACE_LABELS)
    result["by_track_condition_code"] = analyze_axis("track_condition_code", TRACK_LABELS)
    result["by_distance_category"] = analyze_axis("distance_category", DIST_LABELS)
    result["by_horse_count_bucket"] = analyze_axis("horse_count_bucket")
    result["by_grade_code"] = analyze_axis("grade_code")
    result["by_course_code"] = analyze_axis("course_code")

    # 弱点条件の抽出（gap < -5pp かつ n >= 50）
    GAP_THRESHOLD = -0.05
    MIN_N = 50
    worst = []
    for axis_key, axis_data in result.items():
        if not axis_key.startswith("by_"):
            continue
        for val, stats in axis_data.items():
            if (stats["gap_vs_overall"] < GAP_THRESHOLD
                    and stats["n_races"] >= MIN_N
                    and "warning" not in stats):
                worst.append({
                    "axis": axis_key.replace("by_", ""),
                    "value": val,
                    "label": stats["label"],
                    "top1_rate": stats["top1_rate"],
                    "n_races": stats["n_races"],
                    "gap_vs_overall": stats["gap_vs_overall"],
                })

    worst.sort(key=lambda x: x["gap_vs_overall"])
    result["worst_conditions"] = worst

    print(f"\n=== WORST CONDITIONS (gap < -5pp, n >= {MIN_N}) ===")
    if worst:
        for i, w in enumerate(worst, 1):
            print(f"  {i}. [{w['axis']}={w['value']}/{w['label']}]  "
                  f"{w['top1_rate']:.1%}  ({w['n_races']} races)  "
                  f"gap={w['gap_vs_overall']:+.1%}")
    else:
        print("  (なし)")

    out_path = feat_dir / "error_analysis_v33.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
