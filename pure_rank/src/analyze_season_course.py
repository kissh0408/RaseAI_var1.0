"""
analyze_season_course.py — P-30 季節×コース交絡分析

福島・小倉弱点が「コース構造」か「夏開催（月）」かを切り分ける診断スクリプト。
特徴量・モデルは変更しない（分析のみ）。Rule 3: VALID と TEST を完全分離して出力。

出力:
  pure_rank/data/02_features/season_course_analysis_{features_version}.json
  pure_rank/data/02_features/season_course_matrix_{features_version}.csv

第1軸（必須）: month × course_code — Top-1, NDCG@3, n, Δ vs 期間全体
第2軸（参考）: month × course_code × track_condition_code — min_cell_n 未満は除外

仕様: docs/2026-07-05-current-problems-detailed.md P-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, get_feature_cols, load_config
from evaluate import ensemble_predict, load_models, ndcg_at_k

COURSE_NAMES: dict[int, str] = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}
TRACK_LABELS: dict[int, str] = {0: "不明", 1: "良", 2: "稍重", 3: "重", 4: "不良"}
SUMMER_MONTHS = {6, 7, 8, 9}
WEAK_COURSES = {3, 10}
HOKKAIDO_COURSES = {1, 2}


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def _build_race_table(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """レース単位の評価行を構築（Top-1 的中・NDCG@3 含む）。"""
    work = df.copy()
    work["pred_score"] = predictions
    has_horse_num = "horse_num" in work.columns

    rows: list[dict] = []
    for race_id, race in work.groupby("race_id", sort=False):
        if len(race) < 2:
            continue
        if has_horse_num:
            race_sorted = race.sort_values(
                ["pred_score", "horse_num"], ascending=[False, True]
            )
        else:
            race_sorted = race.sort_values("pred_score", ascending=False)

        meta = race.iloc[0]
        race_date = pd.Timestamp(meta["race_date"])
        month = int(race_date.month)
        course = int(meta["course_code"])
        track = int(meta["track_condition_code"])
        surface = int(meta["surface_code"])

        pred_order = race_sorted["finish_rank"].values
        is_hit = int(pred_order[0] == 1)
        ndcg3 = ndcg_at_k(
            race["lr_label"].values,
            race["pred_score"].values,
            k=3,
        )

        rows.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "month": month,
                "course_code": course,
                "course_name": COURSE_NAMES.get(course, f"course{course}"),
                "track_condition_code": track,
                "track_label": TRACK_LABELS.get(track, str(track)),
                "surface_code": surface,
                "is_summer_month": int(month in SUMMER_MONTHS),
                "is_heavy_track": int(track in (3, 4)),
                "is_hit": is_hit,
                "ndcg_at_3": float(ndcg3),
            }
        )
    return pd.DataFrame(rows)


def _period_overall(df_race: pd.DataFrame) -> dict:
    n = len(df_race)
    if n == 0:
        return {"top1_rate": None, "ndcg_at_3": None, "n_races": 0}
    return {
        "top1_rate": round(float(df_race["is_hit"].mean()), 6),
        "ndcg_at_3": round(float(df_race["ndcg_at_3"].mean()), 6),
        "n_races": int(n),
    }


def _aggregate_group(
    df_race: pd.DataFrame,
    group_cols: list[str],
    baseline: dict,
    *,
    min_cell_n: int,
) -> list[dict]:
    """セグメント集計。baseline は同一 period の overall。"""
    b_top1 = baseline.get("top1_rate")
    b_ndcg = baseline.get("ndcg_at_3")
    out: list[dict] = []

    for keys, grp in df_race.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(grp)
        top1 = float(grp["is_hit"].mean())
        ndcg = float(grp["ndcg_at_3"].mean())
        entry: dict = {
            "n_races": n,
            "top1_rate": round(top1, 6),
            "ndcg_at_3": round(ndcg, 6),
            "top1_delta_vs_period": round(top1 - b_top1, 6) if b_top1 is not None else None,
            "ndcg_delta_vs_period": round(ndcg - b_ndcg, 6) if b_ndcg is not None else None,
            "low_n_warning": n < min_cell_n,
        }
        for i, col in enumerate(group_cols):
            val = keys[i]
            entry[col] = int(val) if col in ("month", "course_code", "track_condition_code", "surface_code") else val
            if col == "course_code":
                entry["course_name"] = COURSE_NAMES.get(int(val), str(val))
            if col == "track_condition_code":
                entry["track_label"] = TRACK_LABELS.get(int(val), str(val))
        out.append(entry)
    return out


def _matrix_rows(
    segments: list[dict],
    period: str,
    *,
    axis: str,
) -> list[dict]:
    """CSV 用フラット行。"""
    rows = []
    for s in segments:
        row = {"period": period, "axis": axis, **s}
        rows.append(row)
    return rows


def _confounding_summary(valid_race: pd.DataFrame, test_race: pd.DataFrame) -> dict:
    """福島・小倉弱点のうち夏季月に集中する割合（VALID/TEST 別）。"""

    def _course_month_breakdown(df: pd.DataFrame, course: int) -> dict:
        sub = df[df["course_code"] == course]
        if len(sub) == 0:
            return {"n_races": 0}
        overall = float(sub["is_hit"].mean())
        by_month = (
            sub.groupby("month")["is_hit"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "top1_rate", "count": "n_races"})
        )
        summer = sub[sub["is_summer_month"] == 1]
        non_summer = sub[sub["is_summer_month"] == 0]
        return {
            "n_races": int(len(sub)),
            "overall_top1": round(overall, 6),
            "summer_months_top1": round(float(summer["is_hit"].mean()), 6) if len(summer) else None,
            "summer_months_n": int(len(summer)),
            "non_summer_top1": round(float(non_summer["is_hit"].mean()), 6) if len(non_summer) else None,
            "non_summer_n": int(len(non_summer)),
            "pct_races_in_summer_months": round(len(summer) / len(sub), 4) if len(sub) else None,
            "by_month": [
                {
                    "month": int(r["month"]),
                    "top1_rate": round(float(r["top1_rate"]), 6),
                    "n_races": int(r["n_races"]),
                }
                for _, r in by_month.iterrows()
            ],
        }

    def _period_block(df: pd.DataFrame) -> dict:
        overall = _period_overall(df)
        return {
            "period_overall": overall,
            "fukushima": _course_month_breakdown(df, 3),
            "kokura": _course_month_breakdown(df, 10),
            "hakodate": _course_month_breakdown(df, 2),
            "weak_courses_combined": {
                "n_races": int(len(df[df["course_code"].isin(WEAK_COURSES)])),
                "top1_rate": round(
                    float(df[df["course_code"].isin(WEAK_COURSES)]["is_hit"].mean()), 6
                )
                if len(df[df["course_code"].isin(WEAK_COURSES)])
                else None,
            },
            "hokkaido_combined": {
                "n_races": int(len(df[df["course_code"].isin(HOKKAIDO_COURSES)])),
                "top1_rate": round(
                    float(df[df["course_code"].isin(HOKKAIDO_COURSES)]["is_hit"].mean()), 6
                )
                if len(df[df["course_code"].isin(HOKKAIDO_COURSES)])
                else None,
            },
        }

    return {
        "valid": _period_block(valid_race),
        "test": _period_block(test_race),
    }


def _track_deterioration_freq(valid_race: pd.DataFrame, test_race: pd.DataFrame) -> dict:
    """夏競馬ドメイン参考: 月×コースごとの重/不良馬場出現率（能力指標ではなく分布診断）。"""

    def _freq(df: pd.DataFrame) -> list[dict]:
        if len(df) == 0:
            return []
        g = (
            df.groupby(["month", "course_code"])
            .agg(
                n_races=("race_id", "count"),
                heavy_rate=("is_heavy_track", "mean"),
            )
            .reset_index()
        )
        rows = []
        for _, r in g.iterrows():
            rows.append(
                {
                    "month": int(r["month"]),
                    "course_code": int(r["course_code"]),
                    "course_name": COURSE_NAMES.get(int(r["course_code"]), "?"),
                    "n_races": int(r["n_races"]),
                    "heavy_or_bad_rate": round(float(r["heavy_rate"]), 4),
                }
            )
        return rows

    return {"valid": _freq(valid_race), "test": _freq(test_race)}


def run_analysis(*, include_tier2: bool = True, min_cell_n: int = 20, tier2_min_n: int = 30) -> dict:
    cfg = load_config()
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    version = cfg["data"]["features_version"]
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])

    feat_path = feat_dir / f"features_{version}.parquet"
    print(f"Loading features: {feat_path.name}")
    df = _apply_filters(pd.read_parquet(feat_path), cfg)

    # VALID / TEST 完全分離（Rule 3）
    df_valid = df[df["race_date"] <= valid_end].copy()
    df_test = df[df["race_date"] > valid_end].copy()
    print(f"  VALID (<= {valid_end.date()}): {df_valid['race_id'].nunique():,} races")
    print(f"  TEST  (> {valid_end.date()}): {df_test['race_id'].nunique():,} races")

    feat_cols = get_feature_cols(df, cfg)
    print("Loading models...")
    models = load_models(models_dir)

    print("Predicting VALID...")
    pred_valid = ensemble_predict(models, df_valid[feat_cols])
    print("Predicting TEST...")
    pred_test = ensemble_predict(models, df_test[feat_cols])

    race_valid = _build_race_table(df_valid, pred_valid)
    race_test = _build_race_table(df_test, pred_test)

    valid_overall = _period_overall(race_valid)
    test_overall = _period_overall(race_test)

    print(f"\n=== P-30 Season×Course Analysis ({version}) ===")
    print(
        f"VALID overall: Top-1={valid_overall['top1_rate']:.1%}  "
        f"NDCG@3={valid_overall['ndcg_at_3']:.4f}  n={valid_overall['n_races']:,}"
    )
    print(
        f"TEST  overall: Top-1={test_overall['top1_rate']:.1%}  "
        f"NDCG@3={test_overall['ndcg_at_3']:.4f}  n={test_overall['n_races']:,}"
    )

    tier1_valid = _aggregate_group(
        race_valid, ["month", "course_code"], valid_overall, min_cell_n=min_cell_n
    )
    tier1_test = _aggregate_group(
        race_test, ["month", "course_code"], test_overall, min_cell_n=min_cell_n
    )

    tier2_valid: list[dict] = []
    tier2_test: list[dict] = []
    if include_tier2:
        tier2_valid = [
            s
            for s in _aggregate_group(
                race_valid,
                ["month", "course_code", "track_condition_code"],
                valid_overall,
                min_cell_n=tier2_min_n,
            )
            if s["n_races"] >= tier2_min_n
        ]
        tier2_test = [
            s
            for s in _aggregate_group(
                race_test,
                ["month", "course_code", "track_condition_code"],
                test_overall,
                min_cell_n=tier2_min_n,
            )
            if s["n_races"] >= tier2_min_n
        ]

    # 弱点セル抽出（VALID のみ — 仮説構築用。TEST は報告のみ）
    weak_valid = [
        s for s in tier1_valid
        if s["n_races"] >= min_cell_n
        and s["top1_delta_vs_period"] is not None
        and s["top1_delta_vs_period"] <= -0.05
    ]
    weak_valid.sort(key=lambda x: x["top1_delta_vs_period"])

    result: dict = {
        "meta": {
            "model_version": version,
            "valid_end": str(valid_end.date()),
            "rule3_note": "VALID=仮説構築・TEST=報告のみ。TEST結果で特徴量閾値を決定しないこと。",
            "min_cell_n_tier1": min_cell_n,
            "tier2_min_n": tier2_min_n if include_tier2 else None,
            "baseline_reference_test_top1": 0.3024,
        },
        "overall": {
            "valid": valid_overall,
            "test": test_overall,
        },
        "tier1_month_x_course": {
            "valid": tier1_valid,
            "test": tier1_test,
        },
        "tier2_month_x_course_x_track": {
            "valid": tier2_valid,
            "test": tier2_test,
        }
        if include_tier2
        else None,
        "confounding_decomposition": _confounding_summary(race_valid, race_test),
        "track_deterioration_frequency": _track_deterioration_freq(race_valid, race_test),
        "valid_hypothesis_candidates": weak_valid[:20],
    }

    # CSV マトリクス（long format）
    csv_rows: list[dict] = []
    csv_rows.extend(_matrix_rows(tier1_valid, "valid", axis="month_x_course"))
    csv_rows.extend(_matrix_rows(tier1_test, "test", axis="month_x_course"))
    if include_tier2:
        csv_rows.extend(_matrix_rows(tier2_valid, "valid", axis="month_x_course_x_track"))
        csv_rows.extend(_matrix_rows(tier2_test, "test", axis="month_x_course_x_track"))

    json_path = feat_dir / f"season_course_analysis_{version}.json"
    csv_path = feat_dir / f"season_course_matrix_{version}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n--- VALID 弱点セル (Top-1 Δ<=-5pp, n>={min_cell_n}) top 10 ---")
    for i, s in enumerate(weak_valid[:10], 1):
        warn = " [low-n]" if s["low_n_warning"] else ""
        print(
            f"  {i}. {s['course_name']} {s['month']}月: "
            f"Top-1={s['top1_rate']:.1%} (Δ{s['top1_delta_vs_period']:+.1%})  "
            f"NDCG@3={s['ndcg_at_3']:.3f}  n={s['n_races']}{warn}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="P-30 季節×コース交絡分析")
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="第2軸（月×コース×馬場状態）を出力しない",
    )
    parser.add_argument(
        "--min-cell-n",
        type=int,
        default=20,
        help="第1軸セルの low_n_warning 閾値（既定 20）",
    )
    parser.add_argument(
        "--tier2-min-n",
        type=int,
        default=30,
        help="第2軸セルの最小 n（未満は除外、既定 30）",
    )
    args = parser.parse_args()
    run_analysis(
        include_tier2=not args.no_tier2,
        min_cell_n=args.min_cell_n,
        tier2_min_n=args.tier2_min_n,
    )


if __name__ == "__main__":
    main()
