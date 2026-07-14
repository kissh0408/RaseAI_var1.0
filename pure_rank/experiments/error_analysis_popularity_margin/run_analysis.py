"""error_analysis_popularity_margin: run_analysis.py

診断1: 人気帯別（1人気/2人気/3人気/4人気/5人気以上）にモデルの的中・見逃し傾向を分解する。
診断2: 予測1位馬が外れたケースの、勝ち馬とのタイム差（僅差ニアミス）分布を見る。

市場情報（オッズ由来の人気順位）は評価レイヤーの診断のみに使用し、学習・特徴量には
一切投入しない（README参照）。TEST期間（2025-01-01以降）のみを対象とする。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting.src.backtest import load_scored_odds_frame  # noqa: E402

SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim_fold2_oos.parquet"
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
TEST_START = "2025-01-01"
OUT_PATH = EXP_DIR / "results" / "analysis.json"

POP_BUCKETS = ["1", "2", "3", "4", "5+"]


def _pop_bucket(pop: int) -> str:
    return str(pop) if pop <= 4 else "5+"


def _attach_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """odds昇順の順位を人気とする（評価専用の派生列。特徴量には使わない）。"""
    out = df.copy()
    out["popularity"] = (
        out.groupby("race_id")["odds"].rank(method="first", ascending=True).astype(int)
    )
    return out


def _attach_pred_rank(df: pd.DataFrame) -> pd.DataFrame:
    """レース内で pure_score_z 降順の順位（予測順位）を付与する。"""
    out = df.copy()
    out["pred_rank"] = (
        out.groupby("race_id")["pure_score_z"].rank(method="first", ascending=False).astype(int)
    )
    return out


def popularity_breakdown(df: pd.DataFrame) -> list[dict]:
    """診断1: 人気帯別の base rate（実際の上位入賞率）と model catch rate。"""
    rows = []
    for bucket in POP_BUCKETS:
        if bucket == "5+":
            sub = df[df["popularity"] >= 5]
        else:
            sub = df[df["popularity"] == int(bucket)]
        n = int(len(sub))
        if n == 0:
            continue
        top3_actual = sub["finish_rank"] <= 3
        n_top3_actual = int(top3_actual.sum())
        base_rate_top3 = float(top3_actual.mean())

        top3_pred = sub["pred_rank"] <= 3
        # catch: 実際top3 かつ モデルもtop3に入れていた
        n_caught = int((top3_actual & top3_pred).sum())
        catch_rate = float(n_caught / n_top3_actual) if n_top3_actual > 0 else None

        win_actual = sub["finish_rank"] == 1
        n_win_actual = int(win_actual.sum())
        win_pred_rank1 = sub["pred_rank"] == 1
        n_win_caught = int((win_actual & win_pred_rank1).sum())
        win_catch_rate = float(n_win_caught / n_win_actual) if n_win_actual > 0 else None

        avg_pred_rank_of_bucket = float(sub["pred_rank"].mean())

        rows.append(
            {
                "popularity_bucket": bucket,
                "n_horses": n,
                "base_rate_top3_actual": base_rate_top3,
                "n_top3_actual": n_top3_actual,
                "model_top3_catch_rate": catch_rate,
                "n_win_actual": n_win_actual,
                "model_win_catch_rate_pred_rank1": win_catch_rate,
                "avg_model_pred_rank": avg_pred_rank_of_bucket,
            }
        )
    return rows


def margin_breakdown(df: pd.DataFrame) -> dict:
    """診断2: 予測1位馬がハズレたケースの、勝ち馬とのタイム差分布。"""
    pred_top1 = df[df["pred_rank"] == 1].copy()
    misses = pred_top1[pred_top1["finish_rank"] != 1].copy()

    winner_time = (
        df[df["finish_rank"] == 1]
        .drop_duplicates(subset=["race_id"], keep="first")
        .set_index("race_id")["racetime"]
    )
    misses["winner_time"] = misses["race_id"].map(winner_time)
    misses["time_diff"] = misses["racetime"] - misses["winner_time"]
    misses = misses.dropna(subset=["time_diff"])

    n_total_pred_top1 = int(len(pred_top1))
    n_misses = int(len(misses))
    n_hits = n_total_pred_top1 - n_misses

    bins = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, float("inf")]
    labels = ["<=0.1s", "0.1-0.2s", "0.2-0.3s", "0.3-0.5s", "0.5-1.0s", ">1.0s"]
    misses_second_only = misses[misses["finish_rank"] == 2]
    dist = pd.cut(misses_second_only["time_diff"], bins=bins, labels=labels, right=True)
    dist_counts = dist.value_counts().reindex(labels).fillna(0).astype(int).to_dict()

    n_second_place_misses = int(len(misses_second_only))
    n_nose_le_0_1 = int((misses_second_only["time_diff"] <= 0.1).sum())
    n_narrow_le_0_3 = int((misses_second_only["time_diff"] <= 0.3).sum())

    return {
        "n_pred_top1_total": n_total_pred_top1,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "top1_rate": float(n_hits / n_total_pred_top1) if n_total_pred_top1 else None,
        "n_misses_finished_2nd": n_second_place_misses,
        "second_place_miss_time_diff_distribution": dist_counts,
        "nose_miss_rate_le_0.1s_of_2nd_misses": (
            float(n_nose_le_0_1 / n_second_place_misses) if n_second_place_misses else None
        ),
        "narrow_miss_rate_le_0.3s_of_2nd_misses": (
            float(n_narrow_le_0_3 / n_second_place_misses) if n_second_place_misses else None
        ),
        "nose_miss_rate_le_0.1s_of_all_misses": (
            float(n_nose_le_0_1 / n_misses) if n_misses else None
        ),
        "narrow_miss_rate_le_0.3s_of_all_misses": (
            float(n_narrow_le_0_3 / n_misses) if n_misses else None
        ),
    }


def run_analysis() -> dict:
    df = load_scored_odds_frame(SCORES_PATH, FEATURES_PATH)
    df["race_id"] = df["race_id"].astype(str)
    df = df.dropna(subset=["odds"])
    df = df[pd.to_numeric(df["odds"], errors="coerce") > 0]

    dates = pd.to_datetime(df["race_date"])
    test_df = df.loc[dates >= pd.Timestamp(TEST_START)].copy()

    feat = pd.read_parquet(FEATURES_PATH, columns=["race_id", "horse_num", "racetime"])
    feat["race_id"] = feat["race_id"].astype(str)
    test_df = test_df.merge(feat, on=["race_id", "horse_num"], how="left")

    test_df = _attach_popularity(test_df)
    test_df = _attach_pred_rank(test_df)

    n_races = int(test_df["race_id"].nunique())
    n_rows = int(len(test_df))

    result = {
        "protocol": {
            "test_period": f"{TEST_START}..",
            "score_col": "pure_score_z",
            "n_races": n_races,
            "n_rows": n_rows,
            "popularity_definition": "odds昇順のレース内順位（診断専用の評価派生列。学習・特徴量には未使用）",
            "note": "本診断は評価レイヤー限定でオッズ由来の人気を使用する（CLAUDE.md Rule 1は特徴量投入の禁止であり事後診断は対象外。evaluation/market_baseline.pyのfavorite baseline測定と同じ扱い）。",
        },
        "diagnostic_1_popularity_breakdown": popularity_breakdown(test_df),
        "diagnostic_2_margin_breakdown": margin_breakdown(test_df),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_analysis()
