"""
analyze_course_weakness.py — 福島・小倉（および小回り4場）の構造弱点診断

テストセットでの予測ミスパターンと、学習+valid期間の統計を出力する。
特徴量・モデルは変更しない（分析のみ）。

出力: pure_rank/data/02_features/course_weakness_{features_version}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, get_feature_cols, load_config
from create_features import SMALL_COURSE_CODES
from evaluate import ensemble_predict, load_models

COURSE_NAMES: dict[int, str] = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}

SURFACE_LABELS = {1: "芝", 2: "ダート", 5: "その他"}
DIST_LABELS = {0: "短距離", 1: "マイル", 2: "中距離", 3: "長距離"}
TRACK_LABELS = {0: "不明", 1: "良", 2: "稍重", 3: "重", 4: "不良"}

WEAK_COURSES = {3, 10}
FEATURE_AVAIL_COLS = [
    "hist_same_course_win_rate",
    "hist_same_course_dist_win_rate",
    "hist_jockey_course_win_rate",
    "hist_front_running_pref",
]

RUNNING_STYLE_LABELS = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込", 0: "不明"}


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def _horse_count_bucket(hc: int) -> str:
    if hc <= 8:
        return "5-8"
    if hc <= 12:
        return "9-12"
    if hc <= 16:
        return "13-16"
    return "17+"


def _top1_by_group(df_race: pd.DataFrame, group_cols: list[str]) -> dict:
    """レース単位 DataFrame を group_cols で集計し Top-1 率を返す。"""
    out: dict = {}
    for keys, grp in df_race.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_str = "|".join(str(k) for k in keys)
        rate = grp["is_hit"].mean()
        n = len(grp)
        out[key_str] = {
            "top1_rate": round(float(rate), 6),
            "n_races": int(n),
            "keys": {col: str(keys[i]) for i, col in enumerate(group_cols)},
        }
    return out


def _nan_stats(sub: pd.DataFrame, cols: list[str]) -> dict:
    stats: dict = {}
    for col in cols:
        if col not in sub.columns:
            continue
        s = sub[col]
        stats[col] = {
            "nan_rate": round(float(s.isna().mean()), 4),
            "mean": round(float(s.mean()), 4) if s.notna().any() else None,
            "std": round(float(s.std()), 4) if s.notna().sum() > 1 else None,
        }
    return stats


def _build_race_frame(df_test: pd.DataFrame) -> pd.DataFrame:
    """馬単位 test DataFrame からレース単位の診断フレームを構築する。"""
    race_rows = []
    for race_id, grp in df_test.groupby("race_id"):
        pred_best = grp.loc[grp["pred_score"].idxmax()]
        actual_winner = grp.loc[grp["finish_rank"].idxmin()]
        meta = grp.iloc[0]
        hc = int(meta["horse_count"])
        is_hit = int(pred_best["finish_rank"] == 1)

        pred_sorted = grp.sort_values("pred_score", ascending=False)
        score_gap_12 = float(
            pred_sorted["pred_score"].iloc[0] - pred_sorted["pred_score"].iloc[1]
        ) if len(pred_sorted) >= 2 else np.nan

        race_rows.append({
            "race_id": race_id,
            "is_hit": is_hit,
            "course_code": int(meta["course_code"]),
            "surface_code": int(meta["surface_code"]),
            "track_condition_code": int(meta["track_condition_code"]),
            "distance_category": int(meta["distance_category"]),
            "horse_count_bucket": _horse_count_bucket(hc),
            "horse_count": hc,
            "pred_winner_finish": int(pred_best["finish_rank"]),
            "pred_winner_wakuban": int(pred_best["wakuban"]),
            "actual_winner_wakuban": int(actual_winner["wakuban"]),
            "actual_winner_running_style": int(actual_winner.get("running_style_code", 0) or 0),
            "pred_winner_running_style": int(pred_best.get("running_style_code", 0) or 0),
            "score_gap_top2": score_gap_12,
            "pred_winner_same_course_nan": bool(
                pd.isna(pred_best.get("hist_same_course_win_rate"))
            ),
            "actual_winner_same_course_nan": bool(
                pd.isna(actual_winner.get("hist_same_course_win_rate"))
            ),
            "pred_winner_front_pref": pred_best.get("hist_front_running_pref"),
            "actual_winner_front_pref": actual_winner.get("hist_front_running_pref"),
            "is_small_course": int(meta["course_code"]) in SMALL_COURSE_CODES,
        })
    return pd.DataFrame(race_rows)


def _miss_classification(df_race: pd.DataFrame, course_codes: set[int]) -> dict:
    """外れレースのミス分類タイプを集計する。"""
    sub = df_race[df_race["course_code"].isin(course_codes) & (df_race["is_hit"] == 0)]
    if len(sub) == 0:
        return {"n_miss": 0}

    pred_fav_lost = int((sub["pred_winner_finish"] <= 3).sum())
    pred_long_lost = int((sub["pred_winner_finish"] > 3).sum())
    actual_front_won = int(sub["actual_winner_running_style"].isin([1, 2]).sum())
    pred_front_picked = int(sub["pred_winner_running_style"].isin([1, 2]).sum())
    low_confidence = int((sub["score_gap_top2"] < sub["score_gap_top2"].median()).sum())

    return {
        "n_miss": int(len(sub)),
        "pred_top3_but_not_win": pred_fav_lost,
        "pred_outside_top3": pred_long_lost,
        "actual_winner_front_runner": actual_front_won,
        "pred_picked_front_runner": pred_front_picked,
        "low_confidence_misses": low_confidence,
        "pred_winner_same_course_nan_rate": round(
            float(sub["pred_winner_same_course_nan"].mean()), 4
        ),
        "actual_winner_same_course_nan_rate": round(
            float(sub["actual_winner_same_course_nan"].mean()), 4
        ),
        "median_pred_winner_finish": round(float(sub["pred_winner_finish"].median()), 2),
        "median_score_gap_top2": round(float(sub["score_gap_top2"].median()), 4),
    }


def _running_style_bias(df_test: pd.DataFrame, course_code: int) -> dict:
    """場別: 脚質別の実際1着率 vs モデルがその脚質を1位予測したレースの的中率。"""
    sub = df_test[df_test["course_code"] == course_code]
    if len(sub) == 0:
        return {}

    # 実際の1着率（脚質別）
    winners = sub[sub["finish_rank"] == 1]
    actual_by_style: dict = {}
    for style, grp in winners.groupby("running_style_code"):
        style_key = int(style) if pd.notna(style) else 0
        n_races_style = sub.groupby("race_id").ngroups
        actual_by_style[str(style_key)] = {
            "label": RUNNING_STYLE_LABELS.get(style_key, str(style_key)),
            "win_share": round(len(grp) / max(len(winners), 1), 4),
            "n_wins": int(len(grp)),
        }

    # モデルTop-1予測の脚質別的中率
    model_by_style: dict = {}
    for race_id, grp in sub.groupby("race_id"):
        pred_idx = grp["pred_score"].idxmax()
        style = int(grp.loc[pred_idx, "running_style_code"] or 0)
        hit = int(grp.loc[pred_idx, "finish_rank"] == 1)
        if str(style) not in model_by_style:
            model_by_style[str(style)] = {"hits": 0, "n": 0, "label": RUNNING_STYLE_LABELS.get(style, str(style))}
        model_by_style[str(style)]["n"] += 1
        model_by_style[str(style)]["hits"] += hit

    for k, v in model_by_style.items():
        v["top1_rate"] = round(v["hits"] / v["n"], 4) if v["n"] > 0 else None
        del v["hits"]

    return {
        "actual_winner_by_running_style": actual_by_style,
        "model_top1_by_predicted_running_style": model_by_style,
        "n_races": int(sub.groupby("race_id").ngroups),
    }


def _train_period_stats(df_train_valid: pd.DataFrame) -> dict:
    """学習+valid期間のみ: 場別統計（特徴量設計の閾値決定用）。"""
    stats: dict = {}
    for cc in sorted(SMALL_COURSE_CODES | WEAK_COURSES):
        sub = df_train_valid[df_train_valid["course_code"] == cc]
        if len(sub) == 0:
            continue
        n_races = sub["race_id"].nunique()
        n_days = sub["race_date"].nunique()
        front_win = sub[(sub["finish_rank"] == 1) & sub["running_style_code"].isin([1, 2])]
        n_wins = int((sub["finish_rank"] == 1).sum())
        stats[str(cc)] = {
            "course_name": COURSE_NAMES.get(cc, str(cc)),
            "n_samples": int(len(sub)),
            "n_races": int(n_races),
            "n_race_days": int(n_days),
            "front_runner_win_rate": round(
                len(front_win) / max(n_wins, 1), 4
            ),
            "wakuban_win_rate": _wakuban_win_rates(sub),
            "hist_same_course_nan_rate": round(
                float(sub["hist_same_course_win_rate"].isna().mean()), 4
            ) if "hist_same_course_win_rate" in sub.columns else None,
        }
    return stats


def _wakuban_win_rates(sub: pd.DataFrame) -> dict:
    winners = sub[sub["finish_rank"] == 1]
    out: dict = {}
    for waku, grp in winners.groupby("wakuban"):
        out[str(int(waku))] = round(len(grp) / max(len(winners), 1), 4)
    return out


def main() -> None:
    cfg = load_config()
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    version = cfg["data"]["features_version"]
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])

    feat_path = feat_dir / f"features_{version}.parquet"
    print(f"Loading features: {feat_path.name}")
    df = pd.read_parquet(feat_path)
    df = _apply_filters(df, cfg)

    df_test = df[df["race_date"] > valid_end].copy()
    df_train_valid = df[df["race_date"] <= valid_end].copy()
    print(f"Test: {len(df_test):,} rows / {df_test['race_id'].nunique():,} races")
    print(f"Train+Valid: {len(df_train_valid):,} rows")

    feat_cols = get_feature_cols(df_test, cfg)
    print("Loading models...")
    models = load_models(models_dir)
    df_test = df_test.copy()
    df_test["pred_score"] = ensemble_predict(models, df_test[feat_cols])

    df_race = _build_race_frame(df_test)
    overall_top1 = df_race["is_hit"].mean()

    result: dict = {
        "model_version": version,
        "overall_top1": round(float(overall_top1), 6),
        "n_test_races": int(len(df_race)),
        "block_a_weak_course_crosstab": {},
        "block_b_small_course_comparison": {},
        "block_c_feature_availability": {},
        "block_d_miss_classification": {},
        "block_e_running_style_bias": {},
        "block_f_train_period_stats": _train_period_stats(df_train_valid),
        "hypothesis_signals": {},
    }

    # ── Block A: 福島・小倉 場内クロス集計 ─────────────────────────────────────
    print("\n=== Block A: Weak course crosstab ===")
    for cc in sorted(WEAK_COURSES):
        sub = df_race[df_race["course_code"] == cc]
        cc_key = str(cc)
        result["block_a_weak_course_crosstab"][cc_key] = {
            "course_name": COURSE_NAMES[cc],
            "overall_top1": round(float(sub["is_hit"].mean()), 6),
            "n_races": int(len(sub)),
            "by_surface": _top1_by_group(sub, ["surface_code"]),
            "by_distance": _top1_by_group(sub, ["distance_category"]),
            "by_horse_count": _top1_by_group(sub, ["horse_count_bucket"]),
            "by_track_condition": _top1_by_group(sub, ["track_condition_code"]),
            "by_surface_distance": _top1_by_group(sub, ["surface_code", "distance_category"]),
            "by_surface_horse_count": _top1_by_group(sub, ["surface_code", "horse_count_bucket"]),
        }
        print(f"  {COURSE_NAMES[cc]}({cc}): Top-1={sub['is_hit'].mean():.1%} n={len(sub)}")

    # ── Block B: 小回り4場比較 ────────────────────────────────────────────────
    print("\n=== Block B: Small course comparison ===")
    for cc in sorted(SMALL_COURSE_CODES):
        sub = df_race[df_race["course_code"] == cc]
        cc_key = str(cc)
        result["block_b_small_course_comparison"][cc_key] = {
            "course_name": COURSE_NAMES[cc],
            "top1_rate": round(float(sub["is_hit"].mean()), 6),
            "n_races": int(len(sub)),
            "gap_vs_overall": round(float(sub["is_hit"].mean() - overall_top1), 6),
            "by_surface": _top1_by_group(sub, ["surface_code"]),
            "by_distance": _top1_by_group(sub, ["distance_category"]),
            "median_score_gap_top2": round(float(sub["score_gap_top2"].median()), 4),
        }
        gap = sub["is_hit"].mean() - overall_top1
        print(f"  {COURSE_NAMES[cc]}: {sub['is_hit'].mean():.1%} ({gap:+.1%}) n={len(sub)}")

    # ── Block C: 特徴量可用性（馬単位） ─────────────────────────────────────
    print("\n=== Block C: Feature availability ===")
    for label, codes in [("weak", WEAK_COURSES), ("other", set(range(1, 11)) - WEAK_COURSES)]:
        sub = df_test[df_test["course_code"].isin(codes)]
        result["block_c_feature_availability"][label] = {
            "n_samples": int(len(sub)),
            "courses": sorted(codes),
            **_nan_stats(sub, FEATURE_AVAIL_COLS),
        }
        print(f"  {label}: n={len(sub):,}")
        for col in FEATURE_AVAIL_COLS:
            if col in result["block_c_feature_availability"][label]:
                nr = result["block_c_feature_availability"][label][col]["nan_rate"]
                print(f"    {col} NaN={nr:.1%}")

    per_course_nan: dict = {}
    for cc in sorted(SMALL_COURSE_CODES):
        sub = df_test[df_test["course_code"] == cc]
        per_course_nan[str(cc)] = {
            "course_name": COURSE_NAMES[cc],
            **_nan_stats(sub, FEATURE_AVAIL_COLS),
        }
    result["block_c_feature_availability"]["per_small_course"] = per_course_nan

    # ── Block D: ミス分類タイプ ─────────────────────────────────────────────
    print("\n=== Block D: Miss classification ===")
    for cc in sorted(WEAK_COURSES):
        cc_key = str(cc)
        miss = _miss_classification(df_race, {cc})
        result["block_d_miss_classification"][cc_key] = {
            "course_name": COURSE_NAMES[cc],
            **miss,
        }
        print(f"  {COURSE_NAMES[cc]}: {miss.get('n_miss', 0)} misses, "
              f"pred_top3_not_win={miss.get('pred_top3_but_not_win', 0)}")

    result["block_d_miss_classification"]["weak_combined"] = _miss_classification(
        df_race, WEAK_COURSES
    )

    # ── Block E: 先行脚質バイアス ─────────────────────────────────────────────
    print("\n=== Block E: Running style bias ===")
    for cc in sorted(SMALL_COURSE_CODES):
        cc_key = str(cc)
        bias = _running_style_bias(df_test, cc)
        result["block_e_running_style_bias"][cc_key] = {
            "course_name": COURSE_NAMES[cc],
            **bias,
        }

    # ── 仮説シグナル（planner 向けサマリー） ─────────────────────────────────
    weak_top1 = df_race[df_race["course_code"].isin(WEAK_COURSES)]["is_hit"].mean()
    hokkaido_top1 = df_race[df_race["course_code"].isin({1, 2})]["is_hit"].mean()
    weak_nan = result["block_c_feature_availability"]["weak"]["hist_same_course_win_rate"]["nan_rate"]
    other_nan = result["block_c_feature_availability"]["other"]["hist_same_course_win_rate"]["nan_rate"]

    result["hypothesis_signals"] = {
        "H1_course_sparsity": {
            "weak_courses_nan_rate": weak_nan,
            "other_courses_nan_rate": other_nan,
            "delta_nan": round(weak_nan - other_nan, 4),
            "signal": "strong" if weak_nan - other_nan > 0.05 else "weak",
        },
        "H2_waku_bias": {
            "fukushima_wakuban_win_rate_train": result["block_f_train_period_stats"]
            .get("3", {}).get("wakuban_win_rate", {}),
            "kokura_wakuban_win_rate_train": result["block_f_train_period_stats"]
            .get("10", {}).get("wakuban_win_rate", {}),
        },
        "H3_pace_density": {
            "weak_combined_front_winner_share": result["block_d_miss_classification"]
            .get("weak_combined", {}).get("actual_winner_front_runner"),
            "weak_combined_n_miss": result["block_d_miss_classification"]
            .get("weak_combined", {}).get("n_miss"),
        },
        "H4_fukushima_vs_kokura": {
            "fukushima_top1": result["block_b_small_course_comparison"]["3"]["top1_rate"],
            "kokura_top1": result["block_b_small_course_comparison"]["10"]["top1_rate"],
            "hokkaido_top1": round(float(hokkaido_top1), 6),
            "weak_top1": round(float(weak_top1), 6),
        },
        "recommended_priority": [],
    }

    # 優先度自動推薦（診断のみ、特徴量閾値決定には使わない）
    rec: list[str] = []
    if weak_nan - other_nan > 0.05:
        rec.append("H1: hist_small_course_pool_win_rate_ts")
    miss_combined = result["block_d_miss_classification"]["weak_combined"]
    if miss_combined.get("actual_winner_front_runner", 0) > miss_combined.get("n_miss", 1) * 0.5:
        rec.append("H3: front_pref_x_pace (orthogonalized)")
    rec.append("H2: relative_post_position_x_small")
    result["hypothesis_signals"]["recommended_priority"] = rec

    out_path = feat_dir / f"course_weakness_{version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
