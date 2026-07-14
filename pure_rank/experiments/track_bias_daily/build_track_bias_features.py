"""同日・同場の先行レース結果から馬場バイアス（内外・前後）を日次推定する。

リーク対策: 各レースについて、同日・同場かつ race_num が小さい（=先に行われた）
レースの結果のみを使う（expanding + shift(1) で当該レース自身を除外）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
OUT_PATH = Path(__file__).resolve().parent / "data" / "track_bias_features.parquet"

MIN_TERCILE_N = 5
MIN_OVERALL_N = 15
MIN_FRONT_BACK_N = 5


def main() -> None:
    print(f"Loading: {FEATURES_PATH}")
    cols = ["race_id", "race_date", "course_code", "horse_num", "wakuban",
            "relative_post_position", "finish_rank", "corner_4", "horse_count",
            "hist_front_running_pref"]
    df = pd.read_parquet(FEATURES_PATH, columns=cols)
    df["race_num"] = df["race_id"].astype(str).str[-2:].astype(int)

    df["post_tercile"] = pd.cut(
        df["relative_post_position"], bins=[-0.01, 1 / 3, 2 / 3, 10],
        labels=["inner", "mid", "outer"],
    )

    df["corner4_rank_in_race"] = df.groupby("race_id")["corner_4"].rank(method="first")
    df["is_front"] = df["corner4_rank_in_race"] <= (df["horse_count"] / 2)

    # ─ race単位のサム/カウントを算出 ─
    tercile_agg = (
        df.groupby(["race_id", "post_tercile"], observed=True)["finish_rank"]
        .agg(["sum", "count"]).unstack("post_tercile")
    )
    tercile_agg.columns = [f"{t}_{stat}" for stat, t in tercile_agg.columns]

    overall_agg = df.groupby("race_id")["finish_rank"].agg(["sum", "count"])
    overall_agg.columns = ["overall_sum", "overall_count"]

    fb = df[df["is_front"].notna()].copy()
    fb_group = fb.groupby(["race_id", "is_front"])["finish_rank"].agg(["sum", "count"]).unstack("is_front")
    fb_group.columns = [f"{'front' if c else 'back'}_{s}" for s, c in fb_group.columns]

    race_meta = df.groupby("race_id").agg(
        race_date=("race_date", "first"), course_code=("course_code", "first"),
        race_num=("race_num", "first"),
    )

    race_level = race_meta.join([tercile_agg, overall_agg, fb_group]).fillna(0.0)
    race_level = race_level.sort_values(["race_date", "course_code", "race_num"])

    sum_cols = [c for c in race_level.columns if c.endswith("_sum") or c.endswith("_count")]
    grp = race_level.groupby(["race_date", "course_code"])
    cum_before = grp[sum_cols].cumsum() - race_level[sum_cols]  # 自分自身を除いた累積(先行レースのみ)
    cum_before.columns = [f"{c}_before" for c in cum_before.columns]
    race_level = pd.concat([race_level, cum_before], axis=1)

    def _avg(sum_col: str, cnt_col: str, min_n: int) -> pd.Series:
        cnt = race_level[cnt_col]
        avg = race_level[sum_col] / cnt.replace(0, np.nan)
        return avg.where(cnt >= min_n)

    overall_avg = _avg("overall_sum_before", "overall_count_before", MIN_OVERALL_N)
    for tercile in ["inner", "mid", "outer"]:
        tercile_avg = _avg(f"{tercile}_sum_before", f"{tercile}_count_before", MIN_TERCILE_N)
        race_level[f"post_bias_{tercile}"] = -(tercile_avg - overall_avg)

    front_avg = _avg("front_sum_before", "front_count_before", MIN_FRONT_BACK_N)
    back_avg = _avg("back_sum_before", "back_count_before", MIN_FRONT_BACK_N)
    race_level["pace_bias_race"] = back_avg - front_avg

    bias_cols = ["post_bias_inner", "post_bias_mid", "post_bias_outer", "pace_bias_race"]
    df = df.merge(race_level[bias_cols], left_on="race_id", right_index=True, how="left")

    df["post_bias_today"] = np.select(
        [df["post_tercile"] == "inner", df["post_tercile"] == "mid", df["post_tercile"] == "outer"],
        [df["post_bias_inner"], df["post_bias_mid"], df["post_bias_outer"]],
        default=np.nan,
    )

    # pace_bias_race を連続値のまま hist_front_running_pref に掛けると r=0.86 で
    # 相関ゲートに抵触（既存特徴量の再スケールに過ぎなくなる）。front_pref_x_small の
    # 成功パターン（連続値×二値フラグ）に倣い、pace_bias_race を三値化してから掛ける。
    q_low, q_high = df["pace_bias_race"].quantile([1 / 3, 2 / 3])
    pace_signal = pd.Series(0, index=df.index, dtype="float64")
    pace_signal[df["pace_bias_race"] >= q_high] = 1.0   # 先行有利な日
    pace_signal[df["pace_bias_race"] <= q_low] = -1.0   # 差し有利な日
    pace_signal[df["pace_bias_race"].isna()] = np.nan
    style_centered = df["hist_front_running_pref"] - df["hist_front_running_pref"].mean()
    df["pace_bias_x_style"] = pace_signal * style_centered

    print("\npost_bias_today 記述統計:")
    print(df["post_bias_today"].describe())
    print(f"NaN率: {df['post_bias_today'].isna().mean():.1%}")
    print("\npace_bias_x_style 記述統計:")
    print(df["pace_bias_x_style"].describe())
    print(f"NaN率: {df['pace_bias_x_style'].isna().mean():.1%}")

    from scipy.stats import spearmanr
    for col in ["post_bias_today", "pace_bias_x_style"]:
        sub = df.dropna(subset=[col, "finish_rank"])
        corr, p = spearmanr(sub[col], sub["finish_rank"])
        print(f"\n{col} vs finish_rank: spearman={corr:.4f} (p={p:.2e}, n={len(sub):,})")

    out = df[["race_id", "horse_num", "post_bias_today", "pace_bias_x_style"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
