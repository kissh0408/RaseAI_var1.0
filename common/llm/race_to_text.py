"""
race_to_text.py — レースデータをLLM入力テキストに変換するモジュール

情報漏洩禁止カラム: finish_rank, time, hon_shokin, fuka_shokin,
time_4f_after, time_3f_after, time_diff, mining_* は入力に含めない。
"""

from __future__ import annotations

import os
import pandas as pd
from typing import Optional

# データ所在
DATA_OUTPUT_DIR = r"C:\Users\syugo\AI\RaceAI\common\data\output"

# コードマッピング
COURSE_CODE_MAP = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}

TRACK_CODE_MAP = {
    10: "芝(右)",
    11: "芝(右外)",
    12: "芝(直線)",
    13: "芝(左)",
    14: "芝(左外)",
    15: "芝(左・内→外)",
    17: "芝(右・内→外)",
    18: "芝(直線・外)",
    19: "障害(芝・右)",
    20: "芝(直線・内)",
    21: "芝(右・外→内)",
    22: "芝(左・外→内)",
    23: "芝(左・内→外)",
    24: "ダート(右)",
    25: "ダート(右外)",
    26: "ダート(直線)",
    27: "ダート(左)",
    28: "ダート(左外)",
    29: "障害(ダート・右)",
}

WEATHER_CODE_MAP = {
    1: "晴", 2: "曇", 3: "小雨", 4: "雨", 5: "小雪", 6: "雪", 7: "霧",
}

TURF_CONDITION_MAP = {0: "良", 1: "稍重", 2: "重", 3: "不良"}
DIRT_CONDITION_MAP = {0: "良", 1: "稍重", 2: "重", 3: "不良"}

SEX_CODE_MAP = {1: "牡", 2: "牝", 3: "騸", 4: "牝", 5: "騸"}  # 4=牝(障害), 5=騸(障害)


def make_race_id(row) -> str:
    """
    race_idを生成する。
    フォーマット: YYYY + month_day(4桁) + course_code(2桁) + kai(2桁) + nichi(2桁) + race_num(2桁)
    """
    return (
        f"{int(row['year'])}"
        f"{int(row['month_day']):04d}"
        f"{int(row['course_code']):02d}"
        f"{int(row['kai']):02d}"
        f"{int(row['nichi']):02d}"
        f"{int(row['race_num']):02d}"
    )


def load_race_data(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    指定年の race_se と race_ra を読み込み、race_id を付与して返す。

    Returns:
        (df_se, df_ra): 出走馬データ, レース情報データ
    """
    se_path = os.path.join(DATA_OUTPUT_DIR, "race_se", f"race_se_{year}.csv")
    ra_path = os.path.join(DATA_OUTPUT_DIR, "race_ra", f"race_ra_{year}.csv")

    df_se = pd.read_csv(se_path, low_memory=False)
    df_ra = pd.read_csv(ra_path, low_memory=False)

    df_se["race_id"] = df_se.apply(make_race_id, axis=1)
    df_ra["race_id"] = df_ra.apply(make_race_id, axis=1)

    return df_se, df_ra


def _get_track_type(track_code: int) -> str:
    """トラックコードから芝/ダートを判定。"""
    label = TRACK_CODE_MAP.get(track_code, f"トラック{track_code}")
    if "ダート" in label:
        return "ダート"
    elif "障害" in label:
        return "障害"
    return "芝"


def _get_condition(ra_row) -> str:
    """馬場状態文字列を返す。"""
    track_code = int(ra_row.get("track_code", 0))
    track_type = _get_track_type(track_code)
    if track_type == "ダート":
        cond = DIRT_CONDITION_MAP.get(int(ra_row.get("dirt_condition", 0)), "不明")
    else:
        cond = TURF_CONDITION_MAP.get(int(ra_row.get("turf_condition", 0)), "不明")
    return cond


def race_to_prompt(ra_row, horses_df: pd.DataFrame) -> str:
    """
    コンパクト形式のプロンプトを生成する (目標 ~200トークン)。
    情報漏洩禁止: finish_rank, time, hon_shokin, fuka_shokin,
    time_4f_after, time_3f_after, time_diff, mining_* は含めない。
    """
    course_code = int(ra_row.get("course_code", 0))
    course_name = COURSE_CODE_MAP.get(course_code, str(course_code))
    distance = int(ra_row.get("distance", 0))
    track_code = int(ra_row.get("track_code", 0))
    condition = _get_condition(ra_row)
    running_count = int(ra_row.get("running_count", 0))
    track_type = _get_track_type(track_code)

    lines = [f"R:{course_name} {track_type}{distance}m {condition} {running_count}h"]

    valid_horses = horses_df[horses_df["abnormal_code"] == 0].sort_values("horse_num")
    for _, h in valid_horses.iterrows():
        horse_num = int(h["horse_num"])
        sex_raw = h.get("sex_code", 1)
        sex = SEX_CODE_MAP.get(int(sex_raw) if pd.notna(sex_raw) else 1, "?")
        age = int(h["age"]) if pd.notna(h.get("age")) else 0
        burden_kg = (int(h["burden_weight"]) if pd.notna(h.get("burden_weight")) else 0) / 10.0
        jockey_code = int(h["jockey_code"]) if pd.notna(h.get("jockey_code")) else 0
        hw = int(h["horse_weight"]) if pd.notna(h.get("horse_weight")) else 0
        sign = str(h.get("weight_change_sign", "+")).strip() if pd.notna(h.get("weight_change_sign")) else "+"
        change = int(h["weight_change"]) if pd.notna(h.get("weight_change")) else 0
        wdiff = f"{sign}{change}"
        odds_raw = h.get("odds", 0)
        odds_val = int(odds_raw) / 10.0 if pd.notna(odds_raw) and odds_raw != 0 else 0.0
        pop = int(h["popularity"]) if pd.notna(h.get("popularity")) else 0
        lines.append(f"H{horse_num}:{sex}{age} {burden_kg:.0f}kg J{jockey_code} W{hw}({wdiff}) {odds_val:.1f}x{pop}p")

    return "\n".join(lines)


def build_label(horses_df: pd.DataFrame, race_id: str) -> list[dict]:
    """
    着順とオッズからラベルを生成する。

    rank_score = (出走頭数 - finish_rank + 1) / 出走頭数
        → 1着が最大値 1.0、最下位が 1/N
    ev_score = (単勝オッズ ÷ 10) × (1 / finish_rank) を正規化
        → 全馬の ev_score の合計で除算して相対値に

    abnormal_code != 0 の馬は除外。
    finish_rank == 0 の馬(完走タイム未確定等)も除外。

    Returns:
        [{"horse_num": int, "rank_score": float, "ev_score": float}, ...]
        rank_score の降順でソート済み
    """
    valid = horses_df[
        (horses_df["abnormal_code"] == 0) &
        (horses_df["finish_rank"] > 0)
    ].copy()

    if valid.empty:
        return []

    n = len(valid)

    valid["rank_score"] = (n - valid["finish_rank"] + 1) / n
    valid["odds_val"] = valid["odds"] / 10.0
    valid["ev_raw"] = valid["odds_val"] * (1.0 / valid["finish_rank"])

    ev_sum = valid["ev_raw"].sum()
    if ev_sum > 0:
        valid["ev_score"] = valid["ev_raw"] / ev_sum
    else:
        valid["ev_score"] = 0.0

    result = []
    for _, row in valid.sort_values("rank_score", ascending=False).iterrows():
        result.append({
            "horse_num": int(row["horse_num"]),
            "rank_score": round(float(row["rank_score"]), 4),
            "ev_score": round(float(row["ev_score"]), 4),
        })

    return result
