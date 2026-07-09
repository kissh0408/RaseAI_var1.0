"""当日CSV系の race_id 生成・オッズ読み込みユーティリティ。

旧 model_training/src/prepare_db.py（アーカイブ済み: C:/Users/syugo/AI/_archive/
RaceAI_var1.0/layer2_legacy/model_training/src/prepare_db.py）から必要関数のみ
移植した。main/unified_pipeline.py の当日オッズ結合（_odds_from_se）はこのモジュール
にのみ依存し、builders/create_features_v3/v4/champion_features などアーカイブ済みの
var2.0.0 由来モジュールには依存しない。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_race_id_vec(df: pd.DataFrame) -> pd.Series:
    """year/month_day/course_code/kai/nichi/race_num から16桁race_idを生成する。"""
    return (
        df["year"].astype(int).astype(str).str.zfill(4)
        + df["month_day"].astype(int).astype(str).str.zfill(4)
        + df["course_code"].astype(int).astype(str).str.zfill(2)
        + df["kai"].astype(int).astype(str).str.zfill(2)
        + df["nichi"].astype(int).astype(str).str.zfill(2)
        + df["race_num"].astype(int).astype(str).str.zfill(2)
    )


def _make_race_date_vec(df: pd.DataFrame) -> pd.Series:
    """year/month_day から 'YYYY-MM-DD' を生成する。"""
    md = df["month_day"].astype(int).astype(str).str.zfill(4)
    return df["year"].astype(int).astype(str).str.zfill(4) + "-" + md.str[:2] + "-" + md.str[2:]


def _surface_from_track_code(track_code: int) -> int:
    """JV-Link track_code → surface_code (1=芝, 2=ダート, 3=障害)。"""
    tc = int(track_code) if not pd.isna(track_code) else 0
    if 10 <= tc <= 19:
        return 1
    elif 20 <= tc <= 29:
        return 2
    elif tc >= 50:
        return 3
    return 2


def _make_race_id_from_parts(df: pd.DataFrame) -> pd.Series:
    """realtime系CSV（year/month_day/course/kai/nichi/race_num列）からrace_idを構築する。"""
    for col in ["year", "month_day", "course_code", "kai", "nichi", "race_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return _make_race_id_vec(df)


def load_realtime_odds(path: Path) -> pd.DataFrame:
    """realtime_odds/o1_odds.csv → race_id, horse_num, odds（デシマル）。"""
    df = pd.read_csv(path)
    if "race_id" not in df.columns:
        df["race_id"] = _make_race_id_from_parts(df)
    df["race_id"] = df["race_id"].astype(str)
    df["odds"] = pd.to_numeric(df["odds_raw"], errors="coerce") / 10.0
    df["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce").astype(int)
    return df[["race_id", "horse_num", "odds"]].dropna(subset=["odds"])
