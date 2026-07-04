"""
common.py — RaceAI_var1.0 共通モジュール

train.py / evaluate.py / create_features.py 等に重複していた
設定読み込み・特徴量列選択・禁止列定義・group 配列生成を一元管理する。

禁止列リストは本モジュールが唯一の真実（single source of truth）。
学習と評価で特徴量セットが食い違う事故を防ぐため、
各スクリプトで独自に定義してはならない。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ─── パス解決 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "pure_rank" / "config" / "train_config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── 禁止列定義 ────────────────────────────────────────────────────────────────
# 市場情報（絶対禁止）。create_features.py の混入チェックはこの集合を使う。
# 万一これらの列が DataFrame に存在した場合は生成段階で即エラーにする。
FORBIDDEN_MARKET_COLS: frozenset[str] = frozenset({
    "odds", "popularity", "win_odds", "place_odds",
    "quinella_odds", "market_prob", "market_log_odds",
    "init_score", "ninki",
})

# 学習・評価の特徴量から除外する全列（市場情報 + メタ列 + 後出し情報 + ラベル）。
FORBIDDEN_COLS: frozenset[str] = FORBIDDEN_MARKET_COLS | frozenset({
    # 一時作業列
    "_time_dev",
    # RA / SE のメタ列（特徴量として不要）
    "year", "month_day", "kai", "nichi", "race_num",
    "horse_num", "registered_count", "finish_count",
    "race_type_code", "weight_type", "race_condition_code",
    "race_level", "race_age_type", "course_kubun",
    "track_code",
    "obstacle_mile_time_sec",
    "dead_heat_flag", "dead_heat_count",
    "breed_code", "region_code",
    # 血統 ID（文字列。特徴量としては派生した win_rate 系を使う）
    "sire_id", "bms_id",
    # ─── レース後にしか判明しない後出し情報（特徴量にしてはならない） ───
    # 走破タイム・上がり3F（結果。hist_ 系経由で過去走データは使用可）
    "racetime", "time_3f_after",
    # コーナー通過順（レース中の位置情報。結果）
    "corner_1", "corner_2", "corner_3", "corner_4",
    # 脚質判定（レース後判定）
    "running_style_code",
    # 異常区分（レース後確定）
    "abnormal_code",
    # 賞金（レース後確定。hist_ 系経由で過去走データは使用可）
    "hon_shokin", "fuka_shokin",
    # 生ラベル（全てレース後確定）
    "finish_rank", "is_win", "is_place", "lr_label",
})


# ─── 特徴量列の選択 ────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """学習・評価に使う特徴量列を返す。

    ID 列・ラベル列・禁止列（FORBIDDEN_COLS）を除外し、
    残った数値・カテゴリ列を特徴量とする。
    train.py と evaluate.py で必ず同一の列集合になるよう、この関数のみを使う。
    """
    id_cols = set(cfg["features"]["id_cols"])
    exclude = id_cols | FORBIDDEN_COLS
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype not in ["object", "string"]
    ]


# ─── LambdaRank group 配列 ─────────────────────────────────────────────────────

def get_group_sizes(df: pd.DataFrame, race_id_col: str = "race_id") -> list[int]:
    """LightGBM LambdaRank 用 group 配列（レースごとの頭数リスト）を返す。

    前提: df は (race_date, race_id, horse_num) 順に並んでいなければならない。
    sort=False は行順を尊重するため、parquet の行順序が正しい場合のみ正確な
    グループ割り当てになる。create_features.py でこのソートを保証している。
    """
    return df.groupby(race_id_col, sort=False).size().tolist()
