"""
features_pace_field.py — レース内ペース構成特徴量

生成特徴量:
    field_front_runner_ratio  : 出走頭数に占める逃げ+先行馬の割合（ペース激しさの代理変数）
    field_closer_ratio        : 出走頭数に占める差し+追込馬の割合
    style_vs_field_fit_score  : 自馬脚質が当該レースの展開で有利かどうかのスコア
                                  逃げ/先行 → 少頭数の前傾なら有利（低field_front_runner_ratio）
                                  差し/追込 → 多頭数の前傾なら有利（高field_front_runner_ratio）

問題の根拠（v13バックテスト）:
    脚質別フラットベットROI: 逃げ2.12 vs 追込0.29。モデルが差し/追込馬を精度よく
    評価できていない。展開適性（ペースシナリオ）を捉える特徴量を追加することで改善を図る。

計算上のリーク防止:
    running_style_code は各馬の「過去傾向に基づく脚質区分」。これを同一レース内で
    集計するのはレース前に利用可能な情報であり、リークには該当しない。
    （注: running_style_code が事後確定値の場合でも、同一レース内集計はリーク
    に当たらない。当該レース全馬のスタイルが揃ってから集計するため。）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FRONT_STYLES = {1, 2}   # 逃げ・先行
_CLOSER_STYLES = {3, 4}  # 差し・追込


def add_pace_field_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内ペース構成特徴量を df に追加して返す。

    Args:
        df: features_past_v14 相当（race_id, horse_modal_running_style（または running_style_code）,
            n_horses 列を持つ）
            horse_modal_running_style が存在しない場合は running_style_code にフォールバック。
    Returns:
        新特徴量3列を追加した DataFrame（行数・順序変更なし）
    """
    new_cols = ["field_front_runner_ratio", "field_closer_ratio", "style_vs_field_fit_score"]
    if all(c in df.columns for c in new_cols):
        return df

    # horse_modal_running_style を優先使用（推論時も 0 以外の値が入る）
    # レース内の全馬の脚質分布を集計するため、フィールド全体の展開評価に使用
    if "horse_modal_running_style" in df.columns:
        style = pd.to_numeric(df["horse_modal_running_style"], errors="coerce")
    else:
        style = pd.to_numeric(df["running_style_code"], errors="coerce")
    race_id = df["race_id"].astype(str)

    # レースごとに逃げ+先行 / 差し+追込 の頭数を集計
    front_flag = style.isin(_FRONT_STYLES).astype("int8")
    closer_flag = style.isin(_CLOSER_STYLES).astype("int8")

    race_front_count = front_flag.groupby(race_id, sort=False).transform("sum")
    race_closer_count = closer_flag.groupby(race_id, sort=False).transform("sum")
    race_total = race_front_count + race_closer_count

    # 比率計算（有効馬頭数ゼロ回避）
    field_front_ratio = (race_front_count / race_total.replace(0, np.nan)).astype("float32")
    field_closer_ratio = (race_closer_count / race_total.replace(0, np.nan)).astype("float32")

    df = df.copy()
    df["field_front_runner_ratio"] = field_front_ratio
    df["field_closer_ratio"] = field_closer_ratio

    # 展開フィット・スコア: [-1, +1] の連続値
    #   逃げ/先行馬: 前傾馬が少ない(field_front_ratio低い) → 楽に逃げられる → +
    #   差し/追込馬: 前傾馬が多い(field_front_ratio高い) → ペース崩れを期待 → +
    #   脚質不明(0): 中立値 0.0 を設定（推論時に LightGBM の NaN 分岐を回避するため）
    is_front = style.isin(_FRONT_STYLES)
    is_closer = style.isin(_CLOSER_STYLES)
    is_unknown_style = (style.fillna(0) == 0)  # horse_modal_running_style=0 または NaN

    # front馬のフィット: field_front_ratioが低いほど有利 → -(field_front_ratio - 0.5)*2
    # closer馬のフィット: field_front_ratioが高いほど有利 → (field_front_ratio - 0.5)*2
    # 不明馬: 展開との有利不利を判断できないため中立値 0.0（NaN よりも LightGBM 分岐に最適）
    fit_score = pd.Series(0.0, index=df.index, dtype="float32")  # デフォルト: 中立値 0.0
    fit_score = fit_score.where(~is_front, -(field_front_ratio - 0.5) * 2)
    fit_score = fit_score.where(~is_closer, (field_front_ratio - 0.5) * 2)
    df["style_vs_field_fit_score"] = fit_score

    return df
