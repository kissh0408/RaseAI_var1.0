"""Kelly基準によるベットサイジング。

CLAUDE.md制約:
- kelly_fraction = 0.08 (Quarter Kelly以下、MDD管理のため固定)
- max_bet_ratio = 0.05 (1レースあたり資金の5%上限)
- EV閾値は train_config.json の ev_threshold を参照（EVフィルタは ev_calculator 側）

実運用値は train_config.json の training セクションから渡される。
本モジュールのデフォルト引数は呼び出し側が値を省略した場合の安全側の値。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(
    model_prob: float,
    odds: float,
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> float:
    """Kelly基準でベット比率（資金に対する割合）を計算する。

    Full Kelly: f = (b*p - q) / b
    Fractional Kelly: f* = f × kelly_frac

    b: オッズ - 1 (純利益倍率)
    p: モデル推定勝率
    q: 1 - p (敗率)
    """
    b = odds - 1.0
    p = model_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b
    full_kelly = max(0.0, full_kelly)  # 負のKellyはベットしない

    fractional = full_kelly * kelly_frac
    return min(fractional, max_bet_ratio)


def kelly_bet_amount(
    model_prob: float,
    odds: float,
    bankroll: float,
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> float:
    """ベット金額（円）を返す。100円単位に切り捨て。"""
    ratio = kelly_fraction(model_prob, odds, kelly_frac, max_bet_ratio)
    raw_amount = bankroll * ratio
    # 馬券は100円単位
    return float(int(raw_amount / 100) * 100)


def apply_kelly_sizing(
    df: pd.DataFrame,
    bankroll: float,
    model_prob_col: str = "model_prob",
    odds_col: str = "odds",
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> pd.DataFrame:
    """DataFrameの各行にKellyベット金額を追加する。

    kelly_fraction() / kelly_bet_amount() のスカラー実装と同一の結果を返す
    ベクトル化版（行単位 apply はバックテスト全期間では低速なため）。
    """
    df = df.copy()
    b = df[odds_col].astype(float) - 1.0
    p = df[model_prob_col].astype(float)

    # b <= 0 はベット対象外。b.where(b > 0) でゼロ除算を避け、
    # NaN（オッズ・確率欠損）はスカラー版と同様に 0 扱いにする。
    full_kelly = ((b * p - (1.0 - p)) / b.where(b > 0)).fillna(0.0).clip(lower=0.0)

    df["kelly_ratio"] = (full_kelly * kelly_frac).clip(upper=max_bet_ratio)
    # 馬券は100円単位に切り捨て（非負のため floor == 切り捨て）
    df["kelly_bet_yen"] = np.floor(bankroll * df["kelly_ratio"] / 100.0) * 100.0
    return df
