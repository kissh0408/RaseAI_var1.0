"""期待値（EV）計算モジュール。

model-strategy-generatorフェーズ。
EV = キャリブレーション済み予測勝率 × デシマルオッズ
EV > 1.0 → 正期待値（投資対象）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_ev(model_prob: float, odds: float) -> float:
    """単勝の期待値率を計算する。

    EV = P(win) × odds
    EV > 1.05 → 5%以上の正期待値 → 投資対象
    """
    return model_prob * odds


def calculate_ev_series(model_probs: pd.Series, odds: pd.Series) -> pd.Series:
    """Series単位でEVを計算する。"""
    return model_probs * odds


def apply_ev_filters(
    df: pd.DataFrame,
    ev_col: str = "ev_rate",
    odds_col: str = "odds",
    model_prob_col: str = "model_prob",
    ev_threshold: float = 1.05,
    min_odds: float = 2.0,
    max_odds: float = 50.0,
    min_model_prob: float = 0.05,
) -> pd.Series:
    """投資条件を全て満たすブールマスクを返す。

    CLAUDE.mdの条件:
    - EV > ev_threshold (デフォルト1.05)
    - オッズ >= min_odds (過剰人気馬を除外)
    - オッズ <= max_odds (極端な大穴を除外)
    - model_prob >= min_model_prob (確率極小馬を除外)
    """
    mask = (
        (df[ev_col] >= ev_threshold)
        & (df[odds_col] >= min_odds)
        & (df[odds_col] <= max_odds)
        & (df[model_prob_col] >= min_model_prob)
    )
    return mask


def enrich_predictions(
    df: pd.DataFrame,
    model_prob_col: str = "model_prob",
    odds_col: str = "odds",
) -> pd.DataFrame:
    """予測DataFrameにEV・implied_prob・model_edgeを追加する。"""
    df = df.copy()
    df["ev_rate"] = calculate_ev_series(df[model_prob_col], df[odds_col])
    df["implied_prob"] = 1.0 / df[odds_col].clip(lower=1.01)  # オッズ1倍未満はクリップ
    df["model_edge"] = df[model_prob_col] - df["implied_prob"]
    return df
