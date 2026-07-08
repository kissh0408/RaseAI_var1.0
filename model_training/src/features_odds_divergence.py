"""
features_odds_divergence.py — 市場オッズ乖離特徴量（v25 / Exp-3）

domain-planner 仕様書 docs/specs/domain_planner_spec_v25_odds_divergence.md に基づく。

odds_rank_divergence:
  JV-Link マイニング予測順位と市場単勝人気順位の差
  計算: mining_predicted_rank - popularity
  正値 = モデルが市場より高評価（過小評価馬の可能性）
  リーク防止: mining_predicted_rank はレース前公開値。popularity は O1 暫定値。shift 不要。

field_odds_entropy:
  1 レース内の全馬単勝オッズ分布の Shannon エントロピー
  低値 = オッズが均等（混戦）、高値 = 本命馬が明確
  リーク防止: odds は O1 レコードから取得する暫定値。着順情報は一切使用しない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NEW_COLS = ["odds_rank_divergence", "field_odds_entropy"]


def v25_odds_column_names() -> list[str]:
    return list(_NEW_COLS)


def _calc_field_entropy(odds_series: pd.Series) -> float:
    """1レース内の単勝オッズ分布から Shannon エントロピーを計算する。

    オッズの逆数を確率として正規化し、エントロピーを返す。
    低値 = 本命が明確なレース、高値 = 混戦レース。
    """
    arr = pd.to_numeric(odds_series, errors="coerce").dropna().values
    if len(arr) == 0:
        return np.nan
    # オッズは 1.0 より大きい値のみ有効（最低でも 1.01 にクリップ）
    probs = 1.0 / np.clip(arr, 1.01, None)
    total = probs.sum()
    if total <= 0:
        return np.nan
    probs = probs / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def add_odds_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    odds_rank_divergence と field_odds_entropy を追加する。

    前提: df に mining_predicted_rank, popularity, odds, race_id, n_horses 列が存在すること。
    リーク防止: すべての入力列はレース前確定/暫定値のため shift 不要。

    Parameters
    ----------
    df : pd.DataFrame
        features_past_v23.parquet 相当の入力データ。

    Returns
    -------
    pd.DataFrame
        新列 odds_rank_divergence, field_odds_entropy を追加したデータフレーム。
    """
    # 冪等性ガード: 既存列があれば即時リターン
    if all(c in df.columns for c in _NEW_COLS):
        return df

    df = df.copy()

    # ── odds_rank_divergence ──────────────────────────────────
    # JV-Link マイニング予測順位 と 市場単勝人気順位 の差
    # 正値 = AI がより上位に評価（市場が過小評価している可能性）
    if "mining_predicted_rank" in df.columns and "popularity" in df.columns:
        mining = pd.to_numeric(df["mining_predicted_rank"], errors="coerce")
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        raw = mining - pop
        # NaN は 0（差なし）で補完。int16 で省メモリ化
        df["odds_rank_divergence"] = raw.fillna(0).astype("int16")
    else:
        df["odds_rank_divergence"] = np.int16(0)

    # ── field_odds_entropy ────────────────────────────────────
    # レース内の全馬オッズ分布の Shannon エントロピー
    # 着順情報（finish_rank 等）は一切使用しない（リーク防止）
    if "race_id" in df.columns and "odds" in df.columns:
        entropy_by_race = (
            df.groupby("race_id", sort=False)["odds"]
            .apply(_calc_field_entropy)
            .rename("field_odds_entropy")
        )
        df = df.merge(entropy_by_race, on="race_id", how="left")
        # 欠損補完: n_horses が利用可能な場合は log(n_horses) で補完
        # （均等オッズ分布のエントロピー = log(n) を仮定）
        if "n_horses" in df.columns:
            nh = pd.to_numeric(df["n_horses"], errors="coerce").clip(lower=2)
            fallback = np.log(nh).astype("float32")
            df["field_odds_entropy"] = (
                df["field_odds_entropy"].fillna(fallback).astype("float32")
            )
        else:
            df["field_odds_entropy"] = df["field_odds_entropy"].astype("float32")
    else:
        df["field_odds_entropy"] = np.nan

    return df
