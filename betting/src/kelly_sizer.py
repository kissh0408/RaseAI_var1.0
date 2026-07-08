"""Kelly criterion bet sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(
    model_prob: float,
    odds: float,
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> float:
    """Fractional Kelly bet ratio."""
    b = odds - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    full_kelly = max(0.0, full_kelly)
    fractional = full_kelly * kelly_frac
    return min(fractional, max_bet_ratio)


def kelly_bet_amount(
    model_prob: float,
    odds: float,
    bankroll: float,
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> float:
    """Bet amount in yen (100-yen units)."""
    ratio = kelly_fraction(model_prob, odds, kelly_frac, max_bet_ratio)
    raw_amount = bankroll * ratio
    return float(int(raw_amount / 100) * 100)


def apply_kelly_sizing(
    df: pd.DataFrame,
    bankroll: float,
    model_prob_col: str = "model_prob",
    odds_col: str = "odds",
    kelly_frac: float = 0.08,
    max_bet_ratio: float = 0.05,
) -> pd.DataFrame:
    """Add kelly_ratio and kelly_bet_yen columns."""
    df = df.copy()
    b = df[odds_col].astype(float) - 1.0
    p = df[model_prob_col].astype(float)
    full_kelly = ((b * p - (1.0 - p)) / b.where(b > 0)).fillna(0.0).clip(lower=0.0)
    df["kelly_ratio"] = (full_kelly * kelly_frac).clip(upper=max_bet_ratio)
    df["kelly_bet_yen"] = np.floor(bankroll * df["kelly_ratio"] / 100.0) * 100.0
    return df


def apply_mutually_exclusive_decay(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    kelly_col: str = "kelly_ratio",
) -> pd.DataFrame:
    """Reduce Kelly within race when multiple win bets (approximate mutual exclusivity)."""
    out = df.copy()
    if kelly_col not in out.columns:
        return out

    def _decay(grp: pd.DataFrame) -> pd.Series:
        k = grp[kelly_col].astype(float)
        n = (k > 0).sum()
        if n <= 1:
            return k
        return k / n

    out[kelly_col] = out.groupby(race_id_col, sort=False, group_keys=False).apply(_decay)
    return out
