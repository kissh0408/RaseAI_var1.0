"""Market probability q from win odds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.market_baseline import power_market_prob, proportional_market_prob


def attach_market_q(
    df: pd.DataFrame,
    *,
    odds_col: str = "odds",
    race_id_col: str = "race_id",
    method: str = "proportional",
    power: float = 0.81,
    q_col: str = "market_q",
    ln_q_col: str = "ln_market_q",
) -> pd.DataFrame:
    """Add market probability q and ln(q) per horse."""
    out = df.copy()

    def _race_q(grp: pd.DataFrame) -> pd.Series:
        odds = grp[odds_col].astype(float).values
        if method == "power":
            q = power_market_prob(odds, power=power)
        else:
            q = proportional_market_prob(odds)
        return pd.Series(q, index=grp.index)

    out[q_col] = out.groupby(race_id_col, sort=False, group_keys=False).apply(_race_q)
    out[ln_q_col] = np.log(out[q_col].clip(lower=1e-12))
    return out
