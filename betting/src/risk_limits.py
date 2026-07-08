"""Monthly MDD and consecutive loss stop rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RiskState:
    bankroll: float
    peak_bankroll: float
    consecutive_losses: int = 0
    stopped: bool = False
    stop_reason: str = ""


@dataclass
class RiskLimits:
    monthly_mdd_limit: float = 0.15
    consecutive_loss_stop: int = 10

    def update_after_bet(
        self,
        state: RiskState,
        *,
        stake: float,
        payout: float,
        month_peak: float | None = None,
    ) -> RiskState:
        """Update bankroll and check stop rules."""
        state = RiskState(
            bankroll=state.bankroll - stake + payout,
            peak_bankroll=state.peak_bankroll,
            consecutive_losses=state.consecutive_losses,
            stopped=state.stopped,
            stop_reason=state.stop_reason,
        )
        if payout <= stake:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

        if state.consecutive_losses >= self.consecutive_loss_stop:
            state.stopped = True
            state.stop_reason = "consecutive_losses"

        peak = month_peak if month_peak is not None else state.peak_bankroll
        if state.bankroll > peak:
            peak = state.bankroll
        state.peak_bankroll = peak
        if peak > 0:
            dd = (peak - state.bankroll) / peak
            if dd >= self.monthly_mdd_limit:
                state.stopped = True
                state.stop_reason = "monthly_mdd"
        return state


def compute_mdd(equity_curve: np.ndarray) -> float:
    """Maximum drawdown from equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def monthly_returns(bets: pd.DataFrame, *, date_col: str = "race_date", pnl_col: str = "pnl") -> pd.Series:
    """Aggregate PnL by month."""
    dates = pd.to_datetime(bets[date_col])
    return bets.groupby(dates.dt.to_period("M"))[pnl_col].sum()
