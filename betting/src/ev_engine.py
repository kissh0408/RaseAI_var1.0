"""EV calculation and filtering for Benter fused probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def calculate_ev(model_prob: float, odds: float) -> float:
    return model_prob * odds


def calculate_ev_series(model_probs: pd.Series, odds: pd.Series) -> pd.Series:
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
    return (
        (df[ev_col] >= ev_threshold)
        & (df[odds_col] >= min_odds)
        & (df[odds_col] <= max_odds)
        & (df[model_prob_col] >= min_model_prob)
    )


def enrich_predictions(
    df: pd.DataFrame,
    model_prob_col: str = "model_prob",
    odds_col: str = "odds",
    ev_haircut: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    out["ev_rate"] = calculate_ev_series(out[model_prob_col], out[odds_col])
    out["ev_adjusted"] = out["ev_rate"] * ev_haircut
    out["implied_prob"] = 1.0 / out[odds_col].clip(lower=1.01)
    out["model_edge"] = out[model_prob_col] - out["implied_prob"]
    return out


@dataclass
class EvThresholdResult:
    threshold: float
    valid_n_rows: int
    valid_n_races: int
    grid_evaluated: int
    fallback_used: bool
    warnings: list[str] = field(default_factory=list)


def select_ev_threshold_on_valid(
    valid_df: pd.DataFrame,
    grid: list[float],
    *,
    prob_col: str = "p_win",
    odds_col: str = "odds",
    finish_col: str = "finish_rank",
    bet_type: str = "win",
    ev_haircut: float = 0.95,
    min_bets: int = 50,
    place_odds_col: str | None = None,
) -> EvThresholdResult:
    """Pick EV threshold maximizing ROI on VALID (Rule 3). Never silent without warning."""
    warnings: list[str] = []
    n_rows = len(valid_df)
    n_races = int(valid_df["race_id"].nunique()) if n_rows and "race_id" in valid_df.columns else 0

    if n_rows == 0 or n_races == 0:
        warnings.append(f"VALID empty (rows={n_rows}, races={n_races}); cannot tune threshold")
        return EvThresholdResult(
            threshold=grid[0],
            valid_n_rows=n_rows,
            valid_n_races=n_races,
            grid_evaluated=0,
            fallback_used=True,
            warnings=warnings,
        )

    if bet_type == "place" and (place_odds_col is None or place_odds_col not in valid_df.columns):
        warnings.append("place odds unavailable; place threshold selection skipped")
        return EvThresholdResult(
            threshold=grid[0],
            valid_n_rows=n_rows,
            valid_n_races=n_races,
            grid_evaluated=0,
            fallback_used=True,
            warnings=warnings,
        )

    best_threshold = grid[0]
    best_roi = -1.0
    grid_hits = 0
    work = valid_df.copy()
    work["model_prob"] = work[prob_col]
    payout_odds_col = place_odds_col if bet_type == "place" and place_odds_col else odds_col
    work = enrich_predictions(work, model_prob_col="model_prob", odds_col=odds_col, ev_haircut=ev_haircut)

    for thr in grid:
        mask = apply_ev_filters(work, ev_col="ev_adjusted", ev_threshold=thr)
        picks = work.loc[mask]
        if len(picks) < min_bets:
            continue
        grid_hits += 1
        stake = 100.0 * len(picks)
        if bet_type == "win":
            payout = picks.loc[picks[finish_col].astype(int) == 1, payout_odds_col].astype(float).sum() * 100.0
        else:
            payout = picks.loc[picks[finish_col].astype(int) <= 3, payout_odds_col].astype(float).sum() * 100.0
        roi = payout / stake * 100.0 if stake > 0 else 0.0
        if roi > best_roi:
            best_roi = roi
            best_threshold = thr

    fallback = grid_hits == 0
    if fallback:
        warnings.append(
            f"No grid threshold met min_bets={min_bets} on VALID (n_rows={n_rows}); "
            f"fallback to grid[0]={grid[0]}"
        )

    return EvThresholdResult(
        threshold=best_threshold,
        valid_n_rows=n_rows,
        valid_n_races=n_races,
        grid_evaluated=grid_hits,
        fallback_used=fallback,
        warnings=warnings,
    )
