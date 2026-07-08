"""VALID 期間のみで EV 閾値・max_picks_per_race をグリッド探索（Rule 3 厳守）。"""
from __future__ import annotations

from typing import Callable, Literal

import pandas as pd

from evaluation import calculate_drawdown, calculate_roi_metrics

ObjectiveKind = Literal["roi", "sharpe"]


def _score_metrics(
    metrics: dict,
    drawdown: dict,
    *,
    objective: ObjectiveKind,
) -> float:
    if objective == "roi":
        return float(metrics.get("roi") or 0.0)
    return float(drawdown.get("sharpe_ratio") or 0.0)


def _is_better(
    candidate: dict,
    best: dict,
    *,
    objective: ObjectiveKind,
    sharpe_tie_delta: float,
) -> bool:
    c_score = candidate["objective_score"]
    b_score = best["objective_score"]
    if c_score > b_score + 1e-12:
        return True
    if abs(c_score - b_score) <= sharpe_tie_delta:
        # tie-break: larger n, then lower ev_threshold (more conservative)
        if candidate["valid_n_bets"] > best["valid_n_bets"]:
            return True
        if candidate["valid_n_bets"] == best["valid_n_bets"]:
            return candidate["ev_threshold"] < best["ev_threshold"]
    return False


def tune_bet_params_on_valid(
    valid_df: pd.DataFrame,
    predict_fn: Callable[..., pd.DataFrame],
    *,
    default_ev_threshold: float,
    default_max_picks: int,
    ev_threshold_grid: list[float],
    max_picks_grid: list[int],
    min_valid_bets: int = 100,
    objective: ObjectiveKind = "roi",
    sharpe_tie_delta: float = 0.01,
) -> dict:
    """VALID で ev_threshold / max_picks を選ぶ（テスト期間は使わない）。

    objective='sharpe': Phase 1b — Sharpe 最大、n < min_valid_bets は -1。
    objective='roi': 後方互換 — VALID ROI 最大。
    """
    best: dict | None = None
    for ev_t in ev_threshold_grid:
        for mp in max_picks_grid:
            pred = predict_fn(valid_df, ev_threshold=ev_t, max_picks=mp)
            if len(pred) == 0:
                continue
            metrics = calculate_roi_metrics(pred)
            dd = calculate_drawdown(pred)
            n_bets = int(metrics.get("n_bets") or 0)
            if n_bets < min_valid_bets:
                continue
            obj_score = _score_metrics(metrics, dd, objective=objective)
            candidate = {
                "ev_threshold": float(ev_t),
                "max_picks_per_race": int(mp),
                "valid_roi": float(metrics.get("roi") or 0.0),
                "valid_sharpe": float(dd.get("sharpe_ratio") or 0.0),
                "valid_n_bets": n_bets,
                "objective": objective,
                "objective_score": obj_score,
            }
            if best is None or _is_better(
                candidate, best, objective=objective, sharpe_tie_delta=sharpe_tie_delta
            ):
                best = candidate

    if best is None:
        return {
            "ev_threshold": default_ev_threshold,
            "max_picks_per_race": default_max_picks,
            "valid_roi": None,
            "valid_sharpe": None,
            "valid_n_bets": 0,
            "objective": objective,
            "objective_score": -1.0,
            "tuning_fallback": True,
        }
    best["tuning_fallback"] = False
    return best
