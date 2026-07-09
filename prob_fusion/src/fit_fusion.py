"""Conditional logit fusion: p_i ∝ exp(α·z_i + β·ln q_i + γ·x_i)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats


@dataclass
class FusionParams:
    alpha: float
    beta: float
    nll: float
    success: bool
    gamma: float = 0.0


RaceTuple = tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, np.ndarray, int]


def _unpack_race(
    race: RaceTuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    if len(race) == 3:
        z, ln_q, winner_idx = race
        return z, ln_q, None, winner_idx
    z, ln_q, x, winner_idx = race
    return z, ln_q, x, winner_idx


def fusion_probs(
    z: np.ndarray,
    ln_q: np.ndarray,
    alpha: float,
    beta: float,
    *,
    x: np.ndarray | None = None,
    gamma: float = 0.0,
) -> np.ndarray:
    """Race-normalized fusion probabilities."""
    logits = alpha * z + beta * ln_q
    if x is not None and gamma != 0.0:
        logits = logits + gamma * x
    logits = logits - np.max(logits)
    exp_l = np.exp(logits)
    total = exp_l.sum()
    if total <= 0:
        n = len(z)
        return np.full(n, 1.0 / n)
    return exp_l / total


def race_nll(
    params: np.ndarray,
    z: np.ndarray,
    ln_q: np.ndarray,
    winner_idx: int,
    x: np.ndarray | None = None,
) -> float:
    """Negative log-likelihood for one race."""
    alpha, beta = params[0], params[1]
    gamma = params[2] if len(params) > 2 else 0.0
    p = fusion_probs(z, ln_q, alpha, beta, x=x, gamma=gamma)
    p_w = float(p[winner_idx])
    return -np.log(max(p_w, 1e-15))


def total_nll(params: np.ndarray, races: list[RaceTuple]) -> float:
    total = 0.0
    for race in races:
        z, ln_q, x, winner_idx = _unpack_race(race)
        total += race_nll(params, z, ln_q, winner_idx, x=x)
    return total


def fit_fusion_mle(
    races: list[RaceTuple],
    *,
    alpha_bounds: tuple[float, float] = (0.0, 5.0),
    beta_bounds: tuple[float, float] = (0.0, 3.0),
    gamma_bounds: tuple[float, float] = (0.0, 5.0),
    market_only: bool = False,
    gamma_fixed_zero: bool = False,
) -> FusionParams:
    """MLE for (α, β, γ) on race tuples; γ is optional/backward compatible."""
    if market_only:

        def nll_beta(beta_arr: np.ndarray) -> float:
            return total_nll(np.array([0.0, beta_arr[0]]), races)

        res = optimize.minimize(
            nll_beta,
            x0=np.array([1.0]),
            bounds=[beta_bounds],
            method="L-BFGS-B",
        )
        return FusionParams(alpha=0.0, beta=float(res.x[0]), nll=float(res.fun), success=bool(res.success), gamma=0.0)

    has_candidate = any(len(race) == 4 for race in races)
    if has_candidate and not gamma_fixed_zero:
        res = optimize.minimize(
            total_nll,
            x0=np.array([1.0, 1.0, 0.0]),
            args=(races,),
            bounds=[alpha_bounds, beta_bounds, gamma_bounds],
            method="L-BFGS-B",
        )
        return FusionParams(
            alpha=float(res.x[0]),
            beta=float(res.x[1]),
            gamma=float(res.x[2]),
            nll=float(res.fun),
            success=bool(res.success),
        )

    res = optimize.minimize(
        total_nll,
        x0=np.array([1.0, 1.0]),
        args=(races,),
        bounds=[alpha_bounds, beta_bounds],
        method="L-BFGS-B",
    )
    return FusionParams(
        alpha=float(res.x[0]),
        beta=float(res.x[1]),
        nll=float(res.fun),
        success=bool(res.success),
        gamma=0.0,
    )


def likelihood_ratio_test(
    races: list[RaceTuple],
    fitted: FusionParams,
    *,
    alpha_bounds: tuple[float, float] = (0.0, 5.0),
    beta_bounds: tuple[float, float] = (0.0, 3.0),
) -> dict:
    """LRT: H0 α=0 vs H1 α free."""
    h0 = fit_fusion_mle(races, alpha_bounds=alpha_bounds, beta_bounds=beta_bounds, market_only=True)
    lr = 2.0 * (h0.nll - fitted.nll)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1) if lr > 0 else 1.0
    return {
        "h0_nll": h0.nll,
        "h1_nll": fitted.nll,
        "lr_statistic": lr,
        "p_value": float(p_value),
        "h0_beta": h0.beta,
    }


def gamma_likelihood_ratio_test(
    races: list[RaceTuple],
    fitted: FusionParams,
    *,
    alpha_bounds: tuple[float, float] = (0.0, 5.0),
    beta_bounds: tuple[float, float] = (0.0, 3.0),
    gamma_bounds: tuple[float, float] = (0.0, 5.0),
) -> dict:
    """LRT: H0 γ=0 with α,β free vs H1 γ free."""
    h0 = fit_fusion_mle(
        races,
        alpha_bounds=alpha_bounds,
        beta_bounds=beta_bounds,
        gamma_bounds=gamma_bounds,
        gamma_fixed_zero=True,
    )
    lr = 2.0 * (h0.nll - fitted.nll)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1) if lr > 0 else 1.0
    return {
        "h0_nll": h0.nll,
        "h1_nll": fitted.nll,
        "lr_statistic": lr,
        "p_value": float(p_value),
        "h0_alpha": h0.alpha,
        "h0_beta": h0.beta,
    }


def build_race_tuples(df: pd.DataFrame, x_col: str | None = None) -> list[RaceTuple]:
    """Build (z, ln_q, winner_idx) list from scored dataframe."""
    races = []
    for _, grp in df.groupby("race_id"):
        grp = grp.sort_values("horse_num")
        winners = grp.loc[grp["finish_rank"].astype(int) == 1]
        if winners.empty:
            continue
        w_horse = int(winners["horse_num"].iloc[0])
        horse_nums = grp["horse_num"].astype(int).tolist()
        if w_horse not in horse_nums:
            continue
        w_idx = horse_nums.index(w_horse)
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        if len(z) < 2:
            continue
        if x_col is not None:
            x = grp[x_col].astype(float).values
            races.append((z, ln_q, x, w_idx))
        else:
            races.append((z, ln_q, w_idx))
    return races


def mean_logloss(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    *,
    x_col: str | None = None,
    gamma: float = 0.0,
) -> float:
    """Mean -log p(winner) on dataframe with fusion inputs."""
    losses: list[float] = []
    for _, grp in df.groupby("race_id"):
        winners = grp.loc[grp["finish_rank"].astype(int) == 1]
        if winners.empty:
            continue
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        x = grp[x_col].astype(float).values if x_col is not None else None
        p = fusion_probs(z, ln_q, alpha, beta, x=x, gamma=gamma)
        w_pos = list(grp.index).index(winners.index[0])
        pw = float(p[w_pos])
        if pw > 1e-15:
            losses.append(-np.log(pw))
    return float(np.mean(losses)) if losses else float("nan")


def top1_hit_rate(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    *,
    x_col: str | None = None,
    gamma: float = 0.0,
) -> float:
    """Top-1 accuracy using fusion probabilities."""
    hits = 0
    total = 0
    for _, grp in df.groupby("race_id"):
        winners = grp.loc[grp["finish_rank"].astype(int) == 1]
        if winners.empty:
            continue
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        x = grp[x_col].astype(float).values if x_col is not None else None
        p = fusion_probs(z, ln_q, alpha, beta, x=x, gamma=gamma)
        pred_idx = int(np.argmax(p))
        actual_idx = list(grp.index).index(winners.index[0])
        total += 1
        if pred_idx == actual_idx:
            hits += 1
    return hits / total if total else 0.0


def calibration_bins(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    n_bins: int = 10,
    *,
    x_col: str | None = None,
    gamma: float = 0.0,
) -> dict:
    """10-bin calibration: max |predicted - actual|."""
    preds: list[float] = []
    outcomes: list[int] = []
    for _, grp in df.groupby("race_id"):
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        x = grp[x_col].astype(float).values if x_col is not None else None
        p = fusion_probs(z, ln_q, alpha, beta, x=x, gamma=gamma)
        finish = grp["finish_rank"].astype(int).values
        for i, prob in enumerate(p):
            preds.append(float(prob))
            outcomes.append(1 if finish[i] == 1 else 0)
    if not preds:
        return {"max_error_pp": None, "bins": []}
    preds_arr = np.array(preds)
    outcomes_arr = np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []
    bin_details = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (preds_arr >= lo) & (preds_arr < hi if i < n_bins - 1 else preds_arr <= hi)
        if mask.sum() < 5:
            continue
        pred_mean = preds_arr[mask].mean()
        actual_rate = outcomes_arr[mask].mean()
        err_pp = abs(pred_mean - actual_rate) * 100
        errors.append(err_pp)
        bin_details.append(
            {"bin": i, "pred": pred_mean, "actual": actual_rate, "error_pp": err_pp, "n": int(mask.sum())}
        )
    return {"max_error_pp": max(errors) if errors else None, "bins": bin_details}
