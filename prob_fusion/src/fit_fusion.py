"""Conditional logit fusion: p_i ∝ exp(α·z_i + β·ln q_i)."""

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


def fusion_probs(z: np.ndarray, ln_q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Race-normalized fusion probabilities."""
    logits = alpha * z + beta * ln_q
    logits = logits - np.max(logits)
    exp_l = np.exp(logits)
    total = exp_l.sum()
    if total <= 0:
        n = len(z)
        return np.full(n, 1.0 / n)
    return exp_l / total


def race_nll(params: np.ndarray, z: np.ndarray, ln_q: np.ndarray, winner_idx: int) -> float:
    """Negative log-likelihood for one race."""
    alpha, beta = params
    p = fusion_probs(z, ln_q, alpha, beta)
    p_w = float(p[winner_idx])
    return -np.log(max(p_w, 1e-15))


def total_nll(params: np.ndarray, races: list[tuple[np.ndarray, np.ndarray, int]]) -> float:
    return sum(race_nll(params, z, ln_q, w) for z, ln_q, w in races)


def fit_fusion_mle(
    races: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    alpha_bounds: tuple[float, float] = (0.0, 5.0),
    beta_bounds: tuple[float, float] = (0.0, 3.0),
    market_only: bool = False,
) -> FusionParams:
    """MLE for (α, β) on list of (z, ln_q, winner_index) per race."""
    if market_only:

        def nll_beta(beta_arr: np.ndarray) -> float:
            return total_nll(np.array([0.0, beta_arr[0]]), races)

        res = optimize.minimize(
            nll_beta,
            x0=np.array([1.0]),
            bounds=[beta_bounds],
            method="L-BFGS-B",
        )
        return FusionParams(alpha=0.0, beta=float(res.x[0]), nll=float(res.fun), success=bool(res.success))

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
    )


def likelihood_ratio_test(
    races: list[tuple[np.ndarray, np.ndarray, int]],
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


def build_race_tuples(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
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
        races.append((z, ln_q, w_idx))
    return races


def mean_logloss(df: pd.DataFrame, alpha: float, beta: float) -> float:
    """Mean -log p(winner) on dataframe with fusion inputs."""
    losses: list[float] = []
    for _, grp in df.groupby("race_id"):
        winners = grp.loc[grp["finish_rank"].astype(int) == 1]
        if winners.empty:
            continue
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p = fusion_probs(z, ln_q, alpha, beta)
        w_pos = list(grp.index).index(winners.index[0])
        pw = float(p[w_pos])
        if pw > 1e-15:
            losses.append(-np.log(pw))
    return float(np.mean(losses)) if losses else float("nan")


def top1_hit_rate(df: pd.DataFrame, alpha: float, beta: float) -> float:
    """Top-1 accuracy using fusion probabilities."""
    hits = 0
    total = 0
    for _, grp in df.groupby("race_id"):
        winners = grp.loc[grp["finish_rank"].astype(int) == 1]
        if winners.empty:
            continue
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p = fusion_probs(z, ln_q, alpha, beta)
        pred_idx = int(np.argmax(p))
        actual_idx = list(grp.index).index(winners.index[0])
        total += 1
        if pred_idx == actual_idx:
            hits += 1
    return hits / total if total else 0.0


def calibration_bins(df: pd.DataFrame, alpha: float, beta: float, n_bins: int = 10) -> dict:
    """10-bin calibration: max |predicted - actual|."""
    preds: list[float] = []
    outcomes: list[int] = []
    for _, grp in df.groupby("race_id"):
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p = fusion_probs(z, ln_q, alpha, beta)
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
