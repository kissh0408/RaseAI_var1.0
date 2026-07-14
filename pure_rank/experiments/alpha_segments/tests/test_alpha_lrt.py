"""TDD tests for the alpha LRT machinery reused from prob_fusion.src.fit_fusion
(spec section 9, items 4/5/6/9). Synthetic data only; no real data required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from prob_fusion.src.fit_fusion import (  # noqa: E402
    build_race_tuples,
    fit_fusion_mle,
    fusion_probs,
    likelihood_ratio_test,
    mean_logloss,
)


def _synthetic_races_df(
    n_races: int,
    alpha_true: float,
    *,
    beta_true: float = 1.0,
    seed: int = 42,
    race_id_prefix: str = "R",
) -> pd.DataFrame:
    """Generate synthetic race dataframe: winner sampled from
    softmax(alpha_true * z + beta_true * ln_q). Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        n_horses = int(rng.integers(6, 15))
        z = rng.normal(0, 1, size=n_horses)
        ln_q = rng.normal(0, 1, size=n_horses)
        p = fusion_probs(z, ln_q, alpha_true, beta_true)
        winner = rng.choice(n_horses, p=p)
        race_id = f"{race_id_prefix}{i:05d}"
        for h in range(n_horses):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_num": h + 1,
                    "finish_rank": 1 if h == winner else 2,
                    "pure_score_z": float(z[h]),
                    "ln_market_q": float(ln_q[h]),
                    "race_date": "2023-06-01",
                }
            )
    return pd.DataFrame(rows)


# ─── item 4: positive control (alpha detection power) ──────────────────────


def test_alpha_positive_control_detected():
    df = _synthetic_races_df(n_races=500, alpha_true=0.8, seed=42)
    races = build_race_tuples(df)
    fitted = fit_fusion_mle(races)
    lrt = likelihood_ratio_test(races, fitted)

    assert fitted.alpha > 0.3, f"expected recovered alpha > 0.3, got {fitted.alpha}"
    assert lrt["p_value"] < 0.01, f"expected p<0.01, got {lrt['p_value']}"


# ─── item 5: negative control (alpha=0) ─────────────────────────────────────


def test_alpha_negative_control_not_detected():
    df = _synthetic_races_df(n_races=500, alpha_true=0.0, seed=123)
    races = build_race_tuples(df)
    fitted = fit_fusion_mle(races)
    lrt = likelihood_ratio_test(races, fitted)

    assert lrt["p_value"] > 0.05, f"expected p>0.05 under H0 truth, got {lrt['p_value']}"

    # ΔLL/race near zero (or negative allowed) on a held-out eval split from
    # the same (alpha=0) generative process.
    eval_df = _synthetic_races_df(n_races=500, alpha_true=0.0, seed=456)
    ll_h1 = mean_logloss(eval_df, fitted.alpha, fitted.beta)
    ll_h0 = mean_logloss(eval_df, 0.0, lrt["h0_beta"])
    delta_ll = ll_h0 - ll_h1
    assert abs(delta_ll) < 0.05, f"expected deltaLL near 0 under H0 truth, got {delta_ll}"


# ─── item 6: sign of ΔLL/race on eval split (positive control) ─────────────


def test_delta_ll_per_race_positive_sign_on_eval_split():
    fit_df = _synthetic_races_df(n_races=500, alpha_true=0.8, seed=42)
    eval_df = _synthetic_races_df(n_races=500, alpha_true=0.8, seed=999)

    fit_races = build_race_tuples(fit_df)
    fitted = fit_fusion_mle(fit_races)
    lrt = likelihood_ratio_test(fit_races, fitted)

    ll_h1 = mean_logloss(eval_df, fitted.alpha, fitted.beta)
    ll_h0 = mean_logloss(eval_df, 0.0, lrt["h0_beta"])
    delta_ll = ll_h0 - ll_h1
    assert delta_ll > 0, f"expected mean_logloss(H0) - mean_logloss(H1) > 0, got {delta_ll}"


# ─── item 9: reproducibility ─────────────────────────────────────────────────


def test_synthetic_generation_is_deterministic():
    df1 = _synthetic_races_df(n_races=50, alpha_true=0.8, seed=42)
    df2 = _synthetic_races_df(n_races=50, alpha_true=0.8, seed=42)
    pd.testing.assert_frame_equal(df1, df2)
