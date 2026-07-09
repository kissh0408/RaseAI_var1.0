"""Tests for prob_fusion fit_fusion."""

from __future__ import annotations

import pandas as pd
import numpy as np

from prob_fusion.src.fit_fusion import (
    build_race_tuples,
    fit_fusion_mle,
    fusion_probs,
    gamma_likelihood_ratio_test,
)


def test_fusion_probs_sum_to_one():
    z = np.array([0.0, 1.0, -0.5])
    ln_q = np.log(np.array([0.5, 0.3, 0.2]))
    p = fusion_probs(z, ln_q, alpha=1.0, beta=1.0)
    assert abs(p.sum() - 1.0) < 1e-9


def test_market_only_fusion():
    races = [
        (np.array([0.0, 1.0]), np.log(np.array([0.6, 0.4])), 0),
        (np.array([-0.5, 0.5]), np.log(np.array([0.55, 0.45])), 1),
    ]
    fitted = fit_fusion_mle(races, market_only=True)
    assert fitted.alpha == 0.0
    assert fitted.beta > 0


def test_fusion_probs_accepts_gamma_candidate_term():
    z = np.array([0.0, 0.0, 0.0])
    ln_q = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
    x = np.array([2.0, -1.0, -1.0])

    p = fusion_probs(z, ln_q, alpha=0.0, beta=1.0, x=x, gamma=2.0)

    assert abs(p.sum() - 1.0) < 1e-9
    assert int(np.argmax(p)) == 0


def test_gamma_zero_preserves_two_parameter_probabilities():
    z = np.array([0.0, 1.0, -0.5])
    ln_q = np.log(np.array([0.5, 0.3, 0.2]))
    x = np.array([10.0, -10.0, 3.0])

    p_old = fusion_probs(z, ln_q, alpha=1.2, beta=0.8)
    p_new = fusion_probs(z, ln_q, alpha=1.2, beta=0.8, x=x, gamma=0.0)

    assert np.allclose(p_old, p_new)


def test_build_race_tuples_can_include_candidate_column():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "horse_num": [1, 2, 1, 2],
            "finish_rank": [1, 2, 2, 1],
            "pure_score_z": [0.2, -0.2, 0.1, -0.1],
            "ln_market_q": np.log([0.55, 0.45, 0.52, 0.48]),
            "cand_score_z": [1.0, -1.0, -1.0, 1.0],
        }
    )

    races = build_race_tuples(df, x_col="cand_score_z")

    assert len(races) == 2
    z, ln_q, x, winner_idx = races[0]
    assert winner_idx == 0
    assert np.allclose(x, np.array([1.0, -1.0]))
    assert len(z) == len(ln_q) == len(x)


def test_gamma_lrt_detects_synthetic_candidate_signal():
    races = []
    ln_q = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
    for _ in range(80):
        races.append(
            (
                np.zeros(3),
                ln_q,
                np.array([2.0, -1.0, -1.0]),
                0,
            )
        )

    fitted = fit_fusion_mle(races, gamma_bounds=(0.0, 5.0))
    lrt = gamma_likelihood_ratio_test(races, fitted)

    assert fitted.gamma > 0.0
    assert lrt["p_value"] < 0.01


def test_gamma_lrt_does_not_flag_constant_candidate():
    races = []
    ln_q = np.log(np.array([0.5, 0.3, 0.2]))
    for i in range(40):
        races.append(
            (
                np.array([0.0, 0.0, 0.0]),
                ln_q,
                np.array([1.0, 1.0, 1.0]),
                i % 3,
            )
        )

    fitted = fit_fusion_mle(races, gamma_bounds=(0.0, 5.0))
    lrt = gamma_likelihood_ratio_test(races, fitted)

    assert abs(fitted.gamma) < 1e-6
    assert lrt["p_value"] >= 0.01
