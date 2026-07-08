"""Tests for prob_fusion fit_fusion."""

from __future__ import annotations

import numpy as np

from prob_fusion.src.fit_fusion import fit_fusion_mle, fusion_probs


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
