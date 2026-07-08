"""ev_filters 純粋関数の characterization テスト。"""
from __future__ import annotations

import numpy as np

from betting.src.ev_filters import EvFilterConfig, effective_min_edge, ev_filter_config_from_mapping


class TestEffectiveMinEdge:
    def test_static_min_edge_when_dynamic_disabled(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=False)
        assert effective_min_edge(5.0, cfg) == 0.02

    def test_step_band_at_odds_5(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=True)
        assert effective_min_edge(5.0, cfg) == 0.12

    def test_step_band_at_odds_15(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=True)
        assert effective_min_edge(15.0, cfg) == 999.0

    def test_log_linear_mode(self):
        cfg = EvFilterConfig(
            min_edge=0.02,
            dynamic_edge_enabled=True,
            dynamic_edge_mode="log_linear",
            dynamic_edge_alpha=0.02,
            dynamic_edge_beta=0.08,
        )
        val = effective_min_edge(5.0, cfg)
        expected = max(0.02, 0.02 + 0.08 * np.log(5.0))
        assert np.isclose(val, expected)

    def test_step_band_at_odds_3(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=True)
        assert effective_min_edge(3.0, cfg) == 0.08

    def test_step_band_at_odds_6(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=True)
        assert effective_min_edge(6.0, cfg) == 0.12

    def test_step_band_vectorized(self):
        cfg = EvFilterConfig(min_edge=0.02, dynamic_edge_enabled=True)
        odds = np.array([3.0, 5.0, 15.0])
        expected = np.array([0.08, 0.12, 999.0])
        assert np.allclose(effective_min_edge(odds, cfg), expected)


class TestEvFilterConfigFromMapping:
    def test_builds_from_dict(self):
        cfg = ev_filter_config_from_mapping(
            {"min_edge": 0.03, "dynamic_edge_enabled": True}
        )
        assert cfg.min_edge == 0.03
        assert cfg.dynamic_edge_enabled is True
