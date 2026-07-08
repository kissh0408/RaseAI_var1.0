"""Unit tests for evaluate_going_experiment_gate segment logic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "scripts"))

from evaluate_going_experiment_gate import evaluate_segment_gate


def test_segment_gate_pending_when_n_low() -> None:
    out = evaluate_segment_gate({"heavy": {"n_bets": 50, "roi": 1.2}})
    assert out["pending"] is True
    assert out["passed"] is None


def test_segment_gate_pass_when_n_high_and_roi_ok() -> None:
    out = evaluate_segment_gate(
        {
            "heavy": {"n_bets": 250, "roi": 1.05},
            "soft": {"n_bets": 220, "roi": 1.02},
        }
    )
    assert out["passed"] is True


def test_segment_gate_reject_when_heavy_roi_low() -> None:
    out = evaluate_segment_gate({"heavy": {"n_bets": 300, "roi": 0.95}, "soft": {"n_bets": 250, "roi": 1.1}})
    assert out["passed"] is False
