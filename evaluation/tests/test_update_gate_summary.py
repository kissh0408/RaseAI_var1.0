"""Tests for gate_summary aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.update_gate_summary import build_gate_summary


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_gate_summary_includes_alpha_and_place_sections(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    _write(reports / "market_baseline.json", {"favorite_baseline": {"favorite_top1_rate": 0.329, "favorite_roi": 77.9}})
    _write(reports / "fusion_oos_fold2.json", {"formal": {"gates": {"phase2_pass": False}}})
    _write(reports / "betting_backtest_oos.json", {"status": "skipped", "reason": "no positive EV"})
    _write(
        reports / "alpha_gate_b1.json",
        {
            "candidate": "b1",
            "pass": True,
            "gamma": 0.2,
            "gamma_lrt_p": 0.001,
            "delta_ll_per_race": 0.01,
            "eval_top1": 0.31,
        },
    )
    _write(
        reports / "place_baseline_oos.json",
        {
            "status": "measured",
            "model_top1": {"roi_pct": 102.0, "n_races": 500, "roi_minus_favorite_pp": 4.0},
            "favorite": {"roi_pct": 98.0, "n_races": 500},
            "verdict": "PASS",
            "gates": {"phase3_place_pass": True},
            "known_limitations": ["confirmed HR payouts are settlement data"],
        },
    )

    summary = build_gate_summary(reports, root=tmp_path)

    assert summary["alpha_gate"]["n_reports"] == 1
    assert summary["alpha_gate"]["passing_candidates"] == ["b1"]
    assert summary["track_c_place"]["status"] == "measured"
    assert summary["track_c_place"]["verdict"] == "PASS"
    assert summary["track_c_place"]["gates"] == {"phase3_place_pass": True}
    assert summary["track_c_place"]["model_top1_roi_pct"] == 102.0
    assert summary["track_c_place"]["roi_minus_favorite_pp"] == 4.0
    assert summary["track_c_place"]["known_limitations"] == ["confirmed HR payouts are settlement data"]
