"""Tests for alpha-gate candidate evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.alpha_gate import (
    attach_candidate_z,
    evaluate_leak_warnings,
    run_alpha_gate_on_dataframe,
    split_alpha_gate_cv,
)


def _candidate_frame(signal: bool = True) -> pd.DataFrame:
    rows = []
    for year in (2023, 2024, 2025):
        for race in range(12):
            race_id = f"{year}0101010101{race:02d}"
            winner = race % 3 + 1
            losers = [h for h in (1, 2, 3) if h != winner]
            for horse_num in (1, 2, 3):
                if signal:
                    cand = 1.0 if horse_num == winner else (0.7 if (race + horse_num) % 2 == 0 else -1.0)
                else:
                    cand = 1.0
                finish_rank = 1 if horse_num == winner else (2 if horse_num == losers[race % 2] else 3)
                rows.append(
                    {
                        "race_id": race_id,
                        "horse_num": horse_num,
                        "race_date": pd.Timestamp(f"{year}-06-01"),
                        "finish_rank": finish_rank,
                        "pure_score_z": 0.0,
                        "ln_market_q": np.log(1 / 3),
                        "cand_score": cand,
                    }
                )
    return pd.DataFrame(rows)


def test_split_alpha_gate_cv_uses_2023_fit_and_2024_eval_only():
    df = _candidate_frame()

    fit_df, eval_df = split_alpha_gate_cv(df)

    assert set(pd.to_datetime(fit_df["race_date"]).dt.year.unique()) == {2023}
    assert set(pd.to_datetime(eval_df["race_date"]).dt.year.unique()) == {2024}
    assert 2025 not in set(pd.to_datetime(eval_df["race_date"]).dt.year.unique())


def test_attach_candidate_z_standardizes_within_race():
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1", "R2", "R2", "R2"],
            "cand_score": [3.0, 2.0, 1.0, 5.0, 5.0, 5.0],
        }
    )

    out = attach_candidate_z(df)

    r1 = out.loc[out["race_id"] == "R1", "cand_score_z"]
    r2 = out.loc[out["race_id"] == "R2", "cand_score_z"]
    assert abs(float(r1.mean())) < 1e-8
    assert np.isclose(float(r1.std()), 1.0)
    assert np.allclose(r2.to_numpy(), 0.0)


def test_run_alpha_gate_detects_out_of_year_candidate_signal():
    df = attach_candidate_z(_candidate_frame(signal=True))

    report = run_alpha_gate_on_dataframe(df, candidate_name="synthetic_signal")

    assert report["candidate"] == "synthetic_signal"
    assert report["fit_n_races"] == 12
    assert report["eval_n_races"] == 12
    assert report["gamma"] > 0
    assert report["gamma_lrt_p"] < 0.01
    assert report["delta_ll_per_race"] > 0
    assert report["pass"] is True
    assert report["used_test_2025_plus"] is False


def test_run_alpha_gate_rejects_constant_candidate_control():
    df = attach_candidate_z(_candidate_frame(signal=False))

    report = run_alpha_gate_on_dataframe(df, candidate_name="constant_control")

    assert abs(report["gamma"]) < 1e-6
    assert report["gamma_lrt_p"] >= 0.01
    assert report["pass"] is False


def test_run_alpha_gate_rejects_seeded_random_candidate_control():
    df = _candidate_frame(signal=False)
    rng = np.random.default_rng(42)
    df["cand_score"] = rng.normal(size=len(df))
    df = attach_candidate_z(df)

    report = run_alpha_gate_on_dataframe(df, candidate_name="random_control")

    assert report["gamma_lrt_p"] >= 0.01
    assert report["pass"] is False


def test_run_alpha_gate_flags_finish_rank_proxy_as_positive_leak_control():
    df = _candidate_frame(signal=True)
    # Strong outcome proxy: this should be statistically useful, but must be
    # rejected by the leak gate.
    df["cand_score"] = -pd.to_numeric(df["finish_rank"], errors="coerce")
    df = attach_candidate_z(df)

    report = run_alpha_gate_on_dataframe(df, candidate_name="finish_rank_leak")

    assert report["gamma_lrt_p"] < 0.01
    assert report["leak_warnings"]["fit"]["leak_warning"] is True
    assert report["pass"] is False


def test_run_alpha_gate_checks_eval_year_leak_warnings():
    df = _candidate_frame(signal=False)
    eval_mask = pd.to_datetime(df["race_date"]).dt.year == 2024
    df.loc[eval_mask, "cand_score"] = -pd.to_numeric(df.loc[eval_mask, "finish_rank"], errors="coerce")
    df = attach_candidate_z(df)

    report = run_alpha_gate_on_dataframe(df, candidate_name="eval_only_leak")

    assert report["leak_warnings"]["fit"]["leak_warning"] is False
    assert report["leak_warnings"]["eval"]["leak_warning"] is True
    assert report["gates"]["leak_warning_clear"] is False
    assert report["pass"] is False


def test_evaluate_leak_warnings_flags_finish_rank_proxy():
    df = _candidate_frame(signal=True)
    df["cand_score_z"] = -pd.to_numeric(df["finish_rank"], errors="coerce")

    warnings = evaluate_leak_warnings(df)

    assert warnings["finish_rank_abs_corr"] >= 0.7
    assert warnings["leak_warning"] is True
