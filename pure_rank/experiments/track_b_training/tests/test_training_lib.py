"""TDD tests for track_b_training.training_lib (spec section 8, items 1-8, 10, 11).

All tests use synthetic data only -- no real HC/WC/scores parquet required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import training_lib as tl  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# 1. B-1 slope (also covers slope_ols primitives reused by B-4)
# ─────────────────────────────────────────────────────────────────────────

def test_slope_ols_recovers_known_slope():
    # y decreases by 0.1 sec/day as t increases (t is negative day-offset, e.g. -20..-1)
    t = np.array([-20.0, -15.0, -10.0, -5.0, -1.0])
    intercept = 40.0
    y = intercept + 0.1 * t  # slope = +0.1 -> cand_score = -slope = -0.1 in this raw convention
    slope = tl.slope_ols(t, y, min_n=3)
    assert slope == pytest.approx(0.1, abs=1e-8)


def test_slope_ols_nan_below_min_n():
    t = np.array([-5.0, -1.0])
    y = np.array([36.0, 35.5])
    assert np.isnan(tl.slope_ols(t, y, min_n=3))


def test_slope_ols_nan_all_same_day():
    t = np.array([-5.0, -5.0, -5.0])
    y = np.array([36.0, 35.0, 37.0])
    assert np.isnan(tl.slope_ols(t, y, min_n=3))


def test_slope_ols_zero_for_flat_fit():
    t = np.array([-20.0, -10.0, -1.0])
    y = np.array([36.0, 36.0, 36.0])
    slope = tl.slope_ols(t, y, min_n=3)
    assert slope == pytest.approx(0.0, abs=1e-8)


def test_build_b1_intensity_trend_sign_and_window():
    race_date = pd.Timestamp("2024-06-01")
    ketto = "H001"
    # 3 workouts in-window improving (getting faster): slope negative -> cand positive
    hc = pd.DataFrame({
        "ketto_num": [ketto, ketto, ketto],
        "training_date": [
            race_date - pd.Timedelta(days=20),
            race_date - pd.Timedelta(days=10),
            race_date - pd.Timedelta(days=1),
        ],
        "hc_4f_sec": [37.0, 36.0, 35.0],  # improving (decreasing) over time
    })
    race_keys = pd.DataFrame({
        "race_id": ["R1"],
        "horse_num": [1],
        "ketto_num": [ketto],
        "race_date": [race_date],
    })
    out = tl.build_b1_intensity_trend(hc, race_keys, window_days=30, min_n=3)
    assert list(out.columns) == ["race_id", "horse_num", "cand_score"]
    assert out["cand_score"].iloc[0] > 0  # improving horse -> positive candidate score


def test_build_b1_excludes_same_day_and_out_of_window():
    race_date = pd.Timestamp("2024-06-01")
    ketto = "H001"
    hc = pd.DataFrame({
        "ketto_num": [ketto, ketto, ketto, ketto],
        "training_date": [
            race_date,  # same-day: must be excluded (boundary test, item 6)
            race_date - pd.Timedelta(days=1),
            race_date - pd.Timedelta(days=10),
            race_date - pd.Timedelta(days=45),  # out of 30d window
        ],
        "hc_4f_sec": [10.0, 36.0, 37.0, 100.0],
    })
    race_keys = pd.DataFrame({
        "race_id": ["R1"], "horse_num": [1], "ketto_num": [ketto], "race_date": [race_date],
    })
    out = tl.build_b1_intensity_trend(hc, race_keys, window_days=30, min_n=3)
    # Only 2 in-window rows remain (day-1, day-10) -> below min_n=3 -> NaN
    assert np.isnan(out["cand_score"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────
# 2. B-2 frequency ratio
# ─────────────────────────────────────────────────────────────────────────

def test_freq_ratio_basic():
    assert tl.freq_ratio(6.0, 3.0) == pytest.approx(2.0)


def test_freq_ratio_nan_when_baseline_zero():
    assert np.isnan(tl.freq_ratio(4.0, 0.0))


def test_freq_ratio_nan_when_nan_inputs():
    assert np.isnan(tl.freq_ratio(float("nan"), 3.0))
    assert np.isnan(tl.freq_ratio(4.0, float("nan")))


def test_expanding_median_shift1_excludes_current_and_requires_min_periods():
    # n_interval sequence across a horse's races in chronological order
    s = pd.Series([np.nan, 4.0, 5.0, 6.0, 3.0])
    out = tl.expanding_median_shift1(s, min_periods=3)
    # race0: no prior -> NaN
    assert np.isnan(out.iloc[0])
    # race1: prior valid = [] (race0 was NaN) -> NaN
    assert np.isnan(out.iloc[1])
    # race2: prior valid = [4.0] -> only 1 -> NaN (min_periods=3)
    assert np.isnan(out.iloc[2])
    # race3: prior valid = [4.0, 5.0] -> only 2 -> NaN
    assert np.isnan(out.iloc[3])
    # race4: prior valid = [4.0, 5.0, 6.0] -> median 5.0 (does NOT include current 3.0)
    assert out.iloc[4] == pytest.approx(5.0)


def test_build_b2_freq_change_initial_race_and_baseline_zero():
    ketto = "H002"
    dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01")]
    race_history = pd.DataFrame({"ketto_num": [ketto] * 3, "race_date": dates})
    hc = pd.DataFrame({"ketto_num": [], "training_date": []})
    race_keys = pd.DataFrame({
        "race_id": ["R1"], "horse_num": [1], "ketto_num": [ketto], "race_date": [dates[0]],
    })
    out = tl.build_b2_freq_change(hc, race_history, race_keys, min_baseline_n=3)
    # first-ever race: prev_race_date is NaN -> raw NaN
    assert np.isnan(out["cand_score"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────
# 3. B-3 recent-vs-career
# ─────────────────────────────────────────────────────────────────────────

def test_recent_minus_career_basic():
    values = np.array([0.1, 0.2, 0.1, 0.3, 0.5, 0.9])  # 6 values, career>=6
    result = tl.recent_minus_career(values, recent_n=3, min_career_n=6)
    career = values.mean()
    recent = values[-3:].mean()
    assert result == pytest.approx(recent - career)


def test_recent_minus_career_nan_below_min_career():
    values = np.array([0.1, 0.2, 0.1, 0.3, 0.5])  # 5 values < 6
    assert np.isnan(tl.recent_minus_career(values, recent_n=3, min_career_n=6))


def test_build_b3_accel_profile_career_boundary():
    race_date = pd.Timestamp("2024-06-01")
    ketto5 = "H5"
    ketto6 = "H6"
    dates5 = [race_date - pd.Timedelta(days=d) for d in (50, 40, 30, 20, 10)]
    dates6 = [race_date - pd.Timedelta(days=d) for d in (60, 50, 40, 30, 20, 10)]
    hc = pd.DataFrame({
        "ketto_num": [ketto5] * 5 + [ketto6] * 6,
        "training_date": dates5 + dates6,
        "hc_accel_sec": [0.1, 0.1, 0.1, 0.1, 0.1] + [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
    })
    race_keys = pd.DataFrame({
        "race_id": ["R5", "R6"],
        "horse_num": [1, 1],
        "ketto_num": [ketto5, ketto6],
        "race_date": [race_date, race_date],
    })
    out = tl.build_b3_accel_profile(hc, race_keys, recent_n=3, min_career_n=6)
    scores = dict(zip(out["horse_num"].astype(str) + "_" + out["race_id"], out["cand_score"]))
    r5 = out.loc[out["race_id"] == "R5", "cand_score"].iloc[0]
    r6 = out.loc[out["race_id"] == "R6", "cand_score"].iloc[0]
    assert np.isnan(r5)  # only 5 valid workouts -> below min_career_n=6
    assert not np.isnan(r6)  # 6 valid workouts -> passes
    assert r6 > 0  # recent (0.9,0.9,0.1-ish) above career avg -> positive


# ─────────────────────────────────────────────────────────────────────────
# 4. B-4 fade rate
# ─────────────────────────────────────────────────────────────────────────

def test_compute_fade_formula():
    hc_200 = pd.Series([13.0])
    hc_3f = pd.Series([36.0])
    fade = tl.compute_fade(hc_200, hc_3f)
    assert fade.iloc[0] == pytest.approx(13.0 / 12.0)


def test_compute_fade_excludes_zero_or_nan_hc3f():
    hc_200 = pd.Series([13.0, 13.0, 13.0])
    hc_3f = pd.Series([0.0, np.nan, 36.0])
    fade = tl.compute_fade(hc_200, hc_3f)
    assert np.isnan(fade.iloc[0])
    assert np.isnan(fade.iloc[1])
    assert not np.isnan(fade.iloc[2])


def test_build_b4_fade_trend_sign_improving():
    race_date = pd.Timestamp("2024-06-01")
    ketto = "H004"
    # fade decreasing over time (improving) -> cand_score positive
    hc = pd.DataFrame({
        "ketto_num": [ketto, ketto, ketto],
        "training_date": [
            race_date - pd.Timedelta(days=20),
            race_date - pd.Timedelta(days=10),
            race_date - pd.Timedelta(days=1),
        ],
        "hc_200_sec": [14.0, 13.5, 13.0],
        "hc_3f_sec": [36.0, 36.0, 36.0],
    })
    race_keys = pd.DataFrame({
        "race_id": ["R1"], "horse_num": [1], "ketto_num": [ketto], "race_date": [race_date],
    })
    out = tl.build_b4_fade_trend(hc, race_keys, window_days=30, min_n=3)
    assert out["cand_score"].iloc[0] > 0


# ─────────────────────────────────────────────────────────────────────────
# 5. B-5 WC share
# ─────────────────────────────────────────────────────────────────────────

def test_wc_share_and_share_diff():
    assert tl.wc_share(2.0, 4.0) == pytest.approx(0.5)
    assert np.isnan(tl.wc_share(2.0, 0.0))
    assert tl.share_diff(0.5, 0.2) == pytest.approx(0.3)
    assert np.isnan(tl.share_diff(float("nan"), 0.2))


def test_build_b5_wc_switch_window_and_career_min_and_wc_start_exclusion():
    race_date = pd.Timestamp("2022-06-01")
    ketto = "H005"
    wc_start = "2021-07-27"
    # career: 3 HC before wc_start (excluded) + 5 HC after wc_start (included) = 5 career HC
    hc_dates = (
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")]
        + [race_date - pd.Timedelta(days=d) for d in (200, 150, 100, 50, 25)]
    )
    hc = pd.DataFrame({"ketto_num": [ketto] * len(hc_dates), "training_date": hc_dates})
    wc_dates = [race_date - pd.Timedelta(days=d) for d in (10,)]
    wc = pd.DataFrame({"ketto_num": [ketto], "training_date": wc_dates})
    race_keys = pd.DataFrame({
        "race_id": ["R1"], "horse_num": [1], "ketto_num": [ketto], "race_date": [race_date],
    })
    out = tl.build_b5_wc_switch(
        hc, wc, race_keys, window_days=30, wc_start=wc_start, min_window_n=2, min_career_n=5,
    )
    # window (last 30d): 1 HC (day 25) + 1 WC (day 10) = 2 total -> meets min_window_n=2
    # career (post wc_start only): 5 HC + 1 WC = 6 total -> meets min_career_n=5
    assert not np.isnan(out["cand_score"].iloc[0])


def test_build_b5_wc_switch_nan_below_window_min():
    race_date = pd.Timestamp("2022-06-01")
    ketto = "H006"
    wc_start = "2021-07-27"
    hc = pd.DataFrame({
        "ketto_num": [ketto] * 5,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (200, 150, 100, 50, 40)],
    })
    wc = pd.DataFrame({"ketto_num": [], "training_date": []})
    race_keys = pd.DataFrame({
        "race_id": ["R1"], "horse_num": [1], "ketto_num": [ketto], "race_date": [race_date],
    })
    out = tl.build_b5_wc_switch(
        hc, wc, race_keys, window_days=30, wc_start=wc_start, min_window_n=2, min_career_n=5,
    )
    # window: 0 workouts in last 30 days -> below min_window_n=2 -> NaN
    assert np.isnan(out["cand_score"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────
# 7. NaN fill regulation
# ─────────────────────────────────────────────────────────────────────────

def test_fill_race_mean_partial_nan():
    df = pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "horse_num": [1, 2, 3],
        "cand_score": [1.0, np.nan, 3.0],
    })
    out = tl.fill_race_mean(df)
    assert out["cand_score"].isna().sum() == 0
    assert out.loc[out["horse_num"] == 2, "cand_score"].iloc[0] == pytest.approx(2.0)  # mean of 1.0,3.0


def test_fill_race_mean_all_nan_race_filled_zero():
    df = pd.DataFrame({
        "race_id": ["R2", "R2"],
        "horse_num": [1, 2],
        "cand_score": [np.nan, np.nan],
    })
    out = tl.fill_race_mean(df)
    assert (out["cand_score"] == 0.0).all()


def test_fill_race_mean_zscore_becomes_zero_for_filled_horse():
    sys.path.insert(0, str(EXP_DIR.parents[1] / "evaluation"))
    from evaluation.alpha_gate import attach_candidate_z

    df = pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "horse_num": [1, 2, 3],
        "cand_score": [1.0, np.nan, 3.0],
    })
    filled = tl.fill_race_mean(df)
    z = attach_candidate_z(filled)
    # the filled horse (mean-value) should have z == 0
    filled_row_z = z.loc[z["horse_num"] == 2, "cand_score_z"].iloc[0]
    assert filled_row_z == pytest.approx(0.0, abs=1e-6)

    df_all_nan = pd.DataFrame({
        "race_id": ["R2", "R2"],
        "horse_num": [1, 2],
        "cand_score": [np.nan, np.nan],
    })
    filled_all = tl.fill_race_mean(df_all_nan)
    z_all = attach_candidate_z(filled_all)
    assert (z_all["cand_score_z"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────
# 8. Output format
# ─────────────────────────────────────────────────────────────────────────

def test_build_b1_output_columns_and_uniqueness():
    race_date = pd.Timestamp("2024-06-01")
    hc = pd.DataFrame({"ketto_num": [], "training_date": [], "hc_4f_sec": []})
    race_keys = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "horse_num": [1, 2],
        "ketto_num": ["A", "B"],
        "race_date": [race_date, race_date],
    })
    out = tl.build_b1_intensity_trend(hc, race_keys, window_days=30, min_n=3)
    assert list(out.columns) == ["race_id", "horse_num", "cand_score"]
    assert not out.duplicated(subset=["race_id", "horse_num"]).any()


# ─────────────────────────────────────────────────────────────────────────
# 10. Sign convention across all improving-horse candidates
# ─────────────────────────────────────────────────────────────────────────

def test_all_candidate_signs_positive_for_improving_horse():
    race_date = pd.Timestamp("2024-06-01")

    # B-1: improving = time shrinking
    hc1 = pd.DataFrame({
        "ketto_num": ["H1"] * 3,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (20, 10, 1)],
        "hc_4f_sec": [37.0, 36.0, 35.0],
    })
    rk1 = pd.DataFrame({"race_id": ["R1"], "horse_num": [1], "ketto_num": ["H1"], "race_date": [race_date]})
    assert tl.build_b1_intensity_trend(hc1, rk1, window_days=30, min_n=3)["cand_score"].iloc[0] > 0

    # B-3: accel upswing recent vs career
    hc3 = pd.DataFrame({
        "ketto_num": ["H3"] * 6,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (60, 50, 40, 30, 20, 10)],
        "hc_accel_sec": [0.0, 0.0, 0.0, 0.9, 0.9, 0.9],
    })
    rk3 = pd.DataFrame({"race_id": ["R3"], "horse_num": [1], "ketto_num": ["H3"], "race_date": [race_date]})
    assert tl.build_b3_accel_profile(hc3, rk3, recent_n=3, min_career_n=6)["cand_score"].iloc[0] > 0

    # B-4: fade decreasing = improving
    hc4 = pd.DataFrame({
        "ketto_num": ["H4"] * 3,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (20, 10, 1)],
        "hc_200_sec": [14.0, 13.5, 13.0],
        "hc_3f_sec": [36.0, 36.0, 36.0],
    })
    rk4 = pd.DataFrame({"race_id": ["R4"], "horse_num": [1], "ketto_num": ["H4"], "race_date": [race_date]})
    assert tl.build_b4_fade_trend(hc4, rk4, window_days=30, min_n=3)["cand_score"].iloc[0] > 0

    # B-5: WC share increasing
    hc5 = pd.DataFrame({
        "ketto_num": ["H5"] * 5,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (200, 150, 100, 60, 45)],
    })
    wc5 = pd.DataFrame({
        "ketto_num": ["H5"],
        "training_date": [race_date - pd.Timedelta(days=10)],
    })
    rk5 = pd.DataFrame({"race_id": ["R5"], "horse_num": [1], "ketto_num": ["H5"], "race_date": [race_date]})
    out5 = tl.build_b5_wc_switch(
        hc5, wc5, rk5, window_days=30, wc_start="2021-07-27", min_window_n=1, min_career_n=5,
    )
    assert out5["cand_score"].iloc[0] > 0


# ─────────────────────────────────────────────────────────────────────────
# 11. Reproducibility (deterministic, no RNG involved -> re-run equality)
# ─────────────────────────────────────────────────────────────────────────

def test_reproducibility_b1_deterministic():
    race_date = pd.Timestamp("2024-06-01")
    hc = pd.DataFrame({
        "ketto_num": ["H1"] * 3,
        "training_date": [race_date - pd.Timedelta(days=d) for d in (20, 10, 1)],
        "hc_4f_sec": [37.0, 36.2, 35.1],
    })
    rk = pd.DataFrame({"race_id": ["R1"], "horse_num": [1], "ketto_num": ["H1"], "race_date": [race_date]})
    out1 = tl.build_b1_intensity_trend(hc, rk, window_days=30, min_n=3)
    out2 = tl.build_b1_intensity_trend(hc, rk, window_days=30, min_n=3)
    assert out1["cand_score"].iloc[0] == out2["cand_score"].iloc[0]
