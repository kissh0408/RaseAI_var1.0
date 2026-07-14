"""TDD テスト: variable_sizing の純関数（合成データのみ、実データ不要）。

仕様書 docs/specs/2026-07-11-variable-sizing-spec.md §9 の項目1,3,4,5,7,8,9,12 対応。
項目2（階層割当・凍結境界の import 再利用）は test_reuse_guard.py、
項目6（市場情報混入の静的検査）は test_static_guards.py、
項目10（性能主張の不在）は test_no_performance_claim.py、
項目11（危険信号フラグ）は本ファイル末尾に収録。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import sizing_lib as sl  # noqa: E402
from betting.src.flat_top1 import apply_flat_sizing, settle_win_bets  # noqa: E402

MULTIPLIERS = [0.5, 0.75, 1.25, 1.5]
BOUNDARIES = [0.11895233392715454, 0.28122442960739136, 0.5161097198724747]


# ---------------------------------------------------------------------------
# 1. 有界性・単調性
# ---------------------------------------------------------------------------


def test_multipliers_match_config_and_are_bounded():
    sl.validate_multipliers(MULTIPLIERS)  # should not raise


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_multiplier_for_tier_within_bounds(tier):
    m = sl.multiplier_for_tier(tier, MULTIPLIERS)
    assert 0.5 <= m <= 1.5


def test_multipliers_strictly_monotonic():
    m = [sl.multiplier_for_tier(t, MULTIPLIERS) for t in (1, 2, 3, 4)]
    assert m == sorted(m)
    assert len(set(m)) == 4


def test_validate_multipliers_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        sl.validate_multipliers([0.4, 0.75, 1.25, 1.5])
    with pytest.raises(ValueError):
        sl.validate_multipliers([0.5, 0.75, 1.25, 1.6])


def test_validate_multipliers_rejects_non_monotonic():
    with pytest.raises(ValueError):
        sl.validate_multipliers([0.5, 0.9, 0.8, 1.5])


def test_validate_multipliers_rejects_wrong_length():
    with pytest.raises(ValueError):
        sl.validate_multipliers([0.5, 1.0, 1.5])


def test_multiplier_for_tier_extreme_margin_inputs_via_assign_tier():
    # 0・負値ガード・極大値を含む margin 入力でも tier は常に 1..4 に収まる。
    for margin in (0.0, -1.0, -1e9, 1e9, float("inf")):
        tier = sl.assign_tier(margin, BOUNDARIES)
        assert 1 <= tier <= 4
        m = sl.multiplier_for_tier(tier, MULTIPLIERS)
        assert 0.5 <= m <= 1.5


# ---------------------------------------------------------------------------
# 3. リスク予算保存則
# ---------------------------------------------------------------------------


def test_occupancy_equal_weights_gives_m_bar_one():
    tiers = [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25
    occ = sl.compute_tier_occupancy(tiers)
    assert occ[1] == pytest.approx(0.25)
    assert occ[2] == pytest.approx(0.25)
    assert occ[3] == pytest.approx(0.25)
    assert occ[4] == pytest.approx(0.25)
    m_bar = sl.weighted_mean_multiplier(occ, MULTIPLIERS)
    assert m_bar == pytest.approx(1.0)
    assert sl.budget_preserved(m_bar) is True


def test_occupancy_reference_confidence_tiers_counts():
    # confidence-tiers §15 実測占有 (666/677/641/624)。厳密計算値は約0.9885（[0.95,1.05]内）。
    tiers = [1] * 666 + [2] * 677 + [3] * 641 + [4] * 624
    occ = sl.compute_tier_occupancy(tiers)
    m_bar = sl.weighted_mean_multiplier(occ, MULTIPLIERS)
    total = 666 + 677 + 641 + 624
    expected = (
        (666 / total) * 0.5 + (677 / total) * 0.75 + (641 / total) * 1.25 + (624 / total) * 1.5
    )
    assert m_bar == pytest.approx(expected, abs=1e-9)
    assert sl.budget_preserved(m_bar) is True


def test_budget_preserved_false_outside_tolerance():
    # 偏った占有率 (T1のみ) -> M̄=0.5 -> [0.95,1.05] 外
    tiers = [1] * 100
    occ = sl.compute_tier_occupancy(tiers)
    m_bar = sl.weighted_mean_multiplier(occ, MULTIPLIERS)
    assert m_bar == pytest.approx(0.5)
    assert sl.budget_preserved(m_bar) is False

    tiers_high = [4] * 100
    occ_high = sl.compute_tier_occupancy(tiers_high)
    m_bar_high = sl.weighted_mean_multiplier(occ_high, MULTIPLIERS)
    assert m_bar_high == pytest.approx(1.5)
    assert sl.budget_preserved(m_bar_high) is False


def test_occupancy_signature_is_outcome_blind():
    import inspect

    sig = inspect.signature(sl.compute_tier_occupancy)
    params = set(sig.parameters.keys())
    forbidden = {"finish_rank", "payout", "odds", "win", "roi", "hit", "stake"}
    assert not (params & forbidden), f"outcome列を引数に取ってはならない: {params}"


# ---------------------------------------------------------------------------
# 4. 100円丸め
# ---------------------------------------------------------------------------


def test_base_stake_exact_multiples_bankroll_400000():
    base = sl.compute_base_stake(400_000, 0.001)
    assert base == pytest.approx(400.0)
    stakes = {t: base * m for t, m in zip((1, 2, 3, 4), MULTIPLIERS)}
    assert stakes[1] == pytest.approx(200.0)
    assert stakes[2] == pytest.approx(300.0)
    assert stakes[3] == pytest.approx(500.0)
    assert stakes[4] == pytest.approx(600.0)
    for v in stakes.values():
        assert v % 100 == 0


def test_base_stake_rejects_non_multiple_of_400_bankroll():
    with pytest.raises(ValueError):
        sl.compute_base_stake(100_000, 0.001)  # base=100, not multiple of 400
    with pytest.raises(ValueError):
        sl.compute_base_stake(350_000, 0.001)  # base=350, not multiple of 400


def test_base_stake_f_var_0005_needs_bankroll_800000():
    min_bankroll = sl.min_bankroll_variable(0.0005)
    assert min_bankroll == pytest.approx(800_000.0)
    base = sl.compute_base_stake(min_bankroll, 0.0005)
    assert base == pytest.approx(400.0)
    stakes = [base * m for m in MULTIPLIERS]
    assert stakes == pytest.approx([200.0, 300.0, 500.0, 600.0])


def test_min_bankroll_variable_f0001():
    assert sl.min_bankroll_variable(0.001) == pytest.approx(400_000.0)


# ---------------------------------------------------------------------------
# 5. 実効倍率の一致
# ---------------------------------------------------------------------------


def test_effective_multiplier_matches_design_exactly():
    df = pd.DataFrame({"tier": [1, 2, 3, 4, 1, 4]})
    base_stake = 400.0
    sized = sl.apply_variable_stake(df, base_stake=base_stake, multipliers=MULTIPLIERS)
    eff = sl.effective_multiplier(sized["stake"], base_stake)
    expected = [MULTIPLIERS[t - 1] for t in df["tier"]]
    assert list(eff) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. flat比較の決済一致性
# ---------------------------------------------------------------------------


def _synthetic_picks(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(n)],
            "race_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "horse_num": rng.integers(1, 10, size=n),
            "tier": rng.integers(1, 5, size=n),
            "odds": rng.uniform(2.0, 20.0, size=n),
            "finish_rank": rng.integers(1, 10, size=n),
        }
    )


def test_variable_and_flat_series_share_identical_bet_set():
    picks = _synthetic_picks()

    var_sized = sl.apply_variable_stake(picks, base_stake=400.0, multipliers=MULTIPLIERS)
    var_settled = settle_win_bets(var_sized)

    flat_sized = apply_flat_sizing(picks, bankroll=400_000, stake_fraction=0.001)
    flat_settled = settle_win_bets(flat_sized)

    assert len(var_settled) == len(flat_settled) == len(picks)
    assert set(var_settled["race_id"]) == set(flat_settled["race_id"])
    assert list(var_settled["win"]) == list(flat_settled["win"])
    # stake列のみが異なる（他の列は同一値）
    assert not (var_settled["stake"] == flat_settled["stake"]).all()


# ---------------------------------------------------------------------------
# 8. 月次MDDセマンティクス（derive_flat_fraction と同値であることの照合）
# ---------------------------------------------------------------------------


def test_monthly_max_drawdown_matches_hand_calc():
    dates = pd.Series(
        pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-15", "2024-03-01"])
    )
    pnl = pd.Series([-100.0, -50.0, 200.0, -300.0, 10.0])
    bankroll = 1000.0
    # Jan: -150 -> loss_ratio 0.15; Feb: -100 -> 0.10; Mar: +10 -> loss_ratio 0 (clipped)
    worst, by_month = sl.monthly_max_drawdown(dates, pnl, bankroll)
    assert worst == pytest.approx(0.15)
    assert by_month["2024-01"] == pytest.approx(0.15)
    assert by_month["2024-02"] == pytest.approx(0.10)
    assert by_month["2024-03"] == pytest.approx(0.0)


def test_monthly_max_drawdown_linear_in_f():
    dates = pd.Series(pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10"]))
    pnl_base = pd.Series([-100.0, -50.0, -30.0])
    bankroll = 1000.0
    worst_base, _ = sl.monthly_max_drawdown(dates, pnl_base, bankroll)
    worst_2x, _ = sl.monthly_max_drawdown(dates, pnl_base * 2.0, bankroll)
    assert worst_2x == pytest.approx(worst_base * 2.0)


def test_busiest_day_exposure_reused_from_derive_flat_fraction():
    dates = pd.Series(
        pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-05", "2024-01-06"])
    )
    exposure, busiest_day, n = sl.busiest_day_exposure(dates, 0.001)
    assert n == 3
    assert busiest_day == "2024-01-05"
    assert exposure == pytest.approx(3 * 0.001)


# ---------------------------------------------------------------------------
# 9. f_var機械導出
# ---------------------------------------------------------------------------


def test_derive_f_var_matches_decision_rule():
    grid = [0.001, 0.0005, 0.00025]
    result = sl.derive_f_var(0.10, 0.001, grid=grid, monthly_mdd_limit=0.15, safety_factor=0.5)
    f_scale_expected = 0.15 / (0.10 / 0.001)
    f_capped_expected = 0.5 * f_scale_expected
    assert result["f_scale"] == pytest.approx(f_scale_expected)
    assert result["f_capped"] == pytest.approx(f_capped_expected)
    eligible = [f for f in grid if f <= f_capped_expected]
    expected_adopted = max(eligible) if eligible else None
    assert result["adopted_f_var"] == expected_adopted


def test_derive_f_var_none_when_no_candidate_eligible_and_no_upward_extension():
    # worst_month_dd very large -> f_capped tiny -> no grid candidate eligible
    grid = [0.001, 0.0005, 0.00025]
    result = sl.derive_f_var(10.0, 0.001, grid=grid, monthly_mdd_limit=0.15, safety_factor=0.5)
    assert result["adopted_f_var"] is None
    # grid unchanged (no upward extension)
    assert result["grid"] == grid


def test_derive_f_var_high_dd_selects_smaller_grid_value():
    # worst_month_dd@f0 such that f_capped falls strictly between 0.0005 and 0.001
    grid = [0.001, 0.0005, 0.00025]
    # f_scale = 0.15/(dd/0.001); want f_capped in (0.0005, 0.001)
    # choose dd=0.12 -> f_scale=0.00125 -> f_capped=0.000625 -> eligible {0.0005,0.00025} -> adopt 0.0005
    result = sl.derive_f_var(0.12, 0.001, grid=grid, monthly_mdd_limit=0.15, safety_factor=0.5)
    assert result["adopted_f_var"] == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# 11. 危険信号フラグ（tiers_lib の既存フラグ関数を import 再利用）
# ---------------------------------------------------------------------------


def test_danger_flags_reused_from_tiers_lib():
    assert sl.danger_roi_gt_100(1.01) is True
    assert sl.danger_roi_gt_100(0.83) is False
    assert sl.leak_review_flag(0.41) is True
    assert sl.leak_review_flag(0.30) is False


# ---------------------------------------------------------------------------
# 12. 再現性（決定的seed・安定ソート）
# ---------------------------------------------------------------------------


def test_apply_variable_stake_deterministic():
    picks = _synthetic_picks()
    r1 = sl.apply_variable_stake(picks, base_stake=400.0, multipliers=MULTIPLIERS)
    r2 = sl.apply_variable_stake(picks, base_stake=400.0, multipliers=MULTIPLIERS)
    pd.testing.assert_frame_equal(r1, r2)


def test_build_result_envelope_deterministic_and_contains_keys():
    env1 = sl.build_result_envelope({"a": 1})
    env2 = sl.build_result_envelope({"a": 1})
    assert env1 == env2
    assert env1["disclaimer"] == sl.DISCLAIMER
    assert sl.CAVEAT_CONFIDENCE_DOES_NOT_PREDICT_EDGE in env1["caveats"]
    assert sl.CAVEAT_DILUTION_RISK in env1["caveats"]
