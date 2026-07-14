"""TDD テスト: confidence_tiers の純関数（合成データのみ、実データ不要）。

仕様書 docs/specs/2026-07-11-confidence-tiers-spec.md §11 の項目1〜9,13 対応。
項目10（市場情報混入の静的検査）・項目12（L1非追加担保）は test_static_guards.py、
項目11（import再利用の静的検査）は test_reuse_guard.py に分離。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import tiers_lib as tl  # noqa: E402


# ---------------------------------------------------------------------------
# 1. margin 計算
# ---------------------------------------------------------------------------


def test_margin_basic():
    scores = [2.0, 1.5, 0.0]
    horse_nums = [1, 2, 3]
    assert tl.compute_race_margin(scores, horse_nums) == pytest.approx(0.5)


def test_margin_tie_break_by_horse_num():
    # 1位・2位が同値スコアの場合、select_top1_bets同様に馬番昇順でタイブレークする。
    # 1位=馬番1(score=2.0), 2位=馬番2 or 3(同値score=2.0のうち馬番が小さい方) -> margin=0
    scores = [2.0, 2.0, 2.0, 1.0]
    horse_nums = [5, 2, 8, 1]
    margin = tl.compute_race_margin(scores, horse_nums)
    assert margin == pytest.approx(0.0)


def test_margin_all_tied_is_zero():
    scores = [1.0, 1.0, 1.0, 1.0]
    horse_nums = [4, 1, 3, 2]
    assert tl.compute_race_margin(scores, horse_nums) == pytest.approx(0.0)


def test_margin_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        tl.compute_race_margin([1.0, 2.0], [1])


# ---------------------------------------------------------------------------
# 2. 階層割当の境界テスト
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "margin,expected_tier",
    [
        (0.19, 1),
        (0.20, 1),   # 境界値ちょうどは下位階層
        (0.21, 2),
        (0.50, 2),   # 境界値ちょうどは下位階層
        (0.90, 3),   # 境界値ちょうどは下位階層
        (0.91, 4),
    ],
)
def test_assign_tier_boundaries(margin, expected_tier):
    boundaries = [0.2, 0.5, 0.9]
    assert tl.assign_tier(margin, boundaries) == expected_tier


def test_assign_tier_deterministic():
    boundaries = [0.2, 0.5, 0.9]
    r1 = [tl.assign_tier(m, boundaries) for m in [0.1, 0.3, 0.6, 0.95]]
    r2 = [tl.assign_tier(m, boundaries) for m in [0.1, 0.3, 0.6, 0.95]]
    assert r1 == r2 == [1, 2, 3, 4]


def test_assign_tier_batch_matches_scalar():
    boundaries = [0.2, 0.5, 0.9]
    margins = [0.19, 0.20, 0.21, 0.50, 0.90, 0.91]
    batch = tl.assign_tier_batch(margins, boundaries)
    scalar = [tl.assign_tier(m, boundaries) for m in margins]
    assert list(batch) == scalar


# ---------------------------------------------------------------------------
# 3. 四分位境界の再現性・outcome-blind構造
# ---------------------------------------------------------------------------


def test_quartile_boundaries_matches_numpy_quantile():
    margins = [0.05, 0.4, 0.1, 0.8, 0.2, 1.2, 0.02, 0.65, 0.33, 0.55]
    expected = np.quantile(margins, [0.25, 0.5, 0.75], method="linear")
    b1, b2, b3 = tl.compute_quartile_boundaries(margins)
    assert (b1, b2, b3) == pytest.approx(tuple(expected))


def test_quartile_boundaries_signature_is_outcome_blind():
    import inspect

    sig = inspect.signature(tl.compute_quartile_boundaries)
    params = set(sig.parameters.keys())
    forbidden = {"finish_rank", "payout", "odds", "win", "roi", "hit"}
    assert not (params & forbidden), f"outcome列を引数に取ってはならない: {params}"


# ---------------------------------------------------------------------------
# 4. ペア比較の整合
# ---------------------------------------------------------------------------


def test_roi_and_delta_hand_calc():
    # 3レース、stake=100均等。model: 2勝(odds 3.0,5.0), 1敗
    stakes_m = [100.0, 100.0, 100.0]
    payouts_m = [300.0, 500.0, 0.0]
    # favorite: 1勝(odds 2.0), 2敗
    stakes_f = [100.0, 100.0, 100.0]
    payouts_f = [200.0, 0.0, 0.0]

    roi_m = tl.compute_roi(stakes_m, payouts_m)
    roi_f = tl.compute_roi(stakes_f, payouts_f)
    assert roi_m == pytest.approx(800.0 / 300.0)
    assert roi_f == pytest.approx(200.0 / 300.0)
    assert tl.compute_delta(roi_m, roi_f) == pytest.approx(roi_m - roi_f)


def test_per_race_payout_diff_zero_when_model_equals_favorite():
    # レース1,3はモデル1位=1番人気(payout一致) -> diff=0。レース2は不一致。
    payout_model = [300.0, 500.0, 0.0]
    payout_fav = [300.0, 0.0, 0.0]
    diff = tl.per_race_payout_diff(payout_model, payout_fav)
    assert diff[0] == pytest.approx(0.0)
    assert diff[2] == pytest.approx(0.0)
    assert diff[1] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 5. ブートストラップ陽性・陰性コントロール（H1〜H4相当のΔ検定）
# ---------------------------------------------------------------------------


def _make_delta_injected_arrays(n: int, *, roi_model: float, roi_fav: float, odds_model: float, odds_fav: float):
    """n件、stake=100均等。win_rate = roi/odds となるよう決定的に的中パターンを作る。"""
    hit_rate_m = roi_model / odds_model
    hit_rate_f = roi_fav / odds_fav
    n_hit_m = int(round(hit_rate_m * n))
    n_hit_f = int(round(hit_rate_f * n))

    stakes_m = np.full(n, 100.0)
    payouts_m = np.zeros(n)
    payouts_m[:n_hit_m] = 100.0 * odds_model

    stakes_f = np.full(n, 100.0)
    payouts_f = np.zeros(n)
    payouts_f[:n_hit_f] = 100.0 * odds_fav
    return stakes_m, payouts_m, stakes_f, payouts_f


def test_bootstrap_positive_control_delta_plus8pp():
    n = 500
    # ROI_model=88%, ROI_fav=80% (Δ=+8pp), hit_rate=25%均等(odds固定)で決定的に注入
    stakes_m, payouts_m, stakes_f, payouts_f = _make_delta_injected_arrays(
        n, roi_model=0.88, roi_fav=0.80, odds_model=3.52, odds_fav=3.20
    )
    result = tl.cluster_bootstrap_delta_p_value(stakes_m, payouts_m, stakes_f, payouts_f, B=10000, seed=42)
    assert result["delta_hat"] == pytest.approx(0.08, abs=1e-6)
    assert result["p_value"] < 0.002


def test_bootstrap_negative_control_delta_zero():
    n = 500
    # 前半250レースはmodelが勝ちfavが負け、後半250レースは逆。ROI_model==ROI_fav(=100%)、
    # レース単位では相関した変動があるため妥当なブートストラップ分散が生じる。
    stakes_m = np.full(n, 100.0)
    payouts_m = np.zeros(n)
    payouts_m[:250] = 200.0

    stakes_f = np.full(n, 100.0)
    payouts_f = np.zeros(n)
    payouts_f[250:] = 200.0

    result = tl.cluster_bootstrap_delta_p_value(stakes_m, payouts_m, stakes_f, payouts_f, B=10000, seed=42)
    assert result["delta_hat"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] > 0.05


def test_bootstrap_seed_determinism():
    n = 300
    stakes_m, payouts_m, stakes_f, payouts_f = _make_delta_injected_arrays(
        n, roi_model=0.90, roi_fav=0.82, odds_model=3.0, odds_fav=2.8
    )
    r1 = tl.cluster_bootstrap_delta_p_value(stakes_m, payouts_m, stakes_f, payouts_f, B=2000, seed=42)
    r2 = tl.cluster_bootstrap_delta_p_value(stakes_m, payouts_m, stakes_f, payouts_f, B=2000, seed=42)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 6. 順序コントラスト（H_ord: Δ(T4)-Δ(T1)）
# ---------------------------------------------------------------------------


def test_ordering_contrast_positive_control_plus10pp():
    n = 400
    # T1: Δ=0.00 (roi_model=roi_fav=80%), T4: Δ=0.10 (roi_model=90%, roi_fav=80%)
    t1_sm, t1_pm, t1_sf, t1_pf = _make_delta_injected_arrays(
        n, roi_model=0.80, roi_fav=0.80, odds_model=3.2, odds_fav=3.2
    )
    t4_sm, t4_pm, t4_sf, t4_pf = _make_delta_injected_arrays(
        n, roi_model=0.90, roi_fav=0.80, odds_model=3.6, odds_fav=3.2
    )
    result = tl.cluster_bootstrap_ordering_contrast(
        t1_sm, t1_pm, t1_sf, t1_pf, t4_sm, t4_pm, t4_sf, t4_pf, B=10000, seed=42
    )
    assert result["c_hat"] == pytest.approx(0.10, abs=1e-6)
    assert result["p_value"] < 0.002


def test_ordering_contrast_negative_control_zero_diff():
    n = 400
    # T1・T4 とも同一のΔ(=0.05) -> C=0
    t1_sm, t1_pm, t1_sf, t1_pf = _make_delta_injected_arrays(
        n, roi_model=0.85, roi_fav=0.80, odds_model=3.4, odds_fav=3.2
    )
    t4_sm, t4_pm, t4_sf, t4_pf = _make_delta_injected_arrays(
        n, roi_model=0.85, roi_fav=0.80, odds_model=3.4, odds_fav=3.2
    )
    result = tl.cluster_bootstrap_ordering_contrast(
        t1_sm, t1_pm, t1_sf, t1_pf, t4_sm, t4_pm, t4_sf, t4_pf, B=10000, seed=42
    )
    assert result["c_hat"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] > 0.05


def test_monotonicity_flag():
    assert tl.monotonicity_flag([0.01, 0.02, 0.03, 0.04]) is True
    assert tl.monotonicity_flag([0.04, 0.02, 0.03, 0.01]) is False
    assert tl.monotonicity_flag([0.01, 0.01, 0.01, 0.01]) is True  # 等号は単調とみなす


# ---------------------------------------------------------------------------
# 7. 最小サンプル
# ---------------------------------------------------------------------------


def test_min_sample_boundary():
    assert tl.min_sample_ok(199) is False
    assert tl.min_sample_ok(200) is True


def test_all_tiers_min_sample_ok_blocks_ordering_hypothesis():
    ok = {1: 650, 2: 650, 3: 650, 4: 650}
    bad = {1: 650, 2: 650, 3: 650, 4: 199}
    assert tl.all_tiers_min_sample_ok(ok) is True
    assert tl.all_tiers_min_sample_ok(bad) is False


# ---------------------------------------------------------------------------
# 8. Bonferroni
# ---------------------------------------------------------------------------


def test_bonferroni_threshold_fixed_at_0002():
    assert tl.bonferroni_threshold() == pytest.approx(0.002)
    assert tl.bonferroni_threshold(k=tl.K_HYP, alpha=tl.BONFERRONI_ALPHA) == pytest.approx(0.002)


def test_bonferroni_threshold_unaffected_by_hold_status():
    # 保留階層が発生しても閾値計算そのものは常にk=5固定で呼び出される(呼び出し側の責務)。
    thr_full = tl.bonferroni_threshold(k=5)
    thr_still_full = tl.bonferroni_threshold(k=5)  # 保留があっても k を減らして呼ばない
    assert thr_full == thr_still_full == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 9. 危険信号フラグ
# ---------------------------------------------------------------------------


def test_leak_review_flag():
    assert tl.leak_review_flag(0.41) is True
    assert tl.leak_review_flag(0.40) is False  # 閾値ちょうどは超過ではない
    assert tl.leak_review_flag(0.30) is False


def test_danger_roi_gt_100():
    assert tl.danger_roi_gt_100(1.01) is True
    assert tl.danger_roi_gt_100(1.00) is False
    assert tl.danger_roi_gt_100(0.83) is False


def test_large_delta_flag():
    assert tl.large_delta_flag(0.21) is True
    assert tl.large_delta_flag(-0.21) is True
    assert tl.large_delta_flag(0.20) is False
    assert tl.large_delta_flag(0.05) is False


def test_payout_concentration_gate_reused_from_divergence_lib():
    # top1_payout_share=0.31 -> diagnosis_valid=False
    gate1 = tl.payout_concentration_gate([310.0, 690.0], n_hits=10)
    assert gate1["diagnosis_valid"] is False
    # n_hits=9 (<10) -> diagnosis_valid=False
    gate2 = tl.payout_concentration_gate([100.0] * 9, n_hits=9)
    assert gate2["diagnosis_valid"] is False
    # 両方満たす場合は True
    gate3 = tl.payout_concentration_gate([100.0] * 20, n_hits=20)
    assert gate3["diagnosis_valid"] is True


# ---------------------------------------------------------------------------
# 13. 再現性（全体）
# ---------------------------------------------------------------------------


def test_all_bootstrap_functions_deterministic_with_fixed_seed():
    n = 250
    sm, pm, sf, pf = _make_delta_injected_arrays(n, roi_model=0.86, roi_fav=0.81, odds_model=3.1, odds_fav=2.9)
    r1 = tl.cluster_bootstrap_delta_p_value(sm, pm, sf, pf, B=1000, seed=42)
    r2 = tl.cluster_bootstrap_delta_p_value(sm, pm, sf, pf, B=1000, seed=42)
    assert r1 == r2

    o1 = tl.cluster_bootstrap_ordering_contrast(sm, pm, sf, pf, sm, pm, sf, pf, B=1000, seed=42)
    o2 = tl.cluster_bootstrap_ordering_contrast(sm, pm, sf, pf, sm, pm, sf, pf, B=1000, seed=42)
    assert o1 == o2
