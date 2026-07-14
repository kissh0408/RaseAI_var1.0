"""TDD テスト: cross_pool_divergence の純関数（合成データのみ、実データ不要）。

仕様書 docs/specs/2026-07-10-p4-cross-pool-divergence-spec.md §9 の項目 1〜13,15,16 対応。
項目14（L1不使用の静的検査）は test_static_guards.py に分離。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import divergence_lib as dl  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 人気順位付与
# ---------------------------------------------------------------------------

def test_popularity_rank_basic():
    odds = [2.0, 3.5, 3.5, 10.0]
    horse_nums = [1, 2, 3, 4]
    ranks = dl.assign_popularity_rank(odds, horse_nums)
    assert list(ranks) == [1, 2, 3, 4]


def test_popularity_rank_tie_break_by_horse_num():
    # 同オッズの2頭のうち馬番が小さい方が上位順位になる
    odds = [5.0, 2.0, 2.0]
    horse_nums = [3, 5, 1]
    ranks = dl.assign_popularity_rank(odds, horse_nums)
    # horse 1 (odds 2.0, num1) -> rank1, horse 5(odds2.0,num5)->rank2, horse3(odds5.0)->rank3
    rank_by_horse = dict(zip(horse_nums, ranks))
    assert rank_by_horse[1] == 1
    assert rank_by_horse[5] == 2
    assert rank_by_horse[3] == 3


def test_popularity_rank_deterministic_repeat():
    odds = [4.0, 4.0, 4.0, 1.5]
    horse_nums = [7, 2, 5, 1]
    r1 = dl.assign_popularity_rank(odds, horse_nums)
    r2 = dl.assign_popularity_rank(odds, horse_nums)
    assert list(r1) == list(r2)


# ---------------------------------------------------------------------------
# 2. 人気帯割当（単体・複勝）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rank,expected",
    [
        (1, "POP1"),
        (2, "POP2"),
        (3, "POP2"),
        (4, "POP3"),
        (6, "POP3"),
        (7, "POP4"),
        (10, "POP4"),
    ],
)
def test_pop_band_single_boundaries(rank, expected):
    assert dl.assign_pop_band_single(rank) == expected


# ---------------------------------------------------------------------------
# 3. ペア帯割当
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ri,rj,expected",
    [
        (1, 2, "PAIR_TOP"),
        (1, 4, "PAIR_MIX"),
        (4, 5, "PAIR_LONG"),  # 境界: 両方とも4番人気以下
        (3, 4, "PAIR_MIX"),  # 境界: 片方のみ3番人気以内
        (2, 3, "PAIR_TOP"),  # 境界: 両方3番人気以内（(3,3)は同一レースで人気順位が重複しないため不可）
    ],
)
def test_pair_band_boundaries(ri, rj, expected):
    assert dl.assign_pair_band(ri, rj) == expected


# ---------------------------------------------------------------------------
# 4. 頭数帯
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hc,expected",
    [
        (8, "FS_S"),
        (9, "FS_M"),
        (13, "FS_M"),
        (14, "FS_L"),
        (18, "FS_L"),
        (19, None),
    ],
)
def test_fs_band_boundaries(hc, expected):
    assert dl.assign_fs_band(hc) == expected


# ---------------------------------------------------------------------------
# 5. 複勝の m と的中定義切替
# ---------------------------------------------------------------------------

def test_place_m_and_hit_small_field():
    m, y = dl.place_m_and_hit(horse_count=7, finish_rank=2)
    assert m == 2
    assert y is True
    m, y = dl.place_m_and_hit(horse_count=7, finish_rank=3)
    assert m == 2
    assert y is False


def test_place_m_and_hit_normal_field():
    m, y = dl.place_m_and_hit(horse_count=8, finish_rank=3)
    assert m == 3
    assert y is True
    m, y = dl.place_m_and_hit(horse_count=8, finish_rank=4)
    assert m == 3
    assert y is False


# ---------------------------------------------------------------------------
# 6. p_pool 正規化
# ---------------------------------------------------------------------------

def test_p_pool_sums_to_m_wide():
    # 4頭レース、全ペアの確定オッズ既知(合成)
    odds_map = {(1, 2): 3.0, (1, 3): 5.0, (1, 4): 10.0, (2, 3): 4.0, (2, 4): 8.0, (3, 4): 6.0}
    m = 3  # wide
    p_pool_map, or_r = dl.compute_p_pool(odds_map, m)
    total = sum(p_pool_map.values())
    assert math.isclose(total, m, rel_tol=1e-9)
    # OR_r は compute_race_overround と同一定義（Σ1/O）
    expected_or = sum(1.0 / o for o in odds_map.values())
    assert math.isclose(or_r, expected_or, rel_tol=1e-9)


def test_p_pool_sums_to_m_quinella():
    odds_map = {(1, 2): 6.0, (1, 3): 12.0, (2, 3): 20.0}
    m = 1  # quinella
    p_pool_map, or_r = dl.compute_p_pool(odds_map, m)
    total = sum(p_pool_map.values())
    assert math.isclose(total, m, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 7. D_cal / D_adj / ROI_flat の数値検証
# ---------------------------------------------------------------------------

def test_d_cal_and_roi_flat_hand_computed():
    y = np.array([1, 0, 1, 0], dtype=float)
    p_theo = np.array([0.5, 0.3, 0.4, 0.2], dtype=float)
    o = np.array([2.0, 3.0, 2.5, 4.0], dtype=float)

    h = dl.seg_h(y)
    assert math.isclose(h, 0.5)

    p_bar_theo = dl.seg_p_bar_theo(p_theo)
    assert math.isclose(p_bar_theo, 0.35)

    d_cal = dl.seg_d_cal(h, p_bar_theo)
    assert math.isclose(d_cal, 0.15)

    roi_flat = dl.seg_roi_flat(y, o)
    # y*O = [2.0, 0, 2.5, 0] -> mean = 1.125
    assert math.isclose(roi_flat, 1.125)


def test_d_adj_pair_hand_computed():
    h = 0.6
    p_bar_pool = 0.5
    d_adj = dl.seg_d_adj_pair(h, p_bar_pool)
    assert math.isclose(d_adj, 0.1)


# ---------------------------------------------------------------------------
# 8. 複勝 D_adj の恒等式
# ---------------------------------------------------------------------------

def test_place_d_adj_identity_matches():
    y = np.array([1, 0, 1, 1, 0], dtype=float)
    o = np.array([2.5, np.nan, 3.0, 1.8, np.nan], dtype=float)
    t_place = 0.20

    h = dl.seg_h(y)
    o_bar_hit = dl.seg_o_bar_hit(y, o)
    roi_flat = dl.seg_roi_flat(y, o)

    d_adj_a = dl.seg_d_adj_place(h, o_bar_hit, t_place)
    d_adj_b = dl.seg_d_adj_place_from_roi(h, roi_flat, t_place)
    assert math.isclose(d_adj_a, d_adj_b, rel_tol=1e-9, abs_tol=1e-12)


def test_place_d_adj_efficient_pool_near_zero():
    # 効率的プール: O(u) = (1-t)/p_true から生成 (p_true は帯内一定と仮定)
    rng = np.random.default_rng(123)
    t = 0.20
    p_true = 0.5
    o_fixed = (1 - t) / p_true  # 的中時オッズ
    n = 5000
    y = (rng.random(n) < p_true).astype(float)
    o = np.where(y > 0, o_fixed, np.nan)

    h = dl.seg_h(y)
    o_bar_hit = dl.seg_o_bar_hit(y, o)
    d_adj = dl.seg_d_adj_place(h, o_bar_hit, t)
    assert abs(d_adj) < 0.02  # サンプル誤差の範囲でほぼ0


def test_place_d_adj_mispricing_injection_recovers_5pp():
    # +5pp ミスプライス注入: 実測的中率を理論(効率的プール)より5pp高くする
    rng = np.random.default_rng(7)
    t = 0.20
    p_priced = 0.45  # プールが値付けしている確率
    p_true = 0.50  # 実際の的中率（+5pp乖離）
    o_fixed = (1 - t) / p_priced  # プール価格ベースのオッズ
    n = 20000
    y = (rng.random(n) < p_true).astype(float)
    o = np.where(y > 0, o_fixed, np.nan)

    h = dl.seg_h(y)
    o_bar_hit = dl.seg_o_bar_hit(y, o)
    d_adj = dl.seg_d_adj_place(h, o_bar_hit, t)
    assert abs(d_adj - 0.05) < 0.01


# ---------------------------------------------------------------------------
# 9. クラスタ・ブートストラップ
# ---------------------------------------------------------------------------

def _make_synthetic_pair_race_data(n_races: int, seed: int, injected_gap: float, race_corr_sigma: float = 0.15):
    """race_id ごとに複数ユニット(疑似ペア)を持つ合成データ。乖離 injected_gap を注入。

    race_corr_sigma > 0 のとき、レースごとに共有の潜在効果（レース強度）を加えて
    レース内ユニット間に相関を持たせる（クラスタブートストラップの効果検証用）。
    """
    rng = np.random.default_rng(seed)
    p_pool_true = 0.30
    p_actual = p_pool_true + injected_gap
    race_ids = []
    ys = []
    p_pools = []
    for r in range(n_races):
        n_units = 10  # レース内相関を持たせるためのユニット数
        race_shift = rng.normal(0.0, race_corr_sigma) if race_corr_sigma > 0 else 0.0
        p_race = min(max(p_actual + race_shift, 0.01), 0.99)
        y = (rng.random(n_units) < p_race).astype(float)
        race_ids.extend([r] * n_units)
        ys.extend(y.tolist())
        p_pools.extend([p_pool_true] * n_units)
    return np.array(race_ids), np.array(ys), np.array(p_pools)


def test_bootstrap_deterministic_with_fixed_seed():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(200, seed=1, injected_gap=0.05, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))

    def _resample_d(sample_races: np.ndarray) -> float:
        idx = np.concatenate([np.where(race_ids == r)[0] for r in sample_races])
        return dl.seg_d_adj_pair(dl.seg_h(y[idx]), dl.seg_p_bar_theo(p_pool[idx]))

    p1 = dl.cluster_bootstrap_p_value(race_ids, _resample_d, d_hat=d_hat, B=200, seed=42)
    p2 = dl.cluster_bootstrap_p_value(race_ids, _resample_d, d_hat=d_hat, B=200, seed=42)
    assert p1 == p2


def test_bootstrap_positive_control_significant():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(500, seed=2, injected_gap=0.08, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))

    def _resample_d(sample_races: np.ndarray) -> float:
        idx = np.concatenate([np.where(race_ids == r)[0] for r in sample_races])
        return dl.seg_d_adj_pair(dl.seg_h(y[idx]), dl.seg_p_bar_theo(p_pool[idx]))

    p_val = dl.cluster_bootstrap_p_value(race_ids, _resample_d, d_hat=d_hat, B=2000, seed=42)
    assert p_val < 0.01


def test_bootstrap_negative_control_not_significant():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(500, seed=3, injected_gap=0.0, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))

    def _resample_d(sample_races: np.ndarray) -> float:
        idx = np.concatenate([np.where(race_ids == r)[0] for r in sample_races])
        return dl.seg_d_adj_pair(dl.seg_h(y[idx]), dl.seg_p_bar_theo(p_pool[idx]))

    p_val = dl.cluster_bootstrap_p_value(race_ids, _resample_d, d_hat=d_hat, B=2000, seed=42)
    assert p_val > 0.05


def test_bootstrap_cluster_wider_ci_than_unit_independent():
    # レース内相関を持つ合成ペアデータで、クラスタブートストラップの標準偏差が
    # ユニット独立リサンプリング（通常のブートストラップ）より大きいことを確認する。
    race_ids, y, p_pool = _make_synthetic_pair_race_data(300, seed=4, injected_gap=0.0)
    rng = np.random.default_rng(99)

    def _cluster_resample_d(sample_races: np.ndarray) -> float:
        idx = np.concatenate([np.where(race_ids == r)[0] for r in sample_races])
        return dl.seg_d_adj_pair(dl.seg_h(y[idx]), dl.seg_p_bar_theo(p_pool[idx]))

    unique_races = np.array(sorted(set(race_ids)))
    n_units = len(y)

    B = 500
    cluster_draws = np.empty(B)
    unit_draws = np.empty(B)
    for b in range(B):
        sample_races = rng.choice(unique_races, size=len(unique_races), replace=True)
        cluster_draws[b] = _cluster_resample_d(sample_races)

        unit_idx = rng.choice(n_units, size=n_units, replace=True)
        unit_draws[b] = dl.seg_d_adj_pair(dl.seg_h(y[unit_idx]), dl.seg_p_bar_theo(p_pool[unit_idx]))

    assert np.nanstd(cluster_draws) > np.nanstd(unit_draws)


# ---------------------------------------------------------------------------
# 9b. レース集計配列ベースの高速ブートストラップ（大規模セグメント用。
#     ユニット単位版と同値であることを確認）
# ---------------------------------------------------------------------------

def _race_level_arrays(race_ids: np.ndarray, y: np.ndarray, p_pool: np.ndarray):
    unique_races = np.array(sorted(set(race_ids)))
    n_units = np.array([int(np.sum(race_ids == r)) for r in unique_races], dtype=float)
    sum_y = np.array([float(np.sum(y[race_ids == r])) for r in unique_races], dtype=float)
    sum_p_pool = np.array([float(np.sum(p_pool[race_ids == r])) for r in unique_races], dtype=float)
    return {"n_units": n_units, "sum_y": sum_y, "sum_p_pool": sum_p_pool}


def test_fast_bootstrap_positive_control_significant():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(500, seed=2, injected_gap=0.08, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))
    race_arrays = _race_level_arrays(race_ids, y, p_pool)

    def _totals_to_d(totals: dict) -> float:
        if totals["n_units"] <= 0:
            return float("nan")
        h = totals["sum_y"] / totals["n_units"]
        p_bar_pool = totals["sum_p_pool"] / totals["n_units"]
        return dl.seg_d_adj_pair(h, p_bar_pool)

    p_val = dl.cluster_bootstrap_p_value_from_race_arrays(race_arrays, _totals_to_d, d_hat=d_hat, B=2000, seed=42)
    assert p_val < 0.01


def test_fast_bootstrap_negative_control_not_significant():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(500, seed=3, injected_gap=0.0, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))
    race_arrays = _race_level_arrays(race_ids, y, p_pool)

    def _totals_to_d(totals: dict) -> float:
        if totals["n_units"] <= 0:
            return float("nan")
        h = totals["sum_y"] / totals["n_units"]
        p_bar_pool = totals["sum_p_pool"] / totals["n_units"]
        return dl.seg_d_adj_pair(h, p_bar_pool)

    p_val = dl.cluster_bootstrap_p_value_from_race_arrays(race_arrays, _totals_to_d, d_hat=d_hat, B=2000, seed=42)
    assert p_val > 0.05


def test_fast_bootstrap_deterministic():
    race_ids, y, p_pool = _make_synthetic_pair_race_data(200, seed=1, injected_gap=0.05, race_corr_sigma=0.0)
    d_hat = dl.seg_d_adj_pair(dl.seg_h(y), dl.seg_p_bar_theo(p_pool))
    race_arrays = _race_level_arrays(race_ids, y, p_pool)

    def _totals_to_d(totals: dict) -> float:
        if totals["n_units"] <= 0:
            return float("nan")
        h = totals["sum_y"] / totals["n_units"]
        p_bar_pool = totals["sum_p_pool"] / totals["n_units"]
        return dl.seg_d_adj_pair(h, p_bar_pool)

    p1 = dl.cluster_bootstrap_p_value_from_race_arrays(race_arrays, _totals_to_d, d_hat=d_hat, B=200, seed=42)
    p2 = dl.cluster_bootstrap_p_value_from_race_arrays(race_arrays, _totals_to_d, d_hat=d_hat, B=200, seed=42)
    assert p1 == p2


# ---------------------------------------------------------------------------
# 10. 最小サンプル除外
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_units,sum_p_theo,expected",
    [
        (299, 40.0, False),
        (300, 29.9, False),
        (300, 30.0, True),
    ],
)
def test_min_sample_confirmed(n_units, sum_p_theo, expected):
    assert dl.min_sample_confirmed(n_units, sum_p_theo) is expected


# ---------------------------------------------------------------------------
# 11. Bonferroni閾値
# ---------------------------------------------------------------------------

def test_bonferroni_threshold_k30():
    assert math.isclose(dl.bonferroni_threshold(30), 0.01 / 30)


def test_bonferroni_threshold_k1():
    assert math.isclose(dl.bonferroni_threshold(1), 0.01)


def test_bonferroni_threshold_k0_none():
    assert dl.bonferroni_threshold(0) is None


# ---------------------------------------------------------------------------
# 12. 打ち切り verdict
# ---------------------------------------------------------------------------

def test_cutoff_verdict_all_below():
    d_adj_values = [0.01, -0.02, 0.029, float("nan"), 0.0]
    verdict = dl.determine_cutoff_verdict(d_adj_values, threshold=0.03)
    assert verdict == "cross_pool_divergence_within_takeout_wall"


def test_cutoff_verdict_one_pass_no_verdict():
    d_adj_values = [0.01, -0.02, 0.031, float("nan")]
    verdict = dl.determine_cutoff_verdict(d_adj_values, threshold=0.03)
    assert verdict is None


# ---------------------------------------------------------------------------
# 13. payout 集中度ゲート
# ---------------------------------------------------------------------------

def test_payout_gate_top1_share_violation():
    payouts = [310.0] + [69.0] * 9  # top1 share = 310/(310+621)=~0.333 > 0.30
    result = dl.payout_concentration_gate(payouts, n_hits=10)
    assert result["diagnosis_valid"] is False


def test_payout_gate_n_hits_violation():
    payouts = [10.0] * 9  # n_hits=9 < 10
    result = dl.payout_concentration_gate(payouts, n_hits=9)
    assert result["diagnosis_valid"] is False


def test_payout_gate_pass():
    payouts = [10.0] * 20
    result = dl.payout_concentration_gate(payouts, n_hits=20)
    assert result["diagnosis_valid"] is True


# ---------------------------------------------------------------------------
# 15. 符号一貫チェック
# ---------------------------------------------------------------------------

def test_sign_consistency_required_for_primary_pass():
    bonferroni_thr = 0.01 / 30
    # 2023年は正、2024年は負 -> 一次判定は不通過
    passed = dl.primary_pass(
        d_adj=0.05,
        bootstrap_p=0.0001,
        d_adj_year1=0.04,
        d_adj_year2=-0.01,
        bonferroni_thr=bonferroni_thr,
    )
    assert passed is False


def test_sign_consistency_both_positive_can_pass():
    bonferroni_thr = 0.01 / 30
    passed = dl.primary_pass(
        d_adj=0.05,
        bootstrap_p=0.0001,
        d_adj_year1=0.04,
        d_adj_year2=0.03,
        bonferroni_thr=bonferroni_thr,
    )
    assert passed is True


# ---------------------------------------------------------------------------
# 16. 再現性（全体）
# ---------------------------------------------------------------------------

def test_full_pipeline_reproducible():
    odds = [2.0, 3.5, 3.5, 10.0, 6.0]
    horse_nums = [1, 2, 3, 4, 5]
    r1 = dl.assign_popularity_rank(odds, horse_nums)
    r2 = dl.assign_popularity_rank(odds, horse_nums)
    assert list(r1) == list(r2)
    bands1 = [dl.assign_pop_band_single(r) for r in r1]
    bands2 = [dl.assign_pop_band_single(r) for r in r2]
    assert bands1 == bands2
