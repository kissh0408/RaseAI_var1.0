"""cross_pool_divergence: 純関数ライブラリ。

L1（L1ソース実装 / features 系 / scores 系の各parquet資産）を一切参照しない。入力は単勝オッズ由来の
市場確率 q と、券種別の確定オッズ・払戻のみ。人気順位付与・帯割当・p_pool 正規化・
D_cal/D_adj・クラスタブートストラップ・除外判定・Bonferroni閾値・打ち切り判定・
payout集中度ゲートをここに実装する。

仕様書: docs/specs/2026-07-10-p4-cross-pool-divergence-spec.md
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 4.1 人気順位付与・帯割当
# ---------------------------------------------------------------------------

def assign_popularity_rank(odds: Sequence[float], horse_nums: Sequence[int]) -> np.ndarray:
    """レース内人気順位（1=最人気=オッズ最小）。同オッズは馬番昇順で決定的にタイブレーク。

    戻り値は horse_nums / odds と同じ並び順の 1..n の順位配列（重複なし）。
    """
    odds_arr = np.asarray(odds, dtype=float)
    horse_arr = np.asarray(horse_nums, dtype=int)
    if len(odds_arr) != len(horse_arr):
        raise ValueError("odds and horse_nums must have the same length")
    # lexsort: 最後のキーが第一優先。odds 昇順を主、horse_num 昇順を従（タイブレーク）にする。
    order = np.lexsort((horse_arr, odds_arr))
    ranks = np.empty(len(odds_arr), dtype=int)
    ranks[order] = np.arange(1, len(odds_arr) + 1)
    return ranks


def assign_pop_band_single(rank: int) -> str:
    """複勝用の単体人気帯（POP1〜POP4）。"""
    r = int(rank)
    if r == 1:
        return "POP1"
    if 2 <= r <= 3:
        return "POP2"
    if 4 <= r <= 6:
        return "POP3"
    return "POP4"


def assign_pair_band(rank_i: int, rank_j: int, *, top_rank_max: int = 3) -> str:
    """ワイド・馬連用のペア人気帯（PAIR_TOP / PAIR_MIX / PAIR_LONG）。"""
    ri, rj = int(rank_i), int(rank_j)
    top_i = ri <= top_rank_max
    top_j = rj <= top_rank_max
    if top_i and top_j:
        return "PAIR_TOP"
    if top_i or top_j:
        return "PAIR_MIX"
    return "PAIR_LONG"


def assign_fs_band(horse_count: int) -> Optional[str]:
    """頭数帯（FS_S/FS_M/FS_L）。18頭超は None（対象外）。"""
    hc = int(horse_count)
    if hc <= 8:
        return "FS_S"
    if hc <= 13:
        return "FS_M"
    if hc <= 18:
        return "FS_L"
    return None


# ---------------------------------------------------------------------------
# 4.4 複勝の m と的中定義
# ---------------------------------------------------------------------------

def place_m_and_hit(horse_count: int, finish_rank: int, *, small_field_max: int = 7) -> tuple[int, bool]:
    """複勝の勝ちユニット数 m と的中フラグ y。

    horse_count >= 8 は m=3 (finish_rank<=3)、5<=horse_count<=7 は m=2 (finish_rank<=2)。
    """
    hc = int(horse_count)
    m = 2 if hc <= small_field_max else 3
    y = bool(int(finish_rank) <= m)
    return m, y


# ---------------------------------------------------------------------------
# 5.3 p_pool 正規化（ワイド・馬連。全ユニットの O が既知）
# ---------------------------------------------------------------------------

def compute_race_overround_from_odds(odds_values: Sequence[float]) -> float:
    """レース内全ペアの Σ(1/O)。odds_loader/wide_ev_core の compute_race_overround と同一定義。"""
    total = 0.0
    for o in odds_values:
        o = float(o)
        if o > 1.0:
            total += 1.0 / o
    return total if total > 0 else float("nan")


def compute_p_pool(odds_map: dict, m: int) -> tuple[dict, float]:
    """レース内ペア odds_map({key: odds}) から p_pool を計算し (p_pool_map, OR_r) を返す。

    Σ_u p_pool(u) = m となるようレース内で正規化する。
    """
    or_r = compute_race_overround_from_odds(list(odds_map.values()))
    p_pool_map: dict = {}
    if not np.isfinite(or_r) or or_r <= 0:
        for k in odds_map:
            p_pool_map[k] = float("nan")
        return p_pool_map, or_r
    for k, o in odds_map.items():
        o = float(o)
        if o > 1.0:
            p_pool_map[k] = float(m) * (1.0 / o) / or_r
        else:
            p_pool_map[k] = float("nan")
    return p_pool_map, or_r


def effective_takeout_from_or(or_r: float, m: int) -> float:
    """実効控除率 t_hat_r = 1 - m/OR_r。"""
    if not np.isfinite(or_r) or or_r <= 0:
        return float("nan")
    return 1.0 - float(m) / float(or_r)


# ---------------------------------------------------------------------------
# 5.2 基本量 / 5.3 D_adj
# ---------------------------------------------------------------------------

def seg_h(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(y))


def seg_p_bar_theo(p_theo: np.ndarray) -> float:
    p_theo = np.asarray(p_theo, dtype=float)
    if len(p_theo) == 0:
        return float("nan")
    return float(np.mean(p_theo))


def seg_d_cal(h: float, p_bar_theo: float) -> float:
    return float(h - p_bar_theo)


def seg_roi_flat(y: np.ndarray, o: np.ndarray) -> float:
    """ROI_flat(s) = mean(y*O)。O は未的中ユニットで NaN でも 0 として扱ってよい
    （y=0 の項は y*O=0 に潰れるため）。"""
    y = np.asarray(y, dtype=float)
    o = np.asarray(o, dtype=float)
    if len(y) == 0:
        return float("nan")
    contrib = np.where(y > 0, o, 0.0)
    contrib = np.nan_to_num(contrib, nan=0.0)
    return float(np.mean(contrib))  # y*O, 非的中は0


def seg_d_adj_pair(h: float, p_bar_pool: float) -> float:
    """ワイド・馬連の D_adj = h - p_bar_pool。"""
    return float(h - p_bar_pool)


def seg_o_bar_hit(y: np.ndarray, o: np.ndarray) -> float:
    """的中ユニットの平均払戻倍率 Ō_hit。的中ゼロなら NaN。"""
    y = np.asarray(y, dtype=float)
    o = np.asarray(o, dtype=float)
    mask = y > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(o[mask]))


def seg_d_adj_place(h: float, o_bar_hit: float, t_place: float) -> float:
    """複勝の D_adj = h - (1-t)/Ō_hit。Σy=0（Ō_hit=NaN）は NaN。"""
    if o_bar_hit is None or not np.isfinite(o_bar_hit) or o_bar_hit <= 0:
        return float("nan")
    return float(h - (1.0 - t_place) / o_bar_hit)


def seg_d_adj_place_from_roi(h: float, roi_flat: float, t_place: float) -> float:
    """複勝 D_adj の同値形 h*(1 - (1-t)/ROI_flat)。ROI_flat=0（Σy=0）は NaN。"""
    if roi_flat is None or not np.isfinite(roi_flat) or roi_flat <= 0:
        return float("nan")
    return float(h * (1.0 - (1.0 - t_place) / roi_flat))


# ---------------------------------------------------------------------------
# 5.5 クラスタ・ブートストラップ
# ---------------------------------------------------------------------------

def cluster_bootstrap_p_value(
    race_ids: Sequence,
    d_adj_fn,
    *,
    d_hat: float,
    B: int = 10000,
    seed: int = 42,
) -> float:
    """レース単位クラスタ・ブートストラップの両側 p 値（percentile 法）。

    race_ids: 各ユニットの所属レースID配列（長さ n_units）。
    d_adj_fn: race_id の配列（復元抽出されたレースIDの並び。重複あり）を受け取り、
        そのリサンプルに対応するユニット集合で D_adj を再計算して返す callable。
        呼び出し側でレース -> ユニットindex群のマップを閉じ込めておくこと。
    d_hat: 観測 D_adj（元データでの点推定値）。
    """
    rng = np.random.default_rng(seed)
    unique_races = np.asarray(sorted(set(race_ids)))
    n_races = len(unique_races)
    if n_races == 0:
        return float("nan")
    boot_d = np.empty(B, dtype=float)
    for b in range(B):
        sample_races = rng.choice(unique_races, size=n_races, replace=True)
        boot_d[b] = d_adj_fn(sample_races)
    return _percentile_two_sided_p(boot_d, d_hat)


def _percentile_two_sided_p(boot_d: np.ndarray, d_hat: float) -> float:
    """percentile 法の両側 p 値: p = 2*min(P(D*<=0), P(D*>=2*D_hat))。"""
    boot_d = boot_d[np.isfinite(boot_d)]
    if len(boot_d) == 0:
        return float("nan")
    p_le0 = float(np.mean(boot_d <= 0))
    p_ge2d = float(np.mean(boot_d >= 2 * d_hat))
    p_value = 2.0 * min(p_le0, p_ge2d)
    return float(min(p_value, 1.0))


def cluster_bootstrap_p_value_from_race_arrays(
    race_arrays: dict,
    d_adj_from_totals_fn,
    *,
    d_hat: float,
    B: int = 10000,
    seed: int = 42,
) -> float:
    """レース単位クラスタ・ブートストラップの高速版（レース集計配列ベース）。

    race_arrays: {stat_name: np.ndarray}（各配列は「ユニークレースの固定順」で長さ n_races）。
        例: {"n_units": ..., "sum_y": ..., "sum_p_pool": ...}。
    d_adj_from_totals_fn: {stat_name: 合計値} を受け取り D_adj を返す callable。
    ブートストラップ復元抽出をレース単位で行い、各リサンプルの集計配列の合計から
    D_adj を再計算する。ユニット単位でインデックスを毎回集約する方式と数学的に同値
    （D_adj はレース内ユニットの和/平均から構成されるため）だが、大規模セグメント
    （数十万ユニット）でも O(n_races) で計算できる。
    """
    if not race_arrays:
        return float("nan")
    n_races = len(next(iter(race_arrays.values())))
    if n_races == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    boot_d = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n_races, size=n_races)
        totals = {k: float(np.sum(v[idx])) for k, v in race_arrays.items()}
        boot_d[b] = d_adj_from_totals_fn(totals)
    return _percentile_two_sided_p(boot_d, d_hat)


# ---------------------------------------------------------------------------
# 最小サンプル除外・Bonferroni閾値
# ---------------------------------------------------------------------------

def min_sample_confirmed(n_units: int, sum_p_theo: float, *, n_min: int = 300, p_theo_min: float = 30.0) -> bool:
    return int(n_units) >= n_min and float(sum_p_theo) >= p_theo_min


def bonferroni_threshold(k: int, *, alpha: float = 0.01) -> Optional[float]:
    if k <= 0:
        return None
    return float(alpha) / float(k)


# ---------------------------------------------------------------------------
# 一次判定・打ち切り verdict
# ---------------------------------------------------------------------------

def primary_pass(
    d_adj: float,
    bootstrap_p: float,
    d_adj_year1: float,
    d_adj_year2: float,
    *,
    bonferroni_thr: Optional[float],
    d_adj_threshold: float = 0.03,
) -> bool:
    if bonferroni_thr is None:
        return False
    if not np.isfinite(d_adj) or not np.isfinite(bootstrap_p):
        return False
    if not (np.isfinite(d_adj_year1) and np.isfinite(d_adj_year2)):
        return False
    cond_magnitude = d_adj >= d_adj_threshold
    cond_p = bootstrap_p < bonferroni_thr
    cond_sign = (d_adj_year1 > 0) and (d_adj_year2 > 0)
    return bool(cond_magnitude and cond_p and cond_sign)


def determine_cutoff_verdict(d_adj_values: Sequence[float], *, threshold: float = 0.03) -> Optional[str]:
    """全確定セグメントの D_adj が閾値未満なら打ち切り verdict を返す。

    NaN（一次不通過セグメント）は「閾値未満」として扱う（正方向の乖離を主張できないため）。
    1つでも有限値かつ >= threshold の D_adj があれば None（延長しない）。
    """
    vals = [v for v in d_adj_values]
    any_pass = any(np.isfinite(v) and v >= threshold for v in vals)
    if any_pass:
        return None
    return "cross_pool_divergence_within_takeout_wall"


# ---------------------------------------------------------------------------
# payout 集中度ゲート（§7）
# ---------------------------------------------------------------------------

def payout_concentration_gate(
    payouts: Sequence[float],
    n_hits: int,
    *,
    top1_share_max: float = 0.30,
    n_hits_min: int = 10,
) -> dict:
    payouts_arr = np.asarray(payouts, dtype=float)
    total = float(payouts_arr.sum()) if len(payouts_arr) else 0.0
    top1_share = float(payouts_arr.max() / total) if total > 0 else 1.0
    concentrated_ok = top1_share <= top1_share_max
    hits_ok = int(n_hits) >= n_hits_min
    return {
        "top1_payout_share": top1_share,
        "payout_not_concentrated_top1_lte_30pct": concentrated_ok,
        "n_hits_gte_10": hits_ok,
        "diagnosis_valid": bool(concentrated_ok and hits_ok),
    }
