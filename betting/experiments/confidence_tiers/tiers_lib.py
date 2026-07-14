"""confidence_tiers: 純関数ライブラリ。

margin 計算・階層割当・四分位境界決定・Δ（対1番人気ROI差）・レース単位ペアド
クラスタブートストラップ・順序コントラスト（T4-T1）・最小サンプル判定・
Bonferroni閾値・危険信号フラグを実装する。

市場列（odds）は Δ計算・ベースライン関連の関数（`compute_roi`,
`per_race_payout_diff`, `cluster_bootstrap_delta_p_value`,
`cluster_bootstrap_ordering_contrast`）の引数（stake/payout 配列。既に決済済みの
金額であり odds そのものではない）としてのみ現れる。margin 計算・階層割当・境界
決定の各関数は市場列に一切触れない（tests/test_static_guards.py で機械的に担保）。

仕様書: docs/specs/2026-07-11-confidence-tiers-spec.md
参考実装パターン: betting/experiments/cross_pool_divergence/divergence_lib.py
（クラスタブートストラップ・Bonferroni閾値・payout集中度ゲート）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# payout_concentration_gate は cross_pool_divergence の実装をそのまま import 再利用する
# （§10 隔離宣言: betting/src/ への書き込みはしないが、他隔離実験の純関数 import は可）。
_CPD_DIR = Path(__file__).resolve().parents[1] / "cross_pool_divergence"
if str(_CPD_DIR) not in sys.path:
    sys.path.insert(0, str(_CPD_DIR))
from divergence_lib import payout_concentration_gate  # noqa: E402,F401

K_HYP = 5
BONFERRONI_ALPHA = 0.01
MIN_SAMPLE_N = 200

_EPS = 1e-12


# ---------------------------------------------------------------------------
# margin 計算（市場列不使用）
# ---------------------------------------------------------------------------


def compute_race_margin(scores: Sequence[float], horse_nums: Sequence[int]) -> float:
    """レース内 margin = 1位スコア - 2位スコア。

    1位・2位の特定は betting/src/flat_top1.py::select_top1_bets と同一の決定的
    タイブレーク（スコア降順 → 同値は馬番昇順）に従う。1頭のみのレースは margin=0
    （2位が存在しないため。実データでは horse_count>=5 フィルタにより発生しない）。
    全馬同値レースは margin=0。
    """
    s = np.asarray(scores, dtype=float)
    h = np.asarray(horse_nums, dtype=float)
    if len(s) != len(h):
        raise ValueError("scores and horse_nums must have the same length")
    if len(s) == 0:
        return float("nan")
    if len(s) == 1:
        return 0.0
    # np.lexsort は最後のキーが第一優先。-s 昇順（=s 降順）を主、h 昇順を従（タイブレーク）。
    order = np.lexsort((h, -s))
    sorted_scores = s[order]
    return float(sorted_scores[0] - sorted_scores[1])


# ---------------------------------------------------------------------------
# 階層割当・四分位境界（市場列不使用）
# ---------------------------------------------------------------------------


def assign_tier(margin: float, boundaries: Sequence[float]) -> int:
    """境界 [b1,b2,b3] に対し margin を階層 1..4 に割り当てる。

    境界値ちょうどは下位階層に入る（仕様書§4.1）: T1: margin<=b1, T2: b1<margin<=b2,
    T3: b2<margin<=b3, T4: margin>b3。これは
    ``1 + np.searchsorted(boundaries, margin, side="left")`` で得られる
    （side="left" は「boundaries 中 margin 未満の個数」を返すため、境界値と一致する
    要素はカウントされず下位階層側に残る）。
    """
    b = np.asarray(boundaries, dtype=float)
    return int(1 + np.searchsorted(b, margin, side="left"))


def assign_tier_batch(margins: Sequence[float], boundaries: Sequence[float]) -> np.ndarray:
    b = np.asarray(boundaries, dtype=float)
    m = np.asarray(margins, dtype=float)
    return (1 + np.searchsorted(b, m, side="left")).astype(int)


def compute_quartile_boundaries(margins: Sequence[float]) -> tuple[float, float, float]:
    """境界 [b1,b2,b3] = margin の 25/50/75 パーセンタイル（numpy.quantile, method="linear"）。

    outcome-blind 構造担保: このシグネチャは margin 配列のみを受け取り、着順・払戻
    列を一切受け取らない（tests/test_tiers_lib.py 項目3で検査）。
    """
    m = np.asarray(margins, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return (float("nan"), float("nan"), float("nan"))
    q = np.quantile(m, [0.25, 0.5, 0.75], method="linear")
    return (float(q[0]), float(q[1]), float(q[2]))


# ---------------------------------------------------------------------------
# Δ（対1番人気ROI差）・ペア比較
# ---------------------------------------------------------------------------


def compute_roi(stakes: Sequence[float], payouts: Sequence[float]) -> float:
    """ROI = Σpayout / Σstake（比率。1.0=100%）。"""
    s = np.asarray(stakes, dtype=float)
    p = np.asarray(payouts, dtype=float)
    total_stake = float(s.sum())
    if total_stake <= 0:
        return float("nan")
    return float(p.sum() / total_stake)


def compute_delta(roi_model: float, roi_fav: float) -> float:
    """Δ(t) = ROI_model(t) - ROI_fav(t)。"""
    return float(roi_model - roi_fav)


def per_race_payout_diff(payout_model: Sequence[float], payout_fav: Sequence[float]) -> np.ndarray:
    """レース単位の payout 差診断量（payout_model - payout_fav）。

    モデル1位馬=1番人気のレースでは payout_model==payout_fav となり、当該レースの
    差分は 0 になる（仕様書§5.2「Δへの寄与は0」の直接的な確認用）。
    """
    pm = np.asarray(payout_model, dtype=float)
    pf = np.asarray(payout_fav, dtype=float)
    return pm - pf


# ---------------------------------------------------------------------------
# クラスタ・ブートストラップ（レース単位、ペアド）
# ---------------------------------------------------------------------------


def cluster_bootstrap_delta_p_value(
    stakes_model: Sequence[float],
    payouts_model: Sequence[float],
    stakes_fav: Sequence[float],
    payouts_fav: Sequence[float],
    *,
    B: int = 10000,
    seed: int = 42,
) -> dict:
    """H1〜H4: Δ(t)>0 の片側レース単位クラスタ・ブートストラップ検定。

    4配列は同じ順序・同じ長さ（=対象レース数）で、各要素が同一レースのモデル/
    1番人気ペアに対応する。復元抽出はレースインデックスに対して行い、モデル側・
    1番人気側を同一インデックスでリサンプルすることでペア関係を保つ。
    片側 p値 = P(Δ* <= 0)。95%CI（percentile法）も併記する。

    パターン出典: betting/experiments/cross_pool_divergence/divergence_lib.py
    ::cluster_bootstrap_p_value（両側percentile法を片側化）。
    """
    sm = np.asarray(stakes_model, dtype=float)
    pm = np.asarray(payouts_model, dtype=float)
    sf = np.asarray(stakes_fav, dtype=float)
    pf = np.asarray(payouts_fav, dtype=float)
    n = len(sm)
    if n == 0 or not (len(pm) == n and len(sf) == n and len(pf) == n):
        return {
            "delta_hat": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_races": n,
        }

    roi_m_hat = compute_roi(sm, pm)
    roi_f_hat = compute_roi(sf, pf)
    delta_hat = compute_delta(roi_m_hat, roi_f_hat)

    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stake_m_sum = sm[idx].sum()
        stake_f_sum = sf[idx].sum()
        roi_m = pm[idx].sum() / stake_m_sum if stake_m_sum > 0 else float("nan")
        roi_f = pf[idx].sum() / stake_f_sum if stake_f_sum > 0 else float("nan")
        boot[b] = roi_m - roi_f

    finite = boot[np.isfinite(boot)]
    if len(finite) == 0:
        return {
            "delta_hat": delta_hat,
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_races": n,
        }
    p_value = float(np.mean(finite <= 0.0))
    ci_low, ci_high = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
    return {
        "delta_hat": delta_hat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_races": n,
    }


def _bootstrap_roi_diff_samples(
    stakes_model: np.ndarray,
    payouts_model: np.ndarray,
    stakes_fav: np.ndarray,
    payouts_fav: np.ndarray,
    *,
    rng: np.random.Generator,
    B: int,
) -> np.ndarray:
    """1階層分の Δ* リサンプル配列（内部ヘルパー。層別リサンプルの1層分に相当）。"""
    n = len(stakes_model)
    out = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stake_m_sum = stakes_model[idx].sum()
        stake_f_sum = stakes_fav[idx].sum()
        roi_m = payouts_model[idx].sum() / stake_m_sum if stake_m_sum > 0 else float("nan")
        roi_f = payouts_fav[idx].sum() / stake_f_sum if stake_f_sum > 0 else float("nan")
        out[b] = roi_m - roi_f
    return out


def cluster_bootstrap_ordering_contrast(
    t1_stakes_model: Sequence[float],
    t1_payouts_model: Sequence[float],
    t1_stakes_fav: Sequence[float],
    t1_payouts_fav: Sequence[float],
    t4_stakes_model: Sequence[float],
    t4_payouts_model: Sequence[float],
    t4_stakes_fav: Sequence[float],
    t4_payouts_fav: Sequence[float],
    *,
    B: int = 10000,
    seed: int = 42,
) -> dict:
    """H_ord: C = Δ(T4) - Δ(T1) > 0 の片側検定。

    T1・T4 それぞれのレース集合を独立に（層別に）復元抽出し、C* = Δ*(T4) - Δ*(T1)
    を B 回計算する。片側 p値 = P(C* <= 0)。95%CI（percentile法）も併記する。
    """
    t1_sm = np.asarray(t1_stakes_model, dtype=float)
    t1_pm = np.asarray(t1_payouts_model, dtype=float)
    t1_sf = np.asarray(t1_stakes_fav, dtype=float)
    t1_pf = np.asarray(t1_payouts_fav, dtype=float)
    t4_sm = np.asarray(t4_stakes_model, dtype=float)
    t4_pm = np.asarray(t4_payouts_model, dtype=float)
    t4_sf = np.asarray(t4_stakes_fav, dtype=float)
    t4_pf = np.asarray(t4_payouts_fav, dtype=float)

    if len(t1_sm) == 0 or len(t4_sm) == 0:
        return {"c_hat": float("nan"), "p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    delta_t1_hat = compute_delta(compute_roi(t1_sm, t1_pm), compute_roi(t1_sf, t1_pf))
    delta_t4_hat = compute_delta(compute_roi(t4_sm, t4_pm), compute_roi(t4_sf, t4_pf))
    c_hat = float(delta_t4_hat - delta_t1_hat)

    rng = np.random.default_rng(seed)
    boot_t1 = _bootstrap_roi_diff_samples(t1_sm, t1_pm, t1_sf, t1_pf, rng=rng, B=B)
    boot_t4 = _bootstrap_roi_diff_samples(t4_sm, t4_pm, t4_sf, t4_pf, rng=rng, B=B)
    boot_c = boot_t4 - boot_t1

    finite = boot_c[np.isfinite(boot_c)]
    if len(finite) == 0:
        return {"c_hat": c_hat, "p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    p_value = float(np.mean(finite <= 0.0))
    ci_low, ci_high = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
    return {"c_hat": c_hat, "p_value": p_value, "ci_low": ci_low, "ci_high": ci_high}


def monotonicity_flag(deltas: Sequence[float]) -> bool:
    """Δ(T1)<=Δ(T2)<=Δ(T3)<=Δ(T4) が全て成立するかの記述的フラグ（判定には不使用）。"""
    d = [float(x) for x in deltas]
    return all(d[i] <= d[i + 1] for i in range(len(d) - 1))


# ---------------------------------------------------------------------------
# 最小サンプル・Bonferroni
# ---------------------------------------------------------------------------


def min_sample_ok(n: int, *, n_min: int = MIN_SAMPLE_N) -> bool:
    return int(n) >= int(n_min)


def all_tiers_min_sample_ok(n_by_tier: dict, *, n_min: int = MIN_SAMPLE_N) -> bool:
    """全階層が n>=n_min を満たすか。1つでも満たさなければ H_ord も実行しない
    （仕様書§4.3: 4階層すべてのΔが必要なため）。"""
    return all(min_sample_ok(n, n_min=n_min) for n in n_by_tier.values())


def bonferroni_threshold(k: int = K_HYP, *, alpha: float = BONFERRONI_ALPHA) -> float:
    """Bonferroni閾値 = alpha/k。判定保留階層があっても k は常に固定（仕様書§6.3）。"""
    if k <= 0:
        raise ValueError("k must be positive")
    return float(alpha) / float(k)


# ---------------------------------------------------------------------------
# 危険信号フラグ（仕様書§9）
# ---------------------------------------------------------------------------


def leak_review_flag(top1_rate: float, *, threshold: float = 0.40) -> bool:
    """階層別Top-1的中率が閾値超 → 即時停止・evaluator報告(機械的不合格ではない)。"""
    return bool(np.isfinite(top1_rate) and top1_rate > threshold)


def danger_roi_gt_100(roi_ratio: float, *, threshold: float = 1.0) -> bool:
    """ROI(比率。1.0=100%)が閾値超 → 「黒字」と記述せずデータ結合バグを疑う。"""
    return bool(np.isfinite(roi_ratio) and roi_ratio > threshold)


def large_delta_flag(delta: float, *, threshold_abs: float = 0.20) -> bool:
    """|Δ(t)| が閾値超(全体Δ+4〜5ppから大きく外れる) → バグ優先で検証。"""
    return bool(np.isfinite(delta) and abs(delta) > threshold_abs)
