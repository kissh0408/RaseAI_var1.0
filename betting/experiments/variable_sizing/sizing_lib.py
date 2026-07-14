"""variable_sizing: 純関数ライブラリ。

倍率適用・保存則検証・base_stake検証（100円丸め厳密化）・可変stake付与・月次MDD/
最繁忙日エクスポージャ（derive_flat_fraction と同一セマンティクス）・f_var機械導出・
危険信号フラグを実装する。

市場情報境界（仕様書§1・§3.4）: 倍率決定・stake計算・占有率・保存則の各関数
（CORE_PURE_FUNCTIONS）は margin/tier・倍率配列・bankroll・f_var のみを入力とし、
オッズ・払戻・着順・人気・確率のいかなる列も引数に取らない
（tests/test_static_guards.py で機械的に担保）。月次MDD・最繁忙日エクスポージャは
`betting/src/derive_flat_fraction.py` の同名（アンダースコア接頭）関数を import
再利用する（コピー禁止。仕様書§2）。階層割当・危険信号フラグは
`betting/experiments/confidence_tiers/tiers_lib.py` を import 再利用する。

仕様書: docs/specs/2026-07-11-variable-sizing-spec.md
先例実装: betting/experiments/confidence_tiers/tiers_lib.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[2]
_CONFIDENCE_TIERS_DIR = _ROOT / "betting" / "experiments" / "confidence_tiers"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CONFIDENCE_TIERS_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIDENCE_TIERS_DIR))

# import 再利用（コピー禁止。仕様書§2・§10-3）
import tiers_lib as _tiers_lib  # noqa: E402

assign_tier = _tiers_lib.assign_tier
assign_tier_batch = _tiers_lib.assign_tier_batch
compute_race_margin = _tiers_lib.compute_race_margin
leak_review_flag = _tiers_lib.leak_review_flag
danger_roi_gt_100 = _tiers_lib.danger_roi_gt_100

from betting.src.derive_flat_fraction import (  # noqa: E402
    _busiest_day_exposure as busiest_day_exposure,
)
from betting.src.derive_flat_fraction import (  # noqa: E402
    _monthly_max_drawdown as monthly_max_drawdown,
)

# ---------------------------------------------------------------------------
# DISCLAIMER・caveats（仕様書§4.4。全結果JSONの disclaimer/caveats キーに埋め込む）
# ---------------------------------------------------------------------------

DISCLAIMER: str = (
    "本可変サイジングはユーザーの設計選好（自信度に応じたリスク配分）の実装であり、"
    "ROI改善を目的・根拠としない。先行検証（docs/specs/2026-07-11-confidence-tiers-spec.md "
    "§15、verdict=confidence_does_not_predict_market_edge）で自信度は対1番人気ROI優位を"
    "予測しないことが確定しており、自信度加重はモデルの優位性の源泉（過剰人気馬回避）を"
    "薄めるリスクが記述的に示唆されている。本推奨は市場に対する相対的な損失最小化の枠内の"
    "配分変更であり、黒字化を保証するものではない（fold2 OOS実測: ROI 81.89%、"
    "元本の約18%の期待損失）"
)

CAVEAT_CONFIDENCE_DOES_NOT_PREDICT_EDGE: str = (
    "先行検証（confidence-tiers §15、verdict=confidence_does_not_predict_market_edge）で "
    "margin（自信度）は対1番人気ROI優位を予測しないことが統計的に確定している"
    "（H1〜H4・順序仮説すべて p>=0.24、Bonferroni閾値0.002を大きく上回る）。"
)

CAVEAT_DILUTION_RISK: str = (
    "自信度が高い階層ほどモデル予測1位が1番人気と一致する割合が上昇する"
    "（T1:37.4%→T4:74.2%、平均オッズも4.64倍→3.26倍と単調低下）。自信度で重み付けすると、"
    "モデルの数少ない優位性の源泉（過剰人気馬回避）を薄めるリスクがある。"
)

CAVEATS: tuple[str, ...] = (
    CAVEAT_CONFIDENCE_DOES_NOT_PREDICT_EDGE,
    CAVEAT_DILUTION_RISK,
)

ROI_NOTE_TEMPLATE: str = (
    "可変系列とflat系列のROI差は記述的な測定値であり、符号や大きさに関わらず判定・"
    "倍率再調整の材料には用いない（仕様書§0.5・§6.2）。"
)


def build_result_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """結果dictに disclaimer/caveats キーを機械的に付与する（仕様書§4.4・§9-10）。

    呼び出し側の payload に既に "caveats" があれば末尾に連結する（重複時も許容。
    JSON生成関数の出力が disclaimer/caveats を必ず含むことをこの一箇所で担保する）。
    """
    out = dict(payload)
    out["disclaimer"] = DISCLAIMER
    extra = list(payload.get("caveats", []))
    out["caveats"] = list(CAVEATS) + extra
    return out


# ---------------------------------------------------------------------------
# 倍率（有界性・単調性の検証。仕様書§3.1・§9-1）
# ---------------------------------------------------------------------------

MULTIPLIER_MIN = 0.5
MULTIPLIER_MAX = 1.5


def validate_multipliers(multipliers: Sequence[float]) -> None:
    """倍率配列が有界性(0.5<=m<=1.5)・狭義単調増加であることを検証する。"""
    m = [float(x) for x in multipliers]
    if len(m) != 4:
        raise ValueError(f"multipliers must have exactly 4 tiers, got {len(m)}")
    if m[0] < MULTIPLIER_MIN or m[-1] > MULTIPLIER_MAX:
        raise ValueError(
            f"multipliers must be bounded within [{MULTIPLIER_MIN},{MULTIPLIER_MAX}], got {m}"
        )
    if not all(m[i] < m[i + 1] for i in range(len(m) - 1)):
        raise ValueError(f"multipliers must be strictly increasing (monotonic), got {m}")


def multiplier_for_tier(tier: int, multipliers: Sequence[float]) -> float:
    """階層1..4に対応する倍率を返す（tierのみを入力とする市場情報遮断シグネチャ）。"""
    t = int(tier)
    if t < 1 or t > len(multipliers):
        raise ValueError(f"tier must be in 1..{len(multipliers)}, got {t}")
    return float(multipliers[t - 1])


# ---------------------------------------------------------------------------
# リスク予算保存則（占有率・占有率加重平均倍率。outcome-blind。仕様書§3.2・§9-3）
# ---------------------------------------------------------------------------


def compute_tier_occupancy(tiers: Sequence[int], *, k: int = 4) -> dict[int, float]:
    """階層占有率 w_t = n_t / Σn_t。tiers（階層番号の配列）のみを入力とし、着順・払戻
    列を一切受け取らない（outcome-blind構造担保。仕様書§3.2-1）。
    """
    arr = np.asarray(list(tiers), dtype=int)
    total = int(len(arr))
    occupancy: dict[int, float] = {}
    for t in range(1, k + 1):
        n_t = int(np.sum(arr == t))
        occupancy[t] = float(n_t) / float(total) if total > 0 else float("nan")
    return occupancy


def weighted_mean_multiplier(occupancy: dict[int, float], multipliers: Sequence[float]) -> float:
    """M̄ = Σ w_t × m_t。"""
    total = 0.0
    for t, w in occupancy.items():
        total += float(w) * multiplier_for_tier(int(t), multipliers)
    return float(total)


def budget_preserved(m_bar: float, *, tol_low: float = 0.95, tol_high: float = 1.05) -> bool:
    """占有率加重平均倍率 M̄ が保存則許容域 [tol_low, tol_high] に入るか。"""
    return bool(np.isfinite(m_bar) and tol_low <= m_bar <= tol_high)


# ---------------------------------------------------------------------------
# base_stake・100円丸め厳密化（仕様書§3.3・§9-4）
# ---------------------------------------------------------------------------


def min_bankroll_variable(f_var: float, *, rounding_yen: int = 100, multiple: int = 4) -> float:
    """base_stake が multiple*rounding_yen(=400円)の倍数になる最低運用bankroll。

    min_bankroll_variable(f_var) * f_var == multiple * rounding_yen が常に成立する
    ため、この bankroll を使えば compute_base_stake は常に検証を通る。
    """
    return float(multiple) * float(rounding_yen) / float(f_var)


def compute_base_stake(
    bankroll: float,
    f_var: float,
    *,
    rounding_yen: int = 100,
    multiple: int = 4,
) -> float:
    """base_stake = bankroll * f_var。400円(multiple*rounding_yen)の倍数でなければ

    ValueError で拒否する（暗黙の切り捨てで実効倍率が設計値からずれることを禁止する。
    仕様書§3.3）。丸め演算は防御的検証のみで実値は変えない。
    """
    raw = float(bankroll) * float(f_var)
    unit = float(multiple) * float(rounding_yen)
    nearest = round(raw / unit) * unit
    if abs(raw - nearest) > 1e-6:
        raise ValueError(
            f"base_stake={raw} (bankroll={bankroll}, f_var={f_var}) is not an exact multiple "
            f"of {unit} yen. Use bankroll=min_bankroll_variable(f_var)={min_bankroll_variable(f_var, rounding_yen=rounding_yen, multiple=multiple)} "
            "or an integer multiple thereof; refusing implicit rounding (spec §3.3)."
        )
    return raw


# ---------------------------------------------------------------------------
# 可変stake付与（サイジング関数本体。tier/base_stake/multipliersのみを入力とする）
# ---------------------------------------------------------------------------


def apply_variable_stake(
    df: pd.DataFrame,
    *,
    tier_col: str = "tier",
    base_stake: float,
    multipliers: Sequence[float],
    stake_col: str = "stake",
) -> pd.DataFrame:
    """tier列から stake = base_stake * multiplier_for_tier(tier) を付与する。

    引数は (tier列, base_stake, multipliers) のみで、オッズ・払戻・着順・人気・確率の
    いかなる列も参照しない（仕様書§3.4）。
    """
    validate_multipliers(multipliers)
    out = df.copy()
    tiers = out[tier_col].astype(int).to_numpy()
    m_arr = np.asarray([float(x) for x in multipliers], dtype=float)
    out[stake_col] = base_stake * m_arr[tiers - 1]
    out["multiplier"] = m_arr[tiers - 1]
    return out


def effective_multiplier(stake: Sequence[float], base_stake: float) -> np.ndarray:
    """実測 stake/base_stake（設計倍率との一致検出。仕様書§7-4・§9-5）。"""
    s = np.asarray([float(x) for x in stake], dtype=float)
    return s / float(base_stake)


# ---------------------------------------------------------------------------
# f_var の機械的導出（決定規則v2の可変版。仕様書§4 Stage V1）
# ---------------------------------------------------------------------------


def derive_f_var(
    worst_month_dd_at_f0: float,
    f0: float,
    *,
    grid: Sequence[float],
    monthly_mdd_limit: float = 0.15,
    safety_factor: float = 0.5,
) -> dict[str, Any]:
    """f_scale = monthly_mdd_limit / (worst_month_dd_at_f0 / f0);
    f_capped = safety_factor * f_scale;
    adopted = グリッド中 f_capped 以下の最大値（下方拡張のみ、無ければ None）。
    """
    f_scale = float(monthly_mdd_limit) / (float(worst_month_dd_at_f0) / float(f0))
    f_capped = float(safety_factor) * f_scale
    eligible = [float(f) for f in grid if float(f) <= f_capped]
    adopted = max(eligible) if eligible else None
    return {
        "f_scale": f_scale,
        "f_capped": f_capped,
        "grid": [float(f) for f in grid],
        "adopted_f_var": adopted,
    }
