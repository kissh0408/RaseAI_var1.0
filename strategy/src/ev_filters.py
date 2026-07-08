"""Phantom EV 対策: 勝率下限・モデル順位・オッズ連動 min_edge（backtest / 本番共用）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import entropy as scipy_entropy  # type: ignore[import-untyped]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

DEFAULT_DYNAMIC_EDGE_BANDS: list[dict[str, float]] = [
    # オッズ帯ごとに要求 edge を段階的に引き上げる（domain-planner v3 仕様）。
    # strategy_config.json の dynamic_edge_bands と同一値を保つこと（フォールバック時も本番と同じ挙動にするため）。
    # odds_min は参考記録。effective_min_edge の step ロジックは odds_max のみで判定する。
    {"odds_min": 0.0,  "odds_max": 3.0,   "min_edge": 0.08},
    {"odds_min": 3.0,  "odds_max": 6.0,   "min_edge": 0.12},
    {"odds_min": 6.0,  "odds_max": 12.0,  "min_edge": 0.20},
    # 12倍超は事実上ベット禁止（max_odds=12.0 設定と合わせて二重ガードとして機能する）
    {"odds_min": 12.0, "odds_max": 999.0, "min_edge": 999.0},
]


@dataclass
class EvFilterConfig:
    """StrategyConfig / OnlineRecommendationConfig から構築するフィルタ設定。"""

    min_prob: float = 0.01
    min_edge: float = 0.02
    min_odds: float = 2.0
    max_odds: float = 50.0
    max_expected_value: float = 1.5
    min_win_prob: Optional[float] = None
    max_model_rank: Optional[int] = None
    dynamic_edge_enabled: bool = False
    dynamic_edge_mode: str = "step"
    dynamic_edge_bands: list[dict[str, float]] = field(
        default_factory=lambda: [dict(b) for b in DEFAULT_DYNAMIC_EDGE_BANDS]
    )
    dynamic_edge_alpha: float = 0.02
    dynamic_edge_beta: float = 0.08
    race_id_col: str = "race_id"
    score_col: str = "pred_rank1"


def ev_filter_config_from_mapping(mapping: Any, *, score_col: str = "pred_rank1") -> EvFilterConfig:
    """dict または dataclass 互換オブジェクトから EvFilterConfig を構築。"""
    if hasattr(mapping, "__dataclass_fields__"):
        d = {
            f: getattr(mapping, f)
            for f in (
                "min_prob",
                "min_edge",
                "min_odds",
                "max_odds",
                "max_expected_value",
                "min_win_prob",
                "max_model_rank",
                "dynamic_edge_enabled",
                "dynamic_edge_mode",
                "dynamic_edge_bands",
                "dynamic_edge_alpha",
                "dynamic_edge_beta",
                "race_id_col",
            )
            if hasattr(mapping, f)
        }
    else:
        d = dict(mapping) if isinstance(mapping, dict) else {}
    bands = d.get("dynamic_edge_bands")
    if bands is None:
        bands = [dict(b) for b in DEFAULT_DYNAMIC_EDGE_BANDS]
    return EvFilterConfig(
        min_prob=float(d.get("min_prob", 0.01)),
        min_edge=float(d.get("min_edge", 0.02)),
        min_odds=float(d.get("min_odds", 2.0)),
        max_odds=float(d.get("max_odds", 50.0)),
        max_expected_value=float(d.get("max_expected_value", 1.5)),
        min_win_prob=d.get("min_win_prob"),
        max_model_rank=d.get("max_model_rank"),
        dynamic_edge_enabled=bool(d.get("dynamic_edge_enabled", False)),
        dynamic_edge_mode=str(d.get("dynamic_edge_mode", "step")),
        dynamic_edge_bands=[dict(b) for b in bands],
        dynamic_edge_alpha=float(d.get("dynamic_edge_alpha", 0.02)),
        dynamic_edge_beta=float(d.get("dynamic_edge_beta", 0.08)),
        race_id_col=str(d.get("race_id_col", "race_id")),
        score_col=str(d.get("score_col", score_col)),
    )


def effective_min_edge(odds: float | np.ndarray, config: EvFilterConfig) -> float | np.ndarray:
    """オッズに応じた required edge（EV-1）。dynamic 無効時は config.min_edge。"""
    if not config.dynamic_edge_enabled:
        return config.min_edge

    mode = str(config.dynamic_edge_mode).lower()
    if isinstance(odds, np.ndarray):
        o = np.clip(odds.astype(float), 1.01, None)
    else:
        o = max(float(odds), 1.01)

    if mode == "log_linear":
        dyn = config.dynamic_edge_alpha + config.dynamic_edge_beta * np.log(o)
        if isinstance(dyn, np.ndarray):
            return np.maximum(config.min_edge, dyn)
        return max(config.min_edge, float(dyn))

    # step bands
    bands = sorted(config.dynamic_edge_bands, key=lambda b: float(b.get("odds_max", float("inf"))))
    if isinstance(o, np.ndarray):
        out = np.full(o.shape, config.min_edge, dtype=float)
        prev_max = 0.0
        for band in bands:
            hi = float(band.get("odds_max", float("inf")))
            me = float(band.get("min_edge", config.min_edge))
            mask = (o > prev_max) & (o <= hi) if math.isfinite(hi) else (o > prev_max)
            out = np.where(mask, np.maximum(me, config.min_edge), out)
            prev_max = hi if math.isfinite(hi) else prev_max
        return out

    prev_max = 0.0
    for band in bands:
        hi = float(band.get("odds_max", float("inf")))
        me = float(band.get("min_edge", config.min_edge))
        in_band = (o > prev_max) and (o <= hi if math.isfinite(hi) else True)
        if in_band:
            return max(config.min_edge, me)
        prev_max = hi
    return max(config.min_edge, float(bands[-1].get("min_edge", config.min_edge)))


def attach_model_rank(
    df: pd.DataFrame,
    *,
    race_id_col: str,
    prob_col: str = "pred_prob",
    score_col: str = "pred_score_raw",
) -> pd.DataFrame:
    """レース内で pred_prob DESC, score_col DESC の安定ソートにより model_rank を付与。"""
    out = df.copy()
    if score_col not in out.columns:
        raise ValueError(f"score_col not found for tie-break: {score_col}")
    sort_keys = [prob_col, score_col]
    for c in sort_keys:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-np.inf)
    out = out.sort_values(
        [race_id_col, prob_col, score_col],
        ascending=[True, False, False],
        kind="mergesort",
    )
    out["model_rank"] = out.groupby(race_id_col, sort=False).cumcount() + 1
    return out


def _prob_threshold(config: EvFilterConfig) -> float:
    floor = float(config.min_prob)
    if config.min_win_prob is not None:
        floor = max(floor, float(config.min_win_prob))
    return floor


def apply_bet_candidate_mask(
    df: pd.DataFrame,
    config: EvFilterConfig,
    *,
    odds_col: str = "effective_odds",
) -> pd.Series:
    """候補抽出用 bool マスク（backtest / 本番共用）。"""
    if "edge" not in df.columns or odds_col not in df.columns:
        raise ValueError("df must have edge and odds columns before apply_bet_candidate_mask")

    odds = pd.to_numeric(df[odds_col], errors="coerce")
    prob = pd.to_numeric(df["pred_prob"], errors="coerce")
    edge = pd.to_numeric(df["edge"], errors="coerce")
    ev = pd.to_numeric(df.get("expected_value", prob * odds), errors="coerce")

    mask = (
        prob.ge(_prob_threshold(config))
        & odds.ge(config.min_odds)
        & odds.le(config.max_odds)
        & ev.le(config.max_expected_value)
    )

    if config.max_model_rank is not None and "model_rank" in df.columns:
        mask &= pd.to_numeric(df["model_rank"], errors="coerce").le(int(config.max_model_rank))

    req_edge = effective_min_edge(odds.to_numpy(dtype=float), config)
    if isinstance(req_edge, np.ndarray):
        mask &= edge.to_numpy(dtype=float) >= req_edge
    else:
        mask &= edge.ge(float(req_edge))

    # 静的 min_edge も floor（dynamic 無効時は effective_min_edge が min_edge を返す）
    mask &= edge.ge(config.min_edge)

    return pd.Series(mask, index=df.index)


def post_slippage_edge_gate(
    pred_prob: float,
    exec_odds: float,
    config: EvFilterConfig,
) -> bool:
    """約定オッズ基準の edge ゲート（dynamic ON 時は必須）。"""
    exec_edge = float(pred_prob) * float(exec_odds) - 1.0
    required = effective_min_edge(float(exec_odds), config)
    if isinstance(required, np.ndarray):
        required = float(required.item()) if required.size == 1 else float(required.flat[0])
    return exec_edge >= max(float(config.min_edge), float(required))


def should_apply_post_slippage_gate(config: EvFilterConfig, *, enforce_post_slippage_edge: bool) -> bool:
    return bool(config.dynamic_edge_enabled or enforce_post_slippage_edge)


def prepare_ev_columns(
    df: pd.DataFrame,
    config: EvFilterConfig,
    *,
    calibrator: Any = None,
    normalize_probs_in_race: bool = True,
    base_slippage: float = 0.01,
    odds_col: str = "odds",
) -> pd.DataFrame:
    """pred_prob / effective_odds / edge / model_rank を一括付与（backtest 前処理用）。"""
    out = df.copy()
    score_col = config.score_col
    if calibrator is not None:
        out["pred_prob"] = calibrator.transform(out[score_col]).clip(0.0, 1.0)
    else:
        out["pred_prob"] = pd.to_numeric(out[score_col], errors="coerce").clip(0.0, 1.0)

    if normalize_probs_in_race:
        grp = out.groupby(config.race_id_col)["pred_prob"].transform("sum").clip(lower=1e-12)
        out["pred_prob"] = out["pred_prob"] / grp

    out["pred_score_raw"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    out = attach_model_rank(
        out,
        race_id_col=config.race_id_col,
        prob_col="pred_prob",
        score_col="pred_score_raw",
    )
    raw_odds = pd.to_numeric(out[odds_col], errors="coerce")
    out["effective_odds"] = raw_odds * (1.0 - float(base_slippage))
    out["effective_odds"] = out["effective_odds"].clip(lower=1.01)
    out["expected_value"] = out["pred_prob"] * out["effective_odds"]
    out["edge"] = out["expected_value"] - 1.0
    return out


def _safe_div(num: float, denom: float) -> float:
    if denom <= 1e-12:
        return 0.0
    return float(num) / float(denom)


def harville_quinella_pair_prob(p_i: float, p_j: float) -> float:
    """2頭の単勝確率から馬連（1-2着・順不同）の Harville 確率。"""
    p_i = float(np.clip(p_i, 0.0, 1.0 - 1e-12))
    p_j = float(np.clip(p_j, 0.0, 1.0 - 1e-12))
    term1 = p_i * _safe_div(p_j, 1.0 - p_i)
    term2 = p_j * _safe_div(p_i, 1.0 - p_j)
    return float(np.clip(term1 + term2, 0.0, 1.0))


def calculate_harville_quinella(
    p_dict: dict[int, float],
) -> dict[tuple[int, int], float]:
    """単勝勝率 p_dict から馬連(Quinella)の確率を算出（順不同キー min,max）。"""
    quinella_probs: dict[tuple[int, int], float] = {}
    horses = sorted(int(h) for h in p_dict.keys())
    for i in range(len(horses)):
        for j in range(i + 1, len(horses)):
            h1, h2 = horses[i], horses[j]
            quinella_probs[(h1, h2)] = harville_quinella_pair_prob(
                float(p_dict[h1]), float(p_dict[h2])
            )
    return quinella_probs


def harville_wide_pair_prob(p_dict: dict[int, float], h1: int, h2: int) -> float:
    """2頭の単勝確率からワイド（3着以内・順不同）の Harville 確率。"""
    p1 = float(np.clip(float(p_dict[h1]), 0.0, 1.0 - 1e-12))
    p2 = float(np.clip(float(p_dict[h2]), 0.0, 1.0 - 1e-12))
    p_1st_2nd = harville_quinella_pair_prob(p1, p2)

    p_3rd_sum = 0.0
    for k, pk_raw in p_dict.items():
        k = int(k)
        if k == h1 or k == h2:
            continue
        pk = float(np.clip(float(pk_raw), 0.0, 1.0 - 1e-12))
        d1 = 1.0 - p1
        d2 = 1.0 - p2
        dk = 1.0 - pk
        p_a = p1 * _safe_div(pk, d1) * _safe_div(p2, d1 - pk) if (d1 - pk) > 1e-12 else 0.0
        p_b = p2 * _safe_div(pk, d2) * _safe_div(p1, d2 - pk) if (d2 - pk) > 1e-12 else 0.0
        p_c = pk * _safe_div(p1, dk) * _safe_div(p2, dk - p1) if (dk - p1) > 1e-12 else 0.0
        p_d = pk * _safe_div(p2, dk) * _safe_div(p1, dk - p2) if (dk - p2) > 1e-12 else 0.0
        p_3rd_sum += p_a + p_b + p_c + p_d

    return float(np.clip(p_1st_2nd + p_3rd_sum, 0.0, 1.0))


def calculate_harville_wide(
    p_dict: dict[int, float],
) -> dict[tuple[int, int], float]:
    """単勝勝率 p_dict からワイド(Quinella Place)の確率を算出。"""
    wide_probs: dict[tuple[int, int], float] = {}
    horses = sorted(int(h) for h in p_dict.keys())
    for i in range(len(horses)):
        for j in range(i + 1, len(horses)):
            h1, h2 = horses[i], horses[j]
            wide_probs[(h1, h2)] = harville_wide_pair_prob(p_dict, h1, h2)
    return wide_probs


def wide_probs_from_win_probs(p_dict: dict[int, float]) -> dict[tuple[int, int], float]:
    """Alias for Layer 2 wide probability derivation (Step 2)."""
    return calculate_harville_wide(p_dict)


def box_combinations(horse_nums: Sequence[int]) -> list[tuple[int, int]]:
    """上位馬リストから馬連/ワイド BOX の組み合わせ（順不同）。"""
    nums = sorted({int(h) for h in horse_nums})
    return [(nums[i], nums[j]) for i in range(len(nums)) for j in range(i + 1, len(nums))]


# ---------------------------------------------------------------------------
# レースリスク指標（モデル信頼度・フィールドエントロピー）
# ---------------------------------------------------------------------------

def calc_race_risk_metrics(
    rec_df: pd.DataFrame,
    *,
    prob_col: str = "pred_prob",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    レースごとの pred_prob_confidence（1位と2位の確率差）と
    field_entropy（pred_prob 分布のエントロピー）を算出して rec_df に追加する。

    pred_prob_confidence が低い（1・2位の確率差が小さい）レースは
    モデルが優劣を判別できておらず、ファントム EV が混入しやすい。
    field_entropy が高い（出走馬の確率が均一）レースは荒れ展開になりやすい。

    Parameters
    ----------
    rec_df : pd.DataFrame
        モデル推論後の予測 DataFrame。pred_prob と race_id が必須。
    prob_col : str
        予測確率カラム名（既定 "pred_prob"）。
    race_id_col : str
        レース識別子カラム名（既定 "race_id"）。

    Returns
    -------
    pd.DataFrame
        入力 rec_df に以下の 2 カラムを追加して返す:
        - pred_prob_confidence : レース内 1 位と 2 位の確率差（高いほどモデル自信あり）
        - field_entropy        : レース内 pred_prob のシャノンエントロピー（nats）
    """
    if prob_col not in rec_df.columns:
        raise ValueError(f"prob_col '{prob_col}' が rec_df に存在しません。")
    if race_id_col not in rec_df.columns:
        raise ValueError(f"race_id_col '{race_id_col}' が rec_df に存在しません。")

    out = rec_df.copy()

    def _confidence(probs: pd.Series) -> float:
        """1 位と 2 位の確率差（出走頭数が 1 の場合は 1.0）。"""
        sorted_probs = probs.sort_values(ascending=False).to_numpy(dtype=float)
        if len(sorted_probs) < 2:
            return 1.0
        return float(sorted_probs[0] - sorted_probs[1])

    def _entropy(probs: pd.Series) -> float:
        """正規化した予測確率に対するシャノンエントロピー（nats）。"""
        p = probs.to_numpy(dtype=float)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        if _SCIPY_AVAILABLE:
            return float(scipy_entropy(p))
        # scipy 未導入のフォールバック実装
        return float(-np.sum(p * np.log(p)))

    confidence = out.groupby(race_id_col)[prob_col].transform(_confidence)
    entropy = out.groupby(race_id_col)[prob_col].transform(_entropy)

    out["pred_prob_confidence"] = confidence
    out["field_entropy"] = entropy
    return out


# ---------------------------------------------------------------------------
# 月次ドローダウン上限フィルタ（バックテスト / シミュレーション用）
# ---------------------------------------------------------------------------

def apply_monthly_drawdown_filter(
    rec_df: pd.DataFrame,
    monthly_dd_limit: float = -0.10,
    bankroll: float = 750_000.0,
    *,
    date_col: str = "race_date",
    profit_col: str = "profit",
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """
    月次累積損益が monthly_dd_limit（比率）を下回った月の残レースをフィルタリングする。

    バックテスト・シミュレーション用。実稼働では main.py の月次チェックとして使用する。

    Parameters
    ----------
    rec_df : pd.DataFrame
        ベット結果を含む DataFrame。date_col / profit_col / race_id_col が必要。
    monthly_dd_limit : float
        月次ドローダウン上限（比率。例: -0.10 = 月間損失が資金の 10% 超でその月の残ベットを停止）。
    bankroll : float
        初期資金（月次損益比率の分母）。
    date_col : str
        日付カラム名（既定 "race_date"）。
    profit_col : str
        損益カラム名（既定 "profit"）。rec_df に存在しない場合は全レースをスルー。
    race_id_col : str
        レース識別子カラム名（既定 "race_id"）。

    Returns
    -------
    pd.DataFrame
        月次ドローダウン上限を超えた月の、上限超過レース以降のレースを除いた DataFrame。
        上限を超えていない月は変更なし。
    """
    if profit_col not in rec_df.columns:
        # profit カラムがない場合（推論前 DataFrame など）はフィルタをかけずに返す
        return rec_df.copy()
    if date_col not in rec_df.columns:
        return rec_df.copy()

    out = rec_df.copy()

    # date_col を datetime 型に変換
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["_ym"] = out[date_col].dt.to_period("M")

    keep_mask = pd.Series(True, index=out.index)

    for ym, month_df in out.groupby("_ym", sort=True):
        # 月内での race_id 順（race_id が文字列日付込みの場合はソートで日付順になる）
        month_sorted = month_df.sort_values([date_col, race_id_col])
        cumulative_profit = month_sorted[profit_col].cumsum()
        cumulative_ratio = cumulative_profit / max(float(bankroll), 1.0)

        # 上限を初めて下回るインデックスを探す
        breach_mask = cumulative_ratio < monthly_dd_limit
        if not breach_mask.any():
            continue

        # 上限超過が発生した最初のレース以降（当該レース含む）を除外する
        first_breach_pos = int(np.argmax(breach_mask.to_numpy()))
        exclude_idx = month_sorted.index[first_breach_pos:]
        keep_mask.loc[exclude_idx] = False

    out = out.loc[keep_mask].drop(columns=["_ym"])
    return out
