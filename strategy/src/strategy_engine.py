from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd
import warnings


class ScoreCalibrator(Protocol):
    def transform(self, scores: pd.Series) -> pd.Series:
        ...


@dataclass
class OnlineRecommendationConfig:
    score_col: str = "pred_rank1"
    race_id_col: str = "race_id"
    horse_col: str = "horse_num"
    odds_col: str = "odds"
    bet_unit: int = 100
    min_prob: float = 0.01
    min_edge: float = 0.10
    min_odds: float = 2.0
    max_odds: float = 15.0
    max_selections_per_race: int = 2
    fractional_kelly: float = 0.08  # CLAUDE.md: v13バックテストMDD-27.3%のため0.08に固定
    max_total_fraction: float = 0.30
    max_single_fraction: float = 0.15
    max_stake_per_bet: int = 3_000
    max_invest_per_race: int = 50_000
    base_slippage: float = 0.01
    normalize_probs_in_race: bool = True
    recommendation_bankroll: int = 100_000
    pair_top_n: int = 2
    wide_top_n: int = 2
    pair_partner_col: str = "pred_rank2"
    wide_partner_col: str = "pred_rank3"
    odds_timestamp_col: str = "odds_timestamp"
    phase: str = "phase1"
    phase2_enabled: bool = False
    save_snapshot_timestamps: bool = True
    probability_policy: str = "market_shrinkage"
    market_shrinkage_alpha: float = 0.20
    max_expected_value: float = 1.5
    max_odds_for_kelly: float = 30.0
    min_bucket_count: int = 100
    quinella_odds_col: str = "quinella_odds"
    wide_odds_col: str = "wide_odds"
    odds_source: str = "unknown"
    odds_cutoff_policy: str = "unknown"
    # 単勝: 当レースで score_col 最大の馬だけを候補にする（既定オフ・後方互換）
    require_score_rank1: bool = False
    # phase1.5: harville=従来（正規化勝率でアンカー） / top2_pred=score_col 上位2頭固定で Harville ペア確率
    pair_selection_mode: str = "harville"
    # score_rank 等未キャリブレーション経路では False（フラット買い強制）
    allow_kelly: bool = True
    min_win_prob: Optional[float] = 0.05
    max_model_rank: Optional[int] = None
    dynamic_edge_enabled: bool = False
    dynamic_edge_mode: str = "step"
    dynamic_edge_bands: Optional[list] = None
    dynamic_edge_alpha: float = 0.02
    dynamic_edge_beta: float = 0.08
    # フィールドサイズフィルタ
    # 少頭数レース（少頭数はオッズ構造が特殊でEV計算が過楽観になりやすい）を除外する
    min_field_size: int = 9
    # フルゲート付近の大頭数レースで min_edge を追加引き上げるための設定
    large_field_threshold: int = 18
    large_field_extra_edge: float = 0.05
    # 複勝オッズカラム名（place_odds が存在する場合のみ複勝推奨を生成する）
    place_odds_col: str = "place_odds"
    # 馬連確率: Harville と Rank2 直接確率のブレンド比率（0=Harvilleのみ, 1=Rank2のみ）
    rank2_blend: float = 0.35
    # ワイド専用 edge 下限（combo_backtest.wide_min_edge と揃える）
    wide_min_edge: float = 0.05
    wide_bets_enabled: bool = True
    quinella_bets_enabled: bool = True
    place_bets_enabled: bool = True
    # wide: harville=anchor flow / divergence=Strategy D (argmax log_divergence)
    wide_selection: str = "harville"
    wide_ev_threshold: float = 1.05
    wide_div_threshold: float = 0.0
    # 分散調整型Kelly: 高オッズ帯で fractional_kelly をオッズに応じて縮小
    dynamic_kelly_enabled: bool = False
    dynamic_kelly_base_fraction: float = 0.08
    dynamic_kelly_odds_ref: float = 3.0
    dynamic_kelly_power: float = 0.5
    # FLB（Favorite-Longshot Bias）補正
    market_bias_correction_enabled: bool = False
    market_bias_correction_model: str = "isotonic"
    # C2: win+wide 非排他多変量 Kelly（portfolio_kelly_enabled=false で従来比例縮小）
    portfolio_kelly_enabled: bool = False
    portfolio_kelly_mode: str = "portfolio_kelly_fractional"
    portfolio_growth_ratio_min: float = 0.5
    portfolio_ind_cap_ratio: float = 0.85
    portfolio_mc_samples: int = 500
    portfolio_mc_seed: int = 42


def _project_to_capped_simplex(vec: np.ndarray, cap: float) -> np.ndarray:
    vec = np.maximum(vec, 0.0)
    s = float(vec.sum())
    if s <= cap or s == 0:
        return vec
    return vec * (cap / s)


def _single_kelly_fraction(prob: float, odds: float) -> float:
    b = max(float(odds) - 1.0, 1e-12)
    q = 1.0 - float(prob)
    return max((b * float(prob) - q) / b, 0.0)


def get_dynamic_kelly_fraction(
    odds: float,
    base_fraction: float = 0.08,
    odds_ref: float = 3.0,
    power: float = 0.5,
) -> float:
    """オッズが大きいほど Kelly 比率を縮小する（高オッズ帯の過剰ベット抑制）。"""
    odds = max(float(odds), 1.01)
    adjusted = float(base_fraction) * (float(odds_ref) / odds) ** float(power)
    return min(float(base_fraction), adjusted)


def _kelly_fraction_for_odds(config: OnlineRecommendationConfig, odds: float) -> float:
    if config.dynamic_kelly_enabled:
        return get_dynamic_kelly_fraction(
            float(odds),
            base_fraction=float(config.dynamic_kelly_base_fraction or config.fractional_kelly),
            odds_ref=float(config.dynamic_kelly_odds_ref),
            power=float(config.dynamic_kelly_power),
        )
    return config.fractional_kelly


def _apply_fixed_slippage(odds: pd.Series, base_slippage: float) -> pd.Series:
    adjusted = odds.astype(float) * (1.0 - float(base_slippage))
    return adjusted.clip(lower=1.01)


def _market_probabilities(df: pd.DataFrame, *, config: OnlineRecommendationConfig) -> pd.Series:
    odds = pd.to_numeric(df[config.odds_col], errors="coerce").clip(lower=1.01)
    implied = (1.0 / odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = implied.groupby(df[config.race_id_col]).transform("sum").clip(lower=1e-12)
    return (implied / denom).clip(0.0, 1.0)


def _score_to_probabilities(
    df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    calibrator: Optional[ScoreCalibrator] = None,
) -> pd.Series:
    score = pd.to_numeric(df[config.score_col], errors="coerce").fillna(0.0)
    if calibrator is not None:
        model_prob = calibrator.transform(score).clip(0.0, 1.0)
    else:
        model_prob = score.clip(0.0, 1.0)

    policy = str(config.probability_policy).lower()
    if policy == "market_shrinkage":
        market_prob = _market_probabilities(df, config=config)
        alpha = float(np.clip(config.market_shrinkage_alpha, 0.0, 1.0))
        prob = ((1.0 - alpha) * model_prob + alpha * market_prob).clip(0.0, 1.0)
    elif policy == "market":
        prob = _market_probabilities(df, config=config)
    else:
        prob = model_prob

    if config.normalize_probs_in_race:
        race_sum = prob.groupby(df[config.race_id_col]).transform("sum").clip(lower=1e-12)
        prob = prob / race_sum
    return prob.clip(0.0, 1.0)


def recommend_win_phase1(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    calibrator: Optional[ScoreCalibrator] = None,
    generated_at: Optional[str] = None,
    _flb_corrector=None,
) -> pd.DataFrame:
    df = pred_df.copy()
    for c in [config.race_id_col, config.horse_col, config.odds_col, config.score_col]:
        if c not in df.columns:
            raise ValueError(f"required column not found: {c}")

    df[config.race_id_col] = df[config.race_id_col].astype(str)
    df[config.horse_col] = pd.to_numeric(df[config.horse_col], errors="coerce")
    df[config.odds_col] = pd.to_numeric(df[config.odds_col], errors="coerce")
    df = df.dropna(subset=[config.horse_col, config.odds_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df["pred_prob"] = _score_to_probabilities(df, config=config, calibrator=calibrator)
    df["pred_score_raw"] = pd.to_numeric(df[config.score_col], errors="coerce").fillna(0.0)
    try:
        from strategy.src.ev_filters import attach_model_rank, apply_bet_candidate_mask, ev_filter_config_from_mapping
    except ModuleNotFoundError:
        from ev_filters import attach_model_rank, apply_bet_candidate_mask, ev_filter_config_from_mapping

    df = attach_model_rank(
        df,
        race_id_col=config.race_id_col,
        prob_col="pred_prob",
        score_col="pred_score_raw",
    )
    df["effective_odds"] = _apply_fixed_slippage(df[config.odds_col], config.base_slippage)
    df["expected_value"] = df["pred_prob"] * df["effective_odds"]
    df["edge"] = df["expected_value"] - 1.0
    ev_cfg = ev_filter_config_from_mapping(config)

    all_rows = []
    for race_id, race in df.groupby(config.race_id_col, sort=False):
        # フィールドサイズフィルタ:
        # n_horses カラムが存在する場合のみ適用する（後方互換を維持するため）。
        # 少頭数レースはオッズ構造が特殊でEV計算が過楽観になるため除外する。
        if "n_horses" in race.columns:
            n_horses_val = pd.to_numeric(race["n_horses"], errors="coerce").dropna()
            if not n_horses_val.empty:
                n_horses_int = int(n_horses_val.iloc[0])
                if n_horses_int < config.min_field_size:
                    continue
        race_scope = race
        if getattr(config, "require_score_rank1", False):
            sc = pd.to_numeric(race_scope[config.score_col], errors="coerce").fillna(-np.inf)
            idx_top = sc.idxmax()
            hn_top = int(race_scope.loc[idx_top, config.horse_col])
            race_scope = race_scope[race_scope[config.horse_col].eq(hn_top)].copy()
            if race_scope.empty:
                continue
        # フルゲートEV引き上げ:
        # n_horses >= large_field_threshold の場合、large_field_extra_edge 分だけ min_edge を加算する。
        # ev_cfg は共通インスタンスなので、レース単位で上書きせず別インスタンスを作る。
        race_ev_cfg = ev_cfg
        if "n_horses" in race_scope.columns:
            n_horses_vals = pd.to_numeric(race_scope["n_horses"], errors="coerce").dropna()
            if not n_horses_vals.empty:
                n_h = int(n_horses_vals.iloc[0])
                if n_h >= config.large_field_threshold and config.large_field_extra_edge > 0.0:
                    race_ev_cfg = copy.copy(ev_cfg)
                    race_ev_cfg.min_edge = ev_cfg.min_edge + float(config.large_field_extra_edge)
        mask = apply_bet_candidate_mask(race_scope, race_ev_cfg, odds_col="effective_odds")
        cand = race_scope.loc[mask].sort_values("edge", ascending=False)
        cand = cand.head(config.max_selections_per_race)
        if cand.empty:
            continue
        raw_frac = []
        flat_frac = config.bet_unit / max(config.recommendation_bankroll, 1.0)
        for p, o in zip(cand["pred_prob"], cand["effective_odds"]):
            if not config.allow_kelly:
                raw_frac.append(flat_frac)
            elif float(o) > float(config.max_odds_for_kelly):
                raw_frac.append(flat_frac)
            else:
                raw_frac.append(
                    _single_kelly_fraction(p, o) * _kelly_fraction_for_odds(config, float(o))
                )
        raw_frac = np.array(raw_frac, dtype=float)
        fracs = np.clip(raw_frac, 0.0, config.max_single_fraction)
        fracs = _project_to_capped_simplex(fracs, config.max_total_fraction)
        race_invest = 0
        for (_, row), frac in zip(cand.iterrows(), fracs):
            target = config.recommendation_bankroll * float(frac)
            allowed = min(target, config.max_stake_per_bet, config.max_invest_per_race - race_invest)
            stake = int(max(allowed, 0.0) // config.bet_unit) * config.bet_unit
            if stake < config.bet_unit:
                continue
            race_invest += stake
            all_rows.append(
                {
                    "ticket_type": "単勝",
                    "race_id": str(race_id),
                    "horse_num": int(row[config.horse_col]),
                    "ticket": str(int(row[config.horse_col])),
                    "pred_prob": float(row["pred_prob"]),
                    "odds_raw": float(row[config.odds_col]),
                    "odds_effective": float(row["effective_odds"]),
                    "expected_value": float(row["expected_value"]),
                    "edge": float(row["edge"]),
                    "kelly_fraction": float(frac),
                    "suggested_stake": int(stake),
                    "is_executable": True,
                    "phase": "phase1",
                    "modeling_note": f"EV/Kelly from {config.probability_policy} probability",
                    "odds_timestamp": (
                        row.get(config.odds_timestamp_col, None)
                        if config.save_snapshot_timestamps
                        else None
                    ),
                    "generated_at": generated_at if config.save_snapshot_timestamps else None,
                    "odds_source": config.odds_source,
                    "odds_cutoff_policy": config.odds_cutoff_policy,
                }
            )
    return pd.DataFrame(all_rows)


def _pick_phase1_pairs(
    race: pd.DataFrame,
    *,
    score_col: str,
    partner_col: str,
    ticket_type: str,
    top_n: int,
) -> list[dict]:
    if race.empty:
        return []
    local = race.copy()
    local[score_col] = pd.to_numeric(local[score_col], errors="coerce").fillna(0.0)
    local[partner_col] = pd.to_numeric(local.get(partner_col, local[score_col]), errors="coerce").fillna(0.0)
    anchor = local.sort_values(score_col, ascending=False).head(1)
    if anchor.empty:
        return []
    h1 = int(anchor.iloc[0]["horse_num"])
    others = local[local["horse_num"] != h1].sort_values(partner_col, ascending=False).head(max(top_n, 0))
    rows: list[dict] = []
    for _, row in others.iterrows():
        h2 = int(row["horse_num"])
        rows.append(
            {
                "ticket_type": ticket_type,
                "ticket": f"{h1}-{h2}",
                "horse_num": h1,
                "partner_horse_num": h2,
                "heuristic_score": float(anchor.iloc[0][score_col]) * float(row[partner_col]),
                "kelly_fraction": np.nan,
                "suggested_stake": 0,  # recommend_pair_wide_phase1 で固定額を設定
                "modeling_note": f"heuristic using {score_col}/{partner_col}",
            }
        )
    return rows


def recommend_pair_wide_phase1(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    generated_at: Optional[str] = None,
) -> pd.DataFrame:
    df = pred_df.copy()
    required = [config.race_id_col, config.horse_col, config.score_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"required column not found: {c}")
    df[config.race_id_col] = df[config.race_id_col].astype(str)
    df[config.horse_col] = pd.to_numeric(df[config.horse_col], errors="coerce")
    df = df.dropna(subset=[config.horse_col]).copy()

    rows = []
    for race_id, race in df.groupby(config.race_id_col, sort=False):
        local = race.copy()
        local["horse_num"] = local[config.horse_col].astype(int)
        pair_rows = _pick_phase1_pairs(
            local,
            score_col=config.score_col,
            partner_col=config.pair_partner_col if config.pair_partner_col in local.columns else config.score_col,
            ticket_type="馬連",
            top_n=config.pair_top_n,
        )
        wide_rows = _pick_phase1_pairs(
            local,
            score_col=config.score_col,
            partner_col=config.wide_partner_col if config.wide_partner_col in local.columns else config.score_col,
            ticket_type="ワイド",
            top_n=config.wide_top_n,
        )
        for row in pair_rows + wide_rows:
            row["suggested_stake"] = int(config.bet_unit)
            row["is_executable"] = True
            row.update(
                {
                    "race_id": str(race_id),
                    "pred_prob": np.nan,
                    "odds_raw": np.nan,
                    "odds_effective": np.nan,
                    "expected_value": np.nan,
                    "edge": np.nan,
                    "phase": "phase1",
                    "odds_timestamp": (
                        local.iloc[0].get(config.odds_timestamp_col, None)
                        if config.save_snapshot_timestamps
                        else None
                    ),
                    "generated_at": generated_at if config.save_snapshot_timestamps else None,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _harville_quinella_probability(p_i: float, p_j: float) -> float:
    p_i = float(np.clip(p_i, 0.0, 1.0 - 1e-12))
    p_j = float(np.clip(p_j, 0.0, 1.0 - 1e-12))
    ij = p_i * p_j / max(1.0 - p_i, 1e-12)
    ji = p_j * p_i / max(1.0 - p_j, 1e-12)
    return float(np.clip(ij + ji, 0.0, 1.0))


def _harville_wide_probability(local: pd.DataFrame, h1: int, h2: int) -> float:
    """Correct wide (top-3 pair) probability via ev_filters.harville_wide_pair_prob."""
    try:
        from ev_filters import harville_wide_pair_prob
    except ModuleNotFoundError:
        from strategy.src.ev_filters import harville_wide_pair_prob

    p_dict = {int(r["horse_num"]): float(r["harville_prob"]) for _, r in local.iterrows()}
    return float(harville_wide_pair_prob(p_dict, int(h1), int(h2)))


def _pick_wide_divergence_row(
    race_id: str,
    local: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    wide_odds_dict: dict | None,
    generated_at: str | None,
    race_invest: int,
) -> dict | None:
    """Strategy D: argmax log_divergence with EV and divergence thresholds."""
    try:
        from ev_filters import wide_probs_from_win_probs
        from wide_ev_core import collect_divergence_bets_per_race, live_dict_to_race_lookup
    except ModuleNotFoundError:
        from strategy.src.ev_filters import wide_probs_from_win_probs
        from strategy.src.wide_ev_core import collect_divergence_bets_per_race, live_dict_to_race_lookup

    p_dict = {int(r["horse_num"]): float(r["harville_prob"]) for _, r in local.iterrows()}
    p_wide_map = wide_probs_from_win_probs(p_dict)
    lookup = live_dict_to_race_lookup(wide_odds_dict) if wide_odds_dict else {}
    pick = collect_divergence_bets_per_race(
        race_id,
        p_wide_map,
        lookup,
        strategy="D",
        ev_threshold=float(getattr(config, "wide_ev_threshold", 1.05)),
        div_threshold=float(getattr(config, "wide_div_threshold", 0.0)),
    )
    if not pick or not pick.get("bet"):
        return None

    h1, h2 = pick["pair"]
    odds_val = float(pick["wide_odds"])
    pair_prob = float(pick["p_wide"])
    effective_odds = max(odds_val * (1.0 - config.base_slippage), 1.01)
    ev = float(pick["ev_wide"])
    edge = ev - 1.0
    if ev > config.max_expected_value or edge < config.wide_min_edge:
        return None

    frac = (
        config.bet_unit / max(config.recommendation_bankroll, 1.0)
        if effective_odds > config.max_odds_for_kelly
        else _single_kelly_fraction(pair_prob, effective_odds) * _kelly_fraction_for_odds(config, effective_odds)
    )
    stake = int(
        min(
            config.recommendation_bankroll * frac,
            config.max_stake_per_bet,
            config.max_invest_per_race - race_invest,
        )
        // config.bet_unit
    ) * config.bet_unit
    if stake < config.bet_unit:
        return None

    return {
        "ticket_type": "ワイド",
        "race_id": str(race_id),
        "horse_num": int(h1),
        "partner_horse_num": int(h2),
        "ticket": f"{h1}-{h2}",
        "pred_prob": pair_prob,
        "odds_raw": odds_val,
        "odds_effective": effective_odds,
        "expected_value": ev,
        "edge": edge,
        "log_divergence": float(pick["log_divergence"]),
        "kelly_fraction": frac,
        "suggested_stake": int(stake),
        "is_executable": True,
        "phase": "phase1_5",
        "modeling_note": "strategy_d_log_divergence",
        "odds_timestamp": (
            local.iloc[0].get(config.odds_timestamp_col, None) if config.save_snapshot_timestamps else None
        ),
        "generated_at": generated_at if config.save_snapshot_timestamps else None,
        "odds_source": config.odds_source,
        "odds_cutoff_policy": config.odds_cutoff_policy,
    }


def recommend_pair_wide_phase15(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    calibrator: Optional[ScoreCalibrator] = None,
    generated_at: Optional[str] = None,
    quinella_odds_dict: Optional[dict] = None,
    wide_odds_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    quinella_odds_dict : {(race_id, h1_str, h2_str): float} (h1 < h2)
        O2速報オッズ。None の場合は DataFrame の quinella_odds 列を参照（なければ NaN）。
    wide_odds_dict : {(race_id, h1_str, h2_str): float} (h1 < h2)
        O3速報オッズ（最低オッズを使用）。None の場合は wide_odds 列を参照。
    """
    df = pred_df.copy()
    for c in [config.race_id_col, config.horse_col, config.score_col]:
        if c not in df.columns:
            raise ValueError(f"required column not found: {c}")
    if config.odds_col not in df.columns:
        df[config.odds_col] = np.nan
    df[config.race_id_col] = df[config.race_id_col].astype(str)
    df[config.horse_col] = pd.to_numeric(df[config.horse_col], errors="coerce")
    df = df.dropna(subset=[config.horse_col]).copy()
    df["pred_prob"] = _score_to_probabilities(df, config=config, calibrator=calibrator)

    mode = str(getattr(config, "pair_selection_mode", "harville") or "harville").lower().replace("-", "_")
    wide_sel = str(getattr(config, "wide_selection", "harville") or "harville").lower()

    rows = []
    for race_id, race in df.groupby(config.race_id_col, sort=False):
        # フィールドサイズフィルタ（recommend_win_phase1 と同じ基準で連複・ワイドも除外）。
        # フルゲート時は min_edge を large_field_extra_edge 分だけ引き上げる。
        _race_min_edge = config.min_edge
        if "n_horses" in race.columns:
            n_horses_val = pd.to_numeric(race["n_horses"], errors="coerce").dropna()
            if not n_horses_val.empty:
                _n_h = int(n_horses_val.iloc[0])
                if _n_h < config.min_field_size:
                    continue
                if _n_h >= config.large_field_threshold and config.large_field_extra_edge > 0.0:
                    _race_min_edge = config.min_edge + float(config.large_field_extra_edge)
        race_invest = 0
        local = race.copy()
        local["horse_num"] = local[config.horse_col].astype(int)
        # Harvilleは全頭確率の合計1が前提。ここで明示的に再正規化する。
        denom = local["pred_prob"].sum()
        if denom <= 0:
            continue
        local["harville_prob"] = local["pred_prob"] / denom
        local["_score_for_order"] = pd.to_numeric(local[config.score_col], errors="coerce").fillna(-np.inf)

        # Rank2スコアをレース内でsoftmax正規化して「2番目に来る確率」として扱う。
        # pred_rank2 が存在する場合のみ馬連確率のブレンドに使う（ワイドは除外）。
        RANK2_BLEND = float(config.rank2_blend)
        rank2_col = "pred_rank2"
        _has_rank2 = rank2_col in local.columns
        if _has_rank2:
            r2_scores = pd.to_numeric(local[rank2_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            r2_sum = float(r2_scores.sum())
            if r2_sum > 0:
                local["_rank2_prob"] = r2_scores / r2_sum
            else:
                local["_rank2_prob"] = 1.0 / max(len(local), 1)
                _has_rank2 = False

        if config.wide_bets_enabled and wide_sel == "divergence":
            wrow = _pick_wide_divergence_row(
                str(race_id),
                local,
                config=config,
                wide_odds_dict=wide_odds_dict,
                generated_at=generated_at,
                race_invest=race_invest,
            )
            if wrow:
                rows.append(wrow)
                race_invest += int(wrow["suggested_stake"])

        if mode in {"top2_pred", "top2_score"}:
            ordered = local.sort_values("_score_for_order", ascending=False).head(2)
            if len(ordered) < 2:
                continue
            h1 = int(ordered.iloc[0]["horse_num"])
            h2 = int(ordered.iloc[1]["horse_num"])
            p1 = float(ordered.iloc[0]["harville_prob"])
            p2 = float(ordered.iloc[1]["harville_prob"])
            quinella_prob_harville = _harville_quinella_probability(p1, p2)
            partner_row = ordered.iloc[1]
            pair_note = "top2_score_order_harville_pair_prob"
            iter_partners = [(partner_row, 1)]
        else:
            anchor = local.sort_values("harville_prob", ascending=False).head(1)
            if anchor.empty:
                continue
            h1 = int(anchor.iloc[0]["horse_num"])
            p1 = float(anchor.iloc[0]["harville_prob"])
            partners = local[local["horse_num"] != h1].sort_values("harville_prob", ascending=False).head(
                max(config.pair_top_n, config.wide_top_n)
            )
            pair_note = "harville_approx_from_normalized_win_prob"
            iter_partners = [(row, int((partners["horse_num"].tolist()).index(int(row["horse_num"]))) + 1) for _, row in partners.iterrows()]
            quinella_prob_harville = None  # harvilleモードではパートナーごとに計算

        for row, rank_pos in iter_partners:
            h2 = int(row["horse_num"])
            if mode not in {"top2_pred", "top2_score"}:
                p2 = float(row["harville_prob"])
                quinella_prob_harville = _harville_quinella_probability(p1, p2)

            # 馬連: Rank2スコアをブレンドしたペア確率を使う。
            # ワイド: 引き続きHarvilleのみ（Rank3はplace推奨で別途利用）。
            if _has_rank2:
                p2_rank2 = float(
                    local.loc[local["horse_num"] == h2, "_rank2_prob"].iloc[0]
                ) if h2 in local["horse_num"].values else p2
                p1_rank2 = float(
                    local.loc[local["horse_num"] == h1, "_rank2_prob"].iloc[0]
                ) if h1 in local["horse_num"].values else p1
                direct_pair_prob = float(np.clip(p1 * p2_rank2 + p2 * p1_rank2, 0.0, 1.0))
                quinella_prob_blended = float(np.clip(
                    (1.0 - RANK2_BLEND) * quinella_prob_harville + RANK2_BLEND * direct_pair_prob,
                    0.0, 1.0
                ))
            else:
                quinella_prob_blended = quinella_prob_harville

            for ticket_type, odds_col, limit_n, enabled in [
                ("馬連", config.quinella_odds_col, config.pair_top_n, config.quinella_bets_enabled),
                ("ワイド", config.wide_odds_col, config.wide_top_n, config.wide_bets_enabled),
            ]:
                if not enabled:
                    continue
                if ticket_type == "ワイド" and wide_sel == "divergence":
                    continue
                if rank_pos > int(limit_n):
                    continue
                # 馬連: Rank2ブレンド / ワイド: harville_wide_pair_prob
                if ticket_type == "馬連":
                    pair_prob = quinella_prob_blended
                else:
                    pair_prob = _harville_wide_probability(local, h1, h2)

                # O2/O3 速報オッズ dict が渡された場合はそちらを優先使用する。
                # キーは (race_id, min(h1,h2), max(h1,h2)) の文字列タプル。
                h1s, h2s = str(h1).zfill(2), str(h2).zfill(2)
                pair_key = (str(race_id), min(h1s, h2s), max(h1s, h2s))
                _odds_dict = quinella_odds_dict if ticket_type == "馬連" else wide_odds_dict
                if _odds_dict is not None and pair_key in _odds_dict:
                    odds_val = float(_odds_dict[pair_key])
                else:
                    odds_val = pd.to_numeric(row.get(odds_col, np.nan), errors="coerce")
                if pd.notna(odds_val):
                    effective_odds = max(float(odds_val) * (1.0 - config.base_slippage), 1.01)
                    ev = pair_prob * effective_odds
                    edge = ev - 1.0
                    min_edge = (
                        float(config.wide_min_edge)
                        if ticket_type == "ワイド"
                        else _race_min_edge
                    )
                    if ev > config.max_expected_value or edge < min_edge:
                        continue
                    frac = (
                        config.bet_unit / max(config.recommendation_bankroll, 1.0)
                        if effective_odds > config.max_odds_for_kelly
                        else _single_kelly_fraction(pair_prob, effective_odds)
                        * _kelly_fraction_for_odds(config, effective_odds)
                    )
                    stake = int(
                        min(
                            config.recommendation_bankroll * frac,
                            config.max_stake_per_bet,
                            config.max_invest_per_race - race_invest,
                        )
                        // config.bet_unit
                    ) * config.bet_unit
                else:
                    # オッズ不明のままベットするとEV計算不能でCLAUDE.mdの「EV > 1.05が必須」に違反する
                    continue
                if stake < config.bet_unit:
                    continue
                race_invest += stake
                rows.append(
                    {
                        "ticket_type": ticket_type,
                        "race_id": str(race_id),
                        "horse_num": h1,
                        "partner_horse_num": h2,
                        "ticket": f"{h1}-{h2}",
                        "pred_prob": pair_prob,
                        "odds_raw": float(odds_val) if pd.notna(odds_val) else np.nan,
                        "odds_effective": effective_odds,
                        "expected_value": ev,
                        "edge": edge,
                        "kelly_fraction": frac,
                        "suggested_stake": int(stake),
                        "is_executable": True,
                        "phase": "phase1_5",
                        "modeling_note": pair_note,
                        "odds_timestamp": (
                            local.iloc[0].get(config.odds_timestamp_col, None)
                            if config.save_snapshot_timestamps
                            else None
                        ),
                        "generated_at": generated_at if config.save_snapshot_timestamps else None,
                        "odds_source": config.odds_source,
                        "odds_cutoff_policy": config.odds_cutoff_policy,
                    }
                )
    return pd.DataFrame(rows)


def recommend_place_phase15(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    calibrator: Optional[ScoreCalibrator] = None,
    generated_at: Optional[str] = None,
) -> pd.DataFrame:
    """
    複勝推奨: Rank3スコアを3着以内確率として扱いEV/Kellyを計算する。
    place_odds カラムがある場合のみ実際のオッズでEV計算を行う。
    オッズなしの固定額フラット買いは過楽観なため、オッズ不明時は推奨をスキップする。
    """
    rank3_col = "pred_rank3"
    place_odds_col = getattr(config, "place_odds_col", "place_odds")

    if rank3_col not in pred_df.columns:
        return pd.DataFrame()
    if place_odds_col not in pred_df.columns:
        return pd.DataFrame()

    df = pred_df.copy()
    for c in [config.race_id_col, config.horse_col, rank3_col, place_odds_col]:
        if c not in df.columns:
            return pd.DataFrame()

    df[config.race_id_col] = df[config.race_id_col].astype(str)
    df[config.horse_col] = pd.to_numeric(df[config.horse_col], errors="coerce")
    df = df.dropna(subset=[config.horse_col]).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for race_id, race in df.groupby(config.race_id_col, sort=False):
        # フィールドサイズフィルタ（単勝・連複と同じ基準）
        _race_min_edge = config.min_edge
        if "n_horses" in race.columns:
            n_horses_val = pd.to_numeric(race["n_horses"], errors="coerce").dropna()
            if not n_horses_val.empty:
                _n_h = int(n_horses_val.iloc[0])
                if _n_h < config.min_field_size:
                    continue
                if _n_h >= config.large_field_threshold and config.large_field_extra_edge > 0.0:
                    _race_min_edge = config.min_edge + float(config.large_field_extra_edge)

        race_invest = 0
        local = race.copy()
        local["horse_num"] = local[config.horse_col].astype(int)

        # Rank3スコアを3着以内確率として補正する。
        # sum正規化のみでは各馬の確率総和が1になり過楽観（複勝は3頭が3着以内なので
        # 理論上の総和は 3/n_horses）。3/n_horses を乗じることで理論値に補正する。
        r3_scores = pd.to_numeric(local[rank3_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        r3_sum = float(r3_scores.sum())
        if r3_sum <= 0:
            continue
        n_horses = max(len(local), 1)
        local["place_prob"] = (r3_scores / r3_sum) * (3.0 / n_horses)

        # max_selections_per_race 件まで推奨（scoreの高い順）
        cands = local.sort_values("place_prob", ascending=False).head(config.max_selections_per_race)

        for _, row in cands.iterrows():
            place_prob = float(row["place_prob"])
            place_odds_val = pd.to_numeric(row.get(place_odds_col, np.nan), errors="coerce")
            if pd.isna(place_odds_val):
                # オッズ不明時はスキップ（固定額フラット買い禁止）
                continue

            effective_place_odds = max(float(place_odds_val) * (1.0 - config.base_slippage), 1.01)
            ev = place_prob * effective_place_odds
            edge = ev - 1.0

            if ev > config.max_expected_value or edge < _race_min_edge:
                continue

            if effective_place_odds > config.max_odds_for_kelly or not config.allow_kelly:
                frac = config.bet_unit / max(config.recommendation_bankroll, 1.0)
            else:
                frac = (
                    _single_kelly_fraction(place_prob, effective_place_odds)
                    * _kelly_fraction_for_odds(config, effective_place_odds)
                )

            stake = int(
                min(
                    config.recommendation_bankroll * frac,
                    config.max_stake_per_bet,
                    config.max_invest_per_race - race_invest,
                )
                // config.bet_unit
            ) * config.bet_unit

            if stake < config.bet_unit:
                continue

            race_invest += stake
            rows.append(
                {
                    "ticket_type": "複勝",
                    "race_id": str(race_id),
                    "horse_num": int(row["horse_num"]),
                    "ticket": str(int(row["horse_num"])),
                    "pred_prob": place_prob,
                    "odds_raw": float(place_odds_val),
                    "odds_effective": effective_place_odds,
                    "expected_value": ev,
                    "edge": edge,
                    "kelly_fraction": frac,
                    "suggested_stake": int(stake),
                    "is_executable": True,
                    "phase": "phase1_5",
                    "modeling_note": "rank3_softmax_prob_as_place_prob",
                    "odds_timestamp": (
                        local.iloc[0].get(config.odds_timestamp_col, None)
                        if config.save_snapshot_timestamps
                        else None
                    ),
                    "generated_at": generated_at if config.save_snapshot_timestamps else None,
                    "odds_source": config.odds_source,
                    "odds_cutoff_policy": config.odds_cutoff_policy,
                }
            )
    return pd.DataFrame(rows)


def recommend_pair_wide_phase2(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    generated_at: Optional[str] = None,
) -> pd.DataFrame:
    """
    Phase2: 連複専用確率モデルの実装用スタブ。
    誤った擬似確率/EVを出さないよう、未実装時は例外を送出する。
    """
    raise NotImplementedError(
        "Phase2 pair/wide model is not implemented. "
        "Please plug in a dedicated pair classifier before EV evaluation."
    )


def recommend_today(
    pred_df: pd.DataFrame,
    *,
    config: OnlineRecommendationConfig,
    calibrator: Optional[ScoreCalibrator] = None,
    generated_at: Optional[str] = None,
    quinella_odds_dict: Optional[dict] = None,
    wide_odds_dict: Optional[dict] = None,
    _flb_corrector=None,
) -> pd.DataFrame:
    now = generated_at or datetime.now(timezone.utc).isoformat()
    win = recommend_win_phase1(
        pred_df,
        config=config,
        calibrator=calibrator,
        generated_at=now,
        _flb_corrector=_flb_corrector,
    )
    if str(config.phase).lower() == "phase2":
        if not config.phase2_enabled:
            raise NotImplementedError(
                "Phase2 is disabled by config (phase2_enabled=False)."
            )
        pairwide = recommend_pair_wide_phase2(pred_df, config=config, generated_at=now)
        place = pd.DataFrame()
    elif str(config.phase).lower() in {"phase1_5", "phase1.5"}:
        pairwide = recommend_pair_wide_phase15(
            pred_df,
            config=config,
            calibrator=calibrator,
            generated_at=now,
            quinella_odds_dict=quinella_odds_dict,
            wide_odds_dict=wide_odds_dict,
        )
        place = (
            recommend_place_phase15(
                pred_df,
                config=config,
                calibrator=calibrator,
                generated_at=now,
            )
            if config.place_bets_enabled
            else pd.DataFrame()
        )
    else:
        pairwide = recommend_pair_wide_phase1(pred_df, config=config, generated_at=now)
        place = pd.DataFrame()
    out = pd.concat([win, pairwide, place], ignore_index=True, sort=False)
    if out.empty:
        warnings.warn("[recommend_today] no executable recommendations found.")
        return out
    out["race_id"] = out["race_id"].astype(str)

    if getattr(config, "portfolio_kelly_enabled", False):
        try:
            from strategy.src.portfolio_kelly import apply_portfolio_kelly_to_recommendations
        except ModuleNotFoundError:
            from portfolio_kelly import apply_portfolio_kelly_to_recommendations
        out = apply_portfolio_kelly_to_recommendations(
            out,
            pred_df,
            race_id_col=config.race_id_col,
            horse_col=config.horse_col,
            bankroll=float(config.recommendation_bankroll),
            bet_unit=int(config.bet_unit),
            max_stake_per_bet=int(config.max_stake_per_bet),
            max_invest_per_race=int(config.max_invest_per_race),
            max_total_fraction=float(config.max_total_fraction),
            max_single_fraction=float(config.max_single_fraction),
            fractional_kelly=float(config.fractional_kelly),
            mode=str(getattr(config, "portfolio_kelly_mode", "portfolio_kelly_fractional")),
            growth_ratio_min=float(getattr(config, "portfolio_growth_ratio_min", 0.5)),
            ind_cap_ratio=float(getattr(config, "portfolio_ind_cap_ratio", 0.85)),
            mc_samples=int(getattr(config, "portfolio_mc_samples", 500)),
            mc_seed=int(getattr(config, "portfolio_mc_seed", 42)),
        )

    # --- レース単位の合計投資額が max_invest_per_race を超える場合に比例縮小 ---
    # 各推奨関数内の race_invest は関数ごとに独立しているため、
    # 単勝+馬連+ワイド+複勝を合算すると最大 max_invest_per_race × N になりうる。
    # CLAUDE.md「1レースあたりのベット上限：資金の5%以内」を保証するため、ここで合算制御する。
    max_invest = int(getattr(config, "max_invest_per_race", 50_000))
    bet_unit = int(getattr(config, "bet_unit", 100))
    if "suggested_stake" in out.columns and max_invest > 0:
        race_totals = out.groupby("race_id")["suggested_stake"].transform("sum")
        over_mask = race_totals > max_invest
        if over_mask.any():
            # 超過レースのみ比例縮小してから bet_unit 単位に切り捨て
            scale = (max_invest / race_totals).clip(upper=1.0)
            scaled = (out.loc[over_mask, "suggested_stake"] * scale[over_mask])
            out.loc[over_mask, "suggested_stake"] = (
                (scaled // bet_unit) * bet_unit
            ).astype(int)
            # bet_unit 未満になったベットは実行不可にする
            too_small = out["suggested_stake"] < bet_unit
            out.loc[too_small, "is_executable"] = False
            out.loc[too_small, "suggested_stake"] = 0

    return out.sort_values(["race_id", "ticket_type", "expected_value"], ascending=[True, True, False], na_position="last")


def load_prediction_frame(
    csv_path: Path,
    *,
    parquet_path: Optional[Path] = None,
    prefer_parquet: bool = True,
    use_cudf: bool = False,
) -> pd.DataFrame:
    p_csv = Path(csv_path)
    p_parquet = Path(parquet_path) if parquet_path else p_csv.with_suffix(".parquet")
    if prefer_parquet and p_parquet.exists():
        if use_cudf:
            try:
                import cudf  # type: ignore

                return cudf.read_parquet(p_parquet).to_pandas()
            except Exception:
                pass
        return pd.read_parquet(p_parquet)
    return pd.read_csv(p_csv, low_memory=False)
