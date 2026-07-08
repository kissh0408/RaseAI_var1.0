"""
portfolio_backtest.py — 単勝 + ワイド合算ポートフォリオバックテスト
"""
from __future__ import annotations

DEFAULT_IND_CAP_RATIO = 0.85

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from strategy.src.betting_framework import (
        ProbabilityCalibrator,
        StrategyConfig,
        _apply_fixed_slippage,
        _single_kelly_fraction,
        compute_metrics,
    )
    from strategy.src.race_filters import filter_df_by_race_num
    from strategy.src.combo_backtest import _build_pair_candidates, load_combo_odds
    from strategy.src.ev_filters import (
        apply_bet_candidate_mask,
        attach_model_rank,
        ev_filter_config_from_mapping,
    )
    from strategy.src.portfolio_kelly import (
        PortfolioBet,
        optimize_fractional_kelly,
        optimize_full_kelly,
        race_rng,
        sample_return_matrix,
    )
except ModuleNotFoundError:
    from betting_framework import (
        ProbabilityCalibrator,
        StrategyConfig,
        _apply_fixed_slippage,
        _single_kelly_fraction,
        compute_metrics,
    )
    from race_filters import filter_df_by_race_num
    from combo_backtest import _build_pair_candidates, load_combo_odds
    from ev_filters import apply_bet_candidate_mask, attach_model_rank, ev_filter_config_from_mapping
    from portfolio_kelly import (
        PortfolioBet,
        optimize_fractional_kelly,
        optimize_full_kelly,
        race_rng,
        sample_return_matrix,
    )


@dataclass
class _Candidate:
    ticket_type: str
    horse_num: int
    partner_horse_num: Optional[int]
    prob: float
    odds_effective: float
    odds_raw: float
    edge: float


def _runtime_get(runtime: dict, key: str, default: Any) -> Any:
    v = runtime.get(key)
    return default if v is None else v


def _prepare_eval_frame(
    eval_df: pd.DataFrame,
    config: StrategyConfig,
    calibrator: Optional[ProbabilityCalibrator],
) -> pd.DataFrame:
    df = filter_df_by_race_num(
        eval_df,
        race_id_col=config.race_id_col,
        race_num_min=config.race_num_min,
        race_num_max=config.race_num_max,
    )
    sort_candidates = [
        c for c in ["date", "year", "month_day", config.race_id_col, config.horse_col] if c in df.columns
    ]
    if sort_candidates:
        df = df.sort_values(sort_candidates).reset_index(drop=True)

    if calibrator is not None:
        if calibrator.method == "isotonic":
            calibrator.params = dict(calibrator.params)
            calibrator.params["interpolation"] = config.isotonic_interpolation
        df["pred_prob"] = calibrator.transform(df[config.score_col]).clip(0.0, 1.0)
    else:
        df["pred_prob"] = df[config.score_col].clip(0.0, 1.0)

    if config.normalize_probs_in_race:
        grp_sum = df.groupby(config.race_id_col)["pred_prob"].transform("sum").clip(lower=1e-12)
        df["pred_prob"] = df["pred_prob"] / grp_sum

    df["pred_score_raw"] = pd.to_numeric(df[config.score_col], errors="coerce").fillna(0.0)
    df = attach_model_rank(
        df,
        race_id_col=config.race_id_col,
        prob_col="pred_prob",
        score_col="pred_score_raw",
    )
    df["effective_odds"] = _apply_fixed_slippage(df[config.odds_col], config.base_slippage)
    df["expected_value"] = df["pred_prob"] * df["effective_odds"]
    df["edge"] = df["expected_value"] - 1.0
    return df


def _collect_win_candidates(race: pd.DataFrame, config: StrategyConfig) -> List[_Candidate]:
    ev_cfg = ev_filter_config_from_mapping(config)
    mask = apply_bet_candidate_mask(race, ev_cfg, odds_col="effective_odds")
    cand = race.loc[mask].sort_values("edge", ascending=False).head(config.max_selections_per_race)
    out: List[_Candidate] = []
    for _, row in cand.iterrows():
        out.append(
            _Candidate(
                ticket_type="win",
                horse_num=int(row[config.horse_col]),
                partner_horse_num=None,
                prob=float(row["pred_prob"]),
                odds_effective=float(row["effective_odds"]),
                odds_raw=float(row[config.odds_col]),
                edge=float(row["edge"]),
            )
        )
    return out


def _collect_wide_candidates(
    race: pd.DataFrame,
    race_id: str,
    wide_dict: Dict[Tuple[str, int, int], float],
    *,
    wide_top_n: int,
    pair_top_n: int,
    rank2_blend: float,
    wide_min_edge: float,
    max_expected_value: float,
    base_slippage: float,
) -> List[_Candidate]:
    if "pred_rank2" not in race.columns:
        return []
    candidates = _build_pair_candidates(
        race,
        pair_top_n=pair_top_n,
        wide_top_n=wide_top_n,
        rank2_blend=rank2_blend,
    )
    out: List[_Candidate] = []
    for cand in candidates:
        if cand["rank"] > wide_top_n:
            continue
        h1, h2 = int(cand["horse_num_1"]), int(cand["horse_num_2"])
        key = (str(race_id), h1, h2)
        w_odds_raw = wide_dict.get(key)
        if w_odds_raw is None or not np.isfinite(w_odds_raw):
            continue
        w_prob = float(cand["wide_prob"])
        w_odds_eff = max(float(w_odds_raw) * (1.0 - base_slippage), 1.01)
        w_ev = w_prob * w_odds_eff
        w_edge = w_ev - 1.0
        if w_edge < wide_min_edge or w_ev > max_expected_value:
            continue
        out.append(
            _Candidate(
                ticket_type="wide",
                horse_num=h1,
                partner_horse_num=h2,
                prob=w_prob,
                odds_effective=w_odds_eff,
                odds_raw=float(w_odds_raw),
                edge=w_edge,
            )
        )
    return out


def _independent_fractions(
    candidates: List[_Candidate],
    config: StrategyConfig,
    bankroll: float,
) -> np.ndarray:
    raw: List[float] = []
    for c in candidates:
        fk = _single_kelly_fraction(c.prob, c.odds_effective) * config.fractional_kelly
        target = bankroll * fk
        capped = min(target, config.max_stake_per_bet)
        raw.append(capped / max(bankroll, 1.0))
    fr = np.array(raw, dtype=float)
    total_target = float(np.sum(fr)) * bankroll
    if total_target > config.max_invest_per_race and total_target > 0:
        scale = config.max_invest_per_race / total_target
        fr = fr * scale
    return np.clip(fr, 0.0, config.max_single_fraction)


def _portfolio_fractions(
    candidates: List[_Candidate],
    p_dict: dict[int, float],
    config: StrategyConfig,
    race_id: str,
    *,
    sizing_mode: str,
    mc_samples: int,
    mc_seed: int,
    growth_ratio_min: float,
) -> np.ndarray:
    bets = [
        PortfolioBet(
            kind=c.ticket_type,
            horse_a=c.horse_num,
            horse_b=c.partner_horse_num,
            prob=c.prob,
            odds=c.odds_effective,
        )
        for c in candidates
    ]
    rng = race_rng(mc_seed, race_id)
    r_mat = sample_return_matrix(bets, p_dict, mc_samples, rng)
    cap = config.max_total_fraction
    if sizing_mode == "portfolio_kelly_fractional":
        fr, _, _ = optimize_fractional_kelly(
            r_mat,
            total_cap=cap,
            max_single=config.max_single_fraction,
            growth_ratio_min=growth_ratio_min,
            fractional_kelly=config.fractional_kelly,
        )
        return fr
    return optimize_full_kelly(
        r_mat,
        total_cap=cap,
        max_single=config.max_single_fraction,
        fractional_kelly=config.fractional_kelly,
        odds=np.array([c.odds_effective for c in candidates], dtype=float),
        probs=np.array([c.prob for c in candidates], dtype=float),
    )


def _stakes_from_fractions(
    fractions: np.ndarray,
    bankroll: float,
    config: StrategyConfig,
    candidates: List[_Candidate],
    wide_max_stake: int,
) -> List[int]:
    stakes: List[int] = []
    race_invest = 0.0
    for frac, cand in zip(fractions, candidates):
        max_stake = config.max_stake_per_bet if cand.ticket_type == "win" else wide_max_stake
        target = bankroll * float(frac)
        remaining = max(config.max_invest_per_race - race_invest, 0.0)
        capped = min(target, max_stake, remaining)
        stake = int(capped // config.bet_unit) * config.bet_unit
        if stake < config.bet_unit:
            stakes.append(0)
            continue
        race_invest += stake
        stakes.append(stake)
    return stakes


def _hit_win(horse_num: int, rank_map: dict[int, int]) -> bool:
    return rank_map.get(horse_num, 99) == 1


def _hit_wide(h1: int, h2: int, rank_map: dict[int, int]) -> bool:
    return rank_map.get(h1, 99) <= 3 and rank_map.get(h2, 99) <= 3


def run_portfolio_backtest(
    eval_df: pd.DataFrame,
    config: StrategyConfig,
    odds_dir: Path,
    *,
    calibrator: Optional[ProbabilityCalibrator] = None,
    runtime: Optional[dict] = None,
    sizing_mode: str = "baseline",
    mc_samples: int = 500,
    mc_seed: int = 42,
    growth_ratio_min: float = 0.65,
    wide_bets_enabled: bool = True,
    ind_cap_ratio: float = DEFAULT_IND_CAP_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    runtime = runtime or {}
    df = _prepare_eval_frame(eval_df, config, calibrator)

    years = sorted(pd.to_numeric(df["valid_year"], errors="coerce").dropna().astype(int).unique().tolist())
    _, wide_df = load_combo_odds(odds_dir, years=years)
    wide_dict: Dict[Tuple[str, int, int], float] = {
        (str(int(r.race_id)), int(r.horse_num_1), int(r.horse_num_2)): float(r.wide_odds)
        for r in wide_df.itertuples(index=False)
    }

    wide_top_n = int(_runtime_get(runtime, "wide_top_n", 2))
    pair_top_n = int(_runtime_get(runtime, "pair_top_n", 2))
    rank2_blend = float(_runtime_get(runtime, "rank2_blend", 0.35))
    wide_min_edge = float(_runtime_get(runtime, "wide_min_edge", 0.05))
    wide_max_stake = int(_runtime_get(runtime, "wide_max_stake_per_bet", config.max_stake_per_bet))

    bankroll = float(config.initial_bankroll)
    bet_rows: List[Dict] = []
    race_rows: List[Dict] = []

    for race_id, race in df.groupby(config.race_id_col, sort=False):
        race = race.copy()
        race_id_str = str(int(race_id)) if str(race_id).isdigit() else str(race_id)
        before = bankroll

        rank_map = {int(r.horse_num): int(r.finish_rank) for r in race.itertuples(index=False)}
        p_dict = {
            int(r.horse_num): float(r.pred_prob)
            for r in race.itertuples(index=False)
        }

        win_cands = _collect_win_candidates(race, config)
        wide_cands: List[_Candidate] = []
        if wide_bets_enabled:
            wide_cands = _collect_wide_candidates(
                race,
                race_id_str,
                wide_dict,
                wide_top_n=wide_top_n,
                pair_top_n=pair_top_n,
                rank2_blend=rank2_blend,
                wide_min_edge=wide_min_edge,
                max_expected_value=config.max_expected_value,
                base_slippage=config.base_slippage,
            )
        candidates = win_cands + wide_cands
        if not candidates:
            race_rows.append(
                {
                    "race_id": race_id,
                    "n_bets": 0,
                    "invest": 0.0,
                    "return": 0.0,
                    "profit": 0.0,
                    "bankroll_before": before,
                    "bankroll_after": bankroll,
                }
            )
            continue

        if sizing_mode in {"baseline", "independent"}:
            fractions = _independent_fractions(candidates, config, before)
        elif sizing_mode in {"portfolio_kelly", "portfolio_kelly_fractional"}:
            fractions = _portfolio_fractions(
                candidates,
                p_dict,
                config,
                race_id_str,
                sizing_mode=sizing_mode,
                mc_samples=mc_samples,
                mc_seed=mc_seed,
                growth_ratio_min=growth_ratio_min,
            )
            ind = _independent_fractions(candidates, config, before)
            ind_sum = float(ind.sum())
            port_sum = float(fractions.sum())
            cap_total = ind_sum * float(ind_cap_ratio)
            if port_sum > 1e-12 and cap_total > 1e-12 and port_sum > cap_total:
                fractions = fractions * (cap_total / port_sum)
        else:
            raise ValueError(f"unknown sizing_mode: {sizing_mode}")

        stakes = _stakes_from_fractions(fractions, before, config, candidates, wide_max_stake)
        race_invest = 0.0
        race_return = 0.0
        race_bet_count = 0

        for cand, frac, stake in zip(candidates, fractions, stakes):
            if stake < config.bet_unit:
                continue
            if cand.ticket_type == "win":
                hit = _hit_win(cand.horse_num, rank_map)
            else:
                hit = _hit_wide(cand.horse_num, int(cand.partner_horse_num), rank_map)
            payout = float(stake * cand.odds_effective) if hit else 0.0
            profit = payout - stake
            race_invest += stake
            race_return += payout
            race_bet_count += 1
            bet_rows.append(
                {
                    "race_id": race_id,
                    "ticket_type": cand.ticket_type,
                    "horse_num": cand.horse_num,
                    "partner_horse_num": cand.partner_horse_num,
                    "stake": stake,
                    "odds_raw": cand.odds_raw,
                    "odds_effective": cand.odds_effective,
                    "pred_prob": cand.prob,
                    "edge": cand.edge,
                    "kelly_fraction": float(frac),
                    "hit": int(hit),
                    "payout": payout,
                    "profit": profit,
                    "valid_year": int(race["valid_year"].iloc[0]) if "valid_year" in race.columns else None,
                }
            )

        profit = race_return - race_invest
        bankroll += profit
        bankroll = max(bankroll, 0.0)
        race_rows.append(
            {
                "race_id": race_id,
                "n_bets": race_bet_count,
                "invest": race_invest,
                "return": race_return,
                "profit": profit,
                "bankroll_before": before,
                "bankroll_after": bankroll,
            }
        )

    bets_df = pd.DataFrame(bet_rows)
    races_df = pd.DataFrame(race_rows)
    combined = compute_metrics(
        bets_df if not bets_df.empty else pd.DataFrame(columns=["hit", "profit", "stake"]),
        races_df,
        initial_bankroll=config.initial_bankroll,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )

    win_df = bets_df[bets_df["ticket_type"] == "win"] if not bets_df.empty else bets_df
    wide_df_b = bets_df[bets_df["ticket_type"] == "wide"] if not bets_df.empty else bets_df
    win_races = races_df[races_df["race_id"].isin(win_df["race_id"].unique())] if not win_df.empty else races_df.iloc[0:0]
    wide_races = races_df[races_df["race_id"].isin(wide_df_b["race_id"].unique())] if not wide_df_b.empty else races_df.iloc[0:0]

    metrics = {
        "combined": combined,
        "win": compute_metrics(win_df, win_races, config.initial_bankroll) if not win_df.empty else {},
        "wide": compute_metrics(wide_df_b, wide_races, config.initial_bankroll) if not wide_df_b.empty else {},
        "sizing_mode": sizing_mode,
        "mc_samples": mc_samples,
        "mc_seed": mc_seed,
        "growth_ratio_min": growth_ratio_min if sizing_mode == "portfolio_kelly_fractional" else None,
    }
    return bets_df, races_df, metrics
