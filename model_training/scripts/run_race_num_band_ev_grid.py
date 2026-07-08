"""
run_race_num_band_ev_grid.py — 1-7R 帯別 edge 厳格化 grid（学習期間選定 → 2024/2025 確認）

1-12R を対象に、race_num <= early_race_max のレースだけ extra min_edge を加算。
8-12R は本番 dynamic_edge と同一。閾値選定は valid_year <= 2023 のみ（後出しじゃんけん防止）。

比較 baseline:
  - production_8_12: race_num 8-12 のみ（本番）
  - open_1_12: race_num 1-12, extra_edge=0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
STRATEGY_CFG = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
OUT_JSON = PROJECT_ROOT / "model_training" / "data" / "03_train" / "race_num_band_ev_grid_report.json"

TRAIN_YEAR_MAX = 2023
EARLY_RACE_MAX = 7
EXTRA_EDGE_GRID = (0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15)


def _load_eval() -> pd.DataFrame:
    from strategy.src.betting_framework import load_evaluation
    from strategy.src.race_filters import attach_race_num

    df = load_evaluation(SPECv2_OOF)
    if "race_num" not in df.columns:
        df = attach_race_num(df)
    return df


def _load_calibrator():
    from main.pipeline.strategy_pipeline import resolve_strategy_calibration_path
    from strategy.src.betting_framework import ProbabilityCalibrator

    cal_path = resolve_strategy_calibration_path(PROJECT_ROOT)
    if cal_path.is_file():
        return ProbabilityCalibrator.from_json(cal_path)
    return None


def _build_config(runtime: dict):
    from main.pipeline.strategy_pipeline import strategy_config_from_runtime

    return strategy_config_from_runtime(runtime)


def _runtime(**overrides) -> dict:
    cfg = json.loads(STRATEGY_CFG.read_text(encoding="utf-8"))
    cfg.update(overrides)
    return cfg


def _ev_cfg_with_extra(strategy_cfg, extra_edge: float):
    """dynamic_edge バンド全体に extra を加算した EvFilterConfig を返す。"""
    from strategy.src.ev_filters import ev_filter_config_from_mapping

    ev = ev_filter_config_from_mapping(strategy_cfg)
    patched = copy.copy(ev)
    patched.min_edge = float(ev.min_edge) + extra_edge
    if ev.dynamic_edge_enabled and ev.dynamic_edge_bands:
        patched.dynamic_edge_bands = [
            {**b, "min_edge": float(b.get("min_edge", ev.min_edge)) + extra_edge}
            for b in ev.dynamic_edge_bands
        ]
    return patched


def run_band_win_backtest(
    eval_df: pd.DataFrame,
    config,
    calibrator,
    *,
    early_race_max: int,
    early_extra_edge: float,
    year: int | None = None,
) -> dict[str, Any]:
    """1-12R 対象。early 帯のみ extra edge を適用した単勝 Kelly バックテスト。"""
    from strategy.src.betting_framework import (
        _apply_dynamic_pool_odds,
        _apply_fixed_slippage,
        _single_kelly_fraction,
        compute_metrics,
        simultaneous_kelly_fractions,
        simultaneous_kelly_fractions_scipy,
    )
    from strategy.src.ev_filters import (
        apply_bet_candidate_mask,
        attach_model_rank,
        ev_filter_config_from_mapping,
        post_slippage_edge_gate,
        should_apply_post_slippage_gate,
    )
    from strategy.src.race_filters import filter_df_by_race_num

    df = eval_df.copy()
    if year is not None:
        df = df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()
    if df.empty:
        return {"error": "empty", "year": year}

    df = filter_df_by_race_num(
        df,
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

    base_ev_cfg = ev_filter_config_from_mapping(config)
    race_num_map = df.groupby(config.race_id_col)["race_num"].first()

    bankroll = float(config.initial_bankroll)
    bet_rows: list[dict] = []
    race_rows: list[dict] = []

    for race_id, race in df.groupby(config.race_id_col, sort=False):
        before_bankroll = bankroll
        rn = int(pd.to_numeric(race_num_map.get(race_id), errors="coerce") or 0)
        extra = early_extra_edge if rn <= early_race_max else 0.0
        race_ev_cfg = _ev_cfg_with_extra(config, extra) if extra > 0 else base_ev_cfg

        race_scope = race.copy()
        if "n_horses" in race_scope.columns:
            n_h = pd.to_numeric(race_scope["n_horses"], errors="coerce").dropna()
            if not n_h.empty:
                nh = int(n_h.iloc[0])
                if nh < config.min_field_size:
                    race_rows.append(
                        {
                            "race_id": race_id,
                            "race_num": rn,
                            "n_bets": 0,
                            "invest": 0.0,
                            "return": 0.0,
                            "profit": 0.0,
                            "bankroll_before": before_bankroll,
                            "bankroll_after": bankroll,
                        }
                    )
                    continue
                if nh >= config.large_field_threshold and config.large_field_extra_edge > 0:
                    race_ev_cfg = copy.copy(race_ev_cfg)
                    race_ev_cfg.min_edge += float(config.large_field_extra_edge)
                    if race_ev_cfg.dynamic_edge_enabled and race_ev_cfg.dynamic_edge_bands:
                        race_ev_cfg.dynamic_edge_bands = [
                            {
                                **b,
                                "min_edge": float(b.get("min_edge", race_ev_cfg.min_edge))
                                + float(config.large_field_extra_edge),
                            }
                            for b in race_ev_cfg.dynamic_edge_bands
                        ]

        mask = apply_bet_candidate_mask(race_scope, race_ev_cfg, odds_col="effective_odds")
        cand = race_scope.loc[mask].sort_values("edge", ascending=False).head(config.max_selections_per_race)
        if cand.empty:
            race_rows.append(
                {
                    "race_id": race_id,
                    "race_num": rn,
                    "n_bets": 0,
                    "invest": 0.0,
                    "return": 0.0,
                    "profit": 0.0,
                    "bankroll_before": before_bankroll,
                    "bankroll_after": bankroll,
                }
            )
            continue

        if config.sizing_mode == "kelly_simultaneous":
            if config.simultaneous_optimizer == "scipy":
                fractions = simultaneous_kelly_fractions_scipy(
                    probs=cand["pred_prob"].to_numpy(),
                    odds=cand["effective_odds"].to_numpy(),
                    fractional_kelly=config.fractional_kelly,
                    total_cap=config.max_total_fraction,
                )
            else:
                fractions = simultaneous_kelly_fractions(
                    probs=cand["pred_prob"].to_numpy(),
                    odds=cand["effective_odds"].to_numpy(),
                    fractional_kelly=config.fractional_kelly,
                    total_cap=config.max_total_fraction,
                )
            fractions = np.clip(fractions, 0.0, config.max_single_fraction)
            from strategy.src.strategy_engine import _project_to_capped_simplex

            fractions = _project_to_capped_simplex(fractions, config.max_total_fraction)
        else:
            raise ValueError(f"unsupported sizing_mode: {config.sizing_mode}")

        race_invest = 0.0
        race_return = 0.0
        race_bet_count = 0
        winners = set(race.loc[race[config.rank_col] == 1, config.horse_col].astype(int).tolist())

        for (_, row), frac in zip(cand.iterrows(), fractions):
            target_bet = before_bankroll * float(frac)
            remaining_budget = max(config.max_invest_per_race - race_invest, 0.0)
            capped_target = min(target_bet, config.max_stake_per_bet, remaining_budget)
            stake = int(capped_target // config.bet_unit) * config.bet_unit
            if stake < config.bet_unit:
                continue

            exec_odds = float(row["effective_odds"])
            if should_apply_post_slippage_gate(
                race_ev_cfg, enforce_post_slippage_edge=config.enforce_post_slippage_edge
            ):
                if not post_slippage_edge_gate(float(row["pred_prob"]), exec_odds, race_ev_cfg):
                    continue

            horse_num = int(row[config.horse_col])
            hit = horse_num in winners
            payout = float(stake * exec_odds) if hit else 0.0
            profit = payout - stake
            race_invest += stake
            race_return += payout
            race_bet_count += 1
            bet_rows.append(
                {
                    "race_id": race_id,
                    "race_num": rn,
                    "horse_num": horse_num,
                    "stake": stake,
                    "edge": float(row["edge"]),
                    "expected_value": float(row["expected_value"]),
                    "hit": int(hit),
                    "profit": profit,
                    "early_band": rn <= early_race_max,
                }
            )

        race_profit = race_return - race_invest
        bankroll += race_profit
        bankroll = max(bankroll, 0.0)
        race_rows.append(
            {
                "race_id": race_id,
                "race_num": rn,
                "n_bets": race_bet_count,
                "invest": race_invest,
                "return": race_return,
                "profit": race_profit,
                "bankroll_before": before_bankroll,
                "bankroll_after": bankroll,
            }
        )

    bets_df = pd.DataFrame(bet_rows)
    race_df = pd.DataFrame(race_rows)
    metrics = compute_metrics(
        bets_df=bets_df,
        race_df=race_df,
        initial_bankroll=config.initial_bankroll,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    early_bets = int(bets_df["early_band"].sum()) if not bets_df.empty and "early_band" in bets_df.columns else 0
    return {
        "year": year,
        "n_bets": int(metrics.get("n_bets", 0)),
        "n_bets_early": early_bets,
        "n_bets_late": int(metrics.get("n_bets", 0)) - early_bets,
        "roi": float(metrics.get("return_multiple", 0)),
        "mdd": float(metrics.get("max_drawdown_rate", 0)),
        "sharpe": float(metrics.get("sharpe", 0)),
        "invest": float(metrics.get("invest", 0)),
    }


def _eval_period(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is None:
        return df[pd.to_numeric(df["valid_year"], errors="coerce") <= TRAIN_YEAR_MAX].copy()
    return df[pd.to_numeric(df["valid_year"], errors="coerce") == year].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    eval_df = _load_eval()
    calibrator = _load_calibrator()

    scenarios: list[dict] = []

    # Baselines
    baseline_defs = [
        ("production_8_12", {"race_num_min": 8, "race_num_max": 12}, None),
        ("open_1_12", {"race_num_min": 1, "race_num_max": 12}, 0.0),
    ]
    for name, rnum, extra in baseline_defs:
        rt = _runtime(**rnum)
        cfg = _build_config(rt)
        row: dict[str, Any] = {
            "scenario": name,
            "race_num_min": rnum["race_num_min"],
            "race_num_max": rnum["race_num_max"],
            "early_extra_edge": extra,
            "periods": {},
        }
        for period_name, year in [("train", None), ("y2024", 2024), ("y2025", 2025)]:
            sub = _eval_period(eval_df, year)
            extra_val = extra if extra is not None else 0.0
            row["periods"][period_name] = run_band_win_backtest(
                sub,
                cfg,
                calibrator,
                early_race_max=EARLY_RACE_MAX,
                early_extra_edge=extra_val,
                year=year if period_name != "train" else None,
            )
        scenarios.append(row)

    # Grid: 1-12R + early extra edge
    for extra in EXTRA_EDGE_GRID:
        if extra == 0.0:
            continue  # covered by open_1_12
        rt = _runtime(race_num_min=1, race_num_max=12)
        cfg = _build_config(rt)
        row = {
            "scenario": f"band_1_12_extra_{extra:.2f}",
            "race_num_min": 1,
            "race_num_max": 12,
            "early_extra_edge": extra,
            "periods": {},
        }
        for period_name, year in [("train", None), ("y2024", 2024), ("y2025", 2025)]:
            sub = _eval_period(eval_df, year)
            row["periods"][period_name] = run_band_win_backtest(
                sub,
                cfg,
                calibrator,
                early_race_max=EARLY_RACE_MAX,
                early_extra_edge=extra,
                year=year if period_name != "train" else None,
            )
        scenarios.append(row)

    # Select best on train only
    candidates = [s for s in scenarios if s["scenario"].startswith("band_")]
    eligible = []
    for s in candidates:
        tr = s["periods"].get("train", {})
        if tr.get("n_bets", 0) >= 500 and tr.get("roi", 0) >= 1.05:
            eligible.append(s)
    best = None
    if eligible:
        best = max(eligible, key=lambda s: s["periods"]["train"]["mdd"])

    prod = next(s for s in scenarios if s["scenario"] == "production_8_12")
    open_ = next(s for s in scenarios if s["scenario"] == "open_1_12")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_year_max": TRAIN_YEAR_MAX,
        "early_race_max": EARLY_RACE_MAX,
        "extra_edge_grid": list(EXTRA_EDGE_GRID),
        "calibrator": "specv2",
        "scenarios": scenarios,
        "selection": {
            "rule": "train valid_year<=2023, n_bets>=500, roi>=1.05, max mdd",
            "best_scenario": best["scenario"] if best else None,
            "best_train": best["periods"]["train"] if best else None,
            "best_y2025": best["periods"]["y2025"] if best else None,
        },
        "comparison_vs_production": {
            "open_1_12_y2025_mdd": open_["periods"]["y2025"].get("mdd"),
            "production_y2025_mdd": prod["periods"]["y2025"].get("mdd"),
            "open_1_12_y2025_roi": open_["periods"]["y2025"].get("roi"),
            "production_y2025_roi": prod["periods"]["y2025"].get("roi"),
        },
    }
    if best:
        report["comparison_vs_production"]["best_grid_y2025_mdd"] = best["periods"]["y2025"].get("mdd")
        report["comparison_vs_production"]["best_grid_y2025_roi"] = best["periods"]["y2025"].get("roi")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")
    print(f"Best (train): {report['selection']['best_scenario']}")
    if best:
        bt = best["periods"]["train"]
        b5 = best["periods"]["y2025"]
        print(
            f"  train ROI={bt['roi']:.1%} MDD={bt['mdd']:.1%} n={bt['n_bets']} "
            f"(early={bt.get('n_bets_early', 0)})"
        )
        print(f"  2025  ROI={b5['roi']:.1%} MDD={b5['mdd']:.1%} n={b5['n_bets']}")
    p5 = prod["periods"]["y2025"]
    o5 = open_["periods"]["y2025"]
    print(f"production 8-12  2025: ROI={p5['roi']:.1%} MDD={p5['mdd']:.1%} n={p5['n_bets']}")
    print(f"open 1-12        2025: ROI={o5['roi']:.1%} MDD={o5['mdd']:.1%} n={o5['n_bets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
