"""Today bet recommendations from fused probabilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from betting.src.ev_engine import apply_ev_filters, enrich_predictions
from betting.src.kelly_sizer import apply_kelly_sizing, apply_mutually_exclusive_decay


def run_recommendations(
    fused_df: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    odds_timestamp: str | None = None,
    bet_types: list[str] | None = None,
) -> pd.DataFrame:
    """Build L3 recommendation rows from live fused probabilities."""
    ts = odds_timestamp or datetime.now(timezone.utc).isoformat()
    bet_types = bet_types or cfg.get("bet_types", ["win"])
    rows: list[dict] = []

    for bet_type in bet_types:
        work = fused_df.copy()
        prob_col = "p_win" if bet_type == "win" else "p_place"
        work["model_prob"] = work[prob_col]
        work = enrich_predictions(work, ev_haircut=cfg.get("ev_haircut", 0.95))
        mask = apply_ev_filters(
            work,
            ev_col="ev_adjusted",
            ev_threshold=cfg.get("ev_threshold", 1.05),
            min_odds=cfg.get("min_odds", 2.0),
            max_odds=cfg.get("max_odds", 50.0),
            min_model_prob=cfg.get("min_model_prob", 0.05),
        )
        picks = work.loc[mask].copy()
        if picks.empty:
            continue
        picks = apply_kelly_sizing(
            picks,
            bankroll=cfg.get("bankroll", 100000),
            kelly_frac=cfg.get("kelly_fraction", 0.08),
            max_bet_ratio=cfg.get("max_bet_ratio", 0.05),
        )
        picks = apply_mutually_exclusive_decay(picks)
        picks = picks.sort_values("ev_adjusted", ascending=False)
        picks = picks.groupby("race_id", sort=False).head(cfg.get("max_picks_per_race", 2))

        for _, row in picks.iterrows():
            rows.append(
                {
                    "race_id": str(row["race_id"]),
                    "bet_type": bet_type,
                    "selection": int(row.get("horse_number", row.get("horse_num", 0))),
                    "ev": float(row["ev_adjusted"]),
                    "kelly_fraction": float(row["kelly_ratio"]),
                    "stake": float(row["kelly_bet_yen"]),
                    "odds_used": float(row["odds"]),
                    "odds_timestamp": ts,
                }
            )

    return pd.DataFrame(rows)


def save_recommendations(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "today_recommendations.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
