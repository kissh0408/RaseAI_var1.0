"""Flat top-1 win betting: loss-minimization operation (not EV-threshold based).

Rationale (see docs/specs/2026-07-10-goal-redefinition-and-loss-minimization.md and
docs/specs/2026-07-10-loss-minimization-implementation-spec.md): no positive EV exists
in the current 4-layer architecture (Benter rebuild, 2026-07-08〜10, 10 independent
pre-registered checks all failed). The only consistent positive signal is that the
model's rank-1 pick (pure_score_z max, no market info) loses *less* to the takeout
than the market favorite does, when both are flat-bet at 100-yen/unit equivalent
(fold2 OOS: model 81.89% ROI vs favorite 77.89% ROI). This module implements that
"lose less than the market" operation. It never claims profitability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DISCLAIMER: str = (
    "本推奨は市場に対する相対的な損失最小化を目的とし、黒字化を保証するものではない"
    "（fold2 OOS実測: ROI 81.89%、元本の約18%の期待損失）"
)

DEFAULT_SCORE_COL = "pure_score_z"
DEFAULT_STAKE_FRACTION = 0.001
DEFAULT_ROUNDING_YEN = 100
# Minimum bankroll the operation may be run with. Below this, the flat stake
# (bankroll * stake_fraction, floored to rounding_yen) collapses to below the JRA
# minimum purchase unit for the frozen stake_fraction=0.001 (100 / 0.001 = 100,000).
# See docs/specs/2026-07-10-loss-minimization-implementation-spec.md §1.2.
DEFAULT_MIN_BANKROLL_YEN = 100_000

SKIP_REASON_BELOW_MIN = "odds_below_min"
SKIP_REASON_ABOVE_MAX = "odds_above_max"
SKIP_REASON_MISSING = "odds_missing"


def _loss_min_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("loss_min", {}) if cfg else {}


def select_top1_bets(
    df: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    race_id_col: str = "race_id",
    horse_num_col: str = "horse_num",
    odds_col: str = "odds",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the model's rank-1 horse (max score_col) per race, tie-break by horse_num asc.

    Applies min_odds/max_odds exclusion (no fallback to rank-2; a race with an
    out-of-range or missing odds pick is skipped entirely, per spec).

    Returns
    -------
    (picks, skipped) : both pd.DataFrame
        picks has one row per non-skipped race (the selected horse).
        skipped has one row per skipped race with columns
        race_id, top1_horse_num, reason.
    """
    loss_min_cfg = _loss_min_cfg(cfg)
    score_col = loss_min_cfg.get("score_col", DEFAULT_SCORE_COL)
    min_odds = float(cfg.get("min_odds", 2.0))
    max_odds = float(cfg.get("max_odds", 50.0))

    if horse_num_col not in df.columns and "horse_number" in df.columns:
        df = df.rename(columns={"horse_number": horse_num_col})

    work = df.copy()
    work[race_id_col] = work[race_id_col].astype(str)
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_horse_num"] = pd.to_numeric(work[horse_num_col], errors="coerce")

    # Deterministic selection: highest score first, then lowest horse_num as tiebreak.
    work = work.sort_values(
        [race_id_col, "_score", "_horse_num"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    top1 = work.groupby(race_id_col, sort=False, as_index=False).head(1).copy()

    odds = pd.to_numeric(top1[odds_col], errors="coerce")
    missing_mask = odds.isna()
    below_mask = (~missing_mask) & (odds < min_odds)
    above_mask = (~missing_mask) & (odds > max_odds)
    ok_mask = ~(missing_mask | below_mask | above_mask)

    picks = top1.loc[ok_mask].drop(columns=["_score"]).copy()

    skipped_rows = []
    for mask, reason in (
        (missing_mask, SKIP_REASON_MISSING),
        (below_mask, SKIP_REASON_BELOW_MIN),
        (above_mask, SKIP_REASON_ABOVE_MAX),
    ):
        sub = top1.loc[mask]
        for _, row in sub.iterrows():
            skipped_rows.append(
                {
                    "race_id": row[race_id_col],
                    "top1_horse_num": int(row["_horse_num"]) if pd.notna(row["_horse_num"]) else None,
                    "reason": reason,
                }
            )
    skipped = pd.DataFrame(skipped_rows, columns=["race_id", "top1_horse_num", "reason"])
    return picks, skipped


def apply_flat_sizing(
    picks: pd.DataFrame,
    *,
    bankroll: float,
    stake_fraction: float,
    rounding_yen: int = DEFAULT_ROUNDING_YEN,
    min_bankroll_yen: float = DEFAULT_MIN_BANKROLL_YEN,
) -> pd.DataFrame:
    """Attach a flat stake column: floor(bankroll * stake_fraction / rounding_yen) * rounding_yen.

    `bankroll` must always be the operation's *initial* bankroll at the moment sizing
    was frozen — never a running/current-bankroll figure that changes bet-to-bet. Every
    call with the same (bankroll, stake_fraction, rounding_yen) yields the same stake,
    which is what makes this a flat (constant-stake) strategy rather than a proportional
    (percent-of-current-bankroll) one (spec §1.2 R2: "運用開始時の初期bankroll × f の定額").

    Raises
    ------
    ValueError
        If `bankroll` is below `min_bankroll_yen` (default 100,000 JPY — the minimum
        operating bankroll for the frozen stake_fraction=0.001, since
        rounding_yen / stake_fraction = 100,000), or if the resulting flat stake would
        floor below the JRA minimum purchase unit (`rounding_yen`, normally 100 JPY).
        Silently placing a 0-yen (or sub-unit) bet is not an option; the operation must
        either use a larger bankroll or not run at all.
    """
    if bankroll < min_bankroll_yen:
        raise ValueError(
            f"bankroll={bankroll} is below min_bankroll_yen={min_bankroll_yen}: "
            "the flat stake would collapse below the JRA minimum purchase unit "
            "for the frozen stake_fraction. Increase the operating bankroll; "
            "do not lower stake_fraction to work around this (Rule 3: frozen value)."
        )

    out = picks.copy()
    raw = float(bankroll) * float(stake_fraction)
    stake = float(np.floor(raw / rounding_yen) * rounding_yen)

    if stake < rounding_yen:
        min_viable_bankroll = rounding_yen / float(stake_fraction)
        raise ValueError(
            f"computed stake={stake} yen is below the minimum purchase unit "
            f"({rounding_yen} yen) for bankroll={bankroll}, stake_fraction={stake_fraction}. "
            f"Minimum viable bankroll for this stake_fraction is {min_viable_bankroll} yen "
            "(see min_bankroll_yen); refusing to place a sub-unit bet."
        )

    out["stake"] = stake
    out["stake_fraction"] = float(stake_fraction)
    return out


def settle_win_bets(
    picks: pd.DataFrame,
    *,
    finish_rank_col: str = "finish_rank",
    odds_col: str = "odds",
    stake_col: str = "stake",
) -> pd.DataFrame:
    """Backtest settlement: payout = stake * odds if finish_rank==1 else 0; adds pnl."""
    out = picks.copy()
    win = pd.to_numeric(out[finish_rank_col], errors="coerce").astype("Int64") == 1
    stake = pd.to_numeric(out[stake_col], errors="coerce").astype(float)
    odds = pd.to_numeric(out[odds_col], errors="coerce").astype(float)
    payout = np.where(win.fillna(False), stake * odds, 0.0)
    out["win"] = win.fillna(False)
    out["payout"] = payout
    out["pnl"] = payout - stake
    return out


def run_loss_min_recommendations(
    rank_preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    odds_timestamp: str,
    bankroll: float,
    score_col: str | None = None,
    odds_source: str = "race_se_csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Live/today entry point: build recommendation rows per §1.4 CSV schema.

    Parameters
    ----------
    rank_preds_df : DataFrame with race_id, horse_num (or horse_number), and
        the configured score_col (default pure_score_z) already attached.
    odds_df : DataFrame with race_id, horse_num, odds.

    Returns
    -------
    (recs, skipped) : recommendation rows (§1.4 schema) and skipped-race rows.
    """
    loss_min_cfg = _loss_min_cfg(cfg)
    resolved_score_col = score_col or loss_min_cfg.get("score_col", DEFAULT_SCORE_COL)
    stake_fraction = float(loss_min_cfg.get("stake_fraction", DEFAULT_STAKE_FRACTION))
    rounding_yen = int(loss_min_cfg.get("stake_rounding_yen", DEFAULT_ROUNDING_YEN))

    rp = rank_preds_df.copy()
    rp["race_id"] = rp["race_id"].astype(str)
    if "horse_num" not in rp.columns and "horse_number" in rp.columns:
        rp["horse_num"] = pd.to_numeric(rp["horse_number"], errors="coerce")

    od = odds_df.copy()
    od["race_id"] = od["race_id"].astype(str)
    od["horse_num"] = pd.to_numeric(od["horse_num"], errors="coerce")

    merged = rp.merge(od[["race_id", "horse_num", "odds"]], on=["race_id", "horse_num"], how="left")

    picks, skipped = select_top1_bets(merged, cfg=cfg)
    picks = apply_flat_sizing(
        picks, bankroll=bankroll, stake_fraction=stake_fraction, rounding_yen=rounding_yen
    )

    rec_rows = []
    for _, row in picks.iterrows():
        score_val = float(row[resolved_score_col]) if resolved_score_col in row.index else None
        rec_rows.append(
            {
                "race_id": str(row["race_id"]),
                "bet_type": "win",
                "selection": int(row["_horse_num"]),
                "pure_score_z": score_val,
                "odds_used": float(row["odds"]),
                "odds_source": odds_source,
                "stake": float(row["stake"]),
                "stake_fraction": float(row["stake_fraction"]),
                "mode": "loss_min_top1",
                "odds_timestamp": odds_timestamp,
                "note": DISCLAIMER,
            }
        )
    recs = pd.DataFrame(
        rec_rows,
        columns=[
            "race_id",
            "bet_type",
            "selection",
            "pure_score_z",
            "odds_used",
            "odds_source",
            "stake",
            "stake_fraction",
            "mode",
            "odds_timestamp",
            "note",
        ],
    )

    print(DISCLAIMER)
    return recs, skipped


def save_loss_min_recommendations(recs: pd.DataFrame, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "today_recommendations.csv"
    recs.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_skipped_races(skipped: pd.DataFrame, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skipped_races.csv"
    skipped.to_csv(path, index=False, encoding="utf-8-sig")
    return path
