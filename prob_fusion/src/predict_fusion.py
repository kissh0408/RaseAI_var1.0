"""Inference and batch probability fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prob_fusion.src.fit_fusion import fusion_probs
from prob_fusion.src.market_prob import attach_market_q
from prob_fusion.src.place_prob import place_prob_from_p_win


def load_fusion_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (ROOT / "prob_fusion" / "config" / "fusion_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_race_probs(
    z: np.ndarray,
    ln_q: np.ndarray,
    *,
    alpha: float,
    beta: float,
    lam2: float = 1.0,
    lam3: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_win, p_place) for one race."""
    p_win = fusion_probs(z, ln_q, alpha, beta)
    p_place = place_prob_from_p_win(p_win, lam2, lam3)
    return p_win, p_place


def fuse_dataframe(
    df: pd.DataFrame,
    *,
    alpha: float,
    beta: float,
    lam2: float = 1.0,
    lam3: float = 1.0,
    odds_col: str = "odds",
    q_method: str = "proportional",
    q_power: float = 0.81,
    model_version: str = "benter_v1",
) -> pd.DataFrame:
    """Apply fusion to full dataframe; requires pure_score_z and odds."""
    work = attach_market_q(
        df,
        odds_col=odds_col,
        method=q_method,
        power=q_power,
    )
    rows: list[dict] = []
    for race_id, grp in work.groupby("race_id"):
        z = grp["pure_score_z"].astype(float).values
        ln_q = grp["ln_market_q"].astype(float).values
        p_win, p_place = predict_race_probs(z, ln_q, alpha=alpha, beta=beta, lam2=lam2, lam3=lam3)
        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append(
                {
                    "race_id": str(race_id),
                    "horse_number": int(row.get("horse_num", row.get("horse_number", i + 1))),
                    "p_win": float(p_win[i]),
                    "p_place": float(p_place[i]),
                    "alpha": alpha,
                    "beta": beta,
                    "model_version": model_version,
                }
            )
    return pd.DataFrame(rows)


def fuse_from_scores_parquet(
    scores_path: Path,
    odds_df: pd.DataFrame,
    params: dict[str, float],
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Join L1 scores parquet with odds and fuse."""
    cfg = config or load_fusion_config()
    scores = pd.read_parquet(scores_path)
    scores["race_id"] = scores["race_id"].astype(str)
    if "horse_number" not in scores.columns:
        scores["horse_number"] = scores.get("horse_num", scores.get("ketto_num"))
    odds = odds_df.copy()
    odds["race_id"] = odds["race_id"].astype(str)
    merge_col = "horse_num" if "horse_num" in odds.columns else "horse_number"
    if merge_col not in odds.columns and "horse_number" in odds.columns:
        merge_col = "horse_number"
    scores["_join_h"] = scores["horse_number"].astype(int)
    odds["_join_h"] = odds[merge_col].astype(int)
    merged = scores.merge(
        odds[["race_id", "_join_h", "odds", "finish_rank"]],
        on=["race_id", "_join_h"],
        how="inner",
    )
    merged = merged.rename(columns={"_join_h": "horse_num"})
    return fuse_dataframe(
        merged,
        alpha=params["alpha"],
        beta=params["beta"],
        lam2=params.get("lam2", cfg.get("stern_lam2", 1.0)),
        lam3=params.get("lam3", cfg.get("stern_lam3", 1.0)),
        q_method=cfg.get("q_method", "proportional"),
        q_power=cfg.get("q_power", 0.81),
        model_version=cfg.get("version", "benter_v1"),
    )
