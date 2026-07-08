"""Attach win odds to feature/score frames."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import load_config, resolve_project_path  # noqa: E402


def _load_combined_win_odds_lookup() -> dict[str, dict[int, float]]:
    """Load win odds via simulate_ev helpers (WinOdds CSV + SE parquet fallback)."""
    from simulate_ev import _build_win_odds_lookup, _load_win_odds_for_simulation

    cfg = load_config()
    odds_dir = resolve_project_path("common/data/output/odds")
    years = list(range(2018, 2027))
    try:
        return _load_win_odds_for_simulation(cfg, years, odds_dir)
    except Exception:
        raw = _build_win_odds_lookup(years, odds_dir)
        flat: dict[str, dict[int, float]] = {}
        for rid, horses in raw.items():
            flat[rid] = {
                h: float(v[0])
                for h, v in horses.items()
                if v[0] is not None and float(v[0]) > 0
            }
        return flat


def attach_odds_from_lookup(
    df: pd.DataFrame,
    lookup: dict[str, dict[int, float]],
    *,
    race_id_col: str = "race_id",
    horse_num_col: str = "horse_num",
) -> pd.DataFrame:
    out = df.copy()
    out[race_id_col] = out[race_id_col].astype(str)
    if horse_num_col not in out.columns and "horse_number" in out.columns:
        out[horse_num_col] = out["horse_number"]

    def _row_odds(row: pd.Series) -> float | None:
        rid = str(row[race_id_col])
        h = int(row[horse_num_col])
        return lookup.get(rid, {}).get(h)

    out["odds"] = out.apply(_row_odds, axis=1)
    return out


def attach_odds_from_se_parquet(df: pd.DataFrame, se_path: Path | None = None) -> pd.DataFrame:
    """Attach odds using combined WinOdds/SE lookup."""
    lookup = _load_combined_win_odds_lookup()
    if not lookup:
        raise FileNotFoundError(
            "Win odds unavailable. Generate WinOdds_YYYY.csv via fetch_win_odds_yearly() "
            "or place SE_preprocessed.parquet in configured src_parquet_dir."
        )
    return attach_odds_from_lookup(df, lookup)
