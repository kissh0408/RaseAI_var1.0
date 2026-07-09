"""Load confirmed place payouts from JV-Link HR data.

HR payouts are settlement data, not pre-race odds. Use them for ROI
settlement and upper-bound diagnostics only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

PlacePayoutLookup = dict[str, dict[int, int]]


def make_race_id_from_row(row: Mapping[str, object]) -> str:
    """Build canonical 16-digit race_id: YYYY + MMDD + CC + KK + NN + RR."""
    return (
        str(int(row["year"])).zfill(4)
        + str(int(row["month_day"])).zfill(4)
        + str(int(row["course_code"])).zfill(2)
        + str(int(row["kai"])).zfill(2)
        + str(int(row["nichi"])).zfill(2)
        + str(int(row["race_num"])).zfill(2)
    )


def _parse_horse(raw: object) -> int | None:
    s = str(raw).strip()
    if not s or s in {"0", "00"}:
        return None
    try:
        horse = int(s)
    except ValueError:
        return None
    return horse if horse > 0 else None


def _parse_payout(raw: object) -> int | None:
    s = str(raw).strip()
    if not s:
        return None
    try:
        payout = int(float(s))
    except ValueError:
        return None
    return payout if payout > 0 else None


def build_place_payout_lookup_from_frame(hr: pd.DataFrame) -> PlacePayoutLookup:
    """Build race_id -> horse_num -> place payout yen per 100 yen."""
    if "record_id" in hr.columns:
        hr = hr[hr["record_id"].astype(str).str.strip().eq("HR")].copy()

    lookup: PlacePayoutLookup = {}
    for _, row in hr.iterrows():
        try:
            race_id = make_race_id_from_row(row)
        except (KeyError, TypeError, ValueError):
            continue
        race_lookup = lookup.setdefault(race_id, {})
        for i in range(1, 6):
            hcol = f"place_{i}_horse"
            mcol = f"place_{i}_money"
            if hcol not in hr.columns or mcol not in hr.columns:
                continue
            horse = _parse_horse(row.get(hcol))
            payout = _parse_payout(row.get(mcol))
            if horse is None or payout is None:
                continue
            race_lookup[horse] = max(race_lookup.get(horse, 0), payout)

    return {rid: horses for rid, horses in lookup.items() if horses}


def build_place_payout_lookup_from_csvs(hr_dir: Path) -> PlacePayoutLookup:
    """Load all race_hr_*.csv under hr_dir into a place payout lookup."""
    files = sorted(Path(hr_dir).glob("race_hr_*.csv"))
    if not files:
        raise FileNotFoundError(f"No race_hr_*.csv files found under {hr_dir}")
    frames = [pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False) for path in files]
    return build_place_payout_lookup_from_frame(pd.concat(frames, ignore_index=True))


def build_place_payout_lookup_from_parquet(hr_path: Path) -> PlacePayoutLookup:
    """Load place payouts from HR_preprocessed long-format parquet if present."""
    hr = pd.read_parquet(hr_path)
    if "bet_type" not in hr.columns:
        raise ValueError(f"HR parquet has no bet_type column: {hr_path}")
    sub = hr[hr["bet_type"].astype(str).eq("place")]
    lookup: PlacePayoutLookup = {}
    for _, row in sub.iterrows():
        payout = _parse_payout(row.get("payout"))
        if payout is None:
            continue
        race_id = str(row["race_id"])
        horse = int(row["horse_num_1"])
        lookup.setdefault(race_id, {})[horse] = max(lookup.setdefault(race_id, {}).get(horse, 0), payout)
    return lookup


def attach_place_payout(
    df: pd.DataFrame,
    lookup: PlacePayoutLookup,
    *,
    race_id_col: str = "race_id",
    horse_num_col: str = "horse_num",
) -> pd.DataFrame:
    """Attach place payout, multiplier, and paid flag to a dataframe."""
    out = df.copy()
    out[race_id_col] = out[race_id_col].astype(str)
    if horse_num_col not in out.columns and "horse_number" in out.columns:
        out[horse_num_col] = out["horse_number"]

    def _payout(row: pd.Series) -> int:
        return int(lookup.get(str(row[race_id_col]), {}).get(int(row[horse_num_col]), 0))

    out["place_payout"] = out.apply(_payout, axis=1).astype(int)
    out["place_multiplier"] = out["place_payout"].astype(float) / 100.0
    out["place_paid"] = out["place_payout"] > 0
    return out


def build_place_payout_lookup(
    *,
    hr_dir: Path | None = None,
    hr_parquet: Path | None = None,
) -> PlacePayoutLookup:
    """Load place payouts from parquet if available, otherwise from HR CSVs."""
    if hr_parquet is not None and hr_parquet.is_file():
        lookup = build_place_payout_lookup_from_parquet(hr_parquet)
        if lookup:
            return lookup
        if hr_dir is None or not list(Path(hr_dir).glob("race_hr_*.csv")):
            return {}
    if hr_dir is None:
        raise FileNotFoundError("Either hr_parquet or hr_dir is required for place payout lookup")
    return build_place_payout_lookup_from_csvs(hr_dir)
