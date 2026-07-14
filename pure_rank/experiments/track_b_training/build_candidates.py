"""track_b_training: build_candidates.py

HC/WC (JRA official workout timing) + fold2-OOS-scores race keys ->
data/cand_b{n}_{name}.parquet (3 cols: race_id, horse_num, cand_score, no NaN)
+ data/cand_b{n}_{name}.meta.json (raw NaN rate, row counts, generation params).

Usage: python build_candidates.py --candidate b1   (one candidate per run)

Isolation: this script and training_lib.py never read betting-related
columns of any kind (verified by tests/test_market_guard.py and the
README's market boundary section).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import training_lib as tl  # noqa: E402

DATA_DIR = EXP_DIR / "data"


def _load_race_keys(scores_path: Path) -> pd.DataFrame:
    """Unique (race_id, horse_num, ketto_num, race_date) rows from fold2 OOS scores."""
    scores = pd.read_parquet(scores_path, columns=["race_id", "horse_num", "ketto_num", "race_date"])
    scores["race_id"] = scores["race_id"].astype(str)
    keys = scores.drop_duplicates(subset=["race_id", "horse_num"]).reset_index(drop=True)
    return keys


def _nan_rate_by_year(cand: pd.DataFrame, race_keys: pd.DataFrame) -> dict:
    merged = cand.merge(race_keys[["race_id", "horse_num", "race_date"]], on=["race_id", "horse_num"], how="left")
    merged["year"] = pd.to_datetime(merged["race_date"]).dt.year
    out = {"overall": float(merged["cand_score"].isna().mean())}
    for y in (2023, 2024, 2025):
        sub = merged.loc[merged["year"] == y, "cand_score"]
        out[str(y)] = float(sub.isna().mean()) if len(sub) else None
    return out


def _write_meta(name: str, out_path: Path, params: dict, raw_nan_rates: dict, n_rows: int) -> Path:
    meta = {
        "candidate": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": n_rows,
        "params": params,
        "raw_nan_rate": raw_nan_rates,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def _finalize_and_write(name: str, raw: pd.DataFrame, race_keys: pd.DataFrame, params: dict, cfg: dict) -> tuple[Path, Path]:
    assert list(raw.columns) == ["race_id", "horse_num", "cand_score"]
    raw_nan_rates = _nan_rate_by_year(raw, race_keys)
    filled = tl.fill_race_mean(raw)
    assert filled["cand_score"].isna().sum() == 0, "cand_score must have zero NaN after fill_race_mean"
    assert not filled.duplicated(subset=["race_id", "horse_num"]).any(), "(race_id, horse_num) must be unique"

    out_path = DATA_DIR / cfg["candidates"][name]["output"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filled.to_parquet(out_path, index=False)
    meta_path = _write_meta(name, out_path, params, raw_nan_rates, len(filled))
    print(f"[{name}] wrote {out_path} rows={len(filled):,} raw_nan_rate={raw_nan_rates}")
    return out_path, meta_path


def build_b1(cfg: dict) -> tuple[Path, Path]:
    root = ROOT
    race_keys = _load_race_keys(root / cfg["paths"]["scores_fold2_oos"])
    hc = pd.read_parquet(
        root / cfg["paths"]["hc_preprocessed"],
        columns=["ketto_num", "training_date", "hc_4f_sec"],
    )
    hc = hc.loc[hc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    window_days = cfg["window_days"]
    min_n = cfg["min_n"]["b1_intensity_trend"]
    raw = tl.build_b1_intensity_trend(hc, race_keys, window_days=window_days, min_n=min_n)
    return _finalize_and_write("b1", raw, race_keys, {"window_days": window_days, "min_n": min_n}, cfg)


def build_b2(cfg: dict) -> tuple[Path, Path]:
    root = ROOT
    race_keys = _load_race_keys(root / cfg["paths"]["scores_fold2_oos"])
    hc = pd.read_parquet(
        root / cfg["paths"]["hc_preprocessed"],
        columns=["ketto_num", "training_date"],
    )
    hc = hc.loc[hc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    features = pd.read_parquet(
        root / cfg["paths"]["features_v39_course_slim"],
        columns=["ketto_num", "race_date"],
    )
    race_history = features.loc[features["ketto_num"].isin(race_keys["ketto_num"].unique())]
    min_baseline_n = cfg["min_n"]["b2_freq_change_baseline_races"]
    raw = tl.build_b2_freq_change(hc, race_history, race_keys, min_baseline_n=min_baseline_n)
    return _finalize_and_write("b2", raw, race_keys, {"min_baseline_n": min_baseline_n}, cfg)


def build_b3(cfg: dict) -> tuple[Path, Path]:
    root = ROOT
    race_keys = _load_race_keys(root / cfg["paths"]["scores_fold2_oos"])
    hc = pd.read_parquet(
        root / cfg["paths"]["hc_preprocessed"],
        columns=["ketto_num", "training_date", "hc_accel_sec"],
    )
    hc = hc.loc[hc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    recent_n = cfg["min_n"]["b3_accel_profile_recent"]
    min_career_n = cfg["min_n"]["b3_accel_profile_career"]
    raw = tl.build_b3_accel_profile(hc, race_keys, recent_n=recent_n, min_career_n=min_career_n)
    return _finalize_and_write("b3", raw, race_keys, {"recent_n": recent_n, "min_career_n": min_career_n}, cfg)


def build_b4(cfg: dict) -> tuple[Path, Path]:
    root = ROOT
    race_keys = _load_race_keys(root / cfg["paths"]["scores_fold2_oos"])
    hc = pd.read_parquet(
        root / cfg["paths"]["hc_preprocessed"],
        columns=["ketto_num", "training_date", "hc_200_sec", "hc_3f_sec"],
    )
    hc = hc.loc[hc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    window_days = cfg["window_days"]
    min_n = cfg["min_n"]["b4_fade_trend"]
    raw = tl.build_b4_fade_trend(hc, race_keys, window_days=window_days, min_n=min_n)
    return _finalize_and_write("b4", raw, race_keys, {"window_days": window_days, "min_n": min_n}, cfg)


def build_b5(cfg: dict) -> tuple[Path, Path]:
    root = ROOT
    race_keys = _load_race_keys(root / cfg["paths"]["scores_fold2_oos"])
    hc = pd.read_parquet(
        root / cfg["paths"]["hc_preprocessed"],
        columns=["ketto_num", "training_date"],
    )
    hc = hc.loc[hc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    wc = pd.read_parquet(
        root / cfg["paths"]["wc_preprocessed"],
        columns=["ketto_num", "training_date"],
    )
    wc = wc.loc[wc["ketto_num"].isin(race_keys["ketto_num"].unique())]
    wc_start_cfg = cfg["wc_start"]
    wc_full = pd.read_parquet(root / cfg["paths"]["wc_preprocessed"], columns=["training_date"])
    wc_start_actual = str(pd.to_datetime(wc_full["training_date"]).min().date())
    print(f"[b5] WC_START config={wc_start_cfg} actual_min={wc_start_actual}")
    assert wc_start_actual == wc_start_cfg, (
        f"WC_START mismatch: config={wc_start_cfg} actual={wc_start_actual}"
    )
    window_days = cfg["window_days"]
    min_window_n = cfg["min_n"]["b5_wc_switch_window"]
    min_career_n = cfg["min_n"]["b5_wc_switch_career"]
    raw = tl.build_b5_wc_switch(
        hc, wc, race_keys,
        window_days=window_days, wc_start=wc_start_cfg,
        min_window_n=min_window_n, min_career_n=min_career_n,
    )
    return _finalize_and_write(
        "b5", raw, race_keys,
        {
            "window_days": window_days, "wc_start": wc_start_cfg,
            "min_window_n": min_window_n, "min_career_n": min_career_n,
        },
        cfg,
    )


BUILDERS = {"b1": build_b1, "b2": build_b2, "b3": build_b3, "b4": build_b4, "b5": build_b5}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one track_b_training candidate parquet")
    parser.add_argument("--candidate", choices=sorted(BUILDERS.keys()), required=True)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.json")
    args = parser.parse_args()

    cfg = tl.load_config(args.config)
    fn = BUILDERS[args.candidate]
    out_path, meta_path = fn(cfg)
    print(f"Done: {out_path}\nMeta: {meta_path}")


if __name__ == "__main__":
    main()
