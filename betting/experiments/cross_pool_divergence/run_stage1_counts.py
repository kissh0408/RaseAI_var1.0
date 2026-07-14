"""cross_pool_divergence: run_stage1_counts.py

Stage 1（TEST 非接触）: units_{place,wide,quinella}.parquet を 2024-12-31 以前で
フィルタし、30セグメント（bet_type × pop_band/pair_band × fs_band）の n_units・
n_races・Σp_theo を集計する。最小サンプル基準（n_units>=300 かつ Σp_theo>=30）を
満たしたセグメントのみ「確定」とし、K（確定セグメント数）と Bonferroni 閾値（0.01/K）
を出力する。

このスクリプトは 2025 年以降のデータを一切読み込まない
（date フィルタを io 直後に適用。仕様書 §8「期間規律」）。

出力:
    betting/experiments/cross_pool_divergence/results/stage1_counts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import divergence_lib as dl  # noqa: E402

DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
OUT_PATH = RESULTS_DIR / "stage1_counts.json"

STAGE1_CUTOFF = "2024-12-31"  # TEST(2025+) 非接触ガード


def _load_config() -> dict:
    return json.loads((EXP_DIR / "config.json").read_text(encoding="utf-8"))


def _read_units(name: str, cutoff: str) -> pd.DataFrame:
    path = DATA_DIR / f"units_{name}.parquet"
    df = pd.read_parquet(path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    # TEST 非接触ガード: 2024-12-31 以前しか読み込まない（build_dataset 側で既に
    # fit期間のみに制限されているが、Stage1/2 側でも io 直後に再度ガードする）。
    df = df.loc[df["race_date"] <= pd.Timestamp(cutoff)].copy()
    assert df["race_date"].max() <= pd.Timestamp(cutoff), f"Stage1にTEST期間が混入: {name}"
    return df


def run_stage1() -> dict:
    cfg = _load_config()
    n_min = cfg["thresholds"]["n_units_min"]
    p_theo_min = cfg["thresholds"]["sum_p_theo_min"]

    place = _read_units("place", STAGE1_CUTOFF)
    wide = _read_units("wide", STAGE1_CUTOFF)
    quinella = _read_units("quinella", STAGE1_CUTOFF)

    segments: list[dict] = []

    for pop_band in cfg["pop_band_order_single"]:
        for fs_band in cfg["fs_band_order"]:
            sub = place[(place["pop_band"] == pop_band) & (place["fs_band"] == fs_band)]
            n_units = int(len(sub))
            n_races = int(sub["race_id"].nunique())
            sum_p_theo = float(sub["p_theo"].sum())
            confirmed = dl.min_sample_confirmed(n_units, sum_p_theo, n_min=n_min, p_theo_min=p_theo_min)
            segments.append(
                {
                    "bet_type": "place",
                    "pop_band": pop_band,
                    "fs_band": fs_band,
                    "n_units": n_units,
                    "n_races": n_races,
                    "sum_p_theo": sum_p_theo,
                    "confirmed": bool(confirmed),
                }
            )

    for bet_type, df in (("wide", wide), ("quinella", quinella)):
        for pair_band in cfg["pair_bands"]:
            for fs_band in cfg["fs_band_order"]:
                sub = df[(df["pair_band"] == pair_band) & (df["fs_band"] == fs_band)]
                n_units = int(len(sub))
                n_races = int(sub["race_id"].nunique())
                sum_p_theo = float(sub["p_theo"].sum())
                confirmed = dl.min_sample_confirmed(n_units, sum_p_theo, n_min=n_min, p_theo_min=p_theo_min)
                segments.append(
                    {
                        "bet_type": bet_type,
                        "pop_band": pair_band,
                        "fs_band": fs_band,
                        "n_units": n_units,
                        "n_races": n_races,
                        "sum_p_theo": sum_p_theo,
                        "confirmed": bool(confirmed),
                    }
                )

    assert len(segments) == cfg["k_max"], f"segment count mismatch: {len(segments)} != {cfg['k_max']}"

    k = sum(1 for s in segments if s["confirmed"])
    bonferroni = dl.bonferroni_threshold(k, alpha=cfg["thresholds"]["bonferroni_alpha"])

    excluded = [s for s in segments if not s["confirmed"]]

    report = {
        "stage": "stage1_counts",
        "cutoff": STAGE1_CUTOFF,
        "n_units_min": n_min,
        "sum_p_theo_min": p_theo_min,
        "segments": segments,
        "K": k,
        "K_max": cfg["k_max"],
        "bonferroni_threshold": bonferroni,
        "excluded_segments": [
            {"bet_type": s["bet_type"], "pop_band": s["pop_band"], "fs_band": s["fs_band"], "n_units": s["n_units"], "sum_p_theo": s["sum_p_theo"]}
            for s in excluded
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    return report


if __name__ == "__main__":
    run_stage1()
