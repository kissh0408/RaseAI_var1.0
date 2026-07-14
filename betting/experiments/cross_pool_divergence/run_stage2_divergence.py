"""cross_pool_divergence: run_stage2_divergence.py

Stage 2（TEST 非接触）: Stage 1 で確定したセグメント（K件）について fit 期間
（2023-01-01〜2024-12-31）の D_cal / D_adj / ROI_flat / クラスタブートストラップ p値 /
2023・2024 年別 D_adj（符号一貫チェック用）を測定し、一次判定（primary_pass）を行う。

このスクリプトは 2025 年以降のデータを一切読み込まない
（date フィルタを io 直後に適用）。

出力:
    betting/experiments/cross_pool_divergence/results/divergence_fit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import divergence_lib as dl  # noqa: E402

DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
STAGE1_PATH = RESULTS_DIR / "stage1_counts.json"
OUT_PATH = RESULTS_DIR / "divergence_fit.json"

STAGE2_CUTOFF = "2024-12-31"  # TEST(2025+) 非接触ガード


def _load_config() -> dict:
    return json.loads((EXP_DIR / "config.json").read_text(encoding="utf-8"))


def _read_units(name: str, cutoff: str) -> pd.DataFrame:
    path = DATA_DIR / f"units_{name}.parquet"
    df = pd.read_parquet(path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.loc[df["race_date"] <= pd.Timestamp(cutoff)].copy()
    assert df["race_date"].max() <= pd.Timestamp(cutoff), f"Stage2にTEST期間が混入: {name}"
    return df


def _place_race_arrays(sub: pd.DataFrame) -> dict:
    """レース集計配列（高速ブートストラップ用）。O_hit = y=1のときのみOを寄与させる。"""
    work = sub[["race_id", "y", "O"]].copy()
    work["O_hit"] = np.where(work["y"] > 0, work["O"].fillna(0.0), 0.0)
    g = work.groupby("race_id").agg(n_units=("y", "size"), sum_y=("y", "sum"), sum_o_hit=("O_hit", "sum"))
    return {
        "n_units": g["n_units"].to_numpy(dtype=float),
        "sum_y": g["sum_y"].to_numpy(dtype=float),
        "sum_o_hit": g["sum_o_hit"].to_numpy(dtype=float),
    }


def _place_totals_to_d_adj(totals: dict, t_place: float) -> float:
    if totals["n_units"] <= 0:
        return float("nan")
    h = totals["sum_y"] / totals["n_units"]
    if totals["sum_y"] <= 0:
        return float("nan")
    o_bar_hit = totals["sum_o_hit"] / totals["sum_y"]
    return dl.seg_d_adj_place(h, o_bar_hit, t_place)


def _measure_place_segment(sub: pd.DataFrame, t_place: float, B: int, seed: int) -> dict:
    y = sub["y"].to_numpy(dtype=float)
    p_theo = sub["p_theo"].to_numpy(dtype=float)
    o = sub["O"].to_numpy(dtype=float)
    years = pd.to_datetime(sub["race_date"]).dt.year.to_numpy()

    h = dl.seg_h(y)
    p_bar_theo = dl.seg_p_bar_theo(p_theo)
    d_cal = dl.seg_d_cal(h, p_bar_theo)
    roi_flat = dl.seg_roi_flat(y, o)
    o_bar_hit = dl.seg_o_bar_hit(y, o)
    d_adj = dl.seg_d_adj_place(h, o_bar_hit, t_place)

    d_adj_by_year: dict[int, float] = {}
    for yr in (2023, 2024):
        mask = years == yr
        if mask.sum() == 0:
            d_adj_by_year[yr] = float("nan")
            continue
        h_y = dl.seg_h(y[mask])
        o_bar_hit_y = dl.seg_o_bar_hit(y[mask], o[mask])
        d_adj_by_year[yr] = dl.seg_d_adj_place(h_y, o_bar_hit_y, t_place)

    race_arrays = _place_race_arrays(sub)
    bootstrap_p = (
        dl.cluster_bootstrap_p_value_from_race_arrays(
            race_arrays,
            lambda totals: _place_totals_to_d_adj(totals, t_place),
            d_hat=d_adj,
            B=B,
            seed=seed,
        )
        if np.isfinite(d_adj)
        else float("nan")
    )

    return {
        "n_units": int(len(sub)),
        "n_races": int(sub["race_id"].nunique()),
        "n_hits": int(y.sum()),
        "h": h,
        "p_bar_theo": p_bar_theo,
        "d_cal": d_cal,
        "o_bar_hit": o_bar_hit,
        "roi_flat": roi_flat,
        "d_adj": d_adj,
        "d_adj_2023": d_adj_by_year[2023],
        "d_adj_2024": d_adj_by_year[2024],
        "bootstrap_p": bootstrap_p,
    }


def _pair_race_arrays(sub: pd.DataFrame) -> dict:
    work = sub[["race_id", "y", "p_pool"]].copy()
    g = work.groupby("race_id").agg(n_units=("y", "size"), sum_y=("y", "sum"), sum_p_pool=("p_pool", "sum"))
    return {
        "n_units": g["n_units"].to_numpy(dtype=float),
        "sum_y": g["sum_y"].to_numpy(dtype=float),
        "sum_p_pool": g["sum_p_pool"].to_numpy(dtype=float),
    }


def _pair_totals_to_d_adj(totals: dict) -> float:
    if totals["n_units"] <= 0:
        return float("nan")
    h = totals["sum_y"] / totals["n_units"]
    p_bar_pool = totals["sum_p_pool"] / totals["n_units"]
    return dl.seg_d_adj_pair(h, p_bar_pool)


def _measure_pair_segment(sub: pd.DataFrame, B: int, seed: int) -> dict:
    y = sub["y"].to_numpy(dtype=float)
    p_theo = sub["p_theo"].to_numpy(dtype=float)
    p_pool = sub["p_pool"].to_numpy(dtype=float)
    o = sub["O"].to_numpy(dtype=float)
    years = pd.to_datetime(sub["race_date"]).dt.year.to_numpy()
    eff_takeout = sub["effective_takeout"].to_numpy(dtype=float)

    h = dl.seg_h(y)
    p_bar_theo = dl.seg_p_bar_theo(p_theo)
    p_bar_pool = dl.seg_p_bar_theo(p_pool)
    d_cal = dl.seg_d_cal(h, p_bar_theo)
    d_adj = dl.seg_d_adj_pair(h, p_bar_pool)
    roi_flat = dl.seg_roi_flat(y, o)

    d_adj_by_year: dict[int, float] = {}
    for yr in (2023, 2024):
        mask = years == yr
        if mask.sum() == 0:
            d_adj_by_year[yr] = float("nan")
            continue
        h_y = dl.seg_h(y[mask])
        p_bar_pool_y = dl.seg_p_bar_theo(p_pool[mask])
        d_adj_by_year[yr] = dl.seg_d_adj_pair(h_y, p_bar_pool_y)

    race_arrays = _pair_race_arrays(sub)
    bootstrap_p = (
        dl.cluster_bootstrap_p_value_from_race_arrays(
            race_arrays, _pair_totals_to_d_adj, d_hat=d_adj, B=B, seed=seed
        )
        if np.isfinite(d_adj)
        else float("nan")
    )

    return {
        "n_units": int(len(sub)),
        "n_races": int(sub["race_id"].nunique()),
        "n_hits": int(y.sum()),
        "h": h,
        "p_bar_theo": p_bar_theo,
        "d_cal": d_cal,
        "p_bar_pool": p_bar_pool,
        "roi_flat": roi_flat,
        "effective_takeout": float(np.nanmean(eff_takeout)) if len(eff_takeout) else float("nan"),
        "d_adj": d_adj,
        "d_adj_2023": d_adj_by_year[2023],
        "d_adj_2024": d_adj_by_year[2024],
        "bootstrap_p": bootstrap_p,
    }


def run_stage2() -> dict:
    cfg = _load_config()
    stage1 = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
    k = int(stage1["K"])
    bonferroni_thr = stage1["bonferroni_threshold"]
    d_adj_threshold = cfg["thresholds"]["d_adj_pass"]
    t_place = cfg["place"]["t_place"]
    B = cfg["bootstrap"]["B"]
    seed = cfg["bootstrap"]["seed"]

    if k == 0:
        report = {
            "stage": "divergence_fit",
            "K": 0,
            "bonferroni_threshold": None,
            "segments": [],
            "verdict": cfg["verdict"]["all_below_threshold"],
            "protocol": {
                "q_method": cfg["q_method"],
                "period": cfg["protocol"],
                "seed": seed,
                "B": B,
            },
            "caveats": [
                "K=0: 最小サンプル基準を満たすセグメントが皆無だったため Stage2 は空実行終了。",
            ],
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    place = _read_units("place", STAGE2_CUTOFF)
    wide = _read_units("wide", STAGE2_CUTOFF)
    quinella = _read_units("quinella", STAGE2_CUTOFF)

    confirmed_segments = [s for s in stage1["segments"] if s["confirmed"]]

    results = []
    for seg in confirmed_segments:
        bet_type = seg["bet_type"]
        band = seg["pop_band"]
        fs_band = seg["fs_band"]
        if bet_type == "place":
            sub = place[(place["pop_band"] == band) & (place["fs_band"] == fs_band)]
            measured = _measure_place_segment(sub, t_place, B, seed)
        else:
            src = wide if bet_type == "wide" else quinella
            sub = src[(src["pair_band"] == band) & (src["fs_band"] == fs_band)]
            measured = _measure_pair_segment(sub, B, seed)

        primary = dl.primary_pass(
            measured["d_adj"],
            measured["bootstrap_p"],
            measured["d_adj_2023"],
            measured["d_adj_2024"],
            bonferroni_thr=bonferroni_thr,
            d_adj_threshold=d_adj_threshold,
        )

        row = {
            "bet_type": bet_type,
            "pop_band": band,
            "fs_band": fs_band,
            **measured,
            "primary_pass": bool(primary),
        }
        results.append(row)
        print(
            f"{bet_type:9s} {band:10s} {fs_band:5s} "
            f"n={row['n_units']:6d} h={row['h']:.4f} d_cal={row['d_cal']:+.4f} "
            f"d_adj={row['d_adj']:+.4f} p={row['bootstrap_p']:.5f} "
            f"pass={row['primary_pass']}"
        )

    d_adj_values = [r["d_adj"] for r in results]
    verdict = dl.determine_cutoff_verdict(d_adj_values, threshold=d_adj_threshold)

    report = {
        "stage": "divergence_fit",
        "K": k,
        "bonferroni_threshold": bonferroni_thr,
        "segments": results,
        "verdict": verdict,
        "protocol": {
            "q_method": cfg["q_method"],
            "lam_source": cfg["fusion_params_source"],
            "period": cfg["protocol"],
            "seed": seed,
            "B": B,
            "t_place": t_place,
            "d_adj_threshold": d_adj_threshold,
        },
        "caveats": [
            "確定払戻は事前オッズではない。乖離が見えても購入時点で消えている可能性は残る（上限診断）。",
            "複勝 D_adj は Ō_hit（的中ユニットの算術平均払戻倍率）によるセグメントレベル近似であり、"
            "調和平均との乖離（セグメント内オッズ分散に比例するバイアス）が入る。",
            "本実験はワイド=Stern・馬連=Harvilleのみを判定に使用する（逆側モデルは副次報告のみ）。",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"K": k, "verdict": verdict}, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_PATH}")
    return report


if __name__ == "__main__":
    run_stage2()
