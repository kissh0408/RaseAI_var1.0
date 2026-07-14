"""cross_pool_divergence: run_supplement_reverse.py

差し戻し対応（仕様書 §11-6 の副次報告）:
  1. 逆側モデル（ワイド Harville / 馬連 Stern）の d_cal を確定セグメントごとに算出。
  2. Poisson-binomial 正規近似 z 検定の両側 p 値（D_cal 軸の参考値。判定不使用）を
     主モデル・逆側モデルの両方について算出。
  3. `results/divergence_fit.json` に副次フィールドを**追加のみ**行う
     （既存 primary 結果の数値は一切変更しない。変更検知のため更新前後で
     primary キーの値一致を assert する）。

TEST 非接触: fit 期間（〜2024-12-31）のユニットのみ使用。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
for p in (str(ROOT), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from betting.src.ev_filters import harville_wide_pair_prob  # noqa: E402
from betting.src.pair_probs import stern_quinella_pair_prob  # noqa: E402
from betting.src.wide_ev_core import norm_pair  # noqa: E402

from build_dataset import _load_base_frame, _load_config, _load_lambda_params  # noqa: E402

DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
FIT_JSON_PATH = RESULTS_DIR / "divergence_fit.json"

STAGE2_CUTOFF = "2024-12-31"

# primary 結果の不変性を保証する検査対象キー
_PRIMARY_KEYS = [
    "n_units", "n_races", "n_hits", "h", "p_bar_theo", "d_cal",
    "roi_flat", "d_adj", "d_adj_2023", "d_adj_2024", "bootstrap_p", "primary_pass",
]


def _poisson_binomial_z_p(y: np.ndarray, p_theo: np.ndarray) -> float:
    """Σy vs Σp_theo の Poisson-binomial 正規近似 z 検定（両側 p、判定不使用の参考値）。"""
    p = np.clip(np.asarray(p_theo, dtype=float), 1e-12, 1 - 1e-12)
    var = float(np.sum(p * (1.0 - p)))
    if var <= 0:
        return float("nan")
    z = (float(np.sum(y)) - float(np.sum(p))) / np.sqrt(var)
    from scipy import stats

    return float(2.0 * stats.norm.sf(abs(z)))


def _compute_reverse_p_theo(base: pd.DataFrame, cfg: dict, lam2: float, bet_type: str) -> pd.DataFrame:
    """(race_id, unit_key) -> 逆側モデル p_theo。ワイド=Harville、馬連=Stern。"""
    bt_cfg = cfg[bet_type]
    hc_min, hc_max = bt_cfg["horse_count_min"], bt_cfg["horse_count_max"]
    sub = base[(base["horse_count"] >= hc_min) & (base["horse_count"] <= hc_max)]

    rows = []
    t0 = time.perf_counter()
    for race_id, grp in sub.groupby("race_id", sort=False):
        n = len(grp)
        horse_nums = grp["horse_num"].astype(int).to_numpy()
        p_win = grp["market_q"].astype(float).to_numpy()
        p_dict = {int(h): float(q) for h, q in zip(horse_nums, p_win)}
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = int(horse_nums[i]), int(horse_nums[j])
                pair = norm_pair(hi, hj)
                if bet_type == "wide":
                    p_rev = harville_wide_pair_prob(p_dict, hi, hj)
                else:
                    p_rev = stern_quinella_pair_prob(p_win, i, j, lam2)
                rows.append({"race_id": race_id, "unit_key": f"{pair[0]}-{pair[1]}", "p_theo_rev": float(p_rev)})
    print(f"{bet_type} reverse p_theo: {len(rows):,} pairs, elapsed={time.perf_counter() - t0:.1f}s")
    return pd.DataFrame(rows)


def _read_units(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"units_{name}.parquet")
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.loc[df["race_date"] <= pd.Timestamp(STAGE2_CUTOFF)].copy()
    return df


def run_supplement() -> dict:
    cfg = _load_config()
    lam2, _lam3 = _load_lambda_params(cfg)

    report = json.loads(FIT_JSON_PATH.read_text(encoding="utf-8"))
    before_primary = {
        (s["bet_type"], s["pop_band"], s["fs_band"]): {k: s.get(k) for k in _PRIMARY_KEYS}
        for s in report["segments"]
    }

    place = _read_units("place")
    wide = _read_units("wide")
    quinella = _read_units("quinella")

    base = _load_base_frame(cfg)
    rev_wide = _compute_reverse_p_theo(base, cfg, lam2, "wide")
    rev_quinella = _compute_reverse_p_theo(base, cfg, lam2, "quinella")

    wide = wide.merge(rev_wide, on=["race_id", "unit_key"], how="left")
    quinella = quinella.merge(rev_quinella, on=["race_id", "unit_key"], how="left")
    n_missing_w = int(wide["p_theo_rev"].isna().sum())
    n_missing_q = int(quinella["p_theo_rev"].isna().sum())
    print(f"reverse merge: wide missing={n_missing_w}, quinella missing={n_missing_q}")

    for seg in report["segments"]:
        bet_type = seg["bet_type"]
        band, fs_band = seg["pop_band"], seg["fs_band"]
        if bet_type == "place":
            sub = place[(place["pop_band"] == band) & (place["fs_band"] == fs_band)]
            seg["poisson_binomial_z_p"] = _poisson_binomial_z_p(
                sub["y"].to_numpy(dtype=float), sub["p_theo"].to_numpy(dtype=float)
            )
            continue
        src = wide if bet_type == "wide" else quinella
        sub = src[(src["pair_band"] == band) & (src["fs_band"] == fs_band)]
        y = sub["y"].to_numpy(dtype=float)
        p_theo = sub["p_theo"].to_numpy(dtype=float)
        p_rev = sub["p_theo_rev"].to_numpy(dtype=float)
        seg["poisson_binomial_z_p"] = _poisson_binomial_z_p(y, p_theo)
        rev_model = "harville" if bet_type == "wide" else "stern"
        seg["reverse_model"] = rev_model
        seg["d_cal_reverse"] = float(np.mean(y) - np.nanmean(p_rev))
        mask = np.isfinite(p_rev)
        seg["poisson_binomial_z_p_reverse"] = _poisson_binomial_z_p(y[mask], p_rev[mask])

    # primary 結果の不変性検査
    for s in report["segments"]:
        key = (s["bet_type"], s["pop_band"], s["fs_band"])
        for k, v in before_primary[key].items():
            assert s.get(k) == v, f"primary値が変更された: {key} {k}"

    report.setdefault("caveats", []).append(
        "逆側モデル（ワイドHarville/馬連Stern）の d_cal と Poisson-binomial z p値は副次報告であり"
        "判定・打ち切りには一切使用しない（run_supplement_reverse.py で追記）。"
    )

    FIT_JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved (fields added, primary unchanged): {FIT_JSON_PATH}")
    return report


if __name__ == "__main__":
    run_supplement()
