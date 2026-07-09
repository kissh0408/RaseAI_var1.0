"""place_calibration: fit_calibrators.py

fit 期間（2023-01-01..2024-12-31）のみを使い、A1（logloss λ 再フィット・global）、
A2（logloss λ 再フィット・頭数帯別 5-7頭/8頭以上）、B1/B2（isotonic 事後較正、
λ は formal 値 0.6018/0.6381 に固定）を fit する。

TEST(2025+) は一切参照しない（Rule 3: 後出し禁止）。

出力:
    pure_rank/experiments/place_calibration/models/lambda_fit.json  (A1, A2の結果)
    pure_rank/experiments/place_calibration/models/isotonic_b.joblib (B1/B2共通のisotonicモデル)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_DIR))

from calib_lib import (  # noqa: E402
    band_5to7_8plus,
    fit_isotonic,
    fit_lambda_logloss,
    fit_lambda_logloss_banded,
)

from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402

DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
CONFIG_PATH = EXP_DIR / "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_race_lists(fit_df: pd.DataFrame) -> tuple[list, list, list]:
    races_p_win, races_y, races_band = [], [], []
    for _, grp in fit_df.groupby("race_id", sort=False):
        p_win = grp["p_win"].astype(float).to_numpy()
        y = grp["y_place"].astype(float).to_numpy()
        n = len(grp)
        races_p_win.append(p_win)
        races_y.append(y)
        races_band.append("le7" if n <= 7 else "ge8")
    return races_p_win, races_y, races_band


def fit_calibrators() -> dict:
    cfg = _load_config()
    fit_df = pd.read_parquet(DATA_DIR / "fit_2023_2024.parquet")
    fit_df["race_id"] = fit_df["race_id"].astype(str)
    n_fit_races = fit_df["race_id"].nunique()
    print(f"Fit period: rows={len(fit_df):,}, races={n_fit_races:,}")

    lam_cfg = cfg["lambda_fit"]
    bounds = [tuple(b) for b in lam_cfg["bounds"]]
    init_lam2 = lam_cfg["init_lam2"]
    init_lam3 = lam_cfg["init_lam3"]

    races_p_win, races_y, races_band = _build_race_lists(fit_df)

    # ─── A1: global logloss λ 再フィット ────────────────────────────────
    print("\n=== A1: global logloss lambda fit ===")
    a1_lam2, a1_lam3 = fit_lambda_logloss(
        races_p_win, races_y, init_lam2=init_lam2, init_lam3=init_lam3, bounds=bounds
    )
    print(f"A1: lam2={a1_lam2:.6f}, lam3={a1_lam3:.6f}")

    # ─── A2: 頭数帯別 logloss λ 再フィット ────────────────────────────────
    print("\n=== A2: banded (le7/ge8) logloss lambda fit ===")
    a2_bands = fit_lambda_logloss_banded(
        races_p_win, races_y, races_band,
        bands=tuple(lam_cfg["bands"]), init_lam2=init_lam2, init_lam3=init_lam3, bounds=bounds,
    )
    for band, res in a2_bands.items():
        print(f"A2[{band}]: lam2={res['lam2']:.6f}, lam3={res['lam3']:.6f}, n_races={res['n_races']:,}")

    # ─── B1/B2: isotonic 事後較正 (lam は formal 値に固定) ────────────────
    print("\n=== B1/B2: isotonic post-calibration (lambda fixed at formal S0) ===")
    s0 = cfg["s0_baseline"]
    p_stern_fit_list = []
    for p_win in races_p_win:
        p_stern_fit_list.append(place_prob_from_p_win(p_win, s0["lam2"], s0["lam3"]))
    p_stern_fit = np.concatenate(p_stern_fit_list)
    y_fit = np.concatenate(races_y)

    iso_cfg = cfg["isotonic"]
    iso = fit_isotonic(p_stern_fit, y_fit, y_min=iso_cfg["y_min"], y_max=iso_cfg["y_max"])
    n_knots = len(iso.X_thresholds_) if hasattr(iso, "X_thresholds_") else None
    print(f"Isotonic fit: n_fit_points={len(p_stern_fit):,}, n_knots={n_knots}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    lambda_fit_result = {
        "fit_period": cfg["fit_period"],
        "n_fit_races": n_fit_races,
        "n_fit_rows": len(fit_df),
        "a1": {"lam2": a1_lam2, "lam3": a1_lam3},
        "a2": {band: res for band, res in a2_bands.items()},
        "s0_reference": {"lam2": s0["lam2"], "lam3": s0["lam3"]},
    }
    lambda_fit_path = MODELS_DIR / "lambda_fit.json"
    lambda_fit_path.write_text(json.dumps(lambda_fit_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {lambda_fit_path}")

    iso_path = MODELS_DIR / "isotonic_b.joblib"
    joblib.dump(iso, iso_path)
    print(f"Saved: {iso_path}")

    return lambda_fit_result


if __name__ == "__main__":
    fit_calibrators()
