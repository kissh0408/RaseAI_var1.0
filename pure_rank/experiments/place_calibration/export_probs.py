"""place_calibration: export_probs.py

TEST 期間(2025-01-01〜)で 5 系列（S0/A1/A2/B1/B2）の複勝(top3)確率を算出し、
1 parquet にまとめる。

  S0: 現行 Stern（formal λ 固定 0.6018/0.6381）
  A1: logloss λ 再フィット（global）
  A2: logloss λ 再フィット（頭数帯別 5-7頭/8頭以上、レース単位で帯を選択）
  B1: isotonic raw（λ固定S0出力にisotonicを適用、正規化なし）
  B2: isotonic normalized（B1をレース内合計3に正規化、clip+再配分）

出力: pure_rank/experiments/place_calibration/data/probs_test_2025.parquet
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

from calib_lib import normalize_place_probs  # noqa: E402

from prob_fusion.src.place_prob import place_prob_from_p_win  # noqa: E402

DATA_DIR = EXP_DIR / "data"
MODELS_DIR = EXP_DIR / "models"
CONFIG_PATH = EXP_DIR / "config.json"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def export_place_calibration_probs() -> Path:
    cfg = _load_config()
    s0 = cfg["s0_baseline"]

    lambda_fit = json.loads((MODELS_DIR / "lambda_fit.json").read_text(encoding="utf-8"))
    a1_lam2, a1_lam3 = lambda_fit["a1"]["lam2"], lambda_fit["a1"]["lam3"]
    a2_bands = lambda_fit["a2"]
    print(f"S0: lam2={s0['lam2']:.6f}, lam3={s0['lam3']:.6f}")
    print(f"A1: lam2={a1_lam2:.6f}, lam3={a1_lam3:.6f}")
    for band, res in a2_bands.items():
        print(f"A2[{band}]: lam2={res['lam2']:.6f}, lam3={res['lam3']:.6f}")

    # joblib.load: models/isotonic_b.joblib is produced locally by fit_calibrators.py
    # in this same experiment (not an externally-sourced/untrusted artifact).
    iso = joblib.load(MODELS_DIR / "isotonic_b.joblib")

    test_df = pd.read_parquet(DATA_DIR / "test_2025.parquet")
    test_df["race_id"] = test_df["race_id"].astype(str)
    n_races = test_df["race_id"].nunique()
    print(f"TEST rows={len(test_df):,}, races={n_races:,}")

    band_lam = {
        band: (res["lam2"], res["lam3"]) for band, res in a2_bands.items()
    }
    band_max_le7 = cfg["lambda_fit"]["band_boundary_max_le7"]

    out_rows = []
    for _, grp in test_df.groupby("race_id", sort=False):
        p_win = grp["p_win"].astype(float).to_numpy()
        n = len(grp)

        p_s0 = place_prob_from_p_win(p_win, s0["lam2"], s0["lam3"])
        p_a1 = place_prob_from_p_win(p_win, a1_lam2, a1_lam3)

        band = "le7" if n <= band_max_le7 else "ge8"
        lam2_a2, lam3_a2 = band_lam[band]
        p_a2 = place_prob_from_p_win(p_win, lam2_a2, lam3_a2)

        out = grp[["race_id", "ketto_num", "horse_num", "race_date", "horse_count", "finish_rank", "y_place"]].copy()
        out["p_s0"] = p_s0
        out["p_a1"] = p_a1
        out["p_a2"] = p_a2
        out_rows.append(out)

    result = pd.concat(out_rows, ignore_index=True)

    # B1: isotonic raw（正規化なし）
    result["p_b1"] = iso.predict(result["p_s0"].to_numpy())

    # B2: isotonic normalized（レース内合計3、clip+再配分。place_direct の関数を import 再利用）
    p_b2, clip_count = normalize_place_probs(
        result["p_b1"].to_numpy(),
        result["race_id"].to_numpy(),
        max_iter=cfg["normalize"]["max_iter"],
    )
    result["p_b2"] = p_b2
    print(f"B2 normalize: clip発生頭数={clip_count:,} / {len(result):,}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "probs_test_2025.parquet"
    result.to_parquet(out_path, index=False, compression="snappy")
    print(f"Saved: {out_path} ({len(result):,} rows, {result['race_id'].nunique():,} races)")
    return out_path


if __name__ == "__main__":
    export_place_calibration_probs()
