"""4馬場シナリオ推論のレイテンシ計測（Phase 1-5 ゲート用）。

Usage:
  python model_training/scripts/benchmark_baba_inference.py
  python model_training/scripts/benchmark_baba_inference.py --n-rows 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main.pipeline.inference_pipeline import apply_uniform_baba_jv_code, load_models, predict_ranks_for_frame

LOG_DIR = ROOT / "model_training" / "logs" / "going_diagnostics"
MODELS_DIR = ROOT / "model_training" / "models" / "ensemble_v5_specv2"
FEATURES_PATH = ROOT / "model_training" / "data" / "02_features" / "features_past_v25_odds.parquet"


def _synthetic_frame(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_races = max(1, n_rows // 12)
    rows = []
    for i in range(n_races):
        rid = f"20260614_05_{i + 1:02d}"
        n_h = min(18, max(8, n_rows // n_races))
        for h in range(n_h):
            rows.append(
                {
                    "race_id": rid,
                    "track_code": 11,
                    "turf_condition": 0.0,
                    "dirt_condition": 0.0,
                    "track_condition_code": 0.0,
                    "horse_turf_heavy_win_rate": float(rng.uniform(0.05, 0.15)),
                    "horse_turf_very_heavy_win_rate": float(rng.uniform(0.04, 0.12)),
                    "horse_turf_light_win_rate": float(rng.uniform(0.08, 0.20)),
                    "horse_turf_soft_win_rate": float(rng.uniform(0.06, 0.18)),
                    "horse_dirt_heavy_win_rate": 0.0,
                    "horse_dirt_soft_win_rate": 0.0,
                    "going_match_score_turf_imputed": float(rng.uniform(0.5, 2.0)),
                    "going_change_lag1": float(rng.integers(-1, 2)),
                    "going_worsening_flag": 0,
                    "tm_score": float(rng.uniform(45, 55)),
                    "mining_predicted_rank": float(rng.integers(1, 15)),
                    "running_style_code": float(rng.integers(1, 5)),
                }
            )
    return pd.DataFrame(rows[:n_rows])


def _load_sample_frame(n_rows: int) -> pd.DataFrame:
    if FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)
        return df.head(n_rows).copy()
    return _synthetic_frame(n_rows)


def run_benchmark(*, n_rows: int = 500) -> dict:
    df = _load_sample_frame(n_rows)
    models = load_models(MODELS_DIR)

    t0 = time.perf_counter()
    for jv in (1, 2, 3, 4):
        frame = apply_uniform_baba_jv_code(df, jv)
        predict_ranks_for_frame(models, frame)
    elapsed = time.perf_counter() - t0

    per_scenario = elapsed / 4.0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "models_dir": str(MODELS_DIR),
        "total_sec_4_scenarios": round(elapsed, 3),
        "per_scenario_sec": round(per_scenario, 3),
        "gate_target_sec": 120.0,
        "passed_under_120s": elapsed <= 120.0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="4シナリオ推論レイテンシ計測")
    parser.add_argument("--n-rows", type=int, default=500)
    args = parser.parse_args()

    report = run_benchmark(n_rows=args.n_rows)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"baba_inference_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
