"""
fit_specv2_calibrator.py — specv2 OOF から rank1 isotonic calibrator を学習・保存

fit: valid_year < holdout_year（既定 2025）
出力: strategy/models/calibration_isotonic_specv2.json（legacy は触らない）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECv2_OOF = PROJECT_ROOT / "model_training" / "data" / "03_train" / "evaluation_specv2_oof.csv"
OUT = PROJECT_ROOT / "strategy" / "models" / "calibration_isotonic_specv2.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-year", type=int, default=2025)
    parser.add_argument("--eval-path", type=Path, default=SPECv2_OOF)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not args.eval_path.exists():
        print(f"[NG] missing {args.eval_path}")
        return 1

    from model_training.src.calibration_report import compute_rank1_calibration_metrics

    df = pd.read_csv(args.eval_path)
    fit = df[pd.to_numeric(df["valid_year"], errors="coerce") < args.holdout_year].copy()
    x = pd.to_numeric(fit["pred_rank1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = (pd.to_numeric(fit["finish_rank"], errors="coerce") == 1).astype(float).to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)

    metrics = compute_rank1_calibration_metrics(
        fit.rename(columns={"pred_rank1": "pred_score"}),
        score_col="pred_score",
        isotonic_model=iso,
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.eval_path),
        "fit_years": f"< {args.holdout_year}",
        "n_samples": int(len(x)),
        "metrics_fit_period": metrics,
        "params": {
            "method": "isotonic",
            "x_thresholds": iso.X_thresholds_.tolist(),
            "y_thresholds": iso.y_thresholds_.tolist(),
            "interpolation": "linear",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")
    print(f"  n={len(x)} brier_iso={metrics.get('brier_isotonic')} ece={metrics.get('ece_isotonic_quantile')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
