"""prob_fusion/src/run_fit_oos.py と同一プロトコルを v51_cushion (A1-A3,A6 の
4特徴量版) の fold2 OOS スコアで再実行する診断スクリプト。

本番 evaluation/reports/fusion_oos_fold2.json / prob_fusion/data/probs_benter_oos_fold2.parquet
/ prob_fusion/data/manifest.json は一切上書きしない。出力はすべて本ディレクトリ配下。

fit=2023-01-01..2024-12-31 / TEST=2025-01-01.. は prob_fusion/src/oos_protocol.py と同一。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from prob_fusion.src.manifest import file_sha256, write_manifest  # noqa: E402
from prob_fusion.src.market_prob import attach_market_q  # noqa: E402
from prob_fusion.src.oos_protocol import FIT_END, FIT_START, TEST_START, split_oos_periods  # noqa: E402
from prob_fusion.src.predict_fusion import fuse_dataframe, load_fusion_config  # noqa: E402
from prob_fusion.src.run_fit import load_scored_dataset  # noqa: E402
from prob_fusion.src.run_fit_oos import _fit_and_eval  # noqa: E402

VERSION = "cushion_v51_oos_fold2"
EXP_DIR = Path(__file__).resolve().parent

SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v51_cushion_fold2_oos.parquet"
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v51_cushion.parquet"


def main() -> dict:
    cfg = load_fusion_config()
    q_method = cfg.get("q_method", "proportional")
    q_power = cfg.get("q_power", 0.81)

    df = load_scored_dataset(SCORES_PATH, FEATURES_PATH)
    df = attach_market_q(df, method=q_method, power=q_power)

    fit_df, test_df = split_oos_periods(df)
    if fit_df.empty or test_df.empty:
        raise ValueError(
            f"OOS 期間が空です: fit={len(fit_df)} rows, test={len(test_df)} rows。"
        )

    formal = _fit_and_eval(fit_df, test_df, cfg, label="formal_fit_2023_2024")
    fit_2024 = fit_df[pd.to_datetime(fit_df["race_date"]) >= pd.Timestamp("2024-01-01")]
    sensitivity = _fit_and_eval(fit_2024, test_df, cfg, label="sensitivity_fit_2024_only")

    fused = fuse_dataframe(
        df,
        alpha=formal["alpha"],
        beta=formal["beta"],
        lam2=formal["lam2"],
        lam3=formal["lam3"],
        q_method=q_method,
        q_power=q_power,
        model_version=VERSION,
    )
    out_dir = EXP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / f"probs_{VERSION}.parquet"
    fused.to_parquet(probs_path, index=False)

    report = {
        "version": VERSION,
        "protocol": {
            "l1_scores": "v51_cushion (A1-A3,A6) fold2-only 5-seed ensemble "
                          "(train<2023; 2024/2025 fully OOS)",
            "fit_period": f"{FIT_START}..{FIT_END}",
            "test_period": f"{TEST_START}..",
            "caveat": "2023 was fold2 early-stopping year (weak contamination, model selection only)",
            "formal_judgment": "formal_fit_2023_2024",
        },
        "formal": formal,
        "sensitivity": sensitivity,
        "config": cfg,
    }
    report_path = out_dir / "fusion_oos_fold2_cushion_v51.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    inputs = {"scores": file_sha256(SCORES_PATH), "features": file_sha256(FEATURES_PATH)}
    write_manifest(out_dir, inputs=inputs, config=cfg, extra={"oos_report": report["protocol"], "formal": formal})

    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    return report


if __name__ == "__main__":
    main()
