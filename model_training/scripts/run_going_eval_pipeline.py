"""Exp A→C 学習 + ゲート評価の一括ランナー（ローカル parquet / モデル必須）。

Usage:
  python model_training/scripts/run_going_eval_pipeline.py --experiment C --fast
  python model_training/scripts/run_going_eval_pipeline.py --skip-train --experiment C
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="馬場改善 Exp 学習+ゲート")
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C", "a", "b", "c"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--segment-report", type=Path, default=None)
    args = parser.parse_args()

    exp = args.experiment.upper()
    py = sys.executable

    if not args.skip_train:
        cmd = [py, str(ROOT / "model_training/scripts/run_going_experiment.py"), "--experiment", exp]
        if args.fast:
            cmd.append("--fast")
        if args.n_trials is not None:
            cmd.extend(["--n-trials", str(args.n_trials)])
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    gate_cmd = [
        py,
        str(ROOT / "model_training/scripts/evaluate_going_experiment_gate.py"),
        "--experiment",
        exp,
    ]
    if args.segment_report:
        gate_cmd.extend(["--segment-report", str(args.segment_report)])
    subprocess.run(gate_cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
