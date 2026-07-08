"""Champion モデル再学習 + 評価パイプライン。

実行:
  python model_training/scripts/run_champion_pipeline.py
  python model_training/scripts/run_champion_pipeline.py --skip-build --skip-train
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "model_training" / "scripts"
MODELS = ROOT / "model_training" / "models"


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}\n[{label}]\n{'='*60}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--full-features", action="store_true", help="v2→v6 フル再生成")
    parser.add_argument("--eval-only", action="store_true", help="評価のみ")
    args = parser.parse_args()

    py = sys.executable
    summary: dict = {"started_at": datetime.now().isoformat(timespec="seconds"), "steps": []}

    if not args.skip_build and not args.eval_only:
        build_cmd = [py, str(SCRIPTS / "build_champion_features.py"), "--legacy-alias"]
        if args.full_features:
            build_cmd.append("--full")
        _run(build_cmd, "build champion features")
        summary["steps"].append("build_features")

    if not args.skip_train and not args.eval_only:
        _run([py, str(ROOT / "model_training/src/train.py"), "ensemble"], "binary champion")
        summary["steps"].append("train_binary")

        _run([py, str(SCRIPTS / "train_top3_ranker_ensemble.py")], "top3 ranker")
        summary["steps"].append("train_top3")

        _run([py, str(SCRIPTS / "train_lambdarank_top3_ensemble.py")], "lambdarank top3")
        summary["steps"].append("train_lambdarank")

    for source in ("win", "top3", "lambdarank"):
        _run(
            [py, str(SCRIPTS / "combo_rank_hit_rates.py"), "--prob-source", source],
            f"combo KPI ({source})",
        )
        summary["steps"].append(f"eval_{source}")

    _run([py, str(ROOT / "strategy/src/backtest.py")], "backtest")
    summary["steps"].append("backtest")

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = MODELS / "champion_pipeline_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SAVE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
