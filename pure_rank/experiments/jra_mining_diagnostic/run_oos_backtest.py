"""init_score トリック版スコアで L2/L3 OOS 測定を走らせる（本番レポートは退避/復元）。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from betting.src.run_backtest_oos import run_backtest_oos  # noqa: E402
from prob_fusion.src.run_fit_oos import run_fit_oos  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
SCORES_PATH = EXP_DIR / "scores" / "scores_jra_init_score_fold2_oos.parquet"
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
EXP_PROB_OUT = EXP_DIR / "reports"

FUSION_REPORT = ROOT / "evaluation" / "reports" / "fusion_oos_fold2.json"
BETTING_REPORT = ROOT / "evaluation" / "reports" / "betting_backtest_oos.json"


def _backup(path: Path) -> Path | None:
    if not path.is_file():
        return None
    backup = path.with_suffix(path.suffix + ".prod_backup")
    shutil.copy2(path, backup)
    return backup


def _restore(path: Path, backup: Path | None) -> None:
    if backup is not None:
        shutil.copy2(backup, path)
        backup.unlink()
    elif path.is_file():
        path.unlink()


def main() -> None:
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(f"{SCORES_PATH} がありません。先に export_scores_init_score.py を実行してください。")

    fusion_backup = _backup(FUSION_REPORT)
    betting_backup = _backup(BETTING_REPORT)
    try:
        run_fit_oos(SCORES_PATH, FEATURES_PATH, EXP_PROB_OUT)
        shutil.copy2(FUSION_REPORT, EXP_DIR / "reports" / "fusion_oos_fold2_init_score_trick.json")

        run_backtest_oos(SCORES_PATH, FEATURES_PATH, FUSION_REPORT)
        shutil.copy2(BETTING_REPORT, EXP_DIR / "reports" / "betting_backtest_oos_init_score_trick.json")
    finally:
        _restore(FUSION_REPORT, fusion_backup)
        _restore(BETTING_REPORT, betting_backup)
        print("\n[run_oos_backtest] evaluation/reports/ の本番ファイルを復元しました。")

    print("\n実験結果:")
    print(f"  {EXP_DIR / 'reports' / 'fusion_oos_fold2_init_score_trick.json'}")
    print(f"  {EXP_DIR / 'reports' / 'betting_backtest_oos_init_score_trick.json'}")


if __name__ == "__main__":
    main()
