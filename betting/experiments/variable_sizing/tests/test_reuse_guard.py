"""import 再利用の静的検査（仕様書§9項目2・§2・コピー禁止）。

- sizing_lib.py が tiers_lib.assign_tier / assign_tier_batch を import しており、
  同等ロジック（境界割当）の再実装が実験ディレクトリ内に存在しないこと。
- build_dataset.py が betting.src.flat_top1.select_top1_bets / betting.src.backtest.
  load_scored_odds_frame と tiers_lib.compute_race_margin を import していること。
- betting/src/flat_top1.py / betting/experiments/confidence_tiers/tiers_lib.py の
  コピー（同名関数の完全な再実装）が本ディレクトリに存在しないこと。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
SIZING_LIB_PATH = EXP_DIR / "sizing_lib.py"
BUILD_DATASET_PATH = EXP_DIR / "build_dataset.py"

_REIMPLEMENTATION_FORBIDDEN_NAMES = {
    "assign_tier",
    "assign_tier_batch",
    "compute_race_margin",
    "select_top1_bets",
    "settle_win_bets",
    "load_scored_odds_frame",
}


def _all_py_files_excluding_tests() -> list[Path]:
    return sorted(p for p in EXP_DIR.rglob("*.py") if "tests" not in p.relative_to(EXP_DIR).parts)


def test_sizing_lib_imports_tiers_lib_assign_tier():
    text = SIZING_LIB_PATH.read_text(encoding="utf-8")
    assert "_tiers_lib.assign_tier" in text
    assert "_tiers_lib.assign_tier_batch" in text
    assert "import tiers_lib as _tiers_lib" in text


def test_sizing_lib_imports_derive_flat_fraction_mdd_functions():
    text = SIZING_LIB_PATH.read_text(encoding="utf-8")
    assert "from betting.src.derive_flat_fraction import" in text
    assert "_monthly_max_drawdown" in text
    assert "_busiest_day_exposure" in text


def test_no_reimplementation_of_reused_functions_in_experiment_dir():
    """CORE 純関数名を FunctionDef として再定義しているファイルがないこと
    （sizing_lib.py 自体は許可された import エイリアスの束縛のみで FunctionDef を
    持たないため違反にならない）。
    """
    violations = []
    for path in _all_py_files_excluding_tests():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _REIMPLEMENTATION_FORBIDDEN_NAMES:
                violations.append((str(path.relative_to(EXP_DIR)), node.name))
    assert violations == [], f"再利用対象関数の再実装が疑われる定義があります: {violations}"


def test_build_dataset_imports_required_functions():
    if not BUILD_DATASET_PATH.exists():
        return  # created later in the task sequence; guarded separately once present
    text = BUILD_DATASET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(BUILD_DATASET_PATH))

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)

    assert "select_top1_bets" in imported_names
    assert "load_scored_odds_frame" in imported_names
