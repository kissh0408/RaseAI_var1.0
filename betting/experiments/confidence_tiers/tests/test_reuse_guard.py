"""選定ロジックの import 再利用の静的検査（仕様書§11 項目11）。

build_dataset.py が betting.src.flat_top1.select_top1_bets を import しており、
同等ロジック（モデル1位馬選定・オッズ除外）の再実装（コピー）が実験ディレクトリ内に
存在しないことを機械的に確認する。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
BUILD_DATASET_PATH = EXP_DIR / "build_dataset.py"


def _all_py_files_excluding_tests() -> list[Path]:
    return sorted(p for p in EXP_DIR.rglob("*.py") if "tests" not in p.relative_to(EXP_DIR).parts)


def test_build_dataset_imports_select_top1_bets():
    text = BUILD_DATASET_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(BUILD_DATASET_PATH))

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "betting.src.flat_top1":
            if any(alias.name == "select_top1_bets" for alias in node.names):
                found = True
    assert found, "build_dataset.py は betting.src.flat_top1.select_top1_bets を import していません"


def test_no_reimplementation_of_select_top1_in_experiment_dir():
    violations = []
    for path in _all_py_files_excluding_tests():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("select_top1"):
                violations.append((str(path.relative_to(EXP_DIR)), node.name))
    assert violations == [], f"select_top1_bets の再実装が疑われる定義があります: {violations}"
