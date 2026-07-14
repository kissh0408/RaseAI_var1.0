"""L1 不使用の静的検査（仕様書 §9 項目14）。

実験ディレクトリ配下の全 .py（tests 含む）のソース文字列に禁止トークンが
一切含まれないことを確認する。禁止トークン定数自体はエンコード（文字結合）して
自己ヒットを回避する。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]

# 自己ヒット回避のため文字列結合でエンコードする
_FORBIDDEN_TOKENS = [
    "pure" + "_score",
    "scores" + "_v39",
    "03" + "_scores",
    "features" + "_v39",
    "02" + "_features",
    "pure_rank" + ".src",
    "pure_rank" + "/src",
]

_FORBIDDEN_IMPORT_PREFIXES = ("pure_rank.src",)


def _all_py_files() -> list[Path]:
    return sorted(EXP_DIR.rglob("*.py"))


def test_no_forbidden_tokens_in_source():
    violations = []
    for path in _all_py_files():
        if path == Path(__file__):
            continue  # このテスト自身は禁止トークンをエンコードしているため除外
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            if token in text:
                violations.append((str(path.relative_to(EXP_DIR)), token))
    assert violations == [], f"L1資産トークンが混入しています: {violations}"


def test_no_pure_rank_src_imports():
    violations = []
    for path in _all_py_files():
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES):
                        violations.append((str(path.relative_to(EXP_DIR)), alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(mod.startswith(p) for p in _FORBIDDEN_IMPORT_PREFIXES):
                    violations.append((str(path.relative_to(EXP_DIR)), mod))
    assert violations == [], f"禁止importが混入しています: {violations}"
