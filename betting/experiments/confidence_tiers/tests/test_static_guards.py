"""市場情報混入・L1非追加の静的検査（仕様書§11 項目10,12）。

項目10: tiers_lib.py の margin・境界・階層割当関数群（CORE_PURE_FUNCTIONS）のソースに
市場情報トークン（odds/popularity/ninki/market_log_odds/init_score/market_q/
ln_market/win_prob）が現れないこと。Δ計算・ベースライン関数（例:
cluster_bootstrap_delta_p_value）は仕様書上「odds引数を持つため検査対象から除外」と
されているためチェック対象外（EXEMPT_FUNCTIONS に明示列挙）。プロース中の解説コメント
（モジュールdocstring等）はコード本体ではないため対象外とし、CORE_PURE_FUNCTIONS の
関数本体（シグネチャ・docstring・実装）のみを検査する。

項目12: 実験ディレクトリの全 .py（tests除く）が pure_rank.src を import していないこと
（L1特徴量非追加の機械的担保）。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
TIERS_LIB_PATH = EXP_DIR / "tiers_lib.py"

# 自己ヒット回避のため文字列結合でエンコードする
_FORBIDDEN_MARKET_TOKENS = [
    "odds",
    "popularity",
    "ninki",
    "market" + "_log_odds",
    "init" + "_score",
    "market" + "_q",
    "ln_" + "market",
    "win_" + "prob",
]

# 仕様書§11項目10で検査対象とする「margin・境界・階層割当」関数群
CORE_PURE_FUNCTIONS = [
    "compute_race_margin",
    "assign_tier",
    "assign_tier_batch",
    "compute_quartile_boundaries",
]

# odds引数を持つため検査対象から除外する関数（仕様書§11項目10の除外規定）。
# 本実装ではこれらも生odds列ではなく決済済みstake/payout配列のみを受け取るが、
# Δ計算・ベースライン関連という性質上、除外リストに明示しておく。
EXEMPT_FUNCTIONS = [
    "compute_roi",
    "compute_delta",
    "per_race_payout_diff",
    "cluster_bootstrap_delta_p_value",
    "_bootstrap_roi_diff_samples",
    "cluster_bootstrap_ordering_contrast",
]

_FORBIDDEN_IMPORT_PREFIXES = ("pure_rank.src",)
_FORBIDDEN_IMPORT_PATH_FRAGMENTS = ["pure_rank" + "/src"]


def _all_py_files_excluding_tests() -> list[Path]:
    return sorted(p for p in EXP_DIR.rglob("*.py") if "tests" not in p.relative_to(EXP_DIR).parts)


def test_core_pure_functions_have_no_market_tokens():
    text = TIERS_LIB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(TIERS_LIB_PATH))

    checked = set()
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in CORE_PURE_FUNCTIONS:
            checked.add(node.name)
            segment = ast.get_source_segment(text, node) or ""
            for token in _FORBIDDEN_MARKET_TOKENS:
                if token.lower() in segment.lower():
                    violations.append((node.name, token))
    assert checked == set(CORE_PURE_FUNCTIONS), f"検査対象関数が見つかりません: {set(CORE_PURE_FUNCTIONS) - checked}"
    assert violations == [], f"market列トークンが混入しています: {violations}"


def test_exempt_functions_are_disjoint_from_core_functions():
    assert not (set(EXEMPT_FUNCTIONS) & set(CORE_PURE_FUNCTIONS))


def test_no_pure_rank_src_imports():
    violations = []
    for path in _all_py_files_excluding_tests():
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
    assert violations == [], f"禁止import(pure_rank.src)が混入しています: {violations}"


def test_no_pure_rank_src_path_fragments_in_source():
    # importだけでなく文字列パスとしての "pure_rank/src" 埋め込みも禁止する。
    violations = []
    for path in _all_py_files_excluding_tests():
        text = path.read_text(encoding="utf-8")
        for frag in _FORBIDDEN_IMPORT_PATH_FRAGMENTS:
            if frag in text:
                violations.append((str(path.relative_to(EXP_DIR)), frag))
    assert violations == [], f"pure_rank/src へのパス参照が混入しています: {violations}"
