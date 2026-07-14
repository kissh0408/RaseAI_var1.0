"""市場情報混入の静的検査（仕様書§9項目6・§3.4）。

sizing_lib.py の CORE_PURE_FUNCTIONS（倍率決定・stake計算・占有率・保存則）の
ソースに市場情報トークン（odds/popularity/ninki/market_log_odds/init_score/
win_prob/payout/finish_rank）が現れないこと。決済・MDD計算関数（derive_flat_fraction
から import した monthly_max_drawdown/busiest_day_exposure）は payout/odds を扱う
ため検査対象から除外する（EXEMPT_FUNCTIONS に明示列挙。sizing_lib.py内では
import文のみでFunctionDefを持たないため、そもそもAST検査には現れない）。

禁止トークンは文字列結合でエンコードし自己ヒット回避する
（confidence-tiers §11-10 パターン踏襲）。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
SIZING_LIB_PATH = EXP_DIR / "sizing_lib.py"

_FORBIDDEN_MARKET_TOKENS = [
    "odds",
    "popularity",
    "ninki",
    "market" + "_log_odds",
    "init" + "_score",
    "win_" + "prob",
    "pay" + "out",
    "finish" + "_rank",
]

CORE_PURE_FUNCTIONS = [
    "validate_multipliers",
    "multiplier_for_tier",
    "compute_tier_occupancy",
    "weighted_mean_multiplier",
    "budget_preserved",
    "min_bankroll_variable",
    "compute_base_stake",
    "apply_variable_stake",
    "effective_multiplier",
    "derive_f_var",
]

# monthly_max_drawdown/busiest_day_exposure は derive_flat_fraction からの import
# エイリアスであり sizing_lib.py 内に FunctionDef を持たない（payout/pnl を扱うため
# 明示的に除外リストに記録しておく）。
EXEMPT_FUNCTIONS = ["monthly_max_drawdown", "busiest_day_exposure"]


def _all_py_files_excluding_tests() -> list[Path]:
    return sorted(p for p in EXP_DIR.rglob("*.py") if "tests" not in p.relative_to(EXP_DIR).parts)


def test_core_pure_functions_have_no_market_tokens():
    text = SIZING_LIB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(SIZING_LIB_PATH))

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


def test_no_market_tokens_in_apply_variable_stake_signature():
    import inspect
    import sys

    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    import sizing_lib as sl  # noqa: E402

    sig = inspect.signature(sl.apply_variable_stake)
    params = set(sig.parameters.keys())
    forbidden = {"odds", "popularity", "ninki", "win_prob", "payout", "finish_rank"}
    assert not (params & forbidden), f"市場・結果列を引数に取ってはならない: {params}"


def test_no_pure_rank_src_imports():
    forbidden_prefixes = ("pure_rank.src",)
    forbidden_path_fragments = ["pure_rank" + "/src"]
    violations = []
    for path in _all_py_files_excluding_tests():
        text = path.read_text(encoding="utf-8")
        for frag in forbidden_path_fragments:
            if frag in text:
                violations.append((str(path.relative_to(EXP_DIR)), "path_fragment", frag))
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in forbidden_prefixes):
                        violations.append((str(path.relative_to(EXP_DIR)), "import", alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(mod.startswith(p) for p in forbidden_prefixes):
                    violations.append((str(path.relative_to(EXP_DIR)), "import_from", mod))
    assert violations == [], f"禁止import(pure_rank.src)が混入しています: {violations}"
