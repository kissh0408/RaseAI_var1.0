"""性能主張の不在の静的検査（仕様書§9項目10・§0.3・§4.4）。

run_v0/v1/v2 の各スクリプトのソース文字列リテラルに「黒字」「改善」「優位」等の
禁止語が現れないこと（これらのスクリプトは disclaimer/caveats を
sizing_lib.build_result_envelope 経由で埋め込むのみで、独自の自由記述文字列として
禁止語を書かないことを担保する）。sizing_lib.py 自体は DISCLAIMER/CAVEATS 定数の
「定義元」であり、定型文には否定文脈で「黒字化を保証するものではない」「優位を
予測しない」等の語が不可避的に含まれるため検査対象外とする（confidence-tiers
run_stage1/2 と同型のパターン）。
"""

from __future__ import annotations

import ast
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent

RUN_SCRIPTS = [
    EXP_DIR / "run_v0_occupancy.py",
    EXP_DIR / "run_v1_risk_valid.py",
    EXP_DIR / "run_v2_valid_report.py",
]

_BANNED_WORDS = ["黒字", "改善", "優位"]


def _string_literals(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    return literals


def test_run_scripts_contain_no_banned_performance_words():
    violations = []
    for path in RUN_SCRIPTS:
        if not path.exists():
            continue
        for lit in _string_literals(path):
            for word in _BANNED_WORDS:
                if word in lit:
                    violations.append((path.name, word, lit))
    assert violations == [], f"性能主張を示唆する禁止語が見つかりました: {violations}"


def test_run_scripts_use_build_result_envelope():
    for path in RUN_SCRIPTS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "build_result_envelope" in text, f"{path.name} は sizing_lib.build_result_envelope を使用していません"


def test_roi_note_template_is_neutral_and_reused():
    import sys

    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    import sizing_lib as sl  # noqa: E402

    for word in _BANNED_WORDS:
        assert word not in sl.ROI_NOTE_TEMPLATE

    for path in RUN_SCRIPTS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "roi_note" in text:
            assert "ROI_NOTE_TEMPLATE" in text
