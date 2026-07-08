"""fold限定モデル選択（OOSスコアエクスポート用）のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from export_scores import select_fold_model_paths


def _touch_models(models_dir: Path, folds: dict[int, list[int]]) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for fold, seeds in folds.items():
        for seed in seeds:
            (models_dir / f"lambdarank_fold{fold}_seed{seed}.txt").write_text("stub")


def test_selects_only_requested_fold_models(tmp_path: Path) -> None:
    _touch_models(tmp_path, {1: [42, 43], 2: [42, 43, 44, 45, 46], 3: [42]})
    paths = select_fold_model_paths(tmp_path, fold=2)
    assert len(paths) == 5
    assert all("fold2" in p.name for p in paths)
    assert paths == sorted(paths)


def test_raises_when_fold_seed_count_mismatch(tmp_path: Path) -> None:
    _touch_models(tmp_path, {2: [42, 43, 44]})  # 3本しかない
    with pytest.raises(ValueError, match="fold2"):
        select_fold_model_paths(tmp_path, fold=2, expected_seeds=5)


def test_raises_when_no_models(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="fold2"):
        select_fold_model_paths(tmp_path, fold=2)
