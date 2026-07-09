"""市場情報混入禁止のテスト（仕様書 §8, §9, プロジェクト憲法）。

特徴量選択は pure_rank/src/common.py の get_feature_cols をそのまま import して使う
（本実験専用の特徴量選択ロジックを新規実装しない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[4]
_PURE_RANK_SRC = str(ROOT / "pure_rank" / "src")
sys.path.insert(0, _PURE_RANK_SRC)
sys.path.insert(0, str(ROOT / "pure_rank" / "experiments" / "place_direct"))

from common import FORBIDDEN_MARKET_COLS, get_feature_cols, load_config  # noqa: E402
from place_lib import compute_target_place, get_experiment_feature_cols  # noqa: E402

# pure_rank/src/common.py がトップレベル common パッケージをシャドウし、後続テスト
# （tests/test_realtime_jv_parsing.py 等）の `from common.data...` import を壊すため後始末する。
sys.path.remove(_PURE_RANK_SRC)
sys.modules.pop("common", None)

FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"


def test_feature_columns_clean():
    """実データ列名で、禁止パターンが特徴量に一切含まれないこと。"""
    cfg = load_config()
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = get_feature_cols(df, cfg)

    banned_substrings = ["odds", "popularity", "ninki", "market_log_odds", "init_score"]
    for col in feature_cols:
        lowered = col.lower()
        for banned in banned_substrings:
            assert banned not in lowered, f"禁止パターン '{banned}' が特徴量列 '{col}' に含まれる"
        # market_leak_diagnostic 実験が追加する exp_* 列（exp_win_odds 等）のプレフィックス検証。
        # "exp_" を接頭辞として含む列のみ拒否する（hist_top_grade_exp_count 等の正当な特徴量を
        # 誤検出しないよう部分文字列一致ではなく startswith で判定する）。
        assert not lowered.startswith("exp_"), f"market_leak_diagnostic 由来の exp_* 列が混入: '{col}'"

    # exact-match 禁止列も念のため確認
    assert not (set(feature_cols) & FORBIDDEN_MARKET_COLS)


def test_synthetic_forbidden_column_raises():
    """疑わしい列名（例: implied_prob）が混入した場合、get_feature_cols が拒否すること。"""
    cfg = load_config()
    df = pd.DataFrame(
        {
            "race_id": ["r1"],
            "ketto_num": ["k1"],
            "race_date": pd.to_datetime(["2020-01-01"]),
            "finish_rank": [1],
            "is_win": [1],
            "lr_label": [0],
            "surface_code": [1],
            "implied_prob": [0.5],  # 疑わしい市場列名（SUSPICIOUS_MARKET_NAME_PATTERN にヒット）
        }
    )
    with pytest.raises(ValueError):
        get_feature_cols(df, cfg)


def test_synthetic_exact_forbidden_column_excluded():
    """完全一致の禁止列（例: odds）は例外なく特徴量から除外されること。"""
    cfg = load_config()
    df = pd.DataFrame(
        {
            "race_id": ["r1"],
            "ketto_num": ["k1"],
            "race_date": pd.to_datetime(["2020-01-01"]),
            "finish_rank": [1],
            "is_win": [1],
            "lr_label": [0],
            "surface_code": [1],
            "odds": [2.5],
        }
    )
    feature_cols = get_feature_cols(df, cfg)
    assert "odds" not in feature_cols


def test_target_place_excluded_from_features():
    """target_place（本実験が付与するラベル列）が特徴量に絶対に混入しないこと。

    target_place は本番 common.FORBIDDEN_COLS に存在しないため、
    get_experiment_feature_cols() が明示的に除外していることを確認する
    （このガードがないと label leakage になる）。
    """
    cfg = load_config()
    df = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r1"],
            "ketto_num": ["k1", "k2", "k3"],
            "race_date": pd.to_datetime(["2020-01-01"] * 3),
            "finish_rank": [1, 2, 4],
            "is_win": [1, 0, 0],
            "lr_label": [2, 1, 0],
            "surface_code": [1, 1, 1],
            "hist_win_rate": [0.3, 0.2, 0.1],
        }
    )
    df["target_place"] = compute_target_place(df["finish_rank"])
    feature_cols = get_experiment_feature_cols(df, cfg)
    assert "target_place" not in feature_cols
    assert "hist_win_rate" in feature_cols
