"""place_direct: build_dataset.py

features_v39_course_slim.parquet を読み込み、禁止列検証・除外フィルタ・
target_place 付与・fold2 分割（train / es2023 / test2025）を行う。

本番 pure_rank/data/02_features/features_v39_course_slim.parquet は読み取り専用。
出力は pure_rank/experiments/place_direct/data/ 配下のみ。

使用方法:
    python pure_rank/experiments/place_direct/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT / "pure_rank" / "src"))

from common import FORBIDDEN_MARKET_COLS, load_config  # noqa: E402
from train import get_fold_split  # noqa: E402

from place_lib import apply_base_filters, compute_target_place, get_experiment_feature_cols  # noqa: E402

BASE_FEATURES = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
DATA_DIR = EXP_DIR / "data"

FOLD = 2
ES_YEAR = "2023"
TEST_START = "2025-01-01"


def _assert_no_market_columns(df: pd.DataFrame) -> None:
    """禁止列・市場情報混入の検証（build 時に一度だけ実行する）。

    market_leak_diagnostic 実験が追加する exp_* 列が、参照している本番 parquet に
    混在していないことを列名で確認する（仕様書 §2 注意事項）。
    """
    cols = set(df.columns)
    hit_exact = cols & FORBIDDEN_MARKET_COLS
    if hit_exact:
        raise ValueError(f"禁止列（完全一致）が features parquet に混入: {hit_exact}")
    hit_exp_prefix = [c for c in cols if c.lower().startswith("exp_")]
    if hit_exp_prefix:
        raise ValueError(
            f"market_leak_diagnostic 由来と思われる exp_* 列が混入: {hit_exp_prefix}"
        )


def build_place_direct_dataset() -> dict[str, Path]:
    cfg = load_config()  # 本番 train_config.json（読み取り専用）

    print(f"Loading base features (read-only): {BASE_FEATURES}")
    df = pd.read_parquet(BASE_FEATURES)
    print(f"  rows={len(df):,}, cols={df.shape[1]}")

    _assert_no_market_columns(df)
    print("  OK: no forbidden/exp_* market columns in raw features")

    df = apply_base_filters(df, cfg)
    print(f"  After filters: rows={len(df):,}, races={df['race_id'].nunique():,}")

    df = df.copy()
    df["target_place"] = compute_target_place(df["finish_rank"])
    print(f"  target_place positive rate: {df['target_place'].mean():.4f}")

    # feature_cols の妥当性を build 時にも再確認（学習・エクスポートと同一集合を保証）。
    # target_place（本実験のラベル列）が混入していないことも get_experiment_feature_cols 内で保証する。
    feature_cols = get_experiment_feature_cols(df, cfg)
    assert "target_place" not in feature_cols, "target_place が特徴量に混入（label leakage）"
    print(f"  Feature cols: {len(feature_cols)}")

    # fold2 分割: train(<2023) / es(2023) は本番 get_fold_split をそのまま再利用
    train_df, es_df = get_fold_split(df, FOLD, cfg["training"]["fold_valid_years"])
    test_df = df[pd.to_datetime(df["race_date"]) >= pd.Timestamp(TEST_START)].copy()

    print(
        f"  Train:  {len(train_df):,} rows, {train_df['race_id'].nunique():,} races "
        f"({train_df['race_date'].min().date()} - {train_df['race_date'].max().date()})"
    )
    print(
        f"  ES({ES_YEAR}): {len(es_df):,} rows, {es_df['race_id'].nunique():,} races "
        f"({es_df['race_date'].min().date()} - {es_df['race_date'].max().date()})"
    )
    print(
        f"  Test(>={TEST_START}): {len(test_df):,} rows, {test_df['race_id'].nunique():,} races "
        f"({test_df['race_date'].min().date()} - {test_df['race_date'].max().date()})"
    )

    # リーク防止チェック（§9）: train/ES に 2024 年以降のデータが混ざっていないこと
    assert train_df["race_date"].max() < pd.Timestamp("2023-01-01"), "train に2023以降混入"
    assert set(es_df["race_date"].dt.year.unique().tolist()) == {2023}, "ES が2023年以外を含む"
    assert test_df["race_date"].min() >= pd.Timestamp(TEST_START), "test に2025以前混入"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": DATA_DIR / "train_fold2.parquet",
        "es": DATA_DIR / "es_fold2_2023.parquet",
        "test": DATA_DIR / "test_2025.parquet",
    }
    train_df.to_parquet(out_paths["train"], index=False, compression="snappy")
    es_df.to_parquet(out_paths["es"], index=False, compression="snappy")
    test_df.to_parquet(out_paths["test"], index=False, compression="snappy")
    for name, path in out_paths.items():
        print(f"Saved {name}: {path}")

    return out_paths


if __name__ == "__main__":
    build_place_direct_dataset()
