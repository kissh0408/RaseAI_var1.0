"""
export_scores.py — アンサンブルスコアの Parquet エクスポート（R-6 準備）

テスト集合（valid_end 以降）の race_id × ketto_num ごとに
LambdaRank アンサンブルスコアを書き出す。市場情報は一切含めない。

Usage:
    python pure_rank/src/export_scores.py
    python pure_rank/src/export_scores.py --split all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


def _import_pure_rank_modules():
    """pure_rank/src の重い依存を遅延 import する。

    モジュールレベルで sys.path に src を挿入すると pure_rank/src/common.py が
    トップレベルの common パッケージをシャドウし、他テスト・他モジュールの
    `common.data.*` import を壊すため、実行時のみ解決する。
    """
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from common import PROJECT_ROOT, get_feature_cols, load_config
    from evaluate import ensemble_predict, load_models
    from score_utils import attach_pure_score_z

    return PROJECT_ROOT, get_feature_cols, load_config, ensemble_predict, load_models, attach_pure_score_z


def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["filters"]
    return df[
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    ].copy()


def select_fold_model_paths(
    models_dir: Path,
    fold: int,
    expected_seeds: int | None = None,
) -> list[Path]:
    """指定 fold のシードモデルのみを列挙する（OOS スコア生成用）。"""
    paths = sorted(Path(models_dir).glob(f"lambdarank_fold{fold}_seed*.txt"))
    if not paths:
        raise ValueError(f"fold{fold} のモデルが見つかりません: {models_dir}")
    if expected_seeds is not None and len(paths) != expected_seeds:
        raise ValueError(
            f"fold{fold} のモデル数が {len(paths)} 本（期待 {expected_seeds} 本）: {models_dir}"
        )
    return paths


def export_scores(
    split: str = "test",
    fold: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    expected_seeds: int = 5,
) -> Path:
    (
        PROJECT_ROOT,
        get_feature_cols,
        load_config,
        ensemble_predict,
        load_models,
        attach_pure_score_z,
    ) = _import_pure_rank_modules()
    cfg = load_config()
    version = cfg["data"]["features_version"]
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])

    feat_path = feat_dir / f"features_{version}.parquet"
    print(f"Loading: {feat_path.name}")
    df = _apply_filters(pd.read_parquet(feat_path), cfg)

    if split == "test":
        df = df[df["race_date"] > valid_end]
    elif split == "valid":
        train_end = pd.Timestamp(cfg["training"]["train_end"])
        df = df[(df["race_date"] > train_end) & (df["race_date"] <= valid_end)]
    elif split == "train":
        train_end = pd.Timestamp(cfg["training"]["train_end"])
        df = df[df["race_date"] <= train_end]
    # split == "all" → no date filter

    if start_date is not None:
        df = df[df["race_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["race_date"] <= pd.Timestamp(end_date)]

    print(f"Split={split}: {len(df):,} rows, {df['race_id'].nunique():,} races")

    feat_cols = get_feature_cols(df, cfg)
    if fold is not None:
        import lightgbm as lgb

        model_paths = select_fold_model_paths(models_dir, fold, expected_seeds=expected_seeds)
        models = [lgb.Booster(model_file=str(p)) for p in model_paths]
        print(f"fold{fold} のみ {len(models)} モデルでスコアリング（OOS モード）")
    else:
        models = load_models(models_dir)
    df = df.copy()
    df["pure_score"] = ensemble_predict(models, df[feat_cols])
    df = attach_pure_score_z(df, score_col="pure_score", race_id_col="race_id", out_col="pure_score_z")

    out_cols = [
        "race_id", "race_date", "ketto_num", "horse_num", "horse_number",
        "course_code", "finish_rank", "pure_score", "pure_score_z",
    ]
    if "horse_number" not in df.columns:
        df["horse_number"] = df["horse_num"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].sort_values(["race_date", "race_id", "horse_num"])

    scores_root = PROJECT_ROOT / "pure_rank" / "data" / "03_scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    legacy_dir = feat_dir / "exported_scores"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    if fold is not None:
        out_path = scores_root / f"scores_{version}_fold{fold}_oos.parquet"
    elif split == "all":
        out_path = scores_root / f"scores_{version}.parquet"
    else:
        out_path = scores_root / f"scores_{version}_{split}.parquet"
    out_df.to_parquet(out_path, index=False, compression="snappy")
    print(f"Saved: {out_path} ({len(out_df):,} rows)")
    if fold is None:
        # fold限定スコアは全モデル版レガシーコピーを汚染するため書かない
        legacy_path = legacy_dir / f"scores_{version}_{split}.parquet"
        out_df.to_parquet(legacy_path, index=False, compression="snappy")
        print(f"Legacy copy: {legacy_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ensemble scores to parquet")
    parser.add_argument(
        "--split",
        choices=["test", "valid", "train", "all"],
        default="test",
        help="Data split to export (default: test)",
    )
    parser.add_argument("--fold", type=int, default=None, help="指定 fold のモデルのみ使用（OOS スコア）")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--expected-seeds", type=int, default=5)
    args = parser.parse_args()
    export_scores(
        split=args.split,
        fold=args.fold,
        start_date=args.start_date,
        end_date=args.end_date,
        expected_seeds=args.expected_seeds,
    )


if __name__ == "__main__":
    main()
