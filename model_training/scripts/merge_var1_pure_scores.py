"""var1.0 純粋能力スコア（レース内 z-score）を binary 学習用 parquet にマージ（R-6）。

var1 の export_scores.py 出力（市場情報なし）を var2 backtest 特徴量に left join し、
レース内 z-score 列 var1_pure_score_z を追加する。統合作業は var2 側のみ。

Usage:
    python model_training/scripts/merge_var1_pure_scores.py
    python model_training/scripts/merge_var1_pure_scores.py --scores path/to/scores.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import FEATURES_DIR  # noqa: E402

DEFAULT_BASE = FEATURES_DIR / "features_v6_going_v1_top3.parquet"
DEFAULT_OUT = FEATURES_DIR / "features_v6_going_v1_top3_var1.parquet"
DEFAULT_SCORES = (
    ROOT / "pure_rank/data/02_features/exported_scores/scores_v39_course_slim_all.parquet"
)
RAW_COL = "_var1_raw_score"
Z_COL = "var1_pure_score_z"


def _race_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if sd is None or sd < 1e-8 or not np.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype="float32")
    return ((s - mu) / sd).astype("float32")


def attach_var1_z_from_rank_preds(
    df: pd.DataFrame,
    rank_preds: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    horse_id_col: str = "horse_id",
    score_col: str = "pred_score",
) -> pd.DataFrame:
    """当日 LambdaRank 推論結果から var1_pure_score_z を inline 生成（parquet 不要）。"""
    if Z_COL in df.columns:
        return df
    scores = rank_preds.copy()
    if horse_id_col not in scores.columns and "ketto_num" in scores.columns:
        scores[horse_id_col] = scores["ketto_num"].astype(str)
    if score_col not in scores.columns:
        raise ValueError(f"rank_preds に {score_col} 列がありません")
    scores = scores.rename(columns={score_col: RAW_COL})
    scores[race_id_col] = scores[race_id_col].astype(str)
    out = df.copy()
    out[race_id_col] = out[race_id_col].astype(str)
    if horse_id_col not in out.columns and "ketto_num" in out.columns:
        out[horse_id_col] = out["ketto_num"].astype(str)
    if RAW_COL in out.columns:
        out = out.drop(columns=[RAW_COL])
    out = out.merge(
        scores[[race_id_col, horse_id_col, RAW_COL]].drop_duplicates([race_id_col, horse_id_col]),
        on=[race_id_col, horse_id_col],
        how="left",
    )
    missing = out[RAW_COL].isna()
    if missing.any():
        race_med = out.groupby(race_id_col)[RAW_COL].transform("median")
        out.loc[missing, RAW_COL] = race_med[missing]
        still = out[RAW_COL].isna()
        if still.any():
            gmed = float(out[RAW_COL].median()) if out[RAW_COL].notna().any() else 0.0
            out.loc[still, RAW_COL] = gmed
    out[Z_COL] = out.groupby(race_id_col, group_keys=False)[RAW_COL].transform(_race_zscore)
    return out.drop(columns=[RAW_COL])


def attach_var1_score_z(
    df: pd.DataFrame,
    scores_path: Path,
    *,
    race_id_col: str = "race_id",
    horse_id_col: str = "horse_id",
) -> pd.DataFrame:
    """DataFrame に var1_pure_score_z を join する（推論・当日パス向け）。"""
    if Z_COL in df.columns:
        return df
    scores = pd.read_parquet(scores_path)
    scores = scores.rename(columns={"ketto_num": horse_id_col, "ensemble_score": RAW_COL})
    scores[race_id_col] = scores[race_id_col].astype(str)
    out = df.copy()
    out[race_id_col] = out[race_id_col].astype(str)
    if RAW_COL in out.columns:
        out = out.drop(columns=[RAW_COL])
    out = out.merge(
        scores[[race_id_col, horse_id_col, RAW_COL]].drop_duplicates([race_id_col, horse_id_col]),
        on=[race_id_col, horse_id_col],
        how="left",
    )
    missing = out[RAW_COL].isna()
    if missing.any():
        race_med = out.groupby(race_id_col)[RAW_COL].transform("median")
        out.loc[missing, RAW_COL] = race_med[missing]
        still = out[RAW_COL].isna()
        if still.any():
            gmed = float(out[RAW_COL].median()) if out[RAW_COL].notna().any() else 0.0
            out.loc[still, RAW_COL] = gmed
    out[Z_COL] = out.groupby(race_id_col, group_keys=False)[RAW_COL].transform(_race_zscore)
    return out.drop(columns=[RAW_COL])


def merge_var1_scores(
    base_path: Path,
    scores_path: Path,
    out_path: Path,
) -> dict:
    print(f"[INFO] base: {base_path}")
    print(f"[INFO] scores: {scores_path}")
    df = pd.read_parquet(base_path)
    scores = pd.read_parquet(scores_path)

    scores = scores.rename(columns={"ketto_num": "horse_id", "ensemble_score": RAW_COL})
    scores["race_id"] = scores["race_id"].astype(str)
    df["race_id"] = df["race_id"].astype(str)

    for col in (Z_COL, RAW_COL, "var1_pure_score"):
        if col in df.columns:
            df = df.drop(columns=[col])

    n_before = len(df)
    merged = df.merge(
        scores[["race_id", "horse_id", RAW_COL]].drop_duplicates(["race_id", "horse_id"]),
        on=["race_id", "horse_id"],
        how="left",
    )
    assert len(merged) == n_before, "merge changed row count (duplicate keys?)"

    missing = merged[RAW_COL].isna()
    n_missing = int(missing.sum())
    if n_missing > 0:
        race_med = merged.groupby("race_id")[RAW_COL].transform("median")
        merged.loc[missing, RAW_COL] = race_med[missing]
        still_missing = merged[RAW_COL].isna().sum()
        if still_missing > 0:
            global_med = float(merged[RAW_COL].median()) if merged[RAW_COL].notna().any() else 0.0
            merged[RAW_COL] = merged[RAW_COL].fillna(global_med)

    merged[Z_COL] = merged.groupby("race_id", group_keys=False)[RAW_COL].transform(_race_zscore)
    merged = merged.drop(columns=[RAW_COL])
    merged[Z_COL] = merged[Z_COL].astype("float32")
    merged.to_parquet(out_path, index=False)
    print(f"[INFO] saved: {out_path} ({len(merged):,} rows)")

    stats = {
        "n_rows": len(merged),
        "n_races": int(merged["race_id"].nunique()),
        "n_missing_before_fill": n_missing,
        "match_rate": round(1.0 - n_missing / len(merged), 6),
        "var1_pure_score_z_mean": float(merged[Z_COL].mean()),
        "var1_pure_score_z_std": float(merged[Z_COL].std()),
    }
    print(f"  {Z_COL}: missing_before_fill={n_missing:,} ({n_missing/len(merged):.2%})")
    print(
        f"  {Z_COL}: mean={stats['var1_pure_score_z_mean']:.4f}, "
        f"std={stats['var1_pure_score_z_std']:.4f} (global; per-race mean~0)"
    )

    manifest = {
        "name": out_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_base": base_path.name,
        "source_scores": str(scores_path),
        "columns_added": [Z_COL],
        "merge_keys": ["race_id", "horse_id"],
        "stats": stats,
        "note": (
            "var1_pure_score_z = race-internal z-score of v39_course_slim LambdaRank ensemble "
            "(market-free, RaceAI_var1.0)"
        ),
    }
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] manifest: {manifest_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge var1.0 pure score z into var2 backtest features")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"Base parquet not found: {args.base}")
    if not args.scores.exists():
        raise FileNotFoundError(
            f"Scores parquet not found: {args.scores}\n"
            "Run from var1.0: python pure_rank/src/export_scores.py --split all"
        )
    merge_var1_scores(args.base, args.scores, args.out)


if __name__ == "__main__":
    main()
