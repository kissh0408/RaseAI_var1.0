"""Champion binary モデル用特徴量の共通ロジック（学習 parquet / 当日 serve で共有）。

patch スクリプトのロジックをここに集約し、build_features_v6_going_v1 と
build_today_features の両方から呼ぶ。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features_going_v1 import add_going_v1_features, going_v1_column_names
from pipeline_common import FEATURES_DIR

CHAMPION_PARQUET = "features_v6_going_v1_top3.parquet"
LEGACY_CHAMPION_PARQUET = "features_v6_going_v2_top3.parquet"

TOP3_PAST_COLS: tuple[str, ...] = (
    "last5_rank_std",
    "top3_rate_career",
    "top3_rate_class",
)

V6_HEAVY_COLS: tuple[str, ...] = (
    "heavy_track_aptitude",
    "heavy_agari_diff",
    "sire_heavy_win_rate",
    "is_heavy_track",
    "sire_heavy_interaction",
    "agari_heavy_interaction",
)


def _prior_mean(values: pd.Series, group: pd.Series) -> pd.Series:
    filled = values.fillna(0.0)
    notna = values.notna().astype(float)
    cum_sum = filled.groupby(group).cumsum() - filled
    cum_cnt = notna.groupby(group).cumsum() - notna
    return (cum_sum / cum_cnt.replace(0.0, np.nan)).astype(float)


def add_v6_heavy_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """features_v6 の馬×馬場交互作用列を追加（create_features_v6 と同一）。"""
    out = df.copy()
    if all(c in out.columns for c in V6_HEAVY_COLS):
        return out

    out["race_date"] = pd.to_datetime(out["race_date"])
    out = out.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    is_heavy = out["track_condition_code"] >= 2
    norm_rank = (out["finish_rank"] / out["horse_count"]).where(out["finish_rank"] > 0)

    heavy_rank = _prior_mean(norm_rank.where(is_heavy), out["horse_id"])
    good_rank = _prior_mean(norm_rank.where(~is_heavy), out["horse_id"])
    out["heavy_track_aptitude"] = good_rank - heavy_rank

    agari = out["agari3f"].where(out["agari3f"] > 0)
    heavy_agari = _prior_mean(agari.where(is_heavy), out["horse_id"])
    good_agari = _prior_mean(agari.where(~is_heavy), out["horse_id"])
    out["heavy_agari_diff"] = heavy_agari - good_agari

    win = out["is_win"].astype(float).where(out["finish_rank"] > 0)
    out["sire_heavy_win_rate"] = _prior_mean(win.where(is_heavy), out["sire_id"])

    out["is_heavy_track"] = (out["track_condition_code"] >= 3).astype(int)
    out["sire_heavy_interaction"] = out["is_heavy_track"] * out["sire_heavy_win_rate"].fillna(0)
    out["agari_heavy_interaction"] = out["is_heavy_track"] * out["heavy_agari_diff"].fillna(0)
    return out


def add_top3_past_features(df: pd.DataFrame) -> pd.DataFrame:
    """shift(1) 付き top3 past 特徴量（PastPerformanceBuilder と同等 intent）。"""
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"])
    out = out.sort_values(["horse_id", "race_date", "race_id"]).reset_index(drop=True)

    valid_rank = out["finish_rank"].where(out["finish_rank"] > 0)
    grp = out.groupby("horse_id", sort=False)

    out["last5_rank_std"] = grp["finish_rank"].transform(
        lambda x: x.where(x > 0).shift(1).rolling(5, min_periods=1).std()
    )
    out["_is_top3"] = (valid_rank <= 3).astype(float)
    out["top3_rate_career"] = grp["_is_top3"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    if "grade_code" in out.columns:
        out["top3_rate_class"] = (
            out.assign(_t3=out["_is_top3"])
            .sort_values("race_date")
            .groupby(["horse_id", "grade_code"], group_keys=False)["_t3"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    out = out.drop(columns=["_is_top3"], errors="ignore")

    for col in TOP3_PAST_COLS:
        if col in out.columns:
            med = out[col].median() if out[col].notna().any() else 0.0
            out[col] = out[col].fillna(med)
    return out


def apply_champion_feature_stack(df: pd.DataFrame) -> pd.DataFrame:
    """v4 相当以降の champion 追加列を一括適用（当日 serve 用）。"""
    out = add_v6_heavy_track_features(df)
    going_cols = going_v1_column_names()
    if not all(c in out.columns for c in going_cols):
        drop = [c for c in going_cols if c in out.columns]
        if drop:
            out = out.drop(columns=drop)
        out = add_going_v1_features(out)
    if not all(c in out.columns for c in TOP3_PAST_COLS):
        out = add_top3_past_features(out)
    return out


def validate_champion_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """不足列名を返す（空なら OK）。"""
    return [c for c in required if c not in df.columns]


def build_champion_parquet(
    *,
    input_path: Path | None = None,
    output_name: str = CHAMPION_PARQUET,
) -> pd.DataFrame:
    """v6_going_v1 → top3 付与 → champion parquet 保存。"""
    src = input_path or (FEATURES_DIR / "features_v6_going_v1.parquet")
    if not src.exists():
        raise FileNotFoundError(f"入力 parquet がありません: {src}")

    print(f"[champion] load {src.name}")
    df = pd.read_parquet(src)
    df = add_top3_past_features(df)

    out_path = FEATURES_DIR / output_name
    df.to_parquet(out_path, index=False)
    print(f"[champion] saved {out_path.name} ({len(df):,} rows × {df.shape[1]} cols)")

    manifest_path = FEATURES_DIR / output_name.replace(".parquet", "_manifest.json")
    base: dict = {}
    base_manifest = FEATURES_DIR / "features_v6_going_v1_manifest.json"
    if base_manifest.exists():
        base = json.loads(base_manifest.read_text(encoding="utf-8"))
    manifest = {
        **base,
        "name": output_name.replace(".parquet", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": src.name,
        "rows": len(df),
        "pipeline": "champion_features.build_champion_parquet",
        "columns_added": list(base.get("columns_added", [])) + list(TOP3_PAST_COLS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return df
