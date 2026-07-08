"""features_v6: 馬×馬場状態の適性交互作用特徴量。

重要度分析（2026-06-11）で馬場系が1.45%と低く、原因はレース内全馬同値の
馬場特徴量が残差学習で順位づけに寄与できないため。馬ごとの適性差を表す
交互作用特徴量で「ダート×稍重 ROI=69%」等の道悪弱点に対処する。

追加特徴量（全て shift 相当の「当該レース除外・過去のみ」集計）:
  - heavy_track_aptitude : 道悪(稍重以上)と良馬場の過去着順率差（正=道悪巧者）
  - heavy_agari_diff     : 道悪と良馬場の過去上がり3F差（負=道悪で速い）
  - sire_heavy_win_rate  : 父の道悪限定 過去勝率
  - is_heavy_track       : 重・不良のみ1（稍重は0）のスパースフラグ
  - sire_heavy_interaction / agari_heavy_interaction : 重・不良時のみ非ゼロの交互作用

入力: features_v4.parquet → 出力: features_v6.parquet (+manifest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))
from pipeline_common import FEATURES_DIR, save_features


def _prior_mean(values: pd.Series, group: pd.Series) -> pd.Series:
    """グループ内で「現在行を除く過去行のみ」の平均（条件外の行はNaNで渡す）。

    cumsum/cumcount ベースのベクトル化実装。値がNaNの行はカウントされないが、
    その行でも過去平均は参照できる（例: 良馬場出走時にも道悪適性を参照可能）。
    """
    filled = values.fillna(0.0)
    notna = values.notna().astype(float)
    cum_sum = filled.groupby(group).cumsum() - filled
    cum_cnt = notna.groupby(group).cumsum() - notna
    return (cum_sum / cum_cnt.replace(0.0, np.nan)).astype(float)


def create_features_v6() -> None:
    src = FEATURES_DIR / "features_v4.parquet"
    print("features_v4.parquet 読み込み中...")
    df = pd.read_parquet(src)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    is_heavy = df["track_condition_code"] >= 2  # 稍重・重・不良
    # 着順率（頭数正規化、小さいほど好走）。未確定(finish_rank=0)はNaN扱い
    norm_rank = (df["finish_rank"] / df["horse_count"]).where(df["finish_rank"] > 0)

    heavy_rank = _prior_mean(norm_rank.where(is_heavy), df["horse_id"])
    good_rank = _prior_mean(norm_rank.where(~is_heavy), df["horse_id"])
    # 正 = 道悪の方が着順率が良い（道悪巧者）
    df["heavy_track_aptitude"] = good_rank - heavy_rank

    agari = df["agari3f"].where(df["agari3f"] > 0)
    heavy_agari = _prior_mean(agari.where(is_heavy), df["horse_id"])
    good_agari = _prior_mean(agari.where(~is_heavy), df["horse_id"])
    # 負 = 道悪でも上がりが落ちない
    df["heavy_agari_diff"] = heavy_agari - good_agari

    win = df["is_win"].astype(float).where(df["finish_rank"] > 0)
    df["sire_heavy_win_rate"] = _prior_mean(win.where(is_heavy), df["sire_id"])

    # 重・不良のみ非ゼロのスパース交互作用（良馬場ノイズを避ける）
    df["is_heavy_track"] = (df["track_condition_code"] >= 3).astype(int)
    df["sire_heavy_interaction"] = df["is_heavy_track"] * df["sire_heavy_win_rate"].fillna(0)
    df["agari_heavy_interaction"] = df["is_heavy_track"] * df["heavy_agari_diff"].fillna(0)

    new_cols = [
        "heavy_track_aptitude",
        "heavy_agari_diff",
        "sire_heavy_win_rate",
        "is_heavy_track",
        "sire_heavy_interaction",
        "agari_heavy_interaction",
    ]
    for c in new_cols:
        print(f"  {c}: NaN率={df[c].isna().mean():.1%}, mean={df[c].mean():.4f}")

    save_features(df, "features_v6")
    print(f"完了: {len(df):,}行 × {df.shape[1]}列")


if __name__ == "__main__":
    create_features_v6()
