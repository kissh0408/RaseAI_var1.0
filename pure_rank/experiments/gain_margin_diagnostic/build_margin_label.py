"""着差を反映した8段階ラベル lr_label_margin を構築する。

finish_rank==2 の馬のうち、当該レース優勝馬との racetime 差が
CLOSE_THRESHOLD_SEC 以下のものを「僅差2着」として label=6 に格上げする。
それ以外は従来の7段階ラベル体系をそのまま1つずつシフトする
（1着=7, 通常2着=5, 3着=4, ... 7着以下=0）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = ROOT / "pure_rank" / "data" / "02_features" / "features_v39_course_slim.parquet"
OUT_PATH = Path(__file__).resolve().parent / "data" / "lr_label_margin.parquet"
CLOSE_THRESHOLD_SEC = 0.2

LABEL_GAIN_MARGIN = [0, 1, 3, 7, 15, 31, 50, 100]


def build_margin_label(df: pd.DataFrame) -> pd.Series:
    winner_time = (
        df[df["finish_rank"] == 1]
        .drop_duplicates(subset=["race_id"], keep="first")
        .set_index("race_id")["racetime"]
    )
    winner_time_aligned = df["race_id"].map(winner_time)
    time_diff = df["racetime"] - winner_time_aligned

    is_winner = df["finish_rank"] == 1
    is_close_second = (df["finish_rank"] == 2) & (time_diff <= CLOSE_THRESHOLD_SEC)
    is_other_second = (df["finish_rank"] == 2) & ~is_close_second

    label = np.zeros(len(df), dtype="int64")
    label[is_winner.values] = 7
    label[is_close_second.values] = 6
    label[is_other_second.values] = 5
    for finish_rank, lbl in [(3, 4), (4, 3), (5, 2), (6, 1)]:
        label[(df["finish_rank"] == finish_rank).values] = lbl
    # finish_rank>=7 or invalid (0) -> label 0 (デフォルトのまま)
    return pd.Series(label, index=df.index, name="lr_label_margin")


def main() -> None:
    print(f"Loading: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH, columns=["race_id", "horse_num", "finish_rank", "racetime", "lr_label"])
    df["lr_label_margin"] = build_margin_label(df)

    n_close = int(((df["finish_rank"] == 2) & (df["lr_label_margin"] == 6)).sum())
    n_second = int((df["finish_rank"] == 2).sum())
    print(f"2着 総数: {n_second:,} / うち僅差(<= {CLOSE_THRESHOLD_SEC}s): {n_close:,} "
          f"({n_close / n_second:.1%})")
    print("\nlabel分布:")
    print(df["lr_label_margin"].value_counts().sort_index())
    print("\n従来lr_labelとの差分がある行数:", int((df["lr_label_margin"] != df["lr_label"]).sum()))

    out = df[["race_id", "horse_num", "lr_label_margin"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
