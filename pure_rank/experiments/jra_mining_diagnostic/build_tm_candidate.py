"""JRA公式TMマイニングスコアを long 形式（race_id, horse_num, cand_score）に変換する。

Track A（evaluation/alpha_gate.py）の候補特徴量入力として使う。
var2.0.0 の MiningFeatureBuilder._softmax_within_race と同一の
softmax（race内正規化）で jra_tm_implied_prob を計算する。

修正（2026-07-09、docs/JV-Data.md TMレコード仕様と照合して発見）:
公式仕様「予測スコア: 000.0～100.0で設定 右から1バイト目を小数点第一位とする」より、
ming_tm.csv の生値（4桁文字列由来の整数、例"0599"→599）は本来 /10 して 59.9 と
解釈すべきだった。var2.0.0・当初の本スクリプトともにこの/10を欠落させ、生の
0-1000程度のスケールをそのまま使っていた。
ただし var2.0.0 の TM_TEMPERATURE=100 も「TM指数の単位（0-1000程度）に合わせた
温度」という同じ未較正の前提で選ばれていたため、score と temperature を同時に
1/10 すれば softmax(score/T) の出力は数学的に完全一致する
（softmax(s/10 / (T/10)) ≡ softmax(s/T)）。
そのため本修正は過去に測定済みのα-gate/init_scoreトリック実験の結果
（γ=0, Top-1=27.64%等）を変えない。将来 jra_tm_score を softmax を経由せず
直接特徴量として使う場合にのみ、この/10補正が数値的に意味を持つ。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "common" / "data" / "output" / "ming_tm" / "ming_tm.csv"
OUT_PATH = Path(__file__).resolve().parent / "data" / "jra_tm_candidate.parquet"

# 仕様書どおりの正しいスケール（000.0～100.0）とtemperatureを同時に1/10にして
# 過去の測定結果との数学的等価性を保つ。
TM_SCORE_SCALE = 10.0
TM_TEMPERATURE = 10.0
N_SLOTS = 18


def _make_race_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["year"].astype(int).astype(str).str.zfill(4)
        + df["month_day"].astype(int).astype(str).str.zfill(4)
        + df["course_code"].astype(int).astype(str).str.zfill(2)
        + df["kai"].astype(int).astype(str).str.zfill(2)
        + df["nichi"].astype(int).astype(str).str.zfill(2)
        + df["race_num"].astype(int).astype(str).str.zfill(2)
    )


def _softmax_within_race(scores: pd.Series) -> pd.Series:
    s = scores.fillna(scores.mean() if scores.notna().any() else 0)
    s_shifted = s - s.max()
    exp_s = np.exp(s_shifted / TM_TEMPERATURE)
    total = exp_s.sum()
    if total == 0:
        return pd.Series(1.0 / len(scores), index=scores.index)
    return exp_s / total


def build_tm_candidate() -> Path:
    print(f"Loading: {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH)
    raw["race_id"] = _make_race_id(raw)
    print(f"  raw races: {len(raw):,}")

    long_frames = []
    for i in range(1, N_SLOTS + 1):
        hn_col = f"mining_pred_{i}_horse_num"
        sc_col = f"mining_pred_{i}_score"
        if hn_col not in raw.columns:
            continue
        chunk = raw[["race_id", hn_col, sc_col]].rename(
            columns={hn_col: "horse_num", sc_col: "jra_tm_score"}
        )
        long_frames.append(chunk)
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["horse_num"])
    long_df["horse_num"] = long_df["horse_num"].astype(float).astype(int)
    long_df["jra_tm_score"] = pd.to_numeric(long_df["jra_tm_score"], errors="coerce") / TM_SCORE_SCALE
    long_df = long_df.drop_duplicates(subset=["race_id", "horse_num"], keep="first")
    print(f"  long rows: {len(long_df):,}, races: {long_df['race_id'].nunique():,}")

    long_df["jra_tm_implied_prob"] = long_df.groupby("race_id")["jra_tm_score"].transform(
        _softmax_within_race
    )
    p = long_df["jra_tm_implied_prob"].clip(1e-6, 1 - 1e-6)
    long_df["jra_tm_log_odds"] = np.log(p / (1 - p))

    out = long_df[["race_id", "horse_num", "jra_tm_score", "jra_tm_implied_prob", "jra_tm_log_odds"]].copy()
    out["cand_score"] = out["jra_tm_implied_prob"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, compression="snappy")
    print(f"Saved: {OUT_PATH} ({len(out):,} rows)")
    return OUT_PATH


if __name__ == "__main__":
    build_tm_candidate()
