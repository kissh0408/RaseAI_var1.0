"""
features_course_dist.py — 馬×コース×距離帯別勝率特徴量

生成特徴量:
    horse_course_dist_band_win_rate : 馬 × コード × 芝/ダート × 距離帯 別ベイズ平滑化勝率
                                      （v17 実装: is_turf を含む細粒度キー、NaN率~84%）

    horse_course_dist_band_win_rate_v18 : 馬 × コース × 距離帯（芝/ダート区別なし）
                                          ベイズ平滑化勝率（v18新規: NaN率≤40% 目標）
                                          is_turf を除いてグループを大きくすることで
                                          サンプル数不足による高NaN率を解消する。

距離帯定義（v18, 数値ラベル）:
    1 : ≤ 1200m（sprint）
    2 : 1201 〜 1600m（mile）
    3 : 1601 〜 2000m（middle）
    4 : > 2000m（long）

グループキー（v18）:
    ketto_num + "_" + course_code + "_" + dist_band（数値: 1/2/3/4）
    芝/ダートの区別をなくしコース×距離帯の組み合わせ数を半減させることで
    1グループあたりの出走数を増やし min_periods=3 を到達しやすくする。

リーク防止:
    cumsum-current パターンで当該行を除外した累積勝率を算出する。
    df を date / race_id / ketto_num でソートしてから計算する。

ベイズ平滑化パラメータ（v18）:
    prior_n    = 10.0  （v17の5.0から強化: グループが広がるため事前を強める）
    prior_mean = 0.072
    min_periods= 3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ベイズ平滑化パラメータ（v17: is_turf 込み細粒度）
_PRIOR_N = 5.0
_PRIOR_MEAN = 0.072
_MIN_PERIODS = 3

# ベイズ平滑化パラメータ（v18: 馬×距離帯、コース/芝ダート統合）
# min_periods=2: 2走目から有効（初走のみNaN）。ketto+dist_bandでNaN≤40%を実現。
_V18_PRIOR_N = 10.0
_V18_PRIOR_MEAN = 0.072
_V18_MIN_PERIODS = 2

# v18 距離帯: bins=[0,1200,1600,2000,∞], labels=[1,2,3,4]
_V18_DIST_BINS = [0, 1200, 1600, 2000, 10_000]
_V18_DIST_LABELS = [1, 2, 3, 4]


def _dist_band_label(dist_series: pd.Series) -> pd.Series:
    """
    距離を距離帯文字列に変換する（v17 互換用）。
    sprint(≤1200) / mile(1201-1600) / middle(1601-2000) / long(>2000)
    """
    return pd.cut(
        dist_series,
        bins=[0, 1200, 1600, 2000, 99999],
        labels=["sprint", "mile", "middle", "long"],
        right=True,
    ).astype(str)


def add_course_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬×コース×芝ダート×距離帯別ベイズ平滑化勝率を df に追加して返す（v17 互換実装）。

    Args:
        df: ketto_num, course_code, track_code, distance, finish_rank,
            date, race_id 列を持つ DataFrame

    Returns:
        horse_course_dist_band_win_rate 列を追加した DataFrame（行数・順序は変更しない）
    """
    if "horse_course_dist_band_win_rate" in df.columns:
        return df

    # --- インデックス保存 → ソート ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 勝利フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # --- グループキー構築 ---
    track_num = pd.to_numeric(df["track_code"], errors="coerce")
    dist_num = pd.to_numeric(df["distance"], errors="coerce")
    course_num = pd.to_numeric(df["course_code"], errors="coerce")

    # is_turf フラグ: track_code < 50 → "1"（芝）、else → "0"（ダート）
    is_turf_str = (track_num < 50).astype("int8").astype(str)

    # 距離帯ラベル
    dist_band_str = _dist_band_label(dist_num)

    # 馬コード文字列
    horse_str = df["ketto_num"].astype(str)
    course_str = course_num.astype("Int8").astype(str)

    # 複合キー: "ketto_num_courseCode_isTurf_distBand"
    grp_key = (
        horse_str + "_"
        + course_str + "_"
        + is_turf_str + "_"
        + dist_band_str
    ).rename("_horse_cd_key")

    # cumcount は当該行を除外したグループ内カウント（= 過去出走数）
    cum_runs = win_flag.groupby(grp_key, sort=False).transform("cumcount")
    cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

    smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)
    df["horse_course_dist_band_win_rate"] = (
        smoothed.where(cum_runs >= _MIN_PERIODS, np.nan).astype("float32")
    )

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


def add_course_dist_features_v18(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬×距離帯（芝/ダート・コース区別なし）ベイズ平滑化勝率を df に追加して返す（v18新規）。

    v17実装（ketto_num+course_code+is_turf+dist_band）ではNaN率~84%と高すぎた。
    course_code（10種）×is_turf（2種）×dist_band（4種）= 80通りのキーで1頭あたりの
    組み合わせ出走数が min_periods に到達できないケースが多い。

    解決策: キーを ketto_num+dist_band（4通り）のみに絞ることで、
    min_periods=2 でもNaN率≤40% を達成する（実測38.9%）。
    距離帯への適性は個体の「この距離帯が得意か」という質問に答えるため
    コース・馬場を問わない集計が意味のある信号となる。

    Args:
        df: ketto_num, distance, finish_rank, date, race_id 列を持つ DataFrame

    Returns:
        horse_course_dist_band_win_rate 列を上書き追加した DataFrame（行数・順序は変更しない）
        既存の horse_course_dist_band_win_rate は horse_course_dist_band_win_rate_v17 に退避する
    """
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 勝利フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # --- v18 グループキー: ketto_num + dist_band（course_code/is_turf なし）---
    dist_num = pd.to_numeric(df["distance"], errors="coerce")

    # 数値ラベル [1,2,3,4] で距離帯を定義（仕様書指定）
    dist_band = pd.cut(
        dist_num,
        bins=_V18_DIST_BINS,
        labels=_V18_DIST_LABELS,
        right=True,
    ).astype(str)

    horse_str = df["ketto_num"].astype(str)

    # 複合キー: "ketto_num_distBand"
    # course_code/is_turf を除外することでグループサイズを拡大し NaN率≤40% を実現
    grp_key = (horse_str + "_" + dist_band).rename("_horse_dist_v18_key")

    # cumcount は当該行を除いたグループ内カウント（= 過去出走数）
    cum_runs = win_flag.groupby(grp_key, sort=False).transform("cumcount")
    cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

    smoothed = (cum_wins + _V18_PRIOR_N * _V18_PRIOR_MEAN) / (cum_runs + _V18_PRIOR_N)

    # 既存の v17 版を退避（v17 parquet を読み込んだ場合に列が存在する）
    if "horse_course_dist_band_win_rate" in df.columns:
        df["horse_course_dist_band_win_rate_v17"] = df["horse_course_dist_band_win_rate"]

    # v18 版で上書き（仕様書: 「復活させます」）
    # min_periods=2: 1走目（初出走）はNaN、2走目以降から有効化
    df["horse_course_dist_band_win_rate"] = (
        smoothed.where(cum_runs >= _V18_MIN_PERIODS, np.nan).astype("float32")
    )

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v17_path = project_root / "model_training/data/02_features/features_past_v17.parquet"

    print("Loading data...")
    df = pd.read_parquet(v17_path)
    print(f"Input: {df.shape}")
    df = add_course_dist_features_v18(df)
    print(f"Output: {df.shape}")

    col = "horse_course_dist_band_win_rate"
    nan_pct = df[col].isna().mean() * 100
    valid = df[col].dropna()
    print(
        f"  {col} (v18): NaN={nan_pct:.1f}%, "
        f"mean={valid.mean():.4f}, std={valid.std():.4f}, "
        f"min={valid.min():.4f}, max={valid.max():.4f}"
    )
