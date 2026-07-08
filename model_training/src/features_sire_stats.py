"""
features_sire_stats.py — 種牡馬・母父馬別統計特徴量

生成特徴量:
    sire_debut_win_rate          : p_sire 別 grade_code==1 勝率（ベイズ平滑化）
    sire_dirt_win_rate           : p_sire 別 track_code between(20,29) 勝率（ベイズ平滑化）
    dam_sire_dirt_win_rate       : p_dam_sire 別 track_code between(20,29) 勝率（ベイズ平滑化）
    dam_sire_debut_win_rate      : p_dam_sire 別 grade_code==1 勝率（ベイズ平滑化）
    sire_debut_distance_match    : p_sire 産駒の全レースを対象に「当該レースの距離帯（±200m近似ビン）」での平均着順スコア（低いほど良い）。変数名に "debut" を含むが grade_code による絞り込みは行わず、全レース・全馬場種別が集計対象。
    sire_long_turf_win_rate      : p_sire 別 芝長距離（track_code<50 かつ distance>2000）勝率（ベイズ平滑化）
    sire_turf_win_rate           : p_sire 別 芝全体（track_code<50）勝率（ベイズ平滑化）
    nick_win_rate                : (p_sire, p_dam_sire) 複合キー別勝率（ベイズ平滑化）

リーク防止:
    種牡馬統計は「当該レース以前の産駒成績」のみから計算する。
    df をソートしてから cumcount / cumsum を使い、当該行を除外した累積を算出する。
    ベイズ平滑化で少数例時の偏りを抑制する。

ベイズ平滑化パラメータ:
    デビュー戦 (grade_code==1)     : prior_n=20, prior_mean=0.066
    ダート (track_code 20-29)        : prior_n=20, prior_mean=0.103
    芝長距離 (track_code<50, d>2000): prior_n=20, prior_mean=0.072, min_periods=10
    芝全体 (track_code<50)          : prior_n=20, prior_mean=0.0714, min_periods=10
    ニックス (p_sire × p_dam_sire)  : prior_n=30, prior_mean=0.072, min_periods=10
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# ベイズ平滑化勝率の共通計算
# ------------------------------------------------------------------

def _bayesian_win_rate(
    df: pd.DataFrame,
    group_col: str,
    win_flag: pd.Series,
    subset_mask: pd.Series,
    prior_n: float,
    prior_mean: float,
) -> pd.Series:
    """
    subset_mask==True の行のみを対象に group_col 別ベイズ平滑化勝率を計算する。

    subset_mask==False の行では NaN を返す（例: ダート限定統計を芝レースに適用しない）。

    リーク防止: cumcount / cumsum で当該行を除外した累積を使用する。
    """
    # subset_mask 行のみで win_flag を有効化（それ以外は 0）
    masked_win = (win_flag * subset_mask.astype("int8")).astype("int8")

    # subset_mask 行のみの出走数累積（当該行除外）
    cum_subset_runs = (
        subset_mask.astype("int8")
        .groupby(df[group_col], sort=False)
        .cumsum() - subset_mask.astype("int8")
    )
    cum_subset_wins = (
        masked_win
        .groupby(df[group_col], sort=False)
        .cumsum() - masked_win
    )

    smoothed = (
        (cum_subset_wins + prior_n * prior_mean)
        / (cum_subset_runs + prior_n)
    )

    # subset 外の行は NaN
    return smoothed.where(subset_mask, np.nan).astype("float32")


def add_sire_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    種牡馬・母父馬別統計特徴量を df に追加して返す。

    Args:
        df : features_past_v10 など（p_sire, p_dam_sire, grade_code,
             track_code, distance, finish_rank, date 列を持つ DataFrame）

    Returns:
        新特徴量 5 列を追加した DataFrame（行数・順序は変更しない）
    """
    # --- インデックス保存 → 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 勝利フラグ（finish_rank == 1） ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # --- grade_code / track_code を数値化 ---
    grade_num = pd.to_numeric(df["grade_code"], errors="coerce")
    track_num = pd.to_numeric(df["track_code"], errors="coerce")

    # デビュー戦マスク（grade_code == 1）
    debut_mask = (grade_num == 1).fillna(False)
    # ダートマスク: JV-Link track_code 20-29 がダートコース（23=右ダート, 24=左ダート）
    # 旧閾値 >=50 は砂特殊コース(52-57)の3%しか捕捉できず sire_dirt_win_rate が97%NaNになっていた
    dirt_mask = track_num.between(20, 29).fillna(False)

    # ------------------------------------------------------------------
    # 1. sire_debut_win_rate: p_sire 別 grade_code==1 勝率
    # ------------------------------------------------------------------
    if "p_sire" in df.columns and "sire_debut_win_rate" not in df.columns:
        _DEBUT_PRIOR_N = 20.0
        _DEBUT_PRIOR_MEAN = 0.066  # デビュー戦平均勝率
        df["sire_debut_win_rate"] = _bayesian_win_rate(
            df, "p_sire", win_flag, debut_mask,
            _DEBUT_PRIOR_N, _DEBUT_PRIOR_MEAN,
        )
    elif "sire_debut_win_rate" not in df.columns:
        df["sire_debut_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 2. sire_dirt_win_rate: p_sire 別 track_code between(20,29) 勝率
    # ------------------------------------------------------------------
    if "p_sire" in df.columns and "sire_dirt_win_rate" not in df.columns:
        _DIRT_PRIOR_N = 20.0
        _DIRT_PRIOR_MEAN = 0.103  # ダート平均勝率
        df["sire_dirt_win_rate"] = _bayesian_win_rate(
            df, "p_sire", win_flag, dirt_mask,
            _DIRT_PRIOR_N, _DIRT_PRIOR_MEAN,
        )
    elif "sire_dirt_win_rate" not in df.columns:
        df["sire_dirt_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 3. dam_sire_dirt_win_rate: p_dam_sire 別 track_code between(20,29) 勝率
    # ------------------------------------------------------------------
    if "p_dam_sire" in df.columns and "dam_sire_dirt_win_rate" not in df.columns:
        _DIRT_PRIOR_N = 20.0
        _DIRT_PRIOR_MEAN = 0.103
        df["dam_sire_dirt_win_rate"] = _bayesian_win_rate(
            df, "p_dam_sire", win_flag, dirt_mask,
            _DIRT_PRIOR_N, _DIRT_PRIOR_MEAN,
        )
    elif "dam_sire_dirt_win_rate" not in df.columns:
        df["dam_sire_dirt_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 4. dam_sire_debut_win_rate: p_dam_sire 別 grade_code==1 勝率
    # ------------------------------------------------------------------
    if "p_dam_sire" in df.columns and "dam_sire_debut_win_rate" not in df.columns:
        _DEBUT_PRIOR_N = 20.0
        _DEBUT_PRIOR_MEAN = 0.066
        df["dam_sire_debut_win_rate"] = _bayesian_win_rate(
            df, "p_dam_sire", win_flag, debut_mask,
            _DEBUT_PRIOR_N, _DEBUT_PRIOR_MEAN,
        )
    elif "dam_sire_debut_win_rate" not in df.columns:
        df["dam_sire_debut_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 5. sire_debut_distance_match
    #    p_sire 産駒の「当該レースの距離帯（±200m 近似ビン）」での平均着順スコア（低いほど良い）
    #    集計対象: grade_code による絞り込みなし。全レース・全馬場種別を対象とする。
    #    （注意: 変数名に "debut" を含むが、これは命名時の経緯によるもので新馬限定ではない）
    #    着順スコア = 1/finish_rank（1着=1.0, 2着=0.5, ...）とし、距離帯ビン内の平均をとる
    #    リーク防止: 当該レースを除外（同一行を含まないように cumsum - current で計算）
    # ------------------------------------------------------------------
    if (
        "p_sire" in df.columns
        and "distance" in df.columns
        and "finish_rank" in df.columns
        and "sire_debut_distance_match" not in df.columns
    ):
        dist_num = pd.to_numeric(df["distance"], errors="coerce")
        # 着順スコア（1/finish_rank、finish_rankが0やNaNのときはNaN）
        rank_score = (1.0 / finish.replace(0, np.nan)).astype("float32")

        # 距離帯カテゴリ（±200m を 400m 幅ビンで近似）
        # 距離ビン: [0, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 10000]
        dist_bin = pd.cut(
            dist_num,
            bins=[0, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 10000],
            labels=[0, 1, 2, 3, 4, 5, 6, 7],
            right=True,
        ).astype("Int8")

        # 距離ビン境界をまたぐ ±200m を正確に扱うには2ビン分を結合する必要があるが、
        # ここでは同一ビン内の実績のみを使用する（ビン幅≒200m のため近似は許容範囲内）
        grp_key = [df["p_sire"], dist_bin]

        # 距離一致出走数（当該行除外）
        cum_dist_runs = df.groupby(grp_key, sort=False).cumcount()

        # 着順スコアの累積和（当該行除外）
        score_filled = rank_score.fillna(0.0)
        score_valid = rank_score.notna().astype("int8")

        cum_score_sum = (
            score_filled.groupby(grp_key, sort=False).cumsum()
            - score_filled
        )
        cum_score_cnt = (
            score_valid.groupby(grp_key, sort=False).cumsum()
            - score_valid
        )

        avg_rank_score = cum_score_sum / cum_score_cnt.replace(0, np.nan)
        df["sire_debut_distance_match"] = avg_rank_score.where(
            cum_dist_runs > 0, np.nan
        ).astype("float32")
    elif "sire_debut_distance_match" not in df.columns:
        df["sire_debut_distance_match"] = np.nan

    # ------------------------------------------------------------------
    # 6. sire_turf_win_rate: p_sire 別 芝全体（track_code<50）勝率
    #    芝レース（ダートとは別のサーフェス適性）での父馬産駒実績。
    #    ダート行には NaN を返し、モデルが表面違いで参照しないよう設計する。
    #    min_periods=10: 少数例の偏りを防ぐために10走未満はNaN。
    # ------------------------------------------------------------------
    if "p_sire" in df.columns and "sire_turf_win_rate" not in df.columns:
        _TURF_PRIOR_N = 20.0
        _TURF_PRIOR_MEAN = 0.0714  # 芝全体平均勝率
        _TURF_MIN_PERIODS = 10
        turf_mask = (track_num < 50).fillna(False)
        df["sire_turf_win_rate"] = _bayesian_win_rate(
            df, "p_sire", win_flag, turf_mask,
            _TURF_PRIOR_N, _TURF_PRIOR_MEAN,
        )
        # min_periods 未満は NaN（_bayesian_win_rate はサブセット行でのcum_runを使う）
        # min_periods を事後的に適用: cum_turf_runs を再計算してフィルタ
        turf_flag = turf_mask.astype("int8")
        sire_key_for_filter = df["p_sire"].astype(str).fillna("__nan__")
        cum_turf_runs = turf_flag.groupby(sire_key_for_filter, sort=False).cumsum() - turf_flag
        df["sire_turf_win_rate"] = df["sire_turf_win_rate"].where(
            cum_turf_runs >= _TURF_MIN_PERIODS, np.nan
        ).astype("float32")
    elif "sire_turf_win_rate" not in df.columns:
        df["sire_turf_win_rate"] = np.nan

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


def add_nick_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    父馬×母父馬（ニックス）別産駒勝率を df に追加して返す。

    (p_sire, p_dam_sire) の複合キーでグループ化し、ベイズ平滑化勝率を計算する。
    2キー交差のためデータが希薄になりやすく、事前分布を強め（prior_n=30）に設定する。
    全行に値を付与し（NaN条件: min_periods未満のみ）、サーフェスによる絞り込みは行わない。

    ベイズ平滑化パラメータ:
        prior_n    = 30.0  （2キー交差で希薄なため強い事前分布）
        prior_mean = 0.072 （芝ダート混合の全体平均勝率を事前として使用）
        min_periods= 10    （10走未満の場合 NaN）

    リーク防止: date ソート後に cumsum - current で当該行を除外する。

    Args:
        df: p_sire, p_dam_sire, finish_rank, date, race_id, ketto_num 列を持つ DataFrame

    Returns:
        nick_win_rate 列を追加した DataFrame（行数・順序は変更しない）
    """
    if "nick_win_rate" in df.columns:
        return df
    if "p_sire" not in df.columns or "p_dam_sire" not in df.columns:
        df["nick_win_rate"] = np.nan
        return df

    # --- インデックス保存 → 時系列ソート ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 勝利フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    # NaN を "__nan__" に統一してグループキーを作成
    # 理由: GroupBy は NaN をキーとして扱えないため、文字列に変換して集約する
    sire_key = df["p_sire"].astype(str).fillna("__nan__")
    dam_sire_key = df["p_dam_sire"].astype(str).fillna("__nan__")
    grp_key = sire_key + "_x_" + dam_sire_key

    _PRIOR_N = 30.0
    _PRIOR_MEAN = 0.072
    _MIN_PERIODS = 10

    # 出走数累積（当該行除外）: cumsum(1) - 1
    ones = pd.Series(np.ones(len(df), dtype="int8"), index=df.index)
    cum_runs = ones.groupby(grp_key, sort=False).cumsum() - ones

    # 勝利数累積（当該行除外）: cumsum(win_flag) - win_flag
    cum_wins = win_flag.groupby(grp_key, sort=False).cumsum() - win_flag

    smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)

    df["nick_win_rate"] = smoothed.where(
        cum_runs >= _MIN_PERIODS, np.nan
    ).astype("float32")

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


def add_sire_long_turf_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    父馬別 芝長距離（track_code<50 かつ distance>2000）勝率を df に追加して返す。

    集計対象を long_turf_flag==1 の行に限定することで、ダートや短距離レースの
    産駒成績が父馬の「芝長距離適性」評価に混入するのを防ぐ。

    芝長距離以外の行（ダート、短距離、中距離）には NaN を返す。
    これにより、モデルがミスマッチの条件でこの特徴量を参照するのを防ぐ。

    ベイズ平滑化パラメータ:
        prior_n    = 20.0  （長距離産駒成績は絶対数が少ないため中程度の事前）
        prior_mean = 0.072 （芝全体平均勝率を事前として使用）
        min_periods= 10    （10走未満の場合 NaN）

    Args:
        df: p_sire, track_code, distance, finish_rank,
            date, race_id, ketto_num 列を持つ DataFrame

    Returns:
        sire_long_turf_win_rate 列を追加した DataFrame（行数・順序は変更しない）
    """
    if "sire_long_turf_win_rate" in df.columns:
        return df

    # --- インデックス保存 → ソート ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 芝長距離フラグ: track_code<50（芝）かつ distance>2000（長距離）---
    track_num = pd.to_numeric(df["track_code"], errors="coerce")
    dist_num = pd.to_numeric(df["distance"], errors="coerce")
    long_turf_flag = ((track_num < 50) & (dist_num > 2000)).astype("int8")

    # --- 勝利フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")

    _PRIOR_N = 20.0
    _PRIOR_MEAN = 0.072  # 芝全体平均勝率を事前として使用
    _MIN_PERIODS = 10

    if "p_sire" in df.columns:
        # p_sire を文字列正規化（NaN は "__nan__" として 1グループに集約）
        sire_key = df["p_sire"].astype(str).fillna("__nan__")

        # long_turf_flag==1 の行のみを出走・勝利としてカウント
        # cumsum-current で当該行を除外
        cum_runs = long_turf_flag.groupby(sire_key, sort=False).cumsum() - long_turf_flag
        cum_wins = (
            (win_flag * long_turf_flag)
            .groupby(sire_key, sort=False)
            .cumsum()
            - (win_flag * long_turf_flag)
        )

        smoothed = (cum_wins + _PRIOR_N * _PRIOR_MEAN) / (cum_runs + _PRIOR_N)

        # min_periods 未満の場合 NaN、かつ long_turf_flag==0 の行は NaN
        rate = smoothed.where(cum_runs >= _MIN_PERIODS, np.nan)
        rate = rate.where(long_turf_flag.astype(bool), np.nan)
        df["sire_long_turf_win_rate"] = rate.astype("float32")
    else:
        df["sire_long_turf_win_rate"] = np.nan

    # --- 元の順序・インデックスに復元 ---
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v16_path = project_root / "model_training/data/02_features/features_past_v16.parquet"

    print("Loading data...")
    df = pd.read_parquet(v16_path)
    print(f"Input: {df.shape}")
    df = add_sire_stats_features(df)
    df = add_sire_long_turf_feature(df)
    print(f"Output: {df.shape}")

    new_cols = [
        "sire_debut_win_rate", "sire_dirt_win_rate",
        "dam_sire_dirt_win_rate", "dam_sire_debut_win_rate",
        "sire_debut_distance_match",
        "sire_long_turf_win_rate",
    ]
    for col in new_cols:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            valid = df[col].dropna()
            print(
                f"  {col}: NaN={nan_pct:.1f}%, mean={valid.mean():.4f}, "
                f"std={valid.std():.4f}, min={valid.min():.4f}, max={valid.max():.4f}"
            )
