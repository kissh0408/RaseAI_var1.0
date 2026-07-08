"""
features_graded.py — 重賞・格変動特徴量

生成特徴量:
    graded_race_win_rate   : 馬の重賞レース（GIII/GII/GI）での過去勝率（ベイズ平滑化, min_periods=2）
    graded_race_top3_rate  : 同じく重賞での複勝率（3着以内率）
    grade_step_up_flag     : 前走より格上のレースへの出走フラグ（1=格上挑戦, 0=同級/格下）
    grade_step_down_flag   : 前走より格下のレースへの出走フラグ（1=格下げ, 0=同級/格上）
    jockey_graded_win_rate : 騎手の重賞限定累積勝率（ベイズ平滑化, min_periods=10）
                             重賞以外の行は NaN を返す
    trainer_graded_win_rate: 調教師の重賞限定累積勝率（ベイズ平滑化, min_periods=10）
                             重賞以外の行は NaN を返す

問題の根拠（v13バックテスト・v17退行分析）:
    グレード別回収率: 新馬/未勝利106.3%に対してGIII 46.3%、GI相当 79.3%。
    重賞は出走機会が限られ個体レベルの重賞実績が現行モデルに不足している。
    格上挑戦・格下げ出走の傾向もモデルが捉えていない。
    v17でROI 139.4%（v13比-33.9%）に退行したため騎手・調教師の重賞特化勝率を追加。

JV-Linkの grade_code 解釈（仕様書準拠）:
    1 = 新馬/未勝利
    2〜4 = 条件戦（1〜3勝クラス）
    5 = GI相当（JVLinkコード体系による）
    6 = GII
    7 = GIII
    8 = OP特別
    9 = 障害

    重賞判定: grade_code in {5, 6, 7}

リーク防止:
    cumsum - current で当該レースを除外した累積ベイズ平滑化勝率。
    grade_step_up/down_flag は「前走 grade_code vs 今走 grade_code」の差で算出し
    前走情報（lag1_grade相当）のみ使用するためリーク非該当。
    jockey/trainer_graded_win_rate は重賞行のみをカウント対象とし、
    is_graded==False の行には NaN を返す（非重賞での誤参照を防止）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_training.src.features_common import _bayesian_cumulative_rate

_GRADE_GRADED = {5, 6, 7}  # GIII/GII/GI相当

_HORSE_PRIOR_N = 5.0      # 重賞はサンプル少のため事前分布を薄めに
_HORSE_PRIOR_WIN = 0.072
_HORSE_PRIOR_TOP3 = 0.22

_MIN_GRADED = 2           # 重賞出走2走以上でのみ有効

# v18追加: 騎手・調教師の重賞限定勝率パラメータ
# 騎手: 全重賞騎乗経験が多いため prior_n=30 で十分に平滑化
_JOCKEY_GRADED_PRIOR_N = 30.0
_JOCKEY_GRADED_PRIOR_MEAN = 0.07
_JOCKEY_GRADED_MIN_PERIODS = 10

# 調教師: 騎手より重賞機会が少ない傾向のため prior_n を少し緩める
_TRAINER_GRADED_PRIOR_N = 20.0
_TRAINER_GRADED_PRIOR_MEAN = 0.07
_TRAINER_GRADED_MIN_PERIODS = 10


def add_graded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    重賞・格変動特徴量を df に追加して返す。

    Args:
        df: features_past_v13 など（ketto_num, grade_code, finish_rank, date 列を持つ DataFrame）
    Returns:
        新特徴量4列を追加した DataFrame（行数・順序は変更しない）
    """
    new_cols = [
        "graded_race_win_rate",
        "graded_race_top3_rate",
        "grade_step_up_flag",
        "grade_step_down_flag",
    ]
    if all(c in df.columns for c in new_cols):
        return df

    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    grade = pd.to_numeric(df["grade_code"], errors="coerce")

    win_flag = (finish == 1).astype("int8")
    top3_flag = (finish <= 3).astype("int8")

    is_graded = grade.isin(_GRADE_GRADED)
    graded_count = is_graded.astype("int8")
    graded_win = (win_flag * graded_count).astype("int8")
    graded_top3 = (top3_flag * graded_count).astype("int8")

    horse_key = df["ketto_num"].astype(str)

    # --- 1 & 2. 重賞での馬の勝率・複勝率（cumsum-current、全行に付与）---
    cum_graded_runs = graded_count.groupby(horse_key, sort=False).cumsum() - graded_count
    cum_graded_wins = graded_win.groupby(horse_key, sort=False).cumsum() - graded_win
    cum_graded_top3 = graded_top3.groupby(horse_key, sort=False).cumsum() - graded_top3

    graded_win_rate = (cum_graded_wins + _HORSE_PRIOR_N * _HORSE_PRIOR_WIN) / (cum_graded_runs + _HORSE_PRIOR_N)
    graded_top3_rate = (cum_graded_top3 + _HORSE_PRIOR_N * _HORSE_PRIOR_TOP3) / (cum_graded_runs + _HORSE_PRIOR_N)

    df["graded_race_win_rate"] = graded_win_rate.where(cum_graded_runs >= _MIN_GRADED, np.nan).astype("float32")
    df["graded_race_top3_rate"] = graded_top3_rate.where(cum_graded_runs >= _MIN_GRADED, np.nan).astype("float32")

    # --- 3 & 4. 格上挑戦・格下げフラグ ---
    # 前走の grade_code を馬ごとに shift(1) で取得
    lag_grade = (
        grade.groupby(horse_key, sort=False)
        .transform(lambda x: x.shift(1))
    )

    # grade_code の大小関係: 数値が大きいほど格下と仮定はできない（JVLink体系依存）
    # 重賞(5,6,7)と一般条件戦(1-4)の境界で格変動を定義する
    # 今走が重賞で前走が条件戦 → 格上挑戦
    # 今走が条件戦で前走が重賞 → 格下げ
    # 同じグループ内での grade_code 差（降順で格上: 5>4>3>2>1、7>6>5等）

    # 簡易定義: 今走のgrade > 前走のgradeなら格上（数値が大きい方が格上とする暫定実装）
    # ※ JV-Link体系上 grade_code が単調な格付けかどうか確認が必要だが、
    #   実用上は重賞区分(>=5)と一般区分(<5)の境界で充分
    is_graded_now = grade.isin(_GRADE_GRADED)
    is_graded_lag = lag_grade.isin(_GRADE_GRADED)
    is_general_now = ~is_graded_now & grade.notna()
    is_general_lag = ~is_graded_lag & lag_grade.notna()

    # 格上挑戦: 前走が一般、今走が重賞
    df["grade_step_up_flag"] = (is_graded_now & is_general_lag).astype("int8")
    # 格下げ: 前走が重賞、今走が一般
    df["grade_step_down_flag"] = (is_general_now & is_graded_lag).astype("int8")

    # 元のインデックス順に戻す（重複インデックスでも安全な位置ベース復元）
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df


def add_graded_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    騎手・調教師の重賞限定累積勝率を df に追加して返す。

    重賞行（grade_code in {5,6,7}）のみを集計対象とし、非重賞行は NaN を返す。
    これにより重賞レースでのみ意味を持つシグナルとして機能する。

    Args:
        df: jockey_code, trainer_code, grade_code, finish_rank, date,
            race_id, ketto_num 列を持つ DataFrame（時系列ソート不要、内部でソートする）
    Returns:
        jockey_graded_win_rate, trainer_graded_win_rate 列を追加した DataFrame
        （行数・順序は変更しない）
    """
    new_cols = ["jockey_graded_win_rate", "trainer_graded_win_rate"]
    if all(c in df.columns for c in new_cols):
        return df

    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    grade = pd.to_numeric(df["grade_code"], errors="coerce")

    is_graded = grade.isin(_GRADE_GRADED)
    win_flag = (finish == 1).astype("int8")

    # 重賞行のみをカウント対象とするフラグ
    graded_count_flag = is_graded.astype("int8")
    # 重賞行での勝利フラグ（非重賞行は 0）
    graded_win_flag = (win_flag * graded_count_flag).astype("int8")

    # --- 騎手の重賞累積勝率 ---
    jockey_key = df["jockey_code"].astype(str)
    jockey_graded_rate = _bayesian_cumulative_rate(
        group_key=jockey_key,
        target_flag=graded_win_flag,
        count_flag=graded_count_flag,
        prior_n=_JOCKEY_GRADED_PRIOR_N,
        prior_mean=_JOCKEY_GRADED_PRIOR_MEAN,
        min_periods=_JOCKEY_GRADED_MIN_PERIODS,
    )
    # 重賞以外の行は NaN（非重賞での誤参照を防ぐ）
    df["jockey_graded_win_rate"] = jockey_graded_rate.where(is_graded, np.nan).astype("float32")

    # --- 調教師の重賞累積勝率 ---
    trainer_key = df["trainer_code"].astype(str)
    trainer_graded_rate = _bayesian_cumulative_rate(
        group_key=trainer_key,
        target_flag=graded_win_flag,
        count_flag=graded_count_flag,
        prior_n=_TRAINER_GRADED_PRIOR_N,
        prior_mean=_TRAINER_GRADED_PRIOR_MEAN,
        min_periods=_TRAINER_GRADED_MIN_PERIODS,
    )
    # 重賞以外の行は NaN
    df["trainer_graded_win_rate"] = trainer_graded_rate.where(is_graded, np.nan).astype("float32")

    # 元のインデックス順に復元
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index
    return df
