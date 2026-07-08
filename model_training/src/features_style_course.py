"""
features_style_course.py — 脚質×コース相性特徴量

生成特徴量:
    horse_style_course_win_rate   : 馬×脚質×コース別の過去勝率（ベイズ平滑化, min_periods=3）
    horse_style_course_place_rate : 馬×脚質×コース別の複勝率（3着以内率）
    jockey_style_course_win_rate  : 騎手×脚質×コース別の過去勝率（ベイズ平滑化, min_periods=10）
    jockey_style_course_place_rate: 騎手×脚質×コース別の複勝率（min_periods=10）
    sire_style_course_win_rate    : 父馬×脚質×コース別の産駒勝率（ベイズ平滑化, min_periods=5）
    style_course_bias_score       : コース×脚質の全体バイアス（訓練期間定数, リーク防止）

NaN補完優先順位:
    horse_style_course_win_rate → jockey_style_course_win_rate
        → sire_style_course_win_rate → 0.0

リーク防止:
    horse/jockey: shift(1) + expanding() に相当する cumsum - current で当該行除外
    sire        : 種牡馬統計は産駒が複数馬にまたがるため date < 当該レース日でフィルタ
    bias_score  : 訓練期間（2015-2024年末）のデータのみで計算した定数。テスト期間に適用。

Train-serving skew 修正 (c5):
    running_style_code は SE レコードのレース後確定フィールドのため推論時は全行 0。
    horse_modal_running_style（過去 N 走の最頻脚質コード）を使用することで
    学習時・推論時の両方で有効な脚質キーを生成する。
    style_course_bias_score は訓練期間の実測 running_style_code で計算するため変更なし。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_training.src.features_common import _bayesian_cumulative_rate

# ------------------------------------------------------------------
# 訓練期間の定義（バイアスなど定数計算に使用）
# train_config.json の training.train_end_date を参照し、なければ定数にフォールバック
# ------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "train_config.json"
try:
    _TRAIN_END: str = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))["training"]["train_end_date"]
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    _TRAIN_END = "2024-12-31"

# ベイズ平滑化パラメータ
_HORSE_PRIOR_N = 10.0
_HORSE_PRIOR_MEAN = 0.072   # 芝平均勝率

_JOCKEY_PRIOR_N = 20.0
_JOCKEY_PRIOR_MEAN = 0.10   # 騎手の平均勝率

_SIRE_PRIOR_N = 15.0
_SIRE_PRIOR_MEAN = 0.072    # 産駒芝平均勝率


# ------------------------------------------------------------------
# 父馬向けベイズ勝率（date < 当該レース日のフィルタ）
# ------------------------------------------------------------------


def _sire_bayesian_rate_vectorized(
    df: pd.DataFrame,
    sire_col: str,
    target_flag: pd.Series,
    count_flag: pd.Series,
    prior_n: float,
    prior_mean: float,
    min_periods: int,
) -> pd.Series:
    """
    vectorized 版: df が date でソート済みであることを前提とし、
    cumsum - current で当該行以前の累積を計算（同日同着は許容近似）。
    sire ごとに産駒全体の累積を扱うため、sort 後に cumsum を使う。
    """
    # date ソート済み前提（呼び出し側で保証）
    sire_key = df[sire_col].astype(str).fillna("__nan__")

    cum_runs = count_flag.groupby(sire_key, sort=False).cumsum() - count_flag
    cum_wins = target_flag.groupby(sire_key, sort=False).cumsum() - target_flag

    smoothed = (cum_wins + prior_n * prior_mean) / (cum_runs + prior_n)
    return smoothed.where(cum_runs >= min_periods, np.nan).astype("float32")


# ------------------------------------------------------------------
# メイン関数
# ------------------------------------------------------------------

def add_style_course_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質×コース相性特徴量を df に追加して返す。

    Args:
        df: features_past_v12など（ketto_num, jockey_code, p_sire,
            horse_modal_running_style（または running_style_code）,
            course_code, finish_rank, date 列を持つ DataFrame）
            horse_modal_running_style が存在しない場合は running_style_code にフォールバック。
    Returns:
        新特徴量6列を追加した DataFrame（行数・順序は変更しない）
    """
    # --- 時系列ソート保証 ---
    sort_cols = [c for c in ("date", "race_id", "ketto_num") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 基本フラグ ---
    finish = pd.to_numeric(df["finish_rank"], errors="coerce")
    win_flag = (finish == 1).astype("int8")
    place_flag = (finish <= 3).astype("int8")

    # horse_modal_running_style を優先使用（推論時も 0 以外の値が入る）
    # フォールバック: horse_modal_running_style 列が存在しない場合は running_style_code を使用
    if "horse_modal_running_style" in df.columns:
        style_num = pd.to_numeric(df["horse_modal_running_style"], errors="coerce")
    else:
        style_num = pd.to_numeric(df["running_style_code"], errors="coerce")
    valid_style = (style_num >= 1) & (style_num <= 4)

    # --- グループキー（脚質×コース） ---
    # 脚質コードと course_code を文字列連結でキーを作る
    style_str = style_num.astype("Int8").astype(str)
    course_num = pd.to_numeric(df["course_code"], errors="coerce").astype("Int8")
    course_str = course_num.astype(str)

    style_course_key = style_str + "_" + course_str  # e.g. "3_5"
    # 脚質不明行は専用キーにしてグループ計算から実質除外
    style_course_key = style_course_key.where(valid_style, "__invalid__")

    # ------------------------------------------------------------------
    # 1 & 2. 馬×脚質×コース別 勝率・複勝率
    # ------------------------------------------------------------------
    horse_key = df["ketto_num"].astype(str) + "_" + style_course_key

    # 出走フラグ（valid_style かつ当該条件マッチ）
    horse_count = valid_style.astype("int8")
    horse_win = (win_flag * horse_count).astype("int8")
    horse_place = (place_flag * horse_count).astype("int8")

    df["horse_style_course_win_rate"] = _bayesian_cumulative_rate(
        horse_key, horse_win, horse_count,
        _HORSE_PRIOR_N, _HORSE_PRIOR_MEAN, min_periods=3,
    ).where(valid_style, np.nan)

    df["horse_style_course_place_rate"] = _bayesian_cumulative_rate(
        horse_key, horse_place, horse_count,
        _HORSE_PRIOR_N, _HORSE_PRIOR_MEAN * 3, min_periods=3,  # prior_mean を3倍（複勝は約21%）
    ).where(valid_style, np.nan)

    # ------------------------------------------------------------------
    # 3 & 4. 騎手×脚質×コース別 勝率・複勝率
    # ------------------------------------------------------------------
    jockey_key = df["jockey_code"].astype(str) + "_" + style_course_key

    jockey_count = valid_style.astype("int8")
    jockey_win = (win_flag * jockey_count).astype("int8")
    jockey_place = (place_flag * jockey_count).astype("int8")

    df["jockey_style_course_win_rate"] = _bayesian_cumulative_rate(
        jockey_key, jockey_win, jockey_count,
        _JOCKEY_PRIOR_N, _JOCKEY_PRIOR_MEAN, min_periods=10,
    ).where(valid_style, np.nan)

    df["jockey_style_course_place_rate"] = _bayesian_cumulative_rate(
        jockey_key, jockey_place, jockey_count,
        _JOCKEY_PRIOR_N, _JOCKEY_PRIOR_MEAN * 3, min_periods=10,
    ).where(valid_style, np.nan)

    # ------------------------------------------------------------------
    # 5. 父馬×脚質×コース別 産駒勝率（vectorized cumsum 版）
    #    sire特徴量は産駒が複数馬にまたがるため vectorized で近似（date ソート済み）
    # ------------------------------------------------------------------
    if "p_sire" in df.columns:
        sire_count = valid_style.astype("int8")
        sire_win = (win_flag * sire_count).astype("int8")

        sire_key_ser = df["p_sire"].astype(str).fillna("__nan__") + "_" + style_course_key
        cum_sire_runs = sire_count.groupby(sire_key_ser, sort=False).cumsum() - sire_count
        cum_sire_wins = sire_win.groupby(sire_key_ser, sort=False).cumsum() - sire_win

        sire_smoothed = (cum_sire_wins + _SIRE_PRIOR_N * _SIRE_PRIOR_MEAN) / (cum_sire_runs + _SIRE_PRIOR_N)
        df["sire_style_course_win_rate"] = sire_smoothed.where(
            (cum_sire_runs >= 5) & valid_style, np.nan
        ).astype("float32")
    else:
        df["sire_style_course_win_rate"] = np.nan

    # ------------------------------------------------------------------
    # 6. style_course_bias_score
    #    訓練期間（2015-2024年末）のみで計算したコース×脚質の全体バイアス定数
    #    bias = (脚質×コース勝率) / (コース全体勝率)
    # ------------------------------------------------------------------
    train_mask = df["date"] <= _TRAIN_END
    train_df = df[train_mask & valid_style].copy()

    train_finish = pd.to_numeric(train_df["finish_rank"], errors="coerce")
    train_win = (train_finish == 1).astype("int8")

    # コース全体勝率
    course_win_rate = (
        train_df.assign(win=train_win)
        .groupby("course_code")["win"]
        .mean()
    )

    # コース×脚質 勝率
    style_course_win_rate = (
        train_df.assign(win=train_win)
        .groupby(["course_code", "running_style_code"])["win"]
        .mean()
    )

    # バイアス = (脚質×コース勝率) / (コース全体勝率)
    bias_dict: dict[tuple, float] = {}
    for (course, style), sc_wr in style_course_win_rate.items():
        course_wr = course_win_rate.get(course, np.nan)
        if pd.notna(course_wr) and course_wr > 0:
            bias_dict[(int(course), int(style))] = float(sc_wr / course_wr)
        else:
            bias_dict[(int(course), int(style))] = 1.0

    # df に map
    course_int = course_num.astype("Int64")
    style_int = style_num.astype("Int64")
    bias_series = pd.Series(
        [
            bias_dict.get((int(c) if pd.notna(c) else -1, int(s) if pd.notna(s) else -1), np.nan)
            for c, s in zip(course_int, style_int)
        ],
        index=df.index,
        dtype="float32",
    )
    df["style_course_bias_score"] = bias_series.where(valid_style, np.nan)

    return df


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    v12_path = project_root / "model_training/data/02_features/features_past_v12.parquet"

    print("Loading data...")
    df = pd.read_parquet(v12_path)
    print(f"Input: {df.shape}")
    df = add_style_course_features(df)
    print(f"Output: {df.shape}")

    new_cols = [
        "horse_style_course_win_rate",
        "horse_style_course_place_rate",
        "jockey_style_course_win_rate",
        "jockey_style_course_place_rate",
        "sire_style_course_win_rate",
        "style_course_bias_score",
    ]
    for col in new_cols:
        nan_pct = df[col].isna().mean() * 100
        valid = df[col].dropna()
        print(f"  {col}: NaN={nan_pct:.1f}%, mean={valid.mean():.4f}, std={valid.std():.4f}")
