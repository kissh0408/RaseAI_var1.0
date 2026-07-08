# NOTE: standalone 実験では ROI 改善せず REJECT だが、create_pastfeatures v19 パイプラインと
# features_past_v25+ 生成に残留。ensemble 系再学習時に再検討すること。
# REJECTED EXPERIMENT (2026-06): 調教トレンド単独 ablation は効果なし。
"""
features_training_trend.py
--------------------------
調教マルチセッション傾向特徴量を生成するモジュール。

生成特徴量:
    training_time1f_trend_3sess  : 直近3セッションの time_1f 変化量の平均
                                   負値 = 加速傾向（最終ラップが短縮している）
    training_days_before_race    : 最終調教日からレース日までの日数

リーク防止方針:
    HC.train_date < SE.race_date（当日調教は含まない）
    merge_asof(direction="backward", tolerance=28日) で各レースの直前3セッションを取得。
    race_date - 1日 を left_on にすることで train_date < race_date を保証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _merge_asof_session(
    work_df: pd.DataFrame,
    hc: pd.DataFrame,
    ref_col: str,
    result_time_col: str,
    result_date_col: str,
) -> pd.DataFrame:
    """
    work_df の ref_col を参照日として、ketto_num でグループした最直近の
    HC セッション（time_1f, train_date）を取得する内部ヘルパー。

    NaT 行は merge_asof に渡さず NaN で復元することで行数を保持する。
    _row_id を使って結果を元のインデックスに正確に復元する。
    """
    key_cols = ["_row_id", "ketto_num", ref_col]
    valid_mask = work_df[ref_col].notna()

    valid_input = (
        work_df.loc[valid_mask, key_cols]
        .sort_values(ref_col)
        .reset_index(drop=True)
    )

    if len(valid_input) > 0:
        merged = pd.merge_asof(
            valid_input,
            hc.rename(columns={"time_1f": result_time_col, "train_date": result_date_col}),
            left_on=ref_col,
            right_on=result_date_col,
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=28),
        )
    else:
        merged = valid_input.copy()
        merged[result_time_col] = np.nan
        merged[result_date_col] = pd.NaT

    # 全行（NaT 含む）を _row_id で復元
    result = (
        work_df[["_row_id"]]
        .merge(
            merged[["_row_id", result_time_col, result_date_col]],
            on="_row_id",
            how="left",
        )
    )
    return result


def add_training_trend_features(df: pd.DataFrame, hc_path: str | Path) -> pd.DataFrame:
    """
    直近3セッションの調教傾向特徴量を追加する。

    各レース行に対して、レース日より前（HC.train_date < race_date）の
    直近3調教セッションを取得し、time_1f の傾向（平均差分）と
    最終調教日からの経過日数を計算して追加する。

    生成する特徴量:
        training_time1f_trend_3sess : time_1f の直近3セッション差分の平均
                                      (s0 - s2) / 2 = 3本目から最新への単位変化量
                                      負値 = 最終ラップが短縮 = 加速傾向。
                                      s2 欠損で s0・s1 両方ある場合は (s0 - s1)。
                                      s0 のみ（s1 欠損）の場合は NaN。
        training_days_before_race   : 最終調教日からレース日までの日数（整数）

    Parameters
    ----------
    df : pd.DataFrame
        特徴量データフレーム。'ketto_num'（object または int）と
        'date'（datetime 列）が必要。一意な行識別子として 'race_id' があれば
        それも使用する。
    hc_path : str | Path
        HC_preprocessed.parquet（または .csv）のファイルパス。
        必要カラム: train_date（datetime）, ketto_num（int/str）, time_1f（float）

    Returns
    -------
    pd.DataFrame
        入力 df に 2 列を追加した新しい DataFrame（行数は入力と同一）。
        入力 df 自体は変更しない（コピーして返す）。

    Notes
    -----
    - HC に存在しない馬・データ不足（セッション < 2）の行は NaN になる。
    - NaN 率目標: 20% 以内（大半の馬は十分なセッション数を持つ）。
    - 行数保持: 内部で _row_id を付与し、全結合で行数を保証する。
    """
    hc_path = Path(hc_path)
    if not hc_path.exists():
        raise FileNotFoundError(f"HC data not found: {hc_path}")

    # ------------------------------------------------------------------
    # Step 1: HC データを読み込む
    # ------------------------------------------------------------------
    if hc_path.suffix.lower() == ".parquet":
        hc = pd.read_parquet(hc_path, columns=["train_date", "ketto_num", "time_1f"])
    else:
        hc = pd.read_csv(hc_path, usecols=["train_date", "ketto_num", "time_1f"])
        hc["train_date"] = pd.to_datetime(hc["train_date"], errors="coerce")

    # time_1f の外れ値除去（1〜30秒: 100m 最終ラップタイムとして現実的な範囲）
    hc = hc[hc["time_1f"].between(1.0, 30.0, inclusive="both")].copy()
    hc["ketto_num"] = hc["ketto_num"].astype(str)
    hc["train_date"] = pd.to_datetime(hc["train_date"], errors="coerce")
    hc = hc.dropna(subset=["train_date", "ketto_num"])
    # merge_asof の前提: train_date 昇順ソート
    hc = hc.sort_values("train_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 2: df の準備
    # ------------------------------------------------------------------
    df = df.copy()
    horse_col = "ketto_num" if "ketto_num" in df.columns else "horse_id"
    date_col = "date" if "date" in df.columns else "race_date"
    if horse_col not in df.columns or date_col not in df.columns:
        raise KeyError(
            f"add_training_trend_features: ketto_num/horse_id と date/race_date が必要 "
            f"(columns={list(df.columns)[:20]}...)"
        )
    df["ketto_num"] = df[horse_col].astype(str)
    if date_col != "date":
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 行数保持のための一意識別子（重複 (ketto_num, date) ペアに対応）
    df["_row_id"] = np.arange(len(df))

    # ------------------------------------------------------------------
    # Step 3: 作業用 DataFrame を date 昇順にソートして merge_asof に渡す
    #   リーク防止: race_date - 1日 を参照日にすることで
    #   train_date < race_date を厳守する
    # ------------------------------------------------------------------
    work = df[["_row_id", "ketto_num", "date"]].sort_values("date").copy()
    work["_ref_date_s0"] = work["date"] - pd.Timedelta(days=1)

    # Session 0: 最も直近の調教セッション
    s0_result = _merge_asof_session(
        work, hc, "_ref_date_s0", "_time1f_s0", "_train_date_s0"
    )
    work = work.merge(s0_result, on="_row_id", how="left")

    # Session 1: s0 の 1日前を参照日にした次のセッション
    work["_ref_date_s1"] = work["_train_date_s0"] - pd.Timedelta(days=1)
    s1_result = _merge_asof_session(
        work, hc, "_ref_date_s1", "_time1f_s1", "_train_date_s1"
    )
    work = work.merge(s1_result, on="_row_id", how="left")

    # Session 2: s1 の 1日前を参照日にした次のセッション
    work["_ref_date_s2"] = work["_train_date_s1"] - pd.Timedelta(days=1)
    s2_result = _merge_asof_session(
        work, hc, "_ref_date_s2", "_time1f_s2", "_train_date_s2"
    )
    work = work.merge(s2_result, on="_row_id", how="left")

    # ------------------------------------------------------------------
    # Step 4: 特徴量計算
    #
    # training_time1f_trend_3sess:
    #   3本揃い: (s0 - s2) / 2  → 単位セッション当たりの平均 time_1f 変化量
    #   2本のみ: (s0 - s1)       → 1差分
    #   s0 のみ or s0 欠損: NaN
    #
    # training_days_before_race:
    #   race_date - _train_date_s0（整数日数）
    # ------------------------------------------------------------------
    s0 = work["_time1f_s0"].values.astype(float)
    s1 = work["_time1f_s1"].values.astype(float)
    s2 = work["_time1f_s2"].values.astype(float)

    with np.errstate(invalid="ignore"):
        trend_3sess = np.where(
            ~np.isnan(s0) & ~np.isnan(s2),
            (s0 - s2) / 2.0,       # 3本以上: 平均差分
            np.where(
                ~np.isnan(s0) & ~np.isnan(s1),
                s0 - s1,           # 2本のみ: 1差分
                np.nan,            # s0 欠損 or s0 のみ
            ),
        )

    work["_trend_3sess"] = trend_3sess
    work["_days_before"] = (work["date"] - work["_train_date_s0"]).dt.days

    # ------------------------------------------------------------------
    # Step 5: 元の df に _row_id で結合して行数を保持
    # ------------------------------------------------------------------
    result_df = df.merge(
        work[["_row_id", "_trend_3sess", "_days_before"]],
        on="_row_id",
        how="left",
    ).rename(
        columns={
            "_trend_3sess": "training_time1f_trend_3sess",
            "_days_before": "training_days_before_race",
        }
    ).drop(columns=["_row_id"])

    result_df = result_df.reset_index(drop=True)
    return result_df
