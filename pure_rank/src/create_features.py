"""
create_features.py — RaceAI_var1.0 特徴量生成スクリプト

01_preprocessed/ の Parquet から特徴量を生成し
02_features/features_{version}.parquet を出力する。
バージョンは pure_rank/config/train_config.json の features_version で管理する。

禁止事項:
- オッズ・人気 (odds, popularity) を特徴量に含めない
- market_log_odds / init_score を使わない
- shift(1) なしの全データ集計を hist_ 系特徴量に使わない
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# パス解決・設定読み込み・禁止列定義は common.py に一元化
from common import FORBIDDEN_MARKET_COLS, PROJECT_ROOT, get_feature_cols, load_config, resolve_project_path


# ─── コース形態定数テーブル（v39_course 実験1 → v39_course_slim） ────────────────
# 出典仕様書: docs/specs/2026-07-03-d1-course-features-design.md セクション3
# JRA 公表の公知情報（直線長 [m]）。市場情報ではなくチューニング対象でもないため
# train_config.json ではなくモジュール定数として定義する。
# 値: (芝直線[内], 芝直線[外] or None, ダート直線)
#
# v39_course_slim（evaluator 差し戻し再実験）:
#   v39_course の3列中 course_straight_len / course_is_small が15モデルで split=0 の
#   dead weight だったため、出力列は front_pref_x_small の1列のみに削減した。
#   course_is_small はその計算用の中間変数としてのみ生成し parquet には出力しない。
#   COURSE_GEOMETRY 定数と track_code 判別に関する定数は実験記録として残す
#   （evaluator 指示: コード内に残してよいが parquet には出さない）。
COURSE_GEOMETRY: dict[int, tuple[float, float | None, float]] = {
    1:  (266.1, None,  264.3),   # 札幌
    2:  (262.1, None,  260.3),   # 函館
    3:  (292.0, None,  295.7),   # 福島
    4:  (358.7, 658.7, 353.9),   # 新潟
    5:  (525.9, None,  501.6),   # 東京
    6:  (310.0, 310.0, 308.0),   # 中山
    7:  (412.5, None,  410.7),   # 中京
    8:  (328.4, 403.7, 329.1),   # 京都
    9:  (356.5, 473.6, 352.7),   # 阪神
    10: (293.0, None,  291.3),   # 小倉
}

# 小回りコース: 「芝直線長 < 300m」という幾何情報のみに基づく固定定義
# （札幌・函館・福島・小倉。テスト成績を見て場を選んだものではない）
SMALL_COURSE_CODES: frozenset[int] = frozenset({1, 2, 3, 10})

# 芝外回りを示す track_code（12=左外, 18=右外）。新潟・中山・京都・阪神で出現
OUTER_TRACK_CODES: frozenset[int] = frozenset({12, 18})

# 芝・直線コース（新潟1000m）。直線長 = レース距離そのもの
STRAIGHT_TRACK_CODE: int = 10

# v40_waku（実験2）: 枠×コース超過勝率セルの最小累積観測数。
# これ未満のセルは S/N 比が低くノイズになるため NaN とし、
# LightGBM の欠損値分岐に処理を委ねる（MIN_JOCKEY_RACES と同じ流儀）。
# 仕様書: docs/specs/2026-07-03-d1-course-features-design.md セクション4 実験2
MIN_WAKU_SAMPLES: int = 50

# ─── 輸送距離カテゴリ定数（v45_transport, 候補4） ────────────────────────────────
# 仕様書: docs/specs/2026-07-05-summer-racing-structural-features-design.md
#   セクション2 候補4・セクション6
# region_code（東西所属コード。JV-Data.md #2301。SEレコード、オフセット85、1桁）:
#   0=下記以外/未整備（主に地方競馬・海外国際レース）, 1=関東(美浦), 2=関西(栗東),
#   3=地方招待, 4=外国招待
# course_code（開催競馬場。COURSE_GEOMETRY と同一採番。1=札幌...10=小倉）
#
# 3カテゴリ（0=近郊/輸送負担小, 1=中距離, 2=長距離/輸送負担大）は
# トレセン所在地（美浦=茨城県・関東、栗東=滋賀県・関西）と競馬場所在地の
# 実際の地理的関係のみに基づき固定する。学習期間データを見て境界を調整しない
# （後出し調整の禁止。COURSE_GEOMETRY と同じ「地理的事実のみ」の流儀）。
#
# 関東(美浦)所属馬:
#   近郊(0)  = 東京・中山（同一都市圏、実質輸送負担なし）
#   中距離(1) = 福島・新潟・中京・京都・阪神（本州内、数百km規模）
#   長距離(2) = 札幌・函館（北海道、海峡越え）・小倉（九州、本州外れの最遠隔）
# 関西(栗東)所属馬:
#   近郊(0)  = 中京・京都・阪神（同一地域ブロック）
#   中距離(1) = 福島・新潟・東京・中山・小倉（本州/九州本土内、数百km規模）
#   長距離(2) = 札幌・函館（北海道、海峡越え。関西からは関東所属より更に遠い）
# 地方招待(3)・外国招待(4)・未整備(0扱いの馬):
#   標準的な美浦/栗東所属体系の外からの遠征であるため、開催地によらず一律
#   長距離(2) とする（サンプルは全体の約0.16%と極小。実測は生成ログで確認する）
TRANSPORT_MAP: dict[tuple[int, int], int] = {}
for _c in (5, 6):
    TRANSPORT_MAP[(1, _c)] = 0
for _c in (3, 4, 7, 8, 9):
    TRANSPORT_MAP[(1, _c)] = 1
for _c in (1, 2, 10):
    TRANSPORT_MAP[(1, _c)] = 2
for _c in (7, 8, 9):
    TRANSPORT_MAP[(2, _c)] = 0
for _c in (3, 4, 5, 6, 10):
    TRANSPORT_MAP[(2, _c)] = 1
for _c in (1, 2):
    TRANSPORT_MAP[(2, _c)] = 2
for _region in (0, 3, 4):
    for _c in range(1, 11):
        TRANSPORT_MAP[(_region, _c)] = 2
del _c, _region


def _transport_category(region_code: pd.Series, course_code: pd.Series) -> pd.Series:
    """region_code × course_code から輸送距離カテゴリ（0=近郊/1=中距離/2=長距離）を導出する。

    TRANSPORT_MAP に存在しない組み合わせ（未知の region_code 等）は NaN とする。
    """
    keys = list(zip(region_code.astype("int64"), course_code.astype("int64")))
    values = [TRANSPORT_MAP.get(k, np.nan) for k in keys]
    return pd.Series(values, index=region_code.index, dtype="float64")


# ─── 市場情報混入チェック ───────────────────────────────────────────────────────

def _check_no_market_features(df: pd.DataFrame) -> None:
    """DataFrame に市場情報列が含まれていないことを確認する。"""
    found = FORBIDDEN_MARKET_COLS & set(df.columns)
    if found:
        raise ValueError(
            f"[FORBIDDEN] 市場情報が特徴量に混入しています: {sorted(found)}\n"
            f"即座に除去してください。"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: データ読み込み・結合
# ═══════════════════════════════════════════════════════════════════════════════

def _load_data(cfg: dict) -> pd.DataFrame:
    """SE / RA / SK の Parquet を読み込んで結合する。"""
    prep_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    se = pd.read_parquet(prep_dir / "SE_preprocessed.parquet")
    ra = pd.read_parquet(prep_dir / "RA_preprocessed.parquet")
    sk = pd.read_parquet(prep_dir / "SK_preprocessed.parquet")

    print(f"  SE: {len(se):,} rows, {len(se.columns)} cols")
    print(f"  RA: {len(ra):,} rows, {len(ra.columns)} cols")
    print(f"  SK: {len(sk):,} rows, {len(sk.columns)} cols")

    # SE + RA を race_id でマージ（RA の距離・馬場情報を SE に付加）
    # weight_type: v44_handicap（候補3: ハンデ戦フラグ）の生成元列。
    # common.py の FORBIDDEN_COLS でメタ列として除外済み（市場情報ではない。
    # docs/specs/2026-07-05-summer-racing-structural-features-design.md セクション2
    # 候補3参照）。_build_current_features で is_handicap_race 導出後に drop する。
    ra_merge_cols = [
        "race_id", "grade_code", "distance", "track_code", "horse_count",
        "weather_code", "surface_code", "track_condition_code",
        "surface_condition", "distance_category", "race_date", "weight_type",
        # v48_agari_turn: 回り×馬場適性の groupby キー（FORBIDDEN_COLS で特徴量除外）
        "race_type_code",
    ]
    # RA には race_date が既にある。SE の race_date と同一のはずだが RA のものを使う
    se = se.drop(columns=["race_date"], errors="ignore")
    ra_subset = ra[[c for c in ra_merge_cols if c in ra.columns]].copy()

    df = se.merge(ra_subset, on="race_id", how="inner")

    # mining_predicted_rank: v42_mining（Phase 6）専用の生列。
    # 他バージョンでは main() 内で version != "v42_mining" のとき早期に drop する
    # （既存バージョンの列数アサートに影響させないため）。
    # SK（血統）をマージ
    sk_cols = ["ketto_num", "sire_id", "bms_id"]
    sk_subset = sk[[c for c in sk_cols if c in sk.columns]].copy()
    df = df.merge(sk_subset, on="ketto_num", how="left")

    # region_code: v45_transport（候補4: 輸送距離カテゴリ）の生成元列。
    # pure_rank/data/01_preprocessed/SE_preprocessed.parquet には region_code が
    # 含まれていない（pure_rank/src/preprocess.py の _SE_SOURCE_COLS 未収録）ため、
    # var2.0.0 側の軽量な SE_preprocessed.parquet（同一の race_id×ketto_num キー、
    # 行数一致・重複なしを実装時に確認済み）から race_id×ketto_num で直接マージする。
    # pure_rank の 01_preprocessed 資産（他バージョンの再現性に影響する共有ファイル）
    # 自体は変更しない。region_code は common.py の FORBIDDEN_COLS でメタ列除外対象
    # のため、_build_current_features で transport_category 導出後に drop する
    # （weight_type → is_handicap_race と同じ「派生列に一本化して drop」の流儀）。
    # 仕様書: docs/specs/2026-07-05-summer-racing-structural-features-design.md
    #   セクション2 候補4・セクション6
    n_before_region = len(df)
    src_se_path = resolve_project_path(cfg["data"]["src_parquet_dir"]) / "SE_preprocessed.parquet"
    import pyarrow.parquet as _pq

    src_se_has_region = src_se_path.exists() and "region_code" in _pq.read_schema(src_se_path).names
    # region_code は元々 var2.0.0 の軽量 SE_preprocessed.parquet から補完する想定だったが
    # (コメント参照)、cfg["data"]["src_parquet_dir"] が pure_rank 自身の
    # 01_preprocessed を指す運用（L4 復旧確認, 2026-07-10）では region_code 列が
    # 存在しない。存在しない場合は NaN 埋めにフォールバックする
    # （_transport_category は未知組み合わせを NaN とする設計のため安全に劣化する）。
    if src_se_has_region:
        src_se = pd.read_parquet(src_se_path, columns=["race_id", "ketto_num", "region_code"])
        src_se["race_id"] = src_se["race_id"].astype(str)
        src_se["ketto_num"] = src_se["ketto_num"].astype(str)
        df["ketto_num"] = df["ketto_num"].astype(str)
        df = df.merge(src_se, on=["race_id", "ketto_num"], how="left")
        assert len(df) == n_before_region, (
            f"region_code マージで行数が変化しています（ファンアウト検出）: "
            f"{n_before_region:,} → {len(df):,}"
        )
    else:
        print(
            f"  [warn] region_code が {src_se_path} に存在しません。"
            f"transport_category は NaN にフォールバックします（L4復旧確認, 2026-07-10）。"
        )
        df["region_code"] = pd.NA

    print(f"  Merged: {len(df):,} rows, {len(df.columns)} cols")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: フィルタ適用
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """必須フィルタを適用する。

    除外対象:
    - grade_code 8 (未格付け), 9 (障害)
    - abnormal_code 1 (取消), 3 (除外), 4 (落馬)
    - horse_count < 5 (少頭数レース)
    - finish_rank == 0 (着順無効)
    """
    f = cfg["filters"]
    n_before = len(df)

    mask = (
        (~df["grade_code"].isin(f["exclude_grade_codes"]))
        & (~df["abnormal_code"].isin(f["exclude_abnormal_codes"]))
        & (df["horse_count"] >= f["min_horse_count"])
        & (df["finish_rank"] > 0)
    )
    df = df[mask].copy()

    print(f"  Filter applied: {n_before:,} → {len(df):,} rows "
          f"(removed {n_before - len(df):,})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5: COURSE GEOMETRY INTERMEDIATE (v39_course_slim)
# コースの物理形態（公知情報）に基づく中間変数。レース前確定情報でありリークと無関係。
# v39_course_slim では course_is_small を front_pref_x_small の計算用中間変数として
# のみ生成し、parquet には出力しない（_build_current_features で使用後に drop）。
# course_straight_len の出力は v39_course で split=0 の dead weight と判定され廃止
# （track_code による内/外回り判別ロジックは v39_course の実装記録として git 履歴に残る）。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_course_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """コース形態の中間変数を生成する。

    生成列（すべて中間変数。最終 parquet には含めない）:
    - course_is_small: 芝直線長 < 300m の小回り4場（札幌・函館・福島・小倉）フラグ。
      front_pref_x_small の計算にのみ使用し、_build_current_features 内で drop する。

    仕様書: docs/specs/2026-07-03-d1-course-features-design.md セクション3・4（実験1）
    + evaluator 差し戻し指示（v39_course_slim: 出力は front_pref_x_small 1列のみ）
    """
    # track_code の分布確認ログ（想定外コードの検出用）。
    # 既知の懸念: 芝コード 20〜22 は preprocess.py の surface_code = track_code // 10
    # によりダート扱いになる。実測ではフィルタ後 2 レースのみで影響は無視できる。
    print("  track_code value_counts (rows):")
    vc = df["track_code"].value_counts().sort_index()
    for code, cnt in vc.items():
        print(f"    track_code={code}: {cnt:,}")

    df["course_is_small"] = (
        df["course_code"].astype(int).isin(SMALL_COURSE_CODES).astype(np.int8)
    )

    n_small = int(df["course_is_small"].sum())
    print(f"  course_is_small=1 (中間変数): {n_small:,} rows ({n_small / len(df):.1%})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HISTORICAL FEATURES
# 全て shift(1) でリーク防止。horse_id × race_date でソート後に計算。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """過去走成績ベースの特徴量を生成する。

    Notes
    -----
    - df は事前にフィルタ済みであること（DNF 等は除外済み）
    - sort_values(['ketto_num', 'race_date']) 後の順序で shift(1) を適用
    - groupby + transform(lambda x: x.shift(1).rolling/expanding) パターンを使用
    """
    # race_date でソートされた状態を保証する
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)

    # レース内の平均走破タイム（同レース内の相対評価用）
    # finish_rank > 0 の馬のみで計算（既にフィルタ済みなので全行有効）
    race_avg_time = df.groupby("race_id")["racetime"].transform("mean")
    df["_time_dev"] = df["racetime"] - race_avg_time

    # ─── 着順系 ───────────────────────────────────────────────────────────────
    grp_horse = df.groupby("ketto_num")

    df["hist_last_rank"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_rank_3"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_rank_5"] = grp_horse["finish_rank"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    df["hist_win_rate"] = grp_horse["is_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["hist_place_rate"] = grp_horse["is_place"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ─── タイム系 ──────────────────────────────────────────────────────────────
    df["hist_last_last3f"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_last3f_3"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_last3f_5"] = grp_horse["time_3f_after"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["hist_last_time_dev"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1)
    )
    df["hist_avg_time_dev_3"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["hist_avg_time_dev_5"] = grp_horse["_time_dev"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # ─── 上がり3F レース内偏差（v48_agari_turn） ───────────────────────────────
    # race_avg_agari は当走レース内の他馬を含むが shift(1) で当走除外 → リークなし。
    # expanding mean（hist_avg_agari3f_dev）は hist_avg_time_dev_* と |r|>=0.7 のため不採用。
    # 前走1走のみ（hist_last_agari3f_dev）に限定して相関ゲートを通過させる。
    race_avg_agari = df.groupby("race_id")["time_3f_after"].transform("mean")
    df["_agari_dev"] = df["time_3f_after"] - race_avg_agari
    # 上がり3F偏差 − 走破タイム偏差: 「末脚が総合タイム偏差に対してどれだけ上/下か」
    # 生の agari_dev / time_dev 単独列は hist_last_time_dev と |r|>=0.7 のため不採用。
    df["_agari_time_gap"] = df["_agari_dev"] - df["_time_dev"]
    df["hist_last_agari_time_gap"] = grp_horse["_agari_time_gap"].transform(
        lambda x: x.shift(1)
    )

    # ─── 馬場条件別 最速タイム ──────────────────────────────────────────────────
    # 同実距離×同馬場種別×同馬場状態での過去最速タイム（shift(1) で当該レース除外）
    # NOTE: distance_category だと同カテゴリ内の実距離差（例: 1000m と 1400m で
    # 20秒超）が生タイムに混入し疑似信号化するため、実距離でグループ化する
    # （2026-06-30 の速度指数バグ修正と同じ根拠。A-2 で distance 化を採用）
    df["hist_best_time_same_cond"] = (
        df.groupby(
            ["ketto_num", "distance", "surface_code", "track_condition_code"]
        )["racetime"].transform(
            lambda x: x.shift(1).expanding().min()
        )
    )

    # ─── 馬場適性系 ───────────────────────────────────────────────────────────
    # 各グループ内で race_date 順に並んでいる前提（sort_values 済み）
    df["hist_same_surface_win_rate"] = (
        df.groupby(["ketto_num", "surface_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_condition_win_rate"] = (
        df.groupby(["ketto_num", "track_condition_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_surface_condition_win_rate"] = (
        df.groupby(["ketto_num", "surface_condition"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_course_win_rate"] = (
        df.groupby(["ketto_num", "course_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_dist_win_rate"] = (
        df.groupby(["ketto_num", "distance_category"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    # v48_agari_turn: 回り×馬場（race_type_code）勝率 − 同馬場勝率（回り方向の上乗せ）
    # 絶対勝率（hist_same_turn_surface_win_rate）は hist_win_rate と |r|>=0.7 のため不採用。
    if "race_type_code" in df.columns:
        _hist_turn_surface_wr = (
            df.groupby(["ketto_num", "race_type_code"])["is_win"].transform(
                lambda x: x.shift(1).expanding().mean()
            )
        )
        df["hist_turn_surface_win_edge"] = (
            _hist_turn_surface_wr - df["hist_same_surface_win_rate"]
        )
    else:
        df["hist_turn_surface_win_edge"] = np.nan

    # race_type_code は hist 集約キー専用。parquet 列数を v39+2 に保つため drop。
    df.drop(columns=["race_type_code"], errors="ignore", inplace=True)

    # v39_course: hist_track_size_win_rate（馬×コースサイズのプーリング勝率）は
    # 相関ゲートで hist_win_rate と r=+0.851（>= 0.7）となり不採用。
    # 仕様書（docs/specs/2026-07-03-d1-course-features-design.md 実験1）の例外規定
    # 「|r| >= 0.7 なら本列のみ落として3列で学習」に従い実装しない。

    # ─── 状態系 ───────────────────────────────────────────────────────────────
    # diff() は current - previous なので shift 不要（current - prev は過去情報）
    df["hist_days_since_last"] = grp_horse["race_date"].transform(
        lambda x: x.diff().dt.days
    )
    # 前走の馬体重変化（shift(1) で当該レース除外）
    df["hist_weight_change"] = grp_horse["horse_weight_change"].transform(
        lambda x: x.shift(1)
    )

    # ─── 賞金系 ───────────────────────────────────────────────────────────────
    df["hist_total_prize"] = grp_horse["hon_shokin"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df["hist_avg_prize_3"] = grp_horse["hon_shokin"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # ─── 天候適性系 ───────────────────────────────────────────────────────────────
    df["hist_same_weather_win_rate"] = (
        df.groupby(["ketto_num", "weather_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["hist_same_weather_avg_rank"] = (
        df.groupby(["ketto_num", "weather_code"])["finish_rank"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── コース×距離帯複合適性 ────────────────────────────────────────────────────
    df["hist_same_course_dist_win_rate"] = (
        df.groupby(["ketto_num", "course_code", "distance_category"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── グレード適性系 ───────────────────────────────────────────────────────────
    df["hist_same_grade_win_rate"] = (
        df.groupby(["ketto_num", "grade_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df["_is_top_grade"] = (df["grade_code"] >= 5).astype(np.int8)
    df["hist_top_grade_exp_count"] = df.groupby("ketto_num")["_is_top_grade"].transform(
        lambda x: x.shift(1).expanding().sum()
    )

    # ─── 精細距離適性（100m単位） ──────────────────────────────────────────────────
    df["_dist_bin_100"] = (df["distance"] // 100) * 100
    df["hist_exact_dist_win_rate"] = (
        df.groupby(["ketto_num", "_dist_bin_100"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # ─── クラス移動特徴量 ──────────────────────────────────────────────────────────
    # grade_code は大きいほど格上（=1: 条件戦, =5: OP, =7: 重賞）
    # 過去の最高格（最大 grade_code = 格上）
    df["hist_best_grade_ever"] = grp_horse["grade_code"].transform(
        lambda x: x.shift(1).expanding().max()
    )
    # 今回 grade_code と過去最高格の差（正=格下出走=有利, 負=格上挑戦=不利）
    df["hist_grade_diff"] = df["hist_best_grade_ever"] - df["grade_code"]

    # 重賞（grade_code >= 7）での過去平均着順（NaN率80〜90%は想定通り）
    df["_rank_top_grade"] = df["finish_rank"].where(df["grade_code"] >= 7, other=np.nan)
    df["hist_avg_rank_top_grade"] = df.groupby("ketto_num")["_rank_top_grade"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ─── 先行傾向（running_style_code は過去走の値。shift(1) で当該レース除外） ───
    # running_style_code: 1=逃げ, 2=先行, 3=差し, 4=追込 / 先行系 = {1, 2}
    df["_is_front_runner"] = df["running_style_code"].isin([1, 2]).astype(np.int8)
    df["hist_front_running_pref"] = grp_horse["_is_front_runner"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 一時列を削除
    df = df.drop(columns=[
        "_time_dev", "_agari_dev", "_agari_time_gap", "_is_top_grade", "_dist_bin_100",
        "_rank_top_grade", "_is_front_runner",
    ])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3.5: SIX-PARAM HIST FEATURES (v49_six_lap)
# スライド 2-3 の6パラメータを shift(1) + 直近10走で hist 化する。
# race_first_3f / race_last_3f は horse_data 由来（レース確定後だが過去走集約のみ使用）。
# ═══════════════════════════════════════════════════════════════════════════════

SIX_PARAM_ROLL: int = 10


def _merge_race_pace_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """horse_data からレースペース（テン3F・上り3F）を race_id でマージする。"""
    hd_path = resolve_project_path(cfg["data"]["src_parquet_dir"]) / "horse_data.parquet"
    pace = pd.read_parquet(
        hd_path, columns=["race_id", "race_first_3f", "race_last_3f"]
    ).drop_duplicates(subset=["race_id"])
    pace["race_id"] = pace["race_id"].astype(str)
    n_before = len(df)
    df = df.copy()
    df["race_id"] = df["race_id"].astype(str)
    df = df.merge(pace, on="race_id", how="left")
    assert len(df) == n_before, (
        f"race pace マージで行数が変化: {n_before:,} → {len(df):,}"
    )
    return df


def _build_six_param_hist_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """6パラメータ hist 特徴量（v49_six_lap 専用）。

    相関ゲート通過のため、生の上がり3Fベスト（既存 hist_last_last3f と重複）を避け、
    レース内偏差・センタリング・ラップバランスを用いる。

    | 列名 | スライド |
    |------|---------|
    | hist_competitive_spirit | 勝負根性 |
    | hist_explosive_agari_gap | 瞬発力（後方→馬券内の末脚偏差 min） |
    | hist_tracking_power | 追走力 |
    | hist_pref_lap_balance | 得意ラップ（馬券内 lap_balance 平均） |
    | hist_dash_ten3f_centered | ダッシュ力（馬券内 ten3F − 馬自身平均） |
    | hist_stamina_agari_gap | スタミナ（馬券内で遅い末脚偏差 max） |

    スピード（上がり2Fベスト）は JV に2F未収録のため既存 hist_avg_last3f_* で代替。
    """
    df = _merge_race_pace_columns(df, cfg)
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)

    winner_time = df.groupby("race_id")["racetime"].transform("min")
    n_f = (df["distance"] / 200).round().astype(int)
    first3 = df["race_first_3f"]
    last3 = df["race_last_3f"]
    mid_cnt = n_f - 6
    front_cnt = (n_f - 3).clip(lower=1)

    df["_race_middle_3f"] = np.where(
        mid_cnt > 0,
        (winner_time - first3 - last3) / mid_cnt * 3,
        np.nan,
    )
    df["_race_ten_3f"] = first3
    df["_lap_balance"] = (winner_time - last3) / front_cnt * 3 - last3

    race_avg_agari = df.groupby("race_id")["time_3f_after"].transform("mean")
    df["_agari_gap"] = df["time_3f_after"] - race_avg_agari

    second_time = df.groupby("race_id")["racetime"].transform(
        lambda s: s.sort_values().iloc[1] if s.notna().sum() >= 2 else np.nan
    )
    df["_win_margin"] = np.where(
        df["finish_rank"] == 1,
        second_time - df["racetime"],
        np.nan,
    )

    in_money = df["finish_rank"] <= 3
    back_in_money = in_money & (df["corner_4"] >= 4)

    df["_explosive_src"] = np.where(back_in_money, df["_agari_gap"], np.nan)
    df["_tracking_src"] = np.where(in_money, df["_race_middle_3f"], np.nan)
    df["_pref_lap_src"] = np.where(in_money, df["_lap_balance"], np.nan)
    df["_stamina_src"] = np.where(in_money, df["_agari_gap"], np.nan)

    grp = df.groupby("ketto_num")
    df["_dash_mean_past"] = grp["_race_ten_3f"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["_dash_src"] = np.where(
        in_money,
        df["_race_ten_3f"] - df["_dash_mean_past"],
        np.nan,
    )

    roll = SIX_PARAM_ROLL

    df["hist_competitive_spirit"] = grp["_win_margin"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).max()
    )
    df["hist_explosive_agari_gap"] = grp["_explosive_src"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).min()
    )
    df["hist_tracking_power"] = grp["_tracking_src"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).min()
    )
    df["hist_pref_lap_balance"] = grp["_pref_lap_src"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).mean()
    )
    df["hist_dash_ten3f_centered"] = grp["_dash_src"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).min()
    )
    df["hist_stamina_agari_gap"] = grp["_stamina_src"].transform(
        lambda x: x.shift(1).rolling(roll, min_periods=1).max()
    )

    df = df.drop(
        columns=[
            "race_first_3f", "race_last_3f",
            "_race_middle_3f", "_race_ten_3f", "_lap_balance", "_agari_gap",
            "_win_margin", "_explosive_src", "_tracking_src", "_pref_lap_src",
            "_stamina_src", "_dash_mean_past", "_dash_src",
        ],
        errors="ignore",
    )

    for col in NEW_FEATURE_COLS_BY_VERSION["v49_six_lap"]:
        print(f"  {col}: NaN率 {df[col].isna().mean():.1%}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CURRENT FEATURES
# 当該レースの固定情報。リーク防止不要（レース前に観測可能な情報）。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_current_features(df: pd.DataFrame, version: str = "") -> pd.DataFrame:
    """当該レースの現状情報特徴量を生成する。

    version: "v44_handicap" のときのみ is_handicap_race（候補3: ハンデ戦フラグ）、
             "v45_transport" のときのみ transport_category（候補4: 輸送距離
             カテゴリ）を追加する。他バージョンでは生成せず、既存バージョンの
             列数を変えない（1変更1実験の原則）。
    """
    # 季節 × 性別スコア: cos(2π × day_of_year/365) × sex_sign
    # sex_sign: 牝馬(sex_code=2)=+1, 牡馬(1)・騸馬(3)=-1
    day_of_year = df["race_date"].dt.dayofyear
    sex_sign = df["sex_code"].map({1: -1, 2: 1, 3: -1}).fillna(0).astype(float)
    df["season_sex_score"] = np.cos(2 * np.pi * day_of_year / 365) * sex_sign

    # 枠番 × 馬場種別 交互作用: 芝=+1, ダート=−1
    # 芝は内枠有利、ダートは大きな差なし（方向性を数値化）
    surface_sign = df["surface_code"].map({1: 1, 2: -1}).fillna(0).astype(float)
    df["wakuban_surface"] = df["wakuban"].astype(float) * surface_sign

    # v39_course_slim: 小回り × 先行傾向 交互作用。
    # 直線の短いコースでは先行馬が残りやすい物理バイアスを積として明示する
    # （depth-2 分割でも表現可能だが、LambdaRank の相対勾配では積の方が発見が容易）。
    # hist_front_running_pref が NaN（新馬）なら NaN のまま（fillna しない）。
    df["front_pref_x_small"] = (
        df["hist_front_running_pref"] * df["course_is_small"].astype(float)
    )
    # course_is_small は中間変数（evaluator 指示: parquet に出力しない）。使用後に drop
    df = df.drop(columns=["course_is_small"])

    # ─── フィールド強度（SECTION 3完了後に依存） ─────────────────────────────────
    # 新馬(hist_win_rate=NaN)は 0 として fillna してから groupby で平均を取る
    df["_hist_win_rate_filled"] = df["hist_win_rate"].fillna(0)
    df["field_avg_win_rate"] = df.groupby("race_id")["_hist_win_rate_filled"].transform("mean")
    df["field_avg_prize"] = df.groupby("race_id")["hist_avg_prize_3"].transform("mean")
    df["win_rate_vs_field"] = df["hist_win_rate"] - df["field_avg_win_rate"]
    df["prize_vs_field"] = df["hist_avg_prize_3"] - df["field_avg_prize"]
    df = df.drop(columns=["_hist_win_rate_filled"])

    # v44_handicap（候補3）: ハンデ戦フラグ。
    # weight_type（重量種別コード。docs/JV-Data.md #2008.重量種別コード）:
    #   1=ハンデ, 2=別定, 3=馬齢, 4=定量, 0=未設定・未整備（主に地方/海外）。
    # コード値は目視憶測ではなく、学習期間データの実測（burden_weight のレース内
    # 標準偏差・レンジが weight_type=1 で最大: std=1.51/range=5.08、他コードは
    # std<=1.32/range<=3.83）と、weight_type=1 のレースが grade_code 1/2/3
    # （G1/G2/G3）に一切出現しない（実際の JRA 運用でも重賞はほぼ別定/馬齢で
    # 施行されハンデはほぼ皆無という既知の事実と整合）ことの両方で確定した。
    # レース番組情報として出走前確定（市場情報ではない）。
    if version == "v44_handicap":
        vc = df["weight_type"].value_counts(dropna=False).sort_index()
        print(f"  weight_type value_counts (行ベース):\n{vc.to_string()}")
        df["is_handicap_race"] = (df["weight_type"] == 1).astype("int8")
        col = df["is_handicap_race"]
        print(f"  is_handicap_race: 1(ハンデ)の割合 {col.mean():.2%} (n={len(col):,})")

    # weight_type 自体は common.py の FORBIDDEN_COLS でメタ列除外対象。
    # is_handicap_race 生成用の中間ソースとしてのみ使い、生列は parquet に残さない
    # （v42_mining の mining_predicted_rank と同じ「派生列に一本化して drop」の流儀）。
    df = df.drop(columns=["weight_type"], errors="ignore")

    # v45_transport（候補4）: 輸送距離カテゴリ。
    # region_code（東西所属コード。_load_data() で var2.0.0 の SE_preprocessed.parquet
    # から race_id×ketto_num でマージ済み）× course_code から TRANSPORT_MAP
    # （地理的事実に基づく固定マッピング。モジュール定数として定義済み）を引く。
    # 馬の恒常的な所属情報（レース結果に非依存）と開催地（レース前確定）の
    # 組み合わせのため時系列リークはない。
    if version == "v45_transport":
        vc = df["region_code"].value_counts(dropna=False).sort_index()
        print(f"  region_code value_counts (行ベース):\n{vc.to_string()}")
        df["transport_category"] = _transport_category(
            df["region_code"], df["course_code"]
        ).astype("int8")
        col = df["transport_category"]
        print(f"  transport_category value_counts (0=近郊/1=中距離/2=長距離):\n"
              f"{col.value_counts(dropna=False).sort_index().to_string()}")
        print(f"  transport_category: NaN率 {col.isna().mean():.2%} (n={len(col):,})")

    # region_code 自体は common.py の FORBIDDEN_COLS でメタ列除外対象。
    # transport_category 生成用の中間ソースとしてのみ使い、生列は parquet に残さない
    # （weight_type → is_handicap_race と同じ「派生列に一本化して drop」の流儀）。
    df = df.drop(columns=["region_code"], errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4.5: WAKU COURSE BIAS (v40_waku 実験2)
# コース×馬場×距離帯×枠番の「枠有利不利」を過去の実結果から時系列推定する。
# 仕様書: docs/specs/2026-07-03-d1-course-features-design.md セクション4 実験2
# （仕様書では v39_waku 表記。v39 は course_slim が採用済みのため v40_waku に読み替え）
# ═══════════════════════════════════════════════════════════════════════════════

def _build_waku_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    """枠×コースの時系列超過勝率 hist_waku_course_bias_ts を生成する。

    - 集計キー: (course_code, surface_code, distance_category, wakuban)
    - 集計値: 超過勝率 = is_win − 1/horse_count の累積平均。
      頭数によってベースレート 1/n が変わるため、生の勝率ではなく
      期待値との差を使う（8頭立ての勝利と18頭立ての勝利を同列に扱わない）
    - リーク防止: 日次集計 → cumsum → shift(1) → merge
      （Step J-4 hist_jockey_course_win_rate と完全同型。同日の他レース結果を含めない）
    - 累積観測数 < MIN_WAKU_SAMPLES のセルは NaN
    """
    keys = ["course_code", "surface_code", "distance_category", "wakuban"]

    wk_daily = (
        df.assign(_excess=df["is_win"] - 1.0 / df["horse_count"].astype(float))
        .groupby(keys + ["race_date"], observed=True)
        .agg(d_excess=("_excess", "sum"), d_races=("_excess", "count"))
        .reset_index()
        .sort_values(keys + ["race_date"])
        .reset_index(drop=True)
    )
    grp_wk = wk_daily.groupby(keys, observed=True)
    wk_daily["cum_excess"]      = grp_wk["d_excess"].cumsum()
    wk_daily["cum_races"]       = grp_wk["d_races"].cumsum()
    wk_daily["cum_excess_prev"] = grp_wk["cum_excess"].shift(1)
    wk_daily["cum_races_prev"]  = grp_wk["cum_races"].shift(1)
    wk_daily["hist_waku_course_bias_ts"] = (
        wk_daily["cum_excess_prev"] / wk_daily["cum_races_prev"]
    )
    wk_daily.loc[
        wk_daily["cum_races_prev"] < MIN_WAKU_SAMPLES,
        "hist_waku_course_bias_ts",
    ] = np.nan

    # セル数の確認ログ（仕様書見込み: 最大 640 セル。存在しない組み合わせは正常）
    print(f"  waku bias cells (course×surface×dist_cat×waku): {grp_wk.ngroups:,}")

    df = df.merge(
        wk_daily[keys + ["race_date", "hist_waku_course_bias_ts"]],
        on=keys + ["race_date"],
        how="left",
    )

    col = df["hist_waku_course_bias_ts"]
    print(f"  hist_waku_course_bias_ts: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.5f}, std {col.std():.5f}, "
          f"min {col.min():+.5f}, max {col.max():+.5f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4.6: MINING FEATURES (v42_mining 実験1・Phase 6)
# JRA公式データマイニング予想（0B13, race_se.mining_predicted_rank）の着順予想順位。
# 仕様書: docs/specs/2026-07-04-phase6-jra-mining-design.md セクション2
# 当該レースの発走前に JRA が確定・公開する事前予想値であり、過去走の集計値ではないため
# shift(1) は不要（むしろ shift すると前走のマイニング予想を誤って使うことになり不適切）。
# 1変更1実験の原則（仕様書2章）: mining_uncertainty 等の派生特徴量は v42 実験1では追加しない。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_mining_features(df: pd.DataFrame) -> pd.DataFrame:
    """mining_pred_rank（JRAマイニング予想順位）を1列だけ追加する。

    1=予想最速、horse_count=予想最遅。0/欠損は preprocess.py 側で既に NaN 化済み。
    生の元列 mining_predicted_rank は出力しない（mining_pred_rank に一本化して drop）。
    """
    df["mining_pred_rank"] = df["mining_predicted_rank"].astype(float)

    col = df["mining_pred_rank"]
    print(f"  mining_pred_rank: 非欠損率 {col.notna().mean():.2%}, "
          f"mean {col.mean():.3f}, std {col.std():.3f}, "
          f"min {col.min():.1f}, max {col.max():.1f}")

    # リーク簡易チェック（仕様書0-2節・2章）:
    # mining_pred_rank=1 の実際勝率が 100% ではないこと（事前予想が時々外れる健全なデータか）
    top1 = df[df["mining_pred_rank"] == 1.0]
    if len(top1) > 0:
        win_rate = top1["is_win"].mean()
        print(f"  [leak check] mining_pred_rank=1 の実際勝率: {win_rate:.2%} "
              f"(n={len(top1):,}. 100%に近い場合はリーク疑いにつき即座に停止)")
        if win_rate > 0.9:
            raise RuntimeError(
                f"[LEAK SUSPECTED] mining_pred_rank=1 の実際勝率が {win_rate:.2%} と"
                f"異常に高すぎます。100%はレース結果のコピーである強い疑いです。"
                f"即座に停止し evaluator へ報告してください。"
            )

    # 生の元列は出力しない
    df = df.drop(columns=["mining_predicted_rank"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.75: PACE INTERACTION (v41_pace 実験3・D-1 最終実験)
# 自馬を除いたレース内の先行傾向密度と、自馬の先行傾向の交互作用。
# 仕様書: docs/specs/2026-07-03-d1-course-features-design.md セクション4 実験3
# （仕様書表記は v39_pace。v39/v40 は既に使用済みのため v41_pace に読み替える）
# ═══════════════════════════════════════════════════════════════════════════════

# センタリング定数 c（生成実行時に main() で計算しログ・manifest に記録する）
PACE_CENTER_CONST: dict[str, float] = {}


def _build_pace_interaction_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """front_pref_x_pace = hist_front_running_pref × (自馬除外先行密度 − c) を生成する。

    - 自馬除外密度 density_others: レース内の他馬の pref_filled 平均。
      pref_filled = hist_front_running_pref.fillna(0)（新馬は 0 として扱う）。
      効率的な自馬除外計算: (race_sum − 自馬 pref_filled) / (horse_count − 1)
      （既存 field_front_runner_density は自馬を含む全馬平均であり、これとは別物）
    - センタリング定数 c: 学習+バリデーション期間（race_date <= valid_end。
      本プロジェクトの train_config.json では 2024-12-31）の density_others の平均。
      テスト期間（2025+）を含めずに算出し、値をログに記録する
      （Phase A 教訓の直接適用: センタリングなしの素朴な積は自己相関成分により
      hist_front_running_pref と r > 0.9 になることがほぼ確実。自馬除外で自己相関を消し、
      センタリングで積の符号を反転可能にすることで相関を落とす）
    - hist_front_running_pref が NaN（新馬）なら本列も NaN のまま（fillna しない）
    """
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])

    df["_pref_filled_pace"] = df["hist_front_running_pref"].fillna(0)
    race_sum = df.groupby("race_id")["_pref_filled_pace"].transform("sum")
    density_others = (
        (race_sum - df["_pref_filled_pace"])
        / (df["horse_count"].astype(float) - 1.0)
    )

    # センタリング定数 c: 学習+バリデーション期間のみ（テスト期間 2025+ は含めない）
    train_valid_mask = df["race_date"] <= valid_end
    c = float(density_others[train_valid_mask].mean())
    PACE_CENTER_CONST["c"] = c
    print(f"  front_pref_x_pace centering constant c = {c:.6f} "
          f"(race_date <= {valid_end.date()}, train+valid combined, "
          f"n={int(train_valid_mask.sum()):,})")

    df["front_pref_x_pace"] = df["hist_front_running_pref"] * (density_others - c)
    df = df.drop(columns=["_pref_filled_pace"])

    col = df["front_pref_x_pace"]
    print(f"  front_pref_x_pace: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.5f}, std {col.std():.5f}, "
          f"min {col.min():+.5f}, max {col.max():+.5f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.76: Phase E — 福島・小倉弱点対策（v46 / v47）
# 仕様書: docs/specs/2026-07-04-fukushima-kokura-features-design.md
# ═══════════════════════════════════════════════════════════════════════════════

DENSITY_CENTER_CONST: dict[str, float] = {}


def _build_small_course_pool_features(df: pd.DataFrame) -> pd.DataFrame:
    """hist_small_course_pool_win_rate_ts: 小回り4場プールの時系列勝率（shift(1)）。

    course_code ∈ SMALL_COURSE_CODES の過去走のみ集計し、
    hist_same_course_win_rate（course 単位）の NaN を補完する情報を提供する。
    """
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    df["_small_course_win"] = np.where(
        df["course_code"].astype(int).isin(SMALL_COURSE_CODES),
        df["is_win"],
        np.nan,
    )
    df["hist_small_course_pool_win_rate_ts"] = (
        df.groupby("ketto_num")["_small_course_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )
    df = df.drop(columns=["_small_course_win"])

    col = df["hist_small_course_pool_win_rate_ts"]
    print(f"  hist_small_course_pool_win_rate_ts: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():.4f}, std {col.std():.4f}")
    return df


def _build_front_pref_x_density(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """front_pref_x_density = hist_front_running_pref × (field_front_runner_density − c).

    v41_pace（自馬除外密度）とは異なり、既存 field_front_runner_density を使用。
    センタリング定数 c は学習+valid期間のみで算出（Rule 3）。
    """
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    train_valid_mask = df["race_date"] <= valid_end
    c = float(df.loc[train_valid_mask, "field_front_runner_density"].mean())
    DENSITY_CENTER_CONST["c"] = c
    print(f"  front_pref_x_density centering constant c = {c:.6f} "
          f"(race_date <= {valid_end.date()}, n={int(train_valid_mask.sum()):,})")

    df["front_pref_x_density"] = (
        df["hist_front_running_pref"]
        * (df["field_front_runner_density"] - c)
    )

    col = df["front_pref_x_density"]
    print(f"  front_pref_x_density: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.5f}, std {col.std():.5f}")
    return df


def _build_post_position_x_small(df: pd.DataFrame) -> pd.DataFrame:
    """post_position_x_small = relative_post_position × course_is_small（小回り4場）。

    v40_waku（枠×コース時系列バイアス）単体は -0.10pp 不合格だったが、
    既存 relative_post_position との交互作用で小回り枠バイアスを明示する。
    course_is_small は中間変数としてのみ使用し parquet には出力しない。
    """
    course_is_small = df["course_code"].astype(int).isin(SMALL_COURSE_CODES).astype(float)
    df["post_position_x_small"] = df["relative_post_position"] * course_is_small
    col = df["post_position_x_small"]
    print(f"  post_position_x_small: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.5f}, std {col.std():.5f}, "
          f"nonzero {(col != 0).sum():,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.79: CUSHION / MOISTURE FEATURES (v51_cushion, Phase A)
# JRAが当日朝に発表するクッション値・含水率。当該レースの結果情報ではなく
# レース前確定の馬場情報であり、市場情報でもないため L1 特徴量として適合する。
# 仕様書: docs/specs/2026-07-12-alpha-recovery-data-expansion-spec.md Phase A
# データ: common/data/output/cushion/cushion_all.csv（2018〜2025、12,792行）
# ═══════════════════════════════════════════════════════════════════════════════

CUSHION_TRACK_AVG_WINDOW_DAYS: int = 365   # A3: 過去1年平均のウィンドウ


def _load_cushion_daily(cfg: dict) -> pd.DataFrame:
    """cushion_all.csv を読み込み、race_date×course_code×surface_code 単位に集約する。

    measure_point_code（同一馬場の複数測定点。ほぼ全レコードで1レースあたり2点）は
    単純平均で当日値に集約する。クッション値は JRA 仕様上、芝のみに設定される
    （ダートは常時 NaN）。2020-09 以前はクッション値自体が未整備で NaN。
    """
    path = resolve_project_path(cfg["data"]["cushion_dir"]) / "cushion_all.csv"
    raw = pd.read_csv(path)
    raw["race_date"] = pd.to_datetime(raw["race_date"].astype(str), format="%Y%m%d")

    day = (
        raw.groupby(["race_date", "course_code", "surface_code"], observed=True)
        .agg(cushion_value=("cushion_value", "mean"), moisture_pct=("moisture_pct", "mean"))
        .reset_index()
    )
    print(f"  cushion_all.csv: {len(raw):,} rows -> {len(day):,} "
          f"(race_date x course_code x surface_code) cells")
    return day


def _build_cushion_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """クッション値・含水率特徴量（A1〜A3, A6, v51_cushion 専用）を生成する。

    A1 cushion_value / A2 moisture_pct: 当日発表値をそのままマージ（shift不要。
        当該レースの結果情報ではなく、レース前に確定している馬場情報のため）。
    A3 cushion_diff_track_avg: 当日値 − 当該場・当該surfaceの過去1年平均（当日除く）。
    A6 last3f_rank_x_cushion: 当該馬の過去走上がり3F順位（shift(1)+expanding平均）×
        当日クッション値。

    A4 hist_perf_similar_cushion / A5 hist_perf_similar_moisture は相関ゲート不合格
    （既存 hist_place_rate と r=0.75/0.73、帯フィルタが広すぎて実質同一情報）のため
    2026-07-12 に削除。詳細: docs/specs/2026-07-12-alpha-recovery-data-expansion-spec.md
    """
    cushion_day = _load_cushion_daily(cfg)

    n_before = len(df)
    df = df.merge(
        cushion_day, on=["race_date", "course_code", "surface_code"], how="left"
    )
    assert len(df) == n_before, (
        f"cushion マージで行数が変化しています（ファンアウト検出）: "
        f"{n_before:,} -> {len(df):,}"
    )
    print(f"  A1 cushion_value: 非欠損率 {df['cushion_value'].notna().mean():.2%} "
          f"(surface_code=2(ダート)・2020-09以前はNaN想定)")
    print(f"  A2 moisture_pct: 非欠損率 {df['moisture_pct'].notna().mean():.2%}")

    # ─── A3: cushion_diff_track_avg ────────────────────────────────────────
    track_idx = (
        cushion_day.sort_values(["course_code", "surface_code", "race_date"])
        .set_index("race_date")
    )
    track_roll = (
        track_idx.groupby(["course_code", "surface_code"], observed=True)["cushion_value"]
        .rolling(f"{CUSHION_TRACK_AVG_WINDOW_DAYS}D", closed="left")
        .mean()
        .reset_index()
        .rename(columns={"cushion_value": "_track_avg_cushion"})
    )
    df = df.merge(
        track_roll, on=["course_code", "surface_code", "race_date"], how="left"
    )
    df["cushion_diff_track_avg"] = df["cushion_value"] - df["_track_avg_cushion"]
    df = df.drop(columns=["_track_avg_cushion"])
    col = df["cushion_diff_track_avg"]
    print(f"  A3 cushion_diff_track_avg: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.4f}, std {col.std():.4f}")

    # ─── A6: last3f_rank_x_cushion ──────────────────────────────────────────
    # 上がり3F順位はレース内順位（1=最速）。当該レースの結果情報だが shift(1) で
    # 過去走のみを参照するため当該レースは含まれない。
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    df["_agari_rank"] = df.groupby("race_id")["time_3f_after"].rank(
        method="min", ascending=True
    )
    df["_hist_agari_rank_avg"] = df.groupby("ketto_num")["_agari_rank"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["last3f_rank_x_cushion"] = df["_hist_agari_rank_avg"] * df["cushion_value"]
    df = df.drop(columns=["_agari_rank", "_hist_agari_rank_avg"])
    col = df["last3f_rank_x_cushion"]
    print(f"  A6 last3f_rank_x_cushion: NaN率 {col.isna().mean():.1%}, "
          f"mean {col.mean():+.4f}, std {col.std():.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BLOODLINE FEATURES
# 父馬・母父の産駒成績（Phase 3: 時系列正確版）。
# 日次集計 → 累積 → shift(1) でリーク防止済み。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sire_features(df: pd.DataFrame, version: str = "") -> pd.DataFrame:
    """父馬・母父産駒の成績を時系列正確版で計算する。

    アプローチ: 日次集計 → 累積 → shift(1) → メイン df にマージ
    理由: 同一 sire_id の産駒が同日複数レースに出走しうるため、
         ketto_num 単位の shift(1) では同日他産駒の結果が混入する。
         日次集計後に shift(1) することで当日を含まない累計を保証する。

    version: "v43_sire_tc" のときのみ hist_sire_track_condition_win_rate_ts
             （候補2: 父×馬場状態別勝率）を追加する。他バージョンでは生成せず、
             既存バージョンの列数を変えない（1変更1実験の原則）。
    """
    # 産駒数が少ない父馬（新種牡馬等）の累積勝率はS/N比が低くノイズになるため、
    # cum_races_prev < MIN_SIRE_RACES の場合は NaN を設定し、
    # LightGBM の欠損値分岐に処理を委ねる。
    MIN_SIRE_RACES = 30

    # ─── sire 特徴量 ──────────────────────────────────────────────────────────
    if "sire_id" not in df.columns or df["sire_id"].isna().all():
        for col in ["hist_sire_win_rate_ts", "hist_sire_surface_win_rate_ts",
                    "hist_sire_dist_win_rate_ts", "hist_sire_dist_diff"]:
            df[col] = np.nan
        if version == "v43_sire_tc":
            df["hist_sire_track_condition_win_rate_ts"] = np.nan
    else:
        # ── 通算勝率（sire × race_date） ──────────────────────────────────────
        sire_daily = (
            df.groupby(["sire_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "race_date"])
        )
        grp_s = sire_daily.groupby("sire_id", observed=True)
        sire_daily["cum_wins"]  = grp_s["d_wins"].cumsum()
        sire_daily["cum_races"] = grp_s["d_races"].cumsum()
        sire_daily["cum_wins_prev"]  = grp_s["cum_wins"].shift(1)
        sire_daily["cum_races_prev"] = grp_s["cum_races"].shift(1)
        sire_daily["hist_sire_win_rate_ts"] = (
            sire_daily["cum_wins_prev"] / sire_daily["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_daily.loc[sire_daily["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_win_rate_ts"] = np.nan
        df = df.merge(
            sire_daily[["sire_id", "race_date", "hist_sire_win_rate_ts"]],
            on=["sire_id", "race_date"], how="left"
        )

        # ── 同馬場勝率（sire × surface_code × race_date） ───────────────────────
        sire_surf = (
            df.groupby(["sire_id", "surface_code", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "surface_code", "race_date"])
        )
        grp_ss = sire_surf.groupby(["sire_id", "surface_code"], observed=True)
        sire_surf["cum_wins"]  = grp_ss["d_wins"].cumsum()
        sire_surf["cum_races"] = grp_ss["d_races"].cumsum()
        sire_surf["cum_wins_prev"]  = grp_ss["cum_wins"].shift(1)
        sire_surf["cum_races_prev"] = grp_ss["cum_races"].shift(1)
        sire_surf["hist_sire_surface_win_rate_ts"] = (
            sire_surf["cum_wins_prev"] / sire_surf["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_surf.loc[sire_surf["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_surface_win_rate_ts"] = np.nan
        df = df.merge(
            sire_surf[["sire_id", "surface_code", "race_date",
                        "hist_sire_surface_win_rate_ts"]],
            on=["sire_id", "surface_code", "race_date"], how="left"
        )

        # ── 同馬場状態勝率（sire × track_condition_code × race_date） ────────────
        # 候補2（v43_sire_tc、docs/specs/2026-07-05-summer-racing-structural-features
        # -design.md セクション2 候補2）: hist_sire_surface_win_rate_ts と全く同じ
        # 日次集計→cumsum→shift(1)パターンで、キーを surface_code → track_condition_code
        # に置換したもの。track_condition_code=0（不明）もキーに含めて計算し、
        # 結果は NaN 扱いで問題ない（他バージョンとの1変更1実験を守るため
        # version == "v43_sire_tc" のときのみ生成する）。
        if version == "v43_sire_tc":
            sire_tc = (
                df.groupby(["sire_id", "track_condition_code", "race_date"], observed=True)
                .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
                .reset_index()
                .sort_values(["sire_id", "track_condition_code", "race_date"])
            )
            grp_stc = sire_tc.groupby(["sire_id", "track_condition_code"], observed=True)
            sire_tc["cum_wins"]  = grp_stc["d_wins"].cumsum()
            sire_tc["cum_races"] = grp_stc["d_races"].cumsum()
            sire_tc["cum_wins_prev"]  = grp_stc["cum_wins"].shift(1)
            sire_tc["cum_races_prev"] = grp_stc["cum_races"].shift(1)
            sire_tc["hist_sire_track_condition_win_rate_ts"] = (
                sire_tc["cum_wins_prev"] / sire_tc["cum_races_prev"]
            )
            # 産駒データが少ない場合のNaNマスク（ノイズ抑制、MIN_SIRE_RACES を共通利用）
            sire_tc.loc[
                sire_tc["cum_races_prev"] < MIN_SIRE_RACES,
                "hist_sire_track_condition_win_rate_ts"
            ] = np.nan
            df = df.merge(
                sire_tc[["sire_id", "track_condition_code", "race_date",
                          "hist_sire_track_condition_win_rate_ts"]],
                on=["sire_id", "track_condition_code", "race_date"], how="left"
            )

        # ── 同距離帯勝率（sire × distance_category × race_date） ─────────────────
        sire_dist = (
            df.groupby(["sire_id", "distance_category", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["sire_id", "distance_category", "race_date"])
        )
        grp_sd = sire_dist.groupby(["sire_id", "distance_category"], observed=True)
        sire_dist["cum_wins"]  = grp_sd["d_wins"].cumsum()
        sire_dist["cum_races"] = grp_sd["d_races"].cumsum()
        sire_dist["cum_wins_prev"]  = grp_sd["cum_wins"].shift(1)
        sire_dist["cum_races_prev"] = grp_sd["cum_races"].shift(1)
        sire_dist["hist_sire_dist_win_rate_ts"] = (
            sire_dist["cum_wins_prev"] / sire_dist["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        sire_dist.loc[sire_dist["cum_races_prev"] < MIN_SIRE_RACES, "hist_sire_dist_win_rate_ts"] = np.nan
        df = df.merge(
            sire_dist[["sire_id", "distance_category", "race_date",
                        "hist_sire_dist_win_rate_ts"]],
            on=["sire_id", "distance_category", "race_date"], how="left"
        )

        # ── 父産駒の平均勝ち距離との差（時系列累積版・当日除外） ────────────────
        # 旧実装は全期間（テスト期間含む）の勝利で平均勝ち距離を計算しており
        # Rule 2（時系列リーク防止）違反だった（課題 A-1）。
        # 他の _ts 特徴量と同じ「日次集計 → cumsum → shift(1)」パターンに置換。
        win_dist_daily = (
            df[df["is_win"] == 1]
            .groupby(["sire_id", "race_date"], observed=True)
            .agg(d_dist_sum=("distance", "sum"), d_wins=("distance", "count"))
            .reset_index()
            .sort_values(["sire_id", "race_date"])
        )
        if len(win_dist_daily) > 0:
            grp_w = win_dist_daily.groupby("sire_id", observed=True)
            win_dist_daily["cum_dist"] = grp_w["d_dist_sum"].cumsum()
            win_dist_daily["cum_wins"] = grp_w["d_wins"].cumsum()
            # shift(1) で「その勝利日より前」の累積平均にする（当日勝利を除外）
            win_dist_daily["avg_win_dist_prev"] = (
                grp_w["cum_dist"].shift(1) / grp_w["cum_wins"].shift(1)
            )
            # 勝利日ベースの sparse な系列なので、メイン df へは merge_asof(backward)
            # で「当該レース日より前の最新値」を引き当てる。
            # merge_asof(backward) は当日ちょうどの right 行も拾うが、shift(1) 済み
            # のため同日一致でも当日結果は混入しない。設計意図（当該レース日より
            # 前の情報のみを参照する）を保証するため、right 側キーを +1 日ずらして
            # 当日行とのマッチ自体を排除する。
            win_dist_daily["asof_date"] = (
                win_dist_daily["race_date"] + pd.Timedelta(days=1)
            )
            right = (
                win_dist_daily[["sire_id", "asof_date", "avg_win_dist_prev"]]
                # merge_asof は right が on キーでグローバルソート済みであることを要求
                .sort_values("asof_date", kind="stable")
                .reset_index(drop=True)
            )
            left = (
                df[["sire_id", "race_date"]]
                .reset_index()  # 元の行位置を保持して後で並び順を戻す
                .sort_values("race_date", kind="stable")
            )
            # sire_id 欠損行は by キーでマッチしないため NaN のままになる（許容）
            merged = pd.merge_asof(
                left,
                right,
                left_on="race_date",
                right_on="asof_date",
                by="sire_id",
                direction="backward",
            )
            merged = merged.set_index("index").sort_index()
            df["hist_sire_dist_diff"] = (
                df["distance"] - merged["avg_win_dist_prev"]
            ).abs()
        else:
            df["hist_sire_dist_diff"] = np.nan

    # ─── bms 特徴量 ──────────────────────────────────────────────────────────
    if "bms_id" not in df.columns or df["bms_id"].isna().all():
        df["hist_bms_win_rate_ts"] = np.nan
    else:
        bms_daily = (
            df.groupby(["bms_id", "race_date"], observed=True)
            .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
            .reset_index()
            .sort_values(["bms_id", "race_date"])
        )
        grp_b = bms_daily.groupby("bms_id", observed=True)
        bms_daily["cum_wins"]  = grp_b["d_wins"].cumsum()
        bms_daily["cum_races"] = grp_b["d_races"].cumsum()
        bms_daily["cum_wins_prev"]  = grp_b["cum_wins"].shift(1)
        bms_daily["cum_races_prev"] = grp_b["cum_races"].shift(1)
        bms_daily["hist_bms_win_rate_ts"] = (
            bms_daily["cum_wins_prev"] / bms_daily["cum_races_prev"]
        )
        # 産駒データが少ない場合のNaNマスク（ノイズ抑制）
        bms_daily.loc[bms_daily["cum_races_prev"] < MIN_SIRE_RACES, "hist_bms_win_rate_ts"] = np.nan
        df = df.merge(
            bms_daily[["bms_id", "race_date", "hist_bms_win_rate_ts"]],
            on=["bms_id", "race_date"], how="left"
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.5: JOCKEY / TRAINER FEATURES
# 騎手・調教師の成績特徴量（Phase 4: 時系列正確版）。
# 日次集計 → 累積/rolling → shift(1)/closed='left' でリーク防止済み。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の成績特徴量を時系列正確版で計算する。

    アプローチ:
    - 通算勝率: 日次集計 → cumsum → shift(1) でリーク防止
    - 直近N日勝率: 日次集計 → GroupBy.rolling(ND, closed='left') でリーク防止

    騎手/調教師は同日に複数レースに関与しうるため（実測: 騎手76.7%・調教師74.8%）、
    エントリ単位の shift(1) では同日他レースの結果が混入する。
    日次集計後に処理することで当日を完全除外する。
    """
    # 分母が少ない場合は NaN としてノイズを抑制する（LightGBM の欠損値分岐に委ねる）
    MIN_JOCKEY_RACES = 10
    MIN_TRAINER_RACES = 10

    # ══════════════════════════════════════════════════════════════════════
    # 騎手特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step J-1: 日次集計（jockey × date） ─────────────────────────────────
    jockey_daily = (
        df.groupby(["jockey_code", "race_date"], observed=True)
        .agg(
            d_wins=("is_win", "sum"),
            d_races=("is_win", "count"),
            d_place=("is_place", "sum"),
        )
        .reset_index()
        .sort_values(["jockey_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step J-2: 通算勝率（cumulative + shift(1) で当日を除外） ─────────────
    grp_j = jockey_daily.groupby("jockey_code", observed=True)
    jockey_daily["cum_wins"]       = grp_j["d_wins"].cumsum()
    jockey_daily["cum_races"]      = grp_j["d_races"].cumsum()
    jockey_daily["cum_wins_prev"]  = grp_j["cum_wins"].shift(1)
    jockey_daily["cum_races_prev"] = grp_j["cum_races"].shift(1)
    jockey_daily["hist_jockey_win_rate_cum"] = (
        jockey_daily["cum_wins_prev"] / jockey_daily["cum_races_prev"]
    )
    # 出走数が少ない場合はNaN（デビュー直後のノイズ抑制）
    jockey_daily.loc[
        jockey_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_win_rate_cum",
    ] = np.nan

    df = df.merge(
        jockey_daily[["jockey_code", "race_date", "hist_jockey_win_rate_cum"]],
        on=["jockey_code", "race_date"],
        how="left",
    )

    # ─── Step J-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    # GroupBy.rolling を使うことで apply より効率的に時系列ウィンドウを計算する。
    # closed='left': ウィンドウ = [race_date - ND, race_date) → 当日を除外する。
    jd_idx = jockey_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            jd_idx.groupby("jockey_code", observed=True)[["d_wins", "d_races", "d_place"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()  # → columns: jockey_code, race_date, d_wins, d_races, d_place
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
                "d_place": f"roll_place_{n_days}d",
            })
        )

        # 勝率
        roll[f"hist_jockey_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_JOCKEY_RACES,
            f"hist_jockey_win_rate_{n_days}d",
        ] = np.nan

        merge_cols = ["jockey_code", "race_date", f"hist_jockey_win_rate_{n_days}d"]

        # 30D のみ複勝率を追加（60D は重複情報となるため省略）
        if n_days == 30:
            roll["hist_jockey_place_rate_30d"] = (
                roll["roll_place_30d"] / roll["roll_races_30d"]
            )
            roll.loc[
                roll["roll_races_30d"] < MIN_JOCKEY_RACES,
                "hist_jockey_place_rate_30d",
            ] = np.nan
            merge_cols.append("hist_jockey_place_rate_30d")

        df = df.merge(roll[merge_cols], on=["jockey_code", "race_date"], how="left")

    # ─── Step J-4: 騎手×競馬場 通算勝率（cumulative + shift(1)） ────────────────
    # rolling ではなく cumulative を採用する理由: コース別は30日間のサンプルが
    # 極端に少なく（数レース程度）、累積の方が安定した適性スコアを提供する。
    jc_daily = (
        df.groupby(["jockey_code", "course_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["jockey_code", "course_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_jc = jc_daily.groupby(["jockey_code", "course_code"], observed=True)
    jc_daily["cum_wins"]       = grp_jc["d_wins"].cumsum()
    jc_daily["cum_races"]      = grp_jc["d_races"].cumsum()
    jc_daily["cum_wins_prev"]  = grp_jc["cum_wins"].shift(1)
    jc_daily["cum_races_prev"] = grp_jc["cum_races"].shift(1)
    jc_daily["hist_jockey_course_win_rate"] = (
        jc_daily["cum_wins_prev"] / jc_daily["cum_races_prev"]
    )
    jc_daily.loc[
        jc_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_course_win_rate",
    ] = np.nan

    df = df.merge(
        jc_daily[["jockey_code", "course_code", "race_date", "hist_jockey_course_win_rate"]],
        on=["jockey_code", "course_code", "race_date"],
        how="left",
    )

    # ══════════════════════════════════════════════════════════════════════
    # 調教師特徴量
    # ══════════════════════════════════════════════════════════════════════

    # ─── Step T-1: 日次集計（trainer × date） ─────────────────────────────────
    trainer_daily = (
        df.groupby(["trainer_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "race_date"])
        .reset_index(drop=True)
    )

    # ─── Step T-2: 通算勝率（cumulative + shift(1)） ─────────────────────────
    grp_t = trainer_daily.groupby("trainer_code", observed=True)
    trainer_daily["cum_wins"]       = grp_t["d_wins"].cumsum()
    trainer_daily["cum_races"]      = grp_t["d_races"].cumsum()
    trainer_daily["cum_wins_prev"]  = grp_t["cum_wins"].shift(1)
    trainer_daily["cum_races_prev"] = grp_t["cum_races"].shift(1)
    trainer_daily["hist_trainer_win_rate_cum"] = (
        trainer_daily["cum_wins_prev"] / trainer_daily["cum_races_prev"]
    )
    trainer_daily.loc[
        trainer_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_win_rate_cum",
    ] = np.nan

    df = df.merge(
        trainer_daily[["trainer_code", "race_date", "hist_trainer_win_rate_cum"]],
        on=["trainer_code", "race_date"],
        how="left",
    )

    # ─── Step T-3: rolling 30D・60D 勝率（closed='left' で当日除外） ────────────
    td_idx = trainer_daily.set_index("race_date")

    for n_days in [30, 60]:
        roll = (
            td_idx.groupby("trainer_code", observed=True)[["d_wins", "d_races"]]
            .rolling(f"{n_days}D", closed="left")
            .sum()
            .reset_index()
            .rename(columns={
                "d_wins":  f"roll_wins_{n_days}d",
                "d_races": f"roll_races_{n_days}d",
            })
        )
        roll[f"hist_trainer_win_rate_{n_days}d"] = (
            roll[f"roll_wins_{n_days}d"] / roll[f"roll_races_{n_days}d"]
        )
        roll.loc[
            roll[f"roll_races_{n_days}d"] < MIN_TRAINER_RACES,
            f"hist_trainer_win_rate_{n_days}d",
        ] = np.nan

        df = df.merge(
            roll[["trainer_code", "race_date", f"hist_trainer_win_rate_{n_days}d"]],
            on=["trainer_code", "race_date"],
            how="left",
        )

    # ─── Step T-4: 調教師×馬場種別 通算勝率（cumulative + shift(1)） ────────────
    # 芝・ダート適性は安定した長期特性のため cumulative を採用する。
    ts_daily = (
        df.groupby(["trainer_code", "surface_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "surface_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_ts = ts_daily.groupby(["trainer_code", "surface_code"], observed=True)
    ts_daily["cum_wins"]       = grp_ts["d_wins"].cumsum()
    ts_daily["cum_races"]      = grp_ts["d_races"].cumsum()
    ts_daily["cum_wins_prev"]  = grp_ts["cum_wins"].shift(1)
    ts_daily["cum_races_prev"] = grp_ts["cum_races"].shift(1)
    ts_daily["hist_trainer_surface_win_rate"] = (
        ts_daily["cum_wins_prev"] / ts_daily["cum_races_prev"]
    )
    ts_daily.loc[
        ts_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_surface_win_rate",
    ] = np.nan

    df = df.merge(
        ts_daily[["trainer_code", "surface_code", "race_date", "hist_trainer_surface_win_rate"]],
        on=["trainer_code", "surface_code", "race_date"],
        how="left",
    )

    # ─── Step J-5: 騎手×馬場種別 通算勝率（cumulative + shift(1)） ────────────────
    # 芝・ダート適性は安定した長期特性のため cumulative を採用する。
    js_daily = (
        df.groupby(["jockey_code", "surface_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["jockey_code", "surface_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_js = js_daily.groupby(["jockey_code", "surface_code"], observed=True)
    js_daily["cum_wins"]       = grp_js["d_wins"].cumsum()
    js_daily["cum_races"]      = grp_js["d_races"].cumsum()
    js_daily["cum_wins_prev"]  = grp_js["cum_wins"].shift(1)
    js_daily["cum_races_prev"] = grp_js["cum_races"].shift(1)
    js_daily["hist_jockey_surface_win_rate_ts"] = (
        js_daily["cum_wins_prev"] / js_daily["cum_races_prev"]
    )
    js_daily.loc[
        js_daily["cum_races_prev"] < MIN_JOCKEY_RACES,
        "hist_jockey_surface_win_rate_ts",
    ] = np.nan

    df = df.merge(
        js_daily[["jockey_code", "surface_code", "race_date", "hist_jockey_surface_win_rate_ts"]],
        on=["jockey_code", "surface_code", "race_date"],
        how="left",
    )

    # ─── Step T-5: 調教師×競馬場 通算勝率（cumulative + shift(1)） ────────────────
    # コース適性は安定した長期特性のため cumulative を採用する。
    tc_daily = (
        df.groupby(["trainer_code", "course_code", "race_date"], observed=True)
        .agg(d_wins=("is_win", "sum"), d_races=("is_win", "count"))
        .reset_index()
        .sort_values(["trainer_code", "course_code", "race_date"])
        .reset_index(drop=True)
    )
    grp_tc = tc_daily.groupby(["trainer_code", "course_code"], observed=True)
    tc_daily["cum_wins"]       = grp_tc["d_wins"].cumsum()
    tc_daily["cum_races"]      = grp_tc["d_races"].cumsum()
    tc_daily["cum_wins_prev"]  = grp_tc["cum_wins"].shift(1)
    tc_daily["cum_races_prev"] = grp_tc["cum_races"].shift(1)
    tc_daily["hist_trainer_course_win_rate_ts"] = (
        tc_daily["cum_wins_prev"] / tc_daily["cum_races_prev"]
    )
    tc_daily.loc[
        tc_daily["cum_races_prev"] < MIN_TRAINER_RACES,
        "hist_trainer_course_win_rate_ts",
    ] = np.nan

    df = df.merge(
        tc_daily[["trainer_code", "course_code", "race_date", "hist_trainer_course_win_rate_ts"]],
        on=["trainer_code", "course_code", "race_date"],
        how="left",
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.6: SPEED INDEX FEATURES
# タイム速度指数（Phase 5: 歴史的条件別基準による標準化）。
# 日次集計 → cumsum → shift(1) で当日を除外したリーク防止済み計算。
# ═══════════════════════════════════════════════════════════════════════════════

def _build_speed_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """歴史的条件別基準による速度指数特徴量を生成する。

    アプローチ:
    - 条件グループ (distance[m], surface_code, track_condition_code) 別に
      日次集計 → cumsum → shift(1) で当日を除外した平均・標準偏差を計算する
    - distance_category（粗い区分）ではなく distance（実距離m）を使う。
      カテゴリ内の異なる距離が混在すると speed_idx が距離バイアスを吸収してしまうため。
    - _speed_idx = (cond_avg_time - racetime) / cond_std_time を計算し、
      馬別に shift(1) を適用して horse-level 特徴量を生成する
    - _speed_idx 自体（当該レースの結果情報を含む）は最後に削除する

    Notes
    -----
    - df には racetime, distance, surface_code, track_condition_code,
      ketto_num, race_date が必要（_build_hist_features 後の df に全て存在する）
    - _build_hist_features 内で計算・削除済みの _time_dev は再計算しない
    - この関数は _build_hist_features の後・_build_current_features の前に呼ぶこと
    """
    # 速度指数の基準値を計算するのに必要な最低レース数
    # この値未満の条件では標準偏差が不安定なため NaN マスクを適用する
    MIN_COND_RACES = 20

    # ─── Step 1: 条件別・日次集計 ──────────────────────────────────────────────
    # 同じ条件（distance × surface_code × track_condition_code）で
    # 同日に複数レースが開催される場合があるため、日次で先に集約する。
    # distance_category（粗いカテゴリ）ではなく distance（実距離m）で集計する。
    # 理由: カテゴリ内で異なる距離（例: 1000m〜1400m）が混在すると
    #       speed_idx が「馬の能力」ではなく「どの距離を走ったか」を反映してしまう。
    cond_daily = (
        df.groupby(
            ["distance", "surface_code", "track_condition_code", "race_date"],
            observed=True,
        )
        .agg(
            d_sum_time=("racetime", "sum"),
            d_sum_sq_time=("racetime", lambda x: (x ** 2).sum()),
            d_count=("racetime", "count"),
        )
        .reset_index()
        .sort_values(
            ["distance", "surface_code", "track_condition_code", "race_date"]
        )
        .reset_index(drop=True)
    )

    # ─── Step 2: 条件グループ内での cumsum ────────────────────────────────────
    grp_cond = cond_daily.groupby(
        ["distance", "surface_code", "track_condition_code"],
        observed=True,
    )
    cond_daily["cum_sum"]   = grp_cond["d_sum_time"].cumsum()
    cond_daily["cum_sq"]    = grp_cond["d_sum_sq_time"].cumsum()
    cond_daily["cum_count"] = grp_cond["d_count"].cumsum()

    # ─── Step 3: shift(1) で当日を除いた前日以前の累積を取得 ──────────────────
    cond_daily["cum_sum_prev"]   = grp_cond["cum_sum"].shift(1)
    cond_daily["cum_sq_prev"]    = grp_cond["cum_sq"].shift(1)
    cond_daily["cum_count_prev"] = grp_cond["cum_count"].shift(1)

    # ─── Step 4: 平均・分散・標準偏差の計算 ────────────────────────────────────
    # Welford 公式: Var(X) = E[X^2] - (E[X])^2
    # 浮動小数点誤差で分散が微小な負値になることがあるため clip(lower=0) が必須
    cond_daily["cond_avg_time"] = (
        cond_daily["cum_sum_prev"] / cond_daily["cum_count_prev"]
    )
    cond_daily["cond_var_time"] = (
        cond_daily["cum_sq_prev"] / cond_daily["cum_count_prev"]
        - cond_daily["cond_avg_time"] ** 2
    )
    cond_daily["cond_std_time"] = np.sqrt(
        cond_daily["cond_var_time"].clip(lower=0)
    )

    # 最低レース数未満の条件は NaN マスク（標準偏差が不安定なためノイズ抑制）
    low_count_mask = cond_daily["cum_count_prev"] < MIN_COND_RACES
    cond_daily.loc[low_count_mask, "cond_avg_time"] = np.nan
    cond_daily.loc[low_count_mask, "cond_std_time"] = np.nan

    # ─── Step 5: df にマージ ──────────────────────────────────────────────────
    df = df.merge(
        cond_daily[
            [
                "distance", "surface_code", "track_condition_code",
                "race_date", "cond_avg_time", "cond_std_time",
            ]
        ],
        on=["distance", "surface_code", "track_condition_code", "race_date"],
        how="left",
    )

    # ─── Step 6: 速度指数の計算 ────────────────────────────────────────────────
    # 正の値 = 歴史的平均より速い = 高能力
    # cond_std_time == 0 の場合（全馬同タイム）は NaN を設定する
    df["_speed_idx"] = np.where(
        df["cond_std_time"] > 0,
        (df["cond_avg_time"] - df["racetime"]) / df["cond_std_time"],
        np.nan,
    )

    # ─── Step 7: 馬別 shift(1) で horse-level 特徴量を生成 ───────────────────
    # _build_hist_features の sort_values が継続している前提だが、念のため保証する
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    grp_horse = df.groupby("ketto_num")

    # 前走の速度指数（最もリークから遠い、最重要候補）
    df["hist_speed_idx_last"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1)
    )

    # 過去最高速度指数（能力の上限値）
    df["hist_speed_idx_best"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).expanding().max()
    )

    # 直近3走の速度指数平均（安定した能力推定。hist_avg_time_dev_3 の絶対スケール版）
    df["hist_speed_idx_avg3"] = grp_horse["_speed_idx"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # 同条件（実距離×馬場種別）での過去最高速度指数（条件適性の絶対評価）
    # distance_category ではなく distance を使うことで他の speed_idx 系と一貫性を保つ
    df["hist_speed_idx_cond_best"] = (
        df.groupby(["ketto_num", "distance", "surface_code"])["_speed_idx"]
        .transform(lambda x: x.shift(1).expanding().max())
    )

    # ─── Step 8: 一時列を削除 ─────────────────────────────────────────────────
    # _speed_idx は当該レースの結果情報を含むため特徴量として残してはならない
    # cond_avg_time / cond_std_time も中間計算値であり不要
    df = df.drop(
        columns=["_speed_idx", "cond_avg_time", "cond_std_time"],
        errors="ignore",
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5.7: RELATIVE FEATURES (within-race z-score + pace index)
# hist_speed_idx_avg3 生成後に呼び出すこと（依存関係）。
# ═══════════════════════════════════════════════════════════════════════════════

def _field_zscore(df: pd.DataFrame, col: str, z_col: str) -> None:
    """同レース内 z-score を in-place で追加する。"""
    race_mean = df.groupby("race_id")[col].transform("mean")
    race_std = df.groupby("race_id")[col].transform("std")
    df[z_col] = (df[col] - race_mean) / (race_std + 1e-6)


def _build_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量（within-race z-score + ペース指数）を生成する。"""
    z_pairs = [
        ("hist_last_time_dev", "field_z_time_dev"),
        ("hist_total_prize", "field_z_prize"),
        ("hist_last_last3f", "field_z_last3f"),
        ("hist_win_rate", "field_z_win_rate"),
        ("hist_speed_idx_avg3", "field_z_speed_idx"),
        ("hist_place_rate", "field_z_place_rate"),
    ]
    for src, dst in z_pairs:
        _field_zscore(df, src, dst)

    df["_front_pref_filled"] = df["hist_front_running_pref"].fillna(0)
    df["field_front_runner_density"] = df.groupby("race_id")["_front_pref_filled"].transform("mean")
    df = df.drop(columns=["_front_pref_filled"])

    df["relative_post_position"] = (
        df["wakuban"].astype(float) / df["horse_count"].astype(float)
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAINING FEATURES (HC/WC)
# 調教データは race_date より前のセッションのみを参照する（リーク防止）。
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hc(cfg: dict) -> pd.DataFrame:
    """HC_preprocessed.parquet を読み込む。"""
    p = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "HC_preprocessed.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"HC_preprocessed.parquet が見つかりません: {p}\npreprocess.py を先に実行してください。"
        )
    df = pd.read_parquet(p)
    print(f"  HC: {len(df):,} rows")
    return df


def _load_wc(cfg: dict) -> pd.DataFrame:
    """WC_preprocessed.parquet を読み込む。なければ空 DataFrame を返す。"""
    p = PROJECT_ROOT / cfg["data"]["preprocessed_dir"] / "WC_preprocessed.parquet"
    if not p.exists():
        print("  WC: ファイルなし（スキップ）")
        return pd.DataFrame(
            columns=["ketto_num", "training_date", "wc_3f_sec", "wc_4f_sec", "wc_1f_sec"]
        )
    df = pd.read_parquet(p)
    print(f"  WC: {len(df):,} rows")
    return df


def _add_training_features(
    df: pd.DataFrame,
    hc: pd.DataFrame,
    wc: pd.DataFrame,
) -> pd.DataFrame:
    """調教特徴量を df に追加して返す。

    カテゴリA: 絶対値系（最近接・最速・セッション数）
    カテゴリB: 同レース内相対比較（rank / zscore）
    カテゴリC: 過去走との差分（shift(1)）
    """
    # ketto_num を int64 に統一（SE parquet では object の場合がある）
    # merge_asof の by キーおよび後続の merge キーで dtype 一致が必要
    df["ketto_num"] = pd.to_numeric(df["ketto_num"], errors="coerce").astype(np.int64)
    keys = df[["race_id", "ketto_num", "race_date"]].copy()
    active_horses = set(keys["ketto_num"].unique())

    # ─── カテゴリA: HC 系 ──────────────────────────────────────────────────────
    if len(hc) > 0:
        hc_f = hc[hc["ketto_num"].isin(active_horses)].copy()
        # merge_asof は right_on キー（training_date）がグローバルソートされている必要がある
        hc_f = hc_f.sort_values("training_date").reset_index(drop=True)
        keys_sorted = keys.sort_values("race_date").reset_index(drop=True)

        # 最近接セッション (merge_asof: training_date < race_date かつ 14日以内)
        last_hc = pd.merge_asof(
            keys_sorted,
            hc_f[["ketto_num", "training_date", "hc_3f_sec", "hc_4f_sec", "hc_200_sec"]],
            left_on="race_date",
            right_on="training_date",
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=14),
        ).rename(columns={
            "hc_3f_sec": "trn_hc_last_3f_sec",
            "hc_4f_sec": "trn_hc_last_4f_sec",
            "hc_200_sec": "trn_hc_last_200_sec",
        }).drop(columns=["training_date"])

        # 14日ウィンドウ集計（最速タイム・セッション数）
        merged_hc = keys.merge(
            hc_f[["ketto_num", "training_date", "hc_3f_sec", "hc_200_sec"]],
            on="ketto_num", how="left"
        )
        diff_days = (merged_hc["race_date"] - merged_hc["training_date"]).dt.days
        win_hc = merged_hc[(diff_days > 0) & (diff_days <= 14)].copy()
        hc_agg = win_hc.groupby(["race_id", "ketto_num"]).agg(
            trn_hc_best_3f_14d=("hc_3f_sec", "min"),
            trn_hc_best_200_14d=("hc_200_sec", "min"),
            trn_hc_count_14d=("training_date", "count"),
        ).reset_index()

        df = df.merge(
            last_hc[["race_id", "ketto_num", "trn_hc_last_3f_sec", "trn_hc_last_4f_sec", "trn_hc_last_200_sec"]],
            on=["race_id", "ketto_num"], how="left"
        )
        df = df.merge(hc_agg, on=["race_id", "ketto_num"], how="left")
    else:
        for col in ["trn_hc_last_3f_sec", "trn_hc_last_4f_sec", "trn_hc_last_200_sec",
                    "trn_hc_best_3f_14d", "trn_hc_best_200_14d", "trn_hc_count_14d"]:
            df[col] = np.nan

    # ─── カテゴリA: WC 系 ──────────────────────────────────────────────────────
    if len(wc) > 0:
        wc_f = wc[wc["ketto_num"].isin(active_horses)].copy()
        # merge_asof は right_on キー（training_date）がグローバルソートされている必要がある
        wc_f = wc_f.sort_values("training_date").reset_index(drop=True)
        keys_sorted = keys.sort_values("race_date").reset_index(drop=True)

        last_wc = pd.merge_asof(
            keys_sorted,
            wc_f[["ketto_num", "training_date", "wc_3f_sec", "wc_4f_sec", "wc_1f_sec"]],
            left_on="race_date",
            right_on="training_date",
            by="ketto_num",
            direction="backward",
            tolerance=pd.Timedelta(days=14),
        ).rename(columns={
            "wc_3f_sec": "trn_wc_last_3f_sec",
            "wc_4f_sec": "trn_wc_last_4f_sec",
            "wc_1f_sec": "trn_wc_last_1f_sec",
        }).drop(columns=["training_date"])

        merged_wc = keys.merge(
            wc_f[["ketto_num", "training_date", "wc_3f_sec", "wc_1f_sec"]],
            on="ketto_num", how="left"
        )
        diff_days = (merged_wc["race_date"] - merged_wc["training_date"]).dt.days
        win_wc = merged_wc[(diff_days > 0) & (diff_days <= 14)].copy()
        wc_agg = win_wc.groupby(["race_id", "ketto_num"]).agg(
            trn_wc_best_3f_14d=("wc_3f_sec", "min"),
            trn_wc_best_1f_14d=("wc_1f_sec", "min"),
            trn_wc_count_14d=("training_date", "count"),
        ).reset_index()

        df = df.merge(
            last_wc[["race_id", "ketto_num", "trn_wc_last_3f_sec", "trn_wc_last_4f_sec", "trn_wc_last_1f_sec"]],
            on=["race_id", "ketto_num"], how="left"
        )
        df = df.merge(wc_agg, on=["race_id", "ketto_num"], how="left")
    else:
        for col in ["trn_wc_last_3f_sec", "trn_wc_last_4f_sec", "trn_wc_last_1f_sec",
                    "trn_wc_best_3f_14d", "trn_wc_best_1f_14d", "trn_wc_count_14d"]:
            df[col] = np.nan

    # 合計セッション数（HC + WC）
    hc_cnt = df["trn_hc_count_14d"] if "trn_hc_count_14d" in df.columns else pd.Series(np.nan, index=df.index)
    wc_cnt = df["trn_wc_count_14d"] if "trn_wc_count_14d" in df.columns else pd.Series(np.nan, index=df.index)
    df["trn_total_count_14d"] = hc_cnt.fillna(0) + wc_cnt.fillna(0)
    # 両方NaNの場合はNaNに戻す
    both_nan = df["trn_hc_count_14d"].isna() & df["trn_wc_count_14d"].isna()
    df.loc[both_nan, "trn_total_count_14d"] = np.nan

    # ─── カテゴリB: 同レース内相対比較 ───────────────────────────────────────────

    def zscore_within_race(s: pd.Series) -> pd.Series:
        # 全馬NaNの場合はNaNを返す（0.0への暗黙補完を防ぐ）
        if s.isna().all():
            return pd.Series(np.nan, index=s.index)
        mean = s.mean()
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mean) / std

    df["trn_hc_rank_3f"] = df.groupby("race_id")["trn_hc_best_3f_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_hc_rank_200"] = df.groupby("race_id")["trn_hc_best_200_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_hc_zscore_3f"] = df.groupby("race_id")["trn_hc_best_3f_14d"].transform(zscore_within_race)

    df["trn_wc_rank_3f"] = df.groupby("race_id")["trn_wc_best_3f_14d"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["trn_wc_zscore_3f"] = df.groupby("race_id")["trn_wc_best_3f_14d"].transform(zscore_within_race)

    # ─── カテゴリC: 過去走との差分 (shift(1)) ─────────────────────────────────
    df = df.sort_values(["ketto_num", "race_date"]).reset_index(drop=True)
    grp = df.groupby("ketto_num")

    df["trn_hc_3f_delta"]  = df["trn_hc_best_3f_14d"]  - grp["trn_hc_best_3f_14d"].shift(1)
    df["trn_hc_200_delta"] = df["trn_hc_best_200_14d"] - grp["trn_hc_best_200_14d"].shift(1)
    df["trn_wc_3f_delta"]  = df["trn_wc_best_3f_14d"]  - grp["trn_wc_best_3f_14d"].shift(1)
    df["trn_count_delta"]  = df["trn_total_count_14d"]  - grp["trn_total_count_14d"].shift(1)

    return df


def _build_hc_norm_features(df: pd.DataFrame, hc: pd.DataFrame) -> pd.DataFrame:
    """坂路 基準時計差 特徴量（v50_hc_norm 専用）。

    生の坂路タイムは調教場・日ごとの馬場差を含むため、同日×同調教場の
    全馬 median を基準にした相対値（基準時計差）に正規化する
    （src/training.py の 坂路スピード/瞬発力 と同じ発想）。
    集約は expanding（キャリア全期間）で、merge_asof の
    allow_exact_matches=False により training_date < race_date を厳密に保証する。
    """
    new_cols = ["trn_hc_basediff_recent5", "trn_hc_basediff_best", "trn_hc_accel_best"]
    if len(hc) == 0:
        for col in new_cols:
            df[col] = np.nan
        return df

    hc = hc.copy()
    baseline = hc.groupby(["training_date", "training_center"])["hc_4f_sec"].transform("median")
    hc["basediff"] = hc["hc_4f_sec"] - baseline

    hc = hc.sort_values(["ketto_num", "training_date"], kind="stable").reset_index(drop=True)
    g = hc.groupby("ketto_num")
    hc["trn_hc_basediff_best"] = g["basediff"].cummin()
    hc["trn_hc_basediff_recent5"] = (
        g["basediff"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    # cummax は NaN 行で NaN を返すため、馬内 ffill で直前までのベストを引き継ぐ
    hc["trn_hc_accel_best"] = g["hc_accel_sec"].cummax()
    hc["trn_hc_accel_best"] = hc.groupby("ketto_num")["trn_hc_accel_best"].ffill()

    hc_sorted = hc.sort_values("training_date", kind="stable").reset_index(drop=True)
    keys = (
        df[["race_id", "ketto_num", "race_date"]]
        .sort_values("race_date", kind="stable")
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        keys,
        hc_sorted[["ketto_num", "training_date"] + new_cols],
        left_on="race_date",
        right_on="training_date",
        by="ketto_num",
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.merge(
        merged[["race_id", "ketto_num"] + new_cols],
        on=["race_id", "ketto_num"], how="left",
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ラベル生成
# ═══════════════════════════════════════════════════════════════════════════════

def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """LambdaRank / Binary 用ラベルを生成する。

    label_gain（7エントリ。具体値は train_config.json の model.label_gain を参照）
    の制約上、ラベルは 0〜6 の範囲に収める必要がある。

    着順 → ラベル対応（gain 値は config の label_gain[label] が適用される）:
        1着 → 6 (最高)
        2着 → 5
        3着 → 4
        4着 → 3
        5着 → 2
        6着 → 1
        7着以下 → 0 (全て同等)

    公式: label = clip(7 - finish_rank, 0, 6)
    頭数に依存せず、着順のみで決まる絶対ラベル方式を採用する。
    """
    # clip(lower=0) で 7着以下は全て 0
    df["lr_label"] = (7 - df["finish_rank"]).clip(lower=0, upper=6).astype(np.int8)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: NaN 率レポート
# ═══════════════════════════════════════════════════════════════════════════════

def _report_nan_rates(df: pd.DataFrame, threshold: float = 0.3) -> None:
    """NaN 率が高い特徴量を報告する。"""
    n = len(df)
    nan_rates = (df.isnull().sum() / n).sort_values(ascending=False)
    high_nan = nan_rates[nan_rates > threshold]
    if len(high_nan) > 0:
        print("\n  [警告] NaN 率が高い列（新馬・初コースは許容）:")
        for col, rate in high_nan.items():
            print(f"    {col}: {rate:.1%}")
    else:
        print(f"\n  NaN 率 > {threshold:.0%} の列なし")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8.5: 相関ゲート（新規特徴量の採用前チェック）
# 仕様書 docs/specs/2026-07-03-d1-course-features-design.md セクション2.2・7-4:
# - 学習+バリデーション期間（race_date <= valid_end）の行のみで計算（2025+ は使わない）
# - 新列 × 既存特徴量列の Pearson |r| >= 0.7 が1つでもあれば学習に進まない
# - 新列とレース内 is_win 相関（リーク簡易チェック。目安 |r| < 0.15）
# evaluator 申し送り: 実測値を必ず生成ログに数値として出力する（コメント記載のみは不可）
# ═══════════════════════════════════════════════════════════════════════════════

# 相関ゲート対象の新規列（features_version ごとに定義。
# v40_waku は実験2の追加列。v39_course_slim では front_pref_x_small が対象
# （採用済みだが再生成時のチェックとして残す））
NEW_FEATURE_COLS_BY_VERSION: dict[str, list[str]] = {
    "v39_course_slim": ["front_pref_x_small"],
    "v40_waku": ["hist_waku_course_bias_ts"],
    "v41_pace": ["front_pref_x_pace"],
    "v42_mining": ["mining_pred_rank"],
    "v43_sire_tc": ["hist_sire_track_condition_win_rate_ts"],
    "v44_handicap": ["is_handicap_race"],
    "v45_transport": ["transport_category"],
    "v46_small_pool": ["hist_small_course_pool_win_rate_ts"],
    "v47_pace_x": ["front_pref_x_density"],
    "v47_post_small": ["post_position_x_small"],
    "v48_agari_turn": ["hist_last_agari_time_gap", "hist_turn_surface_win_edge"],
    "v49_six_lap": [
        "hist_competitive_spirit",
        "hist_explosive_agari_gap",
        "hist_tracking_power",
        "hist_pref_lap_balance",
        "hist_dash_ten3f_centered",
        "hist_stamina_agari_gap",
    ],
    "v50_hc_norm": [
        "trn_hc_basediff_recent5",
        "trn_hc_basediff_best",
        "trn_hc_accel_best",
    ],
    "v51_cushion": [
        "cushion_value",
        "moisture_pct",
        "cushion_diff_track_avg",
        "last3f_rank_x_cushion",
    ],
}

CORR_GATE_THRESHOLD: float = 0.7      # |r| >= 0.7 → 学習に進まず planner へ報告
INRACE_ISWIN_GUIDE: float = 0.15      # レース内 is_win 相関の目安（超過は警告）


def _run_correlation_gate(df: pd.DataFrame, cfg: dict, new_cols: list[str]) -> None:
    """新規特徴量の相関ゲートを実測し、結果をログに出力する。

    ゲート違反（既存列と |r| >= 0.7）があれば RuntimeError を送出し、
    parquet を保存せずに終了する（学習に進ませないため）。
    """
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    sub = df[df["race_date"] <= valid_end]
    print(f"  対象期間: race_date <= {valid_end.date()} ({len(sub):,} rows)")

    feature_cols = get_feature_cols(df, cfg)
    violations: list[tuple[str, str, float]] = []

    for nc in new_cols:
        others = [c for c in feature_cols if c != nc]
        # category dtype 等が混じっても Pearson を計算できるよう数値列に限定する
        others_num = sub[others].select_dtypes(include=[np.number])
        corr = others_num.corrwith(sub[nc].astype(float)).dropna()
        corr_abs = corr.abs().sort_values(ascending=False)

        print(f"\n  [{nc}] vs 既存 {len(corr)} 列の Pearson 相関（上位5件）:")
        for col in corr_abs.head(5).index:
            print(f"    r = {corr[col]:+.4f}  {col}")
        over = corr_abs[corr_abs >= CORR_GATE_THRESHOLD]
        if len(over) > 0:
            for col in over.index:
                violations.append((nc, col, float(corr[col])))
        print(f"    → max|r| = {corr_abs.iloc[0]:.4f} "
              f"({'NG: >= ' if corr_abs.iloc[0] >= CORR_GATE_THRESHOLD else 'OK: < '}"
              f"{CORR_GATE_THRESHOLD})")

        # レース内 is_win 相関（レース内で demean した within 相関）
        s = sub[["race_id", "is_win", nc]].dropna(subset=[nc])
        if len(s) > 0:
            d_feat = s[nc] - s.groupby("race_id")[nc].transform("mean")
            d_win = s["is_win"] - s.groupby("race_id")["is_win"].transform("mean")
            r_inrace = float(d_feat.corr(d_win))
        else:
            r_inrace = float("nan")
        status = "OK" if abs(r_inrace) < INRACE_ISWIN_GUIDE else "警告: 目安超過"
        print(f"    レース内 is_win 相関: r = {r_inrace:+.4f} "
              f"({status}, 目安 |r| < {INRACE_ISWIN_GUIDE})")

    if violations:
        lines = "\n".join(
            f"  {nc} vs {col}: r = {r:+.4f}" for nc, col, r in violations
        )
        raise RuntimeError(
            f"[相関ゲート NG] |r| >= {CORR_GATE_THRESHOLD} の既存列があります。"
            f"学習に進まず planner へ報告してください:\n{lines}"
        )
    print("\n  相関ゲート: 全列 PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: マニフェスト保存
# ═══════════════════════════════════════════════════════════════════════════════

def _save_manifest(df: pd.DataFrame, out_dir: Path, version: str) -> None:
    """特徴量ファイルのメタ情報を manifest_{version}.json として保存する。

    固定名 manifest.json への上書きは廃止した（実験版の生成で本番メタ情報が
    消える事故を防ぐため）。本番のメタ情報を参照するときは
    train_config.json の features_version に対応する manifest_{version}.json を読むこと。
    """
    n = len(df)
    manifest = {
        "version": version,
        "rows": n,
        "cols": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "min": str(df["race_date"].min().date()),
            "max": str(df["race_date"].max().date()),
        },
        "race_count": df["race_id"].nunique(),
        "nan_rates": {
            col: float(f"{df[col].isnull().mean():.4f}")
            for col in df.columns
            if df[col].isnull().any()
        },
    }
    if version == "v41_pace" and "c" in PACE_CENTER_CONST:
        manifest["pace_center_const"] = {
            "c": PACE_CENTER_CONST["c"],
            "definition": (
                "front_pref_x_pace のセンタリング定数。学習+バリデーション期間"
                "（race_date <= training.valid_end）の自馬除外先行密度の平均。"
                "テスト期間（2025+）は含まない。"
            ),
        }
    if version == "v47_pace_x" and "c" in DENSITY_CENTER_CONST:
        manifest["density_center_const"] = {
            "c": DENSITY_CENTER_CONST["c"],
            "definition": (
                "front_pref_x_density のセンタリング定数。学習+バリデーション期間"
                "（race_date <= training.valid_end）の field_front_runner_density 平均。"
            ),
        }
    manifest_path = out_dir / f"manifest_{version}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  manifest saved: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: メイン
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = load_config()
    version = cfg["data"]["features_version"]
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    out_path = feat_dir / f"features_{version}.parquet"

    # 既存ファイルのバックアップ（上書き前に必ず実行）
    if out_path.exists():
        bk_path = out_path.with_suffix(f".bak.parquet")
        shutil.copy2(out_path, bk_path)
        print(f"[backup] {out_path.name} → {bk_path.name}")

    print("\n[1] Loading preprocessed data...")
    df = _load_data(cfg)

    # mining_predicted_rank は v42_mining（Phase 6）専用の生列。
    # 他バージョンでは既存の列数アサートに影響させないため早期に drop する。
    if version != "v42_mining" and "mining_predicted_rank" in df.columns:
        df = df.drop(columns=["mining_predicted_rank"])

    print("\n[2] Applying filters...")
    df = _apply_filters(df, cfg)

    # 市場情報混入チェック
    _check_no_market_features(df)

    # merge によるファンアウト（キー重複での行数増加）検出用
    n_after_filter = len(df)

    print("\n[2.5] Building course geometry intermediate (v39_course_slim)...")
    df = _build_course_geometry_features(df)

    print("\n[3] Building historical features (shift-1 leak prevention)...")
    df = _build_hist_features(df)

    # v48_agari_turn の2列は _build_hist_features 内で無条件生成される（未ゲート）。
    # v48/v49 以外のバージョンでは列セットを v39 基準に保つため drop する
    # （v42_mining の mining_predicted_rank と同じ扱い）。
    if version not in ("v48_agari_turn", "v49_six_lap"):
        _v48_cols = [c for c in ("hist_last_agari_time_gap", "hist_turn_surface_win_edge") if c in df.columns]
        if _v48_cols:
            df = df.drop(columns=_v48_cols)

    if version == "v49_six_lap":
        print("\n[3.5] Building six-parameter hist features (v49_six_lap)...")
        df = _build_six_param_hist_features(df, cfg)

    print("\n[4] Building current race features...")
    df = _build_current_features(df, version)

    # v40_waku（実験2）は Top-1 ゲート不通過で不採用（30.14% <= 30.24%）。
    # 実験記録として実装は残し、features_version = v40_waku のときのみ生成する
    # （現行採用バージョン v39_course_slim の再生成では 132列を維持するため）。
    if version == "v40_waku":
        print("\n[4.5] Building waku course bias features (v40_waku)...")
        df = _build_waku_bias_features(df)

    # v42_mining（Phase 6実験1）: JRA公式マイニング予想順位を1列だけ追加。
    # features_version = v42_mining のときのみ生成する（v39_course_slim の132列を維持するため）。
    if version == "v42_mining":
        print("\n[4.6] Building mining prediction features (v42_mining)...")
        df = _build_mining_features(df)

    print("\n[5] Building bloodline features...")
    df = _build_sire_features(df, version)

    print("\n[5.5] Building jockey/trainer features...")
    df = _build_jockey_trainer_features(df)

    print("\n[5.6] Building speed index features (hist_speed_idx_*)...")
    df = _build_speed_index_features(df)

    print("\n[5.7] Building relative features (field_z_*, pace index)...")
    df = _build_relative_features(df)

    # v41_pace（実験3・D-1 最終実験）front_pref 系が揃った段階（hist_front_running_pref
    # 生成済み・relative_features 完了後）で計算する。他バージョン再生成時は生成しない。
    if version == "v41_pace":
        print("\n[5.75] Building pace interaction features (v41_pace)...")
        df = _build_pace_interaction_features(df, cfg)

    if version == "v47_pace_x":
        print("\n[5.77] Building front_pref_x_density (v47_pace_x)...")
        df = _build_front_pref_x_density(df, cfg)

    if version == "v47_post_small":
        print("\n[5.78] Building post_position_x_small (v47_post_small)...")
        df = _build_post_position_x_small(df)

    if version == "v46_small_pool":
        print("\n[5.76] Building small course pool win rate (v46_small_pool)...")
        df = _build_small_course_pool_features(df)

    if version == "v51_cushion":
        print("\n[5.79] Building cushion/moisture features (v51_cushion)...")
        df = _build_cushion_features(df, cfg)

    print("\n[6] Building training features (HC/WC)...")
    hc = _load_hc(cfg)
    wc = _load_wc(cfg)
    df = _add_training_features(df, hc, wc)

    if version == "v50_hc_norm":
        print("\n[6.5] Building HC baseline-diff features (v50_hc_norm)...")
        df = _build_hc_norm_features(df, hc)

    print("\n[7] Building labels (lr_label)...")
    df = _build_labels(df)

    # merge のキー重複による行数増加（ファンアウト）がないことを保証
    assert len(df) == n_after_filter, (
        f"行数が merge で変化しています: filter後 {n_after_filter:,} → 現在 {len(df):,}"
    )

    # 最終的な市場情報混入チェック
    _check_no_market_features(df)

    # NaN 率レポート
    print("\n[8] NaN rate report:")
    _report_nan_rates(df, threshold=0.3)

    # 相関ゲート（|r| >= 0.7 で RuntimeError → 保存せず終了）
    print("\n[8.2] Correlation gate for new features:")
    _run_correlation_gate(df, cfg, NEW_FEATURE_COLS_BY_VERSION.get(version, []))

    # 最終列数の保証（バージョン別）。
    # 中間変数（course_is_small）の drop 漏れ・course_straight_len の混入も検出する
    if version == "v39_course_slim":
        assert len(df.columns) == 132, (
            f"v39_course_slim は 132 列（131 + front_pref_x_small）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 132")
    elif version == "v40_waku":
        assert len(df.columns) == 133, (
            f"v40_waku は 133 列（132 + hist_waku_course_bias_ts）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "hist_waku_course_bias_ts" in df.columns, "hist_waku_course_bias_ts がありません"
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v41_pace":
        assert len(df.columns) == 133, (
            f"v41_pace は 133 列（132 + front_pref_x_pace）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "front_pref_x_pace" in df.columns, "front_pref_x_pace がありません"
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v42_mining":
        assert len(df.columns) == 133, (
            f"v42_mining は 133 列（132 + mining_pred_rank）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "mining_pred_rank" in df.columns, "mining_pred_rank がありません"
        assert "mining_predicted_rank" not in df.columns, (
            "生の元列 mining_predicted_rank が残っています（mining_pred_rank に一本化すること）"
        )
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "front_pref_x_pace" not in df.columns, (
            "front_pref_x_pace は v41_pace 専用です（1変更1実験の原則）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v43_sire_tc":
        assert len(df.columns) == 133, (
            f"v43_sire_tc は 133 列（132 + hist_sire_track_condition_win_rate_ts）"
            f"のはずですが {len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "hist_sire_track_condition_win_rate_ts" in df.columns, (
            "hist_sire_track_condition_win_rate_ts がありません"
        )
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "front_pref_x_pace" not in df.columns, (
            "front_pref_x_pace は v41_pace 専用です（1変更1実験の原則）"
        )
        assert "mining_pred_rank" not in df.columns, (
            "mining_pred_rank は v42_mining 専用です（1変更1実験の原則）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v44_handicap":
        assert len(df.columns) == 133, (
            f"v44_handicap は 133 列（132 + is_handicap_race）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "is_handicap_race" in df.columns, "is_handicap_race がありません"
        assert "weight_type" not in df.columns, (
            "生の元列 weight_type が残っています（is_handicap_race に一本化すること。"
            "common.py の FORBIDDEN_COLS でメタ列除外対象）"
        )
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "front_pref_x_pace" not in df.columns, (
            "front_pref_x_pace は v41_pace 専用です（1変更1実験の原則）"
        )
        assert "mining_pred_rank" not in df.columns, (
            "mining_pred_rank は v42_mining 専用です（1変更1実験の原則）"
        )
        assert "hist_sire_track_condition_win_rate_ts" not in df.columns, (
            "hist_sire_track_condition_win_rate_ts は v43_sire_tc 専用です"
            "（相関ゲートNGで不採用。1変更1実験の原則）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v45_transport":
        assert len(df.columns) == 133, (
            f"v45_transport は 133 列（132 + transport_category）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "transport_category" in df.columns, "transport_category がありません"
        assert "region_code" not in df.columns, (
            "生の元列 region_code が残っています（transport_category に一本化すること。"
            "common.py の FORBIDDEN_COLS でメタ列除外対象）"
        )
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "hist_waku_course_bias_ts" not in df.columns, (
            "hist_waku_course_bias_ts は v40_waku 専用です（実験2は不採用）"
        )
        assert "front_pref_x_pace" not in df.columns, (
            "front_pref_x_pace は v41_pace 専用です（1変更1実験の原則）"
        )
        assert "mining_pred_rank" not in df.columns, (
            "mining_pred_rank は v42_mining 専用です（1変更1実験の原則）"
        )
        assert "hist_sire_track_condition_win_rate_ts" not in df.columns, (
            "hist_sire_track_condition_win_rate_ts は v43_sire_tc 専用です"
            "（相関ゲートNGで不採用。1変更1実験の原則）"
        )
        assert "is_handicap_race" not in df.columns, (
            "is_handicap_race は v44_handicap 専用です（dead weightで不採用。"
            "1変更1実験の原則）"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        assert "course_straight_len" not in df.columns, "course_straight_len は出力禁止です"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v46_small_pool":
        assert len(df.columns) == 133, (
            f"v46_small_pool は 133 列（132 + hist_small_course_pool_win_rate_ts）のはずですが "
            f"{len(df.columns)} 列あります"
        )
        assert "hist_small_course_pool_win_rate_ts" in df.columns
        assert "front_pref_x_small" in df.columns
        assert "front_pref_x_density" not in df.columns
        assert "front_pref_x_pace" not in df.columns
        assert "hist_waku_course_bias_ts" not in df.columns
        assert "transport_category" not in df.columns
        assert "is_handicap_race" not in df.columns
        assert "course_is_small" not in df.columns
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v47_pace_x":
        assert len(df.columns) == 133, (
            f"v47_pace_x は 133 列（132 + front_pref_x_density）のはずですが "
            f"{len(df.columns)} 列あります"
        )
        assert "front_pref_x_density" in df.columns
        assert "front_pref_x_small" in df.columns
        assert "hist_small_course_pool_win_rate_ts" not in df.columns
        assert "front_pref_x_pace" not in df.columns
        assert "course_is_small" not in df.columns
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v47_post_small":
        assert len(df.columns) == 133, (
            f"v47_post_small は 133 列（132 + post_position_x_small）のはずですが "
            f"{len(df.columns)} 列あります"
        )
        assert "post_position_x_small" in df.columns
        assert "front_pref_x_small" in df.columns
        assert "front_pref_x_density" not in df.columns
        assert "hist_small_course_pool_win_rate_ts" not in df.columns
        assert "course_is_small" not in df.columns
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 133")
    elif version == "v48_agari_turn":
        assert len(df.columns) == 134, (
            f"v48_agari_turn は 134 列（132 + hist_last_agari_time_gap + "
            f"hist_turn_surface_win_edge）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        assert "hist_last_agari_time_gap" in df.columns
        assert "hist_turn_surface_win_edge" in df.columns
        assert "front_pref_x_small" in df.columns, (
            "front_pref_x_small（v39_course_slim ベース）が失われています"
        )
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 134")
    elif version == "v49_six_lap":
        assert len(df.columns) == 140, (
            f"v49_six_lap は 140 列（134 + 6パラメータ hist）のはずですが "
            f"{len(df.columns)} 列あります"
        )
        for col in NEW_FEATURE_COLS_BY_VERSION["v49_six_lap"]:
            assert col in df.columns, f"{col} がありません"
        assert "hist_last_agari_time_gap" in df.columns
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 140")
    elif version == "v50_hc_norm":
        assert len(df.columns) == 135, (
            f"v50_hc_norm は 135 列（v39_course_slim の 132 + 新規3列）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        for col in NEW_FEATURE_COLS_BY_VERSION["v50_hc_norm"]:
            assert col in df.columns, f"{col} がありません"
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 135")
    elif version == "v51_cushion":
        assert len(df.columns) == 136, (
            f"v51_cushion は 136 列（v39_course_slim の 132 + 新規4列）のはずですが "
            f"{len(df.columns)} 列あります: 差分を確認してください"
        )
        for col in NEW_FEATURE_COLS_BY_VERSION["v51_cushion"]:
            assert col in df.columns, f"{col} がありません"
        assert "course_is_small" not in df.columns, "中間変数 course_is_small が残っています"
        print(f"\n[8.3] Column count assert PASS: {len(df.columns)} == 136")

    # 保存前に行順序を LambdaRank グループ割り当て用に修正する。
    # 中間処理では ketto_num 順（shift(1) 効率化）を使うが、
    # parquet の行順序は (race_date, race_id, horse_num) でなければならない。
    # get_group_sizes(sort=False) が正しいグループを返す前提がこれ。
    sort_cols = ["race_date", "race_id", "horse_num"]
    available_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(available_sort_cols).reset_index(drop=True)
    print(f"\n[8.5] Final row ordering: {available_sort_cols}")

    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n[9] Saved: {out_path}")
    print(f"  rows={len(df):,}, cols={len(df.columns)}")

    # 時系列の統計
    train_end = pd.Timestamp(cfg["training"]["train_end"])
    valid_end = pd.Timestamp(cfg["training"]["valid_end"])
    train = df[df["race_date"] <= train_end]
    valid = df[(df["race_date"] > train_end) & (df["race_date"] <= valid_end)]
    test  = df[df["race_date"] > valid_end]
    print(f"\n  Train: {len(train):,} rows, {train['race_id'].nunique():,} races "
          f"({train['race_date'].min().date()} - {train['race_date'].max().date()})")
    print(f"  Valid: {len(valid):,} rows, {valid['race_id'].nunique():,} races "
          f"({valid['race_date'].min().date()} - {valid['race_date'].max().date()})")
    print(f"  Test:  {len(test):,} rows, {test['race_id'].nunique():,} races "
          f"({test['race_date'].min().date()} - {test['race_date'].max().date()})")

    # マニフェスト保存
    _save_manifest(df, feat_dir, version)

    print("\n[10] create_features Done.")


if __name__ == "__main__":
    main()
