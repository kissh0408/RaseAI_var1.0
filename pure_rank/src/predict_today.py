"""
predict_today.py — 当日レース予測（特徴量生成・推論・出力）

仕様書: docs/specs/2026-07-04-today-prediction-design.md 4-D節・5章・6章。

create_features.py の内部関数（_build_hist_features 等）を import して再利用し、
学習時と全く同じ計算式で当日特徴量を生成する（コード重複禁止の原則）。

処理の骨子（4-D節）:
1. SE/RA/SK を _load_data() 相当のロジックでマージし、_apply_filters() を適用した
   過去走履歴 hist_df を用意する（当日行は含まない）
2. 当日行（today_adapter.build_today_merged() の出力）は track_condition_code=1
   （良）のプレースホルダで hist_df と concat する
   （シナリオ依存5列以外の127列はこの値に依存しないため何でもよいが、
   意図を明確にするため「良」に固定する）
3. create_features.py と同じ順序で _build_* 関数を通す（_build_labels は除く。
   finish_rank が NaN の当日行では clip().astype(int8) がエラーになり、
   かつ推論には不要なため）
4. 当日行のみを抽出し、シナリオ非依存列（127列）を確保する
5. track_condition_code ∈ {1,2,3,4} それぞれについて、
   - シナリオ依存の集計3列（hist_same_condition_win_rate /
     hist_surface_condition_win_rate / hist_best_time_same_cond）を
     hist_df から軽量に再集計する
   - track_condition_code / surface_condition 自体（モデルの直接入力特徴量）も
     実際のシナリオコードに明示的に上書きする（groupbyキーとしてではなく
     単純な値としてシナリオを表すため、仕様書2-2節の「3列」には含まれないが
     実装上はシナリオ依存として扱う必要がある。回帰テストで発見）
   4シナリオ分の132列 DataFrame（127 + 5 + lr_label プレースホルダ）を作る

回帰テスト（regression_test_today.py）で発見・修正した2つの実装上の落とし穴:
- ketto_num の dtype 統一: hist_df（SE_preprocessed 由来）は object/str、
  today 側は数値変換で int64 にしていたため、concat 後に列全体が object dtype に
  昇格し "123"(str) と 123(int) が別グループ扱いになって全 hist_* 特徴量が
  今日の行だけ NaN になっていた。concat 前に両者を int64 に統一して解決。
- 特徴量の列順序: lgb.Booster.predict(DataFrame) は列名ではなく列の「位置」で
  特徴量を解釈する（validate_features=False が既定）。build_today_features() が
  シナリオ依存列を末尾に並べ替えるため、get_feature_cols() の列順が学習時と
  ズレて特徴量がスクランブルされていた。run_today_predictions() で
  models[0].feature_name() の順序に明示的に reindex して解決。

禁止事項:
- オッズ・人気を特徴量に一切含めない
- market_log_odds / init_score を使わない
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import PROJECT_ROOT, get_feature_cols, load_config  # noqa: E402
from create_features import (  # noqa: E402  (既存ロジック再利用。コード重複禁止の原則)
    _add_training_features,
    _apply_filters,
    _build_course_geometry_features,
    _build_current_features,
    _build_hist_features,
    _build_jockey_trainer_features,
    _build_relative_features,
    _build_sire_features,
    _build_speed_index_features,
    _check_no_market_features,
    _load_data,
    _load_hc,
    _load_wc,
)
from today_adapter import build_today_merged  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# 定数（仕様書2-5節・2-6節）
# ═══════════════════════════════════════════════════════════════════════════════

# COURSE_CODE_TO_NAME / TRACK_CONDITION_LABELS は仕様書上 common.py への配置が
# 推奨されているが、implementer への指示「既存の common.py 等のロジックは
# 変更しない」を優先し、当日予測専用モジュールである本ファイルに定義する。
COURSE_CODE_TO_NAME: dict[int, str] = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}
TRACK_CONDITION_LABELS: dict[int, str] = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
TRACK_CONDITION_CODES: list[int] = [1, 2, 3, 4]

# track_condition_code に依存してレース内の集計対象が変わる3列（仕様書2-2節）。
# groupby キーとして track_condition_code/surface_condition を使うため、
# シナリオごとに軽量な groupby 再集計が必要な列。
_SCENARIO_AGGREGATE_COLS: list[str] = [
    "hist_same_condition_win_rate",
    "hist_surface_condition_win_rate",
    "hist_best_time_same_cond",
]

# track_condition_code / surface_condition 自体もモデルの直接入力特徴量
# （train_config.json の features.categorical に含まれる）であり、シナリオ
# そのものを表す値である。仕様書2-2節は「これらの値をgroupbyキーに使っている
# 列」を数えているため3列という記述になっているが、track_condition_code /
# surface_condition 自身も当然シナリオごとに異なる値を持つべきであり
# （そうでなければ4シナリオが同じ track_condition_code=1 のプレースホルダを
# 入力し続けることになり、シナリオ間で本質的に無意味な差しか生まれない）、
# 実装上は「シナリオ依存列」として扱い、共有パイプライン通過後に
# 実際のシナリオコードで明示的に上書きする（回帰テストで発見・修正）。
SCENARIO_DEPENDENT_COLS: list[str] = _SCENARIO_AGGREGATE_COLS + [
    "track_condition_code",
    "surface_condition",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 過去走履歴の読み込み（today行を含まない、学習時と同じフィルタ適用済み）
# ═══════════════════════════════════════════════════════════════════════════════

def _load_historical_base(cfg: dict) -> pd.DataFrame:
    """SE/RA/SK をマージ・フィルタ済みの過去走履歴 DataFrame を返す（create_features.main() の
    [1][2]相当。当日行は含まない）。"""
    version = cfg["data"]["features_version"]
    df = _load_data(cfg)
    if version != "v42_mining" and "mining_predicted_rank" in df.columns:
        df = df.drop(columns=["mining_predicted_rank"])
    df = _apply_filters(df, cfg)
    _check_no_market_features(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# シナリオ依存3列の軽量再計算
# ═══════════════════════════════════════════════════════════════════════════════

def _recompute_condition_dependent_cols(
    hist_df: pd.DataFrame,
    today_keys: pd.DataFrame,
    track_condition_code: int,
) -> pd.DataFrame:
    """
    hist_same_condition_win_rate / hist_surface_condition_win_rate /
    hist_best_time_same_cond の3列を、対象馬ごとに hist_df（フィルタ済み過去走）
    から直接集計する軽量関数。

    今日の行は必ず hist_df の全行より時系列的に後ろなので、"shift(1)" を明示的に
    再現する代わりに「対象条件に一致する過去走全体」を直接集計すればよい
    （create_features._build_hist_features の shift(1).expanding().mean()/min() は
    「そのレースの1つ前の時点までの累積」を意味し、今日は履歴の最後尾なので
    単純に全履歴を対象にするのと数学的に等価。重いパイプライン全体を4回
    再実行するのではなく、対象3列のみの軽量 groupby 集計として実装する）。

    Parameters
    ----------
    hist_df : _load_historical_base() の出力（フィルタ済み過去走。当日行を含まない）
    today_keys : 当日行の ["race_id", "ketto_num", "distance", "surface_code"] のみ
    track_condition_code : 1=良, 2=稍重, 3=重, 4=不良
    """
    h = hist_df.copy()
    h["ketto_num"] = pd.to_numeric(h["ketto_num"], errors="coerce").astype(np.int64)

    keys = today_keys.copy().reset_index(drop=True)
    keys["ketto_num"] = pd.to_numeric(keys["ketto_num"], errors="coerce").astype(np.int64)

    # ── hist_same_condition_win_rate: 馬 × track_condition_code=code の勝率 ──
    same_cond = h[h["track_condition_code"] == track_condition_code]
    win_rate_by_horse = same_cond.groupby("ketto_num")["is_win"].mean()
    keys["hist_same_condition_win_rate"] = keys["ketto_num"].map(win_rate_by_horse)

    # ── hist_surface_condition_win_rate: 馬 × surface_condition(=surface_code*10+code) の勝率 ──
    target_surface_condition = (keys["surface_code"].astype(int) * 10 + track_condition_code)
    win_rate_sc = h.groupby(["ketto_num", "surface_condition"])["is_win"].mean()
    keys["hist_surface_condition_win_rate"] = [
        win_rate_sc.get((k, sc), np.nan)
        for k, sc in zip(keys["ketto_num"], target_surface_condition)
    ]

    # ── hist_best_time_same_cond: 馬 × distance × surface_code × track_condition_code=code の自己ベストタイム ──
    best_time = (
        h[h["track_condition_code"] == track_condition_code]
        .groupby(["ketto_num", "distance", "surface_code"])["racetime"]
        .min()
    )
    keys["hist_best_time_same_cond"] = [
        best_time.get((k, d, s), np.nan)
        for k, d, s in zip(keys["ketto_num"], keys["distance"], keys["surface_code"])
    ]

    return keys[
        [
            "race_id", "ketto_num",
            "hist_same_condition_win_rate",
            "hist_surface_condition_win_rate",
            "hist_best_time_same_cond",
        ]
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 特徴量生成メイン（4-D節）
# ═══════════════════════════════════════════════════════════════════════════════

def build_today_features(today_merged: pd.DataFrame, cfg: dict) -> dict[int, pd.DataFrame]:
    """
    当日の出走馬データ（today_adapter.build_today_merged() の出力）を
    132列特徴量パイプラインに通し、track_condition_code シナリオ（1〜4）ごとの
    DataFrame を返す。

    Returns
    -------
    dict[int, pd.DataFrame]: {track_condition_code: 132列DataFrame(当日行のみ)}
    """
    version = cfg["data"]["features_version"]
    if version != "v39_course_slim":
        print(
            f"  [build_today_features][WARN] v39_course_slim 向けに実装されている関数です。"
            f"features_version={version} では列数アサート(132列)が失敗する可能性があります。"
        )

    hist_df = _load_historical_base(cfg)

    # ketto_num の dtype 統一（重要）: SE_preprocessed.parquet 由来の hist_df では
    # ketto_num が文字列 object dtype（例: "2012101800"）で保存されているのに対し、
    # today_adapter 側は数値変換で int64 にしている。dtype が食い違ったまま concat
    # すると列全体が object dtype に昇格し、文字列"123"と整数123が別グループとして
    # 扱われるため、当日行に対する groupby("ketto_num") ベースの hist_* 系特徴量が
    # 全て自分自身の過去走にマッチせず NaN になる（回帰テストで発見した実装バグ）。
    # concat 前に両側を int64 に統一することで、以後の全 _build_* 関数が正しく
    # 同一馬としてグルーピングできるようにする。
    hist_df["ketto_num"] = pd.to_numeric(hist_df["ketto_num"], errors="coerce").astype(np.int64)
    today_merged = today_merged.copy()
    today_merged["ketto_num"] = pd.to_numeric(today_merged["ketto_num"], errors="coerce").astype(np.int64)

    today_race_ids = set(today_merged["race_id"].unique())
    if not today_race_ids:
        raise ValueError("today_merged にレースが含まれていません。")

    # 当日のrace_idが誤って履歴データ側に含まれている場合は除外する。
    # 本番運用では当日レースはまだ SE/RA_preprocessed.parquet に反映されて
    # いないため通常 no-op だが、回帰テスト（既存の完了済みレースを「未実施」
    # として模擬する場合）やJV-Link更新タイミングのズレでは重複しうるため、
    # 二重カウント（同一 race_id が2行ずつ存在し groupby/shift が壊れる）を
    # 防ぐ防御的チェックとして必須。
    overlap = hist_df["race_id"].isin(today_race_ids)
    if overlap.any():
        print(
            f"  [build_today_features][WARN] hist_df に当日race_idが {int(overlap.sum())} 行"
            f" 含まれていたため除外します（回帰テスト等での意図的な重複の可能性）。"
        )
        hist_df = hist_df[~overlap].copy()

    # プレースホルダ track_condition_code=1（良）で結合する（4-D節手順2）
    today_placeholder = today_merged.copy()
    if version != "v42_mining" and "mining_predicted_rank" in today_placeholder.columns:
        today_placeholder = today_placeholder.drop(columns=["mining_predicted_rank"])
    today_placeholder["track_condition_code"] = np.int8(1)
    today_placeholder["surface_condition"] = (
        today_placeholder["surface_code"].astype(np.int64) * 10 + 1
    ).astype(np.int8)

    n_today = len(today_placeholder)
    combined = pd.concat([hist_df, today_placeholder], ignore_index=True, sort=False)
    print(
        f"  [build_today_features] Combined: {len(combined):,} rows "
        f"(hist={len(hist_df):,}, today={n_today:,})"
    )

    # create_features.py と同じ順序で _build_* を適用（_build_labels は除く。4-D節参照）
    combined = _build_course_geometry_features(combined)
    combined = _build_hist_features(combined)
    combined = _build_current_features(combined)
    combined = _build_sire_features(combined)
    combined = _build_jockey_trainer_features(combined)
    combined = _build_speed_index_features(combined)
    combined = _build_relative_features(combined)

    hc = _load_hc(cfg)
    wc = _load_wc(cfg)
    combined = _add_training_features(combined, hc, wc)

    is_today = combined["race_id"].isin(today_race_ids)
    today_out = combined[is_today].copy()
    if len(today_out) != n_today:
        raise RuntimeError(
            f"当日行数が変化しました: 入力 {n_today} → パイプライン後 {len(today_out)}\n"
            f"merge によるファンアウトの可能性があります。"
        )

    scenario_indep_cols = [c for c in today_out.columns if c not in SCENARIO_DEPENDENT_COLS]
    today_base = today_out[scenario_indep_cols].reset_index(drop=True)

    today_keys = today_placeholder[
        ["race_id", "ketto_num", "distance", "surface_code"]
    ].reset_index(drop=True)

    result: dict[int, pd.DataFrame] = {}
    reference_feature_cols: set[str] | None = None
    for code in TRACK_CONDITION_CODES:
        scenario_cols = _recompute_condition_dependent_cols(hist_df, today_keys, code)
        merged = today_base.merge(scenario_cols, on=["race_id", "ketto_num"], how="left")

        # track_condition_code / surface_condition をこのシナリオの実際の値に
        # 明示的に上書きする（today_base はプレースホルダ track_condition_code=1
        # のまま共有パイプラインを通過した値を保持しているため）。
        merged["track_condition_code"] = np.int8(code)
        merged["surface_condition"] = (
            merged["surface_code"].astype(np.int64) * 10 + code
        ).astype(np.int8)

        # lr_label はプレースホルダ（NaN）として付与する。132列アサートのため。
        # 推論には使わない（get_feature_cols() が id_cols として除外する）。
        merged["lr_label"] = np.nan

        assert len(merged.columns) == 132, (
            f"[track_condition_code={code}] 132列のはずですが {len(merged.columns)} 列です: "
            f"{sorted(merged.columns)}"
        )

        fc = set(get_feature_cols(merged, cfg))
        if reference_feature_cols is None:
            reference_feature_cols = fc
        elif fc != reference_feature_cols:
            raise RuntimeError(
                f"[track_condition_code={code}] 特徴量列集合が他シナリオと不一致: "
                f"diff={fc ^ reference_feature_cols}"
            )
        result[code] = merged

    # 4シナリオ間で3列以外（128列 + lr_label = 129列）が完全一致することを検証する
    # （仕様書6章 第2箇条）。
    first_code = TRACK_CONDITION_CODES[0]
    shared_cols = [c for c in result[first_code].columns if c not in SCENARIO_DEPENDENT_COLS]
    for code in TRACK_CONDITION_CODES[1:]:
        pd.testing.assert_frame_equal(
            result[first_code][shared_cols].reset_index(drop=True),
            result[code][shared_cols].reset_index(drop=True),
            check_dtype=False,
        )
    print(
        f"  [build_today_features] シナリオ間 {len(shared_cols)} 列（132 - "
        f"{len(SCENARIO_DEPENDENT_COLS)}）の完全一致を確認 (PASS)"
    )
    print(f"  [build_today_features] 特徴量列数={len(reference_feature_cols)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 推論（アンサンブル予測・順位変換）
# ═══════════════════════════════════════════════════════════════════════════════

def run_today_predictions(
    today_features: dict[int, pd.DataFrame],
    cfg: dict,
    models: list | None = None,
) -> dict[int, pd.DataFrame]:
    """各シナリオについてアンサンブル推論を行い、pred_score/pred_rank/pred_softmax_prob を付与する。

    6-5節: 出力の主軸は生スコア降順の順位（予想着順）。Softmax(T_opt) は参考列として付与する。
    """
    from evaluate import ensemble_predict, load_models
    from predict import softmax_with_temperature

    if models is None:
        models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
        models = load_models(models_dir)

    T_opt = float(cfg.get("plackett_luce", {}).get("T_opt", 1.0))

    # 重要: lgb.Booster.predict(DataFrame) は列名ではなく列の「位置」で特徴量を
    # 解釈する（validate_features=False が既定のため、名前と学習時の並び順が
    # 一致していることが前提）。build_today_features() で3列
    # （hist_same_condition_win_rate 等）と track_condition_code/surface_condition
    # を末尾に並べ替えているため、get_feature_cols(df, cfg) が返す列順は
    # 学習時（features_v39_course_slim.parquet の列順）とズレる。
    # 回帰テストでこのズレによる特徴量スクランブル（的外れな予測）を検出したため、
    # 必ずモデル自身が記憶している学習時の並び順（Booster.feature_name()）に
    # reindex してから predict する。15モデルは同一特徴量集合・同一順序で
    # 学習されているため models[0] の順序を代表として使う。
    model_feature_order = models[0].feature_name()

    result: dict[int, pd.DataFrame] = {}
    for code, df in today_features.items():
        feature_cols = get_feature_cols(df, cfg)
        if set(feature_cols) != set(model_feature_order):
            raise RuntimeError(
                f"[track_condition_code={code}] 特徴量列集合がモデルの学習時と不一致: "
                f"diff={set(feature_cols) ^ set(model_feature_order)}"
            )
        X = df[model_feature_order]
        preds = ensemble_predict(models, X)

        out = df.copy().reset_index(drop=True)
        out["pred_score"] = preds
        out["pred_rank"] = (
            out.groupby("race_id")["pred_score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        out["pred_softmax_prob"] = np.nan
        for _race_id, grp in out.groupby("race_id"):
            scores = grp["pred_score"].values
            probs = softmax_with_temperature(scores, T_opt)
            out.loc[grp.index, "pred_softmax_prob"] = probs

        result[code] = out

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 出力（フォルダ構造・CSV書き出し。仕様書5章）
# ═══════════════════════════════════════════════════════════════════════════════

def write_predictions(
    today_features_with_preds: dict[int, pd.DataFrame],
    out_root: Path,
) -> None:
    """
    main/predictions/{YYYYMMDD}/{開催場所名}/{良|稍重|重|不良}/
        race_{race_id}_pred.csv
        summary.csv
    の構造で予測結果を CSV 出力する（仕様書5章）。
    """
    required_cols = ["race_id", "race_num", "horse_num", "ketto_num", "pred_score", "pred_rank"]

    for code, df in today_features_with_preds.items():
        # 出力前の最終市場情報混入チェック（仕様書5章「出力前に必ず実行」）
        _check_no_market_features(df)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[write_predictions] 必須列が不足しています: {missing}")

        label = TRACK_CONDITION_LABELS.get(code, f"code{code}")
        date_str = pd.Timestamp(df["race_date"].iloc[0]).strftime("%Y%m%d")

        for course_code, grp_course in df.groupby("course_code"):
            course_name = COURSE_CODE_TO_NAME.get(int(course_code), f"course{course_code}")
            out_dir = out_root / date_str / course_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

            summary_rows = []
            for race_id, grp_race in grp_course.groupby("race_id"):
                grp_sorted = grp_race.sort_values("pred_rank")
                out_cols = [
                    c for c in required_cols + ["pred_softmax_prob"]
                    if c in grp_sorted.columns
                ]
                grp_sorted[out_cols].to_csv(
                    out_dir / f"race_{race_id}_pred.csv", index=False, encoding="utf-8-sig"
                )
                top1 = grp_sorted.iloc[0]
                summary_rows.append({
                    "race_id": race_id,
                    "race_num": int(top1["race_num"]),
                    "distance": top1.get("distance", np.nan),
                    "surface_code": top1.get("surface_code", np.nan),
                    "top1_horse_num": int(top1["horse_num"]),
                    "top1_pred_score": float(top1["pred_score"]),
                })
            pd.DataFrame(summary_rows).sort_values("race_num").to_csv(
                out_dir / "summary.csv", index=False, encoding="utf-8-sig"
            )
            print(f"  [write_predictions] {out_dir}: {len(summary_rows)} races")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI エントリポイント
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RaceAI_var1.0 当日レース予測")
    parser.add_argument("--race-day", type=str, default=None, help="YYYYMMDD（省略時は本日）")
    args = parser.parse_args()

    cfg = load_config()
    race_dir = PROJECT_ROOT / "main" / "data" / "race"
    preprocessed_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    print("[1] Loading & converting today's raw data...")
    today_merged = build_today_merged(race_dir, preprocessed_dir)

    print("\n[2] Building today features (4 track-condition scenarios)...")
    today_features = build_today_features(today_merged, cfg)

    print("\n[3] Running ensemble inference...")
    today_with_preds = run_today_predictions(today_features, cfg)

    print("\n[4] Writing predictions...")
    out_root = PROJECT_ROOT / "main" / "predictions"
    write_predictions(today_with_preds, out_root)

    print("\n[predict_today] Done.")


if __name__ == "__main__":
    main()
