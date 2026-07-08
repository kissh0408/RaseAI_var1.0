"""当日予測エントリーポイント。

このモジュールは薄いオーケストレーター。実装の実体は以下の pipeline サブモジュールにある:
  - main.pipeline.data_pipeline       … データ取得・前処理系
  - main.pipeline.inference_pipeline  … モデルロード・推論・スコアリング系
  - main.pipeline.strategy_pipeline   … 戦略・推奨生成・表示系
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# stdout 抑制コンテキスト（前処理・特徴量の tqdm ログを Notebook に出さないため）
# stderr は抑制しない（例外トレースバックが消えるのを防ぐため）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _silent_pipeline_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    devnull_w = open(os.devnull, "w", encoding="utf-8")
    old_out = sys.stdout
    sys.stdout = devnull_w
    try:
        yield
    finally:
        sys.stdout = old_out
        devnull_w.close()


def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if not (cand / "common").is_dir():
            continue
        if (cand / "main").is_dir() or (cand / "Main").is_dir():
            return cand
    raise RuntimeError(f"プロジェクトルートが見つかりません: {start}")


PROJECT_ROOT = find_project_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data.src.jv_subprocess import run_with_32bit_python  # noqa: E402

for mod_name in ["main.main", "main.notebook_bootstrap"]:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

try:
    from main.notebook_bootstrap import (
        update_jra_data,
        fetch_jra_data,
        fetch_race_only,
        fetch_hc_only,
        fetch_wc_only,
        fetch_se_only,
        fetch_sk_only,
        fetch_hn_only,
        fetch_bt_only,
        fetch_ra_only,
        get_race_data_32bit,
        run_today_se_ra_and_realtime,
    )

    logger.info("main.notebook_bootstrap から読み込みました（32bit委譲対応）: %s", PROJECT_ROOT)
except Exception as e:
    if "common.data.src.get_data" in sys.modules:
        importlib.reload(sys.modules["common.data.src.get_data"])
    from common.data.src.get_data import (
        update_jra_data,
        fetch_jra_data,
        fetch_race_only,
        fetch_hc_only,
        fetch_wc_only,
        fetch_se_only,
        fetch_sk_only,
        fetch_hn_only,
        fetch_bt_only,
        fetch_ra_only,
        run_today_se_ra_and_realtime_merge,
    )

    logger.warning("common.data.src.get_data から直接読み込みました（bootstrap失敗: %s）", e)

    def run_today_se_ra_and_realtime(**kwargs):
        """bootstrap 不可時は同一プロセス（Win64 の JV は使えない可能性あり）。"""
        return run_today_se_ra_and_realtime_merge(**kwargs)

    def get_race_data_32bit(**kwargs):
        """notebook_bootstrap 失敗時も 32bit 子プロセス経由で取得（実装は bootstrap と同じ）。"""
        supported = {
            "start_date_str",
            "end_date_str",
            "output_dir",
            "include_entry_kubun_1",
            "target_kubun",
            "race_day_yyyymmdd",
            "dual_pass_se_then_ra",
        }
        filtered = {k: v for k, v in kwargs.items() if k in supported}
        unsupported = set(kwargs.keys()) - supported
        if unsupported:
            logger.warning("以下の引数は無視されます: %s", ", ".join(sorted(unsupported)))
        args = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
        snippet = (
            f"from common.data.src.get_data import get_race_data; get_race_data({args})"
        )
        return run_with_32bit_python(PROJECT_ROOT, snippet, capture_output=True)


if "common.data.src.jv_pipeline" in sys.modules:
    importlib.reload(sys.modules["common.data.src.jv_pipeline"])
from common.data.src.jv_pipeline import (  # noqa: E402
    dispatch_update_jra_data,
    fetch_realtime_data,
    run_accumulation_update,
    run_mode,
    update_target_outputs_since,
)

from common.data.src.jv_run import (  # noqa: E402
    accumulation_update_since_with_report,
    default_state_log_paths,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 本番イレギュラー処理（取消除外）の純粋関数。循環 import を避けるため
# パッケージ(main)経由ではなくサブモジュールから直接 import する。
from main.race_runtime import filter_scratched  # noqa: E402

# ---------------------------------------------------------------------------
# pipeline サブモジュールから公開 API を re-export
# ---------------------------------------------------------------------------
from main.pipeline.data_pipeline import (  # noqa: E402
    update_accumulation_data as _update_accumulation_data_impl,
    run_today_ra_se_wh_weight_only as _run_today_ra_se_wh_weight_only_impl,
    load_pair_odds_dicts as _load_pair_odds_dicts,
)
from main.pipeline.inference_pipeline import (  # noqa: E402
    load_models as _load_models_impl,
    load_rank1_isotonic_calibrator as _load_rank1_isotonic_calibrator_impl,
    load_rank2_isotonic_calibrator as _load_rank2_isotonic_calibrator_impl,
    load_rank3_isotonic_calibrator as _load_rank3_isotonic_calibrator_impl,
    going_condition_in_model_features as _going_condition_in_model_features,
    predict_ranks_for_frame as _predict_ranks_for_frame,
    apply_uniform_baba_jv_code as _apply_uniform_baba_jv_code,
    win_prob_from_rank1 as _win_prob_from_rank1,
    bootstrap_encoders_via_create_features_main as _bootstrap_encoders,
    DIRT_TRACK_CODE_MIN,
)
from main.pipeline.strategy_pipeline import (  # noqa: E402
    load_strategy_runtime_config as _load_strategy_runtime_config_impl,
    strategy_config_from_runtime as _strategy_config_from_runtime,
    resolve_strategy_calibration_path,
    resolve_recommendation_mode as _resolve_recommendation_mode_impl,
    canonical_race_id_str,
    export_predictions_by_course_baba_race as _export_predictions_by_course_baba_race,
    run_today_strategy_pipeline as _run_today_strategy_pipeline_impl,
    run_today_score_rank_pipeline as _run_today_score_rank_pipeline_impl,
    apply_operating_pair_rule_filter as _apply_operating_pair_rule_filter_impl,
    format_strategy_view,
    format_predictions_export_view,
    display_top_win_prob_predictions,
    display_today_recommendation_summaries as _display_today_recommendation_summaries_impl,
    display_ticket_candidates_by_race,
)

# ---------------------------------------------------------------------------
# パス定数（本番出力先。変更厳禁）
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "model_training" / "models"
MAIN_FEATURES_PAST_PATH = (
    PROJECT_ROOT / "model_training" / "data" / "02_features" / "main_features_past.csv"
)
PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT / "main" / "results" / "today_predictions_with_bets.csv"
)
RECOMMENDATION_OUTPUT_PATH = (
    PROJECT_ROOT / "main" / "results" / "today_recommendations.csv"
)
STRATEGY_CONFIG_PATH = PROJECT_ROOT / "strategy" / "config" / "strategy_config.json"
STRATEGY_CALIBRATION_PATH = resolve_strategy_calibration_path(
    PROJECT_ROOT, STRATEGY_CONFIG_PATH
)
O2_ODDS_PATH = PROJECT_ROOT / "common" / "data" / "output" / "realtime_odds" / "o2_odds.csv"
O3_ODDS_PATH = PROJECT_ROOT / "common" / "data" / "output" / "realtime_odds" / "o3_odds.csv"

# JV 馬場状態（一般的に 1=良 / 2=稍重 / 3=重 / 4=不良）
BABA_SCENARIO_JV_CODES: tuple[int, ...] = (1, 2, 3, 4)
BABA_SCENARIO_LABEL_JA: dict[int, str] = {
    1: "良",
    2: "稍重",
    3: "重",
    4: "不良",
}


def _month_day_from_race_day(race_day_yyyymmdd: str | None) -> int | None:
    """YYYYMMDD → JV month_day 整数（例: 20260614 → 614）。"""
    if not race_day_yyyymmdd:
        return None
    digits = "".join(ch for ch in str(race_day_yyyymmdd).strip() if ch.isdigit())
    if len(digits) < 8:
        return None
    return int(digits[4:8])


# ---------------------------------------------------------------------------
# 公開関数（パス定数をバインドしたシンプルなラッパー）
# ---------------------------------------------------------------------------

def update_accumulation_data(
    last_updated_date: str = "20260322",
    *,
    end_date_str: str | None = None,
    output_dir: str | None = None,
    state_path: str | Path | None = None,
    log_path: str | Path | None = None,
    result_json_path: str | Path | None = None,
    incremental: bool = False,
) -> object:
    """
    ノート用ショートカット。実体は ``common.data.src.jv_run.accumulation_update_since_with_report``。
    """
    return _update_accumulation_data_impl(
        PROJECT_ROOT,
        last_updated_date,
        end_date_str=end_date_str,
        output_dir=output_dir,
        state_path=state_path,
        log_path=log_path,
        result_json_path=result_json_path,
        incremental=incremental,
    )


def run_today_ra_se_wh_weight_only(
    race_day_yyyymmdd: str | None = None,
    *,
    dual_pass_se_then_ra: bool = True,
    target_kubun: str = "both",
) -> object:
    """
    当日（または指定日）の RA/SE を保存し、馬体重(WH)のみ速報として SE に反映する。
    馬場(WE)は取得・RA 反映しない。JV は 32bit 子プロセスで実行される。
    """
    return _run_today_ra_se_wh_weight_only_impl(
        PROJECT_ROOT,
        race_day_yyyymmdd,
        dual_pass_se_then_ra=dual_pass_se_then_ra,
        target_kubun=target_kubun,
    )


def load_strategy_runtime_config() -> dict:
    return _load_strategy_runtime_config_impl(STRATEGY_CONFIG_PATH)


def resolve_recommendation_mode(
    runtime_cfg: dict | None = None,
    *,
    use_score_ranked_picks: bool | None = None,
) -> str:
    if runtime_cfg is None:
        runtime_cfg = load_strategy_runtime_config()
    return _resolve_recommendation_mode_impl(
        runtime_cfg,
        use_score_ranked_picks=use_score_ranked_picks,
    )


def load_models() -> dict[int, object] | dict[int, list[object]]:
    return _load_models_impl(MODELS_DIR)


def load_rank1_isotonic_calibrator() -> tuple[object | None, dict]:
    return _load_rank1_isotonic_calibrator_impl(MODELS_DIR, PROJECT_ROOT)


def load_rank2_isotonic_calibrator() -> tuple[object | None, dict]:
    return _load_rank2_isotonic_calibrator_impl(MODELS_DIR, PROJECT_ROOT)


def load_rank3_isotonic_calibrator() -> tuple[object | None, dict]:
    return _load_rank3_isotonic_calibrator_impl(MODELS_DIR, PROJECT_ROOT)


def run_today_prediction_pipeline(
    fetch_today: bool = False,
    race_day_yyyymmdd: str | None = None,
    *,
    prefer_parquet: bool | None = None,
    use_cudf: bool | None = None,
    reload_create_features_module: bool = False,
    baba_scenario_jv_codes: tuple[int, ...] | None = None,
    strategy_baba_scenario_jv_code: int | None = None,
    pipeline_verbose: bool = True,
    export_by_course_baba_race: bool = True,
) -> pd.DataFrame:
    """
    前処理〜過去特徴量〜モデル推論〜勝率・期待値付与。CSV と Parquet を Main/results に保存して DataFrame を返す。

    出馬表時点では馬場が確定しないため、baba_scenario_jv_codes（既定 1〜4）ごとに
    turf_condition / dirt_condition を上書きしてから推論する。
    """
    from strategy.src.betting_framework import load_today_prediction_frame
    from model_training.src.create_features import create_main_features
    from model_training.src.create_pastfeatures import create_main_pastfeatures
    from model_training.src.preprocessing import preprocess_main_data

    runtime_cfg = load_strategy_runtime_config()
    pv = bool(runtime_cfg.get("prefer_parquet", True))
    uc = bool(runtime_cfg.get("use_cudf", True))
    if prefer_parquet is not None:
        pv = bool(prefer_parquet)
    if use_cudf is not None:
        uc = bool(use_cudf)

    if fetch_today:
        _jv_result = run_today_se_ra_and_realtime(race_day_yyyymmdd=race_day_yyyymmdd)
        if _jv_result is not None and hasattr(_jv_result, "returncode") and _jv_result.returncode != 0:
            raise RuntimeError(
                f"JV-Link data fetch failed (returncode={_jv_result.returncode})"
            )

    with _silent_pipeline_stdout(enabled=not pipeline_verbose):
        preprocess_main_data()
        try:
            if reload_create_features_module:
                mod = importlib.import_module("model_training.src.create_features")
                mod = importlib.reload(mod)

            create_main_features()
        except KeyError as e:
            if "Encoder map for" not in str(e):
                raise
            logger.warning("学習用エンコーダが見つからないため、初回セットアップを実行します...")
            _bootstrap_encoders(PROJECT_ROOT)

        create_main_pastfeatures()

    main_features_parquet = MAIN_FEATURES_PAST_PATH.with_suffix(".parquet")
    if not MAIN_FEATURES_PAST_PATH.exists() and not main_features_parquet.exists():
        raise FileNotFoundError(
            f"特徴量ファイルが見つかりません: {MAIN_FEATURES_PAST_PATH}"
        )

    df = load_today_prediction_frame(
        csv_path=MAIN_FEATURES_PAST_PATH,
        parquet_path=main_features_parquet,
        prefer_parquet=pv,
        use_cudf=uc,
    )

    scenario_codes = (
        tuple(baba_scenario_jv_codes)
        if baba_scenario_jv_codes is not None
        else BABA_SCENARIO_JV_CODES
    )

    pri_eff = strategy_baba_scenario_jv_code
    if pri_eff is None:
        v = runtime_cfg.get("strategy_baba_scenario_jv_code", 1)
        pri_eff = int(v if v is not None else 1)

    models = load_models()
    rank1_iso = None
    if runtime_cfg.get("rank1_isotonic_at_inference", True):
        rank1_iso, _ = load_rank1_isotonic_calibrator()
    rank2_iso = None
    if runtime_cfg.get("rank2_isotonic_at_inference", True):
        rank2_iso, _ = load_rank2_isotonic_calibrator()
    rank3_iso = None
    if runtime_cfg.get("rank3_isotonic_at_inference", True):
        rank3_iso, _ = load_rank3_isotonic_calibrator()

    uses_going = _going_condition_in_model_features(models)
    if len(scenario_codes) > 1 and not uses_going:
        warnings.warn(
            "馬場シナリオごとに turf_condition / dirt_condition を差し替えていますが、"
            "現行モデルの入力特徴に馬場状態が含まれていないため、"
            "pred_rank*_baba* は馬場による差がつきません（同一予測の複製）。"
            "差を出したい場合は、turf_condition / dirt_condition を特徴に含めて再学習してください。",
            UserWarning,
            stacklevel=2,
        )

    base_odds_num: pd.Series | None = None
    if "odds" in df.columns:
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        base_odds_num = df["odds"]

    # 出走取消馬（スクラッチ）を除外する。JRA では odds=0 が取消の標準マーカー。
    # odds==NaN はオッズ未取得（レース前の取消）も検出する。
    # 除外ロジックは race_runtime.filter_scratched に集約済み（テスト可能な純粋関数）。
    if "odds" in df.columns:
        odds_num = df["odds"]
        scratch_mask = (odds_num == 0) | odds_num.isna()
        n_scratch = int(scratch_mask.sum())
        if n_scratch > 0:
            scratch_info = []
            for _, row in df.loc[scratch_mask].iterrows():
                rid = row.get("race_id", "?")
                name = row.get("horse_name", row.get("ketto_num", "?"))
                o = row.get("odds")
                reason = "odds=0" if o == 0 else "odds=NaN"
                scratch_info.append(f"{rid}/{name}({reason})")
            logger.warning(
                "出走取消: %d 頭を除外します -> %s%s",
                n_scratch,
                ", ".join(scratch_info[:10]),
                "..." if n_scratch > 10 else "",
            )
            df = filter_scratched(df)
            base_odds_num = df["odds"] if "odds" in df.columns else None

    if uses_going:
        for jv in scenario_codes:
            suffix = f"_baba{jv}"
            frame = _apply_uniform_baba_jv_code(df, jv)
            preds = _predict_ranks_for_frame(
                models, frame,
                rank1_isotonic=rank1_iso,
                rank2_isotonic=rank2_iso,
                rank3_isotonic=rank3_iso,
            )
            for rank, arr in preds.items():
                df[f"pred_rank{rank}{suffix}"] = arr

            r1_col = f"pred_rank1{suffix}"
            df[f"win_prob_est{suffix}"] = _win_prob_from_rank1(df, r1_col)

            if base_odds_num is not None:
                df[f"expected_return{suffix}"] = (
                    df[f"win_prob_est{suffix}"] * base_odds_num
                )
            else:
                df[f"expected_return{suffix}"] = np.nan
    else:
        preds_once = _predict_ranks_for_frame(
            models, df,
            rank1_isotonic=rank1_iso,
            rank2_isotonic=rank2_iso,
            rank3_isotonic=rank3_iso,
        )
        for jv in scenario_codes:
            suffix = f"_baba{jv}"
            for rank, arr in preds_once.items():
                df[f"pred_rank{rank}{suffix}"] = arr

            r1_col = f"pred_rank1{suffix}"
            df[f"win_prob_est{suffix}"] = _win_prob_from_rank1(df, r1_col)

            if base_odds_num is not None:
                df[f"expected_return{suffix}"] = (
                    df[f"win_prob_est{suffix}"] * base_odds_num
                )
            else:
                df[f"expected_return{suffix}"] = np.nan

    if pri_eff not in scenario_codes:
        raise ValueError(
            f"strategy_baba_scenario_jv_code={pri_eff} は "
            f"baba_scenario_jv_codes {scenario_codes!r} に含まれている必要があります"
        )

    pri = pri_eff
    if "race_id" in df.columns:
        df["race_id"] = canonical_race_id_str(df["race_id"])

    df["pred_rank1"] = df[f"pred_rank1_baba{pri}"]
    df["pred_rank2"] = df[f"pred_rank2_baba{pri}"]
    df["pred_rank3"] = df[f"pred_rank3_baba{pri}"]
    df["win_prob_est"] = df[f"win_prob_est_baba{pri}"]
    df["expected_return"] = df[f"expected_return_baba{pri}"]

    if uses_going:
        pri_frame = _apply_uniform_baba_jv_code(df, pri)
        for col in ("turf_condition", "dirt_condition", "track_condition_code"):
            if col in pri_frame.columns:
                df[col] = pri_frame[col]

    jp = BABA_SCENARIO_LABEL_JA.get(pri, str(pri))
    if pipeline_verbose:
        logger.info(
            "馬場シナリオ別推論: %s",
            ", ".join(f"{c}({BABA_SCENARIO_LABEL_JA.get(c, '?')})" for c in scenario_codes),
        )
        logger.info(
            "主列 pred_rank1 / win_prob_est は JVコード %d（%s）を戦略連携用に複製しました。",
            pri, jp,
        )

    now_jst = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    if "odds_timestamp" not in df.columns:
        df["odds_timestamp"] = now_jst
    df["generated_at"] = now_jst

    if "win_prob_est" in df.columns:
        df = df.sort_values(
            ["race_id", "win_prob_est"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            ["race_id", "pred_rank1"], ascending=[True, False], na_position="last"
        ).reset_index(drop=True)

    PREDICTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    df.to_parquet(PREDICTION_OUTPUT_PATH.with_suffix(".parquet"), index=False)

    logger.info("予測を保存しました: %s", PREDICTION_OUTPUT_PATH)
    if export_by_course_baba_race:
        split_root, n_files = _export_predictions_by_course_baba_race(
            df,
            BABA_SCENARIO_JV_CODES,
            BABA_SCENARIO_LABEL_JA,
            result_root=PREDICTION_OUTPUT_PATH.parent,
            export_month_day=_month_day_from_race_day(race_day_yyyymmdd),
        )
        logger.info("コース・馬場シナリオ別CSV: %s に %d ファイル", split_root, n_files)

    return df


def export_predictions_by_venue(
    pred_df: pd.DataFrame | None = None,
    *,
    race_day_yyyymmdd: str | None = None,
) -> tuple[Path, int]:
    """予測 DataFrame（未指定時は保存済み Parquet/CSV）を競馬場×馬場シナリオ別 CSV に再出力する。"""
    if pred_df is None:
        pq = PREDICTION_OUTPUT_PATH.with_suffix(".parquet")
        pred_df = (
            pd.read_parquet(pq)
            if pq.exists()
            else pd.read_csv(PREDICTION_OUTPUT_PATH)
        )
    return _export_predictions_by_course_baba_race(
        pred_df,
        BABA_SCENARIO_JV_CODES,
        BABA_SCENARIO_LABEL_JA,
        result_root=PREDICTION_OUTPUT_PATH.parent,
        export_month_day=_month_day_from_race_day(race_day_yyyymmdd),
    )


def run_today_strategy_pipeline(pred_df: pd.DataFrame) -> pd.DataFrame:
    return _run_today_strategy_pipeline_impl(
        pred_df,
        strategy_config_path=STRATEGY_CONFIG_PATH,
        strategy_calibration_path=STRATEGY_CALIBRATION_PATH,
        recommendation_output_path=RECOMMENDATION_OUTPUT_PATH,
        o2_odds_path=O2_ODDS_PATH,
        o3_odds_path=O3_ODDS_PATH,
    )


def run_today_score_rank_pipeline(
    pred_df: pd.DataFrame,
    *,
    score_col: str = "pred_rank1",
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    bet_unit: int = 100,
) -> pd.DataFrame:
    return _run_today_score_rank_pipeline_impl(
        pred_df,
        recommendation_output_path=RECOMMENDATION_OUTPUT_PATH,
        score_col=score_col,
        pair_top_n=pair_top_n,
        wide_top_n=wide_top_n,
        bet_unit=bet_unit,
    )


def apply_operating_pair_rule_filter(
    pred_df: pd.DataFrame,
    rec_df: pd.DataFrame,
) -> pd.DataFrame:
    return _apply_operating_pair_rule_filter_impl(
        pred_df,
        rec_df,
        recommendation_output_path=RECOMMENDATION_OUTPUT_PATH,
    )


def display_today_recommendation_summaries(rec_df: pd.DataFrame) -> None:
    _display_today_recommendation_summaries_impl(
        rec_df,
        prediction_output_path=PREDICTION_OUTPUT_PATH,
        recommendation_output_path=RECOMMENDATION_OUTPUT_PATH,
    )


def run_predict_and_recommend_workflow(
    *,
    fetch_today: bool = False,
    race_day_yyyymmdd: str | None = None,
    prefer_parquet: bool | None = None,
    use_cudf: bool | None = None,
    reload_create_features_module: bool = False,
    baba_scenario_jv_codes: tuple[int, ...] | None = None,
    strategy_baba_scenario_jv_code: int | None = None,
    use_score_ranked_picks: bool = False,
    score_rank_col: str = "pred_rank1",
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    show_top_predictions: int = 10,
    pipeline_verbose: bool = True,
    export_by_course_baba_race: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook 実行用ワンストップ:
    predict → （戦略 EV または 予測スコア順）→ CSV/Parquet 保存 → IPython があれば表を表示。

    推奨モードは strategy_config.json の recommendation_mode（既定 ev_kelly）。
    use_score_ranked_picks=True は非推奨の後方互換で score_rank を強制する。
    """
    runtime_cfg = load_strategy_runtime_config()
    mode = resolve_recommendation_mode(runtime_cfg, use_score_ranked_picks=use_score_ranked_picks)
    logger.info("[recommendation] mode=%s", mode)

    pred_df = run_today_prediction_pipeline(
        fetch_today=fetch_today,
        race_day_yyyymmdd=race_day_yyyymmdd,
        prefer_parquet=prefer_parquet,
        use_cudf=use_cudf,
        reload_create_features_module=reload_create_features_module,
        baba_scenario_jv_codes=baba_scenario_jv_codes,
        strategy_baba_scenario_jv_code=strategy_baba_scenario_jv_code,
        pipeline_verbose=pipeline_verbose,
        export_by_course_baba_race=export_by_course_baba_race,
    )
    if mode == "score_rank":
        rec_df = run_today_score_rank_pipeline(
            pred_df,
            score_col=score_rank_col,
            pair_top_n=pair_top_n,
            wide_top_n=wide_top_n,
        )
    else:
        rec_df = run_today_strategy_pipeline(pred_df)
    rec_df = apply_operating_pair_rule_filter(pred_df, rec_df)
    if show_top_predictions > 0:
        display_top_win_prob_predictions(pred_df, top_n=show_top_predictions)
    display_today_recommendation_summaries(rec_df)
    return pred_df, rec_df


# ---------------------------------------------------------------------------
# オッズ取得・チェック・特徴量・ペア表示（Notebook 用 — view_pipeline へ分離）
# ---------------------------------------------------------------------------
from main.pipeline.view_pipeline import (  # noqa: E402
    check_model_features,
    check_today_tan_odds,
    create_main_pastfeatures,
    display_pair_odds_view,
    fetch_today_pair_odds,
    fetch_today_tan_odds,
)


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    pred_df, rec_df = run_predict_and_recommend_workflow(fetch_today=True)
    logger.info("完了。予測 %d 行 / 推奨 %d 行", len(pred_df), len(rec_df))
