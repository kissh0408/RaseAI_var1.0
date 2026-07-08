import inspect
import json
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import joblib
import optuna
from sklearn.isotonic import IsotonicRegression
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model_training.src.pipeline_common import (
    BASE_LEAK_COLS,
    CATEGORICAL_FEATURES,
    FEATURES_DIR,
    ID_COLS,
    MODELS_DIR,
    PAST_EXTRA_LEAK_COLS,
    TRAIN_ONLY_EXCLUDE_COLS,
    assert_no_leak_columns,
    load_parquet_or_csv,
    load_train_config,
    update_state,
)

# --- 設定 ---
EVAL_PATH = PROJECT_ROOT / "model_training/data/03_train/evaluation.csv"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "model_training/config/train_config.json"


TRAIN_CONFIG = load_train_config(TRAIN_CONFIG_PATH)
_feature_file = TRAIN_CONFIG["training"].get("feature_file", "features_past.parquet")
INPUT_PATH = PROJECT_ROOT / "model_training/data/02_features" / _feature_file
INPUT_CSV_FALLBACK_PATH = INPUT_PATH.with_suffix(".csv")
N_TRIALS = int(TRAIN_CONFIG["training"]["n_trials"])
SEED = int(TRAIN_CONFIG["training"]["seed"])
USE_GPU = bool(TRAIN_CONFIG["training"].get("use_gpu", True))
GPU_PLATFORM_ID = int(TRAIN_CONFIG["training"].get("gpu_platform_id", 0))
GPU_DEVICE_ID = int(TRAIN_CONFIG["training"].get("gpu_device_id", 0))
GPU_USE_DP = bool(TRAIN_CONFIG["training"].get("gpu_use_dp", False))
MAX_BIN = int(TRAIN_CONFIG["training"].get("max_bin", 255))

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 二段階学習の rounds 上限 ---
# Optuna 探索フェーズは少ない rounds で高速評価し、最終モデルのみフル rounds で学習する。
# early_stopping が先に発火する場合は自動短縮される（精度への影響なし）。
OPTUNA_MAX_ROUNDS = int(TRAIN_CONFIG["training"].get("optuna_max_rounds", 500))
FINAL_MAX_ROUNDS  = int(TRAIN_CONFIG["training"].get("final_max_rounds", 2000))


def _train_log(msg: str) -> None:
    """学習パイプラインの進行をロガーへ（実行中の位置把握用）。"""
    logger.info(msg)


def _optimize_study_with_progress_bar(
    study: optuna.study.Study,
    objective,
    *,
    n_trials: int,
    show_progress_bar: bool,
    n_jobs: int = 1,
) -> None:
    kwargs: dict = {
        "n_trials": n_trials,
        "show_progress_bar": bool(show_progress_bar),
        "n_jobs": n_jobs,
    }
    try:
        sig = inspect.signature(study.optimize)
        if "show_progress_bar" not in sig.parameters:
            kwargs.pop("show_progress_bar", None)
        if "n_jobs" not in sig.parameters:
            kwargs.pop("n_jobs", None)
    except (TypeError, ValueError):
        kwargs.pop("show_progress_bar", None)
        kwargs.pop("n_jobs", None)
    study.optimize(objective, **kwargs)


PEDIGREE_REQUIRED_COLS = [
    "sibling_winup_rate_ls",
    "sibling_surface_bias_ls",
    "sibling_heavy_score_ls",
    "sibling_money_rel_z",
    "sibling_avg_win_dist_ls",
    "dam_age_at_birth",
    "pedigree_surface_switch_flag",
    "interact_switch_x_sibling_bias_ls",
    "interact_shallow_x_winup_ls",
]


def load_features(path=INPUT_PATH, csv_fallback_path=INPUT_CSV_FALLBACK_PATH):
    return load_parquet_or_csv(path, csv_fallback_path)


def _categorical_in(features_cols):
    return [c for c in CATEGORICAL_FEATURES if c in features_cols]


def _cast_categoricals_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    LightGBM categorical_feature 用に、カテゴリ列を int に揃える。
    欠損/非数値は -1 に寄せる。
    """
    if not cols:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    return df


def _lgb_gpu_device(params: dict) -> bool:
    return str(params.get("device", "")).lower() in ("gpu", "cuda")


def _lgb_cat_feature_arg(categorical_cols: list[str], params: dict) -> list[str] | str:
    """
    GPU ではカテゴリ bins が max_bin を超えられず LightGBMError になることがあるため、
    整数エンコード済み列は categorical_feature に載せず数値ヒストグラムのみ使う。
    """
    if _lgb_gpu_device(params):
        return []
    return categorical_cols or "auto"


def _warn_missing_selected(df: pd.DataFrame) -> None:
    expected = PEDIGREE_REQUIRED_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning(
            "Missing optional pedigree features in input data: "
            + ", ".join(missing)
        )


def _validate_pedigree_requirements(df: pd.DataFrame, min_coverage: float) -> None:
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError(f"min_pedigree_coverage must be in [0,1], got {min_coverage}")

    missing = [c for c in PEDIGREE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Required pedigree features are missing: " + ", ".join(missing)
        )

    n = len(df)
    if n == 0:
        raise ValueError("Input feature dataframe is empty.")

    low_cov = []
    for c in PEDIGREE_REQUIRED_COLS:
        cov = float(df[c].notna().mean())
        if cov < min_coverage:
            low_cov.append(f"{c}={cov:.4f}")
    if low_cov:
        raise ValueError(
            "Pedigree feature coverage below threshold "
            f"({min_coverage:.2f}): " + ", ".join(low_cov)
        )


def _relevance_from_finish_rank(finish_rank: pd.Series) -> pd.Series:
    """
    Ranking用のrelevance（Rank2 向け／上位ほど高いゲイン）。
    1着=3, 2着=2, 3着=1, それ以外=0
    """
    rel = finish_rank.map({1: 3, 2: 2, 3: 1}).fillna(0)
    return pd.to_numeric(rel, errors="coerce").fillna(0).astype(int)


def _relevance_flat_place_top3(finish_rank: pd.Series) -> pd.Series:
    """Rank3 複勝志向: 3着以内 relevance=1、それ以外 0"""
    r = pd.to_numeric(finish_rank, errors="coerce")
    ok = r.notna() & (r <= 3) & (r >= 1)
    return ok.astype(int)


def _relevance_for_lambdarank(rank: int, finish_rank: pd.Series) -> pd.Series:
    if rank == 3:
        return _relevance_flat_place_top3(finish_rank)
    if rank == 2:
        return _relevance_from_finish_rank(finish_rank)
    raise ValueError("_relevance_for_lambdarank is only for rank 2 or 3")




def run_simulation_from_predictions(
    test_df_sim, target_rank, is_ranking=False, score_col="pred_score"
):
    test_df_sim = test_df_sim.copy()
    has_odds = "odds" in test_df_sim.columns and test_df_sim["odds"].notna().any()
    title = f"Target Rank {target_rank} ({'Ranking' if is_ranking else 'Binary'}) Model"
    logger.info("\n%s %s Evaluation %s", "=" * 20, title, "=" * 20)

    logger.info("\n【閾値別的中率】")
    logger.info("  閾値   購入数 | 1着的中率 | 2着的中率 | 3着的中率 | 回収率(単)")

    if is_ranking:
        scores = test_df_sim["pred_score"]
        th_list = [np.percentile(scores, p) for p in [80, 85, 90, 95, 98]]
    else:
        th_list = [0.05, 0.10, 0.15, 0.20, 0.25]

    for th in th_list:
        bets = test_df_sim[test_df_sim[score_col] >= th].copy()
        if len(bets) == 0:
            continue

        h1 = (bets["finish_rank"] == 1).sum() / len(bets) * 100
        h2 = (bets["finish_rank"] == 2).sum() / len(bets) * 100
        h3 = (bets["finish_rank"] == 3).sum() / len(bets) * 100

        invest = len(bets) * 100
        returns = (
            (bets.loc[bets["finish_rank"] == 1, "odds"].fillna(0) * 100).sum()
            if has_odds
            else 0
        )
        roi = (returns / invest * 100) if invest > 0 else 0

        logger.info(
            "%6.2f %7d | %8.2f%% | %8.2f%% | %8.2f%% | %9.2f%%",
            th, len(bets), h1, h2, h3, roi
        )

    logger.info("\n【上位N頭戦略 的中率】")
    logger.info(" 購入数(1R) | 1着的中率 | 2着的中率 | 3着的中率")
    for n in [1, 2, 3]:
        top_bets = (
            test_df_sim.sort_values(["race_id", score_col], ascending=[True, False])
            .groupby("race_id", sort=False, group_keys=False)
            .head(n)
            .reset_index(drop=True)
        )
        h1 = (top_bets["finish_rank"] == 1).sum() / len(top_bets) * 100
        h2 = (top_bets["finish_rank"] == 2).sum() / len(top_bets) * 100
        h3 = (top_bets["finish_rank"] == 3).sum() / len(top_bets) * 100
        logger.info("%10d | %8.2f%% | %8.2f%% | %8.2f%%", n, h1, h2, h3)

    if has_odds:
        odds = pd.to_numeric(test_df_sim["odds"], errors="coerce")
        score = pd.to_numeric(test_df_sim[score_col], errors="coerce").clip(lower=0)
        test_df_sim["expected_value"] = score * odds

        logger.info("\n【期待値上位N頭戦略】")
        logger.info(" 購入数(1R) | 1着的中率 | 回収率(単)")
        for n in [1, 2, 3]:
            top_ev = (
                test_df_sim.sort_values(
                    ["race_id", "expected_value"], ascending=[True, False]
                )
                .groupby("race_id", sort=False, group_keys=False)
                .head(n)
                .reset_index(drop=True)
            )
            if len(top_ev) == 0:
                continue
            hit1 = (top_ev["finish_rank"] == 1).sum() / len(top_ev) * 100
            invest = len(top_ev) * 100
            returns = (
                top_ev.loc[top_ev["finish_rank"] == 1, "odds"].fillna(0) * 100
            ).sum()
            roi = (returns / invest * 100) if invest > 0 else 0
            logger.info("%10d | %8.2f%% | %9.2f%%", n, hit1, roi)


def run_simulation(model, test_df, features_cols, target_rank, is_ranking=False):
    test_df_sim = test_df.copy()
    test_df_sim["pred_score"] = model.predict(test_df[features_cols])
    run_simulation_from_predictions(
        test_df_sim, target_rank=target_rank, is_ranking=is_ranking, score_col="pred_score"
    )




def _select_features(df, feature_set):
    if feature_set == "selected":
        candidates = [
            "course_code",
            "wakuban",
            "horse_num",
            "age",
            "sex_code",
            "burden_weight",
            "horse_weight",
            "horse_weight_change",
            "weight_ratio",
            "weight_change_ratio",
            "grade_code",
            "race_type_code",
            "distance",
            "track_code",
            "course_kubun",
            "weather_code",
            "turf_condition",
            "dirt_condition",
            "n_horses",
            "mining_predicted_rank",
            "mining_confidence",
            "tm_score",
            "sin_date",
            "cos_date",
            "interval",
            "training_acceleration",
            "time_first_3f",
            "is_ritto_slope",
            "avg_speed_diff_5",
            "avg_speed_diff_all",
            "lag1_speed_diff",
            "jockey_code_encoded",
            "sire_id_encoded",
            "running_style_code_encoded",
            # --- newly added past/context features ---
            "days_since_prev",
            "is_holiday",
            "distance_diff",
            "relative_speed_lag1",
            "speed_deviation_lag1",
            "is_past_abnormal",

            # --- pedigree / interaction candidates (存在する列だけ採用) ---
            "sibling_winup_rate_ls",
            "sibling_surface_bias_ls",
            "sibling_heavy_score_ls",
            "sibling_money_rel_z",
            "sibling_avg_win_dist_ls",
            "dam_age_at_birth",
            "pedigree_surface_switch_flag",
            "interact_switch_x_sibling_bias_ls",
            "interact_shallow_x_winup_ls",
            # --- condition-specific win rates & context shifts ---
            "horse_course_win_rate",
            "horse_surface_win_rate",
            "horse_distance_win_rate",
            "jockey_course_win_rate",
            "jockey_surface_win_rate",
            "jockey_trainer_combo_win_rate",
            "trainer_surface_win_rate",
            "trainer_distance_win_rate",
            "tm_score_lag1",
            "horse_interval_bins",
            "corner4_normalized_lag1",
            "burden_weight_diff_lag1",
            "n_horses_diff_lag1",
            "class_change_code",
        ]
        # 存在する列だけ使う（特徴量追加/削除に強くする）
        return [c for c in candidates if c in df.columns]

    if feature_set == "all_non_leak":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 市場情報(odds/popularity)は状況によりリーク/実運用不可になりやすいので明示除外
        exclude = set(
            BASE_LEAK_COLS + PAST_EXTRA_LEAK_COLS + ID_COLS + TRAIN_ONLY_EXCLUDE_COLS
        )
        return [c for c in numeric_cols if c not in exclude]

    if feature_set == "all_non_leak_with_market":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        train_exclude_only = ["finish_rank", "target", "weight"]
        exclude = set(
            BASE_LEAK_COLS + PAST_EXTRA_LEAK_COLS + ID_COLS + train_exclude_only
        )
        # 単勝オッズ・当該レースの人気をモデル入力に含める比較用（実運用は締切前オッズ等の可否に注意）
        return [c for c in numeric_cols if c not in exclude]

    raise ValueError(f"Unknown feature_set: {feature_set}")


def _model_path(rank, feature_set):
    if feature_set == "selected":
        return PROJECT_ROOT / f"model_training/models/lgbm_model_rank{rank}.pkl"
    return PROJECT_ROOT / f"model_training/models/lgbm_model_rank{rank}_{feature_set}.pkl"


def _eval_path(feature_set):
    if feature_set == "selected":
        return EVAL_PATH
    return EVAL_PATH.with_name(f"evaluation_{feature_set}.csv")


def _run_shap_analysis(
    *,
    feature_set: str,
    sample_size: int,
    top_ev_n: int,
    feature_path: Path = INPUT_PATH,
    csv_fallback_path: Path = INPUT_CSV_FALLBACK_PATH,
) -> None:
    model_path = _model_path(1, feature_set)
    if not model_path.exists():
        logger.warning("Skip SHAP: rank1 model not found at %s", model_path)
        return
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "model_training/scripts/cli.py"),
        "shap",
        "--model-path",
        str(model_path),
        "--feature-path",
        str(feature_path),
        "--csv-fallback-path",
        str(csv_fallback_path),
        "--sample-size",
        str(int(sample_size)),
        "--top-ev-n",
        str(int(top_ev_n)),
    ]
    logger.info("Running SHAP analysis for rank1 model: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ndcg_at_k(y_true, y_score, group_sizes, k):
    idx = 0
    ndcgs = []
    for size in group_sizes:
        true = y_true[idx : idx + size]
        score = y_score[idx : idx + size]
        order = np.argsort(score)[::-1]
        true_sorted = true[order]
        dcg = 0.0
        for i, rel in enumerate(true_sorted[:k]):
            dcg += (2**rel - 1) / np.log2(i + 2)
        ideal = np.sort(true)[::-1]
        idcg = 0.0
        for i, rel in enumerate(ideal[:k]):
            idcg += (2**rel - 1) / np.log2(i + 2)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        idx += size
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _stable_time_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        c for c in ["date", "year", "month_day", "race_id", "horse_num"] if c in df.columns
    ]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols).reset_index(drop=True)


def _normalize_race_id_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    race_id が int32 オーバーフロー等で崩れている場合に、構成要素から復元する。
    """
    if "race_id" not in df.columns:
        return df

    required = ["year", "month_day", "course_code", "kai", "nichi", "race_num"]
    if not all(c in df.columns for c in required):
        return df

    out = df.copy()
    race_id_text = (
        out["race_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    has_negative = race_id_text.str.startswith("-").any()
    has_non_16 = (~race_id_text.str.fullmatch(r"\d{16}")).any()
    if not (has_negative or has_non_16):
        return out

    rebuilt = (
        pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(int).astype(str)
        + pd.to_numeric(out["month_day"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
        + pd.to_numeric(out["course_code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
        + pd.to_numeric(out["kai"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
        + pd.to_numeric(out["nichi"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
        + pd.to_numeric(out["race_num"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )
    out["race_id"] = rebuilt
    logger.info("race_id was normalized from year/month_day/... components.")
    return out


def build_yearly_walkforward_folds(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[dict]:
    if "year" not in df.columns:
        raise ValueError("Column 'year' is required for yearly walk-forward validation.")
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    if len(years) < 2:
        raise ValueError("At least two years are required to build walk-forward folds.")

    start_year = int(start_year) if start_year is not None else years[0] + 1
    end_year = int(end_year) if end_year is not None else years[-1]
    if start_year > end_year:
        raise ValueError(f"Invalid year range: start_year={start_year}, end_year={end_year}")

    folds: list[dict] = []
    for valid_year in range(start_year, end_year + 1):
        train_idx = df.index[df["year"] < valid_year].to_numpy()
        valid_idx = df.index[df["year"] == valid_year].to_numpy()
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        train_year_max = int(df.loc[train_idx, "year"].max())
        valid_year_min = int(df.loc[valid_idx, "year"].min())
        assert train_year_max < valid_year_min, (
            f"Leak detected in fold valid_year={valid_year}: "
            f"max(train_year)={train_year_max}, min(valid_year)={valid_year_min}"
        )
        if "race_id" in df.columns:
            train_race_ids = set(df.loc[train_idx, "race_id"].astype(str))
            valid_race_ids = set(df.loc[valid_idx, "race_id"].astype(str))
            overlap = train_race_ids.intersection(valid_race_ids)
            if overlap:
                raise AssertionError(
                    f"Race leakage detected in valid_year={valid_year}: "
                    f"{len(overlap)} overlapping race_id values."
                )
        folds.append(
            {
                "valid_year": valid_year,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
            }
        )

    if not folds:
        raise ValueError(
            "No walk-forward folds were created. "
            f"Check year range start={start_year}, end={end_year} and data coverage."
        )
    return folds


def _filter_small_races(df: pd.DataFrame, min_group_size: int) -> tuple[pd.DataFrame, int]:
    if min_group_size <= 1 or "race_id" not in df.columns:
        return df, 0
    race_sizes = df.groupby("race_id").size()
    valid_race_ids = race_sizes[race_sizes >= min_group_size].index
    dropped = int((race_sizes < min_group_size).sum())
    out = df[df["race_id"].isin(valid_race_ids)].copy()
    return out, dropped


# Optuna スキップ時のフォールバック（CLAUDE.md 保守的パラメータ）
_CONSERVATIVE_LGBM_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
}


def _optuna_params_path(params_dir: Path, rank: int) -> Path:
    return params_dir / f"optuna_best_params_rank{rank}.json"


def _save_optuna_best_params(params_dir: Path, rank: int, best_params: dict) -> None:
    params_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in best_params.items() if k not in (
        "objective", "metric", "verbosity", "feature_pre_filter", "seed",
        "feature_fraction_seed", "bagging_seed", "boost_from_average",
        "max_bin", "device", "gpu_device_id", "interaction_constraints",
        "monotone_constraints",
    )}
    _optuna_params_path(params_dir, rank).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_optuna_best_params(params_dir: Path, rank: int) -> dict | None:
    p = _optuna_params_path(params_dir, rank)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _suggest_common_lgbm_params(trial: optuna.trial.Trial) -> dict:
    # learning_rate を探索空間に含める。early_stopping と組み合わせることで
    # 低 lr × 多ブースト / 高 lr × 少ブーストのトレードオフを Optuna が最適化する。
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 150),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def _log_rank1_weight_stats(weights: np.ndarray, tag: str) -> None:
    """学習直前のサンプル重み分布をログ（clip とオッズ偏りの確認用）。"""
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        logger.info("[rank1 weights %s] empty", tag)
        return
    qs = np.quantile(w, [0.0, 0.05, 0.5, 0.95, 1.0])
    logger.info(
        "[rank1 weights %s] n=%d min=%.4f p05=%.4f p50=%.4f p95=%.4f max=%.4f mean=%.4f",
        tag, w.size, qs[0], qs[1], qs[2], qs[3], qs[4], float(np.mean(w))
    )


def _filled_time_diff_for_weights(
    fold_df: pd.DataFrame, tc: dict
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    非負のタイム差（秒）と欠損マスクを返す。
    missing 時の埋め方は ``rank1_time_diff_missing_fill``:
    数値 / ``\"median\"`` / ``\"w_min\"``（欠損行は後段で重みを w_min に固定）。
    """
    policy_raw = tc.get("rank1_time_diff_missing_fill", 5.0)
    policy_key = (
        str(policy_raw).strip().lower()
        if isinstance(policy_raw, str)
        else "__numeric__"
    )

    n = len(fold_df)
    if "time_diff" not in fold_df.columns:
        if policy_key == "median":
            return (
                np.full(n, 2.5, dtype=np.float64),
                np.ones(n, dtype=bool),
                policy_key,
            )
        if policy_key == "w_min":
            return np.zeros(n, dtype=np.float64), np.ones(n, dtype=bool), policy_key
        try:
            fillv = float(policy_raw)
        except (TypeError, ValueError):
            fillv = 5.0
        return (
            np.full(n, max(fillv, 0.0), dtype=np.float64),
            np.ones(n, dtype=bool),
            policy_key,
        )

    td = pd.to_numeric(fold_df["time_diff"], errors="coerce")
    missing = td.isna().to_numpy()

    if policy_key == "median":
        med = float(td.dropna().median()) if td.notna().any() else 2.5
        td_f = td.fillna(med)
    elif policy_key == "w_min":
        td_f = td.fillna(0.0)
    else:
        try:
            fillv = float(policy_raw)
        except (TypeError, ValueError):
            fillv = 5.0
        td_f = td.fillna(fillv)

    out = np.maximum(td_f.to_numpy(dtype=np.float64), 0.0)
    return out, missing, policy_key


def _rank1_valid_top3_overlap_rate(df: pd.DataFrame, pred_col: str = "pred_score") -> float:
    """
    各レースで pred 上位 3 頭のうち、実着順が 1〜3 着以内の頭数を数え、
    レースあたり 3 で割った値の平均（フェーズB 軽量プロキシ）。
    """
    need = ["race_id", "finish_rank", pred_col]
    if not all(c in df.columns for c in need) or len(df) == 0:
        return float("nan")
    rates: list[float] = []
    for _, g in df.groupby("race_id", sort=False):
        if len(g) < 2:
            continue
        top3 = g.nlargest(3, pred_col)
        fr = pd.to_numeric(top3["finish_rank"], errors="coerce")
        cnt = int(((fr >= 1) & (fr <= 3)).sum())
        rates.append(cnt / 3.0)
    return float(np.mean(rates)) if rates else float("nan")


def _rank1_fold_targets_weights(fold_df: pd.DataFrame) -> None:
    """Rank1 の target / weight を設定（binary または soft + タイム差減衰重み）。列は in-place。

    rank1_weight_mode:
      - "log1p_odds_x_time_decay" (旧デフォルト): weight = log1p(odds) × exp(-α × time_diff)
        高オッズ馬の重みを過剰に増幅し、ロングショット・シーカー現象を引き起こす恐れあり
      - "time_decay_only" (新デフォルト): weight = exp(-α × time_diff) のみ
        オッズ依存バイアスを排除し、的中率低下の二次要因を解消する
    """
    tc = TRAIN_CONFIG.get("training", {})
    mode = str(tc.get("rank1_label_mode", "binary")).lower().strip()
    w_min = float(tc.get("rank1_weight_min", 0.5))
    w_max = float(tc.get("rank1_weight_max", 3.0))
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    # 重みモードを読み取る（デフォルト: time_decay_only でオッズ依存バイアスを排除）
    weight_mode = str(tc.get("rank1_weight_mode", "time_decay_only")).lower().strip()

    odds_num = pd.to_numeric(fold_df["odds"], errors="coerce").fillna(1.0)
    log1p_odds = np.log1p(odds_num.to_numpy(dtype=np.float64))

    if mode != "soft":
        win = pd.to_numeric(fold_df["finish_rank"], errors="coerce").fillna(-1).astype(
            np.int64
        )
        y = (win == 1).astype(np.float64)
        fold_df["target"] = y
        loser_w = float(tc.get("rank1_binary_loser_weight", 1.0))
        if weight_mode == "log1p_odds_x_time_decay":
            # 旧動作: オッズ依存重み（ロングショット・シーカー現象のリスクあり）
            win_w = np.clip(log1p_odds, w_min, w_max)
        else:
            # time_decay_only: タイム差減衰のみ（オッズ依存バイアス排除）
            td, missing_mask, pol = _filled_time_diff_for_weights(fold_df, tc)
            alpha = float(tc.get("rank1_time_decay_alpha", 0.175))
            raw_w = np.exp(-alpha * td)
            win_w = np.clip(raw_w, w_min, w_max)
            if pol == "w_min":
                win_w = win_w.copy()
                win_w[missing_mask] = w_min
        fold_df["weight"] = np.where(y >= 0.999, win_w, loser_w).astype(np.float64)
        return

    raw_map = tc.get("rank1_soft_targets") or {"1": 1.0, "2": 0.6, "3": 0.4}
    soft_map = {}
    for k, v in raw_map.items():
        try:
            soft_map[int(k)] = float(v)
        except (TypeError, ValueError):
            continue

    fr = pd.to_numeric(fold_df["finish_rank"], errors="coerce").to_numpy()
    tgt = np.zeros(len(fr), dtype=np.float64)
    for k, v in soft_map.items():
        tgt[fr == k] = v
    fold_df["target"] = tgt

    td, missing_mask, pol = _filled_time_diff_for_weights(fold_df, tc)
    alpha = float(tc.get("rank1_time_decay_alpha", 0.175))

    if weight_mode == "log1p_odds_x_time_decay":
        # 旧動作: オッズ × 時間減衰（ロングショット・シーカー現象のリスクあり）
        raw_w = log1p_odds * np.exp(-alpha * td)
    else:
        # time_decay_only: オッズ依存を排除し時間減衰のみで重み付け
        raw_w = np.exp(-alpha * td)

    w = np.clip(raw_w, w_min, w_max)
    if pol == "w_min":
        w = w.copy()
        w[missing_mask] = w_min
    fold_df["weight"] = w.astype(np.float64)


def _rank2_fold_targets_weights(fold_df: pd.DataFrame) -> None:
    """
    Rank2 の target / weight を設定（binary cross_entropy 用）。列は in-place。
    2着馬を正例とし、1着・3着にソフトラベルを付与して2着争いの文脈を学習させる。

    weight_mode は Rank1 と統一し time_decay_only を使用する。
    旧実装の log1p_odds 依存はロングショット・シーカー現象を招くため排除。
    """
    tc = TRAIN_CONFIG.get("training", {})
    w_min = float(tc.get("rank1_weight_min", 0.5))
    w_max = float(tc.get("rank1_weight_max", 3.0))
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    fr = pd.to_numeric(fold_df["finish_rank"], errors="coerce")

    # ソフトラベル: 2着=1.0, 1着=0.3（2着争いで惜しかった可能性）, 3着=0.2, その他=0.0
    tgt = np.zeros(len(fr), dtype=np.float64)
    tgt[fr == 2] = 1.0
    tgt[fr == 1] = 0.3
    tgt[fr == 3] = 0.2
    fold_df["target"] = tgt

    # time_decay_only: Rank1 と同パターンでオッズ依存バイアスを排除し時間減衰のみで重み付け
    td, missing_mask, pol = _filled_time_diff_for_weights(fold_df, tc)
    alpha = float(tc.get("rank1_time_decay_alpha", 0.175))
    raw_w = np.exp(-alpha * td)
    win_w = np.clip(raw_w, w_min, w_max)
    if pol == "w_min":
        win_w = win_w.copy()
        win_w[missing_mask] = w_min
    fold_df["weight"] = np.where(fr == 2, win_w, 0.8).astype(np.float64)


def _rank3_fold_targets_weights(fold_df: pd.DataFrame) -> None:
    """
    Rank3 の target / weight を設定（binary cross_entropy 用）。列は in-place。
    3着以内を正例として扱い、着順に応じた重みを付与する。
    """
    fr = pd.to_numeric(fold_df["finish_rank"], errors="coerce")

    # 3着以内を正例(1.0)、それ以外を負例(0.0)
    tgt = np.where((fr >= 1) & (fr <= 3), 1.0, 0.0)
    fold_df["target"] = tgt.astype(np.float64)

    # 上位の正例を重視: 1着=2.0, 2着=1.5, 3着=1.0, 負例=0.8
    w = np.full(len(fr), 0.8, dtype=np.float64)
    w[fr == 1] = 2.0
    w[fr == 2] = 1.5
    w[fr == 3] = 1.0
    fold_df["weight"] = w


def _base_params_for_rank(rank: int, seed: int | None = None) -> dict:
    _seed = seed if seed is not None else SEED
    if rank == 1:
        tc = TRAIN_CONFIG.get("training", {})
        obj = str(tc.get("rank1_objective", "binary")).lower().strip()
        if obj == "cross_entropy":
            params = {
                "objective": "cross_entropy",
                "metric": "cross_entropy",
                "verbosity": -1,
                "feature_pre_filter": False,
                "seed": _seed,
                "feature_fraction_seed": _seed,
                "bagging_seed": _seed,
                "boost_from_average": False,
            }
        else:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "feature_pre_filter": False,
                "seed": _seed,
                "feature_fraction_seed": _seed,
                "bagging_seed": _seed,
            }
    else:
        # Rank2/Rank3は binary cross_entropy で2着/3着以内確率を直接学習する。
        # LambdaRankは出力が確率でないためEV計算に使えないため変更。
        params = {
            "objective": "cross_entropy",
            "metric": "cross_entropy",
            "verbosity": -1,
            "feature_pre_filter": False,
            "seed": _seed,
            "feature_fraction_seed": _seed,
            "bagging_seed": _seed,
            "boost_from_average": False,
        }
    params["max_bin"] = MAX_BIN
    if USE_GPU:
        # device='cuda' は LightGBM 4.x のネイティブ CUDA サポート（OpenCL 'gpu' とは別コード）。
        # gpu_platform_id/gpu_use_dp は OpenCL 専用パラメータのため CUDA では不要。
        params.update(
            {
                "device": "cuda",
                "gpu_device_id": GPU_DEVICE_ID,
            }
        )
    return params


def _rank1_early_stop_metric_key(params: dict) -> str:
    if str(params.get("objective", "")).lower() == "cross_entropy":
        return "cross_entropy"
    return "binary_logloss"


def _early_stop_metric_key(rank: int, params: dict) -> str:
    """
    rank に応じた early stopping 用メトリクスキーを返す。
    Rank2/3 は cross_entropy に変更したため "cross_entropy" を返す。
    """
    if rank == 1:
        return _rank1_early_stop_metric_key(params)
    # rank 2/3 は cross_entropy objective
    return "cross_entropy"


def _fit_and_save_rank1_isotonic(oof_df: pd.DataFrame, feature_set: str) -> IsotonicRegression | None:
    """
    OOF の pred_score と実 1着で Isotonic を学習し、pkl + メタ JSON を保存。
    ``rank1_isotonic_fit_years`` が list のときは該当年のみで fit（リークと解釈のトレードオフ）。
    """
    tc = TRAIN_CONFIG.get("training", {})
    if not bool(tc.get("rank1_isotonic_enabled", True)):
        return None
    need = ["pred_score", "finish_rank"]
    if not all(c in oof_df.columns for c in need):
        logger.warning("Rank1 isotonic: OOF に pred_score / finish_rank が無いためスキップ")
        return None

    sub = oof_df.copy()
    years_filter = tc.get("rank1_isotonic_fit_years")
    if years_filter is not None and "year" in sub.columns:
        ys = pd.to_numeric(sub["year"], errors="coerce")
        if isinstance(years_filter, int):
            all_years = sorted(ys.dropna().astype(int).unique())
            years_filter = all_years[-years_filter:]
        sub = sub[ys.isin(list(years_filter))].copy()

    x_raw = pd.to_numeric(sub["pred_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = (
        pd.to_numeric(sub["finish_rank"], errors="coerce") == 1
    ).astype(np.float64).to_numpy()
    m = np.isfinite(x_raw) & np.isfinite(y)
    x_raw, y = x_raw[m], y[m]
    if len(x_raw) < 500:
        logger.warning(
            "Rank1 isotonic: サンプル不足 (%d), "
            "500 未満のためスキップ（全 OOF で学習するかフィルタを緩めてください）",
            len(x_raw)
        )
        return None

    from model_training.src.calibration_report import compute_rank1_calibration_metrics

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x_raw, y)
    cal_metrics = compute_rank1_calibration_metrics(
        sub, score_col="pred_score", isotonic_model=iso
    )

    model_dir = PROJECT_ROOT / "model_training" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    iso_path = model_dir / "rank1_winprob_isotonic.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)

    meta = {
        "feature_set": feature_set,
        "calibrator_relpath": "model_training/models/rank1_winprob_isotonic.pkl",
        "fit_n_samples": int(len(x_raw)),
        "brier_raw_clip01": cal_metrics.get("brier_raw"),
        "brier_isotonic": cal_metrics.get("brier_isotonic"),
        "ece_raw_quantile": cal_metrics.get("ece_raw_quantile"),
        "ece_isotonic_quantile": cal_metrics.get("ece_isotonic_quantile"),
        "is_degraded": cal_metrics.get("is_degraded"),
        "ece_gate_failed": cal_metrics.get("ece_gate_failed"),
        "calibration_status": cal_metrics.get("status"),
        "apply_at_inference": True,
    }
    if years_filter is not None:
        meta["isotonic_fit_years"] = [int(y) for y in years_filter]
    meta_path = model_dir / "rank1_winprob_isotonic_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Rank1 isotonic saved: %s | n=%d | "
        "Brier(clip01)=%.6f Brier(iso)=%.6f | "
        "ECE(q) raw=%.6f iso=%.6f | status=%s",
        iso_path, len(x_raw),
        meta["brier_raw_clip01"], meta["brier_isotonic"],
        meta["ece_raw_quantile"], meta["ece_isotonic_quantile"],
        meta["calibration_status"]
    )
    return iso


def _fit_and_save_rank2_isotonic(oof_df: pd.DataFrame, feature_set: str) -> IsotonicRegression | None:
    """
    OOF の pred_score と実 2着で Isotonic を学習し、pkl + メタ JSON を保存。
    rank1_isotonic_fit_years と同様に ``rank2_isotonic_fit_years`` が list のときは該当年のみで fit。
    """
    tc = TRAIN_CONFIG.get("training", {})
    if not bool(tc.get("rank2_isotonic_enabled", True)):
        return None
    need = ["pred_score", "finish_rank"]
    if not all(c in oof_df.columns for c in need):
        logger.warning("Rank2 isotonic: OOF に pred_score / finish_rank が無いためスキップ")
        return None

    sub = oof_df.copy()
    years_filter = tc.get("rank2_isotonic_fit_years")
    if years_filter is not None and "year" in sub.columns:
        ys = pd.to_numeric(sub["year"], errors="coerce")
        if isinstance(years_filter, int):
            all_years = sorted(ys.dropna().astype(int).unique())
            years_filter = all_years[-years_filter:]
        sub = sub[ys.isin(list(years_filter))].copy()

    x_raw = pd.to_numeric(sub["pred_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = (
        pd.to_numeric(sub["finish_rank"], errors="coerce") == 2
    ).astype(np.float64).to_numpy()
    m = np.isfinite(x_raw) & np.isfinite(y)
    x_raw, y = x_raw[m], y[m]
    if len(x_raw) < 500:
        logger.warning(
            "Rank2 isotonic: サンプル不足 (%d), "
            "500 未満のためスキップ（全 OOF で学習するかフィルタを緩めてください）",
            len(x_raw)
        )
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x_raw, y)

    model_dir = PROJECT_ROOT / "model_training" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    iso_path = model_dir / "rank2_winprob_isotonic.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)

    meta = {
        "feature_set": feature_set,
        "calibrator_relpath": "model_training/models/rank2_winprob_isotonic.pkl",
        "fit_n_samples": int(len(x_raw)),
        "target_rank": 2,
        "apply_at_inference": True,
    }
    if years_filter is not None:
        meta["isotonic_fit_years"] = [int(y) for y in years_filter]
    meta_path = model_dir / "rank2_winprob_isotonic_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Rank2 isotonic saved: %s | n=%d",
        iso_path, len(x_raw)
    )
    return iso


def _fit_and_save_rank3_isotonic(oof_df: pd.DataFrame, feature_set: str) -> IsotonicRegression | None:
    """
    OOF の pred_score と実 3着以内で Isotonic を学習し、pkl + メタ JSON を保存。
    ``rank3_isotonic_fit_years`` が list のときは該当年のみで fit。
    """
    tc = TRAIN_CONFIG.get("training", {})
    if not bool(tc.get("rank3_isotonic_enabled", True)):
        return None
    need = ["pred_score", "finish_rank"]
    if not all(c in oof_df.columns for c in need):
        logger.warning("Rank3 isotonic: OOF に pred_score / finish_rank が無いためスキップ")
        return None

    sub = oof_df.copy()
    years_filter = tc.get("rank3_isotonic_fit_years")
    if years_filter is not None and "year" in sub.columns:
        ys = pd.to_numeric(sub["year"], errors="coerce")
        if isinstance(years_filter, int):
            all_years = sorted(ys.dropna().astype(int).unique())
            years_filter = all_years[-years_filter:]
        sub = sub[ys.isin(list(years_filter))].copy()

    x_raw = pd.to_numeric(sub["pred_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    fr = pd.to_numeric(sub["finish_rank"], errors="coerce")
    y = ((fr >= 1) & (fr <= 3)).astype(np.float64).to_numpy()
    m = np.isfinite(x_raw) & np.isfinite(y)
    x_raw, y = x_raw[m], y[m]
    if len(x_raw) < 500:
        logger.warning(
            "Rank3 isotonic: サンプル不足 (%d), "
            "500 未満のためスキップ（全 OOF で学習するかフィルタを緩めてください）",
            len(x_raw)
        )
        return None

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x_raw, y)

    model_dir = PROJECT_ROOT / "model_training" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    iso_path = model_dir / "rank3_winprob_isotonic.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)

    meta = {
        "feature_set": feature_set,
        "calibrator_relpath": "model_training/models/rank3_winprob_isotonic.pkl",
        "fit_n_samples": int(len(x_raw)),
        "target_rank": 3,
        "apply_at_inference": True,
    }
    if years_filter is not None:
        meta["isotonic_fit_years"] = [int(y) for y in years_filter]
    meta_path = model_dir / "rank3_winprob_isotonic_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Rank3 isotonic saved: %s | n=%d",
        iso_path, len(x_raw)
    )
    return iso


_GPU_ONLY_PARAM_KEYS = {"device", "gpu_platform_id", "gpu_device_id", "gpu_use_dp"}
_gpu_fallback_logged = False


def _without_gpu_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in _GPU_ONLY_PARAM_KEYS}


def _train_lgbm_with_fallback(params: dict, *args, **kwargs):
    global _gpu_fallback_logged
    try:
        return lgb.train(params, *args, **kwargs)
    except Exception as e:
        if str(params.get("device", "")).lower() not in ("gpu", "cuda"):
            raise
        cpu_params = _without_gpu_params(params)
        if not _gpu_fallback_logged:
            logger.warning("GPU/CUDA training failed, fallback to CPU. reason=%s", e)
            _gpu_fallback_logged = True
        return lgb.train(cpu_params, *args, **kwargs)


def _feature_selection_report_path(feature_set: str) -> Path:
    return PROJECT_ROOT / f"model_training/data/03_train/feature_selection_{feature_set}.json"


def _log_oof_combo_roi(evaluation_df: pd.DataFrame, feature_set: str) -> None:
    """学習完了後、OOF で馬連・ワイドボックス指標を簡易ログ。"""
    rt = PROJECT_ROOT / "strategy" / "data" / "return_tables.csv"
    if not rt.exists():
        logger.warning("OOF ROI: return_tables.csv が無いためスキップ")
        return
    try:
        from model_training.scripts.cli import run_box_top_n_baseline, run_simulate_all_baseline
        import types
        eb = types.SimpleNamespace(run_box_top_n_baseline=run_box_top_n_baseline, run_simulate_all_baseline=run_simulate_all_baseline)
    except ImportError:
        logger.warning("OOF ROI: cli.py の import に失敗したためスキップ")
        return

    df = evaluation_df.copy()
    if "pred_rank1" not in df.columns:
        logger.warning("OOF ROI: pred_rank1 が無いためスキップ")
        return

    if "finish_rank" in df.columns:
        top1 = (
            df.sort_values(["race_id", "pred_rank1"], ascending=[True, False])
            .groupby("race_id", sort=False)
            .head(1)
        )
        hit1 = float(
            (pd.to_numeric(top1["finish_rank"], errors="coerce").fillna(-1) == 1).mean()
        )
        logger.info(
            "[OOF ROI] feature_set=%s 単勝Top1的中率=%.4f (実1着ラベルでの参考値)",
            feature_set, hit1
        )

    box_df = eb.run_box_top_n_baseline(
        evaluation_df=df,
        return_table_path=rt,
        out_path=None,
        n_values=(3,),
        sort_col="pred_rank1",
    )
    if box_df.empty:
        logger.warning("OOF ROI: box_top_n が空のためスキップ")
    else:
        for _, r in box_df[box_df["n"] == 3].iterrows():
            logger.info(
                "[OOF ROI] n=3 %s: hit_rate=%.4f return_rate=%.4f precision_at_n=%.4f",
                r["ticket_type"], r["hit_rate"], r["return_rate"], r["precision_at_n"]
            )

    sim_df = eb.run_simulate_all_baseline(
        evaluation_path=None,
        return_table_path=rt,
        out_path=None,
        score_th=0.0,
        ev_methods=("none",),
        evaluation_df=df,
    )
    if not sim_df.empty:
        for _, r in sim_df.iterrows():
            logger.info(
                "[OOF ROI] simulate ev=none %s: hit=%s return=%s",
                r["ticket_type"], r["hit_rate_pct"], r["return_rate_pct"]
            )


_GOING_FEATURES_NEVER_DROP = frozenset({"turf_condition", "dirt_condition"})


def _merge_going_training_params(
    params: dict,
    features_cols: list[str],
    config: dict,
) -> dict:
    """going_improvement 設定に基づき interaction_constraints と monotone_constraints を params に注入。"""
    from model_training.src.feature_groups import build_interaction_constraints, build_monotone_constraints

    merged = dict(params)

    ic = build_interaction_constraints(features_cols, config)
    if ic is not None:
        merged["interaction_constraints"] = ic

    mc = build_monotone_constraints(features_cols, config)
    if mc is not None:
        merged["monotone_constraints"] = mc

    return merged


def _apply_baba_weight_multiplier(fold_df: pd.DataFrame, config: dict) -> None:
    """稍重・重・不良レースの sample weight を増幅（クラス不均衡対策）。"""
    gi = config.get("going_improvement", {})
    mode = str(gi.get("rank1_baba_weight_mode", "none")).lower().strip()
    if mode != "multiplier":
        return
    if "weight" not in fold_df.columns or "track_condition_code" not in fold_df.columns:
        return
    raw_map = gi.get("rank1_baba_weights") or {"1": 1.0, "2": 1.5, "3": 2.5, "4": 3.0}
    weight_map = {str(int(k)): float(v) for k, v in raw_map.items()}
    mult = (
        pd.to_numeric(fold_df["track_condition_code"], errors="coerce")
        .fillna(1)
        .astype(int)
        .astype(str)
        .map(weight_map)
        .fillna(1.0)
        .to_numpy(dtype=np.float64)
    )
    fold_df["weight"] = fold_df["weight"].to_numpy(dtype=np.float64) * mult


def _run_lightweight_feature_selection(
    df: pd.DataFrame,
    features_cols: list[str],
    categorical_cols: list[str],
    train_until_year: int,
    enabled: bool = True,
    min_importance_gain: float = 0.0,
    max_drop_ratio: float = 0.2,
    n_estimators: int = 100,
    sample_size_for_contrib: int = 4000,
    seed: int | None = None,
) -> tuple[list[str], dict]:
    if not enabled:
        return features_cols, {"enabled": False, "dropped_features": []}

    probe_df = df[df["year"] < train_until_year].copy()
    if probe_df.empty:
        return features_cols, {
            "enabled": True,
            "reason": "no_probe_data",
            "dropped_features": [],
        }

    probe_df = _cast_categoricals_int(probe_df, categorical_cols)
    probe_df["target"] = (
        pd.to_numeric(probe_df["finish_rank"], errors="coerce") == 1
    ).astype(int)
    if probe_df["target"].nunique() < 2:
        return features_cols, {
            "enabled": True,
            "reason": "single_class_target",
            "dropped_features": [],
        }

    train_data = lgb.Dataset(
        probe_df[features_cols],
        label=probe_df["target"],
        categorical_feature=categorical_cols or "auto",
    )
    _seed = seed if seed is not None else SEED
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": _seed,
        "feature_fraction_seed": _seed,
        "bagging_seed": _seed,
        "num_leaves": 31,
        "learning_rate": 0.08,
    }
    model = _train_lgbm_with_fallback(params, train_data, num_boost_round=n_estimators)

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    importance_df = pd.DataFrame(
        {"feature": features_cols, "gain": gain.astype(float), "split": split.astype(float)}
    )

    low_imp = importance_df[importance_df["gain"] <= float(min_importance_gain)][
        "feature"
    ].tolist()
    contrib_sample = probe_df[features_cols]
    if len(contrib_sample) > sample_size_for_contrib:
        contrib_sample = contrib_sample.sample(sample_size_for_contrib, random_state=_seed)
    contrib = model.predict(contrib_sample, pred_contrib=True)
    if contrib.ndim == 2 and contrib.shape[1] >= len(features_cols):
        # 最終列はbiasなので除外
        mean_abs_contrib = np.mean(np.abs(contrib[:, : len(features_cols)]), axis=0)
    else:
        mean_abs_contrib = np.zeros(len(features_cols))
    contrib_map = {f: float(v) for f, v in zip(features_cols, mean_abs_contrib)}

    # SHAP先取りガード: 低importanceでも寄与が中央値以上なら保持
    low_contrib_values = [contrib_map[f] for f in low_imp]
    contrib_threshold = float(np.median(low_contrib_values)) if low_contrib_values else 0.0
    protected = [
        f for f in low_imp if contrib_map[f] >= contrib_threshold and contrib_map[f] > 0
    ]
    drop_candidates = [f for f in low_imp if f not in protected]
    drop_candidates = [
        f for f in drop_candidates if f not in _GOING_FEATURES_NEVER_DROP
    ]

    max_drop_count = int(len(features_cols) * max_drop_ratio)
    if max_drop_count < len(drop_candidates):
        drop_candidates = sorted(
            drop_candidates, key=lambda x: contrib_map.get(x, 0.0)
        )[:max_drop_count]

    selected = [f for f in features_cols if f not in set(drop_candidates)]
    report = {
        "enabled": True,
        "train_until_year": int(train_until_year),
        "input_feature_count": len(features_cols),
        "selected_feature_count": len(selected),
        "dropped_features": drop_candidates,
        "protected_low_importance_features": protected,
        "min_importance_gain": float(min_importance_gain),
        "max_drop_ratio": float(max_drop_ratio),
        "n_estimators": int(n_estimators),
    }
    return selected, report


def train_model(
    feature_set="all_non_leak",
    n_trials=N_TRIALS,
    require_pedigree=False,
    min_pedigree_coverage=0.0,
    walkforward_start_year: int | None = None,
    walkforward_end_year: int | None = None,
    min_rank_group_size: int = 2,
    enable_feature_selection: bool = True,
    min_importance_gain: float = 0.0,
    max_feature_drop_ratio: float = 0.2,
    run_shap: bool = False,
    shap_sample_size: int = 1000,
    shap_top_ev_n: int = 30,
    show_progress: bool | None = None,
    features_path: str | None = None,
    seed: int | None = None,
    optuna_params_dir: str | Path | None = None,
    optuna_max_rounds: int | None = None,
    final_max_rounds: int | None = None,
):
    effective_seed = seed if seed is not None else SEED
    optuna_params_dir_p = Path(optuna_params_dir) if optuna_params_dir else None
    optuna_max_rounds_eff = int(optuna_max_rounds) if optuna_max_rounds is not None else OPTUNA_MAX_ROUNDS
    final_max_rounds_eff = int(final_max_rounds) if final_max_rounds is not None else FINAL_MAX_ROUNDS
    # train_config.json を毎回読み直す（Notebookでカーネル再起動なしに反映させるため）
    _live_config = load_train_config(TRAIN_CONFIG_PATH)
    configured_file = _live_config["training"].get("feature_file")
    if features_path:
        _features_path = Path(features_path)
    elif configured_file:
        _features_path = PROJECT_ROOT / "model_training/data/02_features" / configured_file
    else:
        _features_path = INPUT_PATH
    logger.info("Loading features from: %s", _features_path.name)
    df = load_features(path=_features_path)
    df = _normalize_race_id_if_needed(df)
    df = _stable_time_sort(df)
    # finish_rank=0 は未確定レース（結果未入力）のため除外
    if "finish_rank" in df.columns:
        _before = len(df)
        df = df[df["finish_rank"] != 0].copy()
        if len(df) < _before:
            logger.info(
                "finish_rank=0 の未確定レース除外: %d件 → %d件 (除外=%d件)",
                _before, len(df), _before - len(df),
            )
    if require_pedigree:
        _validate_pedigree_requirements(df, min_pedigree_coverage)
        logger.info("Pedigree requirement check passed (min_coverage=%.2f)", min_pedigree_coverage)
    else:
        _warn_missing_selected(df)
    features_cols = _select_features(df, feature_set)
    if not features_cols:
        raise ValueError(f"No features selected for feature_set={feature_set}")

    # feature_exclusions による除外（v17 等で不要な特徴量を設定ファイルで制御する）
    exclusions_cfg = _live_config.get("feature_exclusions", {})
    for _, excl_list in exclusions_cfg.items():
        features_cols = [c for c in features_cols if c not in excl_list]
    if exclusions_cfg:
        logger.info(
            "feature_exclusions 適用後の特徴量数: %d (除外グループ: %s)",
            len(features_cols), list(exclusions_cfg.keys())
        )

    assert_no_leak_columns(features_cols, BASE_LEAK_COLS)
    categorical_cols = _categorical_in(features_cols)
    folds = build_yearly_walkforward_folds(
        df,
        start_year=walkforward_start_year,
        end_year=walkforward_end_year,
    )
    logger.info(
        "Feature set: %s | Features count (before selection): %d", feature_set, len(features_cols)
    )
    logger.info(
        "Walk-forward folds: %d (valid years: %s)",
        len(folds), [f["valid_year"] for f in folds]
    )
    if categorical_cols:
        logger.info("Categorical features: %d", len(categorical_cols))

    # Optuna HP 探索に使うフォールドは最終 N 年分を除いてホールドアウトする。
    # これにより「ハイパーパラメータ選択リーク」を防ぎ、テスト期間の独立性を保つ。
    optuna_holdout_years = int(
        TRAIN_CONFIG.get("training", {}).get("optuna_holdout_years", 2)
    )
    if len(folds) > optuna_holdout_years:
        optuna_folds = folds[:-optuna_holdout_years]
    else:
        optuna_folds = folds
        logger.warning(
            "フォールド数(%d)が optuna_holdout_years(%d)"
            " 以下のため、Optuna ホールドアウトは無効化されます。",
            len(folds), optuna_holdout_years
        )
    logger.info(
        "Optuna チューニング folds: %d (valid years: %s)",
        len(optuna_folds), [f["valid_year"] for f in optuna_folds]
    )
    logger.info(
        "ホールドアウト folds (OOF 評価専用): %d (valid years: %s)",
        optuna_holdout_years, [f["valid_year"] for f in folds[-optuna_holdout_years:]]
    )

    _train_log(
        f"学習開始 | サンプル {len(df):,} 行 | walk-forward {len(folds)} folds "
        f"(検証年: {[f['valid_year'] for f in folds]}) | "
        f"Optuna チューニング folds: {len(optuna_folds)}"
    )

    if show_progress is None:
        sp = bool(TRAIN_CONFIG.get("training", {}).get("show_progress", True))
    else:
        sp = bool(show_progress)
    n_folds = len(folds)

    if enable_feature_selection:
        _train_log("特徴量プルーニング（軽量 LightGBM）を実行中…")
    else:
        _train_log("特徴量プルーニングは無効のため、候補列をそのまま使用します")
    t_fs = time.perf_counter()
    selected_features, fs_report = _run_lightweight_feature_selection(
        df=df,
        features_cols=features_cols,
        categorical_cols=categorical_cols,
        train_until_year=min(f["valid_year"] for f in folds),
        enabled=enable_feature_selection,
        min_importance_gain=min_importance_gain,
        max_drop_ratio=max_feature_drop_ratio,
        n_estimators=100,
        seed=effective_seed,
    )
    _train_log(
        f"特徴量プルーニング完了 ({time.perf_counter() - t_fs:.1f}s) "
        f"| 採用特徴数 {len(selected_features)}"
    )
    _train_log(
        f"以降の流れ: Rank1→3 それぞれ "
        f"Optuna {n_trials} 試行 × {len(optuna_folds)} fold 内CV → "
        f"OOF {n_folds} fold → 最終モデル保存"
    )
    features_cols = selected_features
    categorical_cols = _categorical_in(features_cols)
    if not features_cols:
        raise ValueError("No features remained after feature selection.")
    assert_no_leak_columns(features_cols, BASE_LEAK_COLS)
    logger.info("Features count (after selection): %d", len(features_cols))
    fs_path = _feature_selection_report_path(feature_set)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    fs_path.write_text(json.dumps(fs_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Feature selection report saved to %s", fs_path)

    metrics_summary: dict[int, dict] = {}
    evaluation_df = None
    fold_metrics_rows = []

    for rank in [1, 2, 3]:
        is_ranking = rank != 1
        r1_obj = str(TRAIN_CONFIG.get("training", {}).get("rank1_objective", "binary")).lower()
        obj_label = f"rank1_objective={r1_obj}" if rank == 1 else "cross_entropy (binary)"
        _train_log(
            f"---------- Rank {rank}/3 開始 | {obj_label} ----------"
        )

        if rank == 1 and TRAIN_CONFIG.get("training", {}).get(
            "rank1_log_weight_quantiles", True
        ):
            probe = df.loc[folds[0]["train_idx"]].copy()
            probe = _cast_categoricals_int(probe, categorical_cols)
            _rank1_fold_targets_weights(probe)
            _log_rank1_weight_stats(probe["weight"].to_numpy(), "train_fold0_preview")

        def _build_fold_frame(fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
            fold_train = _stable_time_sort(df.loc[fold["train_idx"]].copy())
            fold_valid = _stable_time_sort(df.loc[fold["valid_idx"]].copy())
            fold_train = _cast_categoricals_int(fold_train, categorical_cols)
            fold_valid = _cast_categoricals_int(fold_valid, categorical_cols)

            if rank == 1:
                _rank1_fold_targets_weights(fold_train)
                _rank1_fold_targets_weights(fold_valid)
                _apply_baba_weight_multiplier(fold_train, _live_config)
                _apply_baba_weight_multiplier(fold_valid, _live_config)
                return fold_train, fold_valid, 0

            # Rank2/3 は binary cross_entropy のため group 不要。
            # _filter_small_races は呼ばず、target/weight を専用関数で設定する。
            if rank == 2:
                _rank2_fold_targets_weights(fold_train)
                _rank2_fold_targets_weights(fold_valid)
            else:  # rank == 3
                _rank3_fold_targets_weights(fold_train)
                _rank3_fold_targets_weights(fold_valid)
            _apply_baba_weight_multiplier(fold_train, _live_config)
            _apply_baba_weight_multiplier(fold_valid, _live_config)
            return fold_train, fold_valid, 0

        def _objective_cv(trial: optuna.trial.Trial) -> float:
            params = _merge_going_training_params(
                {**_base_params_for_rank(rank, seed=effective_seed), **_suggest_common_lgbm_params(trial)},
                features_cols,
                _live_config,
            )
            scores = []
            log_px = rank == 1 and TRAIN_CONFIG.get("training", {}).get(
                "rank1_log_optuna_top3_proxy", False
            )
            px_scores: list[float] = [] if log_px else []
            for fold in optuna_folds:
                fold_train, fold_valid, _ = _build_fold_frame(fold)
                if fold_train.empty or fold_valid.empty:
                    continue

                cf = _lgb_cat_feature_arg(categorical_cols, params)
                # Rank1 は weight 付き Dataset。Rank2/3 は binary cross_entropy で
                # group 不要（LambdaRank から変更）。weight を設定する。
                ds_train = lgb.Dataset(
                    fold_train[features_cols],
                    label=fold_train["target"],
                    weight=fold_train["weight"],
                    categorical_feature=cf,
                )
                ds_valid = lgb.Dataset(
                    fold_valid[features_cols],
                    label=fold_valid["target"],
                    reference=ds_train,
                    categorical_feature=cf,
                )
                # Optuna 探索フェーズは OPTUNA_MAX_ROUNDS を上限に高速評価する。
                # early_stopping が先に発火した場合はそちらが優先される。
                model = _train_lgbm_with_fallback(
                    params,
                    ds_train,
                    num_boost_round=optuna_max_rounds_eff,
                    valid_sets=[ds_valid],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                metric_key = _early_stop_metric_key(rank, params)
                scores.append(model.best_score["valid_0"][metric_key])
                if log_px:
                    pv = fold_valid.copy()
                    pv["pred_score"] = model.predict(fold_valid[features_cols])
                    px_scores.append(_rank1_valid_top3_overlap_rate(pv, "pred_score"))

                # 中間スコアを報告し、中央値を下回るトライアルを早期打ち切り。
                trial.report(float(np.mean(scores)), step=len(scores))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if not scores:
                # 全 rank で minimize（cross_entropy / binary_logloss）のため np.inf を返す。
                return np.inf
            if log_px and px_scores:
                tn = getattr(trial, "number", -1)
                logger.info(
                    "[Optuna rank1] trial=%d valid_top3_overlap_mean=%.4f",
                    tn, float(np.nanmean(px_scores))
                )
            return float(np.mean(scores))

        if n_trials > 0:
            # Rank2/3 も cross_entropy（loss）のため minimize。
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=effective_seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
            )
            _optuna_n_jobs = 1 if USE_GPU else 2
            _train_log(
                f"Rank {rank}/3: Optuna 探索開始（試行 {n_trials} × 内CV {len(optuna_folds)} folds, "
                f"max_rounds={optuna_max_rounds_eff}, n_jobs={_optuna_n_jobs}）"
            )
            t_opt = time.perf_counter()
            _optimize_study_with_progress_bar(
                study,
                _objective_cv,
                n_trials=n_trials,
                show_progress_bar=sp,
                n_jobs=_optuna_n_jobs,
            )
            _train_log(
                f"Rank {rank}/3: Optuna 完了 ({time.perf_counter() - t_opt:.0f}s) | "
                f"best_value={study.best_value:.6f}"
            )
            best_params = _merge_going_training_params(
                {**_base_params_for_rank(rank, seed=effective_seed), **study.best_params},
                features_cols,
                _live_config,
            )
            if optuna_params_dir_p is not None:
                _save_optuna_best_params(optuna_params_dir_p, rank, study.best_params)
        else:
            loaded = (
                _load_optuna_best_params(optuna_params_dir_p, rank)
                if optuna_params_dir_p is not None
                else None
            )
            if loaded is not None:
                best_params = _merge_going_training_params(
                    {**_base_params_for_rank(rank, seed=effective_seed), **loaded},
                    features_cols,
                    _live_config,
                )
                _train_log(f"Rank {rank}/3: Optuna スキップ — 保存済み HP を再利用 ({optuna_params_dir_p})")
            else:
                cons = _live_config.get("training", {}).get(
                    "conservative_lgbm_params", _CONSERVATIVE_LGBM_PARAMS
                )
                best_params = _merge_going_training_params(
                    {**_base_params_for_rank(rank, seed=effective_seed), **cons},
                    features_cols,
                    _live_config,
                )
                _train_log(f"Rank {rank}/3: Optuna スキップ — conservative HP を使用")

        oof_parts = []
        rank_metrics = defaultdict(list)
        dropped_race_total = 0

        _train_log(f"Rank {rank}/3: ウォークフォワード OOF（全 {n_folds} folds）")
        fold_iterator = tqdm(
            folds,
            desc=f"Rank{rank} OOF",
            unit="fold",
            leave=True,
            disable=not sp,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for fold in fold_iterator:
            fold_train, fold_valid, dropped_valid_races = _build_fold_frame(fold)
            dropped_race_total += dropped_valid_races
            if fold_train.empty or fold_valid.empty:
                logger.warning(
                    "Skip fold valid_year=%d for rank%d (empty train/valid after filtering).",
                    fold["valid_year"], rank
                )
                continue

            cf_best = _lgb_cat_feature_arg(categorical_cols, best_params)
            # Rank2/3 は binary cross_entropy に変更したため group 不要。
            # 全 rank で weight 付き Dataset を使う。
            ds_train = lgb.Dataset(
                fold_train[features_cols],
                label=fold_train["target"],
                weight=fold_train["weight"],
                categorical_feature=cf_best,
            )
            ds_valid = lgb.Dataset(
                fold_valid[features_cols],
                label=fold_valid["target"],
                reference=ds_train,
                categorical_feature=cf_best,
            )

            # OOF フォールドは FINAL_MAX_ROUNDS を上限にフル学習する（精度保証）。
            fold_model = _train_lgbm_with_fallback(
                best_params,
                ds_train,
                num_boost_round=final_max_rounds_eff,
                valid_sets=[ds_valid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = fold_model.predict(fold_valid[features_cols])
            fold_eval = fold_valid.copy()
            fold_eval["pred_score"] = preds
            fold_eval["valid_year"] = fold["valid_year"]
            oof_parts.append(fold_eval)

            if rank == 1:
                try:
                    from sklearn.metrics import roc_auc_score, log_loss

                    y_hard = (
                        pd.to_numeric(
                            fold_valid["finish_rank"], errors="coerce"
                        ).fillna(-1).astype(int)
                        == 1
                    ).astype(int)
                    auc = roc_auc_score(y_hard, preds)
                    ll = log_loss(y_hard, preds, labels=[0, 1])
                    rank_metrics["auc"].append(float(auc))
                    rank_metrics["log_loss"].append(float(ll))
                    row_r1 = {
                        "rank": rank,
                        "valid_year": fold["valid_year"],
                        "auc": float(auc),
                        "log_loss": float(ll),
                        "dropped_valid_races": int(dropped_valid_races),
                    }
                    if TRAIN_CONFIG.get("training", {}).get(
                        "rank1_log_fold_top3_proxy", True
                    ):
                        row_r1["top3_overlap"] = _rank1_valid_top3_overlap_rate(
                            fold_eval, "pred_score"
                        )
                    fold_metrics_rows.append(row_r1)
                except Exception as e:
                    logger.warning(
                        "Failed rank1 fold metrics (year=%d): %s", fold["valid_year"], e
                    )
            else:
                # Rank2/3 は binary cross_entropy に変更したため AUC/log_loss で評価する。
                # target はソフトラベル（rank2: 2着=1.0, rank3: 3着以内=1.0）のため
                # 実着順から正解ラベルを改めて作成して評価する。
                try:
                    from sklearn.metrics import roc_auc_score, log_loss

                    if rank == 2:
                        y_hard = (
                            pd.to_numeric(
                                fold_valid["finish_rank"], errors="coerce"
                            ).fillna(-1).astype(int)
                            == 2
                        ).astype(int)
                    else:  # rank == 3
                        fr_eval = pd.to_numeric(
                            fold_valid["finish_rank"], errors="coerce"
                        ).fillna(-1).astype(int)
                        y_hard = ((fr_eval >= 1) & (fr_eval <= 3)).astype(int)

                    preds_clipped = np.clip(preds, 1e-7, 1.0 - 1e-7)
                    auc = roc_auc_score(y_hard, preds_clipped)
                    ll = log_loss(y_hard, preds_clipped, labels=[0, 1])
                    rank_metrics["auc"].append(float(auc))
                    rank_metrics["log_loss"].append(float(ll))
                    fold_metrics_rows.append(
                        {
                            "rank": rank,
                            "valid_year": fold["valid_year"],
                            "auc": float(auc),
                            "log_loss": float(ll),
                            "dropped_valid_races": int(dropped_valid_races),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "Failed rank%d fold metrics (year=%d): %s", rank, fold["valid_year"], e
                    )

        if not oof_parts:
            raise RuntimeError(f"No valid folds left for rank{rank}.")
        oof_df = pd.concat(oof_parts, ignore_index=True)

        if rank == 1:
            _fit_and_save_rank1_isotonic(oof_df, feature_set)
        elif rank == 2:
            _fit_and_save_rank2_isotonic(oof_df, feature_set)
        elif rank == 3:
            _fit_and_save_rank3_isotonic(oof_df, feature_set)

        # 最終モデルは最後のfoldを使用（学習データが最大）
        _train_log(
            f"Rank {rank}/3: 最終モデル（最終年フォールドの train/valid で early stopping）"
        )
        t_final = time.perf_counter()
        last_fold = folds[-1]
        final_train, final_valid, _ = _build_fold_frame(last_fold)
        cf_final = _lgb_cat_feature_arg(categorical_cols, best_params)
        # Rank2/3 も binary cross_entropy のため group 不要。全 rank で同じ Dataset 構築。
        ds_train = lgb.Dataset(
            final_train[features_cols],
            label=final_train["target"],
            weight=final_train["weight"],
            categorical_feature=cf_final,
        )
        ds_valid = lgb.Dataset(
            final_valid[features_cols],
            label=final_valid["target"],
            reference=ds_train,
            categorical_feature=cf_final,
        )
        # 最終モデルは FINAL_MAX_ROUNDS を上限にフル学習する。
        final_model = _train_lgbm_with_fallback(
            best_params,
            ds_train,
            num_boost_round=final_max_rounds_eff,
            valid_sets=[ds_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        model_path = _model_path(rank, feature_set)
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        _train_log(
            f"Rank {rank}/3: 完了 ({time.perf_counter() - t_final:.1f}s) | 保存 {model_path.name}"
        )

        if evaluation_df is None:
            base_cols = ["race_id", "horse_num", "odds", "finish_rank"]
            base_cols = [c for c in base_cols if c in oof_df.columns]
            evaluation_df = oof_df[base_cols].copy()
            if "popularity" in oof_df.columns:
                evaluation_df["popularity"] = oof_df["popularity"]
            if "valid_year" in oof_df.columns:
                evaluation_df["valid_year"] = oof_df["valid_year"]
            if "race_num" in oof_df.columns:
                evaluation_df["race_num"] = oof_df["race_num"]
            elif "race_id" in evaluation_df.columns:
                from strategy.src.race_filters import attach_race_num

                evaluation_df = attach_race_num(evaluation_df)
        pred_col = f"pred_rank{rank}"
        pred_frame = oof_df[["race_id", "horse_num", "pred_score"]].rename(
            columns={"pred_score": pred_col}
        )
        evaluation_df = evaluation_df.merge(
            pred_frame, on=["race_id", "horse_num"], how="left"
        )

        if rank == 1:
            auc_mean = float(np.mean(rank_metrics["auc"])) if rank_metrics["auc"] else np.nan
            ll_mean = (
                float(np.mean(rank_metrics["log_loss"]))
                if rank_metrics["log_loss"]
                else np.nan
            )
            metrics_summary[rank] = {
                "auc_mean": auc_mean,
                "log_loss_mean": ll_mean,
                "dropped_valid_races": int(dropped_race_total),
            }
            logger.info(
                "Rank1 Walk-forward mean AUC: %.4f | LogLoss: %.4f",
                auc_mean, ll_mean
            )
        else:
            # Rank2/3 は binary cross_entropy に変更したため AUC/log_loss で集計する。
            auc_mean_rk = (
                float(np.mean(rank_metrics["auc"])) if rank_metrics["auc"] else np.nan
            )
            ll_mean_rk = (
                float(np.mean(rank_metrics["log_loss"])) if rank_metrics["log_loss"] else np.nan
            )
            metrics_summary[rank] = {
                "auc_mean": auc_mean_rk,
                "log_loss_mean": ll_mean_rk,
                "dropped_valid_races": int(dropped_race_total),
            }
            logger.info(
                "Rank%d Walk-forward mean AUC: %.4f | LogLoss: %.4f | dropped_races=%d",
                rank, auc_mean_rk, ll_mean_rk, dropped_race_total
            )

        run_simulation_from_predictions(
            oof_df,
            target_rank=rank,
            is_ranking=is_ranking,
            score_col="pred_score",
        )

    # evaluation.csvの保存
    eval_path = _eval_path(feature_set)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(eval_path, index=False)
    logger.info("Saved evaluation results to %s", eval_path)

    _log_oof_combo_roi(evaluation_df, feature_set)
    fold_metrics_path = eval_path.with_name(f"fold_metrics_{feature_set}.csv")
    pd.DataFrame(fold_metrics_rows).to_csv(fold_metrics_path, index=False)
    logger.info("Saved fold metrics to %s", fold_metrics_path)

    if "pred_rank1" in evaluation_df.columns and "finish_rank" in evaluation_df.columns:
        from model_training.src.calibration_report import write_calibration_report

        report_df = evaluation_df.rename(columns={"pred_rank1": "pred_score"})
        iso_model = None
        iso_pkl = PROJECT_ROOT / "model_training" / "models" / "rank1_winprob_isotonic.pkl"
        if iso_pkl.exists():
            with open(iso_pkl, "rb") as f:
                iso_model = pickle.load(f)
        write_calibration_report(
            report_df,
            feature_set,
            eval_path.parent,
            score_col="pred_score",
            isotonic_model=iso_model,
        )

    if run_shap:
        _run_shap_analysis(
            feature_set=feature_set,
            sample_size=shap_sample_size,
            top_ev_n=shap_top_ev_n,
        )
    return metrics_summary


def compare_feature_sets(n_trials=N_TRIALS, show_progress: bool | None = None):
    results = {}
    for feature_set in ["selected", "all_non_leak"]:
        logger.info("%s", "\n" + "=" * 60)
        logger.info("Training with feature set: %s", feature_set)
        logger.info("%s", "=" * 60)
        results[feature_set] = train_model(
            feature_set=feature_set,
            n_trials=n_trials,
            show_progress=show_progress,
        )

    logger.info("=== Feature Set Comparison Summary ===")
    for feature_set, metrics in results.items():
        logger.info("[%s]", feature_set)
        for rank, m in metrics.items():
            stats = ", ".join([f"{k}: {v:.4f}" for k, v in m.items()])
            logger.info("  Rank%d: %s", rank, stats)


def compare_market_features(n_trials=N_TRIALS, **kwargs) -> tuple[dict, pd.DataFrame]:
    """
    デフォルト特徴量（単勝オッズ・人気除外）vs 両方を数値特徴として含める構成の_walk-forward_ メトリクス比較。

    ``kwargs`` は :func:`train_model` にそのまま渡る（試行回数などを揃えること）。
    """
    sets = ["all_non_leak", "all_non_leak_with_market"]
    labels = {
        "all_non_leak": "baseline_no_market_feats",
        "all_non_leak_with_market": "with_odds_popularity_feats",
    }
    results: dict = {}
    for fs in sets:
        logger.info("%s", "\n" + "=" * 60)
        logger.info("[compare_market_features] feature_set=%s (%s)", fs, labels[fs])
        logger.info("%s", "=" * 60)
        results[fs] = train_model(feature_set=fs, n_trials=n_trials, **kwargs)

    rows = []
    for fs in sets:
        flat: dict[str, object] = {"feature_set": fs, "label": labels[fs]}
        for rank, m in results[fs].items():
            for key, val in m.items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    flat[f"rank{rank}_{key}"] = float(val)
        rows.append(flat)

    cmp_df = pd.DataFrame(rows)
    logger.info("=== 人気・単勝オッズを特徴に入れた場合 vs 運用既定（除外）===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        logger.info("%s", cmp_df.to_string(index=False))

    pivot = cmp_df.drop(columns=["label"]).set_index("feature_set").T
    diff = pivot["all_non_leak_with_market"] - pivot["all_non_leak"]
    cmp_df.attrs["delta_vs_baseline_row"] = diff
    logger.info("[差分: with_market - baseline] （指標により大きい方が良い/悪いは objective により異なる）")
    logger.info("%s", diff.to_string())
    return results, cmp_df


def main_train(**kwargs):
    """Notebook 向けの学習エントリ（train_model と同じ引数）。"""
    return train_model(**kwargs)


def update_train(
    *,
    state_path: str | None = None,
    **kwargs,
) -> dict:
    """
    学習を実行し、最終実行日時を state に保存する。kwargs は train_model に渡る。
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    if state_path is None:
        state_file = (
            project_root
            / "model_training"
            / "data"
            / "state"
            / "train_last_update.json"
        )
    else:
        state_file = Path(state_path)

    train_model(**kwargs)

    return update_state(
        state_file,
        updates={
            "feature_set": kwargs.get("feature_set", "all_non_leak"),
            "n_trials": kwargs.get("n_trials", N_TRIALS),
        },
    )


# ---------------------------------------------------------------------------
# binary 残差系（backtest 検証パス）共有ヘルパー
#
# 復旧理由: refactor `2ca2510` で train.py を rank 専用に書き換えた際、
# binary 残差の backtest 検証パスが依存していた以下4関数が削除され、
#   - strategy/src/inference_common.py  -> compute_base_margin
#   - strategy/src/backtest.py          -> get_feature_cols
#   - strategy/src/binary_recommendation.py -> get_feature_cols
#   - model_training/src/diagnostics.py -> compute_base_margin, get_feature_cols
#   - model_training/src/train_foundation.py -> build_base_params, get_feature_cols,
#                                                load_merged_features
# が import 段階で ImportError を起こしていた。
# 実装は削除前の最後の祖先コミット 5d413d4
# ("fix: leak-free residual learning ...") の train.py から原文を復元したもの。
# モデル挙動は変えない（純粋な復旧）。本番 rank 推論パスとは独立。
# ---------------------------------------------------------------------------

def load_merged_features() -> pd.DataFrame:
    """特徴量ファイルを読み込む（binary 残差 backtest 検証パス用）。

    優先順:
    1. train_config.json の training.feature_file が指す現行 active ファイル
       （旧実装は features_v4/v3/v2 を探したが、現在のリポジトリには存在しない。
        active_experiment が指す features_past_v*.parquet が実体のため、まず
        これを使うアダプタを追加した。理由: ImportError 復旧後の参照崩れ防止）
    2. 上記が無い場合のみ、旧来の features_v6/v4/v3/v2 を後方互換で探索

    NOTE: features_v5（内外回り対応の直線距離）はバックテストで Fold 3 が
    ROI 149.4%→131.0% / Sharpe 0.14→0.09 に悪化したためリジェクト（2026-06-11）。
    """
    df = None

    # 1) 現行 config の active feature_file を最優先
    feature_file = TRAIN_CONFIG.get("training", {}).get("feature_file")
    if feature_file:
        active_path = FEATURES_DIR / feature_file
        if active_path.exists():
            df = pd.read_parquet(active_path)
            print(f"  {feature_file} loaded: {len(df)} rows")

    # 2) 後方互換: 旧 features_v* を探索
    if df is None:
        for ver in ["v6", "v4", "v3", "v2"]:
            path = FEATURES_DIR / f"features_{ver}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"  features_{ver}.parquet loaded: {len(df)} rows")
                break

    if df is None:
        # フォールバック: basic + past を結合
        print("Warning: features parquet が見つかりません。basic + past_v1 にフォールバックします。")
        df = pd.read_parquet(FEATURES_DIR / "features_basic.parquet")
        past_path = FEATURES_DIR / "features_past_v1.parquet"
        if past_path.exists():
            past = pd.read_parquet(past_path)
            key = ["race_id", "horse_id"]
            past_feat_cols = [c for c in past.columns if c not in ["race_id", "horse_id", "race_date", "target"]]
            df = df.merge(past[key + past_feat_cols], on=key, how="left")

    df["race_date"] = pd.to_datetime(df["race_date"])
    # ラベルは一度だけ生成（train_fold での fold ごとの copy + 再計算を回避）
    if "is_win" not in df.columns and "finish_rank" in df.columns:
        df["is_win"] = (df["finish_rank"] == 1).astype(int)
    return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)


def get_feature_cols(cfg: dict) -> list[str]:
    """学習に使用する特徴量列を返す。v2 拡張特徴量も含む。"""
    basic = cfg["features"]["basic"]
    past = cfg["features"]["past"]
    v2ext = cfg.get("features", {}).get("v2_extended", [])
    latent = cfg.get("features", {}).get("latent", [])
    all_cols = basic + past + [c for c in v2ext + latent if c not in basic + past]
    # NaN100% の列・base_margin 列・生オッズ由来の列は特徴量から除外
    exclude_always = {
        "tm_index",
        "gate_straight_cross",
        "jra_tm_log_odds",
        "market_log_odds",
        "market_prob_norm",
        # P0/P1: var1 は init_score 統合のみ。特徴量として混ぜると gain 支配で iter 3-8 飽和。
        "var1_pure_score_z",
        "var1_pure_score",
    }
    cols = [c for c in all_cols if c not in exclude_always]
    exclusions_cfg = cfg.get("feature_exclusions", {})
    for _, excl_list in exclusions_cfg.items():
        if isinstance(excl_list, list):
            cols = [c for c in cols if c not in excl_list]
    return cols


def compute_base_margin(df: pd.DataFrame, col: str = "jra_tm_log_odds") -> np.ndarray:
    """JRA TM の log-odds を base_margin として返す。

    TM データが存在しない馬はレース内均一確率 (1/horse_count) の log-odds をフォールバックとして使用する。
    これにより市場コンセンサスを初期スコアとした残差学習が可能になる。
    """
    if col not in df.columns or df[col].isna().all():
        # フォールバック: 均一確率の log-odds
        n = df["horse_count"].fillna(10)
        p = (1.0 / n).clip(1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).values

    # TM データがない行は均一確率で補完
    base = df[col].copy()
    missing_mask = base.isna()
    if missing_mask.any():
        n = df.loc[missing_mask, "horse_count"].fillna(10)
        p = (1.0 / n).clip(1e-6, 1 - 1e-6)
        base.loc[missing_mask] = np.log(p / (1 - p))
    return base.values


def compute_composite_base_margin(df: pd.DataFrame, t_cfg: dict) -> np.ndarray:
    """binary 残差学習用 init_score: market_log_odds + beta * var1_pure_score_z。

    var1_init_score.enabled=false または beta=0 のとき market のみ。
    推論時 var1 列が欠落・全 NaN のときは beta=0 フォールバック（本番フェイルセーフ）。
    """
    vis = t_cfg.get("var1_init_score") or {}
    market_col = vis.get("market_col") or t_cfg.get("base_margin_col") or "market_log_odds"
    margin = compute_base_margin(df, market_col)

    if not vis.get("enabled", False):
        return margin

    beta = float(vis.get("beta", 0.0))
    if beta == 0.0:
        return margin

    z_col = vis.get("z_col", "var1_pure_score_z")
    if z_col not in df.columns:
        return margin

    z = df[z_col].fillna(0.0).values.astype(float)
    if not np.isfinite(z).any():
        return margin

    return margin + beta * z


def build_base_params(t_cfg: dict, verbose: int | None = None) -> dict:
    """train_config.json の training セクションから LightGBM パラメータを構築する。

    train_fold と run_monthly_walkforward で共通（以前は重複定義されていた）。
    """
    params = {
        "objective": t_cfg["objective"],
        "metric": t_cfg["metric"],
        "learning_rate": t_cfg["learning_rate"],
        "num_leaves": t_cfg["num_leaves"],
        "min_child_samples": t_cfg["min_child_samples"],
        "subsample": t_cfg["subsample"],
        "colsample_bytree": t_cfg["colsample_bytree"],
        "reg_alpha": t_cfg["reg_alpha"],
        "reg_lambda": t_cfg["reg_lambda"],
        "n_estimators": t_cfg["n_estimators"],
        "early_stopping_rounds": t_cfg["early_stopping_rounds"],
        "verbose": t_cfg["verbose"] if verbose is None else verbose,
        "seed": t_cfg["seed"],
    }
    # GPU設定を反映
    # device="cuda"  → RTX等 NVIDIA CUDA（最速、要CUDA版LightGBMビルド）
    # device="gpu"   → OpenCL GPU（このデータ規模では CPU より遅い）
    # device="cpu"   → CPU（force_row_wise=True で高速化）
    device = t_cfg.get("device", "cpu")
    if device != "cpu":
        params["device"] = device
        for key in ("gpu_platform_id", "gpu_device_id", "gpu_use_dp", "max_bin"):
            if key in t_cfg:
                params[key] = t_cfg[key]
    else:
        params["force_row_wise"] = t_cfg.get("force_row_wise", True)
    # ソフトラベル（連続値 [0,1]）は binary objective が受け付けないため
    # cross_entropy に切り替える。sigmoid スコアリングは binary と同一なので
    # 推論パス（raw_score + base_margin → sigmoid）は変更不要。
    if t_cfg.get("soft_label", {}).get("enabled", False):
        params["objective"] = "cross_entropy"
        params["metric"] = "cross_entropy"

    # binary 残差学習(train_fold / train_foundation)専用の保守的 HP 上書き。
    # なぜ: グローバル training.num_leaves 等(63/30/0.1/1.0)は rank ensemble 用で
    # シード不安定。CLAUDE.md は binary に保守的 HP(num_leaves=31, min_child_samples=50,
    # reg_alpha=1.0, reg_lambda=2.0)を必須としている。build_base_params の呼び出し元は
    # すべて binary 残差経路（train_fold / run_ensemble_training / train_foundation）の
    # ため、ここで一括上書きしても rank 学習(train_model)には影響しない。
    conservative = t_cfg.get("backtest_conservative_params")
    if conservative:
        params.update(conservative)
    return params


# ---------------------------------------------------------------------------
# binary 残差系（backtest 検証パス）の *書き込み側* サブシステム
#
# 復旧理由: refactor `2ca2510` で train.py を rank 専用に書き換えた際、
# binary 残差モデル lgbm_binary_fold{N}_seed{S}.txt を *生成* する以下の関数群が
# 削除された（read 側ヘルパーのみ後で復元されていた）:
#   - make_lgb_dataset / apply_soft_labels
#   - train_fold / _train_and_save_folds
#   - run_walkforward_training / run_ensemble_training
# backtest.py / inference_common.py はこれらが書き出すモデルファイルに依存するが、
# 生成器が無いため A1 適用後の再学習が不可能だった。
# 実装は削除前の最後の祖先コミット 5d413d4 の train.py から原文を復元し、
# 現行スキーマ（pipeline_common の MODELS_DIR/load_config・config の training/
# walkforward_folds・backtest_feature_file キー）へ整合させたもの。
# 本番 rank 推論パス（lgbm_model_rank*.pkl）とは独立。
# ---------------------------------------------------------------------------


def _load_binary_training_features() -> pd.DataFrame:
    """binary 残差学習の入力 parquet を読み込む。

    本番 rank の training.feature_file（features_past_v25_odds）は rank 専用列構成で
    binary 残差学習には使えないため、binary は backtest と同じ
    training.backtest_feature_file（features_v6）を学習入力に使う。
    これにより「学習に使うモデル」と「backtest が読むモデル」の特徴量系統が一致する。
    """
    t_cfg = load_train_config(TRAIN_CONFIG_PATH).get("training", {})
    bt_file = t_cfg.get("backtest_feature_file")
    df = None
    if bt_file:
        path = FEATURES_DIR / bt_file
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  [binary] {bt_file} loaded: {len(df)} rows")
        else:
            print(f"  [binary] WARNING: backtest_feature_file {bt_file} が見つかりません。")

    # backtest_feature_file 未設定/不在のときのみ後方互換探索にフォールバック
    if df is None:
        df = load_merged_features()
        return df

    df["race_date"] = pd.to_datetime(df["race_date"])
    if "is_win" not in df.columns and "finish_rank" in df.columns:
        df["is_win"] = (df["finish_rank"] == 1).astype(int)
    return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)


def apply_soft_labels(df: pd.DataFrame, soft_cfg: dict) -> pd.DataFrame:
    """着差減衰ソフトラベル列 soft_label を付与する（質量保存）。

    非勝ち馬に alpha*exp(-着差秒/tau) のクレジットを与え、勝ち馬ラベルを
    1 - クレジット合計 とすることでレース内ラベル質量を 1 に保存する。
    質量保存は init_score（市場勝率 log-odds）とのスケール整合に必須。
    """
    alpha = soft_cfg.get("alpha", 0.2)
    tau = soft_cfg.get("tau", 0.2)
    max_total = soft_cfg.get("max_total_credit", 0.5)

    is_winner = df["finish_rank"] == 1
    credit = np.where(
        df["finish_rank"] > 1,
        alpha * np.exp(-df["time_diff"].clip(lower=0) / tau),
        0.0,
    )
    credit = pd.Series(np.nan_to_num(credit), index=df.index)
    race_credit = credit.groupby(df["race_id"]).transform("sum")
    credit = credit * (max_total / race_credit).clip(upper=1.0)
    race_credit = credit.groupby(df["race_id"]).transform("sum")

    n_winners = is_winner.groupby(df["race_id"]).transform("sum").clip(lower=1)
    df["soft_label"] = np.where(
        is_winner, (1.0 - race_credit) / n_winners, credit
    )
    return df


def make_lgb_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "is_win",
    base_margin: np.ndarray | None = None,
) -> lgb.Dataset:
    """LightGBM binary 分類用 Dataset を作成する。

    base_margin（市場 log-odds）を init_score として与えると残差学習が有効になる。
    予測時は backtest 側で raw_score + base_margin を手動加算して sigmoid に通す。
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df[label_col] if label_col in df.columns else (df["finish_rank"] == 1).astype(int)
    return lgb.Dataset(X, label=y, init_score=base_margin, free_raw_data=False)


def train_fold(
    df: pd.DataFrame,
    fold_cfg: dict,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[lgb.Booster, dict]:
    """1フォールドの binary 残差学習を実行し、モデルとメタ情報を返す。

    バリデーション期間は学習期間と非重複（valid_start > train_end）。
    base_margin（市場 log-odds）を init_score とし市場からの残差を学習する。
    学習パラメータは config の training セクション（build_base_params 経由）から読む。
    A1 適用後は n_trials=0 のため Optuna 探索を完全スキップする。
    """
    t_cfg = cfg["training"]
    fold_n = fold_cfg["fold"]
    base_margin_col = t_cfg.get("base_margin_col", "jra_tm_log_odds")

    train_end = pd.Timestamp(fold_cfg["train_end"])
    valid_start = pd.Timestamp(fold_cfg["valid_start"])
    valid_end = pd.Timestamp(fold_cfg.get("valid_end", fold_cfg.get("test_start", "")))
    test_start = pd.Timestamp(fold_cfg["test_start"])

    # バリデーション期間は学習期間外（non-overlapping）
    train_df = df[df["race_date"] <= train_end]
    valid_df = df[(df["race_date"] >= valid_start) & (df["race_date"] <= valid_end)]

    if "is_win" not in train_df.columns:
        train_df = train_df.copy()
        valid_df = valid_df.copy()
        train_df["is_win"] = (train_df["finish_rank"] == 1).astype(int)
        valid_df["is_win"] = (valid_df["finish_rank"] == 1).astype(int)

    print(f"\n=== Fold {fold_n} ===")
    print(f"  Train: {train_df['race_date'].min().date()} 〜 {train_end.date()} ({len(train_df)} rows)")
    print(f"  Valid: {valid_start.date()} 〜 {valid_end.date()} ({len(valid_df)} rows) [out-of-sample]")
    print(f"  Test : {test_start.date()} 〜 {fold_cfg['test_end']} ({len(df[df['race_date'] >= test_start])} rows)")
    tm_cov = (
        train_df[base_margin_col].notna().mean()
        if base_margin_col and base_margin_col in train_df.columns
        else 0
    )
    bm_label = base_margin_col if base_margin_col else "(disabled)"
    print(f"  base_margin ({bm_label}) coverage: {tm_cov:.1%}")

    base_params = build_base_params(t_cfg)
    available = [c for c in feature_cols if c in train_df.columns]

    # ソフトラベル: 有効時のみ soft_label を学習ラベルに使う（A1 では無効）
    soft_enabled = t_cfg.get("soft_label", {}).get("enabled", False)
    label_col = "soft_label" if soft_enabled and "soft_label" in train_df.columns else "is_win"
    if soft_enabled:
        print(f"  ソフトラベル有効: label_col={label_col}, objective=cross_entropy")

    # base_margin_col が falsy のときは純粋 binary（init_score なし）
    if not base_margin_col:
        train_margin = None
        valid_margin = None
    else:
        train_margin = compute_composite_base_margin(train_df, t_cfg)
        valid_margin = (
            compute_composite_base_margin(valid_df, t_cfg) if len(valid_df) > 0 else None
        )
        vis = t_cfg.get("var1_init_score") or {}
        if vis.get("enabled") and float(vis.get("beta", 0.0)) != 0.0:
            print(
                f"  composite init_score: market + beta={vis.get('beta')} * {vis.get('z_col', 'var1_pure_score_z')}"
            )

    # --- Optuna 探索（n_trials>0 のときのみ。binary は backtest_n_trials=0 でスキップ）---
    # なぜ binary 専用キーを読むか: グローバル training.n_trials(150) は rank ensemble
    # 本番学習用。binary 残差学習でそれを読むと Optuna が走り、バリデーション過学習で
    # Fold3 が REJECT になる（CLAUDE.md『binary は n_trials=0 固定』違反）。binary 専用の
    # backtest_n_trials(=0) を優先して読む。後方互換のため未設定時のみ legacy n_trials に
    # フォールバックするが、現行 config では backtest_n_trials=0 が明示されている。
    n_trials = int(t_cfg.get("backtest_n_trials", t_cfg.get("n_trials", 0)))
    if n_trials > 0 and len(valid_df) > 0:
        print(f"  Optuna: {n_trials}試行でハイパーパラメータ最適化中...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: _binary_optuna_objective(
                trial, train_df, valid_df, available, base_params,
                train_margin, valid_margin,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = {**base_params, **study.best_params}
        print(f"  最良パラメータ: {study.best_params} (LogLoss={study.best_value:.4f})")
    else:
        # n_trials=0: config の保守的パラメータをそのまま使う（探索なし＝過学習回避）
        best_params = dict(base_params)

    from model_training.src.feature_groups import build_backtest_monotone_constraints

    mc = build_backtest_monotone_constraints(available, cfg)
    if mc is not None:
        best_params["monotone_constraints"] = mc
        n_plus = sum(1 for c in mc if c == 1)
        print(f"  monotone_constraints: {n_plus} feature(s) with +1 constraint")

    # --- 最終学習 ---
    train_set = make_lgb_dataset(train_df, available, label_col=label_col, base_margin=train_margin)
    valid_set = (
        make_lgb_dataset(valid_df, available, label_col=label_col, base_margin=valid_margin)
        if len(valid_df) > 0
        else None
    )
    valid_sets = [valid_set] if valid_set is not None else []

    if valid_sets:
        callbacks = [
            lgb.early_stopping(
                best_params.get("early_stopping_rounds", t_cfg["early_stopping_rounds"]),
                verbose=True,
            ),
            lgb.log_evaluation(100),
        ]
    else:
        # 検証データなしでは early stopping 不可（params 経由でも ValueError になる）
        best_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
        callbacks = [lgb.log_evaluation(100)]

    model = _train_lgbm_with_fallback(
        best_params,
        train_set,
        num_boost_round=best_params.get("n_estimators", t_cfg["n_estimators"]),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    effective_objective = "cross_entropy_soft" if soft_enabled else t_cfg["objective"]

    # 学習/バリデーションの logloss を記録し、過学習 gap の把握に使う
    metric_key = t_cfg.get("metric", "binary_logloss")
    valid_score = None
    if valid_sets:
        try:
            valid_score = float(model.best_score.get("valid_0", {}).get(metric_key))
        except (TypeError, ValueError):
            valid_score = None

    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=available,
    ).sort_values(ascending=False)
    top10_ratio = importance.head(10).sum() / importance.sum() if importance.sum() > 0 else 0

    serializable_params = {k: v for k, v in best_params.items() if not callable(v)}
    model_feature_names = list(model.feature_name())
    meta = {
        "fold": fold_n,
        "train_end": str(train_end.date()),
        "test_start": str(test_start),
        "objective": effective_objective,
        "base_margin_col": base_margin_col,
        "base_margin_coverage": float(tm_cov),
        "best_params": serializable_params,
        "best_iteration": model.best_iteration,
        "valid_logloss": valid_score,
        "feature_cols": model_feature_names,
        "top10_importance": importance.head(10).to_dict(),
        "top10_concentration": float(top10_ratio),
        "saved_at": pd.Timestamp.now().isoformat(),
        "lgb_version": lgb.__version__,
    }

    if top10_ratio > 0.8:
        print(f"  Warning: 上位10特徴量が重要度の{top10_ratio:.1%}を占めています（過剰集中の可能性）")

    return model, meta


def _binary_optuna_objective(
    trial: "optuna.Trial",
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    base_params: dict,
    train_margin: np.ndarray | None,
    valid_margin: np.ndarray | None,
) -> float:
    """binary 残差学習の Optuna 目的（valid logloss 最小化）。

    NOTE: A1 では n_trials=0 のため呼ばれない。互換目的で残す。
    """
    space = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
    }
    params = {**base_params, **space}
    train_set = make_lgb_dataset(train_df, feature_cols, base_margin=train_margin)
    valid_set = make_lgb_dataset(valid_df, feature_cols, base_margin=valid_margin)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=base_params["n_estimators"],
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(base_params["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model.best_score.get("valid_0", {}).get(
        base_params.get("metric", "binary_logloss"), 1.0
    )


def _train_and_save_folds(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    seed: int | None = None,
    save_joblib: bool = False,
) -> list[tuple[lgb.Booster, dict]]:
    """全フォールドを学習し、モデルとメタ情報を保存する。

    seed 指定時は lgbm_binary_fold{N}_seed{S}.txt として保存（アンサンブル用）。
    backtest.py / inference_common.py がこの命名を自動検出する。
    """
    results = []
    for fold_cfg in cfg["training"]["walkforward_folds"]:
        model, meta = train_fold(df, fold_cfg, feature_cols, cfg)

        fold_n = fold_cfg["fold"]
        suffix = f"_seed{seed}" if seed is not None else ""
        model_path = MODELS_DIR / f"lgbm_binary_fold{fold_n}{suffix}.txt"
        meta_path = MODELS_DIR / f"lgbm_binary_fold{fold_n}{suffix}_meta.json"

        if seed is not None:
            meta["seed"] = seed
        model.save_model(str(model_path))
        if save_joblib:
            joblib.dump(model, MODELS_DIR / f"lgbm_binary_fold{fold_n}{suffix}.joblib")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  Fold {fold_n} 保存完了: {model_path.name}")
        results.append((model, meta))
    return results


def run_walkforward_training() -> list[tuple[lgb.Booster, dict]]:
    """Walk-Forward Validation で全フォールドの binary 残差モデルを学習する。"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)

    print("Loading binary training features...")
    df = _load_binary_training_features()
    if cfg["training"].get("soft_label", {}).get("enabled", False):
        df = apply_soft_labels(df, cfg["training"]["soft_label"])
    feature_cols = get_feature_cols(cfg)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = _train_and_save_folds(df, feature_cols, cfg, save_joblib=True)

    print(f"\n全{len(results)}フォールドの学習完了。")
    return results


def run_ensemble_training(seeds: list[int] | None = None) -> None:
    """複数シードで walk-forward 学習し、各モデルを seed suffix 付きで保存する。

    各シードを lgbm_binary_fold{N}_seed{S}.txt として保存する。
    backtest.py はこれらを自動検出してアンサンブル予測に使用する。
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    cfg = load_train_config(TRAIN_CONFIG_PATH)
    t_cfg = cfg["training"]

    print("Loading binary training features...")
    df = _load_binary_training_features()
    if t_cfg.get("soft_label", {}).get("enabled", False):
        df = apply_soft_labels(df, t_cfg["soft_label"])
    feature_cols = get_feature_cols(cfg)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate(seeds, start=1):
        print(f"\n{'='*60}")
        print(f"シード {seed} 学習開始 ({i}/{len(seeds)})")
        print(f"{'='*60}")
        # train_fold は cfg["training"]["seed"] を build_base_params 経由で参照するため切替
        t_cfg["seed"] = seed
        _train_and_save_folds(df, feature_cols, cfg, seed=seed)

    print(f"\n全{len(seeds)}シード × {len(t_cfg['walkforward_folds'])}フォールドの学習完了。")


if __name__ == "__main__":
    # binary 残差アンサンブルの再学習サブコマンド（rank パイプラインとは独立）。
    # `python model_training/src/train.py ensemble` で 5 シード × 3 fold = 15 モデルを再生成。
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "ensemble":
        _log_dir = PROJECT_ROOT / "model_training" / "logs" / "train"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = _log_dir / f"train_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        _fh = logging.FileHandler(_log_file, encoding="utf-8")
        _fh.setFormatter(_fmt)
        _sh = logging.StreamHandler()
        _sh.setFormatter(_fmt)
        logging.basicConfig(level=logging.INFO, handlers=[_sh, _fh])
        logging.info("Ensemble training log: %s", _log_file)
        seeds_arg = [42, 43, 44, 45, 46]
        run_ensemble_training(seeds=seeds_arg)
        _sys.exit(0)

    _log_dir = PROJECT_ROOT / "model_training" / "logs" / "train"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = _log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(_fmt)
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_fmt)
    logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
    logging.info("Training log: %s", _log_file)
    parser = argparse.ArgumentParser(description="Train LGBM models.")
    parser.add_argument(
        "--feature-set",
        choices=["selected", "all_non_leak", "all_non_leak_with_market", "compare", "market_compare"],
        default=TRAIN_CONFIG["training"]["default_feature_set"],
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--walkforward-start-year", type=int, default=None)
    parser.add_argument("--walkforward-end-year", type=int, default=None)
    parser.add_argument("--min-rank-group-size", type=int, default=2)
    parser.add_argument(
        "--disable-feature-selection",
        action="store_true",
        help="Disable lightweight pre-selection by feature importance.",
    )
    parser.add_argument("--min-importance-gain", type=float, default=0.0)
    parser.add_argument("--max-feature-drop-ratio", type=float, default=0.2)
    parser.add_argument(
        "--require-pedigree",
        action="store_true",
        help="Fail training if required pedigree features are missing.",
    )
    parser.add_argument(
        "--min-pedigree-coverage",
        type=float,
        default=0.0,
        help="Minimum non-null coverage ratio [0,1] for required pedigree features.",
    )
    parser.add_argument(
        "--run-shap",
        action="store_true",
        default=bool(TRAIN_CONFIG["shap"]["enabled"]),
        help="Run SHAP analysis for rank1 model after training.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=int(TRAIN_CONFIG["shap"]["sample_size"]),
    )
    parser.add_argument(
        "--shap-top-ev-n",
        type=int,
        default=int(TRAIN_CONFIG["shap"]["top_ev_n"]),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Optuna/tqdm の進行バーとフェーズログの一部をオフ（バッチ向け）。",
    )
    args = parser.parse_args()
    prog_kw = {"show_progress": not args.no_progress}

    if args.feature_set == "compare":
        compare_feature_sets(n_trials=args.n_trials, **prog_kw)
    elif args.feature_set == "market_compare":
        compare_market_features(
            n_trials=args.n_trials,
            require_pedigree=args.require_pedigree,
            min_pedigree_coverage=args.min_pedigree_coverage,
            walkforward_start_year=args.walkforward_start_year,
            walkforward_end_year=args.walkforward_end_year,
            min_rank_group_size=args.min_rank_group_size,
            enable_feature_selection=not args.disable_feature_selection,
            min_importance_gain=args.min_importance_gain,
            max_feature_drop_ratio=args.max_feature_drop_ratio,
            run_shap=args.run_shap,
            shap_sample_size=args.shap_sample_size,
            shap_top_ev_n=args.shap_top_ev_n,
            **prog_kw,
        )
    else:
        train_model(
            feature_set=args.feature_set,
            n_trials=args.n_trials,
            require_pedigree=args.require_pedigree,
            min_pedigree_coverage=args.min_pedigree_coverage,
            walkforward_start_year=args.walkforward_start_year,
            walkforward_end_year=args.walkforward_end_year,
            min_rank_group_size=args.min_rank_group_size,
            enable_feature_selection=not args.disable_feature_selection,
            min_importance_gain=args.min_importance_gain,
            max_feature_drop_ratio=args.max_feature_drop_ratio,
            run_shap=args.run_shap,
            shap_sample_size=args.shap_sample_size,
            shap_top_ev_n=args.shap_top_ev_n,
            **prog_kw,
        )
