"""Exp D: Two-Stage Going Correction モデル学習スクリプト。

Stage 1 (Exp C) の OOF 残差を重・不良馬場サンプルのみで going 特徴量を使って学習し、
going 補正モデル（Stage 2）を生成する。

Usage:
  python model_training/scripts/train_going_correction.py
  python model_training/scripts/train_going_correction.py --alpha 0.3 --n-trials 30
  python model_training/scripts/train_going_correction.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from feature_groups import going_feature_names

EVAL_CSV = ROOT / "model_training" / "data" / "03_train" / "evaluation_all_non_leak.csv"
FEATURES_PATH = ROOT / "model_training" / "data" / "02_features" / "features_past_v26_going_delta.parquet"
STAGE1_DIR = ROOT / "model_training" / "models" / "ensemble_v6_expC"
OUTPUT_DIR = ROOT / "model_training" / "models" / "ensemble_v6_expD"
LOG_DIR = ROOT / "model_training" / "logs" / "going_diagnostics"

# 重・不良条件フィルタ（芝または ダート）
HEAVY_JV_CODES = {3, 4}

# Stage 2 LightGBM 保守パラメータ（少データ過学習防止）
_STAGE2_BASE_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "min_child_samples": 30,
    "reg_alpha": 2.0,
    "reg_lambda": 4.0,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _load_stage1_oof(eval_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    required = {"race_id", "horse_num", "pred_rank1", "finish_rank", "valid_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluation_all_non_leak.csv に必要列がありません: {missing}")
    return df


def _load_going_features(features_path: Path, stage1_oof: pd.DataFrame) -> pd.DataFrame:
    """OOF に going 特徴量と馬場コードをマージ。"""
    feat_cols_needed = ["race_id", "horse_num", "turf_condition", "dirt_condition", "track_code", "year"]
    feat = pd.read_parquet(features_path, columns=None)

    # going_feature_names() が返す列を特定
    all_feat_cols = list(feat.columns)
    going_cols = list(going_feature_names(all_feat_cols))
    load_cols = list(set(feat_cols_needed + going_cols))
    feat = feat[[c for c in load_cols if c in feat.columns]].copy()

    feat["race_id"] = feat["race_id"].astype(str)
    feat["horse_num"] = pd.to_numeric(feat["horse_num"], errors="coerce")
    stage1_oof["race_id"] = stage1_oof["race_id"].astype(str)
    stage1_oof["horse_num"] = pd.to_numeric(stage1_oof["horse_num"], errors="coerce")

    merged = stage1_oof.merge(feat, on=["race_id", "horse_num"], how="inner")
    return merged, going_cols


def _filter_heavy(df: pd.DataFrame) -> pd.DataFrame:
    """重・不良馬場のレースのみ抽出。"""
    is_turf = df["track_code"] < 23
    is_dirt = ~is_turf
    heavy_mask = (
        (is_turf & df["turf_condition"].isin(HEAVY_JV_CODES)) |
        (is_dirt & df["dirt_condition"].isin(HEAVY_JV_CODES))
    )
    return df[heavy_mask].copy()


def _compute_residuals(df: pd.DataFrame) -> pd.Series:
    """残差 = ソフトラベル近似 - Stage 1 pred_rank1。
    ソフトラベル: 1着=1.0, 2着=0.5, 3着=0.2, それ以外=0.0（Exp C 設定に合わせる）
    """
    soft = df["finish_rank"].map({1: 1.0, 2: 0.5, 3: 0.2}).fillna(0.0)
    return soft - df["pred_rank1"]


def _optuna_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    going_cols: list[str],
    n_trials: int,
    seed: int = 42,
) -> dict:
    """Optuna で Stage 2 のハイパーパラメータを探索。バリデーション RMSE を最小化。"""
    def objective(trial: optuna.Trial) -> float:
        params = {
            **_STAGE2_BASE_PARAMS,
            "seed": seed,
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        ds_train = lgb.Dataset(X_train, y_train)
        ds_valid = lgb.Dataset(X_valid, y_valid, reference=ds_train)
        model = lgb.train(
            params,
            ds_train,
            valid_sets=[ds_valid],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        return model.best_score["valid_0"]["rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _train_stage2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    best_params: dict,
    seed: int = 42,
) -> lgb.Booster:
    params = {**_STAGE2_BASE_PARAMS, **best_params, "seed": seed}
    ds_train = lgb.Dataset(X_train, y_train)
    ds_valid = lgb.Dataset(X_valid, y_valid, reference=ds_train)
    model = lgb.train(
        params,
        ds_train,
        valid_sets=[ds_valid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )
    return model


def _tune_alpha(
    df_valid: pd.DataFrame,
    stage2_model: lgb.Booster,
    going_cols: list[str],
    stage1_col: str = "pred_rank1",
) -> float:
    """バリデーション期間でレース内正規化スコアの NDCG を最大化する alpha を探索。"""
    X_v = df_valid[going_cols].fillna(0).to_numpy(dtype=np.float32)
    correction = stage2_model.predict(X_v)

    best_alpha, best_score = 0.0, -np.inf
    for alpha in np.arange(0.0, 0.8, 0.05):
        final = df_valid[stage1_col].to_numpy() + alpha * correction
        # レース内での rank1 的中率（簡易指標）
        tmp = df_valid.copy()
        tmp["final_score"] = final
        hits = (
            tmp.groupby("race_id")
            .apply(lambda g: int(g.loc[g["final_score"].idxmax(), "finish_rank"] == 1), include_groups=False)
            .mean()
        )
        if hits > best_score:
            best_score = hits
            best_alpha = alpha

    print(f"[expD] alpha チューニング: best_alpha={best_alpha:.2f}, top1_hit={best_score:.4f}")
    return float(best_alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp D: Two-Stage Going Correction 学習")
    parser.add_argument("--eval-csv", type=Path, default=EVAL_CSV, help="Stage 1 OOF 評価 CSV")
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--stage1-dir", type=Path, default=STAGE1_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=None, help="補正強度（None でバリデーション最適化）")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna 試行数（0 でスキップ）")
    parser.add_argument("--valid-year", type=int, default=2024, help="alpha チューニングに使うバリデーション年")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[expD] Stage 1 OOF: {args.eval_csv}")
    print(f"[expD] 特徴量: {args.features_path.name}")
    print(f"[expD] Stage 2 出力: {args.output_dir}")

    # ① OOF ロード + 特徴量マージ
    stage1_oof = _load_stage1_oof(args.eval_csv)
    print(f"[expD] OOF rows: {len(stage1_oof)}, years: {sorted(stage1_oof['valid_year'].unique())}")

    merged, going_cols = _load_going_features(args.features_path, stage1_oof)
    print(f"[expD] Merged rows: {len(merged)}, going_cols: {len(going_cols)}")

    # ② 重・不良フィルタ
    heavy = _filter_heavy(merged)
    print(f"[expD] 重・不良サンプル: {len(heavy)} rows ({heavy['race_id'].nunique()} races)")

    if len(heavy) < 200:
        print("[expD] ❌ 重・不良サンプルが 200 件未満。学習を中断。")
        sys.exit(1)

    # ③ 残差計算
    heavy["residual"] = _compute_residuals(heavy)

    # ④ 学習/バリデーション分割（時系列: valid_year 未満 = train, valid_year = valid）
    train_df = heavy[heavy["valid_year"] < args.valid_year].copy()
    valid_df = heavy[heavy["valid_year"] == args.valid_year].copy()
    # alpha チューニング用（全条件）
    valid_all = merged[merged["valid_year"] == args.valid_year].copy()

    print(f"[expD] 学習: {len(train_df)} rows ({train_df['race_id'].nunique()} races, 重/不良のみ)")
    print(f"[expD] バリデーション: {len(valid_df)} rows ({valid_df['race_id'].nunique()} races, 重/不良のみ)")

    if len(train_df) < 100 or len(valid_df) < 20:
        print("[expD] ❌ 学習/バリデーションサンプルが不足。valid_year を変更してください。")
        sys.exit(1)

    X_train = train_df[going_cols].fillna(0).to_numpy(dtype=np.float32)
    y_train = train_df["residual"].to_numpy(dtype=np.float32)
    X_valid = valid_df[going_cols].fillna(0).to_numpy(dtype=np.float32)
    y_valid = valid_df["residual"].to_numpy(dtype=np.float32)

    if args.dry_run:
        print("[expD] --dry-run: 学習をスキップ")
        return

    # ⑤ Optuna ハイパーパラメータ探索
    if args.n_trials > 0:
        print(f"[expD] Optuna 探索 ({args.n_trials} trials)...")
        best_params = _optuna_study(X_train, y_train, X_valid, y_valid, going_cols, args.n_trials)
        print(f"[expD] Best params: {best_params}")
    else:
        best_params = {}
        print("[expD] Optuna スキップ — デフォルトパラメータ使用")

    # ⑥ Stage 2 学習
    print("[expD] Stage 2 学習中...")
    stage2_model = _train_stage2(X_train, y_train, X_valid, y_valid, best_params)
    valid_rmse = stage2_model.best_score["valid_0"]["rmse"]
    print(f"[expD] Stage 2 valid RMSE: {valid_rmse:.6f} (rounds: {stage2_model.best_iteration})")

    # ⑦ alpha チューニング
    if args.alpha is not None:
        alpha = args.alpha
        print(f"[expD] alpha 固定: {alpha}")
    else:
        alpha = _tune_alpha(valid_all, stage2_model, going_cols)

    # ⑧ 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "lgbm_model_going_correction.pkl"
    # 自プロジェクトの学習スクリプトが生成したモデルを保存（外部ソース不使用）
    with open(model_path, "wb") as f:
        pickle.dump(stage2_model, f)

    meta = {
        "generated_at": datetime.now().isoformat(),
        "stage1_dir": str(args.stage1_dir),
        "features_path": str(args.features_path),
        "going_cols": going_cols,
        "alpha": alpha,
        "valid_year": args.valid_year,
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "valid_rmse": valid_rmse,
        "best_params": best_params,
        "heavy_conditions_jv": sorted(HEAVY_JV_CODES),
        "description": "Two-Stage Going Correction (Exp D). Stage 1=expC, Stage 2=going-only residual correction.",
    }
    meta_path = args.output_dir / "going_correction_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[expD] SAVED: {model_path}")
    print(f"[expD] alpha={alpha:.2f}, valid_RMSE={valid_rmse:.6f}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
