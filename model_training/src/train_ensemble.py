"""
train_ensemble.py — Multi-Seed Averaging アンサンブル学習スクリプト

仕様:
- rank1 / rank2 / rank3 それぞれについて seed=[42, 100, 200] の 3 モデルを学習する。
- 保存先: model_training/models/ensemble_v1/lgbm_model_rank{rank}_seed{seed}.pkl
- 学習後に ensemble_v1/ensemble_meta.json を生成する。
- 推論時のアンサンブル平均・Isotonic 補正は main/main.py 側で行う。
  このスクリプトはモデルファイルの生成のみを担当する。

既存の lgbm_model_rank{1,2,3}_all_non_leak.pkl は削除・上書きしない。
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# train.py の学習関数をそのまま再利用する（コード重複禁止）。
# ただし train.py のモジュールレベルで TRAIN_CONFIG / SEED などのグローバル変数が
# 初期化されるため、seed を差し替えるにはモジュール内の定数を一時的に上書きする。
import model_training.src.train as _train_module
from model_training.src.pipeline_common import load_train_config

TRAIN_CONFIG_PATH = PROJECT_ROOT / "model_training" / "config" / "train_config.json"
MODELS_DIR = PROJECT_ROOT / "model_training" / "models"


def _load_ensemble_config() -> dict:
    """train_config.json の ensemble セクションを読み込んで返す。"""
    cfg = load_train_config(TRAIN_CONFIG_PATH)
    ens = cfg.get("ensemble", {})
    if not ens.get("enabled", False):
        raise RuntimeError(
            "ensemble.enabled が false です。"
            "train_config.json の ensemble.enabled を true に設定してください。"
        )
    return ens


def _ensemble_model_dir(output_dir: str) -> Path:
    """アンサンブルモデルの保存ディレクトリ（なければ作成）。"""
    d = MODELS_DIR / output_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensemble_model_path(output_dir: str, rank: int, seed: int) -> Path:
    """各 seed × rank のモデルファイルパス。"""
    return MODELS_DIR / output_dir / f"lgbm_model_rank{rank}_seed{seed}.pkl"


def _save_model_to_path(model, dest: Path) -> None:
    """学習済み LightGBM モデルを pickle で保存する。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump(model, f)




def _resolve_walkforward_end_year(prod: dict, train_cfg: dict) -> int | None:
    """walkforward_end_year が未設定なら train_end_date の年を使う。"""
    if prod.get("walkforward_end_year") is not None:
        return int(prod["walkforward_end_year"])
    end_date = train_cfg.get("training", {}).get("train_end_date")
    if end_date:
        return int(str(end_date)[:4])
    return None


def load_production_training_kwargs(
    *,
    config_path: Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    production_training セクションから train_ensemble() 用 kwargs を構築する。

    run_production_train.py / Notebook 本番セルから呼ぶ単一エントリポイント。
    """
    cfg_path = config_path or TRAIN_CONFIG_PATH
    train_cfg = load_train_config(cfg_path)
    prod = train_cfg.get("production_training")
    if not prod:
        raise RuntimeError(
            "train_config.json に production_training セクションがありません。"
        )

    features_dir = PROJECT_ROOT / "model_training" / "data" / "02_features"
    feature_file = str(prod.get("feature_file", train_cfg["training"]["feature_file"]))
    features_path = features_dir / feature_file

    kwargs: dict = {
        "feature_set": str(prod.get("feature_set", "all_non_leak")),
        "n_trials": int(prod.get("n_trials", train_cfg["training"].get("n_trials", 50))),
        "require_pedigree": bool(prod.get("require_pedigree", True)),
        "min_pedigree_coverage": float(prod.get("min_pedigree_coverage", 0.10)),
        "walkforward_start_year": int(prod["walkforward_start_year"])
        if prod.get("walkforward_start_year") is not None
        else None,
        "walkforward_end_year": _resolve_walkforward_end_year(prod, train_cfg),
        "min_rank_group_size": int(prod.get("min_rank_group_size", 2)),
        "enable_feature_selection": bool(prod.get("enable_feature_selection", True)),
        "max_feature_drop_ratio": float(prod.get("max_feature_drop_ratio", 0.2)),
        "show_progress": True,
        "features_path": str(features_path),
        "output_dir": str(prod.get("output_dir", "ensemble_v5")),
        "seeds_filter": [int(s) for s in prod.get("seeds", [])] or None,
        "reuse_optuna": bool(prod.get("reuse_optuna_from_first_seed", True)),
        "optuna_max_rounds": int(prod["optuna_max_rounds"])
        if prod.get("optuna_max_rounds") is not None
        else None,
        "final_max_rounds": int(prod["final_max_rounds"])
        if prod.get("final_max_rounds") is not None
        else None,
    }
    if overrides:
        kwargs.update(overrides)
    return kwargs


def train_ensemble(
    *,
    feature_set: str = "all_non_leak",
    n_trials: int | None = None,
    require_pedigree: bool = False,
    min_pedigree_coverage: float = 0.0,
    walkforward_start_year: int | None = None,
    walkforward_end_year: int | None = None,
    min_rank_group_size: int = 2,
    enable_feature_selection: bool | None = None,
    min_importance_gain: float = 0.0,
    max_feature_drop_ratio: float = 0.2,
    show_progress: bool = True,
    features_path: str | None = None,
    output_dir: str | None = None,
    fast_mode: bool = False,
    seeds_filter: list[int] | None = None,
    skip_existing: bool = True,
    reuse_optuna: bool | None = None,
    optuna_max_rounds: int | None = None,
    final_max_rounds: int | None = None,
) -> Path:
    """
    アンサンブル学習のエントリポイント。

    Parameters
    ----------
    feature_set:
        train.py と同じ特徴量セット名（"all_non_leak" 推奨）。
    n_trials:
        Optuna の試行数。None の場合は train_config.json の設定値を使う。
    walkforward_start_year / walkforward_end_year:
        ウォークフォワード期間の上書き。None で train_config.json のデフォルト。
    enable_feature_selection:
        軽量特徴量プルーニングを有効にするか（既定 True）。
    show_progress:
        Optuna / tqdm の進行バーを表示するか。

    Returns
    -------
    Path
        ensemble_meta.json のパス。
    """
    ens_cfg = _load_ensemble_config()
    train_cfg = load_train_config(TRAIN_CONFIG_PATH)

    seeds: list[int] = [int(s) for s in ens_cfg.get("seeds", [42, 100, 200])]
    if seeds_filter is not None:
        seeds = [int(s) for s in seeds_filter]
    ranks: list[int] = [1, 2, 3]
    output_dir_name: str = str(output_dir or ens_cfg.get("output_dir", "ensemble_v1"))
    method: str = str(ens_cfg.get("method", "mean"))

    gi = train_cfg.get("going_improvement", {})
    fast_cfg = gi.get("fast_mode", {})
    fast_on = fast_mode or bool(fast_cfg.get("enabled", False))
    if reuse_optuna is None:
        reuse_hp = bool(ens_cfg.get("reuse_optuna_from_first_seed", False))
        if fast_on:
            reuse_hp = bool(fast_cfg.get("reuse_optuna_from_first_seed", True))
    else:
        reuse_hp = bool(reuse_optuna)

    if n_trials is None:
        if fast_on:
            n_trials = int(fast_cfg.get("n_trials", 20))
        else:
            n_trials = int(train_cfg["training"].get("n_trials", 50))

    if optuna_max_rounds is None:
        optuna_max_rounds = int(fast_cfg.get("optuna_max_rounds", 0)) if fast_on else None
    if final_max_rounds is None:
        final_max_rounds = int(fast_cfg.get("final_max_rounds", 0)) if fast_on else None
    if optuna_max_rounds == 0:
        optuna_max_rounds = None
    if final_max_rounds == 0:
        final_max_rounds = None

    if enable_feature_selection is None:
        if fast_on and "enable_feature_selection" in fast_cfg:
            enable_feature_selection = bool(fast_cfg["enable_feature_selection"])
        else:
            enable_feature_selection = True

    model_dir = _ensemble_model_dir(output_dir_name)
    if features_path:
        feature_file = Path(features_path).name
    else:
        feature_file = train_cfg["training"].get("feature_file", "features_past_v11.parquet")

    # model_paths スキーマ（ensemble_meta.json 用）
    model_paths_meta: dict[str, list[str]] = {
        f"rank{rank}": [
            f"{output_dir_name}/lgbm_model_rank{rank}_seed{seed}.pkl"
            for seed in seeds
        ]
        for rank in ranks
    }

    print(f"\n{'='*60}")
    print(f"アンサンブル学習開始")
    print(f"  ranks={ranks}, seeds={seeds}, feature_set={feature_set}")
    print(f"  output_dir={model_dir}")
    if reuse_hp:
        print(f"  reuse_optuna=True | n_trials(1st seed only)={n_trials}")
    if fast_on:
        print(f"  fast_mode=True | n_trials(1st seed)={n_trials} | reuse_hp={reuse_hp}")
        print(f"  optuna_max_rounds={optuna_max_rounds or 'default'}")
        print(f"  final_max_rounds={final_max_rounds or 'default'}")
    print(f"{'='*60}\n")

    primary_seed = int(ens_cfg.get("seeds", [42])[0]) if ens_cfg.get("seeds") else 42

    for i, seed in enumerate(seeds):
        if skip_existing and all(
            _ensemble_model_path(output_dir_name, rank, seed).exists() for rank in ranks
        ):
            print(f"[ensemble] seed={seed} は全 rank 保存済みのためスキップ")
            continue

        seed_trials = n_trials if (not reuse_hp or seed == primary_seed) else 0
        seed_fs = enable_feature_selection if seed == primary_seed else False

        print(f"\n[ensemble] ===== seed={seed} 学習開始 (n_trials={seed_trials}) =====")
        _train_module.train_model(
            feature_set=feature_set,
            n_trials=seed_trials,
            require_pedigree=require_pedigree,
            min_pedigree_coverage=min_pedigree_coverage,
            walkforward_start_year=walkforward_start_year,
            walkforward_end_year=walkforward_end_year,
            min_rank_group_size=min_rank_group_size,
            enable_feature_selection=seed_fs,
            min_importance_gain=min_importance_gain,
            max_feature_drop_ratio=max_feature_drop_ratio,
            show_progress=show_progress,
            run_shap=False,
            seed=seed,
            features_path=features_path,
            optuna_params_dir=model_dir,
            optuna_max_rounds=optuna_max_rounds,
            final_max_rounds=final_max_rounds,
        )

        # 各 rank のモデルを ensemble 出力ディレクトリに保存する。
        # 既存の lgbm_model_rank{rank}_all_non_leak.pkl は上書きしない。
        for rank in ranks:
            src = _train_module._model_path(rank, feature_set)
            if not src.exists():
                raise FileNotFoundError(
                    f"train_model() 後にモデルが見つかりません: {src}"
                )
            dst = _ensemble_model_path(output_dir_name, rank, seed)
            with open(src, "rb") as f:
                model = pickle.load(f)
            _save_model_to_path(model, dst)
            print(f"[ensemble] 保存: {dst.relative_to(MODELS_DIR)}")

    # ensemble_meta.json を生成する。
    meta = {
        "version": output_dir_name,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "base_feature_file": feature_file,
        "ensemble_method": method,
        "ranks": ranks,
        "seeds": seeds,
        "model_paths": model_paths_meta,
        "train_config_snapshot": {
            "feature_set": feature_set,
            "n_trials": n_trials,
            "reuse_optuna_from_first_seed": reuse_hp,
            "seed_base": train_cfg["training"].get("seed", 42),
            "optuna_holdout_years": train_cfg["training"].get("optuna_holdout_years", 2),
            "walkforward_start_year": walkforward_start_year,
            "walkforward_end_year": walkforward_end_year,
        },
        "backtest_roi": None,
        "backtest_sharpe": None,
        "status": "trained",
    }

    meta_path = model_dir / "ensemble_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ensemble] ensemble_meta.json 保存: {meta_path}")
    print(f"[ensemble] アンサンブル学習完了。backtest_evaluator によるバックテストを実施してください。")

    return meta_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Seed Averaging アンサンブル学習")
    parser.add_argument(
        "--feature-set",
        default="all_non_leak",
        choices=["selected", "all_non_leak", "all_non_leak_with_market"],
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--walkforward-start-year", type=int, default=None)
    parser.add_argument("--walkforward-end-year", type=int, default=None)
    parser.add_argument(
        "--disable-feature-selection",
        action="store_true",
        help="軽量特徴量プルーニングを無効化する（高速化用）。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Optuna/tqdm の進行バーをオフにする（バッチ向け）。",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="特徴量 parquet のパス（train_config の feature_file を上書き）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="アンサンブル出力ディレクトリ名（ensemble.output_dir を上書き）",
    )
    args = parser.parse_args()

    train_ensemble(
        feature_set=args.feature_set,
        n_trials=args.n_trials,
        walkforward_start_year=args.walkforward_start_year,
        walkforward_end_year=args.walkforward_end_year,
        enable_feature_selection=not args.disable_feature_selection,
        show_progress=not args.no_progress,
        features_path=args.features_path,
        output_dir=args.output_dir,
    )
