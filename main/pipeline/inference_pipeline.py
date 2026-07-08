"""推論・スコアリング系。

モデルのロード・アンサンブル平均・馬場シナリオ別推論・勝率正規化を担う。
データ取得・戦略ロジックはここに含まない。
"""
from __future__ import annotations

import importlib
import json
import logging
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_CATEGORICAL_FEATURES = frozenset({
    "surface_code",
    "track_condition_code",
    "course_code",
    "weather_code",
    "distance_category",
    "sex_code",
    "class_code",
})

logger = logging.getLogger(__name__)

# model_training.src.create_features と同様: track_code>=23 をダート扱い
DIRT_TRACK_CODE_MIN = 23

# 馬場適性履歴がない馬へのフォールバック勝率・複勝率（全馬平均の経験値）
_TURF_WIN_RATE_FALLBACK: float = 0.07
_TURF_WIN_RATE_FALLBACK_HEAVY: float = 0.06
_DIRT_WIN_RATE_FALLBACK: float = 0.08
_DIRT_WIN_RATE_FALLBACK_HEAVY: float = 0.07
_TURF_TOP3_FALLBACK_LIGHT: float = 0.30
_TURF_TOP3_FALLBACK_SOFT: float = 0.20
_TURF_TOP3_FALLBACK_HEAVY: float = 0.22

_GOING_MODEL_FEATURES = frozenset({"turf_condition", "dirt_condition"})


_ENSEMBLE_STATUSES = frozenset({"trained", "validated"})


def _find_ensemble_meta(models_dir: Path) -> dict | None:
    """ensemble_v*/ensemble_meta.json を新しい順に探索し、モデルが揃っていれば meta を返す。"""
    candidates = sorted(
        models_dir.glob("ensemble_v*/ensemble_meta.json"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for meta_path in candidates:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("status") not in _ENSEMBLE_STATUSES:
            continue
        model_paths = meta.get("model_paths", {})
        if not model_paths:
            continue
        if all(
            (models_dir / rel_path).is_file()
            for paths in model_paths.values()
            for rel_path in paths
        ):
            return meta
    return None


def _check_ensemble_meta(models_dir: Path) -> dict | None:
    """後方互換エイリアス。"""
    return _find_ensemble_meta(models_dir)


def _resolve_model_paths(models_dir: Path) -> dict[int, Path] | dict[int, list[Path]]:
    """
    アンサンブルモードか単体モードかを判定してモデルパスを返す。

    アンサンブルモード（ensemble_v1/ensemble_meta.json が有効な場合）:
        dict[int, list[Path]] を返す（各 rank に対して複数モデルのリスト）。
    単体モード（フォールバック）:
        dict[int, Path] を返す（従来通り）。
    """
    ensemble_meta = _find_ensemble_meta(models_dir)
    if ensemble_meta is not None:
        model_paths: dict[int, list[Path]] = {}
        for rank_key, paths in ensemble_meta["model_paths"].items():
            rank = int(str(rank_key).replace("rank", ""))
            model_paths[rank] = [models_dir / p for p in paths]
        logger.info(
            "アンサンブルモデルを使用: version=%s ranks=%s",
            ensemble_meta.get("version", "?"),
            sorted(model_paths),
        )
        return model_paths

    warnings.warn(
        "アンサンブルモデル（ensemble_v*/ensemble_meta.json）が見つからないか無効です。"
        "従来の単体モデルにフォールバックします。",
        UserWarning,
        stacklevel=2,
    )
    selected = {
        1: models_dir / "lgbm_model_rank1.pkl",
        2: models_dir / "lgbm_model_rank2.pkl",
        3: models_dir / "lgbm_model_rank3.pkl",
    }
    all_non_leak = {
        1: models_dir / "lgbm_model_rank1_all_non_leak.pkl",
        2: models_dir / "lgbm_model_rank2_all_non_leak.pkl",
        3: models_dir / "lgbm_model_rank3_all_non_leak.pkl",
    }
    if all(p.exists() for p in all_non_leak.values()):
        return all_non_leak
    if all(p.exists() for p in selected.values()):
        return selected

    candidates = list(selected.values()) + list(all_non_leak.values())
    missing = [str(p) for p in candidates if not p.exists()]
    raise FileNotFoundError(
        "推論モデルが見つかりません。以下を確認してください:\n" + "\n".join(missing)
    )


def load_models(models_dir: Path) -> dict[int, object] | dict[int, list[object]]:
    """
    アンサンブルモード時: dict[int, list[object]] を返す（各 rank に複数モデルのリスト）。
    単体モード時: dict[int, object] を返す（従来通り）。
    """
    model_paths = _resolve_model_paths(models_dir)
    sample = next(iter(model_paths.values()))
    if isinstance(sample, list):
        models: dict[int, list[object]] = {}
        for rank, paths in model_paths.items():
            models[rank] = []
            for path in paths:
                with path.open("rb") as f:
                    models[rank].append(pickle.load(f))
        return models
    single_models: dict[int, object] = {}
    for rank, path in model_paths.items():
        with path.open("rb") as f:
            single_models[rank] = pickle.load(f)
    return single_models


def load_rank1_isotonic_calibrator(
    models_dir: Path,
    project_root: Path,
) -> tuple[object | None, dict]:
    """
    学習パイプラインが保存した Rank1 用 Isotonic（存在し、メタが有効な場合のみ）。
    戻り値: (モデル or None, メタ dict)
    """
    meta_path = models_dir / "rank1_winprob_isotonic_meta.json"
    if not meta_path.is_file():
        return None, {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if not meta.get("apply_at_inference", True):
        return None, meta
    rel = meta.get(
        "calibrator_relpath", "model_training/models/rank1_winprob_isotonic.pkl"
    )
    pkl_path = (project_root / str(rel).replace("\\", "/")).resolve()
    if not pkl_path.is_file():
        return None, meta
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f), meta
    except Exception:
        return None, meta


def _load_rank_isotonic_calibrator(
    rank: int,
    models_dir: Path,
    project_root: Path,
) -> tuple[object | None, dict]:
    """
    rank2/rank3 用 Isotonic を汎用的にロードする。
    戻り値: (モデル or None, メタ dict)
    """
    meta_path = models_dir / f"rank{rank}_winprob_isotonic_meta.json"
    if not meta_path.is_file():
        return None, {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if not meta.get("apply_at_inference", True):
        return None, meta
    rel = meta.get(
        "calibrator_relpath",
        f"model_training/models/rank{rank}_winprob_isotonic.pkl",
    )
    pkl_path = (project_root / str(rel).replace("\\", "/")).resolve()
    if not pkl_path.is_file():
        return None, meta
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f), meta
    except Exception:
        return None, meta


def load_rank2_isotonic_calibrator(
    models_dir: Path,
    project_root: Path,
) -> tuple[object | None, dict]:
    """
    学習パイプラインが保存した Rank2 用 Isotonic（存在し、メタが有効な場合のみ）。
    戻り値: (モデル or None, メタ dict)
    """
    return _load_rank_isotonic_calibrator(2, models_dir, project_root)


def load_rank3_isotonic_calibrator(
    models_dir: Path,
    project_root: Path,
) -> tuple[object | None, dict]:
    """
    学習パイプラインが保存した Rank3 用 Isotonic（存在し、メタが有効な場合のみ）。
    戻り値: (モデル or None, メタ dict)
    """
    return _load_rank_isotonic_calibrator(3, models_dir, project_root)


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    for col in feature_cols:
        if col in _CATEGORICAL_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1).astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X[feature_cols]


def going_condition_in_model_features(
    models: dict[int, object] | dict[int, list[object]],
) -> bool:
    """
    学習済みモデルの入力特徴に馬場状態（芝/ダート条件）が含まれるか。
    アンサンブルモード時（models[rank] がリスト）は各 rank の最初の 1 モデルのみで判定する。
    特徴量は全シード間で共通のため最初の 1 モデルで十分。
    """
    names: set[str] = set()
    for m in models.values():
        target = m[0] if isinstance(m, list) else m
        fn = getattr(target, "feature_name", None)
        if callable(fn):
            names.update(fn())
    return bool(names & _GOING_MODEL_FEATURES)


def predict_ranks_for_frame(
    models: dict[int, object] | dict[int, list[object]],
    frame: pd.DataFrame,
    *,
    rank1_isotonic: object | None = None,
    rank2_isotonic: object | None = None,
    rank3_isotonic: object | None = None,
) -> dict[int, np.ndarray]:
    """
    各 rank のモデルで予測スコアを計算して返す。

    アンサンブルモード（models[rank] がリスト）:
        全モデルの predict 結果を np.mean(axis=0) で算術平均する。
        Isotonic 補正はアンサンブル平均後に各 rank に対して 1 回だけ適用する。
        各個別モデルへの Isotonic 適用は禁止（仕様）。

    単体モード（models[rank] が単一モデル）:
        従来通り predict してから Isotonic 補正を適用する。
    """
    _isotonic_map: dict[int, object | None] = {
        1: rank1_isotonic,
        2: rank2_isotonic,
        3: rank3_isotonic,
    }
    out: dict[int, np.ndarray] = {}
    for rank, model_or_list in models.items():
        if isinstance(model_or_list, list):
            preds_list: list[np.ndarray] = []
            for m in model_or_list:
                fc = m.feature_name()
                X = _prepare_features(frame, fc)
                preds_list.append(np.asarray(m.predict(X), dtype=float))
            raw = np.mean(np.stack(preds_list, axis=0), axis=0)
        else:
            fc = model_or_list.feature_name()
            X = _prepare_features(frame, fc)
            raw = np.asarray(model_or_list.predict(X), dtype=float)

        iso = _isotonic_map.get(rank)
        if iso is not None:
            raw = np.asarray(
                iso.predict(raw.reshape(-1)), dtype=float
            ).reshape(raw.shape)
        out[rank] = raw
    return out


def apply_uniform_baba_jv_code(df: pd.DataFrame, jv_code: int) -> pd.DataFrame:
    """後方互換: ``main.pipeline.baba_scenario`` へ委譲。"""
    from main.pipeline.baba_scenario import apply_uniform_baba_jv_code as _apply

    return _apply(df, jv_code)


def _recompute_going_delta_active_score(out: pd.DataFrame, jv_code: int) -> None:
    if "going_delta_active_score" not in out.columns:
        return
    from model_training.src.builders.going_delta import compute_going_delta_active_score

    out.loc[:, "going_delta_active_score"] = compute_going_delta_active_score(out, jv_code)


def _recompute_tm_score_surface_adj(out: pd.DataFrame, jv_code: int) -> None:
    """tm_score_surface_adj が存在する場合、馬場シナリオ分の補正を加える。"""
    if "tm_score_surface_adj" not in out.columns or "tm_score" not in out.columns:
        return
    base_adj = pd.to_numeric(out.get("daily_track_variant", 0.0), errors="coerce").fillna(0.0)
    cond_adj_map = {1: 0.0, 2: -5.0, 3: -12.0, 4: -20.0}
    going_adj = float(cond_adj_map.get(int(jv_code), 0.0))
    tm = pd.to_numeric(out["tm_score"], errors="coerce")
    out.loc[:, "tm_score_surface_adj"] = (tm - base_adj + going_adj).astype(np.float32)


def win_prob_from_rank1(df: pd.DataFrame, rank1_col: str) -> pd.Series:
    rank1_score = (
        pd.to_numeric(df[rank1_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    if "race_id" not in df.columns:
        warnings.warn(
            "win_prob_from_rank1: race_id 列が存在しないためレース内正規化不可。"
            "最大頭数（18頭）均等シェア 1/18 をフォールバック値として使用します。",
            UserWarning,
            stacklevel=2,
        )
        fallback = 1.0 / 18
        return pd.Series([fallback] * len(df), index=df.index, dtype=float)
    race_sum = rank1_score.groupby(df["race_id"]).transform("sum").replace(0, np.nan)
    return (rank1_score / race_sum).fillna(0.0).clip(0.0, 1.0)


def bootstrap_encoders_via_create_features_main(project_root: Path) -> None:
    """
    学習用エンコーダが未作成の場合に create_features_main を実行してエンコーダを生成する。
    初回セットアップ時のみ呼ばれる想定。
    """
    import model_training.src.create_features as create_features_module

    create_features_module = importlib.reload(create_features_module)
    create_features_main = create_features_module.create_features_main

    mt = project_root / "model_training" / "data"
    create_features_main(
        input_path=mt / "01_preprocessed" / "horse_data.csv",
        output_path=mt / "02_features" / "features_basic.csv",
        return_df=False,
        fit_encoders=True,
        apply_training_filters=True,
        abnormal_exclude_codes=(1, 3, 4),
        min_horses=6,
        exempt_track_codes=None,
    )
    create_features_main(
        input_path=mt / "01_preprocessed" / "main_horse_data.csv",
        output_path=mt / "02_features" / "main_features_basic.csv",
        return_df=False,
        fit_encoders=False,
        apply_training_filters=False,
    )
