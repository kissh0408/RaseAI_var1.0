"""backtest 用: binary 残差モデルの推論・EV オーバーライド。

本番 Notebook 経路（`main/pipeline/inference_pipeline.py` の rank pkl 推論）とは
別系統。詳細は `docs/refactor/inference_paths_map.md` を参照。
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training" / "src"))

from pipeline_common import MODELS_DIR
from train import compute_base_margin, compute_composite_base_margin


def _load_booster_crlf_safe(path: Path) -> lgb.Booster:
    """CRLF 改行に耐性のある Booster ロード。

    なぜ: Windows チェックアウト（core.autocrlf）で .txt モデルの改行が LF→CRLF に
    変換されると、LightGBM のテキストモデルパーサが行末の \\r を解釈できず
    "Model format error, expect a tree here." で失敗する。ファイルを破壊的に
    書き換えず、読み込み時に \\r\\n→\\n へ正規化して model_str でロードする。
    """
    raw = path.read_bytes()
    if b"\r\n" in raw:
        model_str = raw.replace(b"\r\n", b"\n").decode("utf-8")
        return lgb.Booster(model_str=model_str)
    return lgb.Booster(model_file=str(path))


def load_single_model(fold: int) -> lgb.Booster:
    """単一の binary モデルをロードする（シードアンサンブル未生成時のフォールバック）。"""
    binary_path = MODELS_DIR / f"lgbm_binary_fold{fold}.txt"
    if not binary_path.exists():
        raise FileNotFoundError(f"Fold {fold} モデルが見つかりません: {binary_path}")
    return _load_booster_crlf_safe(binary_path)


def load_ensemble_models(fold: int) -> list[lgb.Booster]:
    """シード付きアンサンブルモデルを全てロードする。

    lgbm_binary_fold{N}_seed*.txt が存在する場合はそれを全てロード。
    存在しない場合は単一モデルをリストで返す。
    """
    seed_paths = sorted(MODELS_DIR.glob(f"lgbm_binary_fold{fold}_seed*.txt"))
    if seed_paths:
        models = [_load_booster_crlf_safe(p) for p in seed_paths]
        seeds = [p.stem.split("seed")[-1] for p in seed_paths]
        print(f"  アンサンブル: {len(models)}モデル (seeds={seeds})")
        return models
    return [load_single_model(fold)]


def load_top3_ensemble_models(fold: int) -> list[lgb.Booster]:
    """3着以内専用 binary モデル（lgbm_top3_fold*）をロード。"""
    seed_paths = sorted(MODELS_DIR.glob(f"lgbm_top3_fold{fold}_seed*.txt"))
    if not seed_paths:
        raise FileNotFoundError(
            f"top3 モデルがありません: {MODELS_DIR / f'lgbm_top3_fold{fold}_seed*.txt'}"
        )
    models = [_load_booster_crlf_safe(p) for p in seed_paths]
    seeds = [p.stem.split("seed")[-1] for p in seed_paths]
    print(f"  top3アンサンブル: {len(models)}モデル (seeds={seeds})")
    return models


def load_lambdarank_top3_ensemble_models(fold: int) -> list[lgb.Booster]:
    """Lambdarank top3 relevance モデル（lgbm_lambdarank_top3_fold*）をロード。"""
    seed_paths = sorted(MODELS_DIR.glob(f"lgbm_lambdarank_top3_fold{fold}_seed*.txt"))
    if not seed_paths:
        raise FileNotFoundError(
            f"lambdarank top3 モデルがありません: "
            f"{MODELS_DIR / f'lgbm_lambdarank_top3_fold{fold}_seed*.txt'}"
        )
    models = [_load_booster_crlf_safe(p) for p in seed_paths]
    seeds = [p.stem.split("seed")[-1] for p in seed_paths]
    print(f"  lambdarank top3: {len(models)}モデル (seeds={seeds})")
    return models


def predict_lambdarank_scores(
    models: list[lgb.Booster],
    df_period: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Lambdarank 生スコア（高いほど上位）のアンサンブル平均。"""
    available = [c for c in feature_cols if c in df_period.columns]
    X = df_period[available].values
    raw_list = [m.predict(X, num_iteration=m.best_iteration) for m in models]
    return pd.Series(np.mean(raw_list, axis=0), index=df_period.index)


def _place_log_odds_margin(df: pd.DataFrame) -> np.ndarray:
    n = pd.to_numeric(df["horse_count"], errors="coerce").fillna(16).clip(lower=3)
    p = (3.0 / n).astype(float)
    return np.log(p / (1.0 - p)).to_numpy()


def predict_top3_probs(
    models: list[lgb.Booster],
    df_period: pd.DataFrame,
    feature_cols: list[str],
    temperature: float = 1.0,
) -> pd.Series:
    """top3 残差モデル → レース内正規化確率。"""
    available_here = [c for c in feature_cols if c in df_period.columns]
    X = df_period[available_here].values
    raw_list = [m.predict(X, num_iteration=m.best_iteration, raw_score=True) for m in models]
    raw = np.mean(raw_list, axis=0)
    margin = _place_log_odds_margin(df_period)
    final_log_odds = raw + margin
    t = max(float(temperature), 1e-6)
    if abs(t - 1.0) > 1e-9:
        final_log_odds = final_log_odds / t
    probs = 1.0 / (1.0 + np.exp(-np.clip(final_log_odds, -20, 20)))
    return pd.Series(normalize_within_race(probs, df_period), index=df_period.index)


def normalize_within_race(probs: np.ndarray, df_period: pd.DataFrame) -> np.ndarray:
    """レース内で確率を sum=1 に正規化する（RangeIndex の連番を仮定せず iloc ベースで処理）。"""
    result = np.empty(len(df_period))
    df_reset = df_period.reset_index(drop=True)
    for _, grp in df_reset.groupby("race_id"):
        idx = grp.index.tolist()
        p_grp = probs[idx].clip(1e-7, 1.0)
        total = p_grp.sum()
        result[idx] = (p_grp / total) if total > 0 else (1.0 / len(idx))
    return result


def predict_model_probs(
    models: list[lgb.Booster],
    df_period: pd.DataFrame,
    feature_cols: list[str],
    base_margin_col: str | None,
    temperature: float = 1.0,
    t_cfg: dict | None = None,
) -> pd.Series:
    """binary モデルの出力に init_score（市場 log-odds [+ beta*var1_z]）を加算し確率に変換する。

    LightGBM は init_score を学習時のみ使用し予測時は含めないため、
    base_margin_col / var1_init_score が設定されている場合は手動で加算する。
    t_cfg 指定時は compute_composite_base_margin（方針 A）を優先。
    temperature: 1.0 以外で log-odds をスケール（T>1 で保守的、T<1 で鋭化）。
    """
    available_here = [c for c in feature_cols if c in df_period.columns]
    X = df_period[available_here].values
    # アンサンブル: 全モデルのraw_scoreを平均してからsigmoid変換
    raw_list = [m.predict(X, num_iteration=m.best_iteration, raw_score=True) for m in models]
    raw = np.mean(raw_list, axis=0)

    if t_cfg and t_cfg.get("base_margin_col"):
        final_log_odds = raw + compute_composite_base_margin(df_period, t_cfg)
    elif base_margin_col and base_margin_col in df_period.columns:
        final_log_odds = raw + compute_base_margin(df_period, base_margin_col)
    else:
        final_log_odds = raw

    t = max(float(temperature), 1e-6)
    if abs(t - 1.0) > 1e-9:
        final_log_odds = final_log_odds / t

    probs = 1.0 / (1.0 + np.exp(-np.clip(final_log_odds, -20, 20)))
    return pd.Series(normalize_within_race(probs, df_period), index=df_period.index)


def predict_with_uncertainty(
    models: list[lgb.Booster],
    df_period: pd.DataFrame,
    feature_cols: list[str],
    base_margin_col: str | None,
    t_cfg: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    """アンサンブル平均確率と、シード間の予測標準偏差（不確実性）を返す。

    model_prob は predict_model_probs と数値的に同一（raw平均→sigmoid→正規化）。
    不確実性はシードごとのレース内正規化確率の標準偏差で、データが少ない馬や
    荒れやすいレースで大きくなる（フェーズ5: 過剰投資の安全装置に使用）。
    """
    available_here = [c for c in feature_cols if c in df_period.columns]
    X = df_period[available_here].values
    if t_cfg and t_cfg.get("base_margin_col"):
        margin = compute_composite_base_margin(df_period, t_cfg)
    elif base_margin_col and base_margin_col in df_period.columns:
        margin = compute_base_margin(df_period, base_margin_col)
    else:
        margin = np.zeros(len(df_period))

    raw_list, prob_list = [], []
    for m in models:
        raw = m.predict(X, num_iteration=m.best_iteration, raw_score=True)
        raw_list.append(raw)
        p = 1.0 / (1.0 + np.exp(-np.clip(raw + margin, -20, 20)))
        prob_list.append(normalize_within_race(p, df_period))

    mean_raw = np.mean(raw_list, axis=0)
    p_mean = 1.0 / (1.0 + np.exp(-np.clip(mean_raw + margin, -20, 20)))
    model_prob = normalize_within_race(p_mean, df_period)
    unc = np.std(np.vstack(prob_list), axis=0)
    return (
        pd.Series(model_prob, index=df_period.index),
        pd.Series(unc, index=df_period.index),
    )


def apply_max_picks_per_race(
    df_period: pd.DataFrame,
    base_mask: pd.Series,
    max_picks: int,
) -> pd.Series:
    """EV閾値通過後、レース内EV上位 max_picks 頭のみ残す相対的足切り。

    絶対EV閾値だけではキャリブレーション変化時に推奨頭数が爆発しうるため、
    レース内ランキングで上限を設ける（バックテスト・本番で同一ロジック）。
    """
    if max_picks <= 0 or not base_mask.any():
        return base_mask
    ev_col = "ev_rate" if "ev_rate" in df_period.columns else "expected_value"
    if ev_col not in df_period.columns:
        ev_col = "expected_value"
        work = df_period.copy()
        work[ev_col] = work["model_prob"] * work["odds"].fillna(0)
    else:
        work = df_period
    ev_rank = (
        work.loc[base_mask]
        .groupby("race_id")[ev_col]
        .rank(ascending=False, method="first")
    )
    rank_col = pd.Series(np.nan, index=df_period.index, dtype=float)
    rank_col.loc[base_mask] = ev_rank
    return base_mask & (rank_col <= max_picks)


def apply_condition_overrides(
    df_period: pd.DataFrame,
    base_mask: pd.Series,
    cond_overrides: list[dict],
    default_ev_threshold: float,
) -> pd.Series:
    """条件別EV閾値オーバーライドを適用したマスクを返す。

    該当条件の馬には min_ev より厳しいEV閾値が要求される（過ベット抑制）。
    対応キー: surface_code / track_condition_code / course_code / months / horse_age
    レガシー本番スキーマ: condition_col / condition_val / min_edge も受理（MS-2 統一）。
    """
    if not cond_overrides:
        return base_mask
    override_mask = pd.Series(True, index=df_period.index)
    edge_col = "model_edge" if "model_edge" in df_period.columns else "edge"
    for ov in cond_overrides:
        match = pd.Series(True, index=df_period.index)
        # CLAUDE.md 標準キー
        if "surface_code" in ov and "surface_code" in df_period.columns:
            match &= df_period["surface_code"] == ov["surface_code"]
        if "track_condition_code" in ov and "track_condition_code" in df_period.columns:
            match &= df_period["track_condition_code"] == ov["track_condition_code"]
        if "track_condition_codes" in ov and "track_condition_code" in df_period.columns:
            codes = ov["track_condition_codes"]
            if isinstance(codes, (list, tuple, set)):
                match &= df_period["track_condition_code"].isin(codes)
        if "distance_max" in ov and "distance" in df_period.columns:
            match &= pd.to_numeric(df_period["distance"], errors="coerce") <= float(
                ov["distance_max"]
            )
        if "course_code" in ov and "course_code" in df_period.columns:
            match &= df_period["course_code"] == ov["course_code"]
        if "months" in ov and "race_date" in df_period.columns:
            match &= pd.to_datetime(df_period["race_date"]).dt.month.isin(ov["months"])
        if "horse_age" in ov and "horse_age" in df_period.columns:
            match &= df_period["horse_age"].round(0) == ov["horse_age"]
        # レガシー本番スキーマ（race_filters 互換）
        if "condition_col" in ov:
            col = str(ov["condition_col"])
            val = ov.get("condition_val")
            if col in df_period.columns and val is not None:
                match &= pd.to_numeric(df_period[col], errors="coerce") == float(val)
        if "min_edge" in ov and edge_col in df_period.columns:
            override_mask &= ~match | (
                pd.to_numeric(df_period[edge_col], errors="coerce") >= float(ov["min_edge"])
            )
        elif "ev_rate" in df_period.columns:
            min_ev_ov = ov.get("min_ev", default_ev_threshold)
            override_mask &= ~match | (df_period["ev_rate"] > min_ev_ov)
        if "max_model_rank" in ov and "model_rank" in df_period.columns:
            override_mask &= ~match | (
                pd.to_numeric(df_period["model_rank"], errors="coerce")
                <= int(ov["max_model_rank"])
            )
    return base_mask & override_mask


def apply_condition_overrides_to_recommendations(
    df: pd.DataFrame,
    overrides: list[dict],
    default_ev_threshold: float,
) -> pd.DataFrame:
    """本番推奨 DataFrame に条件別 EV オーバーライドを適用（backtest と同一ロジック）。"""
    out = df.copy()
    if not overrides:
        out["_conditional_ev_ok"] = True
        return out
    prob_col = next(
        (c for c in ("win_prob_est", "model_prob", "pred_prob") if c in out.columns),
        None,
    )
    odds_col = next(
        (c for c in ("effective_odds", "odds") if c in out.columns),
        None,
    )
    if "ev_rate" not in out.columns and prob_col and odds_col:
        out["ev_rate"] = (
            pd.to_numeric(out[prob_col], errors="coerce")
            * pd.to_numeric(out[odds_col], errors="coerce")
        )
    if "model_edge" not in out.columns and "edge" in out.columns:
        out["model_edge"] = out["edge"]
    base = pd.Series(True, index=out.index)
    out["_conditional_ev_ok"] = apply_condition_overrides(
        out, base, overrides, default_ev_threshold
    )
    return out


def apply_race_budget_cap(
    df: pd.DataFrame,
    recommended_mask: pd.Series,
    max_bet_ratio: float,
    bankroll: float,
) -> pd.DataFrame:
    """同一レースの推奨ベット合計を資金の max_bet_ratio 以内に制限する。

    kelly_sizer の max_bet_ratio は1頭単位のキャップであり、複数頭推奨時に
    レース合計が上限を超え得る（CLAUDE.md「1レースあたり資金の5%以内」違反）。
    超過レースは kelly_ratio を比例縮小し、kelly_bet_yen を再計算する。
    """
    df = df.copy()
    rec = df[recommended_mask]
    if len(rec) == 0 or "kelly_ratio" not in df.columns:
        return df
    race_sum = rec.groupby("race_id")["kelly_ratio"].transform("sum")
    scale = (max_bet_ratio / race_sum).clip(upper=1.0)
    df.loc[recommended_mask, "kelly_ratio"] = rec["kelly_ratio"] * scale
    df.loc[recommended_mask, "kelly_bet_yen"] = (
        np.floor(bankroll * df.loc[recommended_mask, "kelly_ratio"] / 100.0) * 100.0
    )
    return df


def compute_market_log_odds(df: pd.DataFrame, odds_col: str = "odds") -> pd.DataFrame:
    """単勝オッズから market_prob / market_prob_norm / market_log_odds を生成する。

    builders/basic.py の特徴量生成と同一ロジック。本番でリアルタイムオッズから
    base_margin を計算する際に使う（過去parquetのlookupでは市場情報が陳腐化するため）。
    """
    df = df.copy()
    raw_market = (1.0 / df[odds_col].clip(lower=1.01)).where(df[odds_col] > 0)
    df["market_prob"] = raw_market
    df["market_prob_norm"] = df.groupby("race_id")["market_prob"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )
    p = df["market_prob_norm"].clip(1e-6, 1 - 1e-6)
    df["market_log_odds"] = np.log(p / (1 - p))
    return df


def apply_var1_market_blend_probs(
    df: pd.DataFrame,
    *,
    z_col: str = "var1_pure_score_z",
    odds_col: str = "odds",
    beta: float = 0.30,
    race_id_col: str = "race_id",
) -> pd.Series:
    """var1 ベッティングレイヤー: logit(p) = logit(p_market) + beta * z（RaceAI_var1.0 R-6）。

    市場情報は EV 計算専用。学習特徴量 var1_pure_score_z とは別に、確率合成に使う。
    """
    if z_col not in df.columns:
        raise KeyError(f"{z_col} not in DataFrame — run merge_var1_pure_scores first")
    work = df.copy()
    if "market_prob_norm" not in work.columns:
        work["_mp_raw"] = (1.0 / work[odds_col].clip(lower=1.01)).where(work[odds_col] > 0)
        work["market_prob_norm"] = work.groupby(race_id_col)["_mp_raw"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else x
        )
        work = work.drop(columns=["_mp_raw"])

    p_m = work["market_prob_norm"].clip(1e-6, 1 - 1e-6).values
    z = work[z_col].fillna(0.0).values.astype(float)
    logits = np.log(p_m / (1.0 - p_m)) + float(beta) * z
    raw_p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    return pd.Series(normalize_within_race(raw_p, work), index=df.index)


def predict_combo_ranking_scores(
    fold: int,
    df_period: pd.DataFrame,
    feature_cols: list[str],
    *,
    prob_source: str = "win",
    blend_alpha: float = 0.5,
    base_margin_col: str = "market_log_odds",
    temperature: float = 1.0,
) -> pd.Series:
    """ワイド/馬連の top3 順位付け用スコア（高いほど上位）。

    prob_source: win | top3 | blend | lambdarank
    """
    source = prob_source.lower()
    if source == "win":
        models = load_ensemble_models(fold)
        return predict_model_probs(
            models, df_period, feature_cols, base_margin_col, temperature
        )
    if source == "top3":
        models = load_top3_ensemble_models(fold)
        return predict_top3_probs(models, df_period, feature_cols, temperature)
    if source == "lambdarank":
        models = load_lambdarank_top3_ensemble_models(fold)
        return predict_lambdarank_scores(models, df_period, feature_cols)
    if source == "blend":
        win_models = load_ensemble_models(fold)
        top3_models = load_top3_ensemble_models(fold)
        win_p = predict_model_probs(
            win_models, df_period, feature_cols, base_margin_col, temperature
        )
        top3_p = predict_top3_probs(top3_models, df_period, feature_cols, temperature)
        a = float(blend_alpha)
        blended = (1.0 - a) * win_p + a * top3_p
        return pd.Series(normalize_within_race(blended.values, df_period), index=df_period.index)
    raise ValueError(f"Unknown prob_source: {prob_source}")
