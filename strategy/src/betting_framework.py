import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import importlib.util

import numpy as np
import pandas as pd
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - scipy未導入環境のフォールバック
    minimize = None

try:
    from common.utils.common_utils import load_standard_csv
except ModuleNotFoundError:
    _common_path = Path(__file__).resolve().parents[2] / "common" / "utils" / "common_utils.py"
    _spec = importlib.util.spec_from_file_location("common_utils", _common_path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(_mod)
    load_standard_csv = _mod.load_standard_csv

try:
    from strategy_engine import (
        OnlineRecommendationConfig,
        _project_to_capped_simplex,
        load_prediction_frame,
        recommend_today,
    )
    from race_filters import attach_race_num, filter_df_by_race_num
    from ev_filters import (
        EvFilterConfig,
        apply_bet_candidate_mask,
        attach_model_rank,
        ev_filter_config_from_mapping,
        post_slippage_edge_gate,
        should_apply_post_slippage_gate,
    )
except ModuleNotFoundError:
    from strategy.src.strategy_engine import (
        OnlineRecommendationConfig,
        _project_to_capped_simplex,
        load_prediction_frame,
        recommend_today,
    )
    from strategy.src.race_filters import attach_race_num, filter_df_by_race_num
    from strategy.src.ev_filters import (
        EvFilterConfig,
        apply_bet_candidate_mask,
        attach_model_rank,
        ev_filter_config_from_mapping,
        post_slippage_edge_gate,
        should_apply_post_slippage_gate,
    )


@dataclass
class StrategyConfig:
    score_col: str = "pred_rank1"
    odds_col: str = "odds"
    race_id_col: str = "race_id"
    horse_col: str = "horse_num"
    rank_col: str = "finish_rank"
    bet_unit: int = 100
    initial_bankroll: int = 100_000
    min_prob: float = 0.01
    ev_threshold: float = 1.05
    min_edge: float = 0.10
    min_odds: float = 2.0
    max_odds: float = 15.0
    max_selections_per_race: int = 2
    sizing_mode: str = "kelly_simultaneous"  # flat | kelly_single | kelly_simultaneous
    fractional_kelly: float = 0.08  # CLAUDE.md固定値: 0.08（変更禁止）
    max_total_fraction: float = 0.30
    max_single_fraction: float = 0.15
    max_stake_per_bet: int = 3_000
    max_invest_per_race: int = 50_000
    base_slippage: float = 0.01
    slippage_mode: str = "fixed"  # fixed | dynamic_pool
    payout_rate: float = 0.8
    assumed_win_pool: float = 10_000_000.0
    market_impact_power: float = 1.0
    enforce_post_slippage_edge: bool = False
    normalize_probs_in_race: bool = True
    isotonic_interpolation: str = "linear"  # linear | step
    simultaneous_optimizer: str = "gradient"  # gradient | scipy
    bootstrap_samples: int = 2000
    random_seed: int = 42
    require_score_rank1: bool = False
    pair_selection_mode: str = "harville"
    max_expected_value: float = 1.5
    race_num_min: Optional[int] = None
    race_num_max: Optional[int] = None
    force_flat_staking: bool = False
    min_win_prob: Optional[float] = None
    max_model_rank: Optional[int] = None
    dynamic_edge_enabled: bool = False
    dynamic_edge_mode: str = "step"
    dynamic_edge_bands: Optional[List[Dict]] = None
    dynamic_edge_alpha: float = 0.02
    dynamic_edge_beta: float = 0.08
    # フィールドサイズフィルタ（OnlineRecommendationConfig と同一デフォルト値を維持する）
    min_field_size: int = 9
    large_field_threshold: int = 18
    large_field_extra_edge: float = 0.05
    # 月間ドローダウン上限（CLAUDE.md 合格基準 -20%。運用値は strategy_config.json の monthly_drawdown_limit）
    # monthly_drawdown_limit=-0.08: 本番パイプラインに統合済み (2026-06-05)
    # main/pipeline/monthly_dd_tracker.py + strategy_pipeline.py で実装
    monthly_drawdown_limit: float = -0.20
    # 分散調整型Kelly: 高オッズ帯のリスクをオッズ比例で縮小する
    dynamic_kelly_enabled: bool = False
    dynamic_kelly_base_fraction: float = 0.08
    dynamic_kelly_odds_ref: float = 3.0
    dynamic_kelly_power: float = 0.5


def to_online_recommendation_config(
    config: StrategyConfig,
    *,
    phase: str = "phase1",
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    recommendation_bankroll: Optional[int] = None,
    phase2_enabled: bool = False,
    save_snapshot_timestamps: bool = True,
    probability_policy: str = "market_shrinkage",
    market_shrinkage_alpha: float = 0.2,
    max_expected_value: float = 1.5,
    max_odds_for_kelly: float = 30.0,
    min_bucket_count: int = 100,
    odds_source: str = "unknown",
    odds_cutoff_policy: str = "unknown",
    require_score_rank1: Optional[bool] = None,
    pair_selection_mode: Optional[str] = None,
    rank2_blend: float = 0.35,
    wide_min_edge: float = 0.05,
    wide_bets_enabled: bool = True,
    quinella_bets_enabled: bool = True,
    place_bets_enabled: bool = True,
    wide_selection: str = "harville",
    wide_ev_threshold: float = 1.05,
    wide_div_threshold: float = 0.0,
    portfolio_kelly_enabled: bool = False,
    portfolio_kelly_mode: str = "portfolio_kelly_fractional",
    portfolio_growth_ratio_min: float = 0.5,
    portfolio_ind_cap_ratio: float = 0.85,
    portfolio_mc_samples: int = 500,
    portfolio_mc_seed: int = 42,
) -> OnlineRecommendationConfig:
    eff_r1 = config.require_score_rank1 if require_score_rank1 is None else require_score_rank1
    eff_pair = config.pair_selection_mode if pair_selection_mode is None else pair_selection_mode
    return OnlineRecommendationConfig(
        score_col=config.score_col,
        race_id_col=config.race_id_col,
        horse_col=config.horse_col,
        odds_col=config.odds_col,
        bet_unit=config.bet_unit,
        min_prob=config.min_prob,
        min_edge=config.min_edge,
        min_odds=config.min_odds,
        max_odds=config.max_odds,
        max_selections_per_race=config.max_selections_per_race,
        fractional_kelly=config.fractional_kelly,
        max_total_fraction=config.max_total_fraction,
        max_single_fraction=config.max_single_fraction,
        max_stake_per_bet=config.max_stake_per_bet,
        max_invest_per_race=config.max_invest_per_race,
        base_slippage=config.base_slippage,
        normalize_probs_in_race=config.normalize_probs_in_race,
        recommendation_bankroll=(
            int(recommendation_bankroll)
            if recommendation_bankroll is not None
            else int(config.initial_bankroll)
        ),
        pair_top_n=int(pair_top_n),
        wide_top_n=int(wide_top_n),
        phase=str(phase),
        phase2_enabled=bool(phase2_enabled),
        save_snapshot_timestamps=bool(save_snapshot_timestamps),
        probability_policy=str(probability_policy),
        market_shrinkage_alpha=float(market_shrinkage_alpha),
        max_expected_value=float(max_expected_value),
        max_odds_for_kelly=float(max_odds_for_kelly),
        min_bucket_count=int(min_bucket_count),
        odds_source=str(odds_source),
        odds_cutoff_policy=str(odds_cutoff_policy),
        require_score_rank1=bool(eff_r1),
        pair_selection_mode=str(eff_pair),
        allow_kelly=not bool(config.force_flat_staking),
        min_win_prob=config.min_win_prob,
        max_model_rank=config.max_model_rank,
        dynamic_edge_enabled=config.dynamic_edge_enabled,
        dynamic_edge_mode=config.dynamic_edge_mode,
        dynamic_edge_bands=config.dynamic_edge_bands,
        dynamic_edge_alpha=config.dynamic_edge_alpha,
        dynamic_edge_beta=config.dynamic_edge_beta,
        min_field_size=config.min_field_size,
        large_field_threshold=config.large_field_threshold,
        large_field_extra_edge=config.large_field_extra_edge,
        rank2_blend=float(rank2_blend),
        wide_min_edge=float(wide_min_edge),
        wide_bets_enabled=bool(wide_bets_enabled),
        quinella_bets_enabled=bool(quinella_bets_enabled),
        place_bets_enabled=bool(place_bets_enabled),
        wide_selection=str(wide_selection),
        wide_ev_threshold=float(wide_ev_threshold),
        wide_div_threshold=float(wide_div_threshold),
        portfolio_kelly_enabled=bool(portfolio_kelly_enabled),
        portfolio_kelly_mode=str(portfolio_kelly_mode),
        portfolio_growth_ratio_min=float(portfolio_growth_ratio_min),
        portfolio_ind_cap_ratio=float(portfolio_ind_cap_ratio),
        portfolio_mc_samples=int(portfolio_mc_samples),
        portfolio_mc_seed=int(portfolio_mc_seed),
        dynamic_kelly_enabled=config.dynamic_kelly_enabled,
        dynamic_kelly_base_fraction=config.dynamic_kelly_base_fraction,
        dynamic_kelly_odds_ref=config.dynamic_kelly_odds_ref,
        dynamic_kelly_power=config.dynamic_kelly_power,
    )


class ProbabilityCalibrator:
    """
    calibration_*.json を読み込み、pred_score を確率に変換する。
    対応:
      - platt: 1 / (1 + exp(A * x + B))
      - isotonic: しきい値配列ベースの単調ステップ関数
    """

    def __init__(self, method: str, params: Dict):
        self.method = method
        self.params = params

    @classmethod
    def from_json(cls, calibration_path: Path) -> "ProbabilityCalibrator":
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        params = payload.get("params", {})
        method = params.get("method", "identity")
        return cls(method=method, params=params)

    def transform(self, scores: pd.Series) -> pd.Series:
        x = scores.astype(float).to_numpy()

        if self.method == "platt":
            a = float(self.params["coef"])
            b = float(self.params["intercept"])
            calibrated = 1.0 / (1.0 + np.exp(a * x + b))
            return pd.Series(calibrated, index=scores.index)

        if self.method == "isotonic":
            x_thr = np.asarray(self.params["x_thresholds"], dtype=float)
            y_thr = np.asarray(self.params["y_thresholds"], dtype=float)
            interpolation = self.params.get("interpolation", "linear")
            xs, ys = _deduplicate_isotonic_points(x_thr, y_thr)
            if interpolation == "step":
                idx = np.searchsorted(xs, x, side="right") - 1
                idx = np.clip(idx, 0, len(ys) - 1)
                calibrated = ys[idx]
            else:
                calibrated = np.interp(x, xs, ys, left=ys[0], right=ys[-1])
            return pd.Series(calibrated, index=scores.index)

        # fallback: スコアをそのまま [0,1] にクリップ
        return scores.astype(float).clip(0.0, 1.0)


def _deduplicate_isotonic_points(x_thr: np.ndarray, y_thr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """同一xが複数あるisotonicしきい値を単調点列へ圧縮する。"""
    order = np.argsort(x_thr)
    xs = x_thr[order]
    ys = y_thr[order]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    unique_y = np.zeros_like(unique_x, dtype=float)
    for i in range(len(unique_x)):
        unique_y[i] = np.mean(ys[inverse == i])
    # 単調性を担保
    unique_y = np.maximum.accumulate(unique_y)
    return unique_x, unique_y


def simultaneous_kelly_fractions(
    probs: np.ndarray,
    odds: np.ndarray,
    fractional_kelly: float,
    total_cap: float,
    n_iter: int = 300,
    lr: float = 0.05,
) -> np.ndarray:
    """
    同一レース内で排他的な複数買い目に対する同時Kelly近似。
    目的関数:
      sum_i p_i * log(1 - F + f_i * o_i) + p0 * log(1 - F)
      F = sum_i f_i, p0 = 1 - sum_i p_i
    """
    if len(probs) == 0:
        return np.array([], dtype=float)

    p = np.clip(probs.astype(float), 0.0, 1.0)
    o = np.clip(odds.astype(float), 1.01, None)
    p = p / max(p.sum(), 1e-12)

    # 初期値: 個別Kellyから開始
    b = o - 1.0
    q = 1.0 - p
    f0 = np.maximum((b * p - q) / np.maximum(b, 1e-12), 0.0) * fractional_kelly
    f = _project_to_capped_simplex(f0, total_cap)

    for _ in range(n_iter):
        f = _project_to_capped_simplex(f, total_cap)
        f_sum = float(f.sum())
        if f_sum >= 0.999:
            f = f * 0.95
            f_sum = float(f.sum())

        a = 1.0 - f_sum + f * o
        if np.any(a <= 1e-10) or (1.0 - f_sum) <= 1e-10:
            f = f * 0.9
            continue

        p0 = max(1.0 - float(p.sum()), 0.0)
        s1 = float(np.sum(p / a))
        grad = p * o / a - s1 - p0 / max(1.0 - f_sum, 1e-12)

        f_new = _project_to_capped_simplex(f + lr * grad, total_cap)
        if np.linalg.norm(f_new - f) < 1e-8:
            f = f_new
            break
        f = f_new

    return np.maximum(f, 0.0)


def simultaneous_kelly_fractions_scipy(
    probs: np.ndarray,
    odds: np.ndarray,
    fractional_kelly: float,
    total_cap: float,
) -> np.ndarray:
    """
    scipy.optimize.minimize(SLSQP) を使った同時Kelly最適化。
    scipy未導入時は勾配法へフォールバックする。
    """
    if len(probs) == 0:
        return np.array([], dtype=float)
    if minimize is None:
        return simultaneous_kelly_fractions(
            probs=probs,
            odds=odds,
            fractional_kelly=fractional_kelly,
            total_cap=total_cap,
        )

    p = np.clip(probs.astype(float), 0.0, 1.0)
    o = np.clip(odds.astype(float), 1.01, None)
    p = p / max(p.sum(), 1e-12)
    n = len(p)

    def objective(f: np.ndarray) -> float:
        f = np.clip(f, 0.0, None)
        f_sum = float(f.sum())
        if f_sum >= 0.999999:
            return 1e9
        a = 1.0 - f_sum + f * o
        if np.any(a <= 1e-12):
            return 1e9
        p0 = max(1.0 - float(p.sum()), 0.0)
        val = float(np.sum(p * np.log(a)) + p0 * np.log(max(1.0 - f_sum, 1e-12)))
        return -val

    x0 = simultaneous_kelly_fractions(
        probs=probs, odds=odds, fractional_kelly=fractional_kelly, total_cap=total_cap
    )
    if len(x0) != n:
        x0 = np.full(n, min(total_cap / max(n, 1), 0.01), dtype=float)

    bounds = [(0.0, total_cap) for _ in range(n)]
    constraints = [{"type": "ineq", "fun": lambda f: total_cap - float(np.sum(f))}]

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-10, "disp": False},
    )
    if not res.success:
        return x0
    return np.maximum(_project_to_capped_simplex(res.x, total_cap), 0.0)


def _single_kelly_fraction(prob: float, odds: float) -> float:
    b = max(odds - 1.0, 1e-12)
    q = 1.0 - prob
    return max((b * prob - q) / b, 0.0)


def _apply_fixed_slippage(odds: pd.Series, base_slippage: float) -> pd.Series:
    adjusted = odds.astype(float) * (1.0 - base_slippage)
    return adjusted.clip(lower=1.01)


def _apply_dynamic_pool_odds(
    raw_odds: float,
    stake: float,
    payout_rate: float,
    assumed_win_pool: float,
    impact_power: float,
    base_slippage: float,
) -> float:
    """
    賭け金を加味した簡易パリミュチュエル実行オッズを計算する。
    Oeff = ((W + Q) * R) / (w_i + Q) を基礎に、impact_powerで調整。
    """
    raw = max(float(raw_odds), 1.01)
    q = max(float(stake), 0.0)
    w = max(float(assumed_win_pool), 1.0)
    r = float(np.clip(payout_rate, 0.01, 1.0))
    alpha = max(float(impact_power), 0.0)

    # 表示オッズ raw はおおむね (W * R) / w_i とみなし逆算
    horse_pool = max((w * r) / raw, 1.0)
    impacted = ((w + q) * r) / (horse_pool + q)
    impacted = raw * ((impacted / raw) ** alpha) if alpha > 0 else raw
    impacted = impacted * (1.0 - base_slippage)
    return max(float(impacted), 1.01)


def load_evaluation(evaluation_path: Path) -> pd.DataFrame:
    df = load_standard_csv(evaluation_path)
    required = {
        "race_id",
        "horse_num",
        "odds",
        "finish_rank",
        "pred_rank1",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluation.csv に必要カラムがありません: {sorted(missing)}")
    n_before = len(df)
    df = df.dropna(subset=["race_id", "horse_num"])
    dropped_na = n_before - len(df)
    if dropped_na > 0:
        warnings.warn(f"[load_evaluation] dropped {dropped_na} rows with null race_id/horse_num.")

    dup_mask = df.duplicated(subset=["race_id", "horse_num"], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        warnings.warn(f"[load_evaluation] dropped {dup_count} duplicated (race_id, horse_num) rows.")
        df = df.loc[~dup_mask].copy()

    for col in ["odds", "finish_rank", "pred_rank1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_numeric = int(df[["odds", "finish_rank", "pred_rank1"]].isna().any(axis=1).sum())
    if bad_numeric > 0:
        warnings.warn(
            f"[load_evaluation] dropped {bad_numeric} rows with invalid numeric values in odds/finish_rank/pred_rank1."
        )
        df = df.dropna(subset=["odds", "finish_rank", "pred_rank1"]).copy()

    df["race_id"] = df["race_id"].astype(str)
    df["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce")
    df = df.dropna(subset=["horse_num"]).copy()
    df["horse_num"] = df["horse_num"].astype(int)
    return df


def run_today_recommendation(
    pred_df: pd.DataFrame,
    *,
    config: StrategyConfig,
    calibrator: Optional[ProbabilityCalibrator] = None,
    phase: str = "phase1",
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    recommendation_bankroll: Optional[int] = None,
    phase2_enabled: bool = False,
    save_snapshot_timestamps: bool = True,
    probability_policy: str = "market_shrinkage",
    market_shrinkage_alpha: float = 0.2,
    market_bias_correction_enabled: bool = False,
    market_bias_correction_model: str = "isotonic",
    max_expected_value: float = 1.5,
    max_odds_for_kelly: float = 30.0,
    min_bucket_count: int = 100,
    odds_source: str = "unknown",
    odds_cutoff_policy: str = "unknown",
    require_score_rank1: Optional[bool] = None,
    pair_selection_mode: Optional[str] = None,
    rank2_blend: float = 0.35,
    generated_at: Optional[str] = None,
    quinella_odds_dict: Optional[dict] = None,
    wide_odds_dict: Optional[dict] = None,
    wide_min_edge: float = 0.05,
    wide_bets_enabled: bool = True,
    quinella_bets_enabled: bool = True,
    place_bets_enabled: bool = True,
    wide_selection: str = "harville",
    wide_ev_threshold: float = 1.05,
    wide_div_threshold: float = 0.0,
    portfolio_kelly_enabled: bool = False,
    portfolio_kelly_mode: str = "portfolio_kelly_fractional",
    portfolio_growth_ratio_min: float = 0.5,
    portfolio_ind_cap_ratio: float = 0.85,
    portfolio_mc_samples: int = 500,
    portfolio_mc_seed: int = 42,
) -> pd.DataFrame:
    online_cfg = to_online_recommendation_config(
        config,
        phase=phase,
        pair_top_n=pair_top_n,
        wide_top_n=wide_top_n,
        recommendation_bankroll=recommendation_bankroll,
        phase2_enabled=phase2_enabled,
        save_snapshot_timestamps=save_snapshot_timestamps,
        probability_policy=probability_policy,
        market_shrinkage_alpha=market_shrinkage_alpha,
        max_expected_value=max_expected_value,
        max_odds_for_kelly=max_odds_for_kelly,
        min_bucket_count=min_bucket_count,
        odds_source=odds_source,
        odds_cutoff_policy=odds_cutoff_policy,
        require_score_rank1=require_score_rank1,
        pair_selection_mode=pair_selection_mode,
        rank2_blend=rank2_blend,
        wide_min_edge=wide_min_edge,
        wide_bets_enabled=wide_bets_enabled,
        quinella_bets_enabled=quinella_bets_enabled,
        place_bets_enabled=place_bets_enabled,
        wide_selection=wide_selection,
        wide_ev_threshold=wide_ev_threshold,
        wide_div_threshold=wide_div_threshold,
        portfolio_kelly_enabled=portfolio_kelly_enabled,
        portfolio_kelly_mode=portfolio_kelly_mode,
        portfolio_growth_ratio_min=portfolio_growth_ratio_min,
        portfolio_ind_cap_ratio=portfolio_ind_cap_ratio,
        portfolio_mc_samples=portfolio_mc_samples,
        portfolio_mc_seed=portfolio_mc_seed,
    )
    online_cfg.market_bias_correction_enabled = market_bias_correction_enabled
    online_cfg.market_bias_correction_model = market_bias_correction_model

    # FLB補正モデルのロード（market_bias_correction_enabled=True の場合のみ）
    _flb_corrector = None
    if market_bias_correction_enabled:
        try:
            from strategy.src.market_bias_corrector import load_market_bias_corrector
        except ModuleNotFoundError:
            from market_bias_corrector import load_market_bias_corrector
        _flb_corrector = load_market_bias_corrector()

    try:
        return recommend_today(
            pred_df=pred_df,
            config=online_cfg,
            calibrator=calibrator,
            generated_at=generated_at,
            quinella_odds_dict=quinella_odds_dict,
            wide_odds_dict=wide_odds_dict,
            _flb_corrector=_flb_corrector,
        )
    except NotImplementedError as e:
        warnings.warn(f"[run_today_recommendation] {e}")
        return pd.DataFrame(
            columns=[
                "ticket_type",
                "race_id",
                "ticket",
                "pred_prob",
                "odds_raw",
                "odds_effective",
                "expected_value",
                "edge",
                "suggested_stake",
                "phase",
                "modeling_note",
            ]
        )


def load_today_prediction_frame(
    csv_path: Path,
    *,
    parquet_path: Optional[Path] = None,
    prefer_parquet: bool = True,
    use_cudf: bool = False,
) -> pd.DataFrame:
    return load_prediction_frame(
        csv_path=csv_path,
        parquet_path=parquet_path,
        prefer_parquet=prefer_parquet,
        use_cudf=use_cudf,
    )


def run_betting_backtest(
    eval_df: pd.DataFrame,
    config: StrategyConfig,
    calibrator: Optional[ProbabilityCalibrator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = filter_df_by_race_num(
        eval_df,
        race_id_col=config.race_id_col,
        race_num_min=config.race_num_min,
        race_num_max=config.race_num_max,
    )
    sort_candidates = [c for c in ["date", "year", "month_day", config.race_id_col, config.horse_col] if c in df.columns]
    if sort_candidates:
        df = df.sort_values(sort_candidates).reset_index(drop=True)

    effective_config = config
    if config.force_flat_staking:
        effective_config = StrategyConfig(**{**asdict(config), "sizing_mode": "flat"})

    if calibrator is not None:
        if calibrator.method == "isotonic":
            calibrator.params = dict(calibrator.params)
            calibrator.params["interpolation"] = config.isotonic_interpolation
        df["pred_prob"] = calibrator.transform(df[config.score_col]).clip(0.0, 1.0)
    else:
        df["pred_prob"] = df[config.score_col].clip(0.0, 1.0)

    if config.normalize_probs_in_race:
        grp_sum = df.groupby(config.race_id_col)["pred_prob"].transform("sum").clip(lower=1e-12)
        df["pred_prob"] = df["pred_prob"] / grp_sum

    # 候補抽出時は固定スリッページで一次評価し、約定時に動的スリッページを適用可能
    df["pred_score_raw"] = pd.to_numeric(df[config.score_col], errors="coerce").fillna(0.0)
    df = attach_model_rank(
        df,
        race_id_col=config.race_id_col,
        prob_col="pred_prob",
        score_col="pred_score_raw",
    )
    df["effective_odds"] = _apply_fixed_slippage(df[config.odds_col], config.base_slippage)
    df["expected_value"] = df["pred_prob"] * df["effective_odds"]
    df["edge"] = df["expected_value"] - 1.0
    ev_cfg = ev_filter_config_from_mapping(effective_config)

    bankroll = float(config.initial_bankroll)
    bet_rows: List[Dict] = []
    race_rows: List[Dict] = []

    for race_id, race in df.groupby(config.race_id_col, sort=False):
        race = race.copy()
        before_bankroll = bankroll

        mask = apply_bet_candidate_mask(race, ev_cfg, odds_col="effective_odds")
        cand = race.loc[mask].sort_values("edge", ascending=False)
        cand = cand.head(effective_config.max_selections_per_race)

        if cand.empty:
            race_rows.append(
                {
                    "race_id": race_id,
                    "n_bets": 0,
                    "invest": 0.0,
                    "return": 0.0,
                    "profit": 0.0,
                    "bankroll_before": before_bankroll,
                    "bankroll_after": bankroll,
                }
            )
            continue

        if effective_config.sizing_mode == "flat":
            fractions = np.array(
                [effective_config.bet_unit / max(before_bankroll, 1.0)] * len(cand), dtype=float
            )
            fractions = _project_to_capped_simplex(fractions, effective_config.max_total_fraction)
        elif effective_config.sizing_mode == "kelly_single":
            # 分散調整型Kelly: オッズに応じてkelly係数を動的縮小
            _dyn_kelly_bf = getattr(effective_config, "dynamic_kelly_enabled", False)
            raw_list = []
            for p, o in zip(cand["pred_prob"], cand["effective_odds"]):
                if _dyn_kelly_bf:
                    try:
                        from strategy.src.strategy_engine import get_dynamic_kelly_fraction
                    except ModuleNotFoundError:
                        from strategy_engine import get_dynamic_kelly_fraction
                    fk = get_dynamic_kelly_fraction(
                        float(o),
                        base_fraction=float(getattr(effective_config, "dynamic_kelly_base_fraction", effective_config.fractional_kelly)),
                        odds_ref=float(getattr(effective_config, "dynamic_kelly_odds_ref", 3.0)),
                        power=float(getattr(effective_config, "dynamic_kelly_power", 0.5)),
                    )
                else:
                    fk = effective_config.fractional_kelly
                raw_list.append(_single_kelly_fraction(p, o) * fk)
            raw = np.array(raw_list, dtype=float)
            fractions = np.clip(
                raw,
                0.0,
                effective_config.max_single_fraction,
            )
            fractions = _project_to_capped_simplex(fractions, effective_config.max_total_fraction)
        elif effective_config.sizing_mode == "kelly_simultaneous":
            if effective_config.simultaneous_optimizer == "scipy":
                fractions = simultaneous_kelly_fractions_scipy(
                    probs=cand["pred_prob"].to_numpy(),
                    odds=cand["effective_odds"].to_numpy(),
                    fractional_kelly=effective_config.fractional_kelly,
                    total_cap=effective_config.max_total_fraction,
                )
            else:
                fractions = simultaneous_kelly_fractions(
                    probs=cand["pred_prob"].to_numpy(),
                    odds=cand["effective_odds"].to_numpy(),
                    fractional_kelly=effective_config.fractional_kelly,
                    total_cap=effective_config.max_total_fraction,
                )
            fractions = np.clip(fractions, 0.0, effective_config.max_single_fraction)
            fractions = _project_to_capped_simplex(fractions, effective_config.max_total_fraction)
        else:
            raise ValueError(f"未知の sizing_mode: {effective_config.sizing_mode}")

        race_invest = 0.0
        race_return = 0.0
        race_bet_count = 0
        winners = set(race.loc[race[config.rank_col] == 1, config.horse_col].astype(int).tolist())

        for (_, row), frac in zip(cand.iterrows(), fractions):
            target_bet = before_bankroll * float(frac)
            remaining_budget = max(config.max_invest_per_race - race_invest, 0.0)
            capped_target = min(target_bet, config.max_stake_per_bet, remaining_budget)
            stake = int(capped_target // config.bet_unit) * config.bet_unit
            if stake < config.bet_unit:
                continue

            if config.slippage_mode == "dynamic_pool":
                exec_odds = _apply_dynamic_pool_odds(
                    raw_odds=float(row[config.odds_col]),
                    stake=stake,
                    payout_rate=config.payout_rate,
                    assumed_win_pool=config.assumed_win_pool,
                    impact_power=config.market_impact_power,
                    base_slippage=config.base_slippage,
                )
            else:
                exec_odds = float(row["effective_odds"])

            exec_ev = float(row["pred_prob"]) * exec_odds
            exec_edge = exec_ev - 1.0
            if should_apply_post_slippage_gate(
                ev_cfg, enforce_post_slippage_edge=config.enforce_post_slippage_edge
            ):
                if not post_slippage_edge_gate(float(row["pred_prob"]), exec_odds, ev_cfg):
                    continue

            horse_num = int(row[config.horse_col])
            hit = horse_num in winners
            payout = float(stake * exec_odds) if hit else 0.0
            profit = payout - stake

            race_invest += stake
            race_return += payout
            race_bet_count += 1

            bet_rows.append(
                {
                    "race_id": race_id,
                    "horse_num": horse_num,
                    "stake": stake,
                    "odds_raw": float(row[config.odds_col]),
                    "odds_effective": exec_odds,
                    "pred_prob": float(row["pred_prob"]),
                    "expected_value": exec_ev,
                    "edge": exec_edge,
                    "kelly_fraction": float(frac),
                    "hit": int(hit),
                    "payout": payout,
                    "profit": profit,
                }
            )

        race_profit = race_return - race_invest
        bankroll += race_profit
        bankroll = max(bankroll, 0.0)

        race_rows.append(
            {
                "race_id": race_id,
                "n_bets": race_bet_count,
                "invest": race_invest,
                "return": race_return,
                "profit": race_profit,
                "bankroll_before": before_bankroll,
                "bankroll_after": bankroll,
            }
        )

    bets_df = pd.DataFrame(bet_rows)
    race_df = pd.DataFrame(race_rows)
    metrics = compute_metrics(
        bets_df=bets_df,
        race_df=race_df,
        initial_bankroll=config.initial_bankroll,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
    )
    return bets_df, race_df, metrics


def _bootstrap_roi_p_value(
    race_returns: pd.Series,
    bootstrap_samples: int,
    random_seed: int,
) -> float:
    """
    H0: 期待レースリターンの平均は0以下。
    中心化ブートストラップで one-sided p-value を推定する。
    """
    x = race_returns.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return 1.0
    observed = float(np.mean(x))
    if observed <= 0:
        return 1.0
    centered = x - observed
    rng = np.random.default_rng(random_seed)
    sims = rng.choice(centered, size=(bootstrap_samples, len(centered)), replace=True).mean(axis=1)
    p_value = float(np.mean(sims >= observed))
    return max(min(p_value, 1.0), 0.0)


def compute_metrics(
    bets_df: pd.DataFrame,
    race_df: pd.DataFrame,
    initial_bankroll: float,
    bootstrap_samples: int = 2000,
    random_seed: int = 42,
) -> Dict:
    if race_df.empty:
        return {
            "n_races": 0,
            "n_races_bet": 0,
            "n_bets": 0,
            "invest": 0.0,
            "return": 0.0,
            "net_profit": 0.0,
            "roi": 0.0,
            "return_multiple": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            # L3: 比率換算ドローダウン（CLAUDE.md 合格基準 -20% 以内）
            "max_drawdown_rate": 0.0,
            # L2: CLAUDE.md 定義 mean(r)/std(r) （合格基準 0.10 以上）
            "sharpe": 0.0,
            "recovery_factor": 0.0,
            "expected_gain_per_race": 0.0,
            "expected_gain_per_bet": 0.0,
            "roi_p_value": 1.0,
            "edge_significant_5pct": False,
        }

    n_races = int(len(race_df))
    n_races_bet = int((race_df["invest"] > 0).sum())
    n_bets = int(len(bets_df))
    invest = float(race_df["invest"].sum())
    returned = float(race_df["return"].sum())
    net = returned - invest
    roi = (net / invest) if invest > 0 else 0.0
    return_multiple = (returned / invest) if invest > 0 else 0.0

    if n_bets > 0:
        gross_profit = float(bets_df.loc[bets_df["profit"] > 0, "profit"].sum())
        gross_loss = float(bets_df.loc[bets_df["profit"] < 0, "profit"].sum())
    else:
        gross_profit = 0.0
        gross_loss = 0.0

    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")

    equity = pd.Series(
        [initial_bankroll] + race_df["profit"].cumsum().add(initial_bankroll).tolist()
    )
    running_max = equity.cummax()
    drawdown = equity - running_max
    mdd = float(drawdown.min())

    # L3: 比率換算ドローダウン = 絶対ドローダウン / ピーク資産額
    # ピーク資産額ゼロ除算を防ぐため running_max を 1.0 以上にクリップする
    drawdown_rate = drawdown / running_max.clip(lower=1.0)
    mdd_rate = float(drawdown_rate.min())

    # L2: CLAUDE.md 準拠 Sharpe = mean(profit/invest) / std(profit/invest)（√n 不使用）
    # bets_df 単位ではなくレース単位の profit/invest で計算（バックテスト集計との整合）
    race_returns = race_df.loc[race_df["invest"] > 0, "profit"] / race_df.loc[race_df["invest"] > 0, "invest"]
    if len(race_returns) >= 2 and race_returns.std(ddof=1) > 0:
        sharpe = float(race_returns.mean() / race_returns.std(ddof=1))
    else:
        sharpe = 0.0

    rf = net / abs(mdd) if mdd < 0 else 0.0
    expected_gain_per_race = net / n_races_bet if n_races_bet > 0 else 0.0
    expected_gain_per_bet = net / n_bets if n_bets > 0 else 0.0
    roi_p_value = _bootstrap_roi_p_value(
        race_returns=race_returns,
        bootstrap_samples=max(int(bootstrap_samples), 100),
        random_seed=int(random_seed),
    )

    hit_rate = float(bets_df["hit"].mean()) if n_bets > 0 else 0.0

    return {
        "n_races": n_races,
        "n_races_bet": n_races_bet,
        "n_bets": n_bets,
        "invest": invest,
        "return": returned,
        "net_profit": net,
        "roi": roi,
        "return_multiple": return_multiple,
        "hit_rate": hit_rate,
        "profit_factor": pf,
        "max_drawdown": mdd,
        # L3: 比率換算 MDD（CLAUDE.md 合格基準 -20% 以内、表示形式 {rate:.1%}）
        "max_drawdown_rate": mdd_rate,
        # L2: CLAUDE.md 準拠 Sharpe（合格 0.10 以上、要改善 0.05〜0.10）
        "sharpe": sharpe,
        "recovery_factor": rf,
        "expected_gain_per_race": expected_gain_per_race,
        "expected_gain_per_bet": expected_gain_per_bet,
        "roi_p_value": roi_p_value,
        "edge_significant_5pct": bool(roi_p_value < 0.05),
    }


def create_markdown_report(
    config: StrategyConfig,
    metrics: Dict,
    calibration_path: Optional[Path],
) -> str:
    cal_name = str(calibration_path) if calibration_path else "None (raw score)"
    lines = [
        "# Betting Framework Backtest Report",
        "",
        "## 1. Run Configuration",
        f"- Calibration model: `{cal_name}`",
        f"- Sizing mode: `{config.sizing_mode}`",
        f"- Fractional Kelly: `{config.fractional_kelly}`",
        f"- Min edge (EV-1): `{config.min_edge}`",
        f"- Min prob: `{config.min_prob}`",
        f"- Odds range: `{config.min_odds}` - `{config.max_odds}`",
        f"- Max selections per race: `{config.max_selections_per_race}`",
        f"- Max stake per bet: `{config.max_stake_per_bet}`",
        f"- Max invest per race: `{config.max_invest_per_race}`",
        f"- Slippage mode: `{config.slippage_mode}`",
        f"- Base slippage: `{config.base_slippage}`",
        f"- Payout rate: `{config.payout_rate}`",
        f"- Assumed win pool: `{config.assumed_win_pool}`",
        f"- Market impact power: `{config.market_impact_power}`",
        f"- Post-slippage edge check: `{config.enforce_post_slippage_edge}`",
        f"- Isotonic interpolation: `{config.isotonic_interpolation}`",
        f"- Simultaneous optimizer: `{config.simultaneous_optimizer}`",
        f"- Bootstrap samples: `{config.bootstrap_samples}`",
        "",
        "## 2. Performance Metrics",
    ]
    for k, v in metrics.items():
        if k == "max_drawdown_rate" and isinstance(v, float):
            # L3: 比率ドローダウンは % 形式で表示（CLAUDE.md 合格基準 -20% 以内）
            grade = "OK" if v >= -0.20 else ("要改善" if v >= -0.30 else "REJECT")
            lines.append(f"- {k}: `{v:.1%}` [{grade}]")
        elif k == "sharpe" and isinstance(v, float):
            # L2: Sharpe の合格判定を付記（合格 0.10 以上）
            grade = "OK" if v >= 0.10 else ("要改善" if v >= 0.05 else "REJECT")
            lines.append(f"- {k}: `{v:.6f}` [{grade}]")
        elif isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")

    lines.extend(
        [
            "",
            "## 3. Interpretation Guide",
            "- `profit_factor > 1.0` を最低条件、`1.3` 以上を実運用候補とする。",
            "- `max_drawdown` と `recovery_factor` をセットで見て、資金曲線の健全性を判断する。",
            "- `fractional_kelly` は 0.08 を運用値とする（CLAUDE.md固定値。変遷: 0.10→0.08）。",
        ]
    )
    return "\n".join(lines)


def run_combo_betting_backtest(
    eval_df: pd.DataFrame,
    config: StrategyConfig,
    odds_dir: Optional[Union[str, Path]] = None,
    *,
    pair_top_n: int = 2,
    wide_top_n: int = 2,
    rank2_blend: float = 0.35,
) -> Tuple[pd.DataFrame, Dict]:
    """
    馬連・ワイドバックテストのエントリーポイント。

    combo_backtest.run_combo_backtest をラップし、StrategyConfig の設定値を引き渡す。
    odds_dir が None のとき PROJECT_ROOT/common/data/output/odds をデフォルトとして使用する。
    """
    if odds_dir is None:
        # プロジェクトルートを betting_framework.py の位置から推定する
        _here = Path(__file__).resolve()
        # strategy/src/betting_framework.py → 2段上がりがプロジェクトルート
        _project_root = _here.parents[2]
        odds_dir = _project_root / "common" / "data" / "output" / "odds"

    try:
        from strategy.src.combo_backtest import run_combo_backtest as _run_cb
    except ModuleNotFoundError:
        try:
            from combo_backtest import run_combo_backtest as _run_cb  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ImportError(
                "[run_combo_betting_backtest] combo_backtest モジュールが見つかりません。"
                " strategy/src/combo_backtest.py が存在するか確認してください。"
            ) from exc

    return _run_cb(
        eval_df=eval_df,
        config=config,
        odds_dir=odds_dir,
        pair_top_n=pair_top_n,
        wide_top_n=wide_top_n,
        rank2_blend=rank2_blend,
    )


def save_outputs(
    output_dir: Path,
    report_dir: Path,
    config: StrategyConfig,
    metrics: Dict,
    bets_df: pd.DataFrame,
    race_df: pd.DataFrame,
    calibration_path: Optional[Path],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    bets_path = output_dir / "betting_framework_bets.csv"
    race_path = output_dir / "betting_framework_race_summary.csv"
    metrics_path = output_dir / "betting_framework_metrics.json"
    report_path = report_dir / "betting_framework_backtest_report.md"

    bets_df.to_csv(bets_path, index=False, encoding="utf-8-sig")
    race_df.to_csv(race_path, index=False, encoding="utf-8-sig")

    payload = {
        "config": asdict(config),
        "metrics": metrics,
        "calibration_path": str(calibration_path) if calibration_path else None,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = create_markdown_report(config=config, metrics=metrics, calibration_path=calibration_path)
    report_path.write_text(md, encoding="utf-8")

    return {
        "bets_path": bets_path,
        "race_path": race_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
    }
