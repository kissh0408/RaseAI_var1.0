"""LightGBM interaction_constraints 用の特徴量グループ割当。"""
from __future__ import annotations

import fnmatch
from typing import Any

# 馬場・適性グループ（先にマッチさせる）
_GOING_SUBSTRINGS = (
    "turf_condition",
    "dirt_condition",
    "turf_cond_",
    "dirt_cond_",
    "going_",
    "current_going_",
    "horse_turf_",
    "horse_dirt_",
    "jockey_heavy",
    "jockey_turf_soft",
    "jockey_dirt_",
    "trainer_turf_heavy",
    "trainer_turf_soft",
    "trainer_dirt_",
    "sire_turf_",
    "sire_dirt_",
    "sire_heavy",
    "sire_soft",
    "dam_sire_turf_",
    "delta_sire_",
    "delta_horse_",
    "delta_jockey_",
    "going_delta_active_score",
    "heavy_track_aptitude",
    "going_heavy_aptitude",
    "going_dirt_heavy_aptitude",
    "going_soft_exp_count",
    "horse_soft_turf_",
    "pace_dist_style_win_rate",
    "speed_index_3run_avg",
    "speed_index_trend",
    "tm_score_surface_adj",
    "daily_track_variant",
)

# コース・距離グループ
_COURSE_SUBSTRINGS = (
    "course_code",
    "distance",
    "track_code",
    "weather_code",
    "surface_code",
    "track_condition_code",
    "course_kubun",
    "n_horses",
    "wakuban",
    "horse_num",
    "gate_number",
    "draw_bias",
    "style_course_bias",
    "youshiba_",
    "kokai_koban_",
    "sin_date",
    "cos_date",
    "is_holiday",
    "dist_bucket",
    "distance_diff",
    "distance_band",
)

# 市場・絶対能力（デフォルト）
_MARKET_SUBSTRINGS = (
    "odds",
    "mining_",
    "tm_score",
    "lag1_odds",
    "market_prob",
    "field_odds_entropy",
    "odds_rank_divergence",
    "popularity",
    "weight_diff",
    "weight_relative",
)


def _match_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(p in name or fnmatch.fnmatch(name, p) for p in patterns)


def assign_feature_group(feature_name: str, config: dict[str, Any] | None = None) -> str:
    """特徴量名を market_ability / going_aptitude / course_distance に割当。"""
    cfg = (config or {}).get("going_improvement", {}).get("feature_group_rules", {})
    for group, patterns in cfg.items():
        if isinstance(patterns, list) and _match_any(feature_name, tuple(patterns)):
            return group

    if _match_any(feature_name, _GOING_SUBSTRINGS):
        return "going_aptitude"
    if _match_any(feature_name, _COURSE_SUBSTRINGS):
        return "course_distance"
    if _match_any(feature_name, _MARKET_SUBSTRINGS):
        return "market_ability"
    return "market_ability"


def build_interaction_constraints(
    feature_names: list[str],
    config: dict[str, Any] | None = None,
) -> list[list[int]] | None:
    """
    LightGBM interaction_constraints を構築する。
    各グループ内の特徴量インデックスリストを返す（グループ間 interaction 禁止）。
    """
    gi = (config or {}).get("going_improvement", {})
    if not gi.get("interaction_constraints_enabled", False):
        return None

    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        g = assign_feature_group(name, config)
        groups.setdefault(g, []).append(idx)

    return [indices for indices in groups.values() if len(indices) >= 1]


def going_feature_names(feature_names: list[str]) -> list[str]:
    return [f for f in feature_names if assign_feature_group(f) == "going_aptitude"]


def build_monotone_constraints(
    feature_names: list[str],
    config: dict[str, Any] | None = None,
) -> list[int] | None:
    """
    LightGBM monotone_constraints を構築する。
    feature_names の各特徴量に対応する +1/0/-1 のリストを返す。

    going_improvement.monotone_constraints_enabled が false の場合は None を返す。
    評価優先順位:
      1. zero_patterns に一致 → 0
      2. plus_patterns に一致 → +1
      3. それ以外 → 0
    -1 は今サイクルでは使用しない（一義的に不利な方向の going 特徴量が特定できないため）。
    """
    gi = (config or {}).get("going_improvement", {})
    if not gi.get("monotone_constraints_enabled", False):
        return None

    plus_patterns = tuple(gi.get("monotone_constraints_plus_patterns", []))
    zero_patterns = tuple(gi.get("monotone_constraints_zero_patterns", []))

    constraints = []
    for name in feature_names:
        if _match_any(name, zero_patterns):
            constraints.append(0)
        elif _match_any(name, plus_patterns):
            constraints.append(1)
        else:
            constraints.append(0)

    # 全て 0 なら None を返す（monotone_constraints を LightGBM に渡さない）
    if all(c == 0 for c in constraints):
        return None

    return constraints


def build_backtest_monotone_constraints(
    feature_names: list[str],
    config: dict[str, Any] | None = None,
) -> list[int] | None:
    """binary 残差学習（train_fold）専用の monotone_constraints を構築する。"""
    bt = (config or {}).get("training", {}).get("backtest_monotone_constraints", {})
    if not bt.get("enabled", False):
        return None

    plus_patterns = tuple(bt.get("plus_patterns", []))
    constraints = [1 if _match_any(name, plus_patterns) else 0 for name in feature_names]
    if all(c == 0 for c in constraints):
        return None
    return constraints
