"""バックテスト評価指標: ROI・MDD・Sharpe・過学習検知・Residual IC。

backtest-evaluatorフェーズ。
CLAUDE.md合格基準: ROI>=1.05, MDD>=-0.20, Sharpe>=0.10, n_bets>=500
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 回収率・収支
# ---------------------------------------------------------------------------

def calculate_roi_metrics(df: pd.DataFrame) -> dict:
    """推奨馬券のみを対象に回収率を計算する。

    必須カラム: is_recommended, is_win, odds, kelly_bet_yen, model_prob
    """
    recs = df[df["is_recommended"]].copy()

    if len(recs) == 0:
        return {"error": "推奨なし - EV閾値が高すぎる可能性"}

    recs["payout"] = recs["is_win"] * recs["odds"] * recs["kelly_bet_yen"]
    recs["profit"] = recs["payout"] - recs["kelly_bet_yen"]

    total_bet = recs["kelly_bet_yen"].sum()
    total_payout = recs["payout"].sum()
    roi = total_payout / total_bet if total_bet > 0 else 0.0

    n_bets = len(recs)
    hit_rate = recs["is_win"].mean()
    ci_low, ci_high = wilson_ci(hit_rate, n_bets)
    avg_odds = float(recs["odds"].mean())
    implied_prob = 1.0 / max(avg_odds, 1.01)
    model_prob_mean = recs["model_prob"].mean()

    return {
        "n_bets": n_bets,
        "hit_rate": float(hit_rate),
        "hit_rate_ci_95": (float(ci_low), float(ci_high)),
        "roi": float(roi),
        "total_profit_yen": float(recs["profit"].sum()),
        "avg_odds": float(avg_odds),
        "model_prob_mean": float(model_prob_mean),
        "implied_prob": float(implied_prob),
        "model_edge": float(model_prob_mean - (1.0 / avg_odds)) if avg_odds > 0 else 0.0,
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson信頼区間（二項比率の区間推定）。"""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# 戦略なし順位指標（model_prob ベース）
# ---------------------------------------------------------------------------

def _finish_in_top2(fr: float) -> bool:
    return 1 <= fr <= 2


def _finish_in_top3(fr: float) -> bool:
    return 1 <= fr <= 3


def calculate_ranking_metrics(
    df: pd.DataFrame,
    prob_col: str = "model_prob",
    finish_rank_col: str = "finish_rank",
    race_id_col: str = "race_id",
) -> dict:
    """各レースで prob 上位3頭の3着以内率・1位軸ペア的中率など（EV/Kelly 非依存）。"""
    empty = {
        "n_races": 0,
        "top3_overlap_rate": float("nan"),
        "top1_win_rate": float("nan"),
        "top3_box_rate": float("nan"),
        "quinella_anchor_12": float("nan"),
        "quinella_anchor_13": float("nan"),
        "quinella_anchor_any": float("nan"),
        "wide_anchor_12": float("nan"),
        "wide_anchor_13": float("nan"),
        "wide_anchor_any": float("nan"),
        "top1_in_top3": float("nan"),
    }
    if len(df) == 0 or prob_col not in df.columns or finish_rank_col not in df.columns:
        return empty

    overlap_rates: list[float] = []
    top1_wins: list[int] = []
    top3_boxes: list[int] = []
    q12: list[int] = []
    q13: list[int] = []
    q_any: list[int] = []
    w12: list[int] = []
    w13: list[int] = []
    w_any: list[int] = []
    top1_in3: list[int] = []
    n_races = 0

    for _, g in df.groupby(race_id_col, sort=False):
        if len(g) < 3:
            continue
        n_races += 1
        ranked = g.nlargest(3, prob_col)
        fr = pd.to_numeric(ranked[finish_rank_col], errors="coerce")
        in3 = (fr >= 1) & (fr <= 3)
        overlap_rates.append(float(in3.sum()) / 3.0)
        fr1 = pd.to_numeric(g.nlargest(1, prob_col)[finish_rank_col], errors="coerce").iloc[0]
        top1_wins.append(1 if fr1 == 1 else 0)
        top3_boxes.append(1 if in3.all() else 0)

        fr1f, fr2f, fr3f = float(fr.iloc[0]), float(fr.iloc[1]), float(fr.iloc[2])
        top1_in3.append(1 if _finish_in_top3(fr1f) else 0)
        hit_q12 = _finish_in_top2(fr1f) and _finish_in_top2(fr2f)
        hit_q13 = _finish_in_top2(fr1f) and _finish_in_top2(fr3f)
        q12.append(1 if hit_q12 else 0)
        q13.append(1 if hit_q13 else 0)
        q_any.append(1 if (hit_q12 or hit_q13) else 0)
        hit_w12 = _finish_in_top3(fr1f) and _finish_in_top3(fr2f)
        hit_w13 = _finish_in_top3(fr1f) and _finish_in_top3(fr3f)
        w12.append(1 if hit_w12 else 0)
        w13.append(1 if hit_w13 else 0)
        w_any.append(1 if (hit_w12 or hit_w13) else 0)

    def _mean(xs: list[int]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_races": n_races,
        "top3_overlap_rate": float(np.mean(overlap_rates)) if overlap_rates else float("nan"),
        "top1_win_rate": _mean(top1_wins),
        "top3_box_rate": _mean(top3_boxes),
        "quinella_anchor_12": _mean(q12),
        "quinella_anchor_13": _mean(q13),
        "quinella_anchor_any": _mean(q_any),
        "wide_anchor_12": _mean(w12),
        "wide_anchor_13": _mean(w13),
        "wide_anchor_any": _mean(w_any),
        "top1_in_top3": _mean(top1_in3),
    }


def detect_ranking_overfitting(
    train_metrics: dict,
    valid_metrics: dict,
    test_metrics: dict,
) -> dict:
    """train/valid/test の top3_overlap 差で順位過学習を検知。"""
    warnings: list[str] = []

    def _get(m: dict, key: str) -> float:
        return float(m.get(key, float("nan")))

    train_o = _get(train_metrics, "top3_overlap_rate")
    valid_o = _get(valid_metrics, "top3_overlap_rate")
    test_o = _get(test_metrics, "top3_overlap_rate")

    gap_tv = train_o - valid_o if np.isfinite(train_o) and np.isfinite(valid_o) else float("nan")
    gap_vt = valid_o - test_o if np.isfinite(valid_o) and np.isfinite(test_o) else float("nan")

    if np.isfinite(gap_tv) and gap_tv > 0.005:
        warnings.append(
            f"順位過学習疑い: Train overlap {train_o:.3f} vs Valid {valid_o:.3f} (差: {gap_tv:.3f})"
        )
    if np.isfinite(gap_vt) and gap_vt > 0.005:
        warnings.append(
            f"Valid/Test乖離: Valid overlap {valid_o:.3f} vs Test {test_o:.3f} (差: {gap_vt:.3f})"
        )

    return {
        "is_healthy": len(warnings) == 0,
        "warnings": warnings,
        "overlap_gap_train_valid": float(gap_tv) if np.isfinite(gap_tv) else float("nan"),
        "overlap_gap_valid_test": float(gap_vt) if np.isfinite(gap_vt) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 最大ドローダウン・Sharpe
# ---------------------------------------------------------------------------

def calculate_drawdown(df: pd.DataFrame) -> dict:
    """時系列での資金推移から最大ドローダウンを計算する。

    df は race_date でソート済みかつ is_recommended==True のみを想定。
    """
    recs = df[df["is_recommended"]].sort_values("race_date").copy()
    if len(recs) == 0:
        return {"max_drawdown_rate": 0.0, "sharpe_ratio": 0.0}

    initial_bankroll = 100_000.0
    # バンクロールが0以下になったらベットしない（破産シミュレーション）
    # kelly_bet_yen は apply_kelly_sizing の bankroll に依存するため、
    # 比率 kelly_ratio を使ってシミュレーション用 bankroll でスケーリングする。
    use_ratio = "kelly_ratio" in recs.columns
    bankroll = initial_bankroll
    bankroll_series = []
    profit_series = []
    invest_series = []
    for _, row in recs.iterrows():
        if bankroll <= 0:
            bet = 0.0
        elif use_ratio:
            bet = bankroll * float(row["kelly_ratio"])
        else:
            bet = min(float(row["kelly_bet_yen"]), bankroll)
        payout = float(row["is_win"]) * float(row["odds"]) * bet
        profit = payout - bet
        bankroll = max(0.0, bankroll + profit)
        bankroll_series.append(bankroll)
        profit_series.append(profit)
        invest_series.append(bet)
    recs["bankroll"] = bankroll_series
    recs["profit"] = profit_series
    recs["invest"] = invest_series
    # 初期資金もピークに含める（最初のベットからの連敗ドローダウンを取りこぼさないため）
    recs["peak"] = recs["bankroll"].cummax().clip(lower=initial_bankroll)
    recs["drawdown"] = (recs["bankroll"] - recs["peak"]) / recs["peak"].clip(lower=1.0)

    max_dd = float(recs["drawdown"].min())
    max_dd_date = recs.loc[recs["drawdown"].idxmin(), "race_date"]

    in_dd = recs["drawdown"] < -0.01
    if in_dd.any():
        dd_streaks = (in_dd != in_dd.shift()).cumsum()[in_dd]
        max_dd_duration = int(dd_streaks.value_counts().max())
    else:
        max_dd_duration = 0

    # CLAUDE.md: Sharpe = mean(profit/invest) / std(profit/invest)（等重み・ベット単位）
    recs["return_on_invest"] = recs["profit"] / recs["invest"].replace(0, np.nan)
    sharpe = calculate_sharpe(recs["return_on_invest"].dropna())

    return {
        "max_drawdown_rate": max_dd,
        "max_drawdown_date": str(max_dd_date),
        "max_dd_duration_races": max_dd_duration,
        "final_bankroll": float(recs["bankroll"].iloc[-1]),
        "final_roi_vs_initial": float(recs["bankroll"].iloc[-1] / initial_bankroll - 1),
        "sharpe_ratio": sharpe,
    }


def calculate_sharpe(returns: pd.Series) -> float:
    """等重みSharpeレシオ（平均/標準偏差）。"""
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def calculate_sharpe_weighted(returns: pd.Series, weights: pd.Series) -> float:
    """Kelly加重Sharpeレシオ。

    Kelly比率が高い（=エッジが大きい）ベットほど大きな重みを与える。
    CLAUDE.md: Sharpe = mean(profit/invest) / std(profit/invest)（レース単位）。
    単勝二値分布の理論上限は ~0.20。0.10以上が合格。
    """
    if len(returns) == 0:
        return 0.0
    w = weights.clip(lower=0.0)
    w_sum = w.sum()
    if w_sum == 0:
        return calculate_sharpe(returns)
    w_mean = float((returns * w).sum() / w_sum)
    w_var = float(((returns - w_mean) ** 2 * w).sum() / w_sum)
    if w_var == 0:
        return 0.0
    return w_mean / float(w_var ** 0.5)


# ---------------------------------------------------------------------------
# 過学習・データリーク検知
# ---------------------------------------------------------------------------

def detect_overfitting(
    train_metrics: dict,
    valid_metrics: dict,
    test_metrics: dict,
) -> dict:
    """学習/バリデーション/テスト期間のROI差で過学習を検知する。"""
    warnings = []

    if "roi" not in train_metrics or "roi" not in valid_metrics or "roi" not in test_metrics:
        return {"is_healthy": False, "warnings": ["metrics不完全 - 計算エラーの可能性"]}

    roi_gap_tv = train_metrics["roi"] - valid_metrics["roi"]
    roi_gap_vt = valid_metrics["roi"] - test_metrics["roi"]

    if roi_gap_tv > 0.15:
        warnings.append(
            f"過学習疑い: Train ROI {train_metrics['roi']:.3f} vs Valid ROI {valid_metrics['roi']:.3f} (差: {roi_gap_tv:.3f})"
        )

    if roi_gap_vt > 0.15:
        warnings.append(
            f"バリデーション過剰最適化疑い: Valid ROI {valid_metrics['roi']:.3f} vs Test ROI {test_metrics['roi']:.3f} (差: {roi_gap_vt:.3f})"
        )

    if test_metrics.get("n_bets", 0) < 200:
        warnings.append(f"サンプル数不足: {test_metrics.get('n_bets', 0)}件（最低200件推奨）")

    if test_metrics.get("hit_rate", 0) > 0.5 and test_metrics.get("avg_odds", 10) < 3.0:
        warnings.append(
            f"的中率異常: {test_metrics.get('hit_rate', 0):.3f}（低オッズ帯で高的中率 - リーク要確認）"
        )

    return {
        "is_healthy": len(warnings) == 0,
        "warnings": warnings,
        "roi_gap_train_valid": float(roi_gap_tv),
        "roi_gap_valid_test": float(roi_gap_vt),
    }


# ---------------------------------------------------------------------------
# Residual IC（残差情報係数）
# ---------------------------------------------------------------------------

def calculate_residual_ic(
    df: pd.DataFrame,
    model_prob_col: str = "model_prob",
    odds_col: str = "odds",
    win_col: str = "is_win",
    race_id_col: str = "race_id",
) -> dict:
    """残差情報係数: (model_prob - implied_prob_norm) と実勝利の相関。

    model_prob はレース内で sum=1 に正規化済み。
    implied_prob も同様にレース内正規化して比較可能にする
    （生の 1/odds はオーバーラウンド分 sum>1 になるためバイアスが生じる）。

    Residual IC が高い = AIがオッズの誤りを独自に見抜いている証拠。
    IC > 0.02: 弱いエッジあり
    IC > 0.05: 明確なエッジあり
    """
    df = df.copy().dropna(subset=[model_prob_col, odds_col, win_col])
    if len(df) < 30:
        return {"residual_ic": np.nan, "p_value": np.nan, "n_samples": len(df)}

    raw_implied = 1.0 / df[odds_col].clip(lower=1.01)
    # レース内でオーバーラウンド補正（レース内正規化）
    if race_id_col in df.columns:
        race_sum = df.groupby(race_id_col)[odds_col].transform(
            lambda x: (1.0 / x.clip(lower=1.01)).sum()
        )
        implied_prob = raw_implied / race_sum
    else:
        implied_prob = raw_implied
    residual = df[model_prob_col] - implied_prob

    if residual.std() == 0 or df[win_col].std() == 0:
        return {"residual_ic": 0.0, "p_value": 1.0, "n_samples": len(df)}

    try:
        from scipy import stats
        corr, pval = stats.pearsonr(residual, df[win_col])
    except ImportError:
        # scipy なし: 手計算
        x = residual.values - residual.mean()
        y = df[win_col].values - df[win_col].mean()
        denom = np.sqrt((x**2).sum() * (y**2).sum())
        corr = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        pval = float("nan")

    return {
        "residual_ic": float(corr),
        "p_value": float(pval) if not np.isnan(pval) else None,
        "n_samples": len(df),
        "is_significant": (float(pval) < 0.05) if not np.isnan(pval) else False,
        "residual_mean": float(residual.mean()),
        "edge_above_market": float((residual > 0).mean()),
    }


def calculate_log_loss(
    df: pd.DataFrame,
    model_prob_col: str = "model_prob",
    win_col: str = "is_win",
    race_id_col: str = "race_id",
) -> dict:
    """レースグループ単位のLog Lossを計算する。

    Baseline: implied_prob(1/odds)でのLog Lossと比較。
    model_log_loss < baseline_log_loss → モデルが市場より優秀。
    """
    df = df.copy().dropna(subset=[model_prob_col, win_col])

    probs = np.clip(df[model_prob_col].values, 1e-9, 1.0 - 1e-9)
    wins = df[win_col].values.astype(float)

    model_ll = -float(np.mean(wins * np.log(probs) + (1 - wins) * np.log(1 - probs)))

    if "odds" in df.columns and df["odds"].notna().any():
        # オッズ欠損行はベースライン計算から除外（NaN伝播でbase_llがNaNになるのを防ぐ）
        m = df["odds"].notna().values
        implied = np.clip(1.0 / df.loc[m, "odds"].clip(lower=1.01).values, 1e-9, 1.0 - 1e-9)
        w = wins[m]
        base_ll = -float(np.mean(w * np.log(implied) + (1 - w) * np.log(1 - implied)))
    else:
        base_ll = float("nan")

    return {
        "model_log_loss": model_ll,
        "baseline_log_loss": base_ll,
        "log_loss_improvement": (base_ll - model_ll) if not np.isnan(base_ll) else None,
        "n_samples": len(df),
    }


# ---------------------------------------------------------------------------
# 条件別分析
# ---------------------------------------------------------------------------

def analyze_by_conditions(df: pd.DataFrame) -> dict:
    """馬場・距離・人気・頭数など条件別の回収率を分析する。"""
    conditions = {}
    if "track_condition_code" in df.columns:
        conditions["馬場_良"] = df["track_condition_code"] == 1
        conditions["馬場_悪化"] = df["track_condition_code"] >= 2
    if "horse_count" in df.columns:
        conditions["多頭数(15頭以上)"] = df["horse_count"] >= 15
        conditions["少頭数(8頭以下)"] = df["horse_count"] <= 8
    if "surface_code" in df.columns:
        conditions["芝"] = df["surface_code"] == 1
        conditions["ダート"] = df["surface_code"] == 2
    if "distance" in df.columns:
        conditions["短距離(1200m以下)"] = df["distance"] <= 1200
        conditions["長距離(2400m以上)"] = df["distance"] >= 2400

    results = {}
    for label, mask in conditions.items():
        subset = df[mask & df["is_recommended"]]
        if len(subset) >= 20:
            total_bet = subset["kelly_bet_yen"].sum()
            payout = (subset["is_win"] * subset["odds"] * subset["kelly_bet_yen"]).sum()
            roi = payout / total_bet if total_bet > 0 else 0.0
            results[label] = {
                "n_bets": len(subset),
                "roi": float(roi),
                "hit_rate": float(subset["is_win"].mean()),
            }
    return results


# ---------------------------------------------------------------------------
# 合格判定
# ---------------------------------------------------------------------------

PASS_CRITERIA = {
    "roi": {"pass": 1.15, "warn": 1.05, "label": "ROI"},
    "hit_rate": {"pass": 0.22, "warn": 0.20, "label": "的中率"},
    "max_drawdown_rate": {"pass": -0.15, "warn": -0.20, "label": "MDD（符号注意）"},
    "sharpe_ratio": {"pass": 0.12, "warn": 0.10, "label": "Sharpe"},
    "n_bets": {"pass": 300, "warn": 200, "label": "件数"},
}


def grade_metrics(test_metrics: dict, drawdown_metrics: dict) -> dict:
    """CLAUDE.mdの合格基準で総合評価を返す。"""
    combined = {**test_metrics, **drawdown_metrics}
    grades = {}

    roi = combined.get("roi", 0)
    if roi >= PASS_CRITERIA["roi"]["pass"]:
        grades["roi"] = "PASS"
    elif roi >= PASS_CRITERIA["roi"]["warn"]:
        grades["roi"] = "WARN"
    else:
        grades["roi"] = "REJECT"

    hit = combined.get("hit_rate", 0)
    if hit >= PASS_CRITERIA["hit_rate"]["pass"]:
        grades["hit_rate"] = "PASS"
    elif hit >= PASS_CRITERIA["hit_rate"]["warn"]:
        grades["hit_rate"] = "WARN"
    else:
        grades["hit_rate"] = "REJECT"

    mdd = combined.get("max_drawdown_rate", -1)
    if mdd >= PASS_CRITERIA["max_drawdown_rate"]["pass"]:
        grades["mdd"] = "PASS"
    elif mdd >= PASS_CRITERIA["max_drawdown_rate"]["warn"]:
        grades["mdd"] = "WARN"
    else:
        grades["mdd"] = "REJECT"

    sharpe = combined.get("sharpe_ratio", 0)
    if sharpe >= PASS_CRITERIA["sharpe_ratio"]["pass"]:
        grades["sharpe"] = "PASS"
    elif sharpe >= PASS_CRITERIA["sharpe_ratio"]["warn"]:
        grades["sharpe"] = "WARN"
    else:
        grades["sharpe"] = "REJECT"

    n = combined.get("n_bets", 0)
    if n >= PASS_CRITERIA["n_bets"]["pass"]:
        grades["n_bets"] = "PASS"
    elif n >= PASS_CRITERIA["n_bets"]["warn"]:
        grades["n_bets"] = "WARN"
    else:
        grades["n_bets"] = "HOLD"

    overall = "PASS" if all(v == "PASS" for v in grades.values()) else (
        "REJECT" if "REJECT" in grades.values() else "WARN"
    )
    grades["overall"] = overall
    return grades


def generate_evaluation_report(
    fold: int,
    test_metrics: dict,
    drawdown_metrics: dict,
    overfitting_check: dict,
    condition_analysis: dict,
    residual_ic: dict | None = None,
    log_loss: dict | None = None,
) -> str:
    """Model & Strategy Generatorへのフィードバックレポートを生成する。"""
    grades = grade_metrics(test_metrics, drawdown_metrics)

    ic = residual_ic or {}
    ll = log_loss or {}

    report_lines = [
        f"## バックテスト評価レポート Fold {fold}",
        "",
        "### 総合成績",
        f"- ベット件数: {test_metrics.get('n_bets', 0)}件 [{grades['n_bets']}]",
        f"- 回収率: {test_metrics.get('roi', 0):.1%}（目標: {PASS_CRITERIA['roi']['pass']:.0%}以上） [{grades['roi']}]",
        f"- 的中率: {test_metrics.get('hit_rate', 0):.1%}（目標: {PASS_CRITERIA['hit_rate']['pass']:.0%}以上） [{grades['hit_rate']}]",
        f"- モデルエッジ: {test_metrics.get('model_edge', 0):+.3f}",
        "",
        "### リスク指標",
        f"- 最大ドローダウン: {drawdown_metrics.get('max_drawdown_rate', 0):.1%} [{grades['mdd']}]",
        f"- シャープレシオ: {drawdown_metrics.get('sharpe_ratio', 0):.2f} [{grades['sharpe']}]",
        f"- ドローダウン継続: {drawdown_metrics.get('max_dd_duration_races', 0)}レース",
        "",
        "### 情報係数（IC）",
        f"- Residual IC: {ic.get('residual_ic', float('nan')):.4f}"
        + (" [有意]" if ic.get("is_significant") else " [非有意]"),
        f"- 市場エッジ保有率: {ic.get('edge_above_market', 0):.1%}（model_prob > implied_prob の割合）",
    ]
    if ll:
        report_lines += [
            f"- モデルLog Loss: {ll.get('model_log_loss', float('nan')):.5f}",
            f"- ベースライン(オッズ)Log Loss: {ll.get('baseline_log_loss', float('nan')):.5f}",
            f"- Log Loss改善: {(ll.get('log_loss_improvement') or 0):+.5f}"
            + (" (市場より優秀)" if (ll.get("log_loss_improvement") or 0) > 0 else " (市場未満)"),
        ]
    report_lines += [
        "",
        "### 過学習チェック",
        "[OK] 問題なし" if overfitting_check["is_healthy"] else "[WARN] 要確認",
    ]
    for w in overfitting_check.get("warnings", []):
        report_lines.append(f"  - {w}")

    report_lines += ["", f"### 総合判定: **{grades['overall']}**", ""]

    weak = {k: v for k, v in condition_analysis.items() if v["roi"] < 0.95}
    if weak:
        report_lines.append("### 弱点条件（ROI 95%未満）")
        for label, m in weak.items():
            report_lines.append(f"- {label}: ROI {m['roi']:.1%}（{m['n_bets']}件）")
        report_lines.append("")

    report_lines.append("### 改善依頼")
    if grades["roi"] == "REJECT":
        report_lines.append("- 回収率未達: EV閾値を引き上げるか除外条件を追加してください")
    if grades["mdd"] == "REJECT":
        report_lines.append("- MDD超過: kelly_fractionを0.06に引き下げてください")
    if not overfitting_check["is_healthy"]:
        report_lines.append("- 過学習検知: 特徴量を削減するか正則化を強化してください")
    if not any(r != "PASS" for r in grades.values()):
        report_lines.append("- なし（全基準クリア）")

    return "\n".join(report_lines)
