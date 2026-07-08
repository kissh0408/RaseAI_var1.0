"""
portfolio_kelly.py — win+wide 非排他ベットの多変量 Kelly 最適化

G(a) = sum_i p_i log(1 + r_i^T a). MC 行は等確率。勾配は jac へ解析的に渡す。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

try:
    from strategy.src.strategy_engine import _project_to_capped_simplex
except ModuleNotFoundError:
    from strategy_engine import _project_to_capped_simplex


@dataclass(frozen=True)
class PortfolioBet:
    kind: str  # "win" | "wide"
    horse_a: int
    horse_b: Optional[int]
    prob: float
    odds: float


def race_rng(master_seed: int, race_id: str) -> np.random.Generator:
    """CRN: 同一 race_id は全 sizing モードで同一 MC サンプルを生成する。"""
    token = f"{master_seed}:{race_id}".encode("utf-8")
    seed = int.from_bytes(token[:8].ljust(8, b"\0"), "little") & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _normalize_probs(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(float), 0.0, None)
    s = float(x.sum())
    if s <= 1e-12:
        return np.ones_like(x) / max(len(x), 1)
    return x / s


def harville_sample_top3(
    p_dict: dict[int, float],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    horses = list(p_dict.keys())
    if len(horses) < 3:
        raise ValueError("harville_sample_top3 requires at least 3 horses")

    probs = _normalize_probs(np.array([p_dict[h] for h in horses], dtype=float))
    idx1 = int(rng.choice(len(horses), p=probs))
    h1 = horses[idx1]
    p_h1 = float(p_dict[h1])

    rem2 = [h for h in horses if h != h1]
    cond2 = np.array([p_dict[h] / max(1.0 - p_h1, 1e-12) for h in rem2], dtype=float)
    cond2 = _normalize_probs(cond2)
    h2 = rem2[int(rng.choice(len(rem2), p=cond2))]
    p_h2 = float(p_dict[h2])

    rem3 = [h for h in rem2 if h != h2]
    cond3 = np.array([p_dict[h] / max(1.0 - p_h1 - p_h2, 1e-12) for h in rem3], dtype=float)
    cond3 = _normalize_probs(cond3)
    h3 = rem3[int(rng.choice(len(rem3), p=cond3))]
    return h1, h2, h3


def _bet_hits(bet: PortfolioBet, top3: tuple[int, int, int]) -> bool:
    top3_set = {top3[0], top3[1], top3[2]}
    if bet.kind == "win":
        return bet.horse_a == top3[0]
    if bet.kind == "wide":
        assert bet.horse_b is not None
        return bet.horse_a in top3_set and bet.horse_b in top3_set
    raise ValueError(f"unknown bet kind: {bet.kind}")


def sample_return_matrix(
    bets: Sequence[PortfolioBet],
    p_dict: dict[int, float],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shape (n_samples, n_bets). Net return per unit stake: odds-1 if hit else -1."""
    if not bets:
        return np.zeros((0, 0), dtype=float)
    m = len(bets)
    odds = np.array([max(float(b.odds), 1.01) for b in bets], dtype=float)
    r_mat = np.empty((n_samples, m), dtype=float)
    for s in range(n_samples):
        top3 = harville_sample_top3(p_dict, rng)
        for j, bet in enumerate(bets):
            r_mat[s, j] = float(odds[j] - 1.0) if _bet_hits(bet, top3) else -1.0
    return r_mat


def growth_rate(a: np.ndarray, r_mat: np.ndarray) -> float:
    if r_mat.size == 0:
        return 0.0
    a = np.clip(a.astype(float), 0.0, None)
    wealth = 1.0 + r_mat @ a
    if np.any(wealth <= 1e-12):
        return -1e9
    return float(np.mean(np.log(wealth)))


def growth_gradient(a: np.ndarray, r_mat: np.ndarray) -> np.ndarray:
    a = np.clip(a.astype(float), 0.0, None)
    wealth = 1.0 + r_mat @ a
    if np.any(wealth <= 1e-12):
        return np.zeros_like(a)
    coef = 1.0 / wealth
    return (coef[:, None] * r_mat).mean(axis=0)


def optimize_full_kelly(
    r_mat: np.ndarray,
    *,
    total_cap: float,
    max_single: float,
    fractional_kelly: float = 0.08,
    x0: Optional[np.ndarray] = None,
    odds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = r_mat.shape[1] if r_mat.size else 0
    if n == 0:
        return np.array([], dtype=float)
    cap = float(total_cap)
    if x0 is None and odds is not None and probs is not None:
        raw = []
        for p, o in zip(probs, odds):
            bb = max(float(o) - 1.0, 1e-12)
            q = 1.0 - float(p)
            raw.append(max((bb * float(p) - q) / bb, 0.0) * fractional_kelly)
        x0 = _project_to_capped_simplex(np.array(raw, dtype=float), cap)
    if x0 is None:
        x0 = np.full(n, min(cap / max(n, 1), max_single * 0.5), dtype=float)
    x0 = _project_to_capped_simplex(np.clip(x0, 0.0, max_single), cap)

    if minimize is None:
        return x0

    def objective(x: np.ndarray) -> float:
        return -growth_rate(x, r_mat)

    def jac(x: np.ndarray) -> np.ndarray:
        return -growth_gradient(x, r_mat)

    bounds = [(0.0, max_single) for _ in range(n)]
    constraints = [{"type": "ineq", "fun": lambda x: cap - float(np.sum(x))}]

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-10, "disp": False},
    )
    if not res.success:
        return x0
    return np.maximum(_project_to_capped_simplex(np.clip(res.x, 0.0, max_single), cap), 0.0)


def optimize_fractional_kelly(
    r_mat: np.ndarray,
    *,
    total_cap: float,
    max_single: float,
    growth_ratio_min: float,
    fractional_kelly: float = 0.08,
) -> tuple[np.ndarray, float, float]:
    """
    Two-stage fractional Kelly:
    1) full optimum a* and G*
    2) min sum(a) s.t. G(a)/G* >= k
    """
    a_star = optimize_full_kelly(
        r_mat,
        total_cap=total_cap,
        max_single=max_single,
        fractional_kelly=fractional_kelly,
    )
    g_star = growth_rate(a_star, r_mat)
    cap = float(total_cap)
    if g_star <= 0 or growth_ratio_min >= 0.999:
        return a_star, g_star, g_star

    n = r_mat.shape[1]
    threshold = growth_ratio_min * g_star
    x0 = _project_to_capped_simplex(np.clip(a_star * growth_ratio_min, 0.0, max_single), cap)

    if minimize is None:
        return x0, g_star, growth_rate(x0, r_mat)

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x))

    def jac_sum(_x: np.ndarray) -> np.ndarray:
        return np.ones(n, dtype=float)

    bounds = [(0.0, max_single) for _ in range(n)]
    constraints = [
        {"type": "ineq", "fun": lambda x: cap - float(np.sum(x))},
        {
            "type": "ineq",
            "fun": lambda x, rm=r_mat, th=threshold: growth_rate(x, rm) - th,
            "jac": lambda x, rm=r_mat: growth_gradient(x, rm),
        },
    ]

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        jac=jac_sum,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-9, "disp": False},
    )
    a_opt = x0 if not res.success else np.clip(res.x, 0.0, max_single)
    a_opt = _project_to_capped_simplex(a_opt, cap)
    if growth_rate(a_opt, r_mat) < threshold - 1e-8:
        a_opt = x0
    return a_opt, g_star, growth_rate(a_opt, r_mat)


def apply_portfolio_kelly_to_recommendations(
    rec_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    horse_col: str = "horse_num",
    prob_col: str = "pred_prob",
    bankroll: float,
    bet_unit: int,
    max_stake_per_bet: int,
    max_invest_per_race: int,
    max_total_fraction: float,
    max_single_fraction: float,
    fractional_kelly: float,
    mode: str = "portfolio_kelly_fractional",
    growth_ratio_min: float = 0.5,
    ind_cap_ratio: float = 0.85,
    mc_samples: int = 500,
    mc_seed: int = 42,
) -> pd.DataFrame:
    """
    レース内の単勝+ワイド推奨に多変量 Kelly 配分を適用する。
    portfolio_kelly_enabled=false 時は呼び出し側でスキップすること。
    """
    import pandas as pd

    if rec_df.empty:
        return rec_df
    out = rec_df.copy()
    ticket_map = {"単勝": "win", "ワイド": "wide"}

    pred = pred_df.copy()
    pred[race_id_col] = pred[race_id_col].astype(str)

    for race_id, grp in out.groupby("race_id", sort=False):
        portfolio_mask = grp["ticket_type"].isin(["単勝", "ワイド"])
        if not portfolio_mask.any():
            continue
        sub = grp.loc[portfolio_mask]
        if sub["ticket_type"].nunique() < 2:
            continue

        race_pred = pred[pred[race_id_col] == str(race_id)]
        if race_pred.empty:
            continue
        score_col = "pred_rank1" if "pred_rank1" in race_pred.columns else prob_col
        s = pd.to_numeric(race_pred[score_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        rs = float(s.sum())
        p_dict = {
            int(h): float(p)
            for h, p in zip(race_pred[horse_col].astype(int), (s / rs if rs > 0 else s))
        }
        if len(p_dict) < 3:
            continue

        bets: list[PortfolioBet] = []
        idx_map: list[int] = []
        ind_raw: list[float] = []
        for idx, row in sub.iterrows():
            kind = ticket_map.get(str(row["ticket_type"]))
            if kind is None:
                continue
            prob = float(row.get("pred_prob", 0.0))
            odds = float(row.get("odds_effective", row.get("odds_raw", 1.01)))
            if kind == "win":
                bets.append(PortfolioBet("win", int(row["horse_num"]), None, prob, odds))
            else:
                partner = row.get("partner_horse_num")
                if partner is None and "-" in str(row.get("ticket", "")):
                    parts = str(row["ticket"]).split("-")
                    partner = int(parts[1]) if len(parts) >= 2 else None
                if partner is None:
                    continue
                bets.append(PortfolioBet("wide", int(row["horse_num"]), int(partner), prob, odds))
            idx_map.append(int(idx))
            bb = max(odds - 1.0, 1e-12)
            ind_raw.append(max((bb * prob - (1 - prob)) / bb, 0.0) * fractional_kelly)

        if not bets:
            continue

        rng = race_rng(mc_seed, str(race_id))
        r_mat = sample_return_matrix(bets, p_dict, mc_samples, rng)
        if mode == "portfolio_kelly_fractional":
            fr, _, _ = optimize_fractional_kelly(
                r_mat,
                total_cap=max_total_fraction,
                max_single=max_single_fraction,
                growth_ratio_min=growth_ratio_min,
                fractional_kelly=fractional_kelly,
            )
        else:
            fr = optimize_full_kelly(
                r_mat,
                total_cap=max_total_fraction,
                max_single=max_single_fraction,
                fractional_kelly=fractional_kelly,
                odds=np.array([b.odds for b in bets]),
                probs=np.array([b.prob for b in bets]),
            )

        ind = _project_to_capped_simplex(np.array(ind_raw, dtype=float), max_total_fraction)
        ind_sum = float(ind.sum())
        port_sum = float(fr.sum())
        cap_total = ind_sum * float(ind_cap_ratio)
        if port_sum > 1e-12 and cap_total > 1e-12 and port_sum > cap_total:
            fr = fr * (cap_total / port_sum)

        race_invest = 0.0
        for idx, frac, bet in zip(idx_map, fr, bets):
            target = bankroll * float(frac)
            remaining = max(max_invest_per_race - race_invest, 0.0)
            capped = min(target, max_stake_per_bet, remaining)
            stake = int(capped // bet_unit) * bet_unit
            if stake < bet_unit:
                stake = 0
            race_invest += stake
            out.at[idx, "kelly_fraction"] = float(frac)
            out.at[idx, "suggested_stake"] = int(stake)
            out.at[idx, "is_executable"] = stake >= bet_unit

    return out
