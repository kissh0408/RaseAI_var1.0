"""探索的分析: 時系列オッズ(0B41単複枠)のドリフト信号に、確定オッズだけでは
説明できない着順予測情報があるかを調べる。

重要(必読): ここで使う時系列オッズは JV-Link の保持期間制限(1年)により2025-2026年分しか
取得できない。この期間は evaluation/splits.py の get_walkforward_folds() における
fold3 TEST期間(2025-01-01〜)と完全に重複する。fold3 TESTは既にL1/L2の正式合否判定に
使用済みの神聖な期間であり、本スクリプトでの分析は「同じデータの二度目の使用」に当たる。
したがって、ここで得られる結果はどれだけ良く見えても正式な検証を意味しない。
正式なα再フィット・ゲート判定は、今後の週次自動取得(RaceAI_WeeklyOddsTS)で新規に
溜まるプロスペクティブなデータでのみ行うこと。

本スクリプトは探索専用であり、本番コード(pure_rank/src, prob_fusion/src, betting/src)は
一切変更しない。L1特徴量にこの分析結果を組み込むことも禁止(CLAUDE.md市場情報境界)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.odds_loader import attach_odds_from_se_parquet

ODDS_TS_DIR = ROOT / "common" / "data" / "output" / "odds_ts"
SCORES_PATH = ROOT / "pure_rank" / "data" / "03_scores" / "scores_v39_course_slim.parquet"
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_announce_datetime(series: pd.Series) -> pd.Series:
    """announce_datetime (MMDDHHMM, 先頭ゼロ欠落あり) を8桁ゼロパディング後、int化して返す。

    年またぎの厳密な絶対時刻は復元しない(ファイルは年ごとに分かれているため
    レース内の相対順序が保たれていれば十分、との仕様に基づく)。
    """
    s = series.astype("Int64").astype(str).str.zfill(8)
    return pd.to_numeric(s, errors="coerce")


def load_ts_odds(year: int) -> pd.DataFrame:
    path = ODDS_TS_DIR / f"WinPlaceOddsTS_{year}.csv"
    print(f"  loading {path} ...")
    usecols = ["race_id", "horse_num", "win_odds", "win_odds_status", "announce_datetime"]
    dtype = {
        "race_id": str,
        "horse_num": "int16",
        "win_odds_status": str,
    }
    df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    df = df[df["win_odds_status"] == "ok"].copy()
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
    df = df.dropna(subset=["win_odds"])
    df = df[df["win_odds"] > 0]
    df["announce_dt_num"] = _parse_announce_datetime(df["announce_datetime"])
    df = df.dropna(subset=["announce_dt_num"])
    return df[["race_id", "horse_num", "win_odds", "announce_dt_num"]]


def build_early_late(df_ts: pd.DataFrame) -> pd.DataFrame:
    """race_id x horse_num ごとに最初/最後の ok スナップショットのオッズを抽出。"""
    df_ts = df_ts.sort_values(["race_id", "horse_num", "announce_dt_num"])
    grp = df_ts.groupby(["race_id", "horse_num"], sort=False)
    first = grp.first().rename(columns={"win_odds": "early_odds", "announce_dt_num": "early_dt"})
    last = grp.last().rename(columns={"win_odds": "late_ts_odds", "announce_dt_num": "late_dt"})
    n_obs = grp.size().rename("n_snapshots")
    out = first.join(last, how="inner").join(n_obs, how="inner").reset_index()
    return out


def main() -> None:
    print("=== Step 1: Load time-series odds (2025, 2026) ===")
    ts_2025 = load_ts_odds(2025)
    ts_2026 = load_ts_odds(2026)
    ts_all = pd.concat([ts_2025, ts_2026], ignore_index=True)
    print(f"  rows (ok only): {len(ts_all):,}")

    print("=== Step 2: Build early/late snapshots per race_id x horse_num ===")
    early_late = build_early_late(ts_all)
    print(f"  race_id x horse_num pairs: {len(early_late):,}")
    print(f"  n_snapshots stats: {early_late['n_snapshots'].describe()}")

    print("=== Step 3: Load finish_rank / race_date from scores parquet ===")
    scores = pd.read_parquet(SCORES_PATH, columns=["race_id", "horse_num", "finish_rank", "race_date"])
    scores["race_id"] = scores["race_id"].astype(str)
    scores = scores[scores["race_date"] >= "2025-01-01"].copy()
    print(f"  scores rows (>=2025-01-01): {len(scores):,}, races: {scores['race_id'].nunique():,}")

    print("=== Step 4: Attach final (confirmed) win odds via existing loader ===")
    scores_with_odds = attach_odds_from_se_parquet(scores)
    scores_with_odds = scores_with_odds.rename(columns={"odds": "final_odds"})
    n_missing_final = scores_with_odds["final_odds"].isna().sum()
    print(f"  final_odds missing: {n_missing_final:,} / {len(scores_with_odds):,}")

    print("=== Step 5: Merge ===")
    early_late["race_id"] = early_late["race_id"].astype(str)
    merged = scores_with_odds.merge(early_late, on=["race_id", "horse_num"], how="inner")
    merged = merged.dropna(subset=["final_odds", "early_odds", "late_ts_odds"])
    merged = merged[(merged["final_odds"] > 0) & (merged["early_odds"] > 0) & (merged["late_ts_odds"] > 0)]
    n_races = merged["race_id"].nunique()
    n_horses = len(merged)
    print(f"  merged rows: {n_horses:,}, races: {n_races:,}")

    merged.to_parquet(OUT_DIR / "merged_drift_dataset.parquet", index=False)

    print("=== Step 6: Descriptive stats on drift (ln(final/early)) ===")
    merged["drift"] = np.log(merged["final_odds"] / merged["early_odds"])
    merged["drift_late"] = np.log(merged["final_odds"] / merged["late_ts_odds"])
    desc = merged[["drift", "drift_late"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc)

    print("=== Step 7: Market probability q_final (proportional) ===")
    from evaluation.market_baseline import proportional_market_prob

    q_list = []
    for race_id, grp in merged.groupby("race_id"):
        q = proportional_market_prob(grp["final_odds"].values)
        q_list.append(pd.Series(q, index=grp.index))
    merged["q_final"] = pd.concat(q_list).sort_index()
    merged["is_win"] = (merged["finish_rank"].astype(int) == 1).astype(int)
    merged["ln_q_final"] = np.log(merged["q_final"].clip(lower=1e-9))

    print("=== Step 8: Logistic regression: is_win ~ ln_q_final + drift ===")
    import statsmodels.api as sm

    # NOTE: drift_late = ln(final/late_ts_odds) turned out to have zero variance in
    # practice (the last "ok" snapshot in the time-series feed coincides with the
    # confirmed final odds), so it is degenerate for regression purposes and is
    # reported only as a descriptive-stats sanity check, not fit in the logit.
    drift_cols = ["drift"] if merged["drift_late"].std() < 1e-9 else ["drift", "drift_late"]
    for drift_col in drift_cols:
        print(f"--- using {drift_col} ---")
        X = merged[["ln_q_final", drift_col]].copy()
        X = sm.add_constant(X)
        y = merged["is_win"]
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())

        # logloss comparison: q_final alone vs logistic w/ drift
        p_q_only = merged["q_final"].clip(1e-9, 1 - 1e-9)
        logloss_q_only = -np.mean(y * np.log(p_q_only) + (1 - y) * np.log(1 - p_q_only))

        p_full = model.predict(X)
        p_full = p_full.clip(1e-9, 1 - 1e-9)
        logloss_full = -np.mean(y * np.log(p_full) + (1 - y) * np.log(1 - p_full))

        print(f"  logloss (q_final only, raw prob): {logloss_q_only:.5f}")
        print(f"  logloss (logit w/ ln_q_final + {drift_col}, in-sample fit): {logloss_full:.5f}")

    print("=== Step 9: Bin analysis (quintile of drift within odds-similar groups) ===")
    # Bin by q_final decile, then within each decile split drift into terciles.
    merged["q_decile"] = pd.qcut(merged["q_final"], 10, labels=False, duplicates="drop")
    merged["drift_tercile"] = merged.groupby("q_decile")["drift"].transform(
        lambda x: pd.qcut(x, 3, labels=["down(popularized)", "flat", "up(unpopularized)"], duplicates="drop")
    )
    bin_summary = (
        merged.groupby(["drift_tercile"], observed=True)
        .agg(n=("is_win", "size"), win_rate=("is_win", "mean"), mean_q_final=("q_final", "mean"))
        .reset_index()
    )
    print(bin_summary)

    # chi-square test: drift_tercile vs is_win, controlling roughly for q_decile via aggregation
    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(merged["drift_tercile"], merged["is_win"])
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    print(f"  chi2={chi2:.3f}, p={p_val:.5f}, dof={dof}")
    print(contingency)

    print("=== Step 10: Save summary report ===")
    report_lines = []
    report_lines.append("# 時系列オッズドリフト 探索的分析レポート\n")
    report_lines.append(
        "**注意**: この分析で使用した時系列オッズは2025-2026年分のみ(JV-Link保持期間1年の制約)。"
        "この期間は evaluation/splits.py の fold3 TEST期間と完全に重複しており、"
        "fold3 TESTは既にL1/L2の正式合否判定に使用済みの神聖な期間である。"
        "したがって本結果は正式な検証ではなく、あくまで仮説を潰す/残すための探索的分析である。"
        "正式なα再フィット・ゲート判定は今後の週次自動取得(RaceAI_WeeklyOddsTS)で"
        "新規に溜まるプロスペクティブなデータでのみ行うこと。\n"
    )
    report_lines.append(f"- n_races (merged) = {n_races}")
    report_lines.append(f"- n_horses (merged) = {n_horses}\n")
    report_lines.append("## Drift descriptive stats\n")
    report_lines.append(desc.to_string())
    report_lines.append("\n\n## Bin summary (drift tercile within q_final decile)\n")
    report_lines.append(bin_summary.to_string())
    report_lines.append(f"\n\nchi2={chi2:.3f}, p={p_val:.5f}, dof={dof}\n")
    (OUT_DIR / "drift_analysis_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved summary to {OUT_DIR / 'drift_analysis_summary.md'}")


if __name__ == "__main__":
    main()
