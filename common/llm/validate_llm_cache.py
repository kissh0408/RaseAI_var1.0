"""validate_llm_cache.py — 夜間バッチ完了後の出力崩壊サニティチェック

実行:
    python common/llm/validate_llm_cache.py

目的:
    llm_scores.parquet が sweep_params.py に渡せる品質かを確認する。
    失敗した項目が 1 件でもあればスイープ前に要調査。

チェック項目:
    1. 基本整合性: 行数・レース数・正規化誤差・NaN
    2. fold カバレッジ: F1/F2/F3 の LLM 取得率
    3. 時系列ドリフト: 2023前半 vs 2025後半のスコア分布比較
    4. 出力崩壊検知: エントロピー低下・フラット出力（全馬同スコア）を検出
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LLM_CACHE  = ROOT / "simulator" / "cache" / "llm_scores.parquet"
LGBM_CACHE = ROOT / "simulator" / "cache" / "lgbm_scores.parquet"
FAILED_LOG = ROOT / "simulator" / "cache" / "llm_cache_failed.json"

_EPS = 1e-9
ENTROPY_FLOOR = 0.5   # bits: レース内エントロピーがこれを下回ると崩壊疑い
FLAT_THRESH   = 0.02  # 馬ごとのスコア差がこれ未満なら「全馬同スコア」と判定


def _entropy_bits(probs: np.ndarray) -> float:
    """確率ベクトルのエントロピー (bits)。"""
    p = probs[probs > _EPS]
    return float(-np.sum(p * np.log2(p)))


passed = 0
failed = 0
warnings_ = 0


def _ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")


def _warn(msg: str) -> None:
    global warnings_
    warnings_ += 1
    print(f"  [WARN] {msg}")


def run_validation() -> int:
    """0: 全PASS / 1: 1件以上FAIL"""
    print("=" * 60)
    print("LLM キャッシュ サニティチェック")
    print("=" * 60)

    if not LLM_CACHE.exists():
        print(f"[ERROR] キャッシュが存在しません: {LLM_CACHE}")
        return 1

    llm = pd.read_parquet(LLM_CACHE)
    lgbm = pd.read_parquet(LGBM_CACHE, columns=["race_id", "fold"])

    # ── 1. 基本整合性 ─────────────────────────────────────
    print("\n--- 1. 基本整合性 ---")
    n_rows  = len(llm)
    n_races = llm["race_id"].nunique()
    print(f"  行数: {n_rows:,}  /  レース数: {n_races:,}")
    if n_races == 0:
        _fail("レース数がゼロ")
        return 1
    _ok(f"レース数 {n_races:,} 件")

    # NaN チェック
    nan_count = llm[["llm_ev_score", "llm_rank_score"]].isna().sum().sum()
    if nan_count == 0:
        _ok("NaN なし")
    else:
        _fail(f"NaN あり: {nan_count} 件")

    # 正規化チェック
    sums = llm.groupby("race_id")["llm_ev_score"].sum()
    max_err = float((sums - 1.0).abs().max())
    if max_err < 1e-5:
        _ok(f"レース内合計=1.0 (max_err={max_err:.2e})")
    else:
        _fail(f"正規化誤差 max_err={max_err:.6f}")

    # ── 2. fold カバレッジ ────────────────────────────────
    print("\n--- 2. fold カバレッジ ---")
    lgbm_ids = set(lgbm["race_id"].astype(str).unique())
    llm_ids  = set(llm["race_id"].astype(str).unique())
    overlap  = lgbm_ids & llm_ids
    coverage = len(overlap) / len(lgbm_ids) * 100
    print(f"  lgbm: {len(lgbm_ids):,} / llm: {len(llm_ids):,} / overlap: {len(overlap):,} ({coverage:.1f}%)")
    if coverage >= 80:
        _ok(f"カバレッジ {coverage:.1f}% >= 80%")
    elif coverage >= 60:
        _warn(f"カバレッジ {coverage:.1f}% < 80%（スイープ結果が偏る可能性）")
    else:
        _fail(f"カバレッジ {coverage:.1f}% < 60%（スイープ信頼性低）")

    llm_with_fold = llm[llm["race_id"].isin(lgbm_ids)].merge(
        lgbm.drop_duplicates("race_id"), on="race_id", how="left"
    )
    for fold in sorted(llm_with_fold["fold"].dropna().unique()):
        fold_ids = lgbm[lgbm["fold"] == fold]["race_id"].astype(str).unique()
        hit = len(set(fold_ids) & llm_ids)
        rate = hit / len(fold_ids) * 100
        msg = f"F{fold}: {hit}/{len(fold_ids)} ({rate:.1f}%)"
        if rate >= 70:
            _ok(msg)
        else:
            _warn(msg + " — カバレッジ低")

    # ── 3. 時系列ドリフト ─────────────────────────────────
    print("\n--- 3. 時系列ドリフト (2023前半 vs 2025後半) ---")
    llm_sorted = llm.copy()
    llm_sorted["year"] = llm_sorted["race_id"].str[:4].astype(int)

    early = llm_sorted[llm_sorted["race_id"] < "2023070000000000"]
    late  = llm_sorted[llm_sorted["race_id"] > "2025060000000000"]

    if len(early) == 0 or len(late) == 0:
        _warn("ドリフト比較に十分なデータなし (early/late が空)")
    else:
        for stat_name, early_val, late_val in [
            ("mean",  early["llm_ev_score"].mean(), late["llm_ev_score"].mean()),
            ("std",   early["llm_ev_score"].std(),  late["llm_ev_score"].std()),
            ("max",   early["llm_ev_score"].max(),  late["llm_ev_score"].max()),
        ]:
            ratio = late_val / max(early_val, _EPS)
            flag  = abs(ratio - 1.0) > 0.30
            msg   = f"{stat_name}: early={early_val:.4f}  late={late_val:.4f}  ratio={ratio:.2f}"
            if flag:
                _warn(msg + "  ← 30%超の乖離。出力崩壊の可能性を調査してください")
            else:
                _ok(msg)

    # ── 4. 出力崩壊検知 ──────────────────────────────────
    print("\n--- 4. 出力崩壊検知 ---")

    race_stats = llm.groupby("race_id").agg(
        n_horses   = ("horse_num", "count"),
        score_mean = ("llm_ev_score", "mean"),
        score_std  = ("llm_ev_score", "std"),
        score_max  = ("llm_ev_score", "max"),
        score_min  = ("llm_ev_score", "min"),
    ).reset_index()

    # フラット出力検知（全馬に同スコアを出している）
    flat_mask    = (race_stats["score_max"] - race_stats["score_min"]) < FLAT_THRESH
    flat_count   = flat_mask.sum()
    flat_rate    = flat_count / len(race_stats) * 100
    flat_ids     = race_stats.loc[flat_mask, "race_id"].tolist()
    if flat_rate <= 5.0:
        _ok(f"フラット出力レース: {flat_count} 件 ({flat_rate:.1f}%)")
    elif flat_rate <= 15.0:
        _warn(f"フラット出力レース: {flat_count} 件 ({flat_rate:.1f}%)  サンプル: {flat_ids[:3]}")
    else:
        _fail(f"フラット出力レース: {flat_count} 件 ({flat_rate:.1f}%) — 出力崩壊の強い疑い")

    # エントロピー検知
    def _race_entropy(g: pd.DataFrame) -> float:
        return _entropy_bits(g["llm_ev_score"].values)

    entropies = llm.groupby("race_id").apply(_race_entropy)
    low_ent_mask = entropies < ENTROPY_FLOOR
    low_ent_count = low_ent_mask.sum()
    low_ent_rate  = low_ent_count / len(entropies) * 100
    ent_msg = (
        f"低エントロピーレース (<{ENTROPY_FLOOR} bits): {low_ent_count} 件 ({low_ent_rate:.1f}%)"
        f"  / 中央値={entropies.median():.2f} bits"
    )
    if low_ent_rate <= 5.0:
        _ok(ent_msg)
    elif low_ent_rate <= 15.0:
        _warn(ent_msg)
    else:
        _fail(ent_msg + " — 出力崩壊の可能性")

    # 時系列エントロピートレンド（前半 vs 後半）
    llm_ent = entropies.reset_index()
    llm_ent.columns = ["race_id", "entropy"]
    early_ent = llm_ent[llm_ent["race_id"] < "2023070000000000"]["entropy"]
    late_ent  = llm_ent[llm_ent["race_id"] > "2025060000000000"]["entropy"]
    if len(early_ent) > 0 and len(late_ent) > 0:
        ent_ratio = late_ent.median() / max(early_ent.median(), _EPS)
        drift_msg = (
            f"エントロピー中央値: early={early_ent.median():.2f}  late={late_ent.median():.2f}"
            f"  ratio={ent_ratio:.2f}"
        )
        if abs(ent_ratio - 1.0) > 0.25:
            _warn(drift_msg + "  ← 後半でエントロピー低下。出力崩壊傾向を調査してください")
        else:
            _ok(drift_msg)

    # ── 失敗ログ確認 ─────────────────────────────────────
    print("\n--- 5. 失敗ログ確認 ---")
    import json
    if FAILED_LOG.exists():
        with open(FAILED_LOG, encoding="utf-8") as f:
            failed_list = json.load(f)
        fail_rate = len(failed_list) / max(len(lgbm_ids), 1) * 100
        msg = f"失敗 race_id: {len(failed_list)} 件 ({fail_rate:.1f}%)"
        if fail_rate <= 15:
            _ok(msg)
        elif fail_rate <= 30:
            _warn(msg)
        else:
            _fail(msg + " — 失敗率が高すぎます。アダプタパスや race_se データを確認してください")
    else:
        _ok("失敗ログなし (全レース成功または初回実行)")

    # ── 総評 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  結果: {passed} OK / {warnings_} WARN / {failed} FAIL")
    print(f"{'='*60}")

    if failed > 0:
        print("\n[!!] FAIL があります。sweep_params.py の実行前に原因を調査してください。")
        return 1
    if warnings_ > 0:
        print("\n[!]  WARN があります。内容を確認してから sweep_params.py を実行してください。")
    else:
        print("\n[OK] 全項目クリア。sweep_params.py を実行して llm_beta を最適化できます。")
    return 0


if __name__ == "__main__":
    sys.exit(run_validation())
