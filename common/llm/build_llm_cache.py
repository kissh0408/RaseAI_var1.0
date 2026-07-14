"""build_llm_cache.py — バックテスト用 LLM スコアキャッシュ構築（再開可能版）

lgbm_scores.parquet に含まれる全レース (~11,442件 / 2023-2025年) に対して
LLM 推論を実行し、simulator/cache/llm_scores.parquet を生成する。

防御的設計:
  - チェックポイント: CHECKPOINT_SIZE レース毎に中間セーブ。起動時に済み race_id を検出し resume。
  - 例外隔離:  バッチ単位の try-except + 1件ずつへのフォールバック。全損しない。
  - VRAM管理:  チェックポイント毎に empty_cache + gc.collect() で断片化を防止。

実行:
    conda activate keiba-ml
    python common/llm/build_llm_cache.py [--batch-size 8] [--checkpoint-size 100]

出力:
    simulator/cache/llm_scores.parquet   (最終出力)
    simulator/cache/llm_cache_checkpoint.parquet  (中間状態)
    simulator/cache/llm_cache_failed.json         (失敗 race_id リスト)
    logs/build_llm_cache.log
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent  # RaceAI_var3.0/

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── ログ設定 ──────────────────────────────────────────────
LOG_DIR = _ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "build_llm_cache.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── パス定数 ──────────────────────────────────────────────
LGBM_CACHE      = _ROOT / "simulator" / "cache" / "lgbm_scores.parquet"
OUTPUT_PATH     = _ROOT / "simulator" / "cache" / "llm_scores.parquet"
CHECKPOINT_PATH = _ROOT / "simulator" / "cache" / "llm_cache_checkpoint.parquet"
FAILED_LOG      = _ROOT / "simulator" / "cache" / "llm_cache_failed.json"
DATA_OUTPUT     = _ROOT / "common" / "data" / "output"
ADAPTER_PATH    = str(_THIS_DIR / "models" / "lora_adapters")

# lgbm_scores.parquet に含まれる年の一覧（不足年は警告のみ）
DATA_YEARS = [2023, 2024, 2025]

_EPS = 1e-9


# ── データ読込 ────────────────────────────────────────────

def _load_all_race_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """DATA_YEARS 分の race_se / race_ra を結合して返す。race_id 列付き。"""
    from common.llm.race_to_text import make_race_id

    se_frames, ra_frames = [], []
    for yr in DATA_YEARS:
        se_path = DATA_OUTPUT / "race_se" / f"race_se_{yr}.csv"
        ra_path = DATA_OUTPUT / "race_ra" / f"race_ra_{yr}.csv"
        if not se_path.exists():
            logger.warning("race_se_%d.csv が見つかりません (スキップ): %s", yr, se_path)
            continue
        df_se = pd.read_csv(se_path, low_memory=False)
        df_ra = pd.read_csv(ra_path, low_memory=False)
        df_se["race_id"] = df_se.apply(make_race_id, axis=1)
        df_ra["race_id"] = df_ra.apply(make_race_id, axis=1)
        se_frames.append(df_se)
        ra_frames.append(df_ra)
        logger.info("  %d年: race_se=%d行 / race_ra=%d行", yr, len(df_se), len(df_ra))

    if not se_frames:
        raise FileNotFoundError(f"race_se が1件も見つかりません。DATA_YEARS={DATA_YEARS}")

    return pd.concat(se_frames, ignore_index=True), pd.concat(ra_frames, ignore_index=True)


# ── チェックポイント ───────────────────────────────────────

def _load_done_ids() -> set[str]:
    """チェックポイントから完了済み race_id セットを返す。"""
    if not CHECKPOINT_PATH.exists():
        return set()
    try:
        cp = pd.read_parquet(CHECKPOINT_PATH)
        done = set(cp["race_id"].astype(str).unique())
        logger.info("チェックポイント読込: %d レース完了済み", len(done))
        return done
    except Exception as exc:
        logger.warning("チェックポイント読込失敗（最初から再実行）: %s", exc)
        return set()


def _load_failed_ids() -> list[str]:
    if not FAILED_LOG.exists():
        return []
    with open(FAILED_LOG, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(new_rows: list[dict]) -> None:
    """new_rows を正規化してチェックポイントファイルに追記保存する。"""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    new_df["llm_ev_score"] = _normalize_per_race(new_df, "llm_ev_score")

    if CHECKPOINT_PATH.exists():
        existing = pd.read_parquet(CHECKPOINT_PATH)
        new_df = pd.concat([existing, new_df], ignore_index=True)

    # 並行プロセス由来の重複を防ぐ（race_id × horse_num の初出を保持）
    new_df = new_df.drop_duplicates(subset=["race_id", "horse_num"], keep="first")
    new_df.to_parquet(CHECKPOINT_PATH, index=False)
    logger.info(
        "チェックポイント保存: 累計 %d レース / %d 行",
        new_df["race_id"].nunique(), len(new_df),
    )


def _save_failed(failed: list[str]) -> None:
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        json.dump(sorted(set(failed)), f, ensure_ascii=False, indent=2)


# ── 正規化 ────────────────────────────────────────────────

def _normalize_per_race(df: pd.DataFrame, col: str) -> pd.Series:
    def _norm(x: pd.Series) -> pd.Series:
        s = x.sum()
        return x / s if s > _EPS else pd.Series([1.0 / len(x)] * len(x), index=x.index)
    return df.groupby("race_id")[col].transform(_norm)


# ── メイン推論ループ ──────────────────────────────────────

def _max_tokens_for_horses(n_horses: int) -> int:
    """頭数から動的 max_new_tokens を計算する。

    1頭あたり JSON 出力は約 42 トークン。余裕 +80 トークンを加えた上限。
    固定 768 より短い場合のみ短縮 (安全側クランプ)。
    """
    return min(768, max(300, n_horses * 42 + 80))


def _infer_batch_with_fallback(
    prompts: list[str],
    race_ids: list[str],
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int = 768,
) -> list[list[dict]]:
    """バッチ推論。失敗時は 1 件ずつにフォールバック。"""
    from common.llm.inference import predict_batch

    try:
        return predict_batch(
            prompts, model, tokenizer,
            batch_size=batch_size,
            race_ids=race_ids,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        logger.exception(
            "バッチ推論で例外 (batch_size=%d, %d件)。1件ずつにフォールバック。",
            batch_size, len(prompts),
        )
        results: list[list[dict]] = []
        for p, rid in zip(prompts, race_ids):
            try:
                r = predict_batch([p], model, tokenizer, batch_size=1, race_ids=[rid],
                                  max_new_tokens=max_new_tokens)
                results.extend(r)
            except Exception:
                logger.exception("1件推論でも失敗 (race_id=%s)", rid)
                results.append([])
        return results


def run_build_cache(batch_size: int = 8, checkpoint_size: int = 100, cooldown_secs: int = 0) -> None:
    t_start = time.monotonic()

    # ── 対象 race_ids を決定 ──
    lgbm = pd.read_parquet(LGBM_CACHE, columns=["race_id"])
    all_ids = list(lgbm["race_id"].astype(str).unique())
    logger.info("lgbm_scores.parquet: 全対象 %d レース", len(all_ids))

    done_ids   = _load_done_ids()
    failed_ids = set(_load_failed_ids())
    skip_ids   = done_ids | failed_ids
    pending    = [r for r in all_ids if r not in skip_ids]

    logger.info(
        "未処理: %d  / 完了済: %d  / 過去失敗スキップ: %d",
        len(pending), len(done_ids), len(failed_ids),
    )

    if not pending:
        logger.info("全レース処理済み。最終出力に昇格します。")
        _finalize()
        return

    # ── race_se / race_ra 全年ロード ──
    logger.info("race_se / race_ra 読込中 (年: %s)...", DATA_YEARS)
    all_se, all_ra = _load_all_race_data()
    ra_indexed = all_ra.drop_duplicates("race_id").set_index("race_id")
    logger.info("race_se: %d 行 / race_ra: %d 行 ロード完了", len(all_se), len(all_ra))

    # ── 高速化①: race_id → 馬データを O(1) で引けるようにインデックス化 ──
    se_by_race: dict[str, pd.DataFrame] = {
        rid: grp for rid, grp in all_se.groupby("race_id")
    }

    # ── 高速化②: バケットバッチング — 頭数の近いレースをまとめて同一バッチに入れる ──
    # 頭数が揃うとパディング量が減り、max_new_tokens を動的に短縮できる。
    # 8頭バッチ: max_new_tokens≈416、18頭バッチ: 768 → 混合比較で最大 ~30% 高速化。
    horse_counts: dict[str, int] = {}
    for rid in pending:
        horses_tmp = se_by_race.get(rid)
        horse_counts[rid] = len(horses_tmp) if horses_tmp is not None else 18
    pending = sorted(pending, key=lambda r: horse_counts[r])
    logger.info(
        "バケットバッチング: 頭数範囲 %d〜%d (中央値 %d)",
        min(horse_counts[r] for r in pending),
        max(horse_counts[r] for r in pending),
        sorted(horse_counts[r] for r in pending)[len(pending) // 2],
    )

    # ── モデルロード ──
    from common.llm.inference import load_model
    from common.llm.race_to_text import race_to_prompt

    tokenizer, model = load_model(lora_adapter_path=ADAPTER_PATH)

    # ── バッチ推論ループ ──
    chunk_rows:  list[dict] = []
    chunk_done:  int        = 0   # 今セッションで完了したレース数
    new_failed:  list[str]  = []
    total_done   = len(done_ids)  # 累計（前回分含む）

    for batch_start in range(0, len(pending), batch_size):
        batch_ids = pending[batch_start : batch_start + batch_size]

        # ── プロンプト生成（各 race 個別に try-except）──
        prompts:    list[str] = []
        valid_ids:  list[str] = []
        batch_horse_counts: list[int] = []
        for rid in batch_ids:
            try:
                horses = se_by_race.get(rid)
                if horses is None or horses.empty or rid not in ra_indexed.index:
                    logger.warning("race_se/ra に %s のデータなし。スキップ。", rid)
                    new_failed.append(rid)
                    continue
                ra_row = ra_indexed.loc[rid]
                text   = race_to_prompt(ra_row, horses)
                prompts.append(text)
                valid_ids.append(rid)
                batch_horse_counts.append(len(horses))
            except Exception:
                logger.exception("プロンプト生成失敗 (race_id=%s)。スキップ。", rid)
                new_failed.append(rid)

        if not prompts:
            continue

        # ── 高速化③: 動的 max_new_tokens — バッチ内最大頭数に合わせて短縮 ──
        max_horses_in_batch = max(batch_horse_counts) if batch_horse_counts else 18
        dyn_max_tokens = _max_tokens_for_horses(max_horses_in_batch)

        # ── 推論（バッチ失敗→1件フォールバック）──
        results = _infer_batch_with_fallback(
            prompts, valid_ids, model, tokenizer, batch_size,
            max_new_tokens=dyn_max_tokens,
        )

        # ── 結果収集 ──
        for rid, scores in zip(valid_ids, results):
            if not scores:
                new_failed.append(rid)
                continue
            for s in scores:
                chunk_rows.append({
                    "race_id":        rid,
                    "horse_num":      int(s["horse_num"]),
                    "llm_ev_score":   float(s["ev_score"]),
                    "llm_rank_score": float(s["rank_score"]),
                })
            chunk_done  += 1
            total_done  += 1

        # ── チェックポイント ──
        if chunk_done > 0 and chunk_done % checkpoint_size == 0:
            _save_checkpoint(chunk_rows)
            chunk_rows = []
            _save_failed(list(failed_ids) + new_failed)

            torch.cuda.empty_cache()
            gc.collect()

            # サーマルスロットリング対策: GPU をクールダウンさせる休止
            if cooldown_secs > 0:
                logger.info("GPU クールダウン: %d 秒休止中...", cooldown_secs)
                time.sleep(cooldown_secs)

            elapsed   = time.monotonic() - t_start
            speed     = elapsed / max(1, chunk_done)  # 秒/レース（今セッション）
            remaining = (len(pending) - batch_start - batch_size) * speed
            logger.info(
                "進捗: %d/%d  |  経過 %.0f 分  |  残り推定 %.0f 分  |  速度 %.1f 秒/レース",
                total_done, len(all_ids),
                elapsed / 60, max(0, remaining) / 60, speed,
            )

    # ── 残余チャンクを保存 ──
    if chunk_rows:
        _save_checkpoint(chunk_rows)

    # ── 失敗ログ最終保存 ──
    all_failed = sorted(set(list(failed_ids) + new_failed))
    _save_failed(all_failed)
    if all_failed:
        logger.warning("失敗レース合計: %d 件 → %s", len(all_failed), FAILED_LOG)
    else:
        logger.info("失敗レース: 0 件 (完走)")

    _finalize()

    elapsed_h = (time.monotonic() - t_start) / 3600
    logger.info("=== 全完了: %.2f 時間 ===", elapsed_h)


def _finalize() -> None:
    """チェックポイントを最終出力に昇格する。"""
    if not CHECKPOINT_PATH.exists():
        logger.error("チェックポイントが存在しません。推論が1件も成功しなかった可能性があります。")
        return
    df = pd.read_parquet(CHECKPOINT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(
        "=== 最終出力: %s  %d 行 / %d レース ===",
        OUTPUT_PATH, len(df), df["race_id"].nunique(),
    )


# ── エントリポイント ──────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM バックテストキャッシュ構築")
    parser.add_argument("--batch-size",      type=int, default=8,   help="GPU バッチサイズ (default: 8)")
    parser.add_argument("--checkpoint-size", type=int, default=100, help="中間セーブ間隔 (レース数, default: 100)")
    parser.add_argument("--cooldown-secs",   type=int, default=0,   help="チェックポイント後の GPU 休止秒数 (熱対策, default: 0=無効)")
    args = parser.parse_args()

    logger.info("=== build_llm_cache.py 開始 ===")
    logger.info("  batch_size=%d  checkpoint_size=%d  cooldown_secs=%d",
                args.batch_size, args.checkpoint_size, args.cooldown_secs)
    logger.info("  対象キャッシュ: %s", LGBM_CACHE)
    logger.info("  出力先:         %s", OUTPUT_PATH)

    run_build_cache(batch_size=args.batch_size, checkpoint_size=args.checkpoint_size,
                    cooldown_secs=args.cooldown_secs)
