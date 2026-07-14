"""
run_today_llm_inference.py — 当日レースのLLM推論バッチ

run_today_prediction_pipeline() の前に実行する朝バッチ。
GPU PCで実行し today_llm_scores.parquet を生成することで、
APIサーバー（CPU）側はルックアップのみでLLMスコアを利用できる。

使用法:
    python common/llm/run_today_llm_inference.py [YYYYMMDD]
    # 引数省略時は今日の日付

出力: main/Resulut/today_llm_scores.parquet
  columns: race_id (str), horse_num (int), llm_ev_score (float), llm_rank_score (float)
  - llm_ev_score はレース内で合計=1 に正規化済み（q_llm として score_fusion へ渡せる形式）
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent  # RaceAI_var3.0/

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

DATA_OUTPUT_DIR = _ROOT / "common" / "data" / "output"
MAIN_DATA_DIR   = _ROOT / "main" / "data" / "race"   # 当日分フォールバック
RESULT_DIR = _ROOT / "main" / "Resulut"
OUTPUT_PATH = RESULT_DIR / "today_llm_scores.parquet"
ADAPTER_PATH = str(_THIS_DIR / "models" / "lora_adapters")

_EPS = 1e-9


def _load_today_se_ra(race_day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """当日分の race_se / race_ra を返す。

    優先順位:
      1. common/data/output/race_se/race_se_{year}.csv を month_day でフィルタ
      2. データが空なら main/data/race/race_se.csv (当日フラットファイル) にフォールバック
    """
    year = race_day[:4]
    month_day = int(race_day[4:8])  # e.g. "0622" → 622

    se_path = DATA_OUTPUT_DIR / "race_se" / f"race_se_{year}.csv"
    ra_path = DATA_OUTPUT_DIR / "race_ra" / f"race_ra_{year}.csv"

    from common.llm.race_to_text import make_race_id

    try:
        df_se = pd.read_csv(se_path, low_memory=False)
        df_ra = pd.read_csv(ra_path, low_memory=False)
    except FileNotFoundError as e:
        logger.warning("年次CSVが見つかりません: %s", e)
        df_se = pd.DataFrame()
        df_ra = pd.DataFrame()
    df_se["race_id"] = df_se.apply(make_race_id, axis=1)
    df_ra["race_id"] = df_ra.apply(make_race_id, axis=1)
    df_se = df_se[df_se["month_day"] == month_day].copy()
    df_ra = df_ra[df_ra["month_day"] == month_day].copy()

    # フォールバック: 年次 CSV に当日データがない場合は当日フラットファイルを使用
    if df_se.empty:
        fb_se = MAIN_DATA_DIR / "race_se.csv"
        fb_ra = MAIN_DATA_DIR / "race_ra.csv"
        if fb_se.exists() and fb_ra.exists():
            df_se_fb = pd.read_csv(fb_se, low_memory=False)
            df_ra_fb = pd.read_csv(fb_ra, low_memory=False)
            df_se_fb["race_id"] = df_se_fb.apply(make_race_id, axis=1)
            df_ra_fb["race_id"] = df_ra_fb.apply(make_race_id, axis=1)
            # フラットファイルは当日分のみなので month_day フィルタなし
            logger.warning(
                "年次 CSV に month_day=%d のデータなし。フォールバック: %s (%d 行)",
                month_day, fb_se, len(df_se_fb),
            )
            return df_se_fb, df_ra_fb

    return df_se, df_ra


def _normalize_per_race(df: pd.DataFrame, col: str) -> pd.Series:
    """col をレース内で合計=1 に正規化する。ゼロレースは均等分配。"""
    def _norm(x):
        s = x.sum()
        return x / s if s > _EPS else pd.Series([1.0 / len(x)] * len(x), index=x.index)
    return df.groupby("race_id")[col].transform(_norm)


def run_today_llm_inference(
    race_day: str | None = None,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    当日レースのLLM推論を実行し today_llm_scores.parquet を保存する。

    Args:
        race_day: YYYYMMDD 文字列（None のとき今日）
        batch_size: GPU バッチサイズ（VRAM 8GB なら 8 推奨）

    Returns:
        DataFrame: columns=[race_id, horse_num, llm_ev_score, llm_rank_score]
    """
    if race_day is None:
        race_day = date.today().strftime("%Y%m%d")

    logger.info("当日(%s)のLLM推論を開始します", race_day)

    df_se, df_ra = _load_today_se_ra(race_day)
    if df_se.empty:
        logger.warning("当日の race_se データが空です (race_day=%s)", race_day)
        return pd.DataFrame(columns=["race_id", "horse_num", "llm_ev_score", "llm_rank_score"])

    ra_indexed = df_ra.set_index("race_id")
    race_ids = [rid for rid in df_se["race_id"].unique() if rid in ra_indexed.index]
    logger.info("推論対象レース数: %d", len(race_ids))

    from common.llm.race_to_text import race_to_prompt
    from common.llm.inference import load_model, predict_batch

    tokenizer, model = load_model(lora_adapter_path=ADAPTER_PATH)

    texts, valid_ids = [], []
    for rid in race_ids:
        horses = df_se[df_se["race_id"] == rid]
        text = race_to_prompt(ra_indexed.loc[rid], horses)
        texts.append(text)
        valid_ids.append(rid)

    logger.info("推論開始: %d レース / batch_size=%d", len(texts), batch_size)
    all_scores = predict_batch(
        texts, model, tokenizer,
        batch_size=batch_size,
        race_ids=valid_ids,
    )

    rows = []
    skipped_races = []
    for rid, scores in zip(valid_ids, all_scores):
        if not scores:
            skipped_races.append(rid)
            continue
        for s in scores:
            rows.append({
                "race_id": rid,
                "horse_num": int(s["horse_num"]),
                "llm_ev_score": float(s["ev_score"]),
                "llm_rank_score": float(s["rank_score"]),
            })

    if skipped_races:
        logger.warning(
            "スコア空のレース %d 件をスキップ (全 %d 件中): %s",
            len(skipped_races), len(valid_ids), skipped_races[:5],
        )

    if not rows:
        logger.error(
            "全レースで LLM 推論が失敗しました。アダプタパスを確認してください: %s",
            ADAPTER_PATH,
        )
        return pd.DataFrame(columns=["race_id", "horse_num", "llm_ev_score", "llm_rank_score"])

    result_df = pd.DataFrame(rows)
    result_df["llm_ev_score"] = _normalize_per_race(result_df, "llm_ev_score")

    # 正規化後の合計が 1.0 になっているか検証
    sum_check = result_df.groupby("race_id")["llm_ev_score"].sum()
    bad_races = sum_check[(sum_check - 1.0).abs() > 1e-5].index.tolist()
    if bad_races:
        logger.error("正規化異常レース: %s (合計値: %s)", bad_races[:5], sum_check[bad_races[:5]].tolist())

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(
        "LLMスコア保存完了: %s  %d行 / %d races (スキップ: %d races)",
        OUTPUT_PATH, len(result_df), result_df["race_id"].nunique(), len(skipped_races),
    )

    return result_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    arg_day = sys.argv[1] if len(sys.argv) > 1 else None
    run_today_llm_inference(race_day=arg_day)
