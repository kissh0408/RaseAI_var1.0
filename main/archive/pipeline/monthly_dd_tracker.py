"""本番用月次ドローダウントラッカー。

当月の確定損益を追跡し、monthly_drawdown_limit を超えたら当日のベットを停止する。

バックテスト用の apply_monthly_drawdown_filter()（strategy/src/ev_filters.py）とは別物。
こちらは「過去の確定損益をファイルに記録して、当日推論時にガードとして機能する」本番専用実装。

設計原則:
  - レース当日（推論時点）では invested のみ記録できる。
  - returned（回収額）はレース後（翌日以降の結果確認時）に手動 or 自動で更新する。
  - monthly_pnl_tracker.json は main/results/ に保存し、Git管理外とする（.gitignore 推奨）。
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# トラッカーファイルの保存先（main/results/ 配下）
TRACKER_FILE: Path = Path(__file__).parent.parent / "results" / "monthly_pnl_tracker.json"


def load_monthly_pnl() -> dict:
    """月次P&Lトラッカーファイルを読み込む。存在しない場合は空の構造を返す。

    Returns
    -------
    dict
        {"created_at": str, "note": str, "months": {YYYY-MM: {daily: [...]}}} 形式の辞書。
    """
    if not TRACKER_FILE.exists():
        return {
            "created_at": date.today().isoformat(),
            "note": (
                "本番月次P&Lトラッカー。monthly_drawdown_limit=-0.08 の監視用。"
                "invested は推論時に記録、returned はレース確定後に更新する。"
            ),
            "months": {},
        }
    try:
        with TRACKER_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "months" not in data:
            logger.warning(
                "[monthly_dd_tracker] トラッカーファイルの形式が不正です。空の構造にリセットします: %s",
                TRACKER_FILE,
            )
            return {"created_at": date.today().isoformat(), "note": "", "months": {}}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(
            "[monthly_dd_tracker] トラッカーファイルの読み込みに失敗しました (%s): %s",
            TRACKER_FILE, e,
        )
        return {"created_at": date.today().isoformat(), "note": "", "months": {}}


def save_monthly_pnl(data: dict) -> None:
    """月次P&Lトラッカーファイルを保存する。

    Parameters
    ----------
    data : dict
        load_monthly_pnl() が返す形式の辞書。
    """
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TRACKER_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(
            "[monthly_dd_tracker] トラッカーファイルの保存に失敗しました (%s): %s",
            TRACKER_FILE, e,
        )


def get_current_month_key() -> str:
    """当月のキー（YYYY-MM形式）を返す。

    Returns
    -------
    str
        例: "2026-06"
    """
    return date.today().strftime("%Y-%m")


def record_daily_pnl(
    race_date: str,
    invested: float,
    returned: float,
    *,
    n_recommendations: int | None = None,
    n_hits: int | None = None,
) -> None:
    """当日の投資額と回収額を記録する。

    推論時（レース前）は returned=0.0 で呼び出し、
    レース確定後に returned の実額で再呼び出しして上書き更新する。

    Parameters
    ----------
    race_date : str
        レース日付（YYYY-MM-DD 形式）。
    invested : float
        当日の総投資額（推奨リストの suggested_stake 合計）。0 以上の値。
    returned : float
        当日の総回収額（レース後に確定）。推論時点では 0.0 でよい。
    n_recommendations : int | None
        当日の推奨件数（的中率モニタリング用）。
    n_hits : int | None
        当日の的中件数（レース確定後に更新）。
    """
    if invested < 0:
        raise ValueError(f"invested は 0 以上である必要があります: {invested}")
    if returned < 0:
        raise ValueError(f"returned は 0 以上である必要があります: {returned}")

    # race_date から月キーを導出（YYYY-MM-DD → YYYY-MM）
    try:
        ym_key = race_date[:7]  # "YYYY-MM"
    except (IndexError, TypeError) as e:
        raise ValueError(f"race_date の形式が不正です（YYYY-MM-DD が必要）: {race_date!r}") from e

    data = load_monthly_pnl()
    if ym_key not in data["months"]:
        data["months"][ym_key] = {"daily": []}

    daily_list: list[dict] = data["months"][ym_key]["daily"]

    entry: dict = {
        "date": race_date,
        "invested": float(invested),
        "returned": float(returned),
        "profit": float(returned - invested),
    }
    if n_recommendations is not None:
        entry["n_recommendations"] = int(n_recommendations)
    if n_hits is not None:
        entry["n_hits"] = int(n_hits)
    if entry.get("n_recommendations", 0) > 0 and "n_hits" in entry:
        entry["hit_rate"] = float(entry["n_hits"]) / float(entry["n_recommendations"])

    # 同一日付のエントリーがあれば上書き（returned 更新対応）
    existing = next((e for e in daily_list if e.get("date") == race_date), None)
    if existing is not None:
        logger.info(
            "[monthly_dd_tracker] %s のP&Lを更新します: invested=%.0f, returned=%.0f",
            race_date, invested, returned,
        )
        existing.update(entry)
    else:
        daily_list.append(entry)
        logger.info(
            "[monthly_dd_tracker] %s のP&Lを記録しました: invested=%.0f, returned=%.0f",
            race_date, invested, returned,
        )

    save_monthly_pnl(data)


def check_monthly_dd_limit(
    initial_bankroll: float,
    monthly_drawdown_limit: float,
) -> tuple[bool, float]:
    """当月の累積損益が monthly_drawdown_limit を超えているか確認する。

    当月の daily エントリーが存在しない場合、または current_month_key が months に
    存在しない場合は (False, 0.0) を返す（安全側フォールバック）。

    Parameters
    ----------
    initial_bankroll : float
        初期資金（月次損益比率の分母）。
    monthly_drawdown_limit : float
        月次ドローダウン上限（負の比率。例: -0.08 = 月間損失が資金の 8% 超で停止）。

    Returns
    -------
    tuple[bool, float]
        (is_limit_exceeded, current_monthly_pnl_rate)
        is_limit_exceeded=True なら当日のベットを停止すること。
    """
    if initial_bankroll <= 0:
        logger.warning(
            "[monthly_dd_tracker] initial_bankroll が 0 以下です (%s)。DDチェックをスキップします。",
            initial_bankroll,
        )
        return False, 0.0

    data = load_monthly_pnl()
    ym_key = get_current_month_key()

    month_data = data.get("months", {}).get(ym_key)
    if month_data is None or not month_data.get("daily"):
        # 当月データなし → 安全側フォールバック（ベット継続）
        logger.debug(
            "[monthly_dd_tracker] %s のデータが存在しません。月次DDチェックをスキップします。",
            ym_key,
        )
        return False, 0.0

    daily_list: list[dict] = month_data["daily"]

    # 当月の累積損益（profit 列がある場合はそれを使用、なければ returned - invested を計算）
    total_profit: float = 0.0
    for entry in daily_list:
        if "profit" in entry:
            total_profit += float(entry["profit"])
        else:
            inv = float(entry.get("invested", 0.0))
            ret = float(entry.get("returned", 0.0))
            total_profit += ret - inv

    current_rate = total_profit / initial_bankroll

    is_exceeded = current_rate < monthly_drawdown_limit

    logger.info(
        "[monthly_dd_tracker] 当月(%s) 累積損益: %.0f円 / 損益率: %.1f%% / 閾値: %.1f%%",
        ym_key,
        total_profit,
        current_rate * 100,
        monthly_drawdown_limit * 100,
    )

    if is_exceeded:
        logger.warning(
            "[monthly_dd_tracker] 当月累積損失率 %.1f%% が閾値 %.1f%% を超過。",
            current_rate * 100,
            monthly_drawdown_limit * 100,
        )

    return is_exceeded, current_rate


def check_hit_rate_and_roi_alerts(
    *,
    hit_rate_floor: float = 0.20,
    roi_floor: float = 1.15,
    min_bets_for_alert: int = 20,
) -> dict:
    """当月の的中率・ROI が下限を下回る場合に警告情報を返す（DE-新1）。"""
    data = load_monthly_pnl()
    ym_key = get_current_month_key()
    month_data = data.get("months", {}).get(ym_key, {})
    daily_list: list[dict] = month_data.get("daily", [])

    total_inv = sum(float(e.get("invested", 0)) for e in daily_list)
    total_ret = sum(float(e.get("returned", 0)) for e in daily_list)
    total_rec = sum(int(e.get("n_recommendations", 0)) for e in daily_list)
    total_hits = sum(int(e.get("n_hits", 0)) for e in daily_list)

    hit_rate = (total_hits / total_rec) if total_rec > 0 else None
    roi = (total_ret / total_inv) if total_inv > 0 else None

    alerts: list[str] = []
    if total_rec >= min_bets_for_alert and hit_rate is not None and hit_rate < hit_rate_floor:
        alerts.append(f"hit_rate {hit_rate:.1%} < floor {hit_rate_floor:.0%}")
    if total_inv > 0 and roi is not None and roi < roi_floor:
        alerts.append(f"ROI {roi:.1%} < floor {roi_floor:.0%}")

    for msg in alerts:
        logger.warning("[monthly_dd_tracker] ALERT: %s", msg)

    return {
        "month": ym_key,
        "n_recommendations": total_rec,
        "n_hits": total_hits,
        "hit_rate": hit_rate,
        "roi": roi,
        "alerts": alerts,
    }


def get_monthly_summary(month_key: str | None = None) -> pd.DataFrame:
    """指定月（デフォルト: 当月）のP&L日次サマリーを DataFrame で返す。

    主に Notebook でのデバッグ・確認用。

    Parameters
    ----------
    month_key : str | None
        YYYY-MM 形式の月キー。None の場合は当月。

    Returns
    -------
    pd.DataFrame
        date / invested / returned / profit / cumulative_profit 列を持つ DataFrame。
        データが存在しない場合は空の DataFrame を返す。
    """
    data = load_monthly_pnl()
    key = month_key or get_current_month_key()
    month_data = data.get("months", {}).get(key, {})
    daily_list = month_data.get("daily", [])

    if not daily_list:
        return pd.DataFrame(
            columns=["date", "invested", "returned", "profit", "cumulative_profit"]
        )

    df = pd.DataFrame(daily_list)
    if "profit" not in df.columns:
        df["profit"] = df.get("returned", 0.0) - df.get("invested", 0.0)

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    df["cumulative_profit"] = df["profit"].cumsum()
    return df
