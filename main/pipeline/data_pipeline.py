"""データ取得・前処理系ラッパー。

JV-Link 経由の RA/SE 取得・蓄積更新・リアルタイムオッズ読み込みを担う。
モデル学習・戦略ロジックはここに含まない。
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# デフォルトの蓄積更新開始日：環境変数 DATA_LAST_UPDATED_DATE が未設定なら今日から30日前を使用
def _default_last_updated_date() -> str:
    env_val = os.environ.get("DATA_LAST_UPDATED_DATE", "")
    if env_val:
        return env_val
    return (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")


def update_accumulation_data(
    project_root: Path,
    last_updated_date: str = "",  # 空文字のときは _default_last_updated_date() で動的に決定
    *,
    end_date_str: str | None = None,
    output_dir: str | None = None,
    state_path: str | Path | None = None,
    log_path: str | Path | None = None,
    result_json_path: str | Path | None = None,
    incremental: bool = False,
) -> object:
    """
    蓄積データ更新のショートカット。
    実体は ``common.data.src.jv_run.accumulation_update_since_with_report``。
    """
    from common.data.src.jv_run import accumulation_update_since_with_report

    # 呼び出し側が明示的に渡さなかった場合（空文字）は動的デフォルトを使用
    resolved_date = last_updated_date or _default_last_updated_date()
    return accumulation_update_since_with_report(
        resolved_date,
        end_date_str=end_date_str,
        output_dir=output_dir,
        state_path=state_path,
        log_path=log_path,
        project_root=project_root,
        result_json_path=result_json_path,
        incremental=incremental,
    )


def run_today_ra_se_wh_weight_only(
    project_root: Path,
    race_day_yyyymmdd: str | None = None,
    *,
    dual_pass_se_then_ra: bool = True,
    target_kubun: str = "both",
) -> subprocess.CompletedProcess[str]:
    """
    当日（または指定日）の RA/SE を保存し、馬体重(WH)のみ速報として SE に反映する。
    馬場(WE)は取得・RA 反映しない。JV は 32bit 子プロセスで実行される。
    """
    from main.jv_subprocess import run_with_32bit_python

    day = (race_day_yyyymmdd or datetime.now().strftime("%Y%m%d")).strip()
    s = f"{day}000000"
    e = f"{day}235959"
    snippet = f"""from common.data.src.get_data import (
    get_race_data,
    fetch_wh_only,
    fetch_wh_from_0b11,
    merge_realtime_to_main_race,
)

s, e = {s!r}, {e!r}
get_race_data(
    start_date_str=s,
    end_date_str=e,
    race_day_yyyymmdd={day!r},
    dual_pass_se_then_ra={dual_pass_se_then_ra!r},
    target_kubun={target_kubun!r},
)
wh_0v12 = fetch_wh_only(s, e)
wh_0b11 = fetch_wh_from_0b11(s, e)
merge_res = merge_realtime_to_main_race(
    start_date_str=s,
    end_date_str=e,
    apply_we_to_ra=False,
    apply_wh_to_se=True,
)
print({{"wh_0v12": wh_0v12, "wh_0b11": wh_0b11, "merge": merge_res}})
"""
    cp = run_with_32bit_python(project_root, snippet)
    logger.info("subprocess returncode: %d", cp.returncode)
    if cp.stdout:
        logger.info("%s", cp.stdout)
    if cp.returncode != 0:
        logger.error("%s", cp.stderr or "")
        raise RuntimeError("RA/SE + WH weight-only JV run failed")
    return cp


def load_pair_odds_dicts(
    o2_odds_path: Path,
    o3_odds_path: Path,
) -> tuple[dict, dict]:
    """
    O2速報オッズ（馬連）と O3速報オッズ（ワイド）を CSV から読み込んで
    {(race_id, h1, h2): float} 形式の dict を返す。h1 < h2（ゼロ埋め2桁）。

    ファイルが存在しない場合は空 dict を返す（フォールバック: Harville推定）。
    O3 はペアの最低オッズ (odds_min_raw) を採用する。
    """
    def _load(path: Path, odds_col: str) -> dict:
        if not path.exists():
            return {}
        out: dict = {}
        try:
            import csv as _csv
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for r in _csv.DictReader(f):
                    rid = str(r.get("race_id", "")).strip()
                    h1 = str(r.get("horse_num_1", "")).zfill(2)
                    h2 = str(r.get("horse_num_2", "")).zfill(2)
                    raw = str(r.get(odds_col, "")).strip()
                    if not rid or not raw.isdigit() or int(raw) == 0:
                        continue
                    key = (rid, min(h1, h2), max(h1, h2))
                    # 10倍単位 (e.g. "00300" = 30.0)
                    out[key] = int(raw) / 10.0
        except Exception as exc:
            logger.warning("pair odds load failed (%s): %s", path.name, exc)
        return out

    q_dict = _load(o2_odds_path, "odds_raw")
    w_dict = _load(o3_odds_path, "odds_min_raw")
    logger.info("pair_odds 馬連 %d ペア / ワイド %d ペア 読み込み", len(q_dict), len(w_dict))
    return q_dict, w_dict
