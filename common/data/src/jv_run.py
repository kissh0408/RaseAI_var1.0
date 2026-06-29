"""
ノートブック・スクリプト向けの蓄積更新ヘルパー。

実体は ``jv_pipeline.dispatch_update_jra_data`` / ``update_target_outputs_since``。
ここでは **結果 dict + 任意の JSON 保存** だけ足す。
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from .jv_pipeline import update_target_outputs_since
except ImportError:
    from jv_pipeline import update_target_outputs_since


def default_state_log_paths(project_root: Path | None = None) -> tuple[Path, Path]:
    """``common/data/output/state`` 下の state / jsonl パス。"""
    if project_root is None:
        root = Path(__file__).resolve().parent.parent.parent.parent
    else:
        root = Path(project_root)
    st = root / "common" / "data" / "output" / "state" / "jv_last_update.json"
    lg = root / "common" / "data" / "output" / "state" / "jv_update_history.jsonl"
    return st, lg


def accumulation_update_since_with_report(
    last_updated_date: str,
    *,
    end_date_str: str | None = None,
    output_dir: str | None = None,
    state_path: str | Path | None = None,
    log_path: str | Path | None = None,
    project_root: Path | None = None,
    result_json_path: str | Path | None = None,
    incremental: bool = False,
) -> dict:
    """
    ``last_updated_date`` の翌日から蓄積更新し、結果を dict で返す。``result_json_path`` があれば JSON も書く。

    Returns:
        ``executed_at``, ``status``, ``start_used``, ``end_used``, ``state``, ``error`` などを含む dict。
    """
    st_default, lg_default = default_state_log_paths(project_root)
    st = Path(state_path) if state_path else st_default
    lg = Path(log_path) if log_path else lg_default

    result: dict = {
        "executed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "failed",
        "mode": "incremental" if incremental else "accumulation",
        "last_updated_date": last_updated_date,
        "start_used": None,
        "end_used": None,
        "state": None,
        "log_path": str(lg),
        "error": None,
    }

    try:
        ret = update_target_outputs_since(
            last_updated_date=last_updated_date,
            end_date_str=end_date_str,
            output_dir=output_dir,
            state_path=str(st) if st else None,
            log_path=str(lg) if lg else None,
            incremental=incremental,
        )

        if isinstance(ret, tuple) and len(ret) == 2:
            result["start_used"], result["end_used"] = ret[0], ret[1]
        elif isinstance(ret, subprocess.CompletedProcess):
            result["subprocess_returncode"] = ret.returncode

        result["status"] = "success"

        if st.exists():
            result["state"] = json.loads(st.read_text(encoding="utf-8"))

    except Exception as e:
        result["error"] = str(e)
        raise
    finally:
        if result_json_path is not None:
            p = Path(result_json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
