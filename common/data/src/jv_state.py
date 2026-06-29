import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from .jv_log import jv_warn
except ImportError:
    from jv_log import jv_warn


def _default_state_path(output_dir: Path) -> Path:
    return output_dir / "state" / "jv_last_update.json"

def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        jv_warn(f"state file broken or unreadable ({path}): {e}")
        return {}

def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write to avoid truncation/corruption on interruption.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)

def _yesterday_end() -> str:
    y = datetime.now() - timedelta(days=1)
    return y.strftime("%Y%m%d") + "235959"

def _jv_normalize_bound(s: str, *, is_end: bool) -> str:
    """YYYYMMDD → 時刻付き（開始 000000 / 終了 235959）にそろえる。"""
    t = str(s).strip()
    if len(t) == 8 and t.isdigit():
        return t + ("235959" if is_end else "000000")
    return t

def _jv_resolve_start_datetime(
    state: dict,
    *,
    default: str,
    also_use_last_update_date: bool,
) -> str:
    """
    増分更新の開始日時（YYYYMMDDHHMMSS）を state から決める。
    last_success_end があるときはその日の翌日 00:00:00。
    also_use_last_update_date のときだけ last_update_date もフォールバック。
    """
    last_success_end = state.get("last_success_end")
    if last_success_end:
        last_end_str = str(last_success_end).strip()
        if len(last_end_str) >= 8 and last_end_str[:8].isdigit():
            # Start from the beginning of the same day to avoid missing
            # same-day corrections or late-arriving updates.
            return last_end_str[:8] + "000000"
    if also_use_last_update_date:
        lud = state.get("last_update_date")
        if lud:
            try:
                last_date = datetime.strptime(str(lud).strip(), "%Y%m%d")
                next_date = last_date + timedelta(days=1)
                return next_date.strftime("%Y%m%d") + "000000"
            except ValueError:
                jv_warn(f"invalid last_update_date in state: {lud!r}")
    return _jv_normalize_bound(default, is_end=False)
