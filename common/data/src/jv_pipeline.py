"""
Refactored pipeline entrypoint.

This module keeps public API compatibility while delegating reusable
components to jv_client / jv_parse / jv_state / jv_store / jv_schemas.
"""
from datetime import datetime
from pathlib import Path
import inspect
import struct
import sys

try:
    from . import legacy_get_data_impl as _legacy
    from .jv_client import JRAVANClient, _jv_com_return_code
    from .jv_parse import (
        parse_fixed_width,
        get_schema_fieldnames,
        _extract_record_key,
        _jockey_code_from_jc_raw_hex,
    )
    from .jv_schemas import (
        SCHEMAS,
        _RACE_KUBUN_PRIORITY,
        _FETCH_JRA_RACE_KUBUNS,
    )
    from .jv_state import (
        _default_state_path,
        _load_state,
        _save_state,
        _yesterday_end,
        _jv_normalize_bound,
        _jv_resolve_start_datetime,
    )
    from .jv_store import load_existing_dates, _load_existing_dates_without_pandas, save_to_csv
except ImportError:
    import legacy_get_data_impl as _legacy
    from jv_client import JRAVANClient, _jv_com_return_code
    from jv_parse import (
        parse_fixed_width,
        get_schema_fieldnames,
        _extract_record_key,
        _jockey_code_from_jc_raw_hex,
    )
    from jv_schemas import SCHEMAS, _RACE_KUBUN_PRIORITY, _FETCH_JRA_RACE_KUBUNS
    from jv_state import (
        _default_state_path,
        _load_state,
        _save_state,
        _yesterday_end,
        _jv_normalize_bound,
        _jv_resolve_start_datetime,
    )
    from jv_store import load_existing_dates, _load_existing_dates_without_pandas, save_to_csv


# Inject migrated components so legacy orchestration runs with refactored layers.
_legacy.JRAVANClient = JRAVANClient
_legacy._jv_com_return_code = _jv_com_return_code
_legacy.SCHEMAS = SCHEMAS
_legacy._RACE_KUBUN_PRIORITY = _RACE_KUBUN_PRIORITY
_legacy._FETCH_JRA_RACE_KUBUNS = _FETCH_JRA_RACE_KUBUNS
_legacy.parse_fixed_width = parse_fixed_width
_legacy.get_schema_fieldnames = get_schema_fieldnames
_legacy._extract_record_key = _extract_record_key
_legacy._jockey_code_from_jc_raw_hex = _jockey_code_from_jc_raw_hex
_legacy.load_existing_dates = load_existing_dates
_legacy._load_existing_dates_without_pandas = _load_existing_dates_without_pandas
_legacy.save_to_csv = save_to_csv
_legacy._default_state_path = _default_state_path
_legacy._load_state = _load_state
_legacy._save_state = _save_state
_legacy._yesterday_end = _yesterday_end
_legacy._jv_normalize_bound = _jv_normalize_bound
_legacy._jv_resolve_start_datetime = _jv_resolve_start_datetime


def fetch_realtime_data(dataspec: str, date_key: str, output_dir: str | None = None) -> int:
    """
    速報系 JVRTOpen(dataspec, key) を使って当日データを取得して保存する。
    保存形式は rec_id ごとに `realtime_<dataspec>_<rec_id>_<date>.csv`。
    """
    script_dir = Path(__file__).parent.parent.parent
    out_root = Path(output_dir) if output_dir else (script_dir / "data" / "output" / "realtime")
    out_root.mkdir(parents=True, exist_ok=True)

    client = JRAVANClient()
    total_saved = 0
    grouped = {}
    try:
        client.login()
        for raw_chunk in client.get_realtime_data(dataspec, date_key):
            if isinstance(raw_chunk, bytes):
                raw_chunk = raw_chunk.decode("cp932", "replace")
            if not raw_chunk:
                continue

            for line in raw_chunk.split("\n"):
                line = line.rstrip("\r\n")
                if len(line) < 2:
                    continue
                rec_id = line[:2]
                schema = SCHEMAS.get(rec_id)
                if not schema:
                    continue
                line_bytes = line.encode("cp932", "replace")
                parsed = parse_fixed_width(line_bytes, schema)
                parsed["raw_hex"] = line_bytes.hex()
                grouped.setdefault(rec_id, []).append(parsed)

        for rec_id, rows in grouped.items():
            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
            save_path = out_root / f"realtime_{dataspec.lower()}_{rec_id.lower()}_{date_key}.csv"
            save_to_csv(rows, str(save_path), fields, append=False)
            total_saved += len(rows)
    finally:
        client.close()
    return total_saved


def run_mode(mode: str = "predict", *, today: str | None = None) -> None:
    """
    train: 蓄積系 fetch_jra_data
    predict: 速報系 fetch_realtime_data（0B11/0B12/0B15）
    """
    m = str(mode).strip().lower()
    if m == "train":
        _legacy.fetch_jra_data("20230101000000", "20231231235959")
        return
    if m == "predict":
        d = today or datetime.now().strftime("%Y%m%d")
        for spec in ("0B11", "0B12", "0B15"):
            fetch_realtime_data(spec, d)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def _repo_root() -> Path:
    """``common/data/src/jv_pipeline.py`` からリポジトリルートへ。"""
    return Path(__file__).resolve().parent.parent.parent.parent


def dispatch_update_jra_data(**kwargs):
    """
    ``update_jra_data`` を実行する共通入口。

    - 引数は ``get_data.update_jra_data`` が受け取る名前のみ有効（余分は無視）。
    - ``None`` のキーは渡さない（既定の state / end_through に任せる）。
    - Windows かつ 64bit インタプリタでは ``main.jv_subprocess.run_with_32bit_python`` へ委譲。

    Returns:
        プロセス内: ``(start, end)`` のタプル。32bit 委譲時: ``subprocess.CompletedProcess``。
    """
    sig = inspect.signature(_legacy.update_jra_data)
    filtered = {
        k: v
        for k, v in kwargs.items()
        if k in sig.parameters and v is not None
    }
    if sys.platform == "win32" and struct.calcsize("P") * 8 == 64:
        try:
            from main.jv_subprocess import run_with_32bit_python
        except Exception:
            return _legacy.update_jra_data(**filtered)

        project_root = _repo_root()
        args = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
        code = (
            "from common.data.src.get_data import update_jra_data\n"
            f"update_jra_data({args})\n"
        )
        return run_with_32bit_python(project_root, code)

    return _legacy.update_jra_data(**filtered)


def run_accumulation_update(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    state_path: str | None = None,
    log_path: str | None = None,
    end_through: str = "today",
    start_default: str = "20150101000000",
) -> tuple[str, str] | object:
    """
    蓄積系一括更新の推奨エントリ（**常に incremental=False**）。

    ノートブック・スクリプトはここを import すると、Win64 時の 32bit 委譲も同じ経路に乗る。
    差分（INCR）が必要なときは ``update_jra_data(..., incremental=True)`` を直接使う。
    """
    return dispatch_update_jra_data(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
        state_path=state_path,
        log_path=log_path,
        end_through=end_through,
        start_default=start_default,
        incremental=False,
    )


def update_target_outputs_since(
    last_updated_date: str = "20260322",
    *,
    end_date_str: str | None = None,
    output_dir: str | None = None,
    state_path: str | None = None,
    log_path: str | None = None,
    incremental: bool = False,
):
    """
    指定日より後の蓄積系更新ラッパー。
    jv_pipeline 経由でも同じAPIを使えるように公開する。
    """
    d = str(last_updated_date).strip()
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"last_updated_date は YYYYMMDD 8桁で指定してください: {last_updated_date!r}")
    start_dt = datetime.strptime(d, "%Y%m%d")
    from datetime import timedelta

    start_date_str = (start_dt + timedelta(days=1)).strftime("%Y%m%d") + "000000"
    return dispatch_update_jra_data(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
        state_path=state_path,
        log_path=log_path,
        incremental=incremental,
    )


def _export_public_symbols():
    exported = []
    for name in dir(_legacy):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(_legacy, name)
        exported.append(name)
    return exported


__all__ = _export_public_symbols()
__all__.extend(
    [
        "fetch_realtime_data",
        "run_mode",
        "dispatch_update_jra_data",
        "run_accumulation_update",
        "update_target_outputs_since",
    ]
)
