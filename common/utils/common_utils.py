from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm_type
except ImportError:
    _tqdm_type = None  # type: ignore[misc, assignment]


def normalize_ketto_num(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return s.astype(object).where(s.notna(), np.nan)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    return df


def read_csv_optimized(
    path: str | Path,
    *,
    dtype=None,
    usecols: Iterable[str] | None = None,
    low_memory: bool = False,
    nullable_int_cols: Iterable[str] | None = None,
    chunksize: int | None = None,
    optimize: bool = False,
    fillna_int: dict[str, int] | None = None,
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    parquet_path = path.with_suffix(".parquet")
    read_kwargs = {
        "low_memory": low_memory,
        "dtype": dtype,
        "usecols": usecols,
    }
    read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}

    if prefer_parquet and parquet_path.exists():
        if chunksize is not None:
            raise ValueError("chunksize cannot be used with parquet input in read_csv_optimized.")
        try:
            df = pd.read_parquet(parquet_path, columns=usecols)
        except ImportError as e:
            raise RuntimeError(
                f"Parquet read requires pyarrow. Install with `pip install pyarrow` ({parquet_path})."
            ) from e
    elif chunksize is None:
        df = pd.read_csv(path, **read_kwargs)
    else:
        parts = [part for part in pd.read_csv(path, chunksize=chunksize, **read_kwargs)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    for col in nullable_int_cols or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col, value in (fillna_int or {}).items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(int(value)).astype(int)

    if optimize:
        df = optimize_dtypes(df)
    return df


def get_project_root(
    anchor: str | Path,
    *,
    marker: str = ".git",
    max_hops: int = 10,
) -> Path:
    current = Path(anchor).resolve()
    if current.is_file():
        current = current.parent
    for _ in range(max_hops):
        if (current / marker).exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found from anchor: {anchor}"
    )


def safe_parse_json_list(
    value: object,
    *,
    default: list[Any] | None = None,
    mode: str = "auto",
    context: str = "",
) -> list[Any]:
    fallback = [] if default is None else list(default)
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    if not isinstance(value, str):
        warnings.warn(f"[safe_parse_json_list] non-string value skipped ({context}): {value}")
        return fallback

    s = value.strip()
    if s == "":
        return fallback

    parsed: Any | None = None
    if mode in {"auto", "json"}:
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
    if parsed is None and mode in {"auto", "literal"} and s.startswith("[") and s.endswith("]"):
        # legacy literal format support without ast.literal_eval.
        try:
            parsed = json.loads(s.replace("'", "\""))
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        return parsed
    warnings.warn(f"[safe_parse_json_list] failed to parse list ({context}): {value}")
    return fallback


def load_standard_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    default_dtype = {"race_id": str, "horse_num": "Int64"}
    user_dtype = kwargs.pop("dtype", None)
    if isinstance(user_dtype, dict):
        merged_dtype = {**default_dtype, **user_dtype}
    elif user_dtype is None:
        merged_dtype = default_dtype
    else:
        merged_dtype = user_dtype
    return read_csv_optimized(path, dtype=merged_dtype, **kwargs)


def log_step(
    step_name: str,
    *,
    rows_in: int,
    rows_out: int,
    started_at: float,
    prefix: str = "[pipeline]",
) -> None:
    elapsed = time.perf_counter() - started_at
    msg = (
        f"{prefix} step={step_name} rows_in={rows_in} rows_out={rows_out} elapsed={elapsed:.2f}s"
    )
    if _tqdm_type is not None:
        try:
            _tqdm_type.write(msg)
            return
        except Exception:
            pass
    print(msg)
