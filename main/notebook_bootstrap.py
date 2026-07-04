#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jupyter main.ipynb 用の共通セットアップ（パス・32bit JV ラッパー・import）。

RaceAI_var2.0.0 の main/notebook_bootstrap.py を var1.0 向けに移植したもの。
仕様書: docs/specs/2026-07-04-today-prediction-design.md 4-A節。

ノート先頭で:

    %load_ext autoreload
    %autoreload 2
    from main.notebook_bootstrap import *

``update_jra_data`` は Windows かつ 64bit カーネルのとき、自動的に 32bit サブプロセス
（JV-Link 用）へ委譲する。明示的に 32bit だけ使う場合は ``update_jra_data_32bit``。

var2.0.0 との相違点:
- jv_subprocess は main/ 直下ではなく common/data/src/ 配下（4-B節）
- model_training.src.* への参照は存在しないため削除し、
  pure_rank/src の create_features.py / preprocess.py（bare import。
  pure_rank/src を sys.path に追加して "from create_features import ..." の
  形で読み込む。既存モジュール群と同じ import 規約に合わせるため）を使う
- main.main の load_models/predict/recommend_bets/format_recommendations は
  var2.0.0 固有の市場残差ロジックのため import しない。代わりに
  pure_rank/src/predict_today.py の当日予測関数を import する
- 新規: refresh_today_training_data()（HC/WC 当年再取得 + 前処理更新の
  ショートカット。4-A節参照）
"""

from __future__ import annotations

import csv
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment, misc]
try:
    import pandas as pd
except ImportError:  # Py3.14 32bit 等で pandas ホイールが無い環境
    pd = None  # type: ignore[assignment, misc]

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # オプション依存
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]

try:
    from IPython.display import display
except ImportError:

    def display(obj, **_kwargs):  # type: ignore[misc]
        print(obj)


def _ensure_paths() -> Path:
    # main.main は import しない（main.py が notebook_bootstrap を import する循環を避ける）
    p = Path.cwd().resolve()
    root = None
    for cand in [p, *p.parents]:
        if (cand / "main").is_dir() and (cand / "common").is_dir():
            root = cand
            break
    if root is None:
        raise RuntimeError(
            "プロジェクトルートが見つかりません (main/ と common/ が必要)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # 注意: pure_rank/src はここでは sys.path に追加しない（下記 _load_pure_rank_modules
    # 参照）。理由: pure_rank/src/common.py（bare モジュール）と、JV-Link 用の
    # ルート common/ パッケージ（common.data.src.get_data 等）が同名 "common" で
    # 衝突するため、両方が同時に sys.path 上にあると "from common.data.src.get_data
    # import ..." が "'common' is not a package" で失敗する
    # （実装時に回帰テストとは別の手動検証で発見。common/ にも pure_rank/src/common.py
    # にも __init__.py が無いための Python の名前解決順序の問題）。
    return root


PROJECT_ROOT = _ensure_paths()


def run_with_32bit_python(code: str, *, capture_output: bool = True, timeout: int | None = 600):
    from common.data.src.jv_subprocess import run_with_32bit_python as _run

    return _run(PROJECT_ROOT, code, capture_output=capture_output, timeout=timeout)


def _interpreter_is_64bit() -> bool:
    return struct.calcsize("P") * 8 == 64


def update_jra_data_32bit(start_date_str=None, end_date_str=None, *, timeout: int | None = 600, **kwargs):
    parts: list[str] = []
    if start_date_str is not None:
        parts.append(f"start_date_str={start_date_str!r}")
    if end_date_str is not None:
        parts.append(f"end_date_str={end_date_str!r}")
    parts.extend(f"{k}={v!r}" for k, v in kwargs.items())
    args_str = ", ".join(parts)
    snippet = f"from common.data.src.get_data import update_jra_data; update_jra_data({args_str})"
    return run_with_32bit_python(snippet, timeout=timeout)


def get_race_data_32bit(*, timeout: int | None = 600, **kwargs):
    supported = {
        "start_date_str",
        "end_date_str",
        "output_dir",
        "include_entry_kubun_1",
        "target_kubun",
        "race_day_yyyymmdd",
        "dual_pass_se_then_ra",
    }
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    unsupported = set(kwargs.keys()) - supported
    if unsupported:
        print(f"警告: 以下の引数は無視されます: {', '.join(sorted(unsupported))}")
    args = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
    snippet = (
        f"from common.data.src.get_data import get_race_data; get_race_data({args})"
    )
    return run_with_32bit_python(snippet, timeout=timeout)


def fetch_today_race_and_realtime(
    *,
    use_custom_range: bool = False,
    custom_start: str = "20260321000000",
    custom_end: str = "20260322235959",
    target_kubun: str = "both",
) -> None:
    """
    get_race_data（32bit）→ refresh_today_realtime_data（32bit）→ RA/SE 行数の簡易表示。
    """
    fetch_kwargs: dict = {"target_kubun": target_kubun}
    if use_custom_range:
        fetch_kwargs["start_date_str"] = custom_start
        fetch_kwargs["end_date_str"] = custom_end
    get_race_data_32bit(**fetch_kwargs)
    if use_custom_range:
        rt_call = (
            "from common.data.src.get_data import refresh_today_realtime_data; "
            f"refresh_today_realtime_data(start_date_str={custom_start!r}, "
            f"end_date_str={custom_end!r})"
        )
    else:
        rt_call = (
            "from common.data.src.get_data import refresh_today_realtime_data; "
            "refresh_today_realtime_data()"
        )
    run_with_32bit_python(rt_call)
    ra_path = PROJECT_ROOT / "main" / "data" / "race" / "race_ra.csv"
    se_path = PROJECT_ROOT / "main" / "data" / "race" / "race_se.csv"
    if pd is not None:
        ra_check = pd.read_csv(ra_path, dtype=str)
        se_check = pd.read_csv(se_path, dtype=str)
        print(
            "RA rows:",
            len(ra_check),
            "dates:",
            sorted(ra_check["month_day"].dropna().astype(str).unique().tolist())[:5],
        )
        print(
            "SE rows:",
            len(se_check),
            "dates:",
            sorted(se_check["month_day"].dropna().astype(str).unique().tolist())[:5],
        )
    else:
        def _count_and_month_days(path: Path) -> tuple[int, list[str]]:
            if not path.exists():
                return 0, []
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                rows = list(r)
            md_idx = None
            if r.fieldnames:
                for h in r.fieldnames:
                    if h == "month_day":
                        md_idx = h
                        break
            days: set[str] = set()
            if md_idx:
                for row in rows:
                    v = str(row.get(md_idx, "") or "").strip()
                    if v:
                        days.add(v)
            return len(rows), sorted(days)[:5]

        ra_n, ra_d = _count_and_month_days(ra_path)
        se_n, se_d = _count_and_month_days(se_path)
        print("RA rows:", ra_n, "dates:", ra_d)
        print("SE rows:", se_n, "dates:", se_d)


def run_today_se_ra_and_realtime_32bit(
    race_day_yyyymmdd: str | None = None,
    *,
    dual_pass_se_then_ra: bool = True,
    target_kubun: str = "both",
    output_dir: str | None = None,
    timeout: int | None = 600,
):
    """32bit 子プロセスで ``run_today_se_ra_and_realtime_merge`` を実行する。"""
    parts = [
        f"dual_pass_se_then_ra={dual_pass_se_then_ra!r}",
        f"target_kubun={target_kubun!r}",
    ]
    if race_day_yyyymmdd is not None:
        parts.insert(0, f"race_day_yyyymmdd={race_day_yyyymmdd!r}")
    if output_dir is not None:
        parts.append(f"output_dir={output_dir!r}")
    arg = ", ".join(parts)
    snippet = (
        "from common.data.src.get_data import run_today_se_ra_and_realtime_merge; "
        f"run_today_se_ra_and_realtime_merge({arg})"
    )
    return run_with_32bit_python(snippet, timeout=timeout)


def run_today_se_ra_and_realtime(
    race_day_yyyymmdd: str | None = None,
    *,
    dual_pass_se_then_ra: bool = True,
    target_kubun: str = "both",
    output_dir: str | None = None,
    timeout: int | None = 600,
):
    """
    当日（または指定日）の RA/SE を main/data/race に保存し、続けて速報取得・マージ。

    1. ``get_race_data`` … 既定で JVOpen を2回（先に SE、次に RA）。
    2. ``refresh_today_realtime_data`` … 天候・馬場・馬体重などを RA/SE に反映。
    3. ``refresh_today_odds_data`` … 0B31 速報単勝オッズを取得し RA/SE に反映
       （race_se.csv の odds 列を更新するが、当日予測パイプラインの
       特徴量化ステップ（pure_rank/src/today_adapter.py）で明示的に drop する）。
    4. ``realtime_we_fetcher_v2`` … 取得済み WE CSV から開催単位スナップショットを再生成。

    注意: このセルの実行には JV-Link 接続環境（実機 Windows・32bit Python・
    JRA-VAN 契約回線）が必要です。implementer は自動実行しません。

    Parameters
    ----------
    timeout : int | None
        32bit 子プロセスのタイムアウト秒数（既定 600 秒）。応答が遅い場合は
        `run_today_se_ra_and_realtime("20260705", timeout=1800)` のように延長すること。
    """
    run_date = (race_day_yyyymmdd or datetime.now().strftime("%Y%m%d")).strip()
    if sys.platform == "win32" and _interpreter_is_64bit():
        result = run_today_se_ra_and_realtime_32bit(
            race_day_yyyymmdd,
            dual_pass_se_then_ra=dual_pass_se_then_ra,
            target_kubun=target_kubun,
            output_dir=output_dir,
            timeout=timeout,
        )
        # WE v2 は CSV 再集計のみのため、64bit 側で即時処理して
        # 追加の 32bit サブプロセス起動コストを避ける。
        _run_realtime_we_v2_snapshot_inproc(run_date)
        return result
    from common.data.src.get_data import run_today_se_ra_and_realtime_merge

    result = run_today_se_ra_and_realtime_merge(
        race_day_yyyymmdd,
        dual_pass_se_then_ra=dual_pass_se_then_ra,
        target_kubun=target_kubun,
        output_dir=output_dir,
    )
    _run_realtime_we_v2_snapshot_inproc(run_date)
    return result


def _run_realtime_we_v2_snapshot_inproc(run_date: str):
    we_path = PROJECT_ROOT / "common" / "data" / "output" / "realtime_we" / "we.csv"
    if not we_path.exists():
        print(f"WE v2 snapshot skip: we.csv not found ({we_path})")
        return None
    try:
        from common.data.src.realtime_we_fetcher_v2 import (
            build_course_snapshot,
            load_we_events_from_csv,
        )
    except Exception as e:
        print(f"WE v2 snapshot skip: import failed ({e})")
        return None

    events = load_we_events_from_csv(we_path)
    snapshot = build_course_snapshot(events)
    out_dir = PROJECT_ROOT / "common" / "data" / "output" / "realtime_we_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = out_dir / f"we_events_{run_date}_{ts}.csv"
    snapshot_path = out_dir / f"we_course_snapshot_{run_date}_{ts}.csv"
    latest_events = out_dir / "we_events_latest.csv"
    latest_snapshot = out_dir / "we_course_snapshot_latest.csv"
    events_fields = [
        "record_id",
        "data_kubun",
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "race_num",
        "announce_ddhhmmss",
        "change_id",
        "weather_code",
        "turf_condition",
        "dirt_condition",
        "record_separator",
        "raw_hex",
        "source_line",
        "seq",
    ]
    snapshot_fields = [
        "year",
        "month_day",
        "course_code",
        "kai",
        "nichi",
        "last_announce_ddhhmmss",
        "last_change_id",
        "event_count",
        "weather_code",
        "turf_condition",
        "dirt_condition",
    ]
    for path, rows, fieldnames in (
        (events_path, events, events_fields),
        (latest_events, events, events_fields),
        (snapshot_path, snapshot, snapshot_fields),
        (latest_snapshot, snapshot, snapshot_fields),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)
    print(
        "WE v2 snapshot generated:",
        json.dumps(
            {
                "date": run_date,
                "events": len(events),
                "courses": len(snapshot),
                "events_path": str(events_path),
                "snapshot_path": str(snapshot_path),
            },
            ensure_ascii=False,
        ),
    )
    return {"events": len(events), "courses": len(snapshot)}


from common.data.src.get_data import (
    fetch_bt_only,
    fetch_hc_only,
    fetch_hn_only,
    fetch_jra_data,
    fetch_ra_only,
    fetch_race_only,
    fetch_se_only,
    fetch_sk_only,
    fetch_wc_only,
    fetch_we_only,
    fetch_wh_only,
    get_race_data,
    refresh_today_realtime_data,
    run_today_se_ra_and_realtime_merge,
    update_jra_data as _update_jra_data_inproc,
)


def _training_unavailable(*_a, **_kw):
    raise RuntimeError(
        "pure_rank/src の依存（pandas/numpy/lightgbm 等）が利用できません。"
        " 特徴量・前処理・推論を使うカーネルでは 64bit 等で依存を入れてください。"
    )


def _load_pure_rank_modules() -> dict:
    """create_features / preprocess / today_adapter / predict_today を import する。

    重要な注意（"common" 名前衝突の回避）:
    pure_rank/src/common.py（bare モジュール。FORBIDDEN_MARKET_COLS 等を持つ）と、
    JV-Link 用のルート common/ パッケージ（common.data.src.get_data 等。本ファイル
    冒頭で既に import 済み）は、どちらも __init__.py の無いディレクトリ/ファイルで
    同名 "common" のため、両方が同時に sys.path 上にあると Python の import 解決が
    どちらか一方に固定され、他方が壊れる（"'common' is not a package" 等）。

    対処: pure_rank/src 配下のモジュールを import する直前だけ、一時的に
    (1) プロジェクトルートを sys.path から外し、(2) sys.modules に残っている
    "common"/"common.*" のキャッシュを退避し、(3) pure_rank/src を sys.path 先頭に
    追加してから import する。import 完了後（finally節）に、pure_rank/src の
    import 中に新規生成された "common"/"common.*" の sys.modules エントリを削除し、
    (2) で退避しておいたルート common/ パッケージ側のキャッシュを sys.modules に
    書き戻したうえで、プロジェクトルートを sys.path に戻す。この sys.modules
    の復元が漏れると、この関数の呼び出し後（例: main.notebook_bootstrap を
    import した直後）に "from common.data.src.xxx import ..." のような遅延 import
    が "'common' is not a package" で失敗する（sys.path だけ戻しても、既に
    差し替わった sys.modules["common"] のキャッシュはそのままのため）。
    """
    import importlib

    root_str = str(PROJECT_ROOT)
    pure_rank_src = str(PROJECT_ROOT / "pure_rank" / "src")

    had_root = root_str in sys.path
    if had_root:
        sys.path.remove(root_str)
    if pure_rank_src not in sys.path:
        sys.path.insert(0, pure_rank_src)

    saved_modules = {
        k: v for k, v in list(sys.modules.items())
        if k == "common" or k.startswith("common.")
    }
    for k in saved_modules:
        del sys.modules[k]

    try:
        mods = {
            "common": importlib.import_module("common"),
            "create_features": importlib.import_module("create_features"),
            "preprocess": importlib.import_module("preprocess"),
            # evaluate/predict は predict_today.run_today_predictions() 内部で
            # 遅延 import される（"from evaluate import ensemble_predict, load_models"等）。
            # この関数呼び出しは main.notebook_bootstrap の import が完了しsys.modules["common"]
            # がルート common/ パッケージへ復元された「後」に発生するため、ここで先に
            # import して sys.modules にキャッシュしておかないと、遅延 import 時点で
            # evaluate.py/predict.py の "from common import ..." がルート common/ を
            # 参照してしまい ImportError になる（実際に発生を確認済みのバグ）。
            "evaluate": importlib.import_module("evaluate"),
            "predict": importlib.import_module("predict"),
            "today_adapter": importlib.import_module("today_adapter"),
            "predict_today": importlib.import_module("predict_today"),
        }
        return mods
    finally:
        # pure_rank/src の import 中に新たに作られた "common"/"common.*"
        # （pure_rank/src/common.py 由来のキャッシュ）を破棄し、退避しておいた
        # ルート common/ パッケージ側のキャッシュ（saved_modules）を書き戻す。
        # これを怠ると、この関数の呼び出し後に sys.modules["common"] が
        # pure_rank/src 版のままになり、以後の "from common.data.src.xxx import ..."
        # が "'common' is not a package" で失敗する（本関数末尾で sys.path だけ
        # 戻しても sys.modules のキャッシュは戻らないため）。
        for k in list(sys.modules.keys()):
            if k == "common" or k.startswith("common."):
                del sys.modules[k]
        sys.modules.update(saved_modules)
        if had_root and root_str not in sys.path:
            sys.path.append(root_str)


try:
    _pr_mods = _load_pure_rank_modules()
except ImportError as _e:
    create_main_features = _training_unavailable  # type: ignore[misc]
    preprocess_se = preprocess_ra = preprocess_sk = _training_unavailable  # type: ignore[misc]
    preprocess_hc = preprocess_wc = preprocess_hr = _training_unavailable  # type: ignore[misc]
    preprocess_all = _training_unavailable  # type: ignore[misc]
    build_today_features = run_today_predictions = write_predictions = _training_unavailable  # type: ignore[misc]
    load_config = _training_unavailable  # type: ignore[misc]
    _pure_rank_import_error = _e
    import warnings

    warnings.warn(
        f"notebook_bootstrap: pure_rank/src モジュールの import に失敗しました: {_e}",
        stacklevel=1,
    )
else:
    _pure_rank_import_error = None
    create_main_features = _pr_mods["create_features"].main
    preprocess_se = _pr_mods["preprocess"].preprocess_se
    preprocess_ra = _pr_mods["preprocess"].preprocess_ra
    preprocess_sk = _pr_mods["preprocess"].preprocess_sk
    preprocess_hc = _pr_mods["preprocess"].preprocess_hc
    preprocess_wc = _pr_mods["preprocess"].preprocess_wc
    preprocess_hr = _pr_mods["preprocess"].preprocess_hr
    preprocess_all = _pr_mods["preprocess"].main
    # var2.0.0 の main.main.load_models/predict/recommend_bets/format_recommendations
    # （市場残差ロジック固有）は var1.0 では使わない。代わりに当日予測専用モジュール
    # pure_rank/src/predict_today.py（仕様書 4-D節）の関数を使う。
    build_today_features = _pr_mods["predict_today"].build_today_features
    run_today_predictions = _pr_mods["predict_today"].run_today_predictions
    write_predictions = _pr_mods["predict_today"].write_predictions
    # train_config.json ローダー。today_adapter.build_today_merged() や
    # predict_today.build_today_features()/run_today_predictions() に渡す
    # cfg 引数はこの load_config() の戻り値を想定している。
    load_config = _pr_mods["common"].load_config


def update_jra_data(start_date_str=None, end_date_str=None, **kwargs):
    """
    JRA-VAN 一括更新。Anaconda 等の 64bit カーネルでは JV-Link COM が使えないため、
    Windows では 32bit サブプロセスに任せる。32bit カーネルまたは非 Windows ではプロセス内で実行。
    """
    if sys.platform == "win32" and _interpreter_is_64bit():
        return update_jra_data_32bit(start_date_str, end_date_str, **kwargs)
    return _update_jra_data_inproc(start_date_str, end_date_str, **kwargs)


def refresh_today_training_data_32bit(cur_year: int | None = None, *, timeout: int | None = 600):
    """HC/WC 当年分の fetch のみを 32bit 子プロセスで実行する（JV-Link 接続が必要）。"""
    year = cur_year or datetime.now().year
    snippet = (
        "from common.data.src.get_data import fetch_hc_only, fetch_wc_only\n"
        f"fetch_hc_only({year}, {year})\n"
        f"fetch_wc_only({year}, {year})\n"
    )
    return run_with_32bit_python(snippet, timeout=timeout)


def refresh_today_training_data(cur_year: int | None = None, *, timeout: int | None = 600) -> dict:
    """
    HC/WC（坂路調教・ウッドチップ調教）当年分を再取得し、前処理を更新するショートカット。

    当日予測には直近の調教データ（trn_hc_*/trn_wc_* 12列）が必須だが、
    run_today_se_ra_and_realtime() は RA/SE + 速報 + オッズのみを取得し
    HC/WC はカバーしない（仕様書 2-7節）。このため当日予測の Step 0（事前準備）
    として本関数を別途呼び出す必要がある。

    処理内容:
    1. fetch_hc_only(cur_year, cur_year) / fetch_wc_only(cur_year, cur_year)
       （JV-Link 接続が必要。64bit カーネルでは自動的に 32bit サブプロセスへ委譲）
    2. preprocess_hc / preprocess_wc で HC_preprocessed.parquet / WC_preprocessed.parquet を更新
       （pandas が必要。64bit カーネル側で実行する想定）

    Parameters
    ----------
    timeout : int | None
        32bit 子プロセスのタイムアウト秒数（既定 600 秒）。JRA-VAN の応答が遅い場合や
        HC/WC のデータ量が大きい場合は、例えば `timeout=1800` のように延長すること。

    Returns
    -------
    dict: {"hc_rows": int, "wc_rows": int}
    """
    year = cur_year or datetime.now().year
    print(f"[refresh_today_training_data] fetching HC/WC for year={year} (timeout={timeout}s) ...")

    if sys.platform == "win32" and _interpreter_is_64bit():
        refresh_today_training_data_32bit(year, timeout=timeout)
    else:
        from common.data.src.get_data import fetch_hc_only, fetch_wc_only

        fetch_hc_only(year, year)
        fetch_wc_only(year, year)

    if pd is None:
        raise RuntimeError(
            "pandas が利用できないカーネルでは前処理(preprocess_hc/preprocess_wc)を実行できません。"
            " 64bit の pandas 入りカーネルで再度呼び出してください。"
        )
    if preprocess_hc is _training_unavailable or preprocess_wc is _training_unavailable:
        raise RuntimeError(
            "preprocess_hc/preprocess_wc が import できていません。"
            " pure_rank/src の依存関係を確認してください。"
        )

    from preprocess import load_config as _load_preprocess_config

    cfg = _load_preprocess_config()
    hc_dir = Path(cfg["data"]["hc_dir"])
    wc_dir = Path(cfg["data"]["wc_dir"])
    dst_dir = PROJECT_ROOT / cfg["data"]["preprocessed_dir"]

    hc_df = preprocess_hc(hc_dir, dst_dir / "HC_preprocessed.parquet")
    wc_df = preprocess_wc(wc_dir, dst_dir / "WC_preprocessed.parquet")
    print(
        f"[refresh_today_training_data] HC rows={len(hc_df):,}, WC rows={len(wc_df):,}"
    )
    return {"hc_rows": len(hc_df), "wc_rows": len(wc_df)}


if pd is not None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 200)

__all__ = [
    "PROJECT_ROOT",
    "display",
    "json",
    "np",
    "pd",
    "plt",
    "sns",
    "run_with_32bit_python",
    "update_jra_data_32bit",
    "get_race_data_32bit",
    "fetch_today_race_and_realtime",
    "run_today_se_ra_and_realtime_32bit",
    "run_today_se_ra_and_realtime",
    "run_today_se_ra_and_realtime_merge",
    "update_jra_data",
    "get_race_data",
    "fetch_jra_data",
    "fetch_race_only",
    "fetch_hc_only",
    "fetch_wc_only",
    "fetch_se_only",
    "fetch_sk_only",
    "fetch_hn_only",
    "fetch_bt_only",
    "fetch_ra_only",
    "fetch_we_only",
    "fetch_wh_only",
    "refresh_today_realtime_data",
    "refresh_today_training_data",
    "refresh_today_training_data_32bit",
    "preprocess_all",
    "preprocess_se",
    "preprocess_ra",
    "preprocess_sk",
    "preprocess_hc",
    "preprocess_wc",
    "preprocess_hr",
    "create_main_features",
    "build_today_features",
    "run_today_predictions",
    "write_predictions",
    "load_config",
]
