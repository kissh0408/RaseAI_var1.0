#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JV-Link 用 32bit 子プロセス起動のみ（numpy / notebook_bootstrap に依存しない）。

RaceAI_var2.0.0 の main/jv_subprocess.py を var1.0 向けに移植したもの。
32bit 子プロセス起動の汎用インフラのみを提供し、市場情報とは無関係。
配置先が var2.0.0 では main/ 直下だったのに対し、var1.0 では
common/data/src/ 配下に置く（仕様書 docs/specs/2026-07-04-today-prediction-design.md
4-B節）。ロジックは無変更。
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_with_32bit_python(
    project_root: Path,
    code: str,
    *,
    capture_output: bool = True,
    timeout: int | None = 600,
) -> subprocess.CompletedProcess[str]:
    """
    Windows で `py` ランチャー経由で 32bit Python を起動する。

    Parameters
    ----------
    timeout : int | None
        秒単位のタイムアウト。None で無制限（推奨しない）。
        既定 600 秒（JV-Link の大容量取得でも十分な余裕）。
        タイムアウト超過時は RuntimeError を送出し、子プロセスを強制終了する。
    """
    root = str(Path(project_root).resolve())
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {root!r})\n"
        f"os.chdir({root!r})\n"
        + code
    )
    candidates = (
        ["py", "-3-32", "-c", script],
        ["py", "-3.14-32", "-c", script],
        ["py", "-3.12-32", "-c", script],
    )
    last_err: OSError | None = None
    for args in candidates:
        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError(
                    f"32bit 子プロセスが {timeout} 秒以内に完了しませんでした。"
                    " JV-Link の応答を確認してください。"
                )
            return subprocess.CompletedProcess(
                args=args,
                returncode=proc.returncode,
                stdout=stdout if capture_output else "",
                stderr=stderr if capture_output else "",
            )
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(
        "32bit Python が見つかりません。`py -3-32` を入れるか環境変数で 32bit python を指定してください。"
    ) from last_err
