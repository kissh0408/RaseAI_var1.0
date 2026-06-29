"""
JV-Link 取得まわりのコンソール出力を揃える。

環境変数 ``RACEAI_JV_LOG``:
  - ``normal``（既定）: セクション・結果・警告・エラー（接続の細かい行は省略）
  - ``verbose``: 接続・JVInit・切断などもすべて表示
  - ``quiet``: 警告・エラー・保存行のみ
"""

from __future__ import annotations

import os

__all__ = [
    "jv_level",
    "jv_banner",
    "jv_section",
    "jv_info",
    "jv_verbose",
    "jv_warn",
    "jv_err",
    "jv_saved",
]


def jv_level() -> str:
    return os.environ.get("RACEAI_JV_LOG", "normal").strip().lower()


def _verbose() -> bool:
    return jv_level() == "verbose"


def _quiet() -> bool:
    return jv_level() == "quiet"


def jv_banner(title: str, subtitle: str | None = None) -> None:
    if _quiet():
        return
    w = 66
    print()
    print("=" * w)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * w)


def jv_section(title: str, detail: str | None = None) -> None:
    if _quiet():
        return
    line = f"--- {title}"
    if detail:
        line += f"  |  {detail}"
    print()
    print(line)


def jv_info(msg: str) -> None:
    if not _quiet():
        print(msg)


def jv_verbose(msg: str) -> None:
    if _verbose():
        print(msg)


def jv_warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def jv_err(msg: str) -> None:
    print(f"  [ERR]  {msg}")


def jv_saved(path: str, n: int) -> None:
    """保存結果は quiet でも出す（空更新の確認用）。"""
    print(f"  saved: {n} rows -> {path}")
