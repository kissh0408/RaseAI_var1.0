import win32com.client
import pythoncom
import csv
import os
import traceback
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from .jv_log import jv_err, jv_info, jv_verbose, jv_warn
except ImportError:
    from jv_log import jv_err, jv_info, jv_verbose, jv_warn


def _jv_com_return_code(result):
    """
    JV-Link の Python/COM 戻り値から数値リターンコードを取り出す。
    インタフェース仕様では Long だが、pywin32 では tuple の先頭がリターンコードになることが多い
    （例: JVOpen → (0, 48, 0, '...') で先頭 0 が成功。0 以上を成功とみなす）。
    """
    if result is None:
        return 0
    if isinstance(result, tuple):
        if not result:
            return 0
        try:
            return int(result[0])
        except (TypeError, ValueError):
            return -1
    try:
        return int(result)
    except (TypeError, ValueError):
        return -1

class JRAVANClient:
    def __init__(self):
        self.jv_link = None
        self._initialize_link()

    def _initialize_link(self):
        try:
            # 32bit Python 推奨（JRA-VAN 公式）。64bit では DllSurrogate 等が必要な場合あり。
            self.jv_link = win32com.client.Dispatch("JVDTLab.JVLink")
            jv_verbose("JV-Link instance created via Dispatch.")
        except Exception as e:
            jv_err(f"Failed to create JV-Link instance: {e}")
            jv_info(
                "  Hint: 64-bit Python の場合は DllSurrogate 登録が必要なことがあります。"
            )
            raise e

    def login(self):
        """
        JVInit による認証処理
        """
        sid = "UNKNOWN"  # 定額利用キーが設定されていれば UNKNOWN でOK
        try:
            ret_code = _jv_com_return_code(self.jv_link.JVInit(sid))
            if ret_code < 0:
                raise Exception(f"JVInit failed: Code {ret_code}")
            jv_verbose("JVInit (Login) Successful.")
        except Exception as e:
            jv_err(f"Login failed: {e}")
            raise

    def close(self):
        """
        JVClose による切断（二重呼び出しでも落ちないようにする）
        """
        if self.jv_link:
            try:
                self.jv_link.JVClose()
            except Exception:
                pass
            jv_verbose("JV-Link session closed.")

    def open_accumulation(self, dataspec, start_time, option=2, end_time=None):
        """蓄積系 JVOpen を実行し、return code を返す。"""
        last_time = end_time if end_time else ""
        # Requesting 行は jv_processors 側で1行にまとめて出す（二重表示を避ける）
        res = self.jv_link.JVOpen(dataspec, start_time, option, 0, 0, last_time)
        return _jv_com_return_code(res), res

    def open_realtime(self, dataspec, key):
        """速報系 JVRTOpen を実行し、return code を返す。"""
        jv_verbose(f"Requesting realtime {dataspec} (Key: {key})...")
        res = self.jv_link.JVRTOpen(dataspec, key)
        return _jv_com_return_code(res), res

    def read(self, buff_size=10000000):
        """JVRead ループを共通化したジェネレータ。"""
        buff = ""
        fname = ""
        retry_count = 0
        while True:
            try:
                read_res = self.jv_link.JVRead(buff, buff_size, fname)
            except Exception as e:
                jv_err(f"JVRead Exception: {e}")
                break

            status = 0
            raw_data = ""
            if isinstance(read_res, tuple):
                status = read_res[0]
                if len(read_res) > 1:
                    raw_data = read_res[1]
            else:
                status = read_res

            if status > 0:
                retry_count = 0
                yield raw_data
            elif status == -1:
                continue
            elif status == 0:
                break
            elif status == -3:
                retry_count += 1
                if retry_count % 10 == 0:
                    jv_info(f"  Waiting for download... (-3, retry {retry_count})")
                if retry_count > 600:
                    jv_err("Timeout waiting for download (-3).")
                    break
                time.sleep(0.5)
                continue
            elif status == -402:
                jv_verbose("JVRead: no data (-402).")
                break
            else:
                raise Exception(f"JVRead Error: Code {status}")

    def get_data(self, dataspec, start_date, option=2, end_date=None):
        """
        JVOpen -> JVRead ループによるデータ取得ジェネレータ
        dataspec: "RACE", "DIFN" など
        start_date: "YYYYMMDD000000"
        option: 1(Setup), 2(Update), 4(One-time/Full)
        end_date: "YYYYMMDD235959" (Optional) - 終了日時を指定してデータ取得を制限

        ストリーム終了後に JVClose する（連続 get_data 用）。
        """
        # JVOpen
        # 引数: (DataSpec, FromTime, Option, ReadCount, DownloadCount, LastTime)
        # LastTime: 終了日時を指定することで、取得データの範囲を制限できる可能性がある
        # PythonのCOMでは、参照渡しの引数は戻り値のタプルとして返ってくることが多いが
        # JVLinkの仕様上、戻り値はリターンコードのみの場合が多い（環境による）。
        opened_ok = False
        try:
            try:
                ret_code, res = self.open_accumulation(
                    dataspec=dataspec,
                    start_time=start_date,
                    option=option,
                    end_time=end_date,
                )

                # 0 以上が成功（tuple 先頭が 0 以外の正の値でも成功とみなす）
                if ret_code < 0:
                    if ret_code == -111:
                        jv_warn(
                            f"JVOpen {dataspec}: -111 (access/maintenance) - skip"
                        )
                        return  # Graceful skip
                    elif ret_code == -202:
                        jv_warn(
                            f"JVOpen {dataspec}: -202 (no data from {start_date}) - skip"
                        )
                        return  # Graceful skip (データが存在しない場合)
                    elif ret_code == -303:
                        jv_warn(
                            f"JVOpen {dataspec}: -303 (open failed; 速報/契約/日付範囲など) - skip"
                        )
                        return  # Graceful skip（続きの 0B14 等へ進める）
                    jv_err(f"JVOpen {dataspec} failed: code={ret_code} raw={res!r}")
                    raise RuntimeError(f"JVOpen failed with code {ret_code}")

            except RuntimeError:
                raise
            except Exception as e:
                jv_err(f"JVOpen {dataspec} unexpected: {e}")
                raise

            opened_ok = True
            yield from self.read()
        finally:
            if opened_ok and self.jv_link:
                try:
                    self.jv_link.JVClose()
                    jv_verbose(f"JVClose after stream ({dataspec}).")
                except Exception:
                    pass

    def get_realtime_data(self, dataspec, key):
        """
        JVRTOpen -> JVRead ループによる速報データ取得ジェネレータ。
        dataspec: '0B11', '0B12', '0B15', ...
        key: 通常 YYYYMMDD または race_id
        """
        try:
            ret_code, res = self.open_realtime(dataspec, key)
            if ret_code < 0:
                jv_warn(
                    f"JVRTOpen({dataspec}, {key}) skipped: ret={ret_code} raw={res!r}"
                )
                return
        except Exception as e:
            jv_err(f"JVRTOpen {dataspec}: {e}")
            raise

        yield from self.read(buff_size=256000)
