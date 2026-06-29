from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pythoncom
import win32com.client


def _ret_code(result) -> int:
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


def _decode_chunk(raw_data) -> str:
    if isinstance(raw_data, bytes):
        return raw_data.decode("cp932", "replace")
    if isinstance(raw_data, str):
        return raw_data
    return ""


def _announce_int(value: str) -> int:
    s = str(value or "").strip()
    if not s.isdigit():
        return -1
    return int(s)


def parse_we_ascii_line(line: str, seq: int) -> dict | None:
    if not line or len(line) < 39 or not line.startswith("WE"):
        return None
    row = {
        "record_id": line[0:2],
        "data_kubun": line[2:3],
        "year": line[11:15].strip(),
        "month_day": line[15:19].strip(),
        "course_code": line[19:21].strip(),
        "kai": line[21:23].strip(),
        "nichi": line[23:25].strip(),
        "race_num": line[25:27].strip(),
        "announce_ddhhmmss": line[27:35].strip(),
        "change_id": line[35:36].strip(),
        "weather_code": line[36:37].strip(),
        "turf_condition": line[37:38].strip(),
        "dirt_condition": line[38:39].strip(),
        "record_separator": line[39:40].strip() if len(line) >= 40 else "",
        "raw_hex": line.encode("ascii", "replace").hex(),
        "source_line": line,
        "seq": str(seq),
    }
    return row


def _iter_we_lines_from_chunk(raw_chunk: str):
    for line in raw_chunk.split("\n"):
        line = line.strip("\r\n")
        if not line:
            continue
        if line.startswith("WE"):
            yield line


def fetch_we_events_0b14(date_yyyymmdd: str) -> list[dict]:
    pythoncom.CoInitialize()
    jv = None
    events: list[dict] = []
    seq = 0
    try:
        jv = win32com.client.Dispatch("JVDTLab.JVLink")
        init_ret = _ret_code(jv.JVInit("UNKNOWN"))
        if init_ret < 0:
            raise RuntimeError(f"JVInit failed: {init_ret}")

        open_ret_raw = jv.JVRTOpen("0B14", date_yyyymmdd)
        open_ret = _ret_code(open_ret_raw)
        if open_ret < 0:
            raise RuntimeError(f"JVRTOpen(0B14,{date_yyyymmdd}) failed: {open_ret}")

        while True:
            rr = jv.JVRead("", 1000000, "")
            if isinstance(rr, tuple):
                status = int(rr[0])
                raw_data = rr[1] if len(rr) > 1 else ""
            else:
                status = int(rr)
                raw_data = ""

            if status > 0:
                chunk = _decode_chunk(raw_data)
                for line in _iter_we_lines_from_chunk(chunk):
                    seq += 1
                    parsed = parse_we_ascii_line(line, seq)
                    if parsed:
                        events.append(parsed)
                continue
            if status == -1:
                continue
            if status == -3:
                time.sleep(0.2)
                continue
            if status in (0, -402):
                break
            raise RuntimeError(f"JVRead error: {status}")
    finally:
        if jv is not None:
            try:
                jv.JVClose()
            except Exception:
                pass
        pythoncom.CoUninitialize()
    return events


def load_we_events_from_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    out: list[dict] = []
    seq = 0
    for r in rows:
        seq += 1
        if str(r.get("record_id", "")).strip() != "WE":
            continue
        if r.get("source_line"):
            line = str(r.get("source_line", "")).strip()
        else:
            raw_hex = str(r.get("raw_hex", "")).strip()
            if raw_hex:
                try:
                    line = bytes.fromhex(raw_hex).decode("ascii", errors="ignore")
                except Exception:
                    line = ""
            else:
                line = ""
        parsed = parse_we_ascii_line(line, seq)
        if parsed:
            out.append(parsed)
            continue
        # fallback: at least keep existing columns if fixed-width decode failed
        out.append(
            {
                "record_id": "WE",
                "data_kubun": str(r.get("data_kubun", "")).strip(),
                "year": str(r.get("year", "")).strip(),
                "month_day": str(r.get("month_day", "")).strip(),
                "course_code": str(r.get("course_code", "")).strip(),
                "kai": str(r.get("kai", "")).strip(),
                "nichi": str(r.get("nichi", "")).strip(),
                "race_num": str(r.get("race_num", "")).strip(),
                "announce_ddhhmmss": str(r.get("announce_ddhhmmss", "")).strip(),
                "change_id": str(r.get("change_id", "")).strip(),
                "weather_code": str(r.get("weather_code", "")).strip(),
                "turf_condition": str(r.get("turf_condition", "")).strip(),
                "dirt_condition": str(r.get("dirt_condition", "")).strip(),
                "record_separator": str(r.get("record_separator", "")).strip(),
                "raw_hex": str(r.get("raw_hex", "")).strip(),
                "source_line": "",
                "seq": str(seq),
            }
        )
    return out


def build_course_snapshot(events: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for e in events:
        k = (e["year"], e["month_day"], e["course_code"], e["kai"], e["nichi"])
        grouped.setdefault(k, []).append(e)

    snapshot: list[dict] = []
    for k, arr in sorted(grouped.items()):
        arr = sorted(arr, key=lambda x: (_announce_int(x["announce_ddhhmmss"]), int(x["seq"])))
        state = {"weather_code": "", "turf_condition": "", "dirt_condition": ""}
        last = None
        for e in arr:
            cid = str(e.get("change_id", "")).strip()
            w = str(e.get("weather_code", "")).strip()
            t = str(e.get("turf_condition", "")).strip()
            d = str(e.get("dirt_condition", "")).strip()

            if cid == "2":
                # 天候変更: 馬場は据え置く
                if w != "":
                    state["weather_code"] = w
            elif cid in {"1", "3"}:
                if w != "":
                    state["weather_code"] = w
                if t != "":
                    state["turf_condition"] = t
                if d != "":
                    state["dirt_condition"] = d
            else:
                if w != "":
                    state["weather_code"] = w
                if t != "":
                    state["turf_condition"] = t
                if d != "":
                    state["dirt_condition"] = d
            last = e

        if last is None:
            continue
        snapshot.append(
            {
                "year": k[0],
                "month_day": k[1],
                "course_code": k[2],
                "kai": k[3],
                "nichi": k[4],
                "last_announce_ddhhmmss": last["announce_ddhhmmss"],
                "last_change_id": last["change_id"],
                "event_count": str(len(arr)),
                "weather_code": state["weather_code"],
                "turf_condition": state["turf_condition"],
                "dirt_condition": state["dirt_condition"],
            }
        )
    return snapshot


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="0B14/WE realtime fetcher v2")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="YYYYMMDD")
    ap.add_argument(
        "--output-dir",
        default="common/data/output/realtime_we_v2",
        help="output directory (project-root relative)",
    )
    ap.add_argument(
        "--input-we-csv",
        default="",
        help="use existing we.csv instead of JV fetch",
    )
    args = ap.parse_args()

    date = str(args.date).strip()
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"--date must be YYYYMMDD: {date!r}")

    project_root = Path(__file__).resolve().parents[3]
    out_dir = project_root / args.output_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_we_csv:
        events = load_we_events_from_csv(Path(args.input_we_csv))
    else:
        events = fetch_we_events_0b14(date)

    snapshot = build_course_snapshot(events)

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
    snap_fields = [
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

    events_path = out_dir / f"we_events_{date}_{ts}.csv"
    latest_events_path = out_dir / "we_events_latest.csv"
    snap_path = out_dir / f"we_course_snapshot_{date}_{ts}.csv"
    latest_snap_path = out_dir / "we_course_snapshot_latest.csv"

    _write_csv(events_path, events, events_fields)
    _write_csv(latest_events_path, events, events_fields)
    _write_csv(snap_path, snapshot, snap_fields)
    _write_csv(latest_snap_path, snapshot, snap_fields)

    summary = {
        "date": date,
        "events": len(events),
        "courses": len(snapshot),
        "events_path": str(events_path),
        "snapshot_path": str(snap_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
