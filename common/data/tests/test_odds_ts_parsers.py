"""
0B41(O1時系列単複枠) / 0B42(O2時系列馬連) パーサーのユニットテスト。

JV-Link に接続できない開発環境でも検証できるよう、docs/JV-Data.md の
バイト位置仕様（「7. オッズ1（単複枠）（O1）」「8. オッズ2（馬連）（O2）」）
に従って組み立てたモックの固定長ASCII行を使う。実データでの検証は
別途 JV-Link 接続可能な環境で行うこと。
"""

from __future__ import annotations

from common.data.src.jv_schemas import O1_SCHEMA, O2_SCHEMA
from common.data.src.legacy_get_data_impl import (
    _parse_o1_win_place_odds_ts,
    _parse_o2_quinella_odds,
)


def _build_line(schema: dict, fields: dict, total_len: int) -> str:
    """schema の (1始まり start, length) に従って ASCII 行を組み立てる。"""
    buf = [" "] * total_len
    for name, value in fields.items():
        start, length = schema[name]
        s_idx = start - 1
        value = str(value)
        assert len(value) <= length, f"{name}: {value!r} too long for width {length}"
        value = value.ljust(length)
        buf[s_idx : s_idx + length] = list(value)
    return "".join(buf)


def _o1_line(*, announce_datetime: str, horses: list[dict]) -> str:
    fields = {
        "record_id": "O1",
        "data_kubun": "1",  # 中間
        "date_make": "20260712",
        "year": "2026",
        "month_day": "0712",
        "course_code": "05",
        "kai": "01",
        "nichi": "01",
        "race_num": "11",
        "announce_datetime": announce_datetime,
        "registered_count": str(len(horses)).zfill(2),
        "running_count": str(len(horses)).zfill(2),
        "sale_flag_win": "7",
        "sale_flag_place": "7",
        "sale_flag_bracket_quinella": "7",
        "place_payout_key": "3",
    }
    for i, h in enumerate(horses, start=1):
        fields[f"win_odds_{i:02d}_horse"] = h["horse_num"]
        fields[f"win_odds_{i:02d}_odds"] = h["win_odds_raw"]
        fields[f"win_odds_{i:02d}_pop"] = h["win_pop_raw"]
        fields[f"place_odds_{i:02d}_horse"] = h["horse_num"]
        fields[f"place_odds_{i:02d}_min"] = h["place_min_raw"]
        fields[f"place_odds_{i:02d}_max"] = h["place_max_raw"]
        fields[f"place_odds_{i:02d}_pop"] = h["place_pop_raw"]
    return _build_line(O1_SCHEMA, fields, total_len=700)


def test_parse_o1_win_place_odds_ts_basic():
    line = _o1_line(
        announce_datetime="07121030",
        horses=[
            {
                "horse_num": "01",
                "win_odds_raw": "0035",  # 3.5倍
                "win_pop_raw": "03",
                "place_min_raw": "0080",  # 8.0倍
                "place_max_raw": "0120",  # 12.0倍
                "place_pop_raw": "02",
            },
            {
                "horse_num": "02",
                "win_odds_raw": "0120",  # 12.0倍
                "win_pop_raw": "05",
                "place_min_raw": "0150",
                "place_max_raw": "0200",
                "place_pop_raw": "04",
            },
        ],
    )

    rows = _parse_o1_win_place_odds_ts(line, race_id="2026071205010111")

    assert len(rows) == 2
    row1 = next(r for r in rows if r["horse_num"] == "01")
    assert row1["race_id"] == "2026071205010111"
    assert row1["announce_datetime"] == "07121030"
    assert row1["win_odds"] == 3.5
    assert row1["win_odds_status"] == "ok"
    assert row1["place_odds_min"] == 8.0
    assert row1["place_odds_max"] == 12.0
    assert row1["year"] == "2026"
    assert row1["month_day"] == "0712"
    assert row1["course_code"] == "05"

    row2 = next(r for r in rows if r["horse_num"] == "02")
    assert row2["win_odds"] == 12.0
    assert row2["place_odds_min"] == 15.0
    assert row2["place_odds_max"] == 20.0


def test_parse_o1_win_place_odds_ts_special_values():
    """"----"(発売前取消) と "0000"(無投票) が None + status に変換されることを確認する。"""
    line = _o1_line(
        announce_datetime="07121200",
        horses=[
            {
                "horse_num": "07",
                "win_odds_raw": "----",
                "win_pop_raw": "--",
                "place_min_raw": "0000",
                "place_max_raw": "0000",
                "place_pop_raw": "  ",
            },
        ],
    )

    rows = _parse_o1_win_place_odds_ts(line, race_id="2026071205010111")
    assert len(rows) == 1
    row = rows[0]
    assert row["win_odds"] is None
    assert row["win_odds_status"] == "cancel_before_sale"
    assert row["place_odds_min"] is None
    assert row["place_odds_status"] == "no_vote"


def test_parse_o1_win_place_odds_ts_rejects_other_record_ids():
    assert _parse_o1_win_place_odds_ts("O2" + " " * 100, race_id="x") == []
    assert _parse_o1_win_place_odds_ts("", race_id="x") == []


def _o2_line(*, announce_datetime: str, pairs: list[dict]) -> str:
    fields = {
        "record_id": "O2",
        "data_kubun": "1",
        "date_make": "20260712",
        "year": "2026",
        "month_day": "0712",
        "course_code": "05",
        "kai": "01",
        "nichi": "01",
        "race_num": "11",
        "announce_datetime": announce_datetime,
        "registered_count": "16",
        "running_count": "16",
        "sale_flag_quinella": "7",
    }
    for i, p in enumerate(pairs, start=1):
        fields[f"quinella_odds_{i:03d}_kumi"] = p["kumi"]
        fields[f"quinella_odds_{i:03d}_odds"] = p["odds_raw"]
        fields[f"quinella_odds_{i:03d}_pop"] = p["pop_raw"]
    return _build_line(O2_SCHEMA, fields, total_len=2100)


def test_parse_o2_quinella_odds_ts_basic():
    line = _o2_line(
        announce_datetime="07121030",
        pairs=[
            {"kumi": "0102", "odds_raw": "001250", "pop_raw": "001"},  # 125.0倍
            {"kumi": "0103", "odds_raw": "000305", "pop_raw": "010"},  # 30.5倍
        ],
    )

    rows = _parse_o2_quinella_odds(line, race_id="2026071205010111")
    assert len(rows) == 2
    r1 = next(r for r in rows if r["ticket"] == "1-2")
    assert r1["horse_num_1"] == "01"
    assert r1["horse_num_2"] == "02"
    assert r1["odds"] == 125.0
    assert r1["announce_datetime"] == "07121030"
    assert r1["race_id"] == "2026071205010111"
