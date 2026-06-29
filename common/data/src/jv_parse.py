import os

try:
    from .jv_schemas import SCHEMAS
except ImportError:
    from jv_schemas import SCHEMAS


def parse_fixed_width(byte_data, schema):
    """
    固定長データをパースする関数 (Shift-JISバイト列対応版)
    byte_data: Shift-JISエンコードされたbytes列
    schema: { 'field_name': (start_pos, length), ... }  ※start_posは1始まり
    """
    parsed = {}
    for field, (start, length) in schema.items():
        # 仕様書の「1バイト目」はインデックス0
        s_idx = start - 1
        e_idx = s_idx + length

        chunk = byte_data[s_idx:e_idx]
        try:
            # Shift_JIS (cp932) でデコード
            # errors='replace' で外字や不正バイトを '?' などに置換してクラッシュを防ぐ
            val = chunk.decode("cp932", errors="replace").strip()
            parsed[field] = val
        except Exception as e:
            parsed[field] = None
    return parsed

def get_schema_fieldnames(schema_name):
    """
    スキーマのフィールド名を順序付きで取得する関数
    DM_SCHEMAとTM_SCHEMAのループで追加されたフィールドの順序を保証する

    Args:
        schema_name: スキーマ名（例: "DM", "TM"）

    Returns:
        順序付きフィールド名のリスト
    """
    if schema_name not in SCHEMAS:
        return list(SCHEMAS.get(schema_name, {}).keys())

    schema = SCHEMAS[schema_name]

    # DM_SCHEMAとTM_SCHEMAの場合は、順序を明示的に制御
    if schema_name in ["DM", "TM"]:
        # 固定フィールド（ループ前のフィールド）
        fixed_fields = []
        # ループで追加されるフィールド
        loop_fields = []
        # その他のフィールド（record_separatorなど）
        other_fields = []

        for key in schema.keys():
            if key.startswith("mining_pred_"):
                loop_fields.append(key)
            elif key == "record_separator":
                other_fields.append(key)
            else:
                fixed_fields.append(key)

        # ループフィールドをソート
        # 順序: mining_pred_1_horse_num, mining_pred_1_time, mining_pred_1_error+, mining_pred_1_error-,
        #       mining_pred_2_horse_num, mining_pred_2_time, ...
        def sort_key(x):
            parts = x.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                # 数値部分（1, 2, 3, ...）
                num = int(parts[2])
                # フィールド名部分（horse_num, time, error+, error-, score）
                field_name = "_".join(parts[3:]) if len(parts) > 3 else ""
                # フィールド名の優先順位
                field_order = {
                    "horse_num": 0,
                    "time": 1,
                    "error+": 2,
                    "error-": 3,
                    "score": 1,  # TM_SCHEMA用
                }.get(field_name, 999)
                return (num, field_order, field_name)
            return (999, 999, x)

        loop_fields.sort(key=sort_key)

        # 順序: 固定フィールド → ループフィールド → その他（record_separator）
        return fixed_fields + loop_fields + other_fields
    else:
        # その他のスキーマは通常通り
        return list(schema.keys())

def _extract_record_key(record, rec_id):
    """
    レコードから重複チェック用のキーを抽出する関数

    Args:
        record: パースされたレコード（辞書）
        rec_id: レコードID（"RA", "SE", "HR", "HN", "SK", "BT", "HC", "WC", "DM", "TM"）

    Returns:
        重複チェック用のキー（タプルまたは文字列、Noneの場合はキーなし）
    """
    if rec_id in {"RA", "SE", "HR", "WH", "AV", "TC", "CC", "O2", "O3"}:
        if rec_id in {"SE", "WH"}:
            return (
                record.get("year", ""),
                record.get("month_day", ""),
                record.get("course_code", ""),
                record.get("kai", ""),
                record.get("nichi", ""),
                record.get("race_num", ""),
                record.get("horse_num", ""),
            )
        else:
            return (
                record.get("year", ""),
                record.get("month_day", ""),
                record.get("course_code", ""),
                record.get("kai", ""),
                record.get("nichi", ""),
                record.get("race_num", ""),
            )
    elif rec_id == "JC":
        return (
            record.get("year", ""),
            record.get("month_day", ""),
            record.get("course_code", ""),
            record.get("kai", ""),
            record.get("nichi", ""),
            record.get("race_num", ""),
            record.get("horse_num", ""),
        )
    elif rec_id == "WE":
        return (
            record.get("year", ""),
            record.get("month_day", ""),
            record.get("course_code", ""),
            record.get("kai", ""),
            record.get("nichi", ""),
            record.get("race_num", ""),
        )
    elif rec_id == "HN" and "ketto_num" in record:
        return record.get("ketto_num", "")
    elif rec_id == "SK" and "ketto_num" in record:
        return record.get("ketto_num", "")
    elif rec_id == "BT" and "breeding_reg_num" in record:
        return record.get("breeding_reg_num", "")
    elif rec_id in {"HC", "WC"} and "ketto_num" in record and "training_date" in record:
        return (
            record.get("ketto_num", ""),
            record.get("training_date", ""),
        )
    elif rec_id in {"DM", "TM"}:
        return (
            record.get("year", ""),
            record.get("month_day", ""),
            record.get("course_code", ""),
            record.get("kai", ""),
            record.get("nichi", ""),
            record.get("race_num", ""),
        )
    return None

def _jockey_code_from_jc_raw_hex(raw_hex: str) -> str:
    """
    JC（騎手変更）レコードの raw_hex から変更後騎手コード（5桁）を取り出す。
    バイト位置は JV-Data 仕様に依存するため、環境でずれる場合は要調整。
    """
    if not raw_hex or not str(raw_hex).strip():
        return ""
    try:
        b = bytes.fromhex(str(raw_hex).strip())
        if len(b) >= 37:
            chunk = b[32:37]
            s = chunk.decode("ascii", errors="ignore").strip()
            digits = "".join(c for c in s if c.isdigit())
            if len(digits) >= 5:
                return digits[-5:].zfill(5)
            if digits:
                return digits.zfill(5)[-5:]
    except Exception:
        pass
    return ""
