import sys
import types
import unittest


sys.modules.setdefault("pythoncom", types.ModuleType("pythoncom"))
sys.modules.setdefault("win32com", types.ModuleType("win32com"))
sys.modules.setdefault("win32com.client", types.ModuleType("win32com.client"))


class RealtimeJVParsingTests(unittest.TestCase):
    def test_dm_and_tm_schemas_parse_repeating_prediction_fields(self):
        from common.data.src.jv_parse import parse_fixed_width
        from common.data.src.jv_schemas import SCHEMAS

        dm_raw = (
            "DM22026061220260613020101121730011459200970022021467700990022"
            "031459300970022041459900060006051450200950042061458700970022"
            "071461100630007081459600970043091458400970022101465800980043"
            "111462800980043121457800060006"
        ).ljust(301)
        tm_raw = (
            "TM22026061220260613020101121730010599020501030363040273050649"
            "060390070612080578090491100533110517120488"
        ).ljust(139)

        dm = parse_fixed_width(dm_raw.encode("ascii"), SCHEMAS["DM"])
        tm = parse_fixed_width(tm_raw.encode("ascii"), SCHEMAS["TM"])

        self.assertEqual(dm["mining_pred_1_horse_num"], "01")
        self.assertEqual(dm["mining_pred_1_time"], "14592")
        self.assertEqual(dm["mining_pred_1_error+"], "0097")
        self.assertEqual(tm["mining_pred_1_horse_num"], "01")
        self.assertEqual(tm["mining_pred_1_score"], "0599")

    def test_wh_realtime_record_expands_to_horse_rows(self):
        from common.data.src.legacy_get_data_impl import _expand_wh_realtime_row

        raw = (
            b"H" * 35
            + b"01"
            + b"A" * 36
            + b"430-008"
            + b"02"
            + b"B" * 36
            + b"468+004"
        )
        row = {
            "record_id": "WH",
            "data_kubun": "1",
            "year": "2026",
            "month_day": "0613",
            "course_code": "05",
            "kai": "03",
            "nichi": "03",
            "race_num": "01",
            "horse_num": "00",
            "raw_hex": raw.hex(),
        }

        expanded = _expand_wh_realtime_row(row)

        self.assertEqual(expanded[0]["horse_num"], "01")
        self.assertEqual(expanded[0]["horse_weight"], "430")
        self.assertEqual(expanded[0]["weight_change_sign"], "-")
        self.assertEqual(expanded[0]["weight_change"], "008")
        self.assertEqual(expanded[1]["horse_num"], "02")
        self.assertEqual(expanded[1]["horse_weight"], "468")
        self.assertEqual(expanded[1]["weight_change_sign"], "+")
        self.assertEqual(expanded[1]["weight_change"], "004")

    def test_tm_realtime_record_expands_to_horse_scores(self):
        from common.data.src.jv_parse import parse_fixed_width
        from common.data.src.jv_schemas import SCHEMAS
        from common.data.src.legacy_get_data_impl import _expand_tm_realtime_row

        tm_raw = (
            "TM22026061220260613020101121730010599020501030363040273050649"
            "060390070612080578090491100533110517120488"
        ).ljust(139)
        parsed = parse_fixed_width(tm_raw.encode("ascii"), SCHEMAS["TM"])
        parsed["raw_hex"] = tm_raw.encode("ascii").hex()

        scores = _expand_tm_realtime_row(parsed)

        self.assertEqual(scores[0]["horse_num"], "01")
        self.assertEqual(scores[0]["tm_score"], "0599")
        self.assertEqual(scores[1]["horse_num"], "02")
        self.assertEqual(scores[1]["tm_score"], "0501")

    def test_backfill_mining_predicted_rank_from_time(self):
        from common.data.src.legacy_get_data_impl import (
            _backfill_mining_predicted_rank_from_time,
        )

        se_rows = [
            {
                "year": "2026",
                "month_day": "0614",
                "course_code": "02",
                "kai": "01",
                "nichi": "02",
                "race_num": "01",
                "horse_num": "01",
                "mining_predicted_time": "10950",
                "mining_predicted_rank": "00",
            },
            {
                "year": "2026",
                "month_day": "0614",
                "course_code": "02",
                "kai": "01",
                "nichi": "02",
                "race_num": "01",
                "horse_num": "02",
                "mining_predicted_time": "10978",
                "mining_predicted_rank": "00",
            },
            {
                "year": "2026",
                "month_day": "0614",
                "course_code": "02",
                "kai": "01",
                "nichi": "02",
                "race_num": "01",
                "horse_num": "03",
                "mining_predicted_time": "10905",
                "mining_predicted_rank": "00",
            },
        ]

        updated = _backfill_mining_predicted_rank_from_time(se_rows)

        self.assertEqual(updated, 3)
        self.assertEqual(se_rows[2]["mining_predicted_rank"], "01")
        self.assertEqual(se_rows[0]["mining_predicted_rank"], "02")
        self.assertEqual(se_rows[1]["mining_predicted_rank"], "03")


if __name__ == "__main__":
    unittest.main()
