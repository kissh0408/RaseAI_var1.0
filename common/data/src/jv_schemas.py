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


RA_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "year": (12, 4),  # 開催年
    "month_day": (16, 4),  # 開催月日
    "course_code": (20, 2),  # 競馬場コード
    "kai": (22, 2),  # 開催回
    "nichi": (24, 2),  # 開催日目
    "race_num": (26, 2),  # レース番号
    "grade_code": (615, 1),  # グレードコード
    "race_type_code": (617, 2),  # 競走種別コード
    "weight_type": (622, 1),  # 重量種別コード
    "condition_2yo": (623, 3),  # 競走条件コード 2歳条件
    "condition_3yo": (626, 3),  # 競走条件コード 3歳条件
    "condition_4yo": (629, 3),  # 競走条件コード 4歳条件
    "condition_5yo_plus": (632, 3),  # 競走条件コード 5歳以上条件
    "condition_min_age": (635, 3),  # 競走条件コード 最若年条件
    "distance": (698, 4),  # 距離
    "track_code": (706, 2),  # トラックコード
    "course_kubun": (710, 2),  # コース区分
    "registered_count": (882, 2),  # 登録頭数
    "running_count": (884, 2),  # 出走頭数
    "finish_count": (886, 2),  # 入線頭数
    "weather_code": (888, 1),  # 天候コード
    "turf_condition": (889, 1),  # 芝馬場状態コード
    "dirt_condition": (890, 1),  # ダート馬場状態コード
    "lap_times": (891, 75),  # ラップタイム
    "obstacle_mile_time": (966, 4),  # 障害マイルタイム
    "time_3f_before": (970, 3),  # 前３ハロンタイム
    "time_4f_before": (973, 3),  # 前４ハロンタイム
    "time_3f_after": (976, 3),  # 後３ハロンタイム
    "time_4f_after": (979, 3),  # 後４ハロンタイム
    # 第1コーナー
    "corner_1_id": (982, 1),  # コーナー区分
    "corner_1_lap": (983, 1),  # 周回数
    "corner_1_rank": (984, 70),  # 通過順位
    # 第2コーナー
    "corner_2_id": (1054, 1),
    "corner_2_lap": (1055, 1),
    "corner_2_rank": (1056, 70),
    # 第3コーナー
    "corner_3_id": (1126, 1),
    "corner_3_lap": (1127, 1),
    "corner_3_rank": (1128, 70),
    # 第4コーナー
    "corner_4_id": (1198, 1),
    "corner_4_lap": (1199, 1),
    "corner_4_rank": (1200, 70),
    "update_kubun": (1270, 1),  # レコード更新区分
    "record_separator": (1271, 2),  # レコード区切
}

SE_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "year": (12, 4),  # 開催年
    "month_day": (16, 4),  # 開催月日
    "course_code": (20, 2),  # 競馬場コード
    "kai": (22, 2),  # 開催回
    "nichi": (24, 2),  # 開催日目
    "race_num": (26, 2),  # レース番号
    "wakuban": (28, 1),  # 枠番
    "horse_num": (29, 2),  # 馬番
    "ketto_num": (31, 10),  # 血統登録番号
    "horse_mark_code": (77, 2),  # 馬記号コード
    "sex_code": (79, 1),  # 性別コード
    "breed_code": (80, 1),  # 品種コード
    "age": (83, 2),  # 馬齢
    "region_code": (85, 1),  # 東西所属コード
    "trainer_code": (86, 5),  # 調教師コード
    "owner_code": (99, 6),  # 馬主コード
    "burden_weight": (289, 3),  # 負担重量
    "burden_weight_prev": (292, 3),  # 変更前負担重量
    "blinker_code": (295, 1),  # ブリンカー使用区分
    "jockey_code": (297, 5),  # 騎手コード
    "horse_weight": (325, 3),  # 馬体重
    "weight_change_sign": (328, 1),  # 増減符号
    "weight_change": (329, 3),  # 増減差
    "abnormal_code": (332, 1),  # 異常区分コード
    "finish_rank": (333, 2),  # 入線順位
    "final_rank": (335, 2),  # 確定着順
    "dead_heat_flag": (337, 1),  # 同着区分
    "dead_heat_count": (338, 1),  # 同着頭数
    "time": (339, 4),  # 走破タイム
    "margin_code": (343, 3),  # 着差コード
    "corner_1": (352, 2),  # 1コーナーでの順位
    "corner_2": (354, 2),  # 2コーナーでの順位
    "corner_3": (356, 2),  # 3コーナーでの順位
    "corner_4": (358, 2),  # 4コーナーでの順位
    "odds": (360, 4),  # 単勝オッズ
    "popularity": (364, 2),  # 単勝人気順
    "hon_shokin": (366, 8),  # 獲得本賞金
    "fuka_shokin": (374, 8),  # 獲得付加賞金
    "time_4f_after": (388, 3),  # 後4ハロンタイム
    "time_3f_after": (391, 3),  # 後3ハロンタイム
    "time_diff": (532, 4),  # タイム差
    "mining_kubun": (537, 1),  # マイニング区分
    "mining_predicted_time": (538, 5),  # マイニング予想走破タイム
    "mining_error_plus": (543, 4),  # マイニング予想誤差(信頼度)＋
    "mining_error_minus": (547, 4),  # マイニング予想誤差(信頼度)－
    "mining_predicted_rank": (551, 2),  # マイニング予想順位
    "running_style_code": (553, 1),  # 今回レース脚質判定
    "record_separator": (554, 2),  # レコード区切
}

HR_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "date_make": (4, 8),  # データ作成年月日
    "year": (12, 4),  # 開催年
    "month_day": (16, 4),  # 開催月日
    "course_code": (20, 2),  # 競馬場コード
    "kai": (22, 2),  # 開催回
    "nichi": (24, 2),  # 開催日目
    "race_num": (26, 2),  # レース番号
    # --- 単勝 (繰返3回 / 13byte) ---
    "win_1_horse": (103, 2),
    "win_1_money": (105, 9),
    "win_1_pop": (114, 2),
    "win_2_horse": (116, 2),
    "win_2_money": (118, 9),
    "win_2_pop": (127, 2),
    "win_3_horse": (129, 2),
    "win_3_money": (131, 9),
    "win_3_pop": (140, 2),
    # --- 複勝 (繰返5回 / 13byte) ---
    "place_1_horse": (142, 2),
    "place_1_money": (144, 9),
    "place_1_pop": (153, 2),
    "place_2_horse": (155, 2),
    "place_2_money": (157, 9),
    "place_2_pop": (166, 2),
    "place_3_horse": (168, 2),
    "place_3_money": (170, 9),
    "place_3_pop": (179, 2),
    "place_4_horse": (181, 2),
    "place_4_money": (183, 9),
    "place_4_pop": (192, 2),
    "place_5_horse": (194, 2),
    "place_5_money": (196, 9),
    "place_5_pop": (205, 2),
    # --- 枠連 (繰返3回 / 13byte) ---
    "bracket_q_1_kumi": (207, 2),
    "bracket_q_1_money": (209, 9),
    "bracket_q_1_pop": (218, 2),
    "bracket_q_2_kumi": (220, 2),
    "bracket_q_2_money": (222, 9),
    "bracket_q_2_pop": (231, 2),
    "bracket_q_3_kumi": (233, 2),
    "bracket_q_3_money": (235, 9),
    "bracket_q_3_pop": (244, 2),
    # --- 馬連 (繰返3回 / 16byte) ---
    "quinella_1_kumi": (246, 4),
    "quinella_1_money": (250, 9),
    "quinella_1_pop": (259, 3),
    "quinella_2_kumi": (262, 4),
    "quinella_2_money": (266, 9),
    "quinella_2_pop": (275, 3),
    "quinella_3_kumi": (278, 4),
    "quinella_3_money": (282, 9),
    "quinella_3_pop": (291, 3),
    # --- ワイド (繰返7回 / 16byte) ---
    "wide_1_kumi": (294, 4),
    "wide_1_money": (298, 9),
    "wide_1_pop": (307, 3),
    "wide_2_kumi": (310, 4),
    "wide_2_money": (314, 9),
    "wide_2_pop": (323, 3),
    "wide_3_kumi": (326, 4),
    "wide_3_money": (330, 9),
    "wide_3_pop": (339, 3),
    "wide_4_kumi": (342, 4),
    "wide_4_money": (346, 9),
    "wide_4_pop": (355, 3),
    "wide_5_kumi": (358, 4),
    "wide_5_money": (362, 9),
    "wide_5_pop": (371, 3),
    "wide_6_kumi": (374, 4),
    "wide_6_money": (378, 9),
    "wide_6_pop": (387, 3),
    "wide_7_kumi": (390, 4),
    "wide_7_money": (394, 9),
    "wide_7_pop": (403, 3),
    # --- 馬単 (繰返6回 / 16byte) ---
    "exacta_1_kumi": (454, 4),
    "exacta_1_money": (458, 9),
    "exacta_1_pop": (467, 3),
    "exacta_2_kumi": (470, 4),
    "exacta_2_money": (474, 9),
    "exacta_2_pop": (483, 3),
    "exacta_3_kumi": (486, 4),
    "exacta_3_money": (490, 9),
    "exacta_3_pop": (499, 3),
    "exacta_4_kumi": (502, 4),
    "exacta_4_money": (506, 9),
    "exacta_4_pop": (515, 3),
    "exacta_5_kumi": (518, 4),
    "exacta_5_money": (522, 9),
    "exacta_5_pop": (531, 3),
    "exacta_6_kumi": (534, 4),
    "exacta_6_money": (538, 9),
    "exacta_6_pop": (547, 3),
    # --- 3連複 (繰返3回 / 18byte) ---
    "trio_1_kumi": (550, 6),
    "trio_1_money": (556, 9),
    "trio_1_pop": (565, 3),
    "trio_2_kumi": (568, 6),
    "trio_2_money": (574, 9),
    "trio_2_pop": (583, 3),
    "trio_3_kumi": (586, 6),
    "trio_3_money": (592, 9),
    "trio_3_pop": (601, 3),
    # --- 3連単 (繰返6回 / 19byte) ---
    "trifecta_1_kumi": (604, 6),
    "trifecta_1_money": (610, 9),
    "trifecta_1_pop": (619, 4),
    "trifecta_2_kumi": (623, 6),
    "trifecta_2_money": (629, 9),
    "trifecta_2_pop": (638, 4),
    "trifecta_3_kumi": (642, 6),
    "trifecta_3_money": (648, 9),
    "trifecta_3_pop": (657, 4),
    "trifecta_4_kumi": (661, 6),
    "trifecta_4_money": (667, 9),
    "trifecta_4_pop": (676, 4),
    "trifecta_5_kumi": (680, 6),
    "trifecta_5_money": (686, 9),
    "trifecta_5_pop": (695, 4),
    "trifecta_6_kumi": (699, 6),
    "trifecta_6_money": (705, 9),
    "trifecta_6_pop": (714, 4),
    # --- フッター ---
    "record_separator": (718, 2),  # レコード区切
}

HN_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "breeding_reg_num": (12, 10),  # 繁殖登録番号
    "ketto_num": (30, 10),  # 血統登録番号
    "birth_year": (197, 4),  # 生年
    "sex_code": (201, 1),  # 性別コード
    "breed_code": (202, 1),  # 品種コード
    "breeding_import_flag": (205, 1),  # 繁殖馬持込区分
    "sire_reg_num": (230, 10),  # 父馬繁殖登録番号
    "dam_reg_num": (240, 10),  # 母馬繁殖登録番号
    "record_separator": (250, 2),  # レコード区切
}

SK_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "ketto_num": (12, 10),  # 血統登録番号
    "sex_code": (30, 1),  # 性別コード
    "breed_code": (31, 1),  # 品種コード
    "offspring_import_flag": (34, 1),  # 産駒持込区分
    # 3代血統 (各10バイト)
    "p_sire": (67, 10),  # 1. 父
    "p_dam": (77, 10),  # 2. 母
    "p_sire_sire": (87, 10),  # 3. 父父
    "p_sire_dam": (97, 10),  # 4. 父母
    "p_dam_sire": (107, 10),  # 5. 母父
    "p_dam_dam": (117, 10),  # 6. 母母
    "p_sire_sire_sire": (127, 10),  # 7. 父父父
    "p_sire_sire_dam": (137, 10),  # 8. 父父母
    "p_sire_dam_sire": (147, 10),  # 9. 父母父
    "p_sire_dam_dam": (157, 10),  # 10. 父母母
    "p_dam_sire_sire": (167, 10),  # 11. 母父父
    "p_dam_sire_dam": (177, 10),  # 12. 母父母
    "p_dam_dam_sire": (187, 10),  # 13. 母母父
    "p_dam_dam_dam": (197, 10),  # 14. 母母母
    "record_separator": (207, 2),  # レコード区切
}

BT_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "breeding_reg_num": (12, 10),  # 繁殖登録番号
    "system_id": (22, 30),  # 系統ID
    "record_separator": (6888, 2),  # レコード区切
}

DM_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "year": (12, 4),  # 開催年
    "month_day": (16, 4),  # 開催月日
    "course_code": (20, 2),  # 競馬場コード
    "kai": (22, 2),  # 開催回
    "nichi": (24, 2),  # 開催日目
    "race_num": (26, 2),  # レース番号
}
for _i in range(1, 19):
    _base_pos = 32 + (_i - 1) * 15
    DM_SCHEMA[f"mining_pred_{_i}_horse_num"] = (_base_pos + 0, 2)
    DM_SCHEMA[f"mining_pred_{_i}_time"] = (_base_pos + 2, 5)
    DM_SCHEMA[f"mining_pred_{_i}_error+"] = (_base_pos + 7, 4)
    DM_SCHEMA[f"mining_pred_{_i}_error-"] = (_base_pos + 11, 4)
DM_SCHEMA["record_separator"] = (302, 2)

TM_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "year": (12, 4),  # 開催年
    "month_day": (16, 4),  # 開催月日
    "course_code": (20, 2),  # 競馬場コード
    "kai": (22, 2),  # 開催回
    "nichi": (24, 2),  # 開催日目
    "race_num": (26, 2),  # レース番号
}
for _i in range(1, 19):
    _base_pos = 32 + (_i - 1) * 6
    TM_SCHEMA[f"mining_pred_{_i}_horse_num"] = (_base_pos + 0, 2)
    TM_SCHEMA[f"mining_pred_{_i}_score"] = (_base_pos + 2, 4)
TM_SCHEMA["record_separator"] = (140, 2)

HC_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "training_center": (12, 1),  # トレセン区分
    "training_date": (13, 8),  # 調教年月日
    "training_time": (21, 4),  # 調教時刻
    "ketto_num": (25, 10),  # 血統登録番号
    "time_4f_total": (35, 4),  # 4ハロンタイム合計
    "lap_time_800_600": (39, 3),  # ラップタイム(800-600)
    "time_3f_total": (42, 4),  # 3ハロンタイム合計
    "lap_time_600_400": (46, 3),  # ラップタイム(600-400)
    "time_2f_total": (49, 4),  # 2ハロンタイム合計
    "lap_time_400_200": (53, 3),  # ラップタイム(400-200)
    "lap_time_200_0": (56, 3),  # ラップタイム(200-0)
    "record_separator": (59, 2),  # レコード区切
}

WC_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "training_center": (12, 1),  # トレセン区分
    "training_date": (13, 8),  # 調教年月日
    "training_time": (21, 4),  # 調教時刻
    "ketto_num": (25, 10),  # 血統登録番号
    "course": (35, 1),  # コース
    "track_direction": (36, 1),  # 馬場周り
    "time_10f_total": (38, 4),  # 10ハロンタイム合計
    "lap_time_10f_9f": (42, 3),  # ラップタイム(10F-9F)
    "time_9f_total": (45, 4),  # 9ハロンタイム合計
    "lap_time_9f_8f": (49, 3),  # ラップタイム(9F-8F)
    "time_8f_total": (52, 4),  # 8ハロンタイム合計
    "lap_time_8f_7f": (56, 3),  # ラップタイム(8F-7F)
    "time_7f_total": (59, 4),  # 7ハロンタイム合計
    "lap_time_7f_6f": (63, 3),  # ラップタイム(7F-6F)
    "time_6f_total": (66, 4),  # 6ハロンタイム合計
    "lap_time_6f_5f": (70, 3),  # ラップタイム(6F-5F)
    "time_5f_total": (73, 4),  # 5ハロンタイム合計
    "lap_time_5f_4f": (77, 3),  # ラップタイム(5F-4F)
    "time_4f_total": (80, 4),  # 4ハロンタイム合計
    "lap_time_4f_3f": (84, 3),  # ラップタイム(4F-3F)
    "time_3f_total": (87, 4),  # 3ハロンタイム合計
    "lap_time_3f_2f": (91, 3),  # ラップタイム(3F-2F)
    "time_2f_total": (94, 4),  # 2ハロンタイム合計
    "lap_time_2f_1f": (98, 3),  # ラップタイム(2F-1F)
    "lap_time_1f_0f": (101, 3),  # ラップタイム(1F-0F)
    "record_separator": (104, 2),  # レコード区切
}

WE_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    # 天候/馬場は速報系レコードの代表項目名として扱う
    "weather_code": (28, 1),
    "turf_condition": (29, 1),
    "dirt_condition": (30, 1),
    "record_separator": (31, 2),
}

# WH（馬体重）: docs/JV-Data.md 101.馬体重（WH）と照合して修正（2026-07-09）。
# 旧定義は単一馬前提のフラットな座標（horse_num@29等）で、実レコードの繰返し
# グループ構造（開始位置36、18頭×45バイト）と一致していなかった。
# 実運用の馬体重抽出は raw_hex を再パースする _expand_wh_realtime_row() 側で
# 別途正しい定数（ヘッダ35バイト・ブロック45バイト）を使っており実害はなかったが、
# スキーマ定義自体は仕様書に合わせて修正する。
WH_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
}
for _i in range(1, 19):
    _base_pos = 36 + (_i - 1) * 45
    WH_SCHEMA[f"wh_{_i}_horse_num"] = (_base_pos + 0, 2)
    WH_SCHEMA[f"wh_{_i}_horse_name"] = (_base_pos + 2, 36)
    WH_SCHEMA[f"wh_{_i}_horse_weight"] = (_base_pos + 38, 3)
    WH_SCHEMA[f"wh_{_i}_weight_change_sign"] = (_base_pos + 41, 1)
    WH_SCHEMA[f"wh_{_i}_weight_change"] = (_base_pos + 42, 3)
WH_SCHEMA["record_separator"] = (846, 2)

# AV（出走取消・競走除外）: docs/JV-Data.md 103.出走取消・競走除外（AV）と照合。
# 旧 detail_code(28,2) は実在フィールドではなく発表月日時分の途中を指していた。
AV_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "horse_num": (36, 2),
    "horse_name": (38, 36),
    "reason_code": (74, 3),
    "record_separator": (77, 2),
}

# JC（騎手変更）: docs/JV-Data.md 104.騎手変更（JC）と照合。
JC_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "horse_num": (36, 2),
    "horse_name": (38, 36),
    "after_burden_weight": (74, 3),
    "after_jockey_code": (77, 5),
    "after_jockey_name": (82, 34),
    "after_jockey_apprentice_code": (116, 1),
    "before_burden_weight": (117, 3),
    "before_jockey_code": (120, 5),
    "before_jockey_name": (125, 34),
    "before_jockey_apprentice_code": (159, 1),
    "record_separator": (160, 2),
}

# TC（発走時刻変更）: docs/JV-Data.md 105.発走時刻変更（TC）と照合。
TC_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "after_start_time": (36, 4),
    "before_start_time": (40, 4),
    "record_separator": (44, 2),
}

# CC（コース変更）: docs/JV-Data.md 106.コース変更（CC）と照合。
CC_SCHEMA = {
    "record_id": (1, 2),
    "data_kubun": (3, 1),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "after_distance": (36, 4),
    "after_track_code": (40, 2),
    "before_distance": (42, 4),
    "before_track_code": (46, 2),
    "reason_code": (48, 1),
    "record_separator": (49, 2),
}

O2_SCHEMA = {
    "record_id": (1, 2),  # O2: 馬連オッズ
    "data_kubun": (3, 1),
    "date_make": (4, 8),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "registered_count": (36, 2),
    "running_count": (38, 2),
    "sale_flag_quinella": (40, 1),
    "quinella_vote_count": (2030, 11),
    "record_separator": (2041, 2),
}

for _i in range(1, 154):
    _start = 41 + (_i - 1) * 13
    O2_SCHEMA[f"quinella_odds_{_i:03d}_kumi"] = (_start, 4)
    O2_SCHEMA[f"quinella_odds_{_i:03d}_odds"] = (_start + 4, 6)
    O2_SCHEMA[f"quinella_odds_{_i:03d}_pop"] = (_start + 10, 3)

O3_SCHEMA = {
    "record_id": (1, 2),  # O3: ワイドオッズ
    "data_kubun": (3, 1),
    "date_make": (4, 8),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),
    "registered_count": (36, 2),
    "running_count": (38, 2),
    "sale_flag_wide": (40, 1),
    "wide_vote_count": (2642, 11),
    "record_separator": (2653, 2),
}

for _i in range(1, 154):
    _start = 41 + (_i - 1) * 17
    O3_SCHEMA[f"wide_odds_{_i:03d}_kumi"] = (_start, 4)
    O3_SCHEMA[f"wide_odds_{_i:03d}_min"] = (_start + 4, 5)
    O3_SCHEMA[f"wide_odds_{_i:03d}_max"] = (_start + 9, 5)
    O3_SCHEMA[f"wide_odds_{_i:03d}_pop"] = (_start + 14, 3)

# O1（単複枠オッズ）: docs/JV-Data.md「7. オッズ1（単複枠）（O1）」と照合。
# 0B31（速報オッズ・単複枠）と 0B41（時系列オッズ・単複枠）はどちらもこの
# O1 レコードフォーマットを使う。両者の違いはサーバ側の配信挙動
# （0B31=最新スナップショットのみ、0B41=複数時点を時系列で配信）であり、
# レイアウトは同一。announce_datetime（発表月日時分）は時系列オッズ使用時のみ
# キーとして必須になる（中間オッズのみ設定）。
O1_SCHEMA = {
    "record_id": (1, 2),  # O1: オッズ1（単複枠）
    "data_kubun": (3, 1),
    "date_make": (4, 8),
    "year": (12, 4),
    "month_day": (16, 4),
    "course_code": (20, 2),
    "kai": (22, 2),
    "nichi": (24, 2),
    "race_num": (26, 2),
    "announce_datetime": (28, 8),  # 月日時分各2桁。時系列オッズ使用時のみキー
    "registered_count": (36, 2),
    "running_count": (38, 2),
    "sale_flag_win": (40, 1),
    "sale_flag_place": (41, 1),
    "sale_flag_bracket_quinella": (42, 1),
    "place_payout_key": (43, 1),
    "win_vote_count": (928, 11),
    "place_vote_count": (939, 11),
    "bracket_quinella_vote_count": (950, 11),
    "record_separator": (961, 2),
}

for _i in range(1, 29):
    _start = 44 + (_i - 1) * 8
    O1_SCHEMA[f"win_odds_{_i:02d}_horse"] = (_start, 2)
    O1_SCHEMA[f"win_odds_{_i:02d}_odds"] = (_start + 2, 4)
    O1_SCHEMA[f"win_odds_{_i:02d}_pop"] = (_start + 6, 2)

for _i in range(1, 29):
    _start = 268 + (_i - 1) * 12
    O1_SCHEMA[f"place_odds_{_i:02d}_horse"] = (_start, 2)
    O1_SCHEMA[f"place_odds_{_i:02d}_min"] = (_start + 2, 4)
    O1_SCHEMA[f"place_odds_{_i:02d}_max"] = (_start + 6, 4)
    O1_SCHEMA[f"place_odds_{_i:02d}_pop"] = (_start + 10, 2)

SCHEMAS = {
    "RA": RA_SCHEMA,
    "SE": SE_SCHEMA,
    "HR": HR_SCHEMA,
    "HN": HN_SCHEMA,
    "SK": SK_SCHEMA,
    "BT": BT_SCHEMA,
    "DM": DM_SCHEMA,
    "TM": TM_SCHEMA,
    "HC": HC_SCHEMA,
    "WC": WC_SCHEMA,
    "WE": WE_SCHEMA,
    "WH": WH_SCHEMA,
    "AV": AV_SCHEMA,
    "JC": JC_SCHEMA,
    "TC": TC_SCHEMA,
    "CC": CC_SCHEMA,
    "O1": O1_SCHEMA,
    "O2": O2_SCHEMA,
    "O3": O3_SCHEMA,
}

_RACE_KUBUN_PRIORITY = {"7": 3, "2": 2, "1": 1}

_FETCH_JRA_RACE_KUBUNS = frozenset({"2", "7"})
