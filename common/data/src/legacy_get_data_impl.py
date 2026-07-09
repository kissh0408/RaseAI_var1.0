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
    from tqdm.auto import tqdm
except ImportError:
    # tqdm 未導入環境向けの最小互換。
    class _DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def tqdm(iterable=None, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)


# ==========================================
# 1. スキーマ定義 (バイト位置ベース)
# ==========================================

# レース詳細 (RA)
# レース詳細 (RA)
# 仕様書（レコード長1272バイト）
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

# 馬ごとの詳細 (SE)
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

# 繁殖馬マスタ (HN)
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

# 産駒マスタ (SK)
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

# 系統情報 (BT)
BT_SCHEMA = {
    "record_id": (1, 2),  # レコード種別ID
    "data_kubun": (3, 1),  # データ区分
    "breeding_reg_num": (12, 10),  # 繁殖登録番号
    "system_id": (22, 30),  # 系統ID
    "record_separator": (6888, 2),  # レコード区切
}

# タイム型データマイニング予想（DM）
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
# マイニング予想の18回繰り返しフィールドを動的に追加
for i in range(1, 19):
    base_pos = 32 + (i - 1) * 15
    DM_SCHEMA[f"mining_pred_{i}_horse_num"] = (base_pos + 0, 2)  # 馬番
    DM_SCHEMA[f"mining_pred_{i}_time"] = (base_pos + 2, 5)  # 予想走破タイム
    DM_SCHEMA[f"mining_pred_{i}_error+"] = (base_pos + 7, 4)  # 予想誤差(信頼度)＋
    DM_SCHEMA[f"mining_pred_{i}_error-"] = (base_pos + 11, 4)  # 予想誤差(信頼度)＋
DM_SCHEMA["record_separator"] = (302, 2)  # レコード区切

# 対戦型データマイニング予想（TM）
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
# マイニング予想の18回繰り返しフィールドを動的に追加
# 各繰り返し: 馬番(2) + 予測スコア(4) = 6バイト
for i in range(1, 19):
    base_pos = 32 + (i - 1) * 6
    TM_SCHEMA[f"mining_pred_{i}_horse_num"] = (base_pos + 0, 2)  # 馬番
    TM_SCHEMA[f"mining_pred_{i}_score"] = (base_pos + 2, 4)  # 予測スコア
TM_SCHEMA["record_separator"] = (140, 2)  # レコード区切

# 坂路調教 (HC)
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


# ウッドチップ調教 (WC)
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

# 速報天候・馬場 (WE)
# NOTE:
# WE/WH の詳細位置は運用環境の仕様書に合わせて調整してください。
# ここでは共通キー + 主要項目を最小構成で定義しています。
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

# 速報馬体重 (WH): docs/JV-Data.md 101.馬体重（WH）と照合して修正（2026-07-09）。
# 旧定義は単一馬前提のフラットな座標で、実レコードの繰返しグループ構造
# （開始位置36、18頭×45バイト）と一致していなかった（実運用は raw_hex を
# 再パースする _expand_wh_realtime_row() 側の別定数を使っており実害はなかった）。
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

# 出走取消・競走除外 (AV) / 騎手変更 (JC) / 発走時刻変更 (TC) / コース変更 (CC)
# docs/JV-Data.md 103〜106 と照合して修正（2026-07-09）。旧 detail_code は
# 実在フィールドではなく発表月日時分の途中バイトを指していた。
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
    "record_id": (1, 2),
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
    "record_id": (1, 2),
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
    "O2": O2_SCHEMA,
    "O3": O3_SCHEMA,
}

# ==========================================
# 2. ヘルパー関数 (パース・保存)
# ==========================================


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


def _load_existing_dates_without_pandas(filepath: str, rec_id: str) -> set:
    """
    JV 用 32bit 環境など pandas 未導入時の既存キー読み込み（標準 csv のみ）。
    キーの形は pandas 経路と揃える（HN/SK/BT はスカラー、その他はタプル）。
    """
    keys = set()
    col_specs: dict[str, list[str] | None] = {
        "RA": ["year", "month_day", "course_code", "kai", "nichi", "race_num"],
        "HR": ["year", "month_day", "course_code", "kai", "nichi", "race_num"],
        "SE": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
            "horse_num",
        ],
        "HN": ["ketto_num"],
        "SK": ["ketto_num"],
        "BT": ["breeding_reg_num"],
        "HC": ["ketto_num", "training_date"],
        "WC": ["ketto_num", "training_date"],
        "DM": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "TM": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "WE": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "AV": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "TC": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "CC": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "WH": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
        ],
        "JC": [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
            "horse_num",
        ],
    }
    usecols = col_specs.get(rec_id)
    if not usecols:
        return keys
    try:
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return keys
            fn = {h.strip(): h for h in reader.fieldnames if h}
            for row in reader:
                if rec_id in ("HN", "SK", "BT"):
                    col = usecols[0]
                    h = fn.get(col) or fn.get(col.strip())
                    if not h:
                        continue
                    v = str(row.get(h, "") or "").strip()
                    if v:
                        keys.add(v)
                else:
                    vals = []
                    ok = True
                    for col in usecols:
                        h = fn.get(col) or fn.get(col.strip())
                        if not h:
                            ok = False
                            break
                        vals.append(str(row.get(h, "") or "").strip())
                    if ok and any(vals):
                        keys.add(tuple(vals))
    except Exception as e:
        print(
            f"  Warning: Failed to load existing dates (no-pandas) from {filepath}: {e}"
        )
    return keys


def load_existing_dates(filepath, rec_id):
    """
    既存のCSVファイルから日付キーのセットを読み込む（高速化版）
    レースデータ（RA, SE, HR）: year + month_day + course_code + kai + nichi + race_num
    マスターデータ: IDベース（実装済み）
    """
    existing_keys = set()

    if not os.path.exists(filepath) or os.path.getsize(filepath) <= 1024:
        return existing_keys

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return _load_existing_dates_without_pandas(filepath, rec_id)

    try:

        # 必要な列だけを読み込む（メモリ使用量削減）
        if rec_id in ["RA", "SE", "HR"]:
            if rec_id == "SE":
                usecols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                    "horse_num",
                ]
            else:
                usecols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                ]
        elif rec_id == "HN":
            usecols = ["ketto_num"]
        elif rec_id == "SK":
            usecols = ["ketto_num"]
        elif rec_id == "BT":
            usecols = ["breeding_reg_num"]
        elif rec_id in ["HC", "WC"]:
            usecols = ["ketto_num", "training_date"]
        elif rec_id in ["DM", "TM"]:
            usecols = [
                "year",
                "month_day",
                "course_code",
                "kai",
                "nichi",
                "race_num",
            ]
        elif rec_id in ["WE", "AV", "TC", "CC", "WH"]:
            usecols = [
                "year",
                "month_day",
                "course_code",
                "kai",
                "nichi",
                "race_num",
            ]
        elif rec_id == "JC":
            usecols = [
                "year",
                "month_day",
                "course_code",
                "kai",
                "nichi",
                "race_num",
                "horse_num",
            ]
        else:
            usecols = None

        # 必要な列が存在するか確認してから読み込む
        # engine='c'でCエンジンを使用（デフォルトだが明示的に指定）
        if usecols:
            try:
                df = pd.read_csv(
                    filepath,
                    encoding="utf-8-sig",
                    dtype=str,
                    usecols=usecols,
                    engine="c",
                    low_memory=False,  # メモリに余裕がある場合、型推論を避けて高速化
                )
            except (ValueError, KeyError):
                # 列が存在しない場合は全列読み込み
                df = pd.read_csv(
                    filepath,
                    encoding="utf-8-sig",
                    dtype=str,
                    engine="c",
                    low_memory=False,
                )
        else:
            df = pd.read_csv(
                filepath, encoding="utf-8-sig", dtype=str, engine="c", low_memory=False
            )

        # レースデータの場合（RA, SE, HR）- ベクトル化で高速化
        if rec_id in ["RA", "SE", "HR"]:
            if rec_id == "SE":
                required_cols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                    "horse_num",
                ]
                if all(col in df.columns for col in required_cols):
                    # itertuples()を使って高速化（iterrowsより速い）
                    # fillna("")を先に適用してからitertuples()を使用
                    df_filled = df[required_cols].fillna("")
                    existing_keys = set(
                        tuple(row)
                        for row in df_filled.itertuples(index=False, name=None)
                    )
            else:
                # RA, HRはレース単位のデータ
                required_cols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                ]
                if all(col in df.columns for col in required_cols):
                    # itertuples()を使って高速化（iterrowsより速い）
                    # fillna("")を先に適用してからitertuples()を使用
                    df_filled = df[required_cols].fillna("")
                    existing_keys = set(
                        tuple(row)
                        for row in df_filled.itertuples(index=False, name=None)
                    )

        elif rec_id in ["WE", "AV", "TC", "CC", "WH"]:
            required_cols = [
                "year",
                "month_day",
                "course_code",
                "kai",
                "nichi",
                "race_num",
            ]
            if all(col in df.columns for col in required_cols):
                df_filled = df[required_cols].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )

        elif rec_id == "JC":
            required_cols = [
                "year",
                "month_day",
                "course_code",
                "kai",
                "nichi",
                "race_num",
                "horse_num",
            ]
            if all(col in df.columns for col in required_cols):
                df_filled = df[required_cols].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )

        # マスターデータの場合（HN, SK, BT等）
        else:
            # IDベースでチェック（既存のロジックに合わせる）
            if rec_id == "HN" and "ketto_num" in df.columns:
                # dropna()とunique()を組み合わせるよりも、notna()でフィルタしてからunique()の方が速い
                existing_keys = set(
                    df.loc[df["ketto_num"].notna(), "ketto_num"].unique()
                )
            elif rec_id == "SK" and "ketto_num" in df.columns:
                existing_keys = set(
                    df.loc[df["ketto_num"].notna(), "ketto_num"].unique()
                )
            elif rec_id == "BT" and "breeding_reg_num" in df.columns:
                existing_keys = set(
                    df.loc[df["breeding_reg_num"].notna(), "breeding_reg_num"].unique()
                )
            elif (
                rec_id == "HC"
                and "ketto_num" in df.columns
                and "training_date" in df.columns
            ):
                # itertuples()を使って高速化（iterrowsより速い）
                # fillna("")を先に適用してからitertuples()を使用
                df_filled = df[["ketto_num", "training_date"]].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )
            elif (
                rec_id == "WC"
                and "ketto_num" in df.columns
                and "training_date" in df.columns
            ):
                # itertuples()を使って高速化（iterrowsより速い）
                # fillna("")を先に適用してからitertuples()を使用
                df_filled = df[["ketto_num", "training_date"]].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )
            elif rec_id == "DM" and "year" in df.columns and "month_day" in df.columns:
                # itertuples()を使って高速化（iterrowsより速い）
                required_cols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                ]
                # fillna("")を先に適用してからitertuples()を使用
                df_filled = df[required_cols].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )
            elif rec_id == "TM" and "year" in df.columns and "month_day" in df.columns:
                # itertuples()を使って高速化（iterrowsより速い）
                required_cols = [
                    "year",
                    "month_day",
                    "course_code",
                    "kai",
                    "nichi",
                    "race_num",
                ]
                # fillna("")を先に適用してからitertuples()を使用
                df_filled = df[required_cols].fillna("")
                existing_keys = set(
                    tuple(row) for row in df_filled.itertuples(index=False, name=None)
                )

    except Exception as e:
        print(f"  Warning: Failed to load existing dates from {filepath}: {e}")

    return existing_keys


def _extract_record_key(record, rec_id):
    """
    レコードから重複チェック用のキーを抽出する関数

    Args:
        record: パースされたレコード（辞書）
        rec_id: レコードID（"RA", "SE", "HR", "HN", "SK", "BT", "HC", "WC", "DM", "TM"）

    Returns:
        重複チェック用のキー（タプルまたは文字列、Noneの場合はキーなし）
    """
    if rec_id in {"RA", "SE", "HR", "WH", "AV", "TC", "CC"}:
        # WH は SE と異なり「1レコード=1レース分18頭まとめて」の構造
        # （docs/JV-Data.md 101.馬体重）のため、horse_num は含めずレース単位でキー化する。
        if rec_id == "SE":
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


def save_to_csv(data_list, filepath, fieldnames, *, append: bool = True):
    """データをCSVに保存する。append=False で上書き（RACE の RA/SE マージ書き出し用）。"""
    if not data_list:
        return

    # filepath = os.path.join("data", filename) -> Removed to fix double pathing
    # os.makedirs("data", exist_ok=True) -> Caller handles dir creation now

    # Check dir exists
    dirname = os.path.dirname(filepath)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    file_exists = bool(append) and os.path.isfile(filepath)

    # DM/TMデータの場合は古い順にソート
    # ファイルパスから判定（ming_dm または ming_tm が含まれている場合）
    if "ming_dm" in filepath.lower() or "ming_tm" in filepath.lower():
        # ソートキー: year, month_day, course_code, kai, nichi, race_num
        sort_keys = ["year", "month_day", "course_code", "kai", "nichi", "race_num"]

        def sort_key_func(record):
            """ソート用のキー関数"""
            key_values = []
            for key in sort_keys:
                val = record.get(key, "")
                # 数値として比較できる場合は数値に変換、できない場合は文字列として比較
                try:
                    # year, month_day, course_code, kai, nichi, race_num は数値として比較
                    if key in [
                        "year",
                        "month_day",
                        "course_code",
                        "kai",
                        "nichi",
                        "race_num",
                    ]:
                        key_values.append(int(val) if val and val.strip() else 0)
                    else:
                        key_values.append(val if val else "")
                except (ValueError, TypeError):
                    key_values.append(val if val else "")
            return tuple(key_values)

        # ソート実行
        data_list = sorted(data_list, key=sort_key_func)

    # UTF-8 (BOM付き) または cp932 で保存するかは用途次第。
    # Excelで開くなら cp932 または utf-8-sig が無難。ここでは汎用性の高い utf-8-sig を使用。
    mode = "a" if append else "w"
    if not append:
        file_exists = False
    with open(filepath, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_list)
    print(f"  Saved {len(data_list)} records to {filepath}")


# fetch_jra_data の RACE: 区分2（未確定寄り）と7の両方を取り、7を優先して1行にまとめる
_RACE_KUBUN_PRIORITY = {"7": 3, "2": 2, "1": 1}
_FETCH_JRA_RACE_KUBUNS = frozenset({"2", "7"})


def _load_race_year_merge_map(path: str, rec_id: str) -> dict:
    """年次 race_ra / race_se CSV を読み、キーごとに最優先の data_kubun 行だけ残すマップ。"""
    merge: dict = {}
    if not os.path.isfile(path) or os.path.getsize(path) <= 1024:
        return merge
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dk = str(row.get("data_kubun", "")).strip()
                pri = _RACE_KUBUN_PRIORITY.get(dk, 0)
                if pri == 0:
                    continue
                parsed = dict(row)
                key = _extract_record_key(parsed, rec_id)
                if key is None:
                    continue
                old = merge.get(key)
                if old is None or pri > old[0]:
                    merge[key] = (pri, parsed)
    except Exception as e:
        print(f"  Warning: could not load RACE merge map {path}: {e}")
    return merge


def _race_stream_try_merge(
    merge: dict, parsed: dict, rec_id: str, allowed: frozenset
) -> bool:
    """True = 新規または区分アップグレードで取り込み。False = スキップ。"""
    dk = str(parsed.get("data_kubun", "")).strip()
    if dk not in allowed:
        return False
    pri = _RACE_KUBUN_PRIORITY.get(dk, 0)
    if pri == 0:
        return False
    key = _extract_record_key(parsed, rec_id)
    if key is None:
        return False
    old = merge.get(key)
    if old is None:
        merge[key] = (pri, parsed)
        return True
    if pri > old[0]:
        merge[key] = (pri, parsed)
        return True
    return False


def _se_record_in_time_window(parsed: dict, start14: str, end14: str) -> bool:
    """開催日 year+month_day が start14..end14 の YYYYMMDD 範囲に入るか。"""
    try:
        y = str(parsed.get("year", "")).strip().zfill(4)
        md = str(parsed.get("month_day", "")).strip().zfill(4)
        if len(y) != 4 or len(md) != 4:
            return False
        key = y + md
        return str(start14)[:8] <= key <= str(end14)[:8]
    except (TypeError, ValueError):
        return False


# ==========================================
# 3. JRA-VAN クライアントクラス
# ==========================================


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
            print("JV-Link instance created via Dispatch.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to create JV-Link instance.")
            print(f"Details: {e}")
            print(
                "Hint: If you are on 64-bit Python, ensure DllSurrogate registry setup is complete."
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
            print("JVInit (Login) Successful.")
        except Exception as e:
            print(f"Login failed: {e}")
            raise

    def close(self):
        """
        JVClose による切断
        """
        if self.jv_link:
            try:
                self.jv_link.JVClose()
                print("JV-Link session closed.")
            except Exception as e:
                print(f"JVClose error (ignored): {e}")

    def get_data(self, dataspec, start_date, option=2, end_date=None):
        """
        JVOpen -> JVRead ループによるデータ取得ジェネレータ
        dataspec: "RACE", "DIFN" など
        start_date: "YYYYMMDD000000"
        option: 1(Setup), 2(Update), 4(One-time/Full)
        end_date: "YYYYMMDD235959" (Optional) - 終了日時を指定してデータ取得を制限
        """
        # JVOpen
        # 引数: (DataSpec, FromTime, Option, ReadCount, DownloadCount, LastTime)
        # LastTime: 終了日時を指定することで、取得データの範囲を制限できる可能性がある
        # PythonのCOMでは、参照渡しの引数は戻り値のタプルとして返ってくることが多いが
        # JVLinkの仕様上、戻り値はリターンコードのみの場合が多い（環境による）。
        last_time = end_date if end_date else ""
        try:
            if end_date:
                print(
                    f"Requesting {dataspec} (From: {start_date}, To: {end_date}, Opt: {option})..."
                )
            else:
                print(f"Requesting {dataspec} (From: {start_date}, Opt: {option})...")
            res = self.jv_link.JVOpen(dataspec, start_date, option, 0, 0, last_time)
            ret_code = _jv_com_return_code(res)

            # 0 以上が成功（tuple 先頭が 0 以外の正の値でも成功とみなす）
            if ret_code < 0:
                if ret_code == -111:
                    print(
                        f"JVOpen Note: Access denied or maintenance for {dataspec} (-111). Skipping."
                    )
                    return  # Graceful skip
                elif ret_code == -202:
                    print(
                        f"JVOpen Note: No data available for {dataspec} from {start_date} (Code -202). Skipping."
                    )
                    return  # Graceful skip (データが存在しない場合)
                elif ret_code == -303:
                    print(
                        f"JVOpen Note: {dataspec} が開けません (-303, raw={res!r})。"
                        " 0V/速報系で過去日・未保持・契約外のときに出ることがあります。スキップします。"
                    )
                    return  # Graceful skip（続きの 0B14 等へ進める）
                print(f"JVOpen Failed: Code {ret_code} (raw={res!r})")
                raise Exception(f"JVOpen failed with code {ret_code}")

        except Exception as e:
            print(f"JVOpen Error: {e}")
            raise

        # JVRead Loop
        # Report recommends larger buffer for Setup (Option 3/4) to avoid overflow
        buff_size = 10000000  # 10MB (Previously 200KB)
        buff = ""  # バッファ初期化
        fname = ""

        while True:
            try:
                # JVRead(Buff, BuffSize, FileName)
                # 戻り値: (RetCode, BuffString, FileName) のようなタプルになるのが一般的
                read_res = self.jv_link.JVRead(buff, buff_size, fname)
            except Exception as e:
                print(f"JVRead Exception: {e}")
                break

            # 戻り値の解析
            status = 0
            raw_data = ""

            if isinstance(read_res, tuple):
                # (Status, Data, FileName) or (Status, Data) depending on signature
                status = read_res[0]
                if len(read_res) > 1:
                    raw_data = read_res[1]
            else:
                status = read_res

            if status > 0:
                retry_count = 0  # Reset retry count on success
                # データあり (statusバイト数読み込み)
                yield raw_data
            elif status == -1:
                # ファイルの切れ目 (引き続き読み込み可)
                continue
            elif status == 0:
                # 全データ読み込み完了 (End of Download)
                break
            elif status == -3:
                # ダウンロード中 (Download in progress)
                retry_count = locals().get("retry_count", 0) + 1
                if retry_count % 10 == 0:
                    print(f"Waiting for download... (Status -3, Retry {retry_count})")

                if retry_count > 600:  # 5 minutes max
                    print("Error: Timeout waiting for download.")
                    break

                time.sleep(0.5)
                continue
            elif status == -402:
                # ファイルなし（更新データなしの場合など）
                print("JVRead Note: No data found (-402).")
                break
            else:
                raise Exception(f"JVRead Error: Code {status}")


# ==========================================
# 4. メイン処理
# ==========================================


def fetch_race_only(start_date_year=2017, end_date_year=2025):
    """
    RACEデータ（RA, SE, HR）のみを年別ファイルで取得する関数

    Args:
        start_date_year: 開始年（デフォルト: 2017）
        end_date_year: 終了年（デフォルト: 2025）
    """
    print("=== RACEデータのみ取得（年別ファイル） ===")

    # 出力ディレクトリの準備
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output"
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 年別取得の範囲
    RACE_YEARS = list(range(start_date_year, end_date_year + 1))

    # RACE: レース情報
    RACE_TASK = {
        "dataspec": "RACE",
        "option": 4,
        "target_ids": ["RA", "SE", "HR"],
        "years": RACE_YEARS,
    }

    try:
        # RACEタスクを処理
        for year in RACE_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"

            # RACEタスクを処理
            task = RACE_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            task_start_date = year_start

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # 既存データの日付キーを読み込む（年別ファイル）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "RACE":
                    subdir = f"race_{target_id.lower()}"
                    fname = f"race_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(
                        f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                    )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Year Filtering（指定年のデータのみ）- 早期フィルタリング
                            # 年フィルタリングを最初に行い、不要なデータの処理を避ける
                            if "year" in parsed:
                                try:
                                    rec_year = int(parsed["year"])
                                    if rec_year != year:
                                        skipped_count += 1
                                        continue
                                except:
                                    continue

                            # Data Kubun Filtering（年フィルタリングの後）
                            if (
                                rec_id in {"RA", "SE"}
                                and parsed.get("data_kubun") != "7"
                            ):
                                skipped_count += 1
                                continue

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = _extract_record_key(parsed, rec_id)
                            should_skip = (
                                record_key is not None and record_key in existing_keys
                            )

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            # 年別ファイルに保存（サブディレクトリ内）
                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "RACE":
                                    subdir = f"race_{rec_id.lower()}"
                                    fname = f"race_{rec_id.lower()}_{year}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存（年別ファイル、サブディレクトリ内）
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "RACE":
                        subdir = f"race_{rid.lower()}"
                        fname = f"race_{rid.lower()}_{year}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}_{year}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_hc_only(start_date_year=2015, end_date_year=2025):
    """
    HCデータ（坂路調教）のみを年別ファイルで取得する関数

    Args:
        start_date_year: 開始年（デフォルト: 2015、2016年はデータなしのため2015または2017年以降を推奨）
        end_date_year: 終了年（デフォルト: 2025）
    """
    print("=== HCデータ（坂路調教）のみ取得（年別ファイル） ===")

    # 出力ディレクトリの準備
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output"
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 年別取得の範囲
    HC_YEARS = list(range(start_date_year, end_date_year + 1))

    # SLOP (HC): 坂路調教
    HC_TASK = {
        "dataspec": "SLOP",
        "option": 3,
        "target_ids": ["HC"],
        "years": HC_YEARS,
    }

    # 増分取得用の状態ファイル（HC専用。jv_last_update.json とは別ファイル）。
    # 最新年（HC_TASK["years"][-1]）のみ、前回成功時点からの増分取得を行う。
    # 過去年は従来通り年始からのフル取得を維持する（挙動を変えない）。
    hc_state_path = Path(output_dir) / "state" / "hc_last_update.json"
    hc_state = _load_state(hc_state_path)
    hc_latest_year = HC_TASK["years"][-1]

    try:
        # HCタスクを処理
        for year in HC_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"

            # HCタスクを処理
            task = HC_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            task_start_date = year_start
            if year == hc_latest_year:
                # 最新年のみ、前回成功時点（同日00:00:00から）以降を取得する
                task_start_date = _jv_resolve_start_datetime(
                    hc_state, default=year_start, also_use_last_update_date=False
                )
            year_fetch_error = False

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # 既存データの日付キーを読み込む（年別ファイル）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "SLOP":
                    subdir = f"slop_{target_id.lower()}"
                    fname = f"slop_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(
                        f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                    )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Training date filtering for HC（年フィルタリング）
                            if rec_id == "HC" and "training_date" in parsed:
                                try:
                                    training_date_str = parsed.get(
                                        "training_date", ""
                                    ).strip()
                                    if len(training_date_str) == 8:  # YYYYMMDD形式
                                        training_year = int(training_date_str[:4])
                                        if training_year != year:
                                            skipped_count += 1
                                            continue
                                except:
                                    continue

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = None
                            should_skip = False

                            # 調教データ（HC）の場合
                            if (
                                rec_id == "HC"
                                and "ketto_num" in parsed
                                and "training_date" in parsed
                            ):
                                record_key = (
                                    parsed.get("ketto_num", ""),
                                    parsed.get("training_date", ""),
                                )
                                should_skip = record_key in existing_keys

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            # 年別ファイルに保存（サブディレクトリ内）
                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "SLOP":
                                    subdir = f"slop_{rec_id.lower()}"
                                    fname = f"slop_{rec_id.lower()}_{year}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
                year_fetch_error = True
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存（年別ファイル、サブディレクトリ内）
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "SLOP":
                        subdir = f"slop_{rid.lower()}"
                        fname = f"slop_{rid.lower()}_{year}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}_{year}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

            # 増分取得の状態を更新（最新年かつエラーなしの場合のみ）。
            # 失敗時は state を更新しない（update_jra_data と同じ規約）。
            if year == hc_latest_year and not year_fetch_error:
                today_end = datetime.now().strftime("%Y%m%d") + "235959"
                actual_end = min(year_end, today_end)
                hc_state["last_success_end"] = actual_end
                hc_state["last_run_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
                _save_state(hc_state_path, hc_state)
                print(f"Saved state: {hc_state_path}")

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_wc_only(start_date_year=2021, end_date_year=2025):
    """
    WCデータ（ウッドチップ調教）のみを年別ファイルで取得する関数

    Args:
        start_date_year: 開始年（デフォルト: 2021、2021年は7月27日以降のデータのみ）
        end_date_year: 終了年（デフォルト: 2025）
    """
    print("=== WCデータ（ウッドチップ調教）のみ取得（年別ファイル） ===")

    # 出力ディレクトリの準備
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output"
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 年別取得の範囲
    WC_YEARS = list(range(start_date_year, end_date_year + 1))

    # WOOD (WC): ウッドチップ調教
    WC_TASK = {
        "dataspec": "WOOD",
        "option": 3,
        "target_ids": ["WC"],
        "years": WC_YEARS,
    }

    # 増分取得用の状態ファイル（WC専用。jv_last_update.json / hc_last_update.json とは別ファイル）。
    # 最新年（WC_TASK["years"][-1]）のみ、前回成功時点からの増分取得を行う。
    # 過去年は従来通り年始（または2021年は7/27）からのフル取得を維持する（挙動を変えない）。
    wc_state_path = Path(output_dir) / "state" / "wc_last_update.json"
    wc_state = _load_state(wc_state_path)
    wc_latest_year = WC_TASK["years"][-1]

    try:
        # WCタスクを処理
        for year in WC_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"

            # WCタスクを処理
            task = WC_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            # WCは2021年7月27日以降のデータのみ（JV-Link仕様書より）
            # 2021年の場合は7月27日以降から開始
            if year == 2021:
                task_start_date = "20210727000000"
            else:
                task_start_date = year_start
            if year == wc_latest_year:
                # 最新年のみ、前回成功時点（同日00:00:00から）以降を取得する
                task_start_date = _jv_resolve_start_datetime(
                    wc_state, default=task_start_date, also_use_last_update_date=False
                )
            year_fetch_error = False

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # 既存データの日付キーを読み込む（年別ファイル）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "WOOD":
                    subdir = f"wood_{target_id.lower()}"
                    fname = f"wood_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(
                        f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                    )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Training date filtering for WC（年フィルタリング）
                            if rec_id == "WC" and "training_date" in parsed:
                                try:
                                    training_date_str = parsed.get(
                                        "training_date", ""
                                    ).strip()
                                    if len(training_date_str) == 8:  # YYYYMMDD形式
                                        training_year = int(training_date_str[:4])
                                        if training_year != year:
                                            skipped_count += 1
                                            continue
                                        # 2021年の場合は7月27日以降のみ
                                        if training_year == 2021:
                                            training_month_day = training_date_str[4:8]
                                            if training_month_day < "0727":
                                                skipped_count += 1
                                                continue
                                except:
                                    continue

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = None
                            should_skip = False

                            # 調教データ（WC）の場合
                            if (
                                rec_id == "WC"
                                and "ketto_num" in parsed
                                and "training_date" in parsed
                            ):
                                record_key = (
                                    parsed.get("ketto_num", ""),
                                    parsed.get("training_date", ""),
                                )
                                should_skip = record_key in existing_keys

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            # 年別ファイルに保存（サブディレクトリ内）
                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "WOOD":
                                    subdir = f"wood_{rec_id.lower()}"
                                    fname = f"wood_{rec_id.lower()}_{year}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
                year_fetch_error = True
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存（年別ファイル、サブディレクトリ内）
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "WOOD":
                        subdir = f"wood_{rid.lower()}"
                        fname = f"wood_{rid.lower()}_{year}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}_{year}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

            # 増分取得の状態を更新（最新年かつエラーなしの場合のみ）。
            # 失敗時は state を更新しない（update_jra_data と同じ規約）。
            if year == wc_latest_year and not year_fetch_error:
                today_end = datetime.now().strftime("%Y%m%d") + "235959"
                actual_end = min(year_end, today_end)
                wc_state["last_success_end"] = actual_end
                wc_state["last_run_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
                _save_state(wc_state_path, wc_state)
                print(f"Saved state: {wc_state_path}")

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_jra_data(start_date_str="20180101000000", end_date_str="20251231235959"):
    """
    Core logic to fetch data from JRA-VAN
    BLDN/MING: 全期間を1ファイルで取得
    RACE/SLOP/WOOD: 指定期間に応じて年別ファイルで取得
    """
    print("=== JRA-VAN Data Client (Year-based file saving for RACE/SLOP/WOOD) ===")

    # モード判定は日付正規化・年計算より先（INCR: が付いたまま [:4] すると ValueError になる）
    # - full: 既存のセットアップ/全件寄りの option を使う（BLDN=3, MING=4, RACE=4, SLOP/WOOD=3）
    # - incremental: 差分更新寄りの option=2 を使う（前回成功→昨日などの短い期間向け）
    mode = "full"
    if isinstance(start_date_str, str) and start_date_str.startswith("INCR:"):
        mode = "incremental"
        start_date_str = start_date_str.replace("INCR:", "", 1)

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    # 出力ディレクトリの準備
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output"
    output_dir = str(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 年別取得の範囲（start/end引数に追従）
    START_YEAR = int(start_date_str[:4])
    END_YEAR = int(end_date_str[:4])
    # RACE: max(2015, START_YEAR) - END_YEAR
    RACE_YEARS = list(range(max(2015, START_YEAR), END_YEAR + 1))
    # SLOP (HC): max(2015, START_YEAR) - END_YEAR
    HC_YEARS = list(range(max(2015, START_YEAR), END_YEAR + 1))
    # WOOD (WC): max(2021, START_YEAR) - END_YEAR（WCは2021年7月27日以降のデータのみ）
    WC_YEARS = list(range(max(2021, START_YEAR), END_YEAR + 1))

    print(f"fetch_jra_data: mode={mode!r}, window {start_date_str} -> {end_date_str}")

    try:
        # ==========================================
        # グループ1: 全期間を1ファイルで取得（BLDN, MING）
        # ==========================================
        FULL_PERIOD_TASKS = [
            # BLDN: 血統（1986年から2024年までの全データ）
            {
                "dataspec": "BLDN",
                "option": 2 if mode == "incremental" else 3,
                "start_date": "19860101000000",  # 1986年から
                "target_ids": ["HN", "SK", "BT"],
            },
            # MING: マイニング（2024年までの全データ）
            {
                "dataspec": "MING",
                "option": 2 if mode == "incremental" else 4,
                "start_date": "20150101000000",  # 2015年から
                "target_ids": ["DM", "TM"],
            },
        ]

        for task in FULL_PERIOD_TASKS:
            # クライアント初期化
            try:
                client = JRAVANClient()
            except Exception as e:
                print(
                    f"Failed to initialize JV-Link client for {task['dataspec']}: {e}"
                )
                print("Skipping this task and all subsequent tasks.")
                raise  # JV-Linkの初期化に失敗した場合は、全体の処理を中止
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            # incremental の場合、前回成功以降だけを狙うため start_date_str を優先（ただしデータ開始日より前にはしない）
            task_start_date = max(task["start_date"], start_date_str)

            print(f"\n--- Processing {ds} ---")

            # -----------------------------------------------------------
            # 出力ディレクトリの準備（プロジェクトルートからの相対パス）
            # -----------------------------------------------------------
            # スクリプトの場所を基準にプロジェクトルートを取得
            script_dir = Path(__file__).parent.parent.parent
            output_dir = script_dir / "data" / "output"
            output_dir = str(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # -----------------------------------------------------------

            time.sleep(2)

            # 既存データの日付キーを読み込む（各ターゲットIDごと）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "BLDN":
                    subdir = f"blod_{target_id.lower()}"
                    fname = f"blod_{target_id.lower()}.csv"
                elif ds == "MING":
                    subdir = f"ming_{target_id.lower()}"
                    fname = f"ming_{target_id.lower()}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (From: {task_start_date}, To: {end_date_str}, Opt: {opt})..."
            )
            pbar = tqdm(desc=f"Fetching {ds}", unit="chunks", position=0, leave=True)

            try:
                for raw_chunk in client.get_data(
                    ds, task_start_date, opt, end_date_str
                ):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Data Kubun Filtering
                            # HRは data_kubun の値に関係なく有効なデータなので、フィルタリングから除外
                            if (
                                rec_id in {"RA", "SE"}
                                and parsed.get("data_kubun") != "7"
                            ):
                                continue

                            # Date Filtering (BLDN/MINGは全期間取得のため、年フィルタリング不要)
                            # セットに変換してinチェックを高速化
                            is_master_like = rec_id in {
                                "HN",
                                "SK",
                                "BT",
                                "HC",
                                "WC",
                                "UM",
                            }
                            # BLDN/MINGは全期間取得なので、年フィルタリングは行わない
                            if not is_master_like and ds not in ["BLDN", "MING"]:
                                if "year" in parsed:
                                    try:
                                        rec_year = int(parsed["year"])
                                        # end_date_str の年まで取得
                                        if rec_year > END_YEAR:
                                            continue
                                    except:
                                        pass

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = _extract_record_key(parsed, rec_id)
                            should_skip = (
                                record_key is not None and record_key in existing_keys
                            )

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "BLDN":
                                    subdir = f"blod_{rec_id.lower()}"
                                    fname = f"blod_{rec_id.lower()}.csv"
                                elif ds == "MING":
                                    subdir = f"ming_{rec_id.lower()}"
                                    fname = f"ming_{rec_id.lower()}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)

                                # 保存したデータのキーを既存キーセットに追加（重複防止のため）
                                saved_keys = set()
                                for record in records_buffer[rec_id]:
                                    key = _extract_record_key(record, rec_id)
                                    if key is not None:
                                        saved_keys.add(key)

                                # 既存キーセットに追加
                                existing_keys.update(saved_keys)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                        )

            except Exception as e:
                print(f"!!! Error processing {ds}: {e}")
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "BLDN":
                        subdir = f"blod_{rid.lower()}"
                        fname = f"blod_{rid.lower()}.csv"
                    elif ds == "MING":
                        subdir = f"ming_{rid.lower()}"
                        fname = f"ming_{rid.lower()}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)
                else:
                    # デバッグ: バッファが空のrec_idを表示
                    if rid in targets:
                        print(
                            f"  Warning: {rid}データは処理されましたが、保存するレコードがありませんでした。"
                        )

            # デバッグ: 各ターゲットIDごとの処理状況を表示
            for target_id in targets:
                count = len(records_buffer.get(target_id, []))
                if count == 0 and target_id not in records_buffer:
                    print(f"  Warning: {target_id}レコードは処理されませんでした。")
                elif count > 0:
                    print(f"  {target_id}: {count}件がバッファにあります（保存処理へ）")

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds}.")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
                )

        # ==========================================
        # グループ2: 年別ファイルで取得（RACE, SLOP, WOOD）
        # ==========================================
        # RACE: レース情報
        RACE_TASK = {
            "dataspec": "RACE",
            "option": 2 if mode == "incremental" else 4,
            "target_ids": ["RA", "SE", "HR"],
            "years": RACE_YEARS,
        }
        # SLOP (HC): 坂路調教（2015-2024年）
        HC_TASK = {
            "dataspec": "SLOP",
            "option": 2 if mode == "incremental" else 3,
            "target_ids": ["HC"],
            "years": HC_YEARS,
        }
        # WOOD (WC): ウッドチップ調教（2021-2024年）
        WC_TASK = {
            "dataspec": "WOOD",
            "option": 2 if mode == "incremental" else 3,
            "target_ids": ["WC"],
            "years": WC_YEARS,
        }

        # RACEタスクを処理（2016-2024年）
        for year in RACE_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"
            # 期間境界を絞る（無駄な年全体取得を避ける）
            if year == START_YEAR:
                year_start = max(year_start, start_date_str)
            if year == END_YEAR:
                year_end = min(year_end, end_date_str)

            # RACEタスクを処理
            task = RACE_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            task_start_date = year_start

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # RACE: RA/SE は data_kubun 2+7 をマージ（7優先）、HR は従来のキー重複スキップ
            existing_data_keys = {}
            ra_merge_map = {}
            se_merge_map = {}
            for target_id in targets:
                if ds == "RACE":
                    subdir = f"race_{target_id.lower()}"
                    fname = f"race_{target_id.lower()}_{year}.csv"
                elif ds == "SLOP":
                    subdir = f"slop_{target_id.lower()}"
                    fname = f"slop_{target_id.lower()}_{year}.csv"
                elif ds == "WOOD":
                    subdir = f"wood_{target_id.lower()}"
                    fname = f"wood_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                if ds == "RACE" and target_id == "RA":
                    ra_merge_map = _load_race_year_merge_map(fpath, "RA")
                    if ra_merge_map:
                        print(
                            f"  RA: {len(ra_merge_map)} keys loaded for kubun merge ({year})"
                        )
                elif ds == "RACE" and target_id == "SE":
                    se_merge_map = _load_race_year_merge_map(fpath, "SE")
                    if se_merge_map:
                        print(
                            f"  SE: {len(se_merge_map)} keys loaded for kubun merge ({year})"
                        )
                else:
                    existing_keys = load_existing_dates(fpath, target_id)
                    existing_data_keys[target_id] = existing_keys
                    if existing_keys:
                        print(
                            f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                        )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Year Filtering（指定年のデータのみ）- 早期フィルタリング
                            if "year" in parsed:
                                try:
                                    rec_year = int(parsed["year"])
                                    if rec_year != year:
                                        skipped_count += 1
                                        continue
                                except:
                                    continue

                            # RACE ターゲットのみ（このループでは RA/SE/HR）
                            if ds == "RACE" and rec_id in {"RA", "SE"}:
                                parsed["raw_hex"] = line_bytes.hex()
                                merge = ra_merge_map if rec_id == "RA" else se_merge_map
                                if _race_stream_try_merge(
                                    merge,
                                    parsed,
                                    rec_id,
                                    _FETCH_JRA_RACE_KUBUNS,
                                ):
                                    total_count += 1
                                else:
                                    skipped_count += 1
                                continue

                            if ds == "RACE" and rec_id == "HR":
                                existing_keys = existing_data_keys.get("HR", set())
                                record_key = _extract_record_key(parsed, "HR")
                                if not record_key or record_key in existing_keys:
                                    skipped_count += 1
                                    continue
                                parsed["raw_hex"] = line_bytes.hex()
                                records_buffer.setdefault("HR", []).append(parsed)
                                existing_keys.add(record_key)
                                total_count += 1
                                hr_buf = records_buffer["HR"]
                                if len(hr_buf) >= 100000:
                                    subdir_path = os.path.join(output_dir, "race_hr")
                                    if not os.path.exists(subdir_path):
                                        os.makedirs(subdir_path, exist_ok=True)
                                    save_path = os.path.join(
                                        subdir_path, f"race_hr_{year}.csv"
                                    )
                                    fields = get_schema_fieldnames("HR") + ["raw_hex"]
                                    save_to_csv(hr_buf, save_path, fields)
                                    records_buffer["HR"] = []
                                continue

                            # 上記以外（SLOP/WOOD 等の別ループで来ない想定だが念のため）
                            skipped_count += 1

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # RACE: RA/SE はマージ結果を年次ファイルへ上書き
            if ds == "RACE":
                if ra_merge_map:
                    subdir_path = os.path.join(output_dir, "race_ra")
                    os.makedirs(subdir_path, exist_ok=True)
                    save_path = os.path.join(subdir_path, f"race_ra_{year}.csv")
                    rows = [t[1] for t in ra_merge_map.values()]
                    fields = get_schema_fieldnames("RA") + ["raw_hex"]
                    save_to_csv(rows, save_path, fields, append=False)
                if se_merge_map:
                    subdir_path = os.path.join(output_dir, "race_se")
                    os.makedirs(subdir_path, exist_ok=True)
                    save_path = os.path.join(subdir_path, f"race_se_{year}.csv")
                    rows = [t[1] for t in se_merge_map.values()]
                    fields = get_schema_fieldnames("SE") + ["raw_hex"]
                    save_to_csv(rows, save_path, fields, append=False)
                for rid, data_list in records_buffer.items():
                    if rid == "HR" and data_list:
                        subdir_path = os.path.join(output_dir, "race_hr")
                        if not os.path.exists(subdir_path):
                            os.makedirs(subdir_path, exist_ok=True)
                        save_path = os.path.join(subdir_path, f"race_hr_{year}.csv")
                        fields = get_schema_fieldnames("HR") + ["raw_hex"]
                        save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

        # HCタスクを処理（2015-2024年）
        for year in HC_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"
            if year == START_YEAR:
                year_start = max(year_start, start_date_str)
            if year == END_YEAR:
                year_end = min(year_end, end_date_str)

            # HCタスクを処理
            task = HC_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            task_start_date = year_start

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # 既存データの日付キーを読み込む（年別ファイル）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "SLOP":
                    subdir = f"slop_{target_id.lower()}"
                    fname = f"slop_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(
                        f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                    )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Training date filtering for HC（年フィルタリング）
                            if rec_id == "HC" and "training_date" in parsed:
                                try:
                                    training_date_str = parsed.get(
                                        "training_date", ""
                                    ).strip()
                                    if len(training_date_str) == 8:  # YYYYMMDD形式
                                        training_year = int(training_date_str[:4])
                                        if training_year != year:
                                            skipped_count += 1
                                            continue
                                except:
                                    continue

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = None
                            should_skip = False

                            # 調教データ（HC）の場合
                            if (
                                rec_id == "HC"
                                and "ketto_num" in parsed
                                and "training_date" in parsed
                            ):
                                record_key = (
                                    parsed.get("ketto_num", ""),
                                    parsed.get("training_date", ""),
                                )
                                should_skip = record_key in existing_keys

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            # 年別ファイルに保存（サブディレクトリ内）
                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "SLOP":
                                    subdir = f"slop_{rec_id.lower()}"
                                    fname = f"slop_{rec_id.lower()}_{year}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存（年別ファイル、サブディレクトリ内）
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "SLOP":
                        subdir = f"slop_{rid.lower()}"
                        fname = f"slop_{rid.lower()}_{year}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}_{year}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

        # WCタスクを処理（2021-2024年、2021年7月27日以降）
        for year in WC_TASK["years"]:
            year_start = f"{year}0101000000"
            year_end = f"{year}1231235959"
            if year == START_YEAR:
                year_start = max(year_start, start_date_str)
            if year == END_YEAR:
                year_end = min(year_end, end_date_str)

            # WCタスクを処理
            task = WC_TASK
            # クライアント初期化
            client = JRAVANClient()
            try:
                client.login()
            except:
                print("Login failed, skipping task.")
                if client:
                    client.close()
                continue

            ds = task["dataspec"]
            opt = task["option"]
            targets = set(task["target_ids"])  # セットに変換してinチェックを高速化
            # WCは2021年7月27日以降のデータのみ（JV-Link仕様書より）
            # 2021年の場合は7月27日以降から開始
            if year == 2021:
                task_start_date = max("20210727000000", year_start)
            else:
                task_start_date = year_start

            print(f"\n--- Processing {ds} (Year: {year}) ---")

            time.sleep(2)

            # 既存データの日付キーを読み込む（年別ファイル）
            existing_data_keys = {}
            for target_id in targets:
                # サブディレクトリのパスを構築
                if ds == "WOOD":
                    subdir = f"wood_{target_id.lower()}"
                    fname = f"wood_{target_id.lower()}_{year}.csv"
                else:
                    subdir = f"{ds.lower()}_{target_id.lower()}"
                    fname = f"{ds.lower()}_{target_id.lower()}_{year}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                fpath = os.path.join(subdir_path, fname)
                existing_keys = load_existing_dates(fpath, target_id)
                existing_data_keys[target_id] = existing_keys
                if existing_keys:
                    print(
                        f"  {target_id}: 既存データ {len(existing_keys)}件を検出 ({year}年)"
                    )

            records_buffer = {}
            total_count = 0
            skipped_count = 0
            chunk_count = 0  # チャンクカウンタ（進捗バー更新頻度制御用）

            print(
                f"Requesting {ds} (Year: {year}, From: {task_start_date}, To: {year_end}, Opt: {opt})..."
            )
            pbar = tqdm(
                desc=f"Fetching {ds} ({year})",
                unit="chunks",
                position=0,
                leave=True,
            )

            try:
                # 年別取得の場合は、終了日時を指定してJV-Link側でデータ取得を制限
                for raw_chunk in client.get_data(ds, task_start_date, opt, year_end):
                    chunk_count += 1
                    pbar.update(1)

                    # 文字列エンコーディングを一度だけ実行
                    if isinstance(raw_chunk, bytes):
                        try:
                            raw_chunk = raw_chunk.decode("cp932", "replace")
                        except:
                            continue

                    if not raw_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        # rstrip()を使用（末尾の改行のみ削除、先頭の空白は保持）
                        line = line.rstrip("\r\n")
                        if not line:
                            continue

                        # 早期フィルタリング: レコードIDを先にチェック
                        if len(line) < 2:
                            continue
                        rec_id = line[:2]
                        if targets and rec_id not in targets:
                            continue

                        # バイト変換（必要な場合のみ）
                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue
                        if len(line_bytes) < 2:
                            continue

                        if rec_id in SCHEMAS:
                            parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                            # Training date filtering for WC（年フィルタリング）
                            if rec_id == "WC" and "training_date" in parsed:
                                try:
                                    training_date_str = parsed.get(
                                        "training_date", ""
                                    ).strip()
                                    if len(training_date_str) == 8:  # YYYYMMDD形式
                                        training_year = int(training_date_str[:4])
                                        if training_year != year:
                                            skipped_count += 1
                                            continue
                                        # 2021年の場合は7月27日以降のみ
                                        if training_year == 2021:
                                            training_month_day = training_date_str[4:8]
                                            if training_month_day < "0727":
                                                skipped_count += 1
                                                continue
                                except:
                                    continue

                            # 既存データチェック（重複スキップ）- 最適化
                            existing_keys = existing_data_keys.get(rec_id, set())
                            record_key = None
                            should_skip = False

                            # 調教データ（WC）の場合
                            if (
                                rec_id == "WC"
                                and "ketto_num" in parsed
                                and "training_date" in parsed
                            ):
                                record_key = (
                                    parsed.get("ketto_num", ""),
                                    parsed.get("training_date", ""),
                                )
                                should_skip = record_key in existing_keys

                            if should_skip:
                                skipped_count += 1
                                continue

                            # 新しいデータのみ追加
                            parsed["raw_hex"] = line_bytes.hex()
                            if rec_id not in records_buffer:
                                records_buffer[rec_id] = []
                            records_buffer[rec_id].append(parsed)

                            # 既存キーセットに追加（メモリ内で重複チェック）
                            if record_key:
                                existing_keys.add(record_key)

                            total_count += 1

                            # 年別ファイルに保存（サブディレクトリ内）
                            if len(records_buffer[rec_id]) >= 100000:
                                # サブディレクトリのパスを構築
                                if ds == "WOOD":
                                    subdir = f"wood_{rec_id.lower()}"
                                    fname = f"wood_{rec_id.lower()}_{year}.csv"
                                else:
                                    subdir = f"{ds.lower()}_{rec_id.lower()}"
                                    fname = f"{ds.lower()}_{rec_id.lower()}_{year}.csv"

                                subdir_path = os.path.join(output_dir, subdir)
                                if not os.path.exists(subdir_path):
                                    os.makedirs(subdir_path, exist_ok=True)

                                save_path = os.path.join(subdir_path, fname)
                                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                                save_to_csv(records_buffer[rec_id], save_path, fields)
                                records_buffer[rec_id] = []

                    # 進捗バーの更新頻度を下げる（10チャンクごと）
                    if chunk_count % 10 == 0:
                        pbar.set_postfix(
                            {
                                "new": f"{total_count:,}",
                                "skipped": f"{skipped_count:,}",
                            }
                        )

            except Exception as e:
                print(f"!!! Error processing {ds} ({year}): {e}")
            finally:
                pbar.close()
                if client:
                    client.close()
                    client = None

            # 残りのデータを保存（年別ファイル、サブディレクトリ内）
            for rid, data_list in records_buffer.items():
                if data_list:
                    # サブディレクトリのパスを構築
                    if ds == "WOOD":
                        subdir = f"wood_{rid.lower()}"
                        fname = f"wood_{rid.lower()}_{year}.csv"
                    else:
                        subdir = f"{ds.lower()}_{rid.lower()}"
                        fname = f"{ds.lower()}_{rid.lower()}_{year}.csv"

                    subdir_path = os.path.join(output_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path, exist_ok=True)

                    save_path = os.path.join(subdir_path, fname)
                    fields = get_schema_fieldnames(rid) + ["raw_hex"]
                    save_to_csv(data_list, save_path, fields)

            if total_count == 0:
                if skipped_count > 0:
                    print(
                        f"No new records found for {ds} ({year}). ({skipped_count} existing records skipped)"
                    )
                else:
                    print(f"No relevant records found for {ds} ({year}).")
            else:
                print(
                    f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds} ({year})."
                )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
        raise  # 例外を再発生させて、呼び出し元にエラーを伝える
    finally:
        if "client" in locals() and client:
            try:
                client.close()
            except:
                pass  # クリーンアップ時のエラーは無視
        print("\nExiting.")


def _default_state_path(output_dir: Path) -> Path:
    return output_dir / "state" / "jv_last_update.json"


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: state file is broken or unreadable ({path}): {e}")
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write to avoid truncation/corruption on interruption.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def _yesterday_end() -> str:
    y = datetime.now() - timedelta(days=1)
    return y.strftime("%Y%m%d") + "235959"


def _jv_normalize_bound(s: str, *, is_end: bool) -> str:
    """YYYYMMDD → 時刻付き（開始 000000 / 終了 235959）にそろえる。"""
    t = str(s).strip()
    if len(t) == 8 and t.isdigit():
        return t + ("235959" if is_end else "000000")
    return t


def _jv_resolve_start_datetime(
    state: dict,
    *,
    default: str,
    also_use_last_update_date: bool,
) -> str:
    """
    増分更新の開始日時（YYYYMMDDHHMMSS）を state から決める。
    last_success_end があるときはその日の翌日 00:00:00。
    also_use_last_update_date のときだけ last_update_date もフォールバック。
    """
    last_success_end = state.get("last_success_end")
    if last_success_end:
        last_end_str = str(last_success_end).strip()
        if len(last_end_str) >= 8 and last_end_str[:8].isdigit():
            # Start from the beginning of the same day to avoid missing
            # same-day corrections or late-arriving updates.
            return last_end_str[:8] + "000000"
    if also_use_last_update_date:
        lud = state.get("last_update_date")
        if lud:
            try:
                last_date = datetime.strptime(str(lud).strip(), "%Y%m%d")
                next_date = last_date + timedelta(days=1)
                return next_date.strftime("%Y%m%d") + "000000"
            except ValueError:
                print(f"Warning: invalid last_update_date in state: {lud!r}")
    return _jv_normalize_bound(default, is_end=False)


def _jv_fetch_output_bundle(
    start: str,
    end: str,
    *,
    incremental: bool,
) -> None:
    """fetch_jra_data 呼び出し（incremental 時は INCR: を先頭に付与）。"""
    fetch_start = start
    if incremental and not str(fetch_start).startswith("INCR:"):
        fetch_start = "INCR:" + str(fetch_start)
    fetch_jra_data(start_date_str=fetch_start, end_date_str=end)


def update_common_output_all(
    *,
    output_dir: str | None = None,
    state_path: str | None = None,
    start_default: str = "20150101000000",
    end_time: str | None = None,
) -> tuple[str, str]:
    """
    common/data/output を対象に、RA/SE/HR/HN/SK/BT/DM/TM/HC/WC をまとめて差分更新する。

    更新範囲:
      start = 前回成功の end_time（state から取得） もしくは start_default
      end   = end_time もしくは「昨日 23:59:59」

    state（前回取得の決め方）:
      - `common/data/output/state/jv_last_update.json` に `last_success_end` を保存して参照する
    """
    script_dir = Path(__file__).parent.parent.parent
    out_dir = Path(output_dir) if output_dir else (script_dir / "data" / "output")
    st_path = Path(state_path) if state_path else _default_state_path(out_dir)

    state = _load_state(st_path)

    start = _jv_resolve_start_datetime(
        state, default=start_default, also_use_last_update_date=False
    )
    start = _jv_normalize_bound(start, is_end=False)
    end = _jv_normalize_bound(str(end_time or _yesterday_end()), is_end=True)

    # 期間が逆転していたら何もしない
    if start > end:
        print(f"Nothing to update (start={start} > end={end})")
        return start, end

    _jv_fetch_output_bundle(start, end, incremental=True)

    # 成功したら state を更新
    state["last_success_end"] = end
    state["last_run_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
    _save_state(st_path, state)
    print(f"Saved state: {st_path}")

    return start, end


def update_jra_data(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    **kwargs,
) -> tuple[str, str]:
    """
    common/data/output 向けの JRA-VAN 一括取得（fetch_jra_data）を実行し、成功時のみ state を更新する。

    オプションはすべてキーワードで ``**kwargs`` として受け取る（Jupyter の autoreload で
    シグネチャが古いまま残り TypeError になるのを避けるため）。

    流れ:
      1. 開始・終了の日時範囲を決める（引数 > state > 既定）。
      2. ``_jv_fetch_output_bundle`` で ``fetch_jra_data`` を呼ぶ（既定は INCR 付き差分＝BLDN/MING の -303 を避けやすい）。
      3. 例外がなければ ``jv_last_update.json`` に ``last_update_date`` / ``last_success_end`` / ``last_run_at`` を書く。

    Args:
        start_date_str:
            開始（YYYYMMDD または YYYYMMDDHHMMSS）。省略時は state の
            ``last_success_end``（翌日 00:00:00）→ なければ ``last_update_date`` 翌日 → ``start_default``。
        end_date_str:
            終了。省略時は ``end_through`` に応じて「今日 23:59:59」または「昨日 23:59:59」。
        kwargs:
            - ``output_dir``: 出力ルート（未指定時は ``common/data/output``）
            - ``state_path``: 状態 JSON（未指定時は ``output_dir/state/jv_last_update.json``）
            - ``incremental``: 既定 True。False で ``INCR:`` なし（フル寄り）
            - ``start_default``: 既定 ``\"20150101000000\"``（8 桁も可）
            - ``end_through``: ``\"today\"``（既定）または ``\"yesterday\"``

    Returns:
        実際に使った ``(start_date_str, end_date_str)``（正規化済み。``INCR:`` は含まない）。

    状態ファイルのキー:
        - ``last_update_date``: 終了日の YYYYMMDD
        - ``last_success_end``: 終了の YYYYMMDDHHMMSS
        - ``last_run_at``: 実行時刻
    """
    output_dir = kwargs.pop("output_dir", None)
    state_path = kwargs.pop("state_path", None)
    incremental = kwargs.pop("incremental", True)
    start_default = kwargs.pop("start_default", "20150101000000")
    end_through = kwargs.pop("end_through", "today")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"update_jra_data() got unexpected keyword argument(s): {bad}")

    script_dir = Path(__file__).parent.parent.parent
    out_dir = Path(output_dir) if output_dir else (script_dir / "data" / "output")
    st_path = Path(state_path) if state_path else _default_state_path(out_dir)
    state = _load_state(st_path)

    if start_date_str is None:
        start_date_str = _jv_resolve_start_datetime(
            state, default=start_default, also_use_last_update_date=True
        )
    else:
        start_date_str = _jv_normalize_bound(str(start_date_str).strip(), is_end=False)

    if end_date_str is None:
        et = str(end_through).strip().lower()
        if et == "yesterday":
            end_date_str = _yesterday_end()
        elif et == "today":
            end_date_str = datetime.now().strftime("%Y%m%d") + "235959"
        else:
            raise ValueError(
                f"end_through は 'today' または 'yesterday' です: {end_through!r}"
            )
    end_date_str = _jv_normalize_bound(str(end_date_str).strip(), is_end=True)

    if start_date_str > end_date_str:
        print(f"Nothing to update (start={start_date_str} > end={end_date_str})")
        return start_date_str, end_date_str

    print(f"=== update_jra_data: {start_date_str} -> {end_date_str} ===")
    if incremental:
        print("  fetch: incremental (INCR:) / BLDN-MING Opt2")

    try:
        _jv_fetch_output_bundle(start_date_str, end_date_str, incremental=incremental)
        update_date = end_date_str[:8]
        state["last_update_date"] = update_date
        state["last_success_end"] = end_date_str
        state["last_run_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
        _save_state(st_path, state)
        print(f"Saved update date: {update_date} (YYYYMMDD format)")
        print(f"Saved state: {st_path}")
    except Exception as e:
        print(f"Error during update: {e}")
        print("State file will NOT be updated due to error.")
        raise

    return start_date_str, end_date_str


def get_race_data(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    include_entry_kubun_1: bool = False,
    target_kubun: str = "both",
) -> tuple[str, str]:
    """
    SEとRAのデータのみを取得し、予測対象（今日以降）を抽出して保存する。

    Args:
        start_date_str: 開始日時（YYYYMMDDまたはYYYYMMDDHHMMSS形式）
                       指定しない場合、昨日の日付を使用
        end_date_str: 終了日時（YYYYMMDDまたはYYYYMMDDHHMMSS形式）
                     指定しない場合、本日の日付を使用
        output_dir: 出力ディレクトリ（デフォルト: main/data/race）
        include_entry_kubun_1: Trueの場合、data_kubun='1'（出走馬名）も含める
                              Falseの場合、target_kubun のみを対象にする
        target_kubun: 取得するデータ区分
                      - "7": 確定系（当日情報を含む）
                      - "2": 出馬表
                      - "both": 2と7の両方（同一キー重複時は7を優先）

    Returns:
        (start_date_str, end_date_str) のタプル

    保存先:
        - main/data/race/race_ra.csv
        - main/data/race/race_se.csv
    """
    print("=== get_race_data: 予測対象レースデータ取得 ===")

    # プロジェクトルートの取得
    # __file__ = common/data/src/get_data.py
    # parent.parent.parent.parent = プロジェクトルート
    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    main_race_dir.mkdir(parents=True, exist_ok=True)

    # 今日と昨日の日付を取得
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y%m%d")
    yesterday_str = yesterday.strftime("%Y%m%d")

    # 日付の設定（デフォルト: 昨日から今日まで）
    if start_date_str is None:
        start_date_str = yesterday_str + "000000"
    if end_date_str is None:
        end_date_str = today_str + "235959"

    # 日付形式の正規化
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    # option=1（通常データ）を使用（過去データを含むため）
    # option=2（今週データ）は未来のデータのみを取得するため、昨日のデータは取得できない
    fetch_option = 1
    fetch_start_date = start_date_str[:8] + "000000"
    # 同一開催日のみの RACE + option=1 で JVOpen が -1 になる環境があるため、
    # JVOpen の From だけ 1 日前に広げる（保存対象の解釈は既存フィルタのまま）。
    jv_open_from = fetch_start_date
    if fetch_start_date[:8] == end_date_str[:8]:
        d0 = datetime.strptime(fetch_start_date[:8], "%Y%m%d")
        jv_open_from = (d0 - timedelta(days=1)).strftime("%Y%m%d") + "000000"

    print(f"取得期間: {fetch_start_date} -> {end_date_str}")
    if jv_open_from != fetch_start_date:
        print(f"JVOpen From 補正: {jv_open_from} (単日指定時の JVOpen -1 回避)")
    print(f"保存先: {main_race_dir.resolve()}")
    print(f"Option: {fetch_option} (通常データ)")

    # クライアント初期化
    client = JRAVANClient()
    try:
        client.login()
    except Exception as e:
        print(f"Login failed: {e}")
        if client:
            client.close()
        raise

    # データ取得とフィルタリング
    records_ra = {}
    records_se = {}
    total_count = 0
    filtered_count = 0
    replaced_count = 0

    try:
        print("\n--- RACEデータ取得中 (RA, SE) ---")
        pbar = tqdm(desc="Fetching RACE", unit="chunks", position=0, leave=True)

        # data_kubun の許可リスト
        mode = str(target_kubun).strip().lower()
        if mode not in {"2", "7", "both"}:
            raise ValueError(
                f"target_kubun must be one of '2', '7', 'both', got: {target_kubun}"
            )

        if mode == "both":
            allowed_kubun = {"2", "7"}
        elif mode == "7":
            allowed_kubun = {"7"}
        else:
            allowed_kubun = {"2"}

        if include_entry_kubun_1 and mode in {"2", "both"}:
            allowed_kubun.add("1")

        # 同一キー重複時の優先順位（high is better）
        kubun_priority = {"7": 3, "2": 2, "1": 1}

        for raw_chunk in client.get_data(
            "RACE", jv_open_from, option=fetch_option, end_date=end_date_str
        ):
            pbar.update(1)

            # 文字列エンコーディング
            if isinstance(raw_chunk, bytes):
                try:
                    raw_chunk = raw_chunk.decode("cp932", "replace")
                except:
                    continue

            if not raw_chunk:
                continue

            lines = raw_chunk.split("\n")

            for line in lines:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                # レコードIDをチェック
                if len(line) < 2:
                    continue
                rec_id = line[:2]
                if rec_id not in {"RA", "SE"}:
                    continue

                # バイト変換
                try:
                    line_bytes = line.encode("cp932", "replace")
                except:
                    continue
                if len(line_bytes) < 2:
                    continue

                if rec_id in SCHEMAS:
                    parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])
                    total_count += 1

                    # フィルタリング条件1: data_kubun が 1 または 2
                    data_kubun = parsed.get("data_kubun", "").strip()
                    if data_kubun not in allowed_kubun:
                        filtered_count += 1
                        continue

                    # フィルタリング条件2: 開催年月日が今日以降
                    year = parsed.get("year", "").strip()
                    month_day = parsed.get("month_day", "").strip()
                    if not year or not month_day:
                        filtered_count += 1
                        continue

                    try:
                        # YYYYMMDD形式に変換
                        race_date_str = year.zfill(4) + month_day.zfill(4)
                        if race_date_str < today_str:
                            filtered_count += 1
                            continue
                    except (ValueError, TypeError):
                        filtered_count += 1
                        continue

                    # フィルタリング通過したデータを追加
                    parsed["raw_hex"] = line_bytes.hex()
                    rec_key_base = (
                        str(parsed.get("year", "")),
                        str(parsed.get("month_day", "")),
                        str(parsed.get("course_code", "")),
                        str(parsed.get("kai", "")),
                        str(parsed.get("nichi", "")),
                        str(parsed.get("race_num", "")),
                    )
                    current_priority = kubun_priority.get(data_kubun, 0)

                    if rec_id == "RA":
                        existing = records_ra.get(rec_key_base)
                        if existing is None:
                            records_ra[rec_key_base] = parsed
                        else:
                            existing_priority = kubun_priority.get(
                                str(existing.get("data_kubun", "")),
                                0,
                            )
                            if current_priority >= existing_priority:
                                records_ra[rec_key_base] = parsed
                                replaced_count += 1
                    elif rec_id == "SE":
                        se_key = rec_key_base + (str(parsed.get("horse_num", "")),)
                        existing = records_se.get(se_key)
                        if existing is None:
                            records_se[se_key] = parsed
                        else:
                            existing_priority = kubun_priority.get(
                                str(existing.get("data_kubun", "")),
                                0,
                            )
                            if current_priority >= existing_priority:
                                records_se[se_key] = parsed
                                replaced_count += 1

                    # 進捗表示
                    if (len(records_ra) + len(records_se)) % 1000 == 0:
                        pbar.set_postfix(
                            {
                                "RA": len(records_ra),
                                "SE": len(records_se),
                                "filtered": filtered_count,
                                "replaced": replaced_count,
                            }
                        )

        pbar.close()

    except Exception as e:
        print(f"!!! Error during data fetch: {e}")
        raise
    finally:
        if client:
            client.close()

    # データを保存
    print(f"\n--- データ保存中 ---")
    records_ra_list = list(records_ra.values())
    records_se_list = list(records_se.values())

    print(f"RA: {len(records_ra_list)}件, SE: {len(records_se_list)}件")
    print(f"フィルタ除外: {filtered_count}件 / 総取得: {total_count}件")
    print(f"重複置換(優先kubun採用): {replaced_count}件")

    # RAデータの保存（上書きモード）
    ra_path = main_race_dir / "race_ra.csv"
    fields_ra = get_schema_fieldnames("RA") + ["raw_hex"]
    if records_ra_list:
        # 既存ファイルを削除してから保存（上書きモード）
        if ra_path.exists():
            ra_path.unlink()
        save_to_csv(records_ra_list, str(ra_path), fields_ra)
        print(f"Saved RA: {ra_path} ({len(records_ra_list)} records)")
    else:
        # 既存ファイルがある場合は保持（空データでの上書きを避ける）
        if ra_path.exists():
            print(f"No RA records. Kept existing file: {ra_path}")
        else:
            with open(ra_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields_ra)
                writer.writeheader()
            print(f"Created empty RA file: {ra_path}")

    # SEデータの保存（上書きモード）
    se_path = main_race_dir / "race_se.csv"
    fields_se = get_schema_fieldnames("SE") + ["raw_hex"]
    if records_se_list:
        # 既存ファイルを削除してから保存（上書きモード）
        if se_path.exists():
            se_path.unlink()
        save_to_csv(records_se_list, str(se_path), fields_se)
        print(f"Saved SE: {se_path} ({len(records_se_list)} records)")
    else:
        # 既存ファイルがある場合は保持（空データでの上書きを避ける）
        if se_path.exists():
            print(f"No SE records. Kept existing file: {se_path}")
        else:
            with open(se_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields_se)
                writer.writeheader()
            print(f"Created empty SE file: {se_path}")

    print(f"\n=== get_race_data 完了 ===")
    return start_date_str, end_date_str


def _normalize_datetime_range(start_date_str=None, end_date_str=None):
    """YYYYMMDD / YYYYMMDDHHMMSS を YYYYMMDDHHMMSS に正規化"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    if start_date_str is None:
        start_date_str = yesterday.strftime("%Y%m%d") + "000000"
    if end_date_str is None:
        end_date_str = today.strftime("%Y%m%d") + "235959"

    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    return start_date_str, end_date_str


# 0B11 速報馬体重: ヘッダ35byte + (馬番2 + 馬名36 + 体重3 + 符号1 + 増減3) × N
_WH_RT_HEADER_BYTES = 35
_WH_RT_BLOCK_BYTES = 45


def _looks_like_record_hex(val: str) -> bool:
    """DM/TM の生レコード全体が誤って1列に入った hex か判定する。"""
    s = str(val or "").strip()
    return len(s) > 40 and s[:4].upper() in ("444D", "544D")


def _normalize_dm_row(row: dict) -> dict:
    """破損した dm.csv 行を raw_hex から mining_pred_* を再パースする。"""
    out = dict(row)
    hex_src = str(row.get("raw_hex", "") or "").strip()
    if not hex_src and _looks_like_record_hex(row.get("mining_pred_1_horse_num", "")):
        hex_src = str(row["mining_pred_1_horse_num"]).strip()
    if not hex_src:
        return out
    try:
        raw = bytes.fromhex(hex_src)
    except ValueError:
        return out
    parsed = parse_fixed_width(raw, SCHEMAS["DM"])
    for key, val in parsed.items():
        if key.startswith("mining_pred_") or key in ("data_kubun", "record_separator"):
            out[key] = val
    out["raw_hex"] = hex_src
    return out


def _expand_wh_realtime_row(row: dict) -> list[dict]:
    """0B11 WH レース行を馬単位の dict リストに展開する。"""
    raw_hex = str(row.get("raw_hex", "") or "").strip()
    if not raw_hex:
        return []
    try:
        raw = bytes.fromhex(raw_hex)
    except ValueError:
        return []
    if len(raw) < _WH_RT_HEADER_BYTES + _WH_RT_BLOCK_BYTES:
        return []

    skip_keys = {
        "horse_num",
        "horse_weight",
        "weight_change_sign",
        "weight_change",
        "raw_hex",
        "record_separator",
    }
    base = {k: v for k, v in row.items() if k not in skip_keys}
    expanded: list[dict] = []
    pos = _WH_RT_HEADER_BYTES
    while pos + _WH_RT_BLOCK_BYTES <= len(raw):
        block = raw[pos : pos + _WH_RT_BLOCK_BYTES]
        horse_num = block[0:2].decode("ascii", errors="replace").strip()
        weight = block[38:41].decode("ascii", errors="replace").strip()
        sign = block[41:42].decode("ascii", errors="replace").strip()
        diff = block[42:45].decode("ascii", errors="replace").strip()
        if not horse_num.isdigit() or horse_num in ("", "00"):
            break
        if not weight.strip(" 0") and sign.strip(" ") in ("", "0"):
            break
        horse_row = dict(base)
        horse_row["horse_num"] = horse_num.zfill(2)
        horse_row["horse_weight"] = weight
        horse_row["weight_change_sign"] = sign
        horse_row["weight_change"] = diff
        expanded.append(horse_row)
        pos += _WH_RT_BLOCK_BYTES
    return expanded


def _expand_tm_realtime_row(row: dict) -> list[dict]:
    """TM レコードから馬番・タイム指数スコアのリストを返す。"""
    norm = dict(row)
    if _looks_like_record_hex(row.get("mining_pred_1_horse_num", "")):
        hex_src = str(row["mining_pred_1_horse_num"]).strip()
        try:
            parsed = parse_fixed_width(bytes.fromhex(hex_src), SCHEMAS["TM"])
            norm.update(parsed)
        except ValueError:
            pass
    scores: list[dict] = []
    for i in range(1, 19):
        hn = str(norm.get(f"mining_pred_{i}_horse_num", "")).strip()
        sc = str(norm.get(f"mining_pred_{i}_score", "")).strip()
        if not hn.isdigit() or hn in ("", "00"):
            continue
        scores.append({"horse_num": hn.zfill(2), "tm_score": sc})
    return scores


def _key_ra(row):
    return (
        str(row.get("year", "")),
        str(row.get("month_day", "")),
        str(row.get("course_code", "")),
        str(row.get("kai", "")),
        str(row.get("nichi", "")),
        str(row.get("race_num", "")),
    )


def _key_se(row):
    return _key_ra(row) + (str(row.get("horse_num", "")),)


def _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
    y = str(row.get("year", "")).zfill(4)
    md = str(row.get("month_day", "")).zfill(4)
    if not y.strip() or not md.strip():
        return False
    d = y + md
    return start_yyyymmdd <= d <= end_yyyymmdd


def _we_merge_usefulness_score(row: dict) -> int:
    """WE 1行のうち、天候・馬場として上書きに使える列の数（0 だけの値は除外）。"""
    n = 0
    for col in ("weather_code", "turf_condition", "dirt_condition"):
        v = _we_effective_code(row, col)
        if v and not all(ch == "0" for ch in v):
            n += 1
    return n


def _we_raw_tail_fields(row: dict) -> dict[str, str]:
    """
    WE 生行(raw_hex)末尾側の補助情報を取り出す。

    実運用では仕様差/版差で天候・馬場の有効値が末尾側に入るケースがあるため、
    通常カラムが 0 の場合のみ補助候補として使う。
    """
    raw_hex = str(row.get("raw_hex", "")).strip()
    if not raw_hex:
        return {}
    try:
        s = bytes.fromhex(raw_hex).decode("ascii", errors="ignore")
    except Exception:
        return {}
    if len(s) < 39:
        return {}
    return {
        # 1-indexed 28..35: 発表時刻（運用上は DDhhmmss として扱う）
        "announce_ddhhmmss": s[27:35].strip(),
        # 1-indexed 36..39 相当（0-indexed 35..38）
        "change_id": s[35:36].strip(),
        "weather_code_tail": s[36:37].strip(),
        "turf_condition_tail": s[37:38].strip(),
        "dirt_condition_tail": s[38:39].strip(),
    }


def _we_fieldnames_with_extras() -> list[str]:
    return get_schema_fieldnames("WE") + [
        "announce_ddhhmmss",
        "change_id",
        "weather_code_tail",
        "turf_condition_tail",
        "dirt_condition_tail",
        "raw_hex",
    ]


def _we_effective_code(row: dict, col: str) -> str:
    """
    WE の天候/馬場コードを実用上の有効値で返す。
    - まず通常カラム（現行スキーマ）
    - それが 0/空 のときのみ raw_hex 末尾側候補へフォールバック
    """
    tail = _we_raw_tail_fields(row)
    key_map = {
        "weather_code": "weather_code_tail",
        "turf_condition": "turf_condition_tail",
        "dirt_condition": "dirt_condition_tail",
    }
    tail_v = str(tail.get(key_map.get(col, ""), "")).strip()
    # raw_hex 末尾側が取れる場合はそちらを優先（0 も有効値として扱う）
    if tail_v != "":
        return tail_v

    primary = str(row.get(col, "")).strip()
    return primary


def _we_change_id(row: dict) -> str:
    cid = str(row.get("change_id", "")).strip()
    if cid:
        return cid
    return str(_we_raw_tail_fields(row).get("change_id", "")).strip()


def _we_announce_key(row: dict) -> int:
    t = str(row.get("announce_ddhhmmss", "")).strip()
    if not t:
        t = str(_we_raw_tail_fields(row).get("announce_ddhhmmss", "")).strip()
    if not t or not t.isdigit():
        return -1
    return int(t)


def _we_pick_priority(row: dict) -> tuple[int, int, int, int]:
    """
    WE 採用優先度（大きいほど優先）
    1) 変更識別 1/3 を優先（2 は劣後）
    2) 天候・馬場の有効値数
    3) 発表時刻（announce_ddhhmmss）
    4) 同時刻同内容なら change_id=3 を優先
    """
    cid = _we_change_id(row)
    cid_ok = 1 if cid in {"1", "3"} else 0
    useful = _we_merge_usefulness_score(row)
    announce = _we_announce_key(row)
    cid_pref = 1 if cid == "3" else 0
    return (cid_ok, useful, announce, cid_pref)


def _we_choose_better(prev: dict | None, cand: dict) -> dict:
    if prev is None:
        return cand
    return cand if _we_pick_priority(cand) >= _we_pick_priority(prev) else prev


def _parse_race_num(value: str | int | None) -> int:
    s = str(value or "").strip()
    if not s:
        return -1
    try:
        return int(s)
    except (TypeError, ValueError):
        return -1


def _we_select_day_row_for_race(
    day_rows: list[dict], target_race_num: str | int | None
) -> dict | None:
    """
    同一開催(day_key)の WE から、対象レースに適用すべき1行を選ぶ。

    ルール:
    - race_num=00 は基準値として常に候補
    - race_num>00 は対象レース番号以下のみ候補（前方補完）
    - 候補内で race_num が大きいもの（より直近の更新）を優先
    - 同一 race_num では発表時刻と _we_pick_priority で比較
    """
    if not day_rows:
        return None

    target = _parse_race_num(target_race_num)
    candidates: list[dict] = []
    for row in day_rows:
        rn = _parse_race_num(row.get("race_num"))
        if rn < 0:
            continue
        if rn > 0 and target >= 0 and rn > target:
            continue
        if _we_merge_usefulness_score(row) <= 0:
            continue
        candidates.append(row)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda r: (
            _parse_race_num(r.get("race_num")),
            _we_announce_key(r),
            _we_pick_priority(r),
        ),
    )


def purge_realtime_csv_rows_outside_date_range(
    filepath: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> tuple[int, int]:
    """
    開催日(year+month_day)が [start_yyyymmdd, end_yyyymmdd] に **含まれない** 行を削除する。

    realtime_* CSV は「今回の取得ウィンドウ以外の過去日」を溜めない運用向け。
    year/month_day が空の行は日付判定できないためそのまま残す。
    """
    if not os.path.isfile(filepath) or os.path.getsize(filepath) <= 0:
        return 0, 0
    try:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields:
                return 0, 0
            kept: list[dict] = []
            removed = 0
            for row in reader:
                y = str(row.get("year", "")).strip()
                md = str(row.get("month_day", "")).strip()
                if not y or not md:
                    kept.append(row)
                    continue
                d = y.zfill(4) + md.zfill(4)
                if start_yyyymmdd <= d <= end_yyyymmdd:
                    kept.append(row)
                else:
                    removed += 1
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        return removed, len(kept)
    except Exception as e:
        print(f"  Warning: purge (outside range) failed for {filepath}: {e}")
        return 0, 0


def purge_realtime_csv_rows_in_date_range(
    filepath: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> tuple[int, int]:
    """
    開催日(year+month_day)が [start_yyyymmdd, end_yyyymmdd] に入る行を CSV から削除する。

    速報開催情報の「取り消し」対応: JV-Link では取り消し後は当該レコードが配信されないため、
    取得のたびに当該期間の行を捨ててから今回の取得結果だけを追記すれば、
    「取得時点で含まれる情報のみ」を使う運用に近づけられる。
    """
    if not os.path.isfile(filepath) or os.path.getsize(filepath) <= 0:
        return 0, 0
    try:
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields:
                return 0, 0
            kept: list[dict] = []
            removed = 0
            for row in reader:
                if _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
                    removed += 1
                    continue
                kept.append(row)
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        return removed, len(kept)
    except Exception as e:
        print(f"  Warning: purge failed for {filepath}: {e}")
        return 0, 0


def overlay_kubun7_to_main_race(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    source_output_dir: str | None = None,
):
    """
    common/data/output の kubun=7 を main/data/race に上書き反映する。
    - SE: horse_weight, weight_change_sign, weight_change
    - RA: weather_code, turf_condition, dirt_condition,
          running_count, finish_count（kubun=7 に値があるとき）
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    source_dir = (
        Path(source_output_dir)
        if source_output_dir
        else (project_root / "common" / "data" / "output")
    )

    main_race_dir.mkdir(parents=True, exist_ok=True)

    ra_main_path = main_race_dir / "race_ra.csv"
    se_main_path = main_race_dir / "race_se.csv"

    if not ra_main_path.exists() or not se_main_path.exists():
        print("main/data/race の RA/SE が見つからないため overlay をスキップします。")
        return {"ra_updated": 0, "se_updated": 0, "ra_target": 0, "se_target": 0}

    # main 側を読み込み
    with open(ra_main_path, "r", encoding="utf-8-sig", newline="") as f:
        ra_reader = csv.DictReader(f)
        ra_fields = ra_reader.fieldnames or []
        ra_rows = list(ra_reader)

    with open(se_main_path, "r", encoding="utf-8-sig", newline="") as f:
        se_reader = csv.DictReader(f)
        se_fields = se_reader.fieldnames or []
        se_rows = list(se_reader)

    ra_map = {_key_ra(r): r for r in ra_rows}
    se_map = {_key_se(r): r for r in se_rows}

    ra_target = len(ra_map)
    se_target = len(se_map)

    # kubun=7 の候補を読み込み（年分割ファイル）
    ra_src_dir = source_dir / "race_ra"
    se_src_dir = source_dir / "race_se"
    ra_src_files = sorted(ra_src_dir.glob("race_ra_*.csv"))
    se_src_files = sorted(se_src_dir.glob("race_se_*.csv"))

    ra_update_fields = (
        "weather_code",
        "turf_condition",
        "dirt_condition",
        "running_count",
        "finish_count",
    )
    se_update_fields = ("horse_weight", "weight_change_sign", "weight_change")

    ra_updated = 0
    se_updated = 0

    # RA overlay
    for src in ra_src_files:
        try:
            with open(src, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("data_kubun", "")).strip() != "7":
                        continue
                    if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
                        continue
                    k = _key_ra(row)
                    if k not in ra_map:
                        continue
                    target_row = ra_map[k]
                    changed = False
                    for col in ra_update_fields:
                        new_val = str(row.get(col, "")).strip()
                        if not new_val:
                            continue
                        if col in ("running_count", "finish_count") and all(
                            ch == "0" for ch in new_val
                        ):
                            continue
                        if target_row.get(col, "") != new_val:
                            target_row[col] = new_val
                            changed = True
                    if changed:
                        ra_updated += 1
        except Exception as e:
            print(f"Warning: failed to overlay RA from {src}: {e}")

    # SE overlay
    for src in se_src_files:
        try:
            with open(src, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("data_kubun", "")).strip() != "7":
                        continue
                    if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
                        continue
                    k = _key_se(row)
                    if k not in se_map:
                        continue
                    target_row = se_map[k]
                    changed = False
                    for col in se_update_fields:
                        new_val = str(row.get(col, ""))
                        if new_val and target_row.get(col, "") != new_val:
                            target_row[col] = new_val
                            changed = True
                    if changed:
                        se_updated += 1
        except Exception as e:
            print(f"Warning: failed to overlay SE from {src}: {e}")

    # 保存
    with open(ra_main_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ra_fields)
        writer.writeheader()
        writer.writerows(ra_rows)

    with open(se_main_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=se_fields)
        writer.writeheader()
        writer.writerows(se_rows)

    print(
        f"overlay_kubun7_to_main_race: RA {ra_updated}/{ra_target}, "
        f"SE {se_updated}/{se_target} updated."
    )
    return {
        "ra_updated": ra_updated,
        "se_updated": se_updated,
        "ra_target": ra_target,
        "se_target": se_target,
    }


def _fetch_realtime_record_only(
    *,
    dataspec: str,
    target_rec_id: str,
    start_date_str: str,
    end_date_str: str,
    output_subdir: str,
    output_filename: str,
    option: int = 1,
):
    """
    速報系 dataspec から特定レコードのみ取得して CSV 保存する共通処理。
    """
    script_dir = Path(__file__).parent.parent.parent
    output_root = script_dir / "data" / "output" / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_filename

    start_y = start_date_str[:8]
    end_y = end_date_str[:8]
    dropped_past, _ = purge_realtime_csv_rows_outside_date_range(
        str(output_path), start_y, end_y
    )
    if dropped_past:
        print(
            f"  Dropped {dropped_past} past {target_rec_id} rows "
            f"(outside {start_y}..{end_y}) from {output_path.name}"
        )
    removed, _k = purge_realtime_csv_rows_in_date_range(str(output_path), start_y, end_y)
    if removed:
        print(
            f"  Purged {removed} {target_rec_id} rows in snapshot range "
            f"({output_path.name})"
        )

    existing_keys = load_existing_dates(str(output_path), target_rec_id)
    records = []
    total_count = 0
    skipped_count = 0

    client = JRAVANClient()
    try:
        client.login()
        pbar = tqdm(
            desc=f"Fetching {dataspec} ({target_rec_id} only)",
            unit="chunks",
            position=0,
            leave=True,
        )
        try:
            for raw_chunk in client.get_data(
                dataspec, start_date_str, option=option, end_date=end_date_str
            ):
                pbar.update(1)
                if not raw_chunk:
                    continue

                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue

                for line in raw_chunk.split("\n"):
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != target_rec_id:
                        continue

                    if rec_id not in SCHEMAS:
                        continue

                    parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                    # 日付範囲フィルタ（キーがある場合のみ）
                    year = str(parsed.get("year", "")).strip()
                    month_day = str(parsed.get("month_day", "")).strip()
                    if year and month_day:
                        d = year.zfill(4) + month_day.zfill(4)
                        if d < start_date_str[:8] or d > end_date_str[:8]:
                            skipped_count += 1
                            continue

                    record_key = _extract_record_key(parsed, rec_id)
                    if record_key is not None and record_key in existing_keys:
                        skipped_count += 1
                        continue

                    parsed["raw_hex"] = line_bytes.hex()
                    if rec_id == "WE":
                        parsed.update(_we_raw_tail_fields(parsed))
                    records.append(parsed)
                    if record_key is not None:
                        existing_keys.add(record_key)
                    total_count += 1
        finally:
            pbar.close()

    finally:
        if client:
            client.close()

    if records:
        if target_rec_id == "WE":
            fields = _we_fieldnames_with_extras()
            save_to_csv(records, str(output_path), fields, append=False)
        else:
            fields = get_schema_fieldnames(target_rec_id) + ["raw_hex"]
            save_to_csv(records, str(output_path), fields)
    else:
        print(f"No relevant records found for {dataspec} ({target_rec_id}).")

    return {
        "path": str(output_path),
        "added": total_count,
        "skipped": skipped_count,
    }


def _fetch_realtime_bundle_from_0b14(start_date_str: str, end_date_str: str):
    """
    速報開催情報一括(0B14)を JVRTOpen で日単位取得し、
    WE/AV/JC/TC/CC を保存する。
    """
    script_dir = Path(__file__).parent.parent.parent
    output_root = script_dir / "data" / "output"
    we_path = output_root / "realtime_we" / "we.csv"
    av_path = output_root / "realtime_av" / "av.csv"
    jc_path = output_root / "realtime_jc" / "jc.csv"
    tc_path = output_root / "realtime_tc" / "tc.csv"
    cc_path = output_root / "realtime_cc" / "cc.csv"
    we_path.parent.mkdir(parents=True, exist_ok=True)
    av_path.parent.mkdir(parents=True, exist_ok=True)
    jc_path.parent.mkdir(parents=True, exist_ok=True)
    tc_path.parent.mkdir(parents=True, exist_ok=True)
    cc_path.parent.mkdir(parents=True, exist_ok=True)

    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]
    for path, label in (
        (we_path, "WE"),
        (av_path, "AV"),
        (jc_path, "JC"),
        (tc_path, "TC"),
        (cc_path, "CC"),
    ):
        dropped_past, _ = purge_realtime_csv_rows_outside_date_range(
            str(path), start_yyyymmdd, end_yyyymmdd
        )
        if dropped_past:
            print(
                f"  Dropped {dropped_past} past {label} rows "
                f"(outside {start_yyyymmdd}..{end_yyyymmdd})"
            )
        removed, kept = purge_realtime_csv_rows_in_date_range(
            str(path), start_yyyymmdd, end_yyyymmdd
        )
        if removed:
            print(
                f"  Purged {removed} {label} rows in {start_yyyymmdd}..{end_yyyymmdd} "
                f"(snapshot refresh; {kept} rows remain in file after purge)"
            )

    existing_we = load_existing_dates(str(we_path), "WE")
    existing_av = load_existing_dates(str(av_path), "AV")
    existing_jc = load_existing_dates(str(jc_path), "JC")
    existing_tc = load_existing_dates(str(tc_path), "TC")
    existing_cc = load_existing_dates(str(cc_path), "CC")
    we_records = []
    av_records = []
    jc_records = []
    tc_records = []
    cc_records = []

    start_day = datetime.strptime(start_date_str[:8], "%Y%m%d").date()
    end_day = datetime.strptime(end_date_str[:8], "%Y%m%d").date()
    if end_day < start_day:
        start_day, end_day = end_day, start_day

    client = None
    add_we = 0
    add_av = 0
    add_jc = 0
    add_tc = 0
    add_cc = 0
    skip_we = 0
    skip_av = 0
    skip_jc = 0
    skip_tc = 0
    skip_cc = 0

    try:
        client = JRAVANClient()
        client.login()

        current = start_day
        while current <= end_day:
            day_key = current.strftime("%Y%m%d")
            print(f"Requesting realtime bundle 0B14 (key={day_key})...")
            try:
                rt_ret = client.jv_link.JVRTOpen("0B14", day_key)
                rc = _jv_com_return_code(rt_ret)
            except Exception as e:
                print(f"JVRTOpen exception (0B14, {day_key}): {e}")
                current += timedelta(days=1)
                continue

            if rc < 0:
                print(f"JVRTOpen(0B14, {day_key}) skipped: ret={rc} (raw={rt_ret!r})")
                current += timedelta(days=1)
                continue

            read_loops = 0
            while True:
                read_loops += 1
                rr = client.jv_link.JVRead("", 1000000, "")
                if isinstance(rr, tuple):
                    status = int(rr[0])
                    raw_data = rr[1] if len(rr) > 1 else ""
                else:
                    status = int(rr)
                    raw_data = ""

                if status == 0:
                    break
                if status == -1:
                    continue
                if status == -3:
                    time.sleep(0.2)
                    continue
                if status == -402:
                    break
                if status < -1:
                    print(f"JVRead error in 0B14 loop: {status}")
                    break

                if isinstance(raw_data, bytes):
                    try:
                        raw_data = raw_data.decode("cp932", "replace")
                    except Exception:
                        raw_data = ""
                if not isinstance(raw_data, str) or not raw_data:
                    continue

                for line in raw_data.split("\n"):
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id not in {"WE", "AV", "JC", "TC", "CC"}:
                        continue
                    if rec_id not in SCHEMAS:
                        continue

                    parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])
                    parsed["raw_hex"] = line_bytes.hex()
                    if rec_id == "WE":
                        parsed.update(_we_raw_tail_fields(parsed))
                    key = _extract_record_key(parsed, rec_id)

                    if rec_id == "WE":
                        if key is not None and key in existing_we:
                            skip_we += 1
                            continue
                        we_records.append(parsed)
                        if key is not None:
                            existing_we.add(key)
                        add_we += 1
                    elif rec_id == "AV":
                        if key is not None and key in existing_av:
                            skip_av += 1
                            continue
                        av_records.append(parsed)
                        if key is not None:
                            existing_av.add(key)
                        add_av += 1
                    elif rec_id == "JC":
                        if key is not None and key in existing_jc:
                            skip_jc += 1
                            continue
                        jc_records.append(parsed)
                        if key is not None:
                            existing_jc.add(key)
                        add_jc += 1
                    elif rec_id == "TC":
                        if key is not None and key in existing_tc:
                            skip_tc += 1
                            continue
                        tc_records.append(parsed)
                        if key is not None:
                            existing_tc.add(key)
                        add_tc += 1
                    else:
                        if key is not None and key in existing_cc:
                            skip_cc += 1
                            continue
                        cc_records.append(parsed)
                        if key is not None:
                            existing_cc.add(key)
                        add_cc += 1

            current += timedelta(days=1)

    finally:
        if client:
            client.close()

    if we_records:
        fields_we = _we_fieldnames_with_extras()
        save_to_csv(we_records, str(we_path), fields_we, append=False)
    else:
        print("No new WE records from 0B14.")

    if av_records:
        fields_av = get_schema_fieldnames("AV") + ["raw_hex"]
        save_to_csv(av_records, str(av_path), fields_av)
    else:
        print("No new AV records from 0B14.")

    if jc_records:
        fields_jc = get_schema_fieldnames("JC") + ["raw_hex"]
        save_to_csv(jc_records, str(jc_path), fields_jc)
    else:
        print("No new JC records from 0B14.")

    if tc_records:
        fields_tc = get_schema_fieldnames("TC") + ["raw_hex"]
        save_to_csv(tc_records, str(tc_path), fields_tc)
    else:
        print("No new TC records from 0B14.")

    if cc_records:
        fields_cc = get_schema_fieldnames("CC") + ["raw_hex"]
        save_to_csv(cc_records, str(cc_path), fields_cc)
    else:
        print("No new CC records from 0B14.")

    return {
        "we": {"path": str(we_path), "added": add_we, "skipped": skip_we},
        "av": {"path": str(av_path), "added": add_av, "skipped": skip_av},
        "jc": {"path": str(jc_path), "added": add_jc, "skipped": skip_jc},
        "tc": {"path": str(tc_path), "added": add_tc, "skipped": skip_tc},
        "cc": {"path": str(cc_path), "added": add_cc, "skipped": skip_cc},
    }


def _fetch_realtime_daily_jvrt(
    *,
    dataspec: str,
    target_rec_id: str,
    start_date_str: str,
    end_date_str: str,
    output_subdir: str,
    output_filename: str,
):
    """
    JVRTOpen(dataspec, YYYYMMDD) を日単位で呼び出し、
    指定レコードのみ抽出してCSV保存する。
    """
    script_dir = Path(__file__).parent.parent.parent
    output_root = script_dir / "data" / "output" / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_filename

    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]
    dropped_past, _ = purge_realtime_csv_rows_outside_date_range(
        str(output_path), start_yyyymmdd, end_yyyymmdd
    )
    if dropped_past:
        print(
            f"  Dropped {dropped_past} past {target_rec_id} rows "
            f"(outside {start_yyyymmdd}..{end_yyyymmdd}) from {output_path.name}"
        )
    removed, _kept = purge_realtime_csv_rows_in_date_range(
        str(output_path), start_yyyymmdd, end_yyyymmdd
    )
    if removed:
        print(
            f"  Purged {removed} {target_rec_id} rows in {start_yyyymmdd}..{end_yyyymmdd} "
            f"from {output_path.name} (snapshot refresh for cancellations)"
        )

    existing_keys = load_existing_dates(str(output_path), target_rec_id)
    records = []
    total_count = 0
    skipped_count = 0

    start_day = datetime.strptime(start_date_str[:8], "%Y%m%d").date()
    end_day = datetime.strptime(end_date_str[:8], "%Y%m%d").date()
    if end_day < start_day:
        start_day, end_day = end_day, start_day

    client = None
    try:
        client = JRAVANClient()
        client.login()
        current = start_day
        while current <= end_day:
            day_key = current.strftime("%Y%m%d")
            print(f"Requesting {dataspec} (key={day_key})...")
            try:
                rt_ret = client.jv_link.JVRTOpen(dataspec, day_key)
                rc = _jv_com_return_code(rt_ret)
            except Exception as e:
                print(f"JVRTOpen exception ({dataspec}, {day_key}): {e}")
                current += timedelta(days=1)
                continue
            if rc < 0:
                print(
                    f"JVRTOpen({dataspec}, {day_key}) skipped: ret={rc} (raw={rt_ret!r})"
                )
                current += timedelta(days=1)
                continue

            while True:
                rr = client.jv_link.JVRead("", 1000000, "")
                if isinstance(rr, tuple):
                    status = int(rr[0])
                    raw_data = rr[1] if len(rr) > 1 else ""
                else:
                    status = int(rr)
                    raw_data = ""

                if status == 0:
                    break
                if status == -1:
                    continue
                if status == -3:
                    time.sleep(0.2)
                    continue
                if status == -402:
                    break
                if status < -1:
                    print(f"JVRead error in {dataspec} loop: {status}")
                    break

                if isinstance(raw_data, bytes):
                    try:
                        raw_data = raw_data.decode("cp932", "replace")
                    except Exception:
                        raw_data = ""
                if not isinstance(raw_data, str) or not raw_data:
                    continue

                for line in raw_data.split("\n"):
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue
                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != target_rec_id:
                        continue
                    if rec_id not in SCHEMAS:
                        continue
                    parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])
                    parsed["raw_hex"] = line_bytes.hex()
                    if rec_id == "WE":
                        parsed.update(_we_raw_tail_fields(parsed))
                    key = _extract_record_key(parsed, rec_id)
                    if key is not None and key in existing_keys:
                        skipped_count += 1
                        continue
                    records.append(parsed)
                    if key is not None:
                        existing_keys.add(key)
                    total_count += 1
            current += timedelta(days=1)
    finally:
        if client:
            client.close()

    if records:
        if target_rec_id == "WE":
            fields = _we_fieldnames_with_extras()
            save_to_csv(records, str(output_path), fields, append=False)
        else:
            fields = get_schema_fieldnames(target_rec_id) + ["raw_hex"]
            save_to_csv(records, str(output_path), fields, append=False)
    else:
        print(f"No new {target_rec_id} records from {dataspec}.")

    return {"path": str(output_path), "added": total_count, "skipped": skipped_count}


def fetch_wh_from_0b11(start_date_str="20240101000000", end_date_str="20241231235959"):
    """速報馬体重（0B11 / WH）を JVRTOpen(0B11, YYYYMMDD) の日単位で取得する。

    レース単位16桁キーでの取得は common/data/scripts/probe_wh_by_race_id.py を参照。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    print("=== WHデータ取得 (0B11) ===")
    return _fetch_realtime_daily_jvrt(
        dataspec="0B11",
        target_rec_id="WH",
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_subdir="realtime_wh",
        output_filename="wh.csv",
    )


def fetch_dm_from_0b13(start_date_str="20240101000000", end_date_str="20241231235959"):
    """速報タイム型データマイニング予想（0B13 / DM）を取得する。"""
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    print("=== DMデータ取得 (0B13) ===")
    return _fetch_realtime_daily_jvrt(
        dataspec="0B13",
        target_rec_id="DM",
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_subdir="realtime_dm",
        output_filename="dm.csv",
    )


def fetch_tm_from_0b17(start_date_str="20240101000000", end_date_str="20241231235959"):
    """速報対戦型データマイニング予想（0B17 / TM）を取得する。"""
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    print("=== TMデータ取得 (0B17) ===")
    return _fetch_realtime_daily_jvrt(
        dataspec="0B17",
        target_rec_id="TM",
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_subdir="realtime_tm",
        output_filename="tm.csv",
    )


def fetch_we_only(start_date_str="20240101000000", end_date_str="20241231235959"):
    """速報天候・馬場（WE）を取得する。

    優先順:
    1) 0V13（従来の速報系）
    2) 0B14（速報開催情報一括, JVRTOpen）をフォールバック
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    today_start = datetime.now().strftime("%Y%m%d") + "000000"
    if start_date_str[:8] < today_start[:8]:
        print(
            f"WEは速報系(0V13)のため開始日を当日に補正: "
            f"{start_date_str} -> {today_start}"
        )
        start_date_str = today_start
    if end_date_str < start_date_str:
        end_date_str = datetime.now().strftime("%Y%m%d") + "235959"
    print("=== WEデータ取得: primary 0V13 ===")
    res_0v13 = _fetch_realtime_record_only(
        dataspec="0V13",
        target_rec_id="WE",
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_subdir="realtime_we",
        output_filename="we.csv",
        option=2,
    )
    # 0V13 が契約外/メンテ等で空振りする環境向け。
    # 仕様書の開催日単位キー（YYYYMMDD）で 0B14 一括から WE を再取得する。
    res_0b14 = {"path": res_0v13.get("path", ""), "added": 0, "skipped": 0}
    if int(res_0v13.get("added", 0) or 0) <= 0:
        print("=== WEデータ取得: fallback 0B14 (JVRTOpen) ===")
        res_0b14 = _fetch_realtime_daily_jvrt(
            dataspec="0B14",
            target_rec_id="WE",
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            output_subdir="realtime_we",
            output_filename="we.csv",
        )

    return {
        "path": res_0b14.get("path") or res_0v13.get("path", ""),
        "added": int(res_0v13.get("added", 0) or 0)
        + int(res_0b14.get("added", 0) or 0),
        "skipped": int(res_0v13.get("skipped", 0) or 0)
        + int(res_0b14.get("skipped", 0) or 0),
        "primary_0v13": res_0v13,
        "fallback_0b14": res_0b14,
    }


def fetch_wh_only(start_date_str="20240101000000", end_date_str="20241231235959"):
    """速報馬体重（0V12 / WH）を取得する。"""
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    today_start = datetime.now().strftime("%Y%m%d") + "000000"
    if start_date_str[:8] < today_start[:8]:
        print(
            f"WHは速報系(0V12)のため開始日を当日に補正: "
            f"{start_date_str} -> {today_start}"
        )
        start_date_str = today_start
    if end_date_str < start_date_str:
        end_date_str = datetime.now().strftime("%Y%m%d") + "235959"
    print("=== WHデータのみ取得 (0V12) ===")
    return _fetch_realtime_record_only(
        dataspec="0V12",
        target_rec_id="WH",
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_subdir="realtime_wh",
        output_filename="wh.csv",
        option=2,
    )


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


def merge_jc_to_main_se(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    source_output_dir: str | None = None,
):
    """
    realtime_jc/jc.csv を race_se の jockey_code に反映（取得スナップショット前提）。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    source_dir = (
        Path(source_output_dir)
        if source_output_dir
        else (project_root / "common" / "data" / "output")
    )
    se_path = main_race_dir / "race_se.csv"
    jc_path = source_dir / "realtime_jc" / "jc.csv"
    if not se_path.exists():
        print("merge_jc_to_main_se: race_se.csv なし")
        return {"updated": 0, "total": 0}
    if not jc_path.exists():
        print("merge_jc_to_main_se: jc.csv なし")
        return {"updated": 0, "total": 0}

    with open(se_path, "r", encoding="utf-8-sig", newline="") as f:
        se_reader = csv.DictReader(f)
        se_fields = se_reader.fieldnames or []
        se_rows = list(se_reader)

    jc_by_key: dict[tuple, str] = {}
    with open(jc_path, "r", encoding="utf-8-sig", newline="") as f:
        for jc in csv.DictReader(f):
            if not _in_date_range(jc, start_yyyymmdd, end_yyyymmdd):
                continue
            k = _key_se(jc)
            jock = _jockey_code_from_jc_raw_hex(str(jc.get("raw_hex", "")))
            if jock:
                jc_by_key[k] = jock

    updated = 0
    for row in se_rows:
        if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
            continue
        jock = jc_by_key.get(_key_se(row))
        if not jock:
            continue
        cur = str(row.get("jockey_code", "")).strip()
        if cur != jock:
            row["jockey_code"] = jock
            updated += 1

    with open(se_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=se_fields)
        writer.writeheader()
        writer.writerows(se_rows)

    print(f"merge_jc_to_main_se: {updated} rows updated.")
    return {"updated": updated, "total": len(se_rows)}


def _backfill_mining_predicted_rank_from_time(
    se_rows: list[dict],
    *,
    start_yyyymmdd: str | None = None,
    end_yyyymmdd: str | None = None,
) -> int:
    """mining_predicted_time のみ入っている行に、レース内タイム昇順で順位を補完する。"""
    from collections import defaultdict

    by_race: dict[tuple, list[dict]] = defaultdict(list)
    for row in se_rows:
        if (
            start_yyyymmdd
            and end_yyyymmdd
            and not _in_date_range(row, start_yyyymmdd, end_yyyymmdd)
        ):
            continue
        by_race[_key_ra(row)].append(row)

    updated = 0
    for race_rows in by_race.values():
        candidates: list[tuple[int, dict]] = []
        for row in race_rows:
            rank = str(row.get("mining_predicted_rank", "")).strip()
            time_str = str(row.get("mining_predicted_time", "")).strip()
            if rank not in ("", "00", "0") or not time_str or time_str in ("00000", "0"):
                continue
            try:
                time_val = int(time_str)
            except ValueError:
                continue
            if time_val <= 0:
                continue
            candidates.append((time_val, row))

        candidates.sort(key=lambda item: item[0])
        for rank_i, (_, row) in enumerate(candidates, start=1):
            new_rank = str(rank_i).zfill(2)
            if str(row.get("mining_predicted_rank", "")).strip() != new_rank:
                row["mining_predicted_rank"] = new_rank
                updated += 1
    return updated


def merge_dm_mining_to_main_se(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    source_output_dir: str | None = None,
):
    """
    0B13 の DM（realtime_dm/dm.csv）を race_se のマイニング列へ反映する。
    SE の mining_predicted_time / mining_error_plus / mining_error_minus /
    mining_predicted_rank / mining_kubun を更新。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    source_dir = (
        Path(source_output_dir)
        if source_output_dir
        else (project_root / "common" / "data" / "output")
    )
    se_path = main_race_dir / "race_se.csv"
    dm_path = source_dir / "realtime_dm" / "dm.csv"
    if not se_path.exists():
        print("merge_dm_mining_to_main_se: race_se.csv なし")
        return {"updated": 0, "total": 0}
    if not dm_path.exists():
        print("merge_dm_mining_to_main_se: dm.csv なし")
        return {"updated": 0, "total": 0}

    with open(se_path, "r", encoding="utf-8-sig", newline="") as f:
        se_reader = csv.DictReader(f)
        se_fields = se_reader.fieldnames or []
        se_rows = list(se_reader)

    dm_by_race: dict[tuple, dict] = {}
    with open(dm_path, "r", encoding="utf-8-sig", newline="") as f:
        for dm in csv.DictReader(f):
            if not _in_date_range(dm, start_yyyymmdd, end_yyyymmdd):
                continue
            dm_norm = _normalize_dm_row(dm)
            dm_by_race[_key_ra(dm_norm)] = dm_norm

    updated = 0
    for row in se_rows:
        if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
            continue
        dm = dm_by_race.get(_key_ra(row))
        if not dm:
            continue
        hn = str(row.get("horse_num", "")).strip().zfill(2)
        if hn in ("", "00"):
            continue
        for i in range(1, 19):
            dhn = str(dm.get(f"mining_pred_{i}_horse_num", "")).strip().zfill(2)
            if dhn != hn or dhn == "00":
                continue
            t = str(dm.get(f"mining_pred_{i}_time", "")).strip()
            ep = str(dm.get(f"mining_pred_{i}_error+", "")).strip()
            em = str(dm.get(f"mining_pred_{i}_error-", "")).strip()
            rank = str(i).zfill(2)
            mk = str(dm.get("data_kubun", "")).strip()
            changed = False
            if t and str(row.get("mining_predicted_time", "")).strip() != t:
                row["mining_predicted_time"] = t
                changed = True
            if ep and str(row.get("mining_error_plus", "")).strip() != ep:
                row["mining_error_plus"] = ep
                changed = True
            if em and str(row.get("mining_error_minus", "")).strip() != em:
                row["mining_error_minus"] = em
                changed = True
            if rank and str(row.get("mining_predicted_rank", "")).strip() != rank:
                row["mining_predicted_rank"] = rank
                changed = True
            if mk and str(row.get("mining_kubun", "")).strip() != mk:
                row["mining_kubun"] = mk
                changed = True
            if changed:
                updated += 1
            break

    backfill_updated = _backfill_mining_predicted_rank_from_time(
        se_rows,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
    )

    with open(se_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=se_fields)
        writer.writeheader()
        writer.writerows(se_rows)

    print(
        f"merge_dm_mining_to_main_se: {updated} rows updated"
        f" (rank backfill: {backfill_updated})."
    )
    return {"updated": updated, "rank_backfill": backfill_updated, "total": len(se_rows)}


def merge_realtime_to_main_race(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
    source_output_dir: str | None = None,
):
    """
    速報 WE/WH を main/data/race にマージする。

    RA への反映（レースキー一致、日付範囲内）:
    - realtime_we/we.csv の weather_code, turf_condition, dirt_condition を上書き
      （出馬表だけでは空になりやすい項目の想定）。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    source_dir = (
        Path(source_output_dir)
        if source_output_dir
        else (project_root / "common" / "data" / "output")
    )

    ra_main_path = main_race_dir / "race_ra.csv"
    se_main_path = main_race_dir / "race_se.csv"
    we_path = source_dir / "realtime_we" / "we.csv"
    wh_path = source_dir / "realtime_wh" / "wh.csv"

    if not ra_main_path.exists() or not se_main_path.exists():
        print(
            "main/data/race の RA/SE が見つからないため realtime merge をスキップします。"
        )
        return {"ra_updated": 0, "se_updated": 0}

    with open(ra_main_path, "r", encoding="utf-8-sig", newline="") as f:
        ra_reader = csv.DictReader(f)
        ra_fields = ra_reader.fieldnames or []
        ra_rows = list(ra_reader)
    with open(se_main_path, "r", encoding="utf-8-sig", newline="") as f:
        se_reader = csv.DictReader(f)
        se_fields = se_reader.fieldnames or []
        se_rows = list(se_reader)

    # WE 読み込み（レース単位 + 開催日単位）
    we_by_race = {}
    we_by_day_rows = {}
    if we_path.exists():
        with open(we_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
                    continue
                race_key = _key_ra(row)
                day_key = race_key[:5]
                we_by_race[race_key] = _we_choose_better(we_by_race.get(race_key), row)
                we_by_day_rows.setdefault(day_key, []).append(row)

    # WH 読み込み（0B11 は1レース1行 → 馬単位に展開）
    wh_by_horse = {}
    if wh_path.exists():
        with open(wh_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
                    continue
                for horse_row in _expand_wh_realtime_row(row):
                    wh_by_horse[_key_se(horse_row)] = horse_row

    ra_updated = 0
    se_updated = 0

    # RA 更新
    for row in ra_rows:
        if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
            continue
        k = _key_ra(row)
        race_we = we_by_race.get(k)
        day_we = _we_select_day_row_for_race(
            we_by_day_rows.get(k[:5], []), row.get("race_num")
        )
        if race_we and day_we:
            we = _we_choose_better(day_we, race_we)
        else:
            we = race_we or day_we
        if not we:
            continue
        changed = False
        for col in ("weather_code", "turf_condition", "dirt_condition"):
            v = _we_effective_code(we, col)
            if not v or all(ch == "0" for ch in v):
                continue
            if row.get(col, "") != v:
                row[col] = v
                changed = True
        if changed:
            ra_updated += 1

    # SE 更新
    for row in se_rows:
        if not _in_date_range(row, start_yyyymmdd, end_yyyymmdd):
            continue
        wh = wh_by_horse.get(_key_se(row))
        if not wh:
            continue
        changed = False

        hw = str(wh.get("horse_weight", "")).strip()
        if hw and not all(ch == "0" for ch in hw) and row.get("horse_weight", "") != hw:
            row["horse_weight"] = hw
            changed = True

        sign = str(wh.get("weight_change_sign", "")).strip()
        if sign and row.get("weight_change_sign", "") != sign:
            row["weight_change_sign"] = sign
            changed = True

        diff = str(wh.get("weight_change", "")).strip()
        if diff and row.get("weight_change", "") != diff:
            row["weight_change"] = diff
            changed = True

        if changed:
            se_updated += 1

    with open(ra_main_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ra_fields)
        writer.writeheader()
        writer.writerows(ra_rows)
    with open(se_main_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=se_fields)
        writer.writeheader()
        writer.writerows(se_rows)

    print(f"merge_realtime_to_main_race: RA {ra_updated}, SE {se_updated} updated.")
    return {"ra_updated": ra_updated, "se_updated": se_updated}


def refresh_today_realtime_data(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
):
    """
    当日用ユーティリティ:
    1) 速報系 WE/WH を取得
    2) main/data/race にマージ反映
    3) 参考として kubun=7 overlay も実施
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    print(f"=== refresh_today_realtime_data: {start_date_str} -> {end_date_str} ===")

    # 0V 系は取得前に we.csv / wh.csv の当該期間を purge する。
    # 契約外(-111 等)で 0V が空振りすると、ここを **0B14/0B11 の後** に実行すると
    # 直前まで取れた WE/WH を消してしまうため、先に試してから JVRTOpen 系で埋め直す。
    we_res = fetch_we_only(start_date_str, end_date_str)
    wh_res = fetch_wh_only(start_date_str, end_date_str)

    # 0B14(JVRTOpen) 一括取得（WE/AV/JC/TC/CC）
    bundle_res = _fetch_realtime_bundle_from_0b14(start_date_str, end_date_str)

    # 0B11 で WH（速報馬体重）を取得
    wh_0b11_res = fetch_wh_from_0b11(start_date_str, end_date_str)

    # 速報データマイニング予想
    dm_res = fetch_dm_from_0b13(start_date_str, end_date_str)
    tm_res = fetch_tm_from_0b17(start_date_str, end_date_str)

    merge_res = merge_realtime_to_main_race(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
    )
    jc_merge_res = merge_jc_to_main_se(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
    )
    # fallback overlay（kubun=7がある場合）
    overlay_res = overlay_kubun7_to_main_race(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
    )
    # 0B13 DM は成績確定後の SE 内マイニングより優先（オーバーレイの後に上書き）
    dm_merge_res = merge_dm_mining_to_main_se(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
    )
    return {
        "bundle_0b14": bundle_res,
        "wh_0b11": wh_0b11_res,
        "we": we_res,
        "wh": wh_res,
        "dm_0b13": dm_res,
        "tm_0b17": tm_res,
        "merge": merge_res,
        "merge_jc": jc_merge_res,
        "merge_dm": dm_merge_res,
        "overlay": overlay_res,
    }


_O1_DEBUG_DUMPED = False  # 1回だけダンプするフラグ


def _parse_o1_win_odds(raw_line: str):
    """
    速報オッズ O1 レコードから単勝オッズ・人気順を抽出（簡易）。
    仕様書の定義に依存するため、環境差異がある場合は要調整。

    人気順: 1993年6月以前のデータは仕様書上の特殊扱いがある。当日・近年データでは通常 2 桁でよい。
    """
    if not raw_line or not raw_line.startswith("O1"):
        return []

    # raw_line を1回だけデバッグダンプ（フォーマット調査用）
    global _O1_DEBUG_DUMPED
    if not _O1_DEBUG_DUMPED:
        _O1_DEBUG_DUMPED = True
        try:
            import pathlib
            _dbg = pathlib.Path(__file__).parent.parent.parent.parent / "o1_raw_debug.txt"
            _dbg.write_text(repr(raw_line), encoding="utf-8")
        except Exception:
            pass

    # O1: ヘッダ(43byte)の後ろに「1頭あたり8byte × 最大18頭」の繰り返し
    # 馬番(2) + 単勝オッズ(4) + 単勝人気順(2) = 8byte
    # ※実測: section0=[0:171] (header43 + 16頭×8=128), section1/2はスペース区切りで複勝等
    start = 43
    block = 8
    out = []
    for i in range(28):
        s = start + i * block
        e = s + block
        if e > len(raw_line):
            break
        part = raw_line[s:e]
        horse_num = part[0:2]
        odds = part[2:6]
        popularity = part[6:8] if len(part) >= 8 else ""
        if not horse_num.isdigit():
            continue
        if not odds.isdigit():
            continue
        pop_raw = popularity.zfill(2) if popularity.isdigit() else ""
        out.append(
            {
                "horse_num": horse_num.zfill(2),
                "odds_raw": odds.zfill(4),
                "popularity_raw": pop_raw,
            }
        )
    return out


def _parse_odds_tenth(raw: str, *, width: int) -> tuple[float | None, str]:
    s = str(raw or "")
    if not s.strip():
        return None, "not_registered"
    if set(s) == {"-"}:
        return None, "cancel_before_sale"
    if set(s) == {"*"}:
        return None, "cancel_after_sale"
    if not s.isdigit():
        return None, "invalid"
    if int(s) == 0:
        return None, "no_vote"
    if width == 6 and s == "099999":
        return 9999.9, "over_limit"
    if width == 5 and s == "99999":
        return 9999.9, "over_limit"
    return int(s) / 10.0, "ok"


def _parse_pair_kumi(kumi: str) -> tuple[str, str, str]:
    s = str(kumi or "").strip()
    if len(s) != 4 or not s.isdigit():
        return "", "", ""
    h1 = s[:2]
    h2 = s[2:]
    if h1 == "00" or h2 == "00":
        return "", "", ""
    return h1, h2, f"{int(h1)}-{int(h2)}"


def _base_odds_record_fields(parsed: dict, race_id: str) -> dict:
    return {
        "year": str(parsed.get("year", "")).zfill(4),
        "month_day": str(parsed.get("month_day", "")).zfill(4),
        "course_code": str(parsed.get("course_code", "")).zfill(2),
        "kai": str(parsed.get("kai", "")).zfill(2),
        "nichi": str(parsed.get("nichi", "")).zfill(2),
        "race_num": str(parsed.get("race_num", "")).zfill(2),
        "race_id": race_id,
        "data_kubun": str(parsed.get("data_kubun", "")),
        "date_make": str(parsed.get("date_make", "")),
        "announce_datetime": str(parsed.get("announce_datetime", "")),
        "registered_count": str(parsed.get("registered_count", "")),
        "running_count": str(parsed.get("running_count", "")),
    }


def _parse_o2_quinella_odds(raw_line: str, race_id: str) -> list[dict]:
    """速報オッズ O2 から馬連オッズを1組み合わせ1行へ展開する。"""
    if not raw_line or not raw_line.startswith("O2") or "O2" not in SCHEMAS:
        return []
    parsed = parse_fixed_width(raw_line.encode("cp932", "replace"), SCHEMAS["O2"])
    base = _base_odds_record_fields(parsed, race_id)
    base["sale_flag"] = str(parsed.get("sale_flag_quinella", ""))
    base["vote_count"] = str(parsed.get("quinella_vote_count", ""))
    rows = []
    for i in range(1, 154):
        kumi = str(parsed.get(f"quinella_odds_{i:03d}_kumi", ""))
        h1, h2, ticket = _parse_pair_kumi(kumi)
        if not ticket:
            continue
        odds_raw = str(parsed.get(f"quinella_odds_{i:03d}_odds", ""))
        odds, status = _parse_odds_tenth(odds_raw, width=6)
        pop_raw = str(parsed.get(f"quinella_odds_{i:03d}_pop", ""))
        rows.append(
            {
                **base,
                "record_id": "O2",
                "ticket_type": "馬連",
                "ticket": ticket,
                "horse_num_1": h1,
                "horse_num_2": h2,
                "kumi": kumi,
                "odds_raw": odds_raw,
                "odds": odds,
                "odds_status": status,
                "popularity_raw": pop_raw,
            }
        )
    return rows


def _parse_o3_wide_odds(raw_line: str, race_id: str) -> list[dict]:
    """速報オッズ O3 からワイドオッズを1組み合わせ1行へ展開する。"""
    if not raw_line or not raw_line.startswith("O3") or "O3" not in SCHEMAS:
        return []
    parsed = parse_fixed_width(raw_line.encode("cp932", "replace"), SCHEMAS["O3"])
    base = _base_odds_record_fields(parsed, race_id)
    base["sale_flag"] = str(parsed.get("sale_flag_wide", ""))
    base["vote_count"] = str(parsed.get("wide_vote_count", ""))
    rows = []
    for i in range(1, 154):
        kumi = str(parsed.get(f"wide_odds_{i:03d}_kumi", ""))
        h1, h2, ticket = _parse_pair_kumi(kumi)
        if not ticket:
            continue
        min_raw = str(parsed.get(f"wide_odds_{i:03d}_min", ""))
        max_raw = str(parsed.get(f"wide_odds_{i:03d}_max", ""))
        min_odds, min_status = _parse_odds_tenth(min_raw, width=5)
        max_odds, max_status = _parse_odds_tenth(max_raw, width=5)
        odds_mid = (
            (float(min_odds) + float(max_odds)) / 2.0
            if min_odds is not None and max_odds is not None
            else None
        )
        pop_raw = str(parsed.get(f"wide_odds_{i:03d}_pop", ""))
        rows.append(
            {
                **base,
                "record_id": "O3",
                "ticket_type": "ワイド",
                "ticket": ticket,
                "horse_num_1": h1,
                "horse_num_2": h2,
                "kumi": kumi,
                "odds_min_raw": min_raw,
                "odds_max_raw": max_raw,
                "odds_min": min_odds,
                "odds_max": max_odds,
                "odds": odds_mid,
                "odds_status": (
                    min_status if min_status == max_status else f"{min_status}/{max_status}"
                ),
                "popularity_raw": pop_raw,
            }
        )
    return rows


QUINELLA_ODDS_FIELDS = [
    "year",
    "month_day",
    "course_code",
    "kai",
    "nichi",
    "race_num",
    "race_id",
    "record_id",
    "ticket_type",
    "ticket",
    "horse_num_1",
    "horse_num_2",
    "kumi",
    "odds_raw",
    "odds",
    "odds_status",
    "popularity_raw",
    "data_kubun",
    "date_make",
    "announce_datetime",
    "sale_flag",
    "vote_count",
    "registered_count",
    "running_count",
]

WIDE_ODDS_FIELDS = [
    "year",
    "month_day",
    "course_code",
    "kai",
    "nichi",
    "race_num",
    "race_id",
    "record_id",
    "ticket_type",
    "ticket",
    "horse_num_1",
    "horse_num_2",
    "kumi",
    "odds_min_raw",
    "odds_max_raw",
    "odds_min",
    "odds_max",
    "odds",
    "odds_status",
    "popularity_raw",
    "data_kubun",
    "date_make",
    "announce_datetime",
    "sale_flag",
    "vote_count",
    "registered_count",
    "running_count",
]


def _odds_output_dir() -> Path:
    return Path(__file__).parent.parent.parent / "data" / "output" / "odds"


def _quinella_odds_path(year: int | str) -> Path:
    return _odds_output_dir() / f"QuinellaOdds_{int(year)}.csv"


def _wide_odds_path(year: int | str) -> Path:
    return _odds_output_dir() / f"WideOdds_{int(year)}.csv"


def _race_id_from_race_row(row: dict) -> str:
    return (
        str(row.get("year", "")).zfill(4)
        + str(row.get("month_day", "")).zfill(4)
        + str(row.get("course_code", "")).zfill(2)
        + str(row.get("kai", "")).zfill(2)
        + str(row.get("nichi", "")).zfill(2)
        + str(row.get("race_num", "")).zfill(2)
    )


# ─── 確定単勝オッズ・単勝人気順（WinOdds） ──────────────────────────────────
#
# 注記: JV-Link の速報オッズ(O1)ストリームを新たに取得するのではなく、
# SE_SCHEMA（jv_schemas.py）に既に定義されている "odds"（確定単勝オッズ,
# offset 360 len4）・"popularity"（確定単勝人気順, offset 364 len2）フィールドを
# 使う。これらは RACE ストリームの SE レコードそのものに含まれており、
# 既存の fetch_related_data_from_se 等のパイプラインで
# common/data/output/race_se/race_se_YYYY.csv へ既に保存済みである。
# そのため fetch_win_odds_yearly() は JV-Link に一切接続しない
# （ネットワークI/Oなし、ローカル race_se_YYYY.csv -> WinOdds_YYYY.csv 変換のみ）。

WIN_ODDS_FIELDS = [
    "year",
    "month_day",
    "course_code",
    "kai",
    "nichi",
    "race_num",
    "race_id",
    "horse_num",
    "odds_raw",
    "odds",
    "odds_status",
    "popularity_raw",
    "popularity",
    "data_kubun",
    "announce_datetime",
]


def _win_odds_path(year: int | str) -> Path:
    return _odds_output_dir() / f"WinOdds_{int(year)}.csv"


def _se_csv_dir() -> Path:
    return Path(__file__).parent.parent.parent / "data" / "output" / "race_se"


def _parse_win_odds_from_se_row(row: dict) -> dict | None:
    """race_se_YYYY.csv の1行（SE_SCHEMA由来）から確定単勝オッズ・人気順を抽出する。

    SE レコードの "odds"（4byte, 1/10単位）・"popularity"（2byte）フィールドを
    そのまま整形する。異常値・欠場（odds="0000" 等）は odds=None, odds_status で
    理由を残す（_parse_odds_tenth と同じ規約）。
    """
    horse_raw = str(row.get("horse_num", "")).strip()
    if not horse_raw or not horse_raw.isdigit():
        return None
    horse_num = int(horse_raw)
    if horse_num <= 0:
        return None

    try:
        race_id = _race_id_from_race_row(row)
    except Exception:
        return None
    if len(race_id) != 16 or not race_id.isdigit():
        return None

    odds_raw = str(row.get("odds", "")).strip()
    odds, odds_status = _parse_odds_tenth(odds_raw, width=4)

    pop_raw = str(row.get("popularity", "")).strip()
    popularity = int(pop_raw) if pop_raw.isdigit() and int(pop_raw) > 0 else None

    return {
        "year": str(row.get("year", "")).zfill(4),
        "month_day": str(row.get("month_day", "")).zfill(4),
        "course_code": str(row.get("course_code", "")).zfill(2),
        "kai": str(row.get("kai", "")).zfill(2),
        "nichi": str(row.get("nichi", "")).zfill(2),
        "race_num": str(row.get("race_num", "")).zfill(2),
        "race_id": race_id,
        "horse_num": horse_num,
        "odds_raw": odds_raw,
        "odds": odds,
        "odds_status": odds_status,
        "popularity_raw": pop_raw,
        "popularity": popularity,
        "data_kubun": str(row.get("data_kubun", "")),
        # SE の確定値には速報アナウンス時刻が無いため常に空文字（O2/O3 との列互換のため保持）
        "announce_datetime": "",
    }


def fetch_win_odds_yearly(
    start_year: int = 2015,
    end_year: int | None = None,
    *,
    overwrite: bool = True,
    se_dir: Path | str | None = None,
) -> dict:
    """確定単勝オッズ・単勝人気順を年別 CSV として保存する（1番人気ベースライン用）。

    JV-Link には接続しない。既に common/data/output/race_se/race_se_YYYY.csv に
    保存されている SE レコードの odds/popularity フィールドをローカルで再整形するだけ。
    race_se_YYYY.csv が無い年は警告を出してスキップする（グレースフルデグラデーション）。

    保存先:
      common/data/output/odds/WinOdds_YYYY.csv

    Parameters
    ----------
    start_year, end_year : 対象年範囲（end_year 省略時は今年まで）
    overwrite : True（デフォルト）の場合、対象年の WinOdds_YYYY.csv を
        race_se_YYYY.csv から毎回フルリビルドする（race_se が正なので差分マージはしない）。
        False の場合、既に WinOdds_YYYY.csv が存在する年はスキップする。
    se_dir : race_se_YYYY.csv の格納ディレクトリ（テスト用に上書き可能。省略時は既定パス）

    Returns
    -------
    dict[year, {"path": str, "added": int, "skipped_reason": str | None}]
    """
    if end_year is None:
        end_year = datetime.now().year
    se_dir_path = Path(se_dir) if se_dir is not None else _se_csv_dir()

    results: dict = {}
    for year in range(int(start_year), int(end_year) + 1):
        se_path = se_dir_path / f"race_se_{year}.csv"
        win_path = _win_odds_path(year)

        if not se_path.exists():
            print(f"  [warn] {se_path.name} not found, skipping WinOdds_{year}.csv (no fetch performed)")
            results[year] = {"path": str(win_path), "added": 0, "skipped_reason": "se_csv_missing"}
            continue

        if win_path.exists() and not overwrite:
            print(f"  {win_path.name} already exists, skipping (overwrite=False)")
            results[year] = {"path": str(win_path), "added": 0, "skipped_reason": "already_exists"}
            continue

        rows_out: list[dict] = []
        with open(se_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = _parse_win_odds_from_se_row(row)
                if parsed is None:
                    continue
                rows_out.append(parsed)

        if rows_out:
            win_path.parent.mkdir(parents=True, exist_ok=True)
            save_to_csv(rows_out, str(win_path), WIN_ODDS_FIELDS, append=False)
            print(f"  WinOdds_{year}.csv: {len(rows_out):,} rows written from {se_path.name}")
        else:
            print(f"  No valid win-odds rows extracted for {year} ({se_path.name})")

        results[year] = {"path": str(win_path), "added": len(rows_out), "skipped_reason": None}

    return results


def _load_race_ids_from_ra_rows(
    ra_rows: list[dict],
    *,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
) -> list[str]:
    race_ids = []
    seen = set()
    for r in ra_rows:
        y = str(r.get("year", "")).zfill(4)
        md = str(r.get("month_day", "")).zfill(4)
        d = y + md
        if d < start_yyyymmdd or d > end_yyyymmdd:
            continue
        race_id = _race_id_from_race_row(r)
        if len(race_id) == 16 and race_id.isdigit() and race_id not in seen:
            seen.add(race_id)
            race_ids.append(race_id)
    return race_ids


def _load_year_race_ids_for_odds(year: int, *, source_output_dir: str | None = None) -> list[str]:
    source_root = (
        Path(source_output_dir)
        if source_output_dir
        else Path(__file__).parent.parent.parent / "data" / "output"
    )
    ra_path = source_root / "race_ra" / f"race_ra_{int(year)}.csv"
    if not ra_path.exists():
        print(f"race_ra_{year}.csv not found: {ra_path}")
        return []
    with open(ra_path, "r", encoding="utf-8-sig", newline="") as f:
        return _load_race_ids_from_ra_rows(
            list(csv.DictReader(f)),
            start_yyyymmdd=f"{int(year)}0101",
            end_yyyymmdd=f"{int(year)}1231",
        )


def _load_existing_pair_odds_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                keys.add(
                    (
                        row.get("race_id", ""),
                        row.get("ticket", ""),
                        row.get("data_kubun", ""),
                        row.get("announce_datetime", ""),
                    )
                )
    except Exception:
        pass
    return keys


def _fetch_pairwide_odds_0b31_for_race_ids(
    race_ids: list[str],
    *,
    quinella_path: Path,
    wide_path: Path,
    append: bool = True,
) -> dict:
    quinella_path.parent.mkdir(parents=True, exist_ok=True)
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    o2_rows: list[dict] = []
    o3_rows: list[dict] = []
    o2_keys = _load_existing_pair_odds_keys(quinella_path) if append else set()
    o3_keys = _load_existing_pair_odds_keys(wide_path) if append else set()

    client = None
    try:
        client = JRAVANClient()
        client.login()
        for race_id in race_ids:
            for dataspec, target_rec_id in (("0B32", "O2"), ("0B33", "O3")):
                try:
                    rt_ret = client.jv_link.JVRTOpen(dataspec, race_id)
                    rc = _jv_com_return_code(rt_ret)
                except Exception as e:
                    print(f"JVRTOpen({dataspec}, {race_id}) exception: {e}")
                    continue
                if rc < 0:
                    continue

                while True:
                    rr = client.jv_link.JVRead("", 512000, "")
                    if isinstance(rr, tuple):
                        st = int(rr[0])
                        raw = rr[1] if len(rr) > 1 else ""
                    else:
                        st = int(rr)
                        raw = ""

                    if st == 0:
                        break
                    if st == -1:
                        continue
                    if st == -3:
                        time.sleep(0.2)
                        continue
                    if st < -1:
                        break

                    if isinstance(raw, bytes):
                        try:
                            raw = raw.decode("cp932", "replace")
                        except Exception:
                            raw = ""
                    if not isinstance(raw, str) or not raw:
                        continue

                    for line in raw.split("\n"):
                        line = line.strip("\r\n")
                        if target_rec_id == "O2" and line.startswith("O2"):
                            for row in _parse_o2_quinella_odds(line, race_id):
                                key = (
                                    row["race_id"],
                                    row["ticket"],
                                    row["data_kubun"],
                                    row["announce_datetime"],
                                )
                                if key in o2_keys:
                                    continue
                                o2_keys.add(key)
                                o2_rows.append(row)
                        elif target_rec_id == "O3" and line.startswith("O3"):
                            for row in _parse_o3_wide_odds(line, race_id):
                                key = (
                                    row["race_id"],
                                    row["ticket"],
                                    row["data_kubun"],
                                    row["announce_datetime"],
                                )
                                if key in o3_keys:
                                    continue
                                o3_keys.add(key)
                                o3_rows.append(row)

                # 次のJVRTOpenが正しく動作するようにセッションを閉じる
                try:
                    client.jv_link.JVClose()
                except Exception:
                    pass
    finally:
        if client:
            client.close()

    if o2_rows:
        save_to_csv(o2_rows, str(quinella_path), QUINELLA_ODDS_FIELDS, append=append)
    else:
        print(f"No new O2 quinella odds records found: {quinella_path.name}")
    if o3_rows:
        save_to_csv(o3_rows, str(wide_path), WIDE_ODDS_FIELDS, append=append)
    else:
        print(f"No new O3 wide odds records found: {wide_path.name}")

    return {
        "quinella": {"path": str(quinella_path), "added": len(o2_rows)},
        "wide": {"path": str(wide_path), "added": len(o3_rows)},
        "races": len(race_ids),
    }


def fetch_odds_0b31_for_main_races(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
):
    """
    main/data/race/race_ra.csv に載っているレースID単位で 0B31 を JVRTOpen し、
    速報単勝オッズ(O1)を収集する。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    ra_path = main_race_dir / "race_ra.csv"
    if not ra_path.exists():
        print(f"race_ra.csv not found: {ra_path}")
        return {"path": "", "added": 0, "skipped": 0, "races": 0}

    with open(ra_path, "r", encoding="utf-8-sig", newline="") as f:
        ra_rows = list(csv.DictReader(f))

    race_ids = []
    seen = set()
    for r in ra_rows:
        y = str(r.get("year", "")).zfill(4)
        md = str(r.get("month_day", "")).zfill(4)
        d = y + md
        if d < start_yyyymmdd or d > end_yyyymmdd:
            continue
        rid = (
            y
            + md
            + str(r.get("course_code", "")).zfill(2)
            + str(r.get("kai", "")).zfill(2)
            + str(r.get("nichi", "")).zfill(2)
            + str(r.get("race_num", "")).zfill(2)
        )
        if len(rid) == 16 and rid.isdigit() and rid not in seen:
            seen.add(rid)
            race_ids.append((rid, r))

    script_dir = Path(__file__).parent.parent.parent
    out_dir = script_dir / "data" / "output" / "realtime_odds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "o1_odds.csv"

    dropped_past, _ = purge_realtime_csv_rows_outside_date_range(
        str(out_path), start_yyyymmdd, end_yyyymmdd
    )
    if dropped_past:
        print(
            f"  Dropped {dropped_past} past O1 odds rows "
            f"(outside {start_yyyymmdd}..{end_yyyymmdd})"
        )
    removed, _ok = purge_realtime_csv_rows_in_date_range(
        str(out_path), start_yyyymmdd, end_yyyymmdd
    )
    if removed:
        print(
            f"  Purged {removed} O1 odds rows in {start_yyyymmdd}..{end_yyyymmdd} "
            f"(snapshot refresh)"
        )

    existing_keys = set()
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    existing_keys.add(
                        (
                            row.get("year", ""),
                            row.get("month_day", ""),
                            row.get("course_code", ""),
                            row.get("kai", ""),
                            row.get("nichi", ""),
                            row.get("race_num", ""),
                            row.get("horse_num", ""),
                        )
                    )
        except Exception:
            pass

    rows_to_save = []
    added = 0
    skipped = 0

    client = None
    try:
        client = JRAVANClient()
        client.login()

        for race_id, race_row in race_ids:
            try:
                rt_ret = client.jv_link.JVRTOpen("0B31", race_id)
                rc = _jv_com_return_code(rt_ret)
            except Exception as e:
                print(f"JVRTOpen(0B31, {race_id}) exception: {e}")
                continue

            if rc < 0:
                # 速報未配信等
                continue

            while True:
                rr = client.jv_link.JVRead("", 256000, "")
                if isinstance(rr, tuple):
                    st = int(rr[0])
                    raw = rr[1] if len(rr) > 1 else ""
                else:
                    st = int(rr)
                    raw = ""

                if st == 0:
                    break
                if st == -1:
                    continue
                if st == -3:
                    time.sleep(0.2)
                    continue
                if st < -1:
                    break

                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("cp932", "replace")
                    except Exception:
                        raw = ""
                if not isinstance(raw, str) or not raw:
                    continue

                # 複数行混在に備える
                for line in raw.split("\n"):
                    line = line.strip("\r\n")
                    if not line.startswith("O1"):
                        continue
                    odds_rows = _parse_o1_win_odds(line)
                    for o in odds_rows:
                        row = {
                            "year": str(race_row.get("year", "")).zfill(4),
                            "month_day": str(race_row.get("month_day", "")).zfill(4),
                            "course_code": str(race_row.get("course_code", "")).zfill(
                                2
                            ),
                            "kai": str(race_row.get("kai", "")).zfill(2),
                            "nichi": str(race_row.get("nichi", "")).zfill(2),
                            "race_num": str(race_row.get("race_num", "")).zfill(2),
                            "horse_num": o["horse_num"],
                            "odds_raw": o["odds_raw"],
                            "popularity_raw": o.get("popularity_raw", ""),
                            "race_id": race_id,
                            "record_id": "O1",
                        }
                        key = (
                            row["year"],
                            row["month_day"],
                            row["course_code"],
                            row["kai"],
                            row["nichi"],
                            row["race_num"],
                            row["horse_num"],
                        )
                        if key in existing_keys:
                            skipped += 1
                            continue
                        existing_keys.add(key)
                        rows_to_save.append(row)
                        added += 1

            # 次のレースのJVRTOpenが正しく動作するようにセッションを閉じる
            try:
                client.jv_link.JVClose()
            except Exception:
                pass

    finally:
        if client:
            client.close()

    if rows_to_save:
        fields = [
            "year",
            "month_day",
            "course_code",
            "kai",
            "nichi",
            "race_num",
            "horse_num",
            "odds_raw",
            "popularity_raw",
            "race_id",
            "record_id",
        ]
        save_to_csv(rows_to_save, str(out_path), fields)
    else:
        print("No new O1 odds records found.")

    return {
        "path": str(out_path),
        "added": added,
        "skipped": skipped,
        "races": len(race_ids),
    }


def fetch_pairwide_odds_0b31_for_main_races(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
):
    """
    main/data/race/race_ra.csv のレースID単位で 0B32/O2 と 0B33/O3 を JVRTOpen し、
    速報の馬連(O2)・ワイド(O3)オッズをリアルタイムスナップショットとして保存する。

    保存先（O1と同じ realtime_odds/ ディレクトリ）:
      common/data/output/realtime_odds/o2_odds.csv
      common/data/output/realtime_odds/o3_odds.csv
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    start_yyyymmdd = start_date_str[:8]
    end_yyyymmdd = end_date_str[:8]

    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    ra_path = main_race_dir / "race_ra.csv"
    if not ra_path.exists():
        print(f"race_ra.csv not found: {ra_path}")
        return {"o2": {"path": "", "added": 0}, "o3": {"path": "", "added": 0}, "races": 0}

    with open(ra_path, "r", encoding="utf-8-sig", newline="") as f:
        ra_rows = list(csv.DictReader(f))

    race_ids = _load_race_ids_from_ra_rows(
        ra_rows,
        start_yyyymmdd=start_yyyymmdd,
        end_yyyymmdd=end_yyyymmdd,
    )

    # O1と同じ realtime_odds/ ディレクトリに保存（スナップショット方式）
    rt_dir = Path(__file__).parent.parent.parent / "data" / "output" / "realtime_odds"
    rt_dir.mkdir(parents=True, exist_ok=True)
    o2_rt_path = rt_dir / "o2_odds.csv"
    o3_rt_path = rt_dir / "o3_odds.csv"

    # O1と同様: 当日以外の古いデータを削除してからスナップショット更新
    for rt_path, label in ((o2_rt_path, "O2"), (o3_rt_path, "O3")):
        dropped, _ = purge_realtime_csv_rows_outside_date_range(
            str(rt_path), start_yyyymmdd, end_yyyymmdd
        )
        if dropped:
            print(f"  Dropped {dropped} past {label} odds rows (outside {start_yyyymmdd}..{end_yyyymmdd})")
        removed, _ = purge_realtime_csv_rows_in_date_range(
            str(rt_path), start_yyyymmdd, end_yyyymmdd
        )
        if removed:
            print(f"  Purged {removed} {label} odds rows in {start_yyyymmdd}..{end_yyyymmdd} (snapshot refresh)")

    result = _fetch_pairwide_odds_0b31_for_race_ids(
        race_ids,
        quinella_path=o2_rt_path,
        wide_path=o3_rt_path,
        append=True,
    )
    result["races"] = len(race_ids)
    return result


def fetch_pairwide_odds_0b31_yearly(
    start_year: int = 2015,
    end_year: int | None = None,
    *,
    source_output_dir: str | None = None,
    overwrite: bool = False,
):
    """
    common/data/output/race_ra/race_ra_YYYY.csv からレースIDを作り、
    0B32/O2 と 0B33/O3 を年別CSVとして保存する。

    保存先:
      common/data/output/odds/QuinellaOdds_YYYY.csv
      common/data/output/odds/WideOdds_YYYY.csv
    """
    if end_year is None:
        end_year = datetime.now().year
    results = {}
    for year in range(int(start_year), int(end_year) + 1):
        race_ids = _load_year_race_ids_for_odds(
            year,
            source_output_dir=source_output_dir,
        )
        if not race_ids:
            results[year] = {
                "quinella": {"path": str(_quinella_odds_path(year)), "added": 0},
                "wide": {"path": str(_wide_odds_path(year)), "added": 0},
                "races": 0,
            }
            continue
        results[year] = _fetch_pairwide_odds_0b31_for_race_ids(
            race_ids,
            quinella_path=_quinella_odds_path(year),
            wide_path=_wide_odds_path(year),
            append=not overwrite,
        )
    return results


def fetch_pairwide_odds_yearly(
    start_year: int = 2015,
    end_year: int | None = None,
    *,
    overwrite: bool = False,
):
    """
    蓄積系 RACE から確定オッズ O2/O3 を取得し、年別CSVとして保存する。

    保存先:
      common/data/output/odds/QuinellaOdds_YYYY.csv
      common/data/output/odds/WideOdds_YYYY.csv
    """
    if end_year is None:
        end_year = datetime.now().year

    results = {}
    for year in range(int(start_year), int(end_year) + 1):
        year_start = f"{year}0101000000"
        year_end = f"{year}1231235959"
        quinella_path = _quinella_odds_path(year)
        wide_path = _wide_odds_path(year)
        quinella_path.parent.mkdir(parents=True, exist_ok=True)

        o2_rows: list[dict] = []
        o3_rows: list[dict] = []
        o2_keys = set() if overwrite else _load_existing_pair_odds_keys(quinella_path)
        o3_keys = set() if overwrite else _load_existing_pair_odds_keys(wide_path)

        client = None
        try:
            client = JRAVANClient()
            client.login()
            print(f"Fetching RACE O2/O3 year={year} ...", flush=True)
            for raw_chunk in client.get_data("RACE", year_start, 4, year_end):
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue
                if not raw_chunk:
                    continue
                for line in raw_chunk.split("\n"):
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    if line.startswith("O2"):
                        parsed = parse_fixed_width(line.encode("cp932", "replace"), SCHEMAS["O2"])
                        race_id = _race_id_from_race_row(parsed)
                        for row in _parse_o2_quinella_odds(line, race_id):
                            if str(row.get("year", "")).zfill(4) != str(year):
                                continue
                            key = (
                                row["race_id"],
                                row["ticket"],
                                row["data_kubun"],
                                row["announce_datetime"],
                            )
                            if key in o2_keys:
                                continue
                            o2_keys.add(key)
                            o2_rows.append(row)
                    elif line.startswith("O3"):
                        parsed = parse_fixed_width(line.encode("cp932", "replace"), SCHEMAS["O3"])
                        race_id = _race_id_from_race_row(parsed)
                        for row in _parse_o3_wide_odds(line, race_id):
                            if str(row.get("year", "")).zfill(4) != str(year):
                                continue
                            key = (
                                row["race_id"],
                                row["ticket"],
                                row["data_kubun"],
                                row["announce_datetime"],
                            )
                            if key in o3_keys:
                                continue
                            o3_keys.add(key)
                            o3_rows.append(row)
        finally:
            if client:
                client.close()

        if o2_rows:
            save_to_csv(
                o2_rows,
                str(quinella_path),
                QUINELLA_ODDS_FIELDS,
                append=not overwrite,
            )
        else:
            print(f"No new historical O2 records found: {quinella_path.name}")
        if o3_rows:
            save_to_csv(
                o3_rows,
                str(wide_path),
                WIDE_ODDS_FIELDS,
                append=not overwrite,
            )
        else:
            print(f"No new historical O3 records found: {wide_path.name}")

        results[year] = {
            "quinella": {"path": str(quinella_path), "added": len(o2_rows)},
            "wide": {"path": str(wide_path), "added": len(o3_rows)},
        }
    return results


def fetch_pairwide_odds_bulk(
    start_year: int = 2015,
    end_year: int | None = None,
    *,
    overwrite: bool = False,
    flush_rows: int = 50000,
):
    """
    蓄積系 RACE を start_year から1回だけ読み、O2/O3を年別CSVへ逐次保存する。

    古い年を年別に JVOpen すると同じ後続年データを何度も読むため、この関数は
    大量取得向けにストリーミング保存する。
    """
    if end_year is None:
        end_year = datetime.now().year
    start_year = int(start_year)
    end_year = int(end_year)

    odds_dir = _odds_output_dir()
    odds_dir.mkdir(parents=True, exist_ok=True)
    results = {
        year: {
            "quinella": {"path": str(_quinella_odds_path(year)), "added": 0},
            "wide": {"path": str(_wide_odds_path(year)), "added": 0},
        }
        for year in range(start_year, end_year + 1)
    }
    if overwrite:
        for year in range(start_year, end_year + 1):
            for path in (_quinella_odds_path(year), _wide_odds_path(year)):
                if path.exists():
                    path.unlink()

    buffers: dict[tuple[int, str], list[dict]] = {}
    append_written: set[Path] = set()

    def _flush(force: bool = False) -> None:
        for (year, rec_id), rows in list(buffers.items()):
            if not rows:
                continue
            if not force and len(rows) < flush_rows:
                continue
            if rec_id == "O2":
                path = _quinella_odds_path(year)
                fields = QUINELLA_ODDS_FIELDS
            else:
                path = _wide_odds_path(year)
                fields = WIDE_ODDS_FIELDS
            save_to_csv(rows, str(path), fields, append=path in append_written)
            append_written.add(path)
            results[year]["quinella" if rec_id == "O2" else "wide"]["added"] += len(rows)
            buffers[(year, rec_id)] = []

    client = None
    chunk_count = 0
    start_time = f"{start_year}0101000000"
    end_time = f"{end_year}1231235959"
    try:
        client = JRAVANClient()
        client.login()
        print(
            f"Streaming RACE O2/O3 from {start_time} to {end_time} ...",
            flush=True,
        )
        for raw_chunk in client.get_data("RACE", start_time, 4, end_time):
            chunk_count += 1
            if isinstance(raw_chunk, bytes):
                try:
                    raw_chunk = raw_chunk.decode("cp932", "replace")
                except Exception:
                    continue
            if not raw_chunk:
                continue
            for line in raw_chunk.split("\n"):
                line = line.strip("\r\n")
                if not line:
                    continue
                if line.startswith("O2"):
                    parsed = parse_fixed_width(line.encode("cp932", "replace"), SCHEMAS["O2"])
                    year = int(str(parsed.get("year", "0") or "0"))
                    if year < start_year or year > end_year:
                        continue
                    race_id = _race_id_from_race_row(parsed)
                    buffers.setdefault((year, "O2"), []).extend(
                        _parse_o2_quinella_odds(line, race_id)
                    )
                elif line.startswith("O3"):
                    parsed = parse_fixed_width(line.encode("cp932", "replace"), SCHEMAS["O3"])
                    year = int(str(parsed.get("year", "0") or "0"))
                    if year < start_year or year > end_year:
                        continue
                    race_id = _race_id_from_race_row(parsed)
                    buffers.setdefault((year, "O3"), []).extend(
                        _parse_o3_wide_odds(line, race_id)
                    )
            _flush(force=False)
            if chunk_count % 10 == 0:
                total_o2 = sum(v["quinella"]["added"] for v in results.values())
                total_o3 = sum(v["wide"]["added"] for v in results.values())
                print(
                    f"  chunks={chunk_count:,} saved_o2={total_o2:,} saved_o3={total_o3:,}",
                    flush=True,
                )
    finally:
        _flush(force=True)
        if client:
            client.close()

    print("Completed streaming RACE O2/O3.", flush=True)
    return results


def merge_odds_to_main_se(
    *,
    output_dir: str | None = None,
    source_output_dir: str | None = None,
):
    """
    realtime_odds/o1_odds.csv の odds_raw / popularity_raw を race_se の odds / popularity に反映。
    人気順は近年・当日データ向け。1993年6月以前の特殊仕様のデータでは要検証。
    """
    project_root = Path(__file__).parent.parent.parent.parent
    main_race_dir = (
        Path(output_dir) if output_dir else (project_root / "main" / "data" / "race")
    )
    source_dir = (
        Path(source_output_dir)
        if source_output_dir
        else (project_root / "common" / "data" / "output")
    )

    se_path = main_race_dir / "race_se.csv"
    odds_path = source_dir / "realtime_odds" / "o1_odds.csv"
    if not se_path.exists():
        print(f"race_se.csv not found: {se_path}")
        return {"updated": 0, "total": 0}
    if not odds_path.exists():
        print(f"odds source not found: {odds_path}")
        return {"updated": 0, "total": 0}

    with open(se_path, "r", encoding="utf-8-sig", newline="") as f:
        se_reader = csv.DictReader(f)
        se_fields = se_reader.fieldnames or []
        se_rows = list(se_reader)

    odds_map: dict[tuple, str] = {}
    pop_map: dict[tuple, str] = {}
    with open(odds_path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            k = (
                str(r.get("year", "")).zfill(4),
                str(r.get("month_day", "")).zfill(4),
                str(r.get("course_code", "")).zfill(2),
                str(r.get("kai", "")).zfill(2),
                str(r.get("nichi", "")).zfill(2),
                str(r.get("race_num", "")).zfill(2),
                str(r.get("horse_num", "")).zfill(2),
            )
            odds_map[k] = str(r.get("odds_raw", "")).zfill(4)
            pr = str(r.get("popularity_raw", "")).strip()
            if pr.isdigit() and int(pr) > 0:
                pop_map[k] = pr.zfill(2)

    updated_odds = 0
    updated_pop = 0
    for r in se_rows:
        k = (
            str(r.get("year", "")).zfill(4),
            str(r.get("month_day", "")).zfill(4),
            str(r.get("course_code", "")).zfill(2),
            str(r.get("kai", "")).zfill(2),
            str(r.get("nichi", "")).zfill(2),
            str(r.get("race_num", "")).zfill(2),
            str(r.get("horse_num", "")).zfill(2),
        )
        v = odds_map.get(k, "")
        if v and v.isdigit() and int(v) > 0:
            if str(r.get("odds", "")).zfill(4) != v:
                r["odds"] = v
                updated_odds += 1
        pv = pop_map.get(k, "")
        if pv and str(r.get("popularity", "")).strip().zfill(2) != pv:
            r["popularity"] = pv
            updated_pop += 1

    with open(se_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=se_fields)
        writer.writeheader()
        writer.writerows(se_rows)

    print(
        f"merge_odds_to_main_se: odds {updated_odds}, popularity {updated_pop} "
        f"(total se rows {len(se_rows)})."
    )
    return {
        "updated": updated_odds,
        "updated_popularity": updated_pop,
        "total": len(se_rows),
    }


def refresh_today_odds_data(
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    *,
    output_dir: str | None = None,
):
    """
    main/data/race の対象レースに対して 0B31(速報単勝オッズ)を取得し、
    race_se.csv の odds 列へ反映する。
    """
    start_date_str, end_date_str = _normalize_datetime_range(
        start_date_str, end_date_str
    )
    print(f"=== refresh_today_odds_data: {start_date_str} -> {end_date_str} ===")
    fetch_res = fetch_odds_0b31_for_main_races(
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        output_dir=output_dir,
    )
    merge_res = merge_odds_to_main_se(output_dir=output_dir)
    return {"fetch": fetch_res, "merge": merge_res}


def fetch_related_data_from_se(
    se_filepath=None,
    start_date_str="20180101000000",
    end_date_str="20241231999999",
):
    """
    SEデータから血統登録番号を取得し、関連データ（HN, SK, BT, WC, HC）を取得する関数

    Args:
        se_filepath: SEデータのCSVファイルパス
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
        end_date_str: 終了日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    import pandas as pd

    print("=== SEデータから関連データを取得 ===")

    # 出力ディレクトリの準備（プロジェクトルートからの相対パス）
    script_dir = Path(__file__).parent.parent.parent
    output_dir = script_dir / "data" / "output"
    output_dir = str(output_dir)

    # se_filepathが指定されていない場合はデフォルトパスを使用
    if se_filepath is None:
        se_filepath = os.path.join(output_dir, "race_se.csv")

    # 日付形式の調整
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    START_YEAR = int(start_date_str[:4])
    END_YEAR = int(end_date_str[:4])

    # SEデータを読み込み
    if not os.path.exists(se_filepath):
        print(f"Error: SEファイルが見つかりません: {se_filepath}")
        return

    try:
        df_se = pd.read_csv(se_filepath, encoding="utf-8-sig", dtype=str)
        print(f"SEデータ読み込み完了: {len(df_se)}件")
    except Exception as e:
        print(f"Error loading SE data: {e}")
        return

    # SEから血統登録番号のリストを取得（重複除去）
    ketto_nums = df_se["ketto_num"].dropna().unique().tolist()
    print(f"取得対象の血統登録番号数: {len(ketto_nums)}")

    if not ketto_nums:
        print("血統登録番号が見つかりませんでした。")
        return

    # 出力ディレクトリが既に設定されている場合はそれを使用
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 取得対象のデータタイプ
    target_schemas = {
        # HN, SK, BT は BLDN dataspec から取得
        "HN": {"dataspec": "BLDN", "key_field": "ketto_num", "is_master": True},
        "SK": {"dataspec": "BLDN", "key_field": "ketto_num", "is_master": True},
        "BT": {
            "dataspec": "BLDN",
            "key_field": "breeding_reg_num",
            "is_master": True,
            "via": "HN",
        },
        "WC": {
            "dataspec": "WOOD",
            "key_field": "ketto_num",
            "is_master": False,
            "date_field": "training_date",
        },
        "HC": {
            "dataspec": "SLOP",
            "key_field": "ketto_num",
            "is_master": False,
            "date_field": "training_date",
        },
    }

    client = None

    try:
        for rec_id, config in target_schemas.items():
            if rec_id not in SCHEMAS:
                print(
                    f"Warning: {rec_id}スキーマが定義されていません。スキップします。"
                )
                continue

            ds = config["dataspec"]
            key_field = config["key_field"]
            is_master = config.get("is_master", False)
            date_field = config.get("date_field", None)
            via = config.get("via", None)

            print(f"\n--- {rec_id}データを取得中 ({ds}) ---")

            # 出力ファイルパス（サブディレクトリ内）
            if ds == "BLDN":
                subdir = f"blod_{rec_id.lower()}"
                fname = f"blod_{rec_id.lower()}.csv"
            elif ds == "MING":
                subdir = f"ming_{rec_id.lower()}"
                fname = f"ming_{rec_id.lower()}.csv"
            elif ds == "RACE":
                subdir = f"race_{rec_id.lower()}"
                fname = f"race_{rec_id.lower()}.csv"
            elif ds == "SLOP":
                subdir = f"slop_{rec_id.lower()}"
                fname = f"slop_{rec_id.lower()}.csv"
            elif ds == "WOOD":
                subdir = f"wood_{rec_id.lower()}"
                fname = f"wood_{rec_id.lower()}.csv"
            else:
                subdir = f"{ds.lower()}_{rec_id.lower()}"
                fname = f"{ds.lower()}_{rec_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            output_file = os.path.join(subdir_path, fname)

            # 既存ファイルがある場合は読み込んで既存のketto_numを取得
            existing_ketto_nums = set()
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                try:
                    df_existing = pd.read_csv(
                        output_file, encoding="utf-8-sig", dtype=str
                    )
                    if key_field in df_existing.columns:
                        existing_ketto_nums = set(
                            df_existing[key_field].dropna().unique()
                        )
                    print(f"既存データ: {len(existing_ketto_nums)}件")
                except:
                    pass

            # BTの場合は、HNを経由してbreeding_reg_numを取得
            if via == "HN":
                # HNデータを読み込んで、ketto_numからbreeding_reg_numへのマッピングを作成
                hn_subdir = os.path.join(output_dir, "blod_hn")
                hn_file = os.path.join(hn_subdir, "blod_hn.csv")
                if not os.path.exists(hn_file):
                    print(
                        f"Warning: HNファイルが見つかりません。BTの取得にはHNが必要です。"
                    )
                    continue

                try:
                    df_hn = pd.read_csv(hn_file, encoding="utf-8-sig", dtype=str)
                    # ketto_num -> breeding_reg_num のマッピング
                    ketto_to_breeding = dict(
                        zip(df_hn["ketto_num"], df_hn["breeding_reg_num"])
                    )
                    # SEのketto_numに対応するbreeding_reg_numを取得
                    target_keys = [
                        ketto_to_breeding.get(k, None)
                        for k in ketto_nums
                        if k in ketto_to_breeding
                    ]
                    target_keys = [
                        k for k in target_keys if k and k not in existing_ketto_nums
                    ]
                    print(f"BT取得対象のbreeding_reg_num数: {len(target_keys)}")
                except Exception as e:
                    print(f"Error loading HN data: {e}")
                    continue
            else:
                # 通常のketto_numベースの取得
                target_keys = [k for k in ketto_nums if k not in existing_ketto_nums]
                print(f"{rec_id}取得対象の{key_field}数: {len(target_keys)}")

            if not target_keys:
                print(f"{rec_id}: すべてのデータが既に取得済みです。")
                continue

            # データ取得
            client = JRAVANClient()
            try:
                client.login()
            except Exception as e:
                print(f"Login failed: {e}")
                if client:
                    client.close()
                continue

            records_buffer = []
            total_count = 0

            # SLOP (HC) と BLDN (HN, SK, BT), WOOD(WC) は Option 3 (Setup) を使用
            # BLDNは1986年から、SLOP/WOODは2023年から取得
            if ds == "BLDN":
                fetch_option = 3
                fetch_start_date = "19860101000000"  # BLDNは1986年から
            elif ds in ["SLOP", "WOOD"]:
                fetch_option = 3
                fetch_start_date = "20230101000000"  # SLOP/WOODは2023年から
            else:
                fetch_option = 4
                fetch_start_date = start_date_str

            print(f"  Fetch option: {fetch_option}, Start date: {fetch_start_date}")

            try:
                for raw_chunk in client.get_data(ds, fetch_start_date, fetch_option):
                    try:
                        byte_chunk = raw_chunk.encode("cp932", "replace")
                    except Exception:
                        byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                    if not byte_chunk:
                        continue

                    lines = raw_chunk.split("\n")

                    for line in lines:
                        line = line.strip("\r\n")
                        if not line:
                            continue

                        try:
                            line_bytes = line.encode("cp932", "replace")
                        except:
                            continue

                        if len(line_bytes) < 2:
                            continue

                        rec_id_found = line_bytes[:2].decode("ascii", errors="ignore")

                        if rec_id_found != rec_id:
                            continue

                        if rec_id_found in SCHEMAS:
                            parsed = parse_fixed_width(
                                line_bytes, SCHEMAS[rec_id_found]
                            )

                            # キーフィールドでフィルタリング
                            key_value = parsed.get(key_field, "").strip()
                            if not key_value or key_value not in target_keys:
                                continue

                            # 日付フィルタリング（調教データの場合）
                            if date_field and date_field in parsed:
                                try:
                                    training_date_str = parsed[date_field].strip()
                                    if len(training_date_str) == 8:  # YYYYMMDD形式
                                        rec_year = int(training_date_str[:4])
                                        rec_month_day = training_date_str[4:8]

                                        # 年でフィルタリング
                                        if rec_year < START_YEAR or rec_year > END_YEAR:
                                            continue

                                        # 同じ年の場合、月日でフィルタリング
                                        if rec_year == START_YEAR:
                                            start_month_day = start_date_str[4:8]
                                            if rec_month_day < start_month_day:
                                                continue

                                        if rec_year == END_YEAR:
                                            end_month_day = end_date_str[4:8]
                                            if rec_month_day > end_month_day:
                                                continue
                                except:
                                    pass

                            parsed["raw_hex"] = line_bytes.hex()
                            records_buffer.append(parsed)
                            total_count += 1

                            # バッチ保存
                            if len(records_buffer) >= 10000:
                                fields = get_schema_fieldnames(rec_id_found) + [
                                    "raw_hex"
                                ]
                                save_to_csv(records_buffer, output_file, fields)
                                records_buffer = []

            except Exception as e:
                print(f"Error processing {rec_id}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                if client:
                    client.close()
                    client = None

            # 最終保存
            if records_buffer:
                fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                save_to_csv(records_buffer, output_file, fields)
                print(f"  Saved {total_count} records to {output_file}")

            # デバッグ情報
            if total_count == 0:
                print(f"  Warning: {rec_id}データが取得できませんでした。")
            else:
                print(f"  取得完了: {total_count}件")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if client:
            client.close()
        print("\n完了しました。")


def fetch_se_only(start_date_str="20240101000000", end_date_str="20241231235959"):
    """
    SEデータのみを取得する関数

    Args:
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
        end_date_str: 終了日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    print("=== SEデータのみ取得 ===")

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    START_DATE = start_date_str
    START_DATE_YYYYMMDD = START_DATE[:8]
    END_DATE_YYYYMMDD = end_date_str[:8]

    try:
        # SEデータのみを取得するタスク
        task = {
            "dataspec": "RACE",
            "option": 4,
            "start_date": START_DATE,
            "target_ids": ["SE"],
        }

        # クライアント初期化
        client = JRAVANClient()
        try:
            client.login()
        except:
            print("Login failed, aborting.")
            if client:
                client.close()
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = task["target_ids"]
        task_start_date = task["start_date"]

        print(f"\n--- Processing {ds} (SE only) ---")

        # 出力ディレクトリの準備
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "data" / "output"
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time.sleep(2)

        # 既存データの日付キーを読み込む（サブディレクトリ内）
        existing_data_keys = {}
        for target_id in targets:
            # サブディレクトリのパスを構築
            subdir = f"race_{target_id.lower()}"
            fname = f"race_{target_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            existing_keys = load_existing_dates(fpath, target_id)
            existing_data_keys[target_id] = existing_keys
            if existing_keys:
                print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

        records_buffer = {}
        total_count = 0
        skipped_count = 0

        print(f"Requesting {ds} (From: {task_start_date}, Opt: {opt})...")
        pbar = tqdm(
            desc=f"Fetching {ds} (SE only)", unit="chunks", position=0, leave=True
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt):
                pbar.update(1)
                try:
                    byte_chunk = raw_chunk.encode("cp932", "replace")
                except Exception:
                    byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                if not byte_chunk:
                    continue
                lines = raw_chunk.split("\n")

                for line in lines:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != "SE":  # SEのみ処理
                        continue

                    if rec_id in SCHEMAS:
                        parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                        # Data Kubun Filtering
                        if parsed.get("data_kubun") != "7":
                            continue

                        # Date Filtering (strict YYYYMMDD range)
                        rec_date = str(parsed.get("year", "")).strip().zfill(4) + str(
                            parsed.get("month_day", "")
                        ).strip().zfill(4)
                        if len(rec_date) != 8 or not rec_date.isdigit():
                            continue
                        if (
                            rec_date < START_DATE_YYYYMMDD
                            or rec_date > END_DATE_YYYYMMDD
                        ):
                            continue

                        # 既存データチェック（重複スキップ）
                        existing_keys = existing_data_keys.get(rec_id, set())
                        # SEは馬単位のデータなので、horse_numもキーに含める
                        record_key = (
                            str(parsed.get("year", "")),
                            str(parsed.get("month_day", "")),
                            str(parsed.get("course_code", "")),
                            str(parsed.get("kai", "")),
                            str(parsed.get("nichi", "")),
                            str(parsed.get("race_num", "")),
                            str(parsed.get("horse_num", "")),
                        )
                        if record_key in existing_keys:
                            skipped_count += 1
                            continue

                        # 新しいデータのみ追加
                        parsed["raw_hex"] = line_bytes.hex()
                        if rec_id not in records_buffer:
                            records_buffer[rec_id] = []
                        records_buffer[rec_id].append(parsed)

                        # 既存キーセットに追加（メモリ内で重複チェック）
                        existing_keys.add(record_key)

                        total_count += 1

                        if len(records_buffer[rec_id]) >= 100000:
                            # サブディレクトリのパスを構築
                            subdir = f"race_{rec_id.lower()}"
                            fname = f"race_{rec_id.lower()}.csv"

                            subdir_path = os.path.join(output_dir, subdir)
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path, exist_ok=True)

                            save_path = os.path.join(subdir_path, fname)
                            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                            save_to_csv(records_buffer[rec_id], save_path, fields)
                            records_buffer[rec_id] = []

                # チャンク処理後、進捗バーにレコード数を表示
                pbar.set_postfix(
                    {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                )

        except Exception as e:
            print(f"!!! Error processing {ds}: {e}")
        finally:
            pbar.close()
            if client:
                client.close()
                client = None

        # 残りのデータを保存（サブディレクトリ内）
        for rid, data_list in records_buffer.items():
            if data_list:
                # サブディレクトリのパスを構築
                subdir = f"race_{rid.lower()}"
                fname = f"race_{rid.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                save_path = os.path.join(subdir_path, fname)
                fields = get_schema_fieldnames(rid) + ["raw_hex"]
                save_to_csv(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                print(
                    f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                )
            else:
                print(f"No relevant records found for {ds}.")
        else:
            print(
                f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
            )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_se_date_range_to_csv(
    start_date_str: str,
    end_date_str: str,
    output_csv: str | Path,
    *,
    include_kubun_2: bool = True,
) -> int:
    """
    RACE 仕様で SE レコードだけを取得し、指定パスに1ファイルで保存する（JV-Link 要・32bit 推奨）。

    - 日付は ``start_date_str`` / ``end_date_str`` のカレンダー区間（8桁なら時刻を補完）。
    - ``include_kubun_2=True`` のとき data_kubun 2 と 7 の両方を取り、7 を優先して1行にマージ。
    - 既存の ``common/data/output/race_se/race_se.csv`` は読まず、常に今回のストリームのみで組み立てる。

    Args:
        start_date_str: 開始（YYYYMMDD または YYYYMMDDHHMMSS）
        end_date_str: 終了
        output_csv: 出力 CSV（親ディレクトリは存在しなくても作成）
        include_kubun_2: False のとき従来どおり区分7のみ

    Returns:
        書き出した SE 行数（キー単位＝馬単位）
    """
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    out = Path(output_csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    allowed = _FETCH_JRA_RACE_KUBUNS if include_kubun_2 else frozenset({"7"})
    se_merge: dict = {}
    client = None

    print(
        f"=== fetch_se_date_range_to_csv: {start_date_str[:8]} .. {end_date_str[:8]} -> {out} ==="
    )

    try:
        client = JRAVANClient()
        client.login()
        time.sleep(1)

        pbar = tqdm(desc="SE -> file", unit="chunks", position=0, leave=True)
        try:
            for raw_chunk in client.get_data("RACE", start_date_str, 4, end_date_str):
                pbar.update(1)
                if isinstance(raw_chunk, bytes):
                    try:
                        raw_chunk = raw_chunk.decode("cp932", "replace")
                    except Exception:
                        continue
                if not raw_chunk:
                    continue

                for line in raw_chunk.split("\n"):
                    line = line.rstrip("\r\n")
                    if not line or len(line) < 2:
                        continue
                    if line[:2] != "SE":
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    parsed = parse_fixed_width(line_bytes, SCHEMAS["SE"])
                    if not _se_record_in_time_window(
                        parsed, start_date_str, end_date_str
                    ):
                        continue
                    parsed["raw_hex"] = line_bytes.hex()
                    _race_stream_try_merge(se_merge, parsed, "SE", allowed)
                pbar.set_postfix(rows=len(se_merge))
        finally:
            pbar.close()

        rows = [t[1] for t in se_merge.values()]
        fields = get_schema_fieldnames("SE") + ["raw_hex"]
        if rows:
            save_to_csv(rows, str(out), fields, append=False)
        print(f"  Wrote {len(rows)} SE rows to {out}")
        return len(rows)
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass


def fetch_sk_only(start_date_str="19860101000000"):
    """
    SKデータのみを取得する関数

    Args:
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    print("=== SKデータのみ取得 ===")

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"

    try:
        # SKデータのみを取得するタスク
        task = {
            "dataspec": "BLDN",
            "option": 3,
            "start_date": start_date_str,
            "target_ids": ["SK"],
        }

        # クライアント初期化
        client = JRAVANClient()
        try:
            client.login()
        except:
            print("Login failed, aborting.")
            if client:
                client.close()
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = task["target_ids"]
        task_start_date = task["start_date"]

        print(f"\n--- Processing {ds} (SK only) ---")

        # 出力ディレクトリの準備
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "data" / "output"
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time.sleep(2)

        # 既存データのキーを読み込む（サブディレクトリ内）
        existing_data_keys = {}
        for target_id in targets:
            # サブディレクトリのパスを構築
            subdir = f"blod_{target_id.lower()}"
            fname = f"blod_{target_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            existing_keys = load_existing_dates(fpath, target_id)
            existing_data_keys[target_id] = existing_keys
            if existing_keys:
                print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

        records_buffer = {}
        total_count = 0
        skipped_count = 0

        print(f"Requesting {ds} (From: {task_start_date}, Opt: {opt})...")
        pbar = tqdm(
            desc=f"Fetching {ds} (SK only)", unit="chunks", position=0, leave=True
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt):
                pbar.update(1)
                try:
                    byte_chunk = raw_chunk.encode("cp932", "replace")
                except Exception:
                    byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                if not byte_chunk:
                    continue
                lines = raw_chunk.split("\n")

                for line in lines:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != "SK":  # SKのみ処理
                        continue

                    if rec_id in SCHEMAS:
                        parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                        # 既存データチェック（重複スキップ）
                        existing_keys = existing_data_keys.get(rec_id, set())
                        # SKはketto_numベースでチェック
                        record_key = str(parsed.get("ketto_num", ""))
                        if record_key and record_key in existing_keys:
                            skipped_count += 1
                            continue

                        # 新しいデータのみ追加
                        parsed["raw_hex"] = line_bytes.hex()
                        if rec_id not in records_buffer:
                            records_buffer[rec_id] = []
                        records_buffer[rec_id].append(parsed)

                        # 既存キーセットに追加（メモリ内で重複チェック）
                        if record_key:
                            existing_keys.add(record_key)

                        total_count += 1

                        if len(records_buffer[rec_id]) >= 100000:
                            # サブディレクトリのパスを構築
                            subdir = f"blod_{rec_id.lower()}"
                            fname = f"blod_{rec_id.lower()}.csv"

                            subdir_path = os.path.join(output_dir, subdir)
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path, exist_ok=True)

                            save_path = os.path.join(subdir_path, fname)
                            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                            save_to_csv(records_buffer[rec_id], save_path, fields)
                            records_buffer[rec_id] = []

                # チャンク処理後、進捗バーにレコード数を表示
                pbar.set_postfix(
                    {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                )

        except Exception as e:
            print(f"!!! Error processing {ds}: {e}")
        finally:
            pbar.close()
            if client:
                client.close()
                client = None

        # 残りのデータを保存（サブディレクトリ内）
        for rid, data_list in records_buffer.items():
            if data_list:
                # サブディレクトリのパスを構築
                subdir = f"blod_{rid.lower()}"
                fname = f"blod_{rid.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                save_path = os.path.join(subdir_path, fname)
                fields = get_schema_fieldnames(rid) + ["raw_hex"]
                save_to_csv(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                print(
                    f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                )
            else:
                print(f"No relevant records found for {ds}.")
        else:
            print(
                f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
            )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_hn_only(start_date_str="19860101000000"):
    """
    HNデータのみを取得する関数

    Args:
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    print("=== HNデータのみ取得 ===")

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"

    try:
        # HNデータのみを取得するタスク
        task = {
            "dataspec": "BLDN",
            "option": 3,
            "start_date": start_date_str,
            "target_ids": ["HN"],
        }

        # クライアント初期化
        client = JRAVANClient()
        try:
            client.login()
        except:
            print("Login failed, aborting.")
            if client:
                client.close()
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = task["target_ids"]
        task_start_date = task["start_date"]

        print(f"\n--- Processing {ds} (HN only) ---")

        # 出力ディレクトリの準備
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "data" / "output"
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time.sleep(2)

        # 既存データのキーを読み込む（サブディレクトリ内）
        existing_data_keys = {}
        for target_id in targets:
            # サブディレクトリのパスを構築
            subdir = f"blod_{target_id.lower()}"
            fname = f"blod_{target_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            existing_keys = load_existing_dates(fpath, target_id)
            existing_data_keys[target_id] = existing_keys
            if existing_keys:
                print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

        records_buffer = {}
        total_count = 0
        skipped_count = 0

        print(f"Requesting {ds} (From: {task_start_date}, Opt: {opt})...")
        pbar = tqdm(
            desc=f"Fetching {ds} (HN only)", unit="chunks", position=0, leave=True
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt):
                pbar.update(1)
                try:
                    byte_chunk = raw_chunk.encode("cp932", "replace")
                except Exception:
                    byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                if not byte_chunk:
                    continue
                lines = raw_chunk.split("\n")

                for line in lines:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != "HN":  # HNのみ処理
                        continue

                    if rec_id in SCHEMAS:
                        parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                        # 既存データチェック（重複スキップ）
                        existing_keys = existing_data_keys.get(rec_id, set())
                        # HNはketto_numベースでチェック
                        record_key = str(parsed.get("ketto_num", ""))
                        if record_key and record_key in existing_keys:
                            skipped_count += 1
                            continue

                        # 新しいデータのみ追加
                        parsed["raw_hex"] = line_bytes.hex()
                        if rec_id not in records_buffer:
                            records_buffer[rec_id] = []
                        records_buffer[rec_id].append(parsed)

                        # 既存キーセットに追加（メモリ内で重複チェック）
                        if record_key:
                            existing_keys.add(record_key)

                        total_count += 1

                        if len(records_buffer[rec_id]) >= 100000:
                            # サブディレクトリのパスを構築
                            subdir = f"blod_{rec_id.lower()}"
                            fname = f"blod_{rec_id.lower()}.csv"

                            subdir_path = os.path.join(output_dir, subdir)
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path, exist_ok=True)

                            save_path = os.path.join(subdir_path, fname)
                            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                            save_to_csv(records_buffer[rec_id], save_path, fields)
                            records_buffer[rec_id] = []

                # チャンク処理後、進捗バーにレコード数を表示
                pbar.set_postfix(
                    {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                )

        except Exception as e:
            print(f"!!! Error processing {ds}: {e}")
        finally:
            pbar.close()
            if client:
                client.close()
                client = None

        # 残りのデータを保存（サブディレクトリ内）
        for rid, data_list in records_buffer.items():
            if data_list:
                # サブディレクトリのパスを構築
                subdir = f"blod_{rid.lower()}"
                fname = f"blod_{rid.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                save_path = os.path.join(subdir_path, fname)
                fields = get_schema_fieldnames(rid) + ["raw_hex"]
                save_to_csv(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                print(
                    f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                )
            else:
                print(f"No relevant records found for {ds}.")
        else:
            print(
                f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
            )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_bt_only(start_date_str="19860101000000"):
    """
    BTデータのみを取得する関数

    Args:
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    print("=== BTデータのみ取得 ===")

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"

    try:
        # BTデータのみを取得するタスク
        task = {
            "dataspec": "BLDN",
            "option": 3,
            "start_date": start_date_str,
            "target_ids": ["BT"],
        }

        # クライアント初期化
        client = JRAVANClient()
        try:
            client.login()
        except:
            print("Login failed, aborting.")
            if client:
                client.close()
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = task["target_ids"]
        task_start_date = task["start_date"]

        print(f"\n--- Processing {ds} (BT only) ---")

        # 出力ディレクトリの準備
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "data" / "output"
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time.sleep(2)

        # 既存データのキーを読み込む（サブディレクトリ内）
        existing_data_keys = {}
        for target_id in targets:
            # サブディレクトリのパスを構築
            subdir = f"blod_{target_id.lower()}"
            fname = f"blod_{target_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            existing_keys = load_existing_dates(fpath, target_id)
            existing_data_keys[target_id] = existing_keys
            if existing_keys:
                print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

        records_buffer = {}
        total_count = 0
        skipped_count = 0

        print(f"Requesting {ds} (From: {task_start_date}, Opt: {opt})...")
        pbar = tqdm(
            desc=f"Fetching {ds} (BT only)", unit="chunks", position=0, leave=True
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt):
                pbar.update(1)
                try:
                    byte_chunk = raw_chunk.encode("cp932", "replace")
                except Exception:
                    byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                if not byte_chunk:
                    continue
                lines = raw_chunk.split("\n")

                for line in lines:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != "BT":  # BTのみ処理
                        continue

                    if rec_id in SCHEMAS:
                        parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                        # 既存データチェック（重複スキップ）
                        existing_keys = existing_data_keys.get(rec_id, set())
                        # BTはbreeding_reg_numベースでチェック
                        record_key = str(parsed.get("breeding_reg_num", ""))
                        if record_key and record_key in existing_keys:
                            skipped_count += 1
                            continue

                        # 新しいデータのみ追加
                        parsed["raw_hex"] = line_bytes.hex()
                        if rec_id not in records_buffer:
                            records_buffer[rec_id] = []
                        records_buffer[rec_id].append(parsed)

                        # 既存キーセットに追加（メモリ内で重複チェック）
                        if record_key:
                            existing_keys.add(record_key)

                        total_count += 1

                        if len(records_buffer[rec_id]) >= 100000:
                            # サブディレクトリのパスを構築
                            subdir = f"blod_{rec_id.lower()}"
                            fname = f"blod_{rec_id.lower()}.csv"

                            subdir_path = os.path.join(output_dir, subdir)
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path, exist_ok=True)

                            save_path = os.path.join(subdir_path, fname)
                            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                            save_to_csv(records_buffer[rec_id], save_path, fields)
                            records_buffer[rec_id] = []

                # チャンク処理後、進捗バーにレコード数を表示
                pbar.set_postfix(
                    {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                )

        except Exception as e:
            print(f"!!! Error processing {ds}: {e}")
        finally:
            pbar.close()
            if client:
                client.close()
                client = None

        # 残りのデータを保存（サブディレクトリ内）
        for rid, data_list in records_buffer.items():
            if data_list:
                # サブディレクトリのパスを構築
                subdir = f"blod_{rid.lower()}"
                fname = f"blod_{rid.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                save_path = os.path.join(subdir_path, fname)
                fields = get_schema_fieldnames(rid) + ["raw_hex"]
                save_to_csv(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                print(
                    f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                )
            else:
                print(f"No relevant records found for {ds}.")
        else:
            print(
                f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
            )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def fetch_ra_only(start_date_str="20240101000000", end_date_str="20241231235959"):
    """
    RAデータのみを取得する関数

    Args:
        start_date_str: 開始日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
        end_date_str: 終了日時（YYYYMMDD形式またはYYYYMMDDHHMMSS形式）
    """
    print("=== RAデータのみ取得 ===")

    # Ensure correct formats
    if len(start_date_str) == 8:
        start_date_str += "000000"
    if len(end_date_str) == 8:
        end_date_str += "235959"

    START_DATE = start_date_str
    START_DATE_YYYYMMDD = START_DATE[:8]
    END_DATE_YYYYMMDD = end_date_str[:8]

    try:
        # RAデータのみを取得するタスク
        task = {
            "dataspec": "RACE",
            "option": 4,
            "start_date": START_DATE,
            "target_ids": ["RA"],
        }

        # クライアント初期化
        client = JRAVANClient()
        try:
            client.login()
        except:
            print("Login failed, aborting.")
            if client:
                client.close()
            return

        ds = task["dataspec"]
        opt = task["option"]
        targets = task["target_ids"]
        task_start_date = task["start_date"]

        print(f"\n--- Processing {ds} (RA only) ---")

        # 出力ディレクトリの準備
        script_dir = Path(__file__).parent.parent.parent
        output_dir = script_dir / "data" / "output"
        output_dir = str(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time.sleep(2)

        # 既存データの日付キーを読み込む（サブディレクトリ内）
        existing_data_keys = {}
        for target_id in targets:
            # サブディレクトリのパスを構築
            subdir = f"race_{target_id.lower()}"
            fname = f"race_{target_id.lower()}.csv"

            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path, exist_ok=True)

            fpath = os.path.join(subdir_path, fname)
            existing_keys = load_existing_dates(fpath, target_id)
            existing_data_keys[target_id] = existing_keys
            if existing_keys:
                print(f"  {target_id}: 既存データ {len(existing_keys)}件を検出")

        records_buffer = {}
        total_count = 0
        skipped_count = 0

        print(f"Requesting {ds} (From: {task_start_date}, Opt: {opt})...")
        pbar = tqdm(
            desc=f"Fetching {ds} (RA only)", unit="chunks", position=0, leave=True
        )

        try:
            for raw_chunk in client.get_data(ds, task_start_date, opt):
                pbar.update(1)
                try:
                    byte_chunk = raw_chunk.encode("cp932", "replace")
                except Exception:
                    byte_chunk = raw_chunk if isinstance(raw_chunk, bytes) else b""

                if not byte_chunk:
                    continue
                lines = raw_chunk.split("\n")

                for line in lines:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    try:
                        line_bytes = line.encode("cp932", "replace")
                    except Exception:
                        continue
                    if len(line_bytes) < 2:
                        continue

                    rec_id = line_bytes[:2].decode("ascii", errors="ignore")
                    if rec_id != "RA":  # RAのみ処理
                        continue

                    if rec_id in SCHEMAS:
                        parsed = parse_fixed_width(line_bytes, SCHEMAS[rec_id])

                        # Data Kubun Filtering
                        if parsed.get("data_kubun") != "7":
                            continue

                        # Date Filtering (strict YYYYMMDD range)
                        rec_date = str(parsed.get("year", "")).strip().zfill(4) + str(
                            parsed.get("month_day", "")
                        ).strip().zfill(4)
                        if len(rec_date) != 8 or not rec_date.isdigit():
                            continue
                        if (
                            rec_date < START_DATE_YYYYMMDD
                            or rec_date > END_DATE_YYYYMMDD
                        ):
                            continue

                        # 既存データチェック（重複スキップ）
                        existing_keys = existing_data_keys.get(rec_id, set())
                        # RAはレース単位のデータ
                        record_key = (
                            str(parsed.get("year", "")),
                            str(parsed.get("month_day", "")),
                            str(parsed.get("course_code", "")),
                            str(parsed.get("kai", "")),
                            str(parsed.get("nichi", "")),
                            str(parsed.get("race_num", "")),
                        )
                        if record_key in existing_keys:
                            skipped_count += 1
                            continue

                        # 新しいデータのみ追加
                        parsed["raw_hex"] = line_bytes.hex()
                        if rec_id not in records_buffer:
                            records_buffer[rec_id] = []
                        records_buffer[rec_id].append(parsed)

                        # 既存キーセットに追加（メモリ内で重複チェック）
                        existing_keys.add(record_key)

                        total_count += 1

                        if len(records_buffer[rec_id]) >= 100000:
                            # サブディレクトリのパスを構築
                            subdir = f"race_{rec_id.lower()}"
                            fname = f"race_{rec_id.lower()}.csv"

                            subdir_path = os.path.join(output_dir, subdir)
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path, exist_ok=True)

                            save_path = os.path.join(subdir_path, fname)
                            fields = get_schema_fieldnames(rec_id) + ["raw_hex"]
                            save_to_csv(records_buffer[rec_id], save_path, fields)
                            records_buffer[rec_id] = []

                # チャンク処理後、進捗バーにレコード数を表示
                pbar.set_postfix(
                    {"new": f"{total_count:,}", "skipped": f"{skipped_count:,}"}
                )

        except Exception as e:
            print(f"!!! Error processing {ds}: {e}")
        finally:
            pbar.close()
            if client:
                client.close()
                client = None

        # 残りのデータを保存（サブディレクトリ内）
        for rid, data_list in records_buffer.items():
            if data_list:
                # サブディレクトリのパスを構築
                subdir = f"race_{rid.lower()}"
                fname = f"race_{rid.lower()}.csv"

                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)

                save_path = os.path.join(subdir_path, fname)
                fields = get_schema_fieldnames(rid) + ["raw_hex"]
                save_to_csv(data_list, save_path, fields)

        if total_count == 0:
            if skipped_count > 0:
                print(
                    f"No new records found for {ds}. ({skipped_count} existing records skipped)"
                )
            else:
                print(f"No relevant records found for {ds}.")
        else:
            print(
                f"  Added {total_count} new records, skipped {skipped_count} existing records for {ds}."
            )

    except Exception as e:
        print(f"\nTerminating due to error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "client" in locals() and client:
            client.close()
        print("\nExiting.")


def run_today_se_ra_and_realtime_merge(
    race_day_yyyymmdd: str | None = None,
    *,
    dual_pass_se_then_ra: bool = True,
    target_kubun: str = "both",
    output_dir: str | None = None,
) -> dict:
    """
    当日（または指定日）の RA/SE を main/data/race に保存し、その後速報系を取得してマージする。

    1. get_race_data … 開催日で絞り、SE 用 JV パス → RA 用 JV パスの順。
    2. refresh_today_realtime_data … WE/WH 速報取得後、RA/SE に反映。
    3. refresh_today_odds_data … 0B31(O1) 単勝オッズを取得し RA/SE に反映。

    Windows 64bit では main.notebook_bootstrap.run_today_se_ra_and_realtime の利用を推奨
    （32bit 子プロセスへ委譲）。
    """
    d = (race_day_yyyymmdd or datetime.now().strftime("%Y%m%d")).strip()
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"race_day_yyyymmdd は YYYYMMDD 8桁: {race_day_yyyymmdd!r}")
    start_s = d + "000000"
    end_s = d + "235959"
    span = get_race_data(
        start_date_str=start_s,
        end_date_str=end_s,
        output_dir=output_dir,
        target_kubun=target_kubun,
    )
    rt = refresh_today_realtime_data(
        start_date_str=start_s,
        end_date_str=end_s,
        output_dir=output_dir,
    )
    odds = refresh_today_odds_data(
        start_date_str=start_s,
        end_date_str=end_s,
        output_dir=output_dir,
    )
    return {"race_day": d, "get_race_span": span, "realtime": rt, "odds": odds}


if __name__ == "__main__":
    s_arg = sys.argv[1] if len(sys.argv) > 1 else "20180101000000"
    e_arg = sys.argv[2] if len(sys.argv) > 2 else "20251231235959"
    fetch_jra_data(s_arg, e_arg)
