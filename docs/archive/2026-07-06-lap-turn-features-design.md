# 実装仕様書: v48_agari_turn — 上がり3F偏差・回り×馬場適性 — 2026-07-06

作成: implementer（ユーザー Draft を正式化）  
ベース: `v39_course_slim`（132列、本番凍結）  
実験版: `v48_agari_turn`（134列 = 132 + 2）

---

## 0. 禁止事項の確認

- [ ] オッズ・人気・市場情報を特徴量に使わない
- [ ] テストデータ（2025+）で閾値・特徴量を後付け調整しない
- [ ] Top-1 > 40% または Spearman > 0.6 → 即停止・evaluator 報告
- [ ] 実装前後: `grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"`
- [ ] `features_v39_course_slim.parquet` / 本番 models は上書きしない

---

## 1. 目的

レース×馬の一括テーブル（1行 = 1頭）上で、過去走から集約した以下2列を追加し LambdaRank の精度改善を検証する。

| 列名 | 定義 |
|------|------|
| `hist_last_agari_time_gap` | 前走の `(上がり3Fレース内偏差 − 走破タイムレース内偏差)`（shift(1)） |
| `hist_turn_surface_win_edge` | 同一 `race_type_code` 勝率 − 同 `surface_code` 勝率（回り方向の上乗せ） |

**相関ゲート対応（2026-07-06 実測）**

- 当初案 `hist_avg_agari3f_dev` は `hist_avg_time_dev_5` と r=+0.776 → **不採用**
- 当初案 `hist_same_turn_surface_win_rate` は `hist_win_rate` と r=+0.724 → **不採用**
- 採用: `hist_last_agari_time_gap`（max|r|=0.534）、`hist_turn_surface_win_edge`（max|r|=0.193）

---

## 2. データ構造

- **粒度**: 1行 = 1レース × 1頭（`race_id` + `ketto_num`）
- **グループキー**: `race_id`（LambdaRank）
- **馬ID**: `ketto_num`（仕様書の `horse_id` に相当）

---

## 3. 処理フローとリーク防止

既存 [`pure_rank/src/create_features.py`](../pure_rank/src/create_features.py) のフローを踏襲する。

1. `_load_data`: SE + RA + SK merge（`race_type_code` を RA から merge）
2. `_apply_filters`: 障害・少頭数等を除外
3. `_build_hist_features`:
   - **冒頭で必ず** `sort_values(["ketto_num", "race_date"])`（L280）
   - `_agari_dev` = 当走のレース内上がり3F偏差（他馬含む race mean との差）
   - `shift(1).expanding().mean()` で当走を除外した過去集約のみを特徴量化
   - 一時列 `_agari_dev` は drop（`FORBIDDEN_COLS` に登録）
4. 保存前: `sort_values(["race_date", "race_id", "horse_num"])` で LambdaRank group 順を保証

### 3.1 時系列ソートの担保（実装確認ポイント）

| 段階 | ソートキー | 目的 |
|------|-----------|------|
| `_build_hist_features` 冒頭 | `ketto_num`, `race_date` 昇順 | groupby + shift(1) が「過去走」を指す |
| parquet 保存直前 | `race_date`, `race_id`, `horse_num` | LambdaRank group 配列 |

同一 `race_date` に複数走がある場合は `race_date` 単独では順序が不定になりうるが、**shift(1) は同一馬内の直前行**を指すため、同日複数走馬では rare edge case。v39 以来の既存パターンと同一。

### 3.2 `_agari_dev` のリーク安全性

```
race_avg_agari = mean(time_3f_after) within race_id   # 当走レース内（結果情報）
_agari_dev = time_3f_after - race_avg_agari           # 当走の偏差
hist_avg_agari3f_dev = shift(1).expanding().mean(_agari_dev) per ketto_num
```

予測時点では **前走以前** の `_agari_dev` のみが集約される。`_time_dev` / `hist_avg_time_dev_*` と同型でリークなし。

---

## 4. race_type_code（回り×馬場）

JV-Link 競走種別（平地）:

| code | 意味 |
|------|------|
| 11 | 右回り芝 |
| 12 | 左回り芝 |
| 13 | 右回りダート |
| 14 | 左回りダート |

- ソース: `RA_preprocessed.parquet`（preprocess 済み）
- 特徴量列としては使わず（`FORBIDDEN_COLS`）、hist 集約キーのみ
- 障害 18/19 は平地 11–14 と group 分離

---

## 5. 欠損値

- 新馬・初回条件: `hist_*` = NaN（fillna しない）
- 上がり3F未記録走: `_agari_dev` = NaN → expanding mean も NaN 伝播

---

## 6. 相関ゲート

- 対象期間: `race_date <= 2024-12-31`
- 新列 × 既存特徴量 |r| >= 0.7 → RuntimeError（学習禁止）
- 新列: `hist_avg_agari3f_dev`, `hist_same_turn_surface_win_rate`

---

## 7. 評価

| 指標 | 合格 |
|------|------|
| Top-1 | > 30.24%（v39 ベースライン）かつ <= 40% |
| NDCG@3 | > 0.5359 |
| Spearman | > 0.5048 |
| 相関ゲート | PASS |
| 市場情報 | 混入なし |

コマンド:

```bash
python pure_rank/src/create_features.py
python pure_rank/src/train.py          # 動作確認: 単一 seed
python pure_rank/src/evaluate.py
```

---

## 8. 変更ファイル

| ファイル | 内容 |
|---------|------|
| `pure_rank/src/create_features.py` | merge + hist 2列 + v48 assert + gate |
| `pure_rank/src/common.py` | `_agari_dev` を FORBIDDEN_COLS に追加 |
| `pure_rank/config/train_config.json` | `features_version: v48_agari_turn`（実験ブランチのみ） |

**変更しない**: v39 parquet/models、`preprocess.py`（`race_type_code` 済み）

---

## 9. ブランチ

`feature/v48-agari-turn-surface`
