# 実装仕様書: Phase 6 — JRA公式タイム指数（TM）特徴量
# 日付: 2026-06-30

---

## 1. 目的

v6（27.8%）に JRA 公式タイム指数（TM: タイム指数）を追加し、28.5% 以上を達成する。
TM は市場バイアスを含まない JRA 発行の走破タイム標準化スコアであり、
既存の速度指数（hist_speed_idx_*）が持つ「同条件の歴史的平均との差」を
JRA 独自の補正で算出した異なる信号を提供する可能性がある。

ベンチマーク: 1番人気 Top-1 ≈ 31%。本 Phase の目標は 28.5%（Phase 7 基準）の突破。

---

## 2. データ確認結果

### 2.1 JV-Link に「JRAマイニング予想スコア」専用レコードは存在しない

調査対象パスの全ファイル一覧:

```
C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\data\01_preprocessed\
  HC_preprocessed.parquet  (25 MB) — 坂路調教
  RA_preprocessed.parquet  (0.6 MB) — レース詳細
  SE_preprocessed.parquet  (8.7 MB) — 成績
  SK_preprocessed.parquet  (0.6 MB) — 血統リンク (ketto_num / sire_id / bms_id のみ)
  WC_preprocessed.parquet  (4.4 MB) — ウッドチップ調教（v6 で既に使用）

C:\Users\syugo\AI\RaceAI_var2.0.0\model_training\data\01_preprocessed\
  TM_preprocessed.parquet  (1.1 MB) — JRA公式タイム指数（未使用）
```

JV-Link のレコード種別として「JRAマイニング予想」という独立したデータ種別は存在しない。
JRA 公式が発行する馬の能力指数は **TM（タイム指数）のみ**。

### 2.2 TM_preprocessed.parquet の詳細

| 項目 | 内容 |
|------|------|
| パス | `C:\Users\syugo\AI\RaceAI_var2.0.0\model_training\data\01_preprocessed\TM_preprocessed.parquet` |
| レコード数 | 533,400 行 |
| 列数 | 3 列 |
| 対象期間 | 2015年〜2026年 |
| ユニーク race_id | 37,906 レース |

| 列名 | 型 | NaN率 | 説明 |
|------|-----|-------|------|
| race_id | object | 0.0% | レースID (SE の race_id と対応) |
| horse_num | int32 | 0.0% | 馬番 (SE の horse_num と対応) |
| tm_score | float32 | 0.2% | JRA公式タイム指数 |

**tm_score の統計**: min=5.6, max=93.3, mean=49.9

### 2.3 市場情報混入チェック — 合格

TM（タイム指数）は走破タイムを距離・馬場状態・コースで補正した客観的な時間ベース指数であり、
オッズ・人気・市場情報に依存しない。CLAUDE.md 第1条（市場情報排除）に抵触しない。

禁止列チェック:
```
grep -n "odds\|popularity\|market_log_odds\|init_score" TM_preprocessed の列名
→ ヒットなし（race_id / horse_num / tm_score の 3 列のみ）
```

### 2.4 既存 hist_speed_idx_* との関係

v6 は `_build_speed_index_features()` で以下を既に実装済み:

| 特徴量 | 説明 |
|--------|------|
| hist_speed_idx_last | 前走の速度指数（当プロジェクト計算） |
| hist_speed_idx_avg3 | 直近3走の速度指数平均 |
| hist_speed_idx_best | 過去最高速度指数 |
| hist_speed_idx_cond_best | 同距離帯×馬場での最高速度指数 |

計算式: `(cond_avg_time_prev - racetime) / cond_std_time_prev`（shift(1)済み累積統計）

TM の tm_score との相関は 0.6〜0.8 と推定される（同じ走破タイムを標準化しているが、
JRA の補正アルゴリズムが異なるため完全な重複にはならない）。
相関が 0.8 以下であれば追加の情報を持ち、+0.3〜+0.8 pp の改善が期待できる。

---

## 3. 特徴量設計

### 3.1 TM 特徴量一覧

| 特徴量名 | 計算方法 | リーク防止 | 期待効果 | 優先度 |
|---------|---------|-----------|--------|-------|
| hist_tm_last | 前走の tm_score（shift(1)） | ○ shift済 | 高 | 高 |
| hist_tm_avg3 | 直近3走の tm_score 平均（shift(1)+rolling(3)） | ○ shift済 | 高 | 高 |
| hist_tm_best | 過去最高 tm_score（shift(1)+expanding().max()） | ○ shift済 | 中 | 高 |
| hist_tm_cond_best | 同距離帯×surface_code での過去最高 tm_score | ○ shift済 | 中 | 中 |
| hist_tm_trend | (hist_tm_last - hist_tm_avg3)（TM の上昇/下降トレンド） | ○（上記から派生） | 低〜中 | 低 |

**実装スコープ**: 第1弾は `hist_tm_last / hist_tm_avg3 / hist_tm_best` の3列。
`hist_tm_cond_best` と `hist_tm_trend` は第1弾の評価後に追加判断。

### 3.2 実装上の制約（重要）

**TM は horse_num で識別される（ketto_num ではない）**

SE にも horse_num 列が存在する（int8, NaN=0.0%）。
以下の手順で ketto_num に紐付ける:

```python
# Step 1: SE から race_id × horse_num → ketto_num の対応テーブルを作成
id_map = df[["race_id", "horse_num", "ketto_num", "race_date"]].copy()

# Step 2: TM を id_map にマージして ketto_num を付与
tm_mapped = tm.merge(id_map, on=["race_id", "horse_num"], how="left")

# Step 3: ketto_num × race_date でソート後に shift(1)
tm_mapped = tm_mapped.sort_values(["ketto_num", "race_date"])
grp = tm_mapped.groupby("ketto_num")
tm_mapped["hist_tm_last"] = grp["tm_score"].transform(lambda x: x.shift(1))
tm_mapped["hist_tm_avg3"] = grp["tm_score"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
tm_mapped["hist_tm_best"] = grp["tm_score"].transform(
    lambda x: x.shift(1).expanding().max()
)

# Step 4: df に ketto_num × race_id でマージ
df = df.merge(
    tm_mapped[["ketto_num", "race_id", "hist_tm_last", "hist_tm_avg3", "hist_tm_best"]],
    on=["ketto_num", "race_id"], how="left"
)
```

### 3.3 データファイルの配置

TM ファイルは現在 var2.0.0 にのみ存在する。以下のどちらかで対処する:

**推奨方法: train_config.json に参照パスを追加**
```json
{
  "data": {
    "tm_parquet": "C:/Users/syugo/AI/RaceAI_var2.0.0/model_training/data/01_preprocessed/TM_preprocessed.parquet"
  }
}
```
（コピーではなく参照でよい。var1.0 と var2.0.0 で同一ファイルを共有）

### 3.4 NaN 率の見通し

| 特徴量 | 予想 NaN 率 | 理由 |
|--------|----------|------|
| hist_tm_last | 35〜45% | 初出走・TM記録なし・出走取消歴など |
| hist_tm_avg3 | 35〜45% | hist_tm_last と同等 |
| hist_tm_best | 35〜45% | hist_tm_last と同等 |

NaN が多い場合、LightGBM の欠損値分岐が処理する（追加対処不要）。
ただし NaN 率が 70% を超える場合は情報量が少なすぎるため、
その特徴量の追加は evaluator の判断を待つ。

---

## 4. 期待改善量の根拠

### 4.1 保守的シナリオ（相関が高い場合: r = 0.75〜0.85）

hist_speed_idx_* と TM の相関が 0.8 前後の場合、追加情報は限定的。
LightGBM は冗長な特徴量を自然に低重要度として扱うため、
精度低下リスクは低いが、改善も小さい。

期待: **v6_TM ≈ 27.9〜28.2%**（+0.1〜+0.4 pp）

### 4.2 楽観的シナリオ（相関が中程度: r = 0.55〜0.70）

JRA の標準化アルゴリズムが独自の補正（コース特性・馬場作り・馬体差等）を持つ場合、
我々の速度指数では捉えられない信号を持つ可能性がある。

期待: **v6_TM ≈ 28.3〜28.8%**（+0.5〜+1.0 pp）

### 4.3 Phase 6 の目標設定

TM 特徴量単独で 28.5% 達成が難しい可能性がある。
v6 改善仕様書（2026-06-30-v6-improvement-plan.md）と並行実施することで、
相補的に目標値を突破する戦略を推奨する。

---

## 5. 実装優先順位

| 優先度 | 実施内容 | 期待効果 |
|--------|---------|--------|
| 高 | hist_tm_last / hist_tm_avg3 / hist_tm_best を追加（TM 3列） | +0.3〜+0.8 pp |
| 中 | hist_tm_cond_best を追加（同条件TM最高値） | +0.1〜0.2 pp |
| 低 | hist_tm_trend を追加（TM トレンド） | +0.1 pp 以下 |

**注意**: hist_speed_idx_* との相関を evaluator が確認した後に、
不要な特徴量削除（高相関 r>0.9 の場合）を refactorer に依頼する。

---

## 6. 禁止事項確認

- [x] オッズ系データを一切含まないことを確認した（tm_score は時間ベース）
- [x] 人気順位を含まないことを確認した
- [x] shift(1) でリーク防止を実施する設計になっていることを確認した
- [x] 当該レースの tm_score を当該レースの予測に使わない（過去走のみ）

---

## 7. implementer への引き渡し事項

### 必須タスク

1. `create_features.py` に `_build_tm_features(df, tm_path)` 関数を追加する
   - 配置場所: SECTION 5.6（speed index）の直後、SECTION 6（training）の前
   - 引数: `df`（SE+RA+SK マージ済み）、`tm_path`（TM parquet のパス）

2. `train_config.json` に `"tm_parquet"` キーを追加する
   ```json
   "tm_parquet": "C:/Users/syugo/AI/RaceAI_var2.0.0/model_training/data/01_preprocessed/TM_preprocessed.parquet"
   ```

3. `main()` に `_build_tm_features` の呼び出しを追加する（ステップ 5.7 として）

4. `features_version` を `v3cj` → `v7` に更新する（features_v7.parquet として出力）

5. 実装後、以下の禁止チェックを実行すること:
   ```
   grep -rn "odds\|popularity\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
   ```

### 確認事項（実装前）

- TM の horse_num と SE の horse_num が同一 race_id 内で一致するか確認する
  （SE: horse_num int8, TM: horse_num int32 — merge 時に dtype 統一が必要）
- TM の NaN 率が 70% を超える場合は evaluator に報告する

### 評価後の確認ポイント

- hist_tm_last と hist_speed_idx_last の相関係数（Pearson r）を報告する
- r > 0.9 の場合: hist_tm_last を削除し hist_speed_idx_* のみ維持することを evaluator と協議
- r < 0.9 の場合: 両方を維持する

---

## 8. リーク停止閾値（再掲）

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止し evaluator に報告
```
