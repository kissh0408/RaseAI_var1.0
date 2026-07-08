# 実装仕様書: 福島・小倉弱点対策特徴量 — 2026-07-04

作成: planner（Phase E 調査結果に基づく）
入力データ: `pure_rank/data/02_features/course_weakness_v39_course_slim.json`
版系列: **v46_small_pool** / **v47_pace_x**（v45_transport まで使用済み）

---

## 0. 禁止事項の確認

- [x] オッズ・人気・市場情報を一切使わない
- [x] テスト Top-1 を見て閾値・カテゴリ境界を後付け調整しない
- [x] 相関ゲート・NaN 率・センタリング定数 `c` は `race_date <= valid_end` のみで決定
- [ ] リーク停止閾値: Top-1 > 40% または Spearman > 0.6 → 即停止

---

## 1. 調査結果サマリー（course_weakness_v39_course_slim.json）

| 発見 | 数値 | 含意 |
|------|------|------|
| 福島 Top-1 | 22.8%（-7.5pp, n=312） | v33〜v45 で不変の構造弱点 |
| 小倉 Top-1 | 22.5%（-7.8pp, n=383） | 同上 |
| 函館 Top-1 | 40.3%（+10.0pp） | 同じ SMALL_COURSE でも強い — 「小回り=弱い」は不成立 |
| H1: hist_same_course NaN | 弱点場 51.7% vs 他場 38.3%（**+13.4pp**） | **コース実績疎性が最強シグナル** |
| H3: 外れレースの先行1着 | 538 miss 中 396 が逃げ/先行（**73.6%**） | 展開密度×脚質の交互作用余地 |
| 福島ダート短距離 | surface=2 × dist=0: Top-1=18.8%（n=48） | 特定セグメント集中だが n 小 |
| 小倉ミス | pred_outside_top3=167/297（56%） | 穴馬見落とし比率高（福島 140/241=58%） |

### 仮説の棄却/採択

| 仮説 | 判定 | 根拠 |
|------|------|------|
| H1 コース実績疎性 | **採択 → v46** | delta_nan=+13.4pp、miss 時 pred/actual winner の same_course NaN 率 48〜60% |
| H2 枠×小回り | 保留 → v48 候補 | train期間の枠勝率分布は平坦（内枠優位は弱い） |
| H3 展開密度×脚質 | **採択 → v47** | 73.6% front-runner wins in misses |
| H4 福島≠小倉 | 部分採択 | 同一弱点だが NaN パターンが異なる（福島 58% vs 小倉 39%） |
| H5 距離帯特異 | 棄却 | 全距離帯で弱点、特定距離のみではない |

---

## 2. 実験1: v46_small_pool

### 特徴量: `hist_small_course_pool_win_rate_ts`

**定義**: 馬の過去走のうち `course_code ∈ {1,2,3,10}`（SMALL_COURSE_CODES）でのみ
勝敗を集計した時系列勝率。`shift(1)` で当該レースを除外。

```python
df["_small_course_win"] = np.where(
    df["course_code"].isin(SMALL_COURSE_CODES), df["is_win"], np.nan
)
df["hist_small_course_pool_win_rate_ts"] = (
    df.groupby("ketto_num")["_small_course_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
)
```

**設計根拠**:
- D-1 で `hist_track_size_win_rate`（size プール）は hist_win_rate と r=0.851 で不採用だったが、
  本列は **course_code 粒度** ではなく **小回り4場プール** であり、
  `hist_same_course_win_rate`（course 単位、NaN 51.7%）の**補完情報**を提供する
- 札幌・函館の好成績（+10pp）の証拠を福島・小倉初出走馬に転用可能

**相関ゲート注意**: `hist_same_course_win_rate` との r を実測。≥ 0.7 なら v46 不採用とし
v48（`post_position_x_small`）へ切替。

**期待改善**: +0.1〜0.3pp（695レース × 部分改善 ≈ 全体 +0.15pp 上限）

---

## 3. 実験2: v47_pace_x（v46 不合格時または v46 合格後）

### 特徴量: `front_pref_x_density`

**定義**: 既存 `field_front_runner_density`（レース内先行密度）を学習+valid期間平均 `c` で
センタリングし、`hist_front_running_pref` と積を取る。

```python
c = df.loc[df["race_date"] <= valid_end, "field_front_runner_density"].mean()
df["front_pref_x_density"] = (
    df["hist_front_running_pref"] * (df["field_front_runner_density"] - c)
)
```

**v41_pace との差分**:
- v41 は自馬除外密度（`_build_pace_interaction_features`）で r=0.774 ゲートNG
- v47 は既存 `field_front_runner_density`（全馬平均）を使用 — 相関構造が異なる
- 新馬（pref NaN）は NaN のまま fillna しない

**期待改善**: +0.05〜0.2pp

---

## 4. 実験3（条件付き）: v48_handicap_x

v46/v47 不合格かつハンデ弱点が残る場合のみ:
- `handicap_x_win_rate_vs_field` = `is_handicap_race × win_rate_vs_field`
- v44 で単体 `is_handicap_race` は gain=0、ハンデ Top-1=22.7%（-7.5pp）

---

## 5. 評価基準（v39_course_slim 比）

| 指標 | 合格 |
|------|------|
| Top-1 | > 30.24% |
| NDCG@3 | ≥ 0.5329 |
| Spearman | ≥ 0.4998 |

副次（合否に不使用）: 福島+小倉 Top-1 合計 +2pp 以上

---

## 6. 実装手順

1. モデル15個を `pure_rank/models_backup_v39_before_v46/` に退避
2. `train_config.json` の `features_version` を `v46_small_pool` に変更
3. `create_features.py` 実行 → 相関ゲート PASS 確認
4. `train.py --ensemble` → `evaluate.py` → `analyze_errors.py`
5. evaluator 合否判定
6. 不合格: ロールバック → v47 へ

---

## 7. 変更ファイル

| ファイル | 変更 |
|---------|------|
| `pure_rank/src/create_features.py` | v46/v47 列生成、`NEW_FEATURE_COLS_BY_VERSION` 追記 |
| `pure_rank/config/train_config.json` | `features_version` 切替 |

---

## 8. Phase E 実験結果（2026-07-04 implementer + evaluator）

| 版 | 特徴量 | 相関ゲート | Top-1 | NDCG@3 | Spearman | 福島 | 小倉 | 判定 |
|----|--------|-----------|-------|--------|----------|------|------|------|
| v39（baseline） | — | — | **30.24%** | 0.5359 | 0.5048 | 22.8% | 22.5% | 採用維持 |
| v46_small_pool | hist_small_course_pool_win_rate_ts | PASS (max r=0.58) | 30.14% | 0.5359 | 0.5049 | 22.8% | 23.0% | **不合格** |
| v47_pace_x | front_pref_x_density | **NG** (r=0.78 vs density) | — | — | — | — | — | 学習未実施 |
| v47_post_small | post_position_x_small | PASS (max r=0.64) | 30.03% | 0.5348 | 0.5052 | 22.4% | 23.0% | **不合格** |

**結論**: 福島・小倉の -7pp 弱点は v46/v47 でも解消せず。正式ベースライン v39_course_slim をロールバック確認済み（Top-1=30.24% バイト一致）。

**ROI 並行タスク**:
- R-7-lite v39 再実行: VALID 6条件 → TEST 全条件 ROI<100%（OR 複合 75.6%）→ 条件フィルタ凍結確認
- R-1 正式判定: ロードマップ §7 に記録済み
- `export_scores.py` 初版: `exported_scores/scores_v39_course_slim_test.parquet`（66,020行）

**次の方向性**: R-6（var2.0.0 残差統合）最優先。ランキング側は福島・小倉以外の軸（多頭数展開・ハンデ×能力交互作用）またはモデリング変更が必要。

