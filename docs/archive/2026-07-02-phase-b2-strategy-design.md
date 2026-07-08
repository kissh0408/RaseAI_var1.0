# 実装仕様書: Phase B2 次戦略設計 — 2026-07-02

## 目的

特徴量飽和後（v33_jt_ext、Top-1=30.37%、131列）の次フェーズ戦略を策定する。
Phase A の全6候補が失敗した原因を整理し、真に独立した信号を持つ候補を優先順位付けする。

ベンチマーク（市場1番人気 Top-1≈31%）まで残り 0.6pp。この領域では 1 候補ずつ
丁寧に効果を検証する。

## 禁止特徴量の確認

- [x] オッズ・人気を含む候補を一切検討しない
- [x] 人気順位を含まない
- [x] テストデータを見て後付け選択しない（エラー分析の結果は特徴量設計の根拠とするが、
      threshold を事後調整するために使わない）

---

## Phase A 失敗の構造的分析

Phase A の全6候補が失敗した共通原因：

| 失敗パターン | 代表候補 | 既存特徴量との重複 |
|------------|---------|-----------------|
| 脚質系の飽和 | running_style 過去平均 / corner4 通過順位 | hist_front_running_pref (r=0.74〜0.79) |
| 細分化してもシグナルなし | 上がり3F最速率 | field_z_last3f (r=0.74) |
| 生値で弱い信号は変換しても弱い | 斤量負担率 | burden_weight との Spearman rho=0.034 |
| 信号の粒度が粗い | 距離差 | hist_same_dist_win_rate と重複、かつ弱い |

**結論**: v33_jt_ext は脚質・位置取り・斤量・距離差の信号を既に内包済みで飽和している。
次の候補は「完全に直交するシグナル源」を持つものに限定する。

---

## 候補評価

### 候補 A: 馬×コース×距離 特化勝率 (`hist_same_course_dist_win_rate`)

**評価結果: 実装不要**

`hist_same_course_dist_win_rate` は v33_jt_ext に**既に存在する**（131列中に確認済み）。

```python
# v33_jt_ext の列リストで確認
'hist_same_course_dist_win_rate'  # EXISTS
```

実装コスト: 不要 / 重複リスク: 既実装 / 期待改善: なし

---

### 候補 B: 血統×馬場状態 交互作用 (`hist_sire_track_condition_win_rate_ts`)

**評価結果: 新規シグナル。実装を推奨**

v33_jt_ext での存在確認：

```
hist_sire_surface_win_rate_ts      → EXISTS（芝/ダート別）
hist_sire_track_condition_win_rate_ts → MISSING
```

これら2つは**異なる次元**を測定している：

| 特徴量 | 測定する次元 | 値の範囲 |
|--------|------------|---------|
| hist_sire_surface_win_rate_ts | 芝 vs ダート | surface_code ∈ {1, 2} |
| hist_sire_track_condition_win_rate_ts | 良/稍重/重/不良 | track_condition_code ∈ {1,2,3,4} |

芝ダートの適性を測定できていても、馬場悪化への父系統的な対応力は別情報である。
特に不良馬場（81レース）・重馬場（296レース）で父産駒の実績が独自シグナルになりうる。

**実装方法**（`hist_sire_surface_win_rate_ts` と同じパターン）：

```python
def _build_sire_track_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """父×馬場状態別の産駒勝率（時系列安全）。"""
    df = df.sort_values(["race_date", "race_id", "horse_num"])

    # sire_id × track_condition_code での時系列累積勝率
    df["_sire_tc_win"] = df["is_win"]
    sire_tc = (
        df.groupby(["sire_id", "track_condition_code"])["_sire_tc_win"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["hist_sire_track_condition_win_rate_ts"] = sire_tc
    df.drop(columns=["_sire_tc_win"], inplace=True)
    return df
```

注意: `track_condition_code=0`（不明）はリーク防止上学習に含めるが、
NaN 扱いで問題ない（LightGBM が欠損を自動処理する）。

**実装コスト**: 低（hist_sire_surface_win_rate_ts の実装コピーでほぼ完成）
**重複リスク**: 低（surface vs track_condition は直交する次元）
**期待改善**: 0〜0.3pp（エラー分析で不良・重馬場での弱点が確認できれば効果増）
**判断ルール**: エラー分析で track_condition 別 Top-1 の格差が 5pp 以上ある場合に優先度「高」に昇格する。

---

### 候補 C: 斤量のフィールド相対化 (`weight_vs_field`)

**評価結果: 実装しない**

v33_jt_ext での存在確認：

```
burden_weight      → EXISTS（生値）
weight_vs_field    → MISSING
```

しかし実装しない理由が2点ある：

1. **過去の斤量信号の一貫した弱さ**: burden_weight_ratio（v16）で Spearman rho=0.034。
   Phase A の「斤量負担率」でも -0.17pp。斤量シグナル自体が弱い。
   生値を相対化しても弱いシグナルが強くなる保証はない。

2. **burden_weight の特性**: 競馬の斤量格差は最大 3〜4kg（57kg vs 54kg）で変動幅が小さく、
   ランキング内での相対化をしても情報量が増えにくい。

「斤量の相対化」は試みる価値があるが、他の候補より優先度は低い。
エラー分析で特定条件（例: 57kg超の斤量格差が大きいレース）での弱点が確認された
場合のみ再検討する。

**実装コスト**: 低 / 重複リスク: 低（ただしシグナル自体が弱い）/ 期待改善: 0.1pp未満

---

### 候補 D: 馬体重変化 (`horse_weight_change`)

**評価結果: 実装不要**

v33_jt_ext に**既に存在する**（131列中に確認済み）：

```
horse_weight_change  → EXISTS
hist_weight_change   → EXISTS
```

実装コスト: 不要 / 期待改善: なし

---

## モデルアーキテクチャ変更の検討

### 候補 E: label_gain 調整

**現行**: `[0, 1, 3, 7, 15, 31, 100]`

label_gain のインデックスは lr_label 値に対応する：

| lr_label | 着順 | 現行 gain | 試案1 | 試案2 |
|----------|------|----------|-------|-------|
| 6 | 1着 | **100** | **100** | **100** |
| 5 | 2着 | 31 | 16 | 31 |
| 4 | 3着 | 15 | 6 | 8 |
| 3 | 4着 | 7 | 2 | 3 |
| 2 | 5着 | 3 | 1 | 1 |
| 1 | 6着 | 1 | 0 | 0 |
| 0 | 7着以下 | 0 | 0 | 0 |

**試案1**: `[0, 0, 1, 2, 6, 16, 100]` — 1着への集中強化
- 1着/2着 比率: 100/16 = 6.25（現行 100/31=3.23 の 2倍）
- 2着以下の重みを削り、1着の識別に特化させる
- リスク: Top-3 的中率・NDCG@3 が低下する可能性がある

**試案2**: `[0, 1, 3, 7, 15, 63, 127]`（ユーザー提示案）— 1着・2着の重みを増加
- 1着/2着 比率: 127/63 = 2.02（現行より**低い**）
- これは 1着より 2着の重みを大きく増やす方向。Top-1 向けには不適
- **非推奨**: 1着特化という目的と逆方向

**推奨**: 試案1 のみを試験する。ただし以下の前提条件を設ける：
1. エラー分析を先に実施する
2. 候補 B（hist_sire_track_condition_win_rate_ts）を先に試験する
3. 両方が効果なしと判明した場合の第3手として試す

**実装コスト**: 低（train_config.json の label_gain を変更するだけ）
**期待改善**: 0〜0.4pp（v29_fixed → A3 の +0.3pp という前例あり）
**リスク**: Top-3・NDCG@3 の低下を伴いながら Top-1 だけ改善する可能性がある

---

### 候補 F: 条件別サブモデル

**評価結果: 現時点では非推奨**

弱い条件（例: 不良馬場）に特化した LambdaRank を追加する案：

- **実装コスト**: 高（train.py / evaluate.py の改修、モデル管理の複雑化）
- **リスク**: 条件フィルタ後のサンプル数が小さい（不良馬場 81レース）のでテスト精度が不安定
- **判断**: エラー分析で「特定条件での Top-1 が全体より 10pp 以上低い」が判明した場合に限り検討する

現時点では候補 B と候補 E（試案1）を試してから再評価する。

---

## 実装優先順位

```
優先度 1 [即時]: エラー分析スクリプト実行
  → analyze_errors.py を実装して error_analysis_v33.json を生成
  → リスク: なし（既存モデルを使うだけ）
  → 目的: データ駆動で候補 B・E の優先順位を確定する

優先度 2 [エラー分析後]: 候補 B の実装
  → hist_sire_track_condition_win_rate_ts を create_features.py に追加
  → 実装コスト: 低（hist_sire_surface_win_rate_ts のパターンを流用）
  → トリガー条件: track_condition 別格差が 5pp 以上 OR 無条件で試験

優先度 3 [候補 B 不採用後]: 候補 E 試案1（label_gain 調整）
  → train_config.json の label_gain を [0,0,1,2,6,16,100] に変更
  → 実装コスト: 極低（設定値変更のみ）
  → 注意: Top-1 改善を優先するため Top-3・NDCG@3 が低下しても許容する

優先度 4 [候補 B・E 両方不採用後]: 候補 C（weight_vs_field）
  → 期待値は低いが試験コストも低い

優先度 5 [長期検討]: 候補 F（条件別サブモデル）
  → エラー分析で 10pp 以上の条件格差が確認された場合のみ
```

---

## 評価基準（全候補共通）

| 指標 | 合格 | 要改善 | 不合格 |
|------|------|--------|--------|
| Top-1 的中率 | >30.37%（v33超え） | 30.0〜30.37% | <30.0% |
| NDCG@3 | 現行値以上 | ±0.001 | <現行値-0.003 |
| Spearman | 現行値以上 | ±0.002 | <現行値-0.005 |

リーク停止閾値: Top-1 > 40% または Spearman > 0.6 の場合は即座に実装停止して evaluator に報告。

---

## implementerへの引き渡し事項

### ステップ 1（即時）
`analyze_errors.py` の実装・実行（仕様書: `2026-07-02-error-analysis-design.md` 参照）

### ステップ 2（エラー分析完了後）
evaluator が error_analysis_v33.json の結果を確認し planner に報告した後、
planner が候補 B の実装指示を発行する。

候補 B を実装する場合:
- `create_features.py` の `_build_sire_features` 関数に `hist_sire_track_condition_win_rate_ts` を追加
- `features_v34_sire_tc.parquet` として保存（v33 バックアップ後）
- 学習後に evaluate.py で合否判定
- `train_config.json` の `features_version` は実装後に `v34_sire_tc` に変更

### ステップ 3（候補 B の結果確定後）
結果によって候補 E または 候補 C に進む。
