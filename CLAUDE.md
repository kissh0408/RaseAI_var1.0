# RaceAI_var1.0 — Claude Code ルールブック

## プロジェクト概要

**市場情報（オッズ・人気）を一切使わない**純粋能力ベースの競馬着順予想AI。
JV-Linkから取得した馬の能力・血統・状態データのみで着順を予測する。

**目標**: 純粋能力評価で1番人気の的中率（Top-1≈30〜33%）を超える。  
**主モデル**: LightGBM LambdaRank（着順を直接最適化）  
**比較ベースライン**: LightGBM Binary（市場情報なし）  
**評価軸**: ROIではなく Top-1的中率・NDCG@3・Spearman相関

---

## プロジェクト憲法（全エージェント・全作業に適用）

### 1. 市場情報排除（最重要）

以下を**特徴量として絶対に使用しない**：

| 禁止データ | 禁止理由 |
|-----------|---------|
| 単勝オッズ・複勝オッズ | 市場の集合知。排除対象 |
| 馬連・ワイド・三連複オッズ | 同上 |
| 人気順位 | オッズから導出される市場情報 |
| 前日オッズ・当日オッズ変動 | 同上 |
| `market_log_odds` / `market_prob` | RaceAI_var2.0.0の残差学習で使う変数。このプロジェクトでは禁止 |
| `init_score`（市場オッズ由来） | 同上 |

実装前後に必ず確認：
```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

### 2. 時系列リーク防止

当該レースの情報を当該レースの特徴量に使わない。

```python
# ❌ 禁止：全データで集計
df['horse_win_rate'] = df.groupby('horse_id')['is_win'].transform('mean')

# ✅ 必須：shift(1)で当該レースを除外
df['horse_win_rate'] = (
    df.sort_values('race_date')
    .groupby('horse_id')['is_win']
    .transform(lambda x: x.shift(1).expanding().mean())
)
```

### 3. 後出しじゃんけん禁止

- テストデータの結果を見て特徴量・閾値・パラメータを後付けで調整しない
- 学習期間データの分析のみで設計を決定する

### 4. リーク停止閾値

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止・evaluatorへ報告
```

これを超える精度はデータリークの強い疑いがあり、合格ではなく危険信号として扱う。

---

## 5エージェント・アーキテクチャ

このプロジェクトは以下の5エージェントで開発する。**役割分担を守ること。**

| エージェント | 担当 | やらないこと |
|------------|------|------------|
| `planner` | Phase設計・特徴量仕様・評価基準策定 | コード実装・パラメータ設定 |
| `implementer` | データパイプライン・特徴量実装・LambdaRank学習 | 評価判定・仕様策定 |
| `evaluator` | Top-N/NDCG/Spearman計算・合否判定・差し戻し指示 | パラメータ調整・特徴量追加 |
| `refactorer` | デッドコード除去・市場情報混入チェック・バグ修正 | 機能追加・精度変更を伴う変更 |
| `orchestrator` | Phase管理・エージェント調整・状態把握 | 直接実装・直接評価 |

### 標準ワークフロー（Phase サイクル）

```
===========================================================
【Phase開始】
===========================================================
1. planner      → 特徴量仕様書・タスクリスト作成
        ↓ 仕様確定（市場情報不使用を確認）

===========================================================
【実装】
===========================================================
2. implementer  → 特徴量生成・LambdaRank学習・スクリプト実行
        ↓ 実装完了（コード・モデル・実行ログ）

===========================================================
【評価】
===========================================================
3. evaluator    → 市場情報混入チェック → Top-N/NDCG/Spearman計算 → 合否判定
        ↓
   [不合格・実装の問題] → implementerへ差し戻し → 再評価
   [不合格・仕様の問題] → plannerへ差し戻し → 再実装 → 再評価
        ↓ [合格]

===========================================================
【Phase完了 → 定期メンテ】
===========================================================
4. refactorer   → 市場情報チェック + デッドコード除去 + バグ修正
5. evaluator    → リファクタリング後の精度一致確認
```

---

## ディレクトリ構造

```
C:\Users\syugo\AI\RaceAI_var1.0\
├── .claude/
│   └── agents/               # 5エージェント定義
│       ├── planner.md
│       ├── implementer.md
│       ├── evaluator.md
│       ├── refactorer.md
│       └── orchestrator.md
├── pure_rank/
│   ├── config/
│   │   └── train_config.json # 学習設定（一元管理。ハードコード禁止）
│   ├── data/
│   │   ├── 01_preprocessed/  # 前処理済みParquet（SE/RA/HC/PED/TM）
│   │   └── 02_features/      # 特徴量Parquet + manifest.json
│   ├── models/               # 学習済みモデル（lambdarank_fold*_seed*.txt）
│   └── src/
│       ├── create_features.py
│       ├── train.py
│       └── evaluate.py
├── common/
│   └── data/src/             # JV-Link データ取得（既存参照可）
├── docs/
│   └── specs/                # Phase別仕様書
├── CLAUDE.md                 # このファイル
└── .gitignore
```

---

## Phase ロードマップ

| Phase | 追加特徴量カテゴリ | 目標 Top-1 | 状態 |
|-------|----------------|-----------|------|
| Phase 1 | 過去走成績ベースライン（着順・上がり3F・走破タイム） | >25% | 参考実績あり |
| Phase 2 | レース条件（course_code/weather_code/track_condition） | >27% | — |
| Phase 3 | 血統（PED: 父適性・母父・ニックス） | >28% | — |
| Phase 4 | 騎手・調教師（直近30日成績・コース適性） | >29% | — |
| Phase 5 | TMタイム指数・通算賞金 | >30% | — |
| Phase 6 | JRAマイニング予想追加（あり/なし比較） | >32% | **承認後のみ** |

**参照ベースライン（Phase 7実績）**: Top-1=28.5% / NDCG@3=0.497 / Spearman=0.489  
新しい実装はこの値を上回ることが最低条件。

---

## 評価基準

### 合否判定

| 指標 | 合格 | 要改善 | 不合格 |
|------|------|--------|--------|
| Top-1 的中率 | >30% | 28〜30% | <28%（Phase 7基準割れ） |
| Top-3 的中率 | >55% | 52〜55% | <52% |
| NDCG@3 | >0.52 | 0.50〜0.52 | <0.50 |
| Spearman相関 | >0.50 | 0.47〜0.50 | <0.47 |
| テスト件数 | 500レース以上 | 200〜500 | 200未満（判定保留） |

> ベンチマーク（市場）: 1番人気 Top-1≈30〜33%、Top-3≈60〜65%

---

## 確定アーキテクチャ

### LambdaRank（主モデル）

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "label_gain": [0, 1, 3, 7, 15, 31, 63],  # 指数的重み（上位馬重視）
    "num_leaves": 31,           # 保守的設定（変更する場合は planner を通す）
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "seed": 42,                 # 5シードアンサンブル: 42〜46
}
```

**`init_score` は使わない**（RaceAI_var2.0.0との根本的な違い）。

### カテゴリ特徴量（必ず指定する）

```python
CATEGORICAL_FEATURES = [
    'surface_code',          # 1=芝, 2=ダート
    'track_condition_code',  # 1=良, 2=稍重, 3=重, 4=不良
    'course_code',           # 競馬場コード
    'weather_code',          # 天候コード
    'distance_category',     # 距離カテゴリ
    'sex_code',              # 性別
    'class_code',            # クラス
]
# lgb.Dataset の categorical_feature に必ず渡す
```

### データ除外条件（必須フィルタ）

```python
df = df[
    (~df['grade_code'].isin([8, 9])) &       # 未格付け・障害を除外
    (~df['abnormal_code'].isin([1, 3, 4])) &  # 取消・除外・落馬を除外
    (df['horse_count'] >= 5) &                # 5頭未満レースを除外
    (df['finish_rank'] > 0)                   # 着順が有効なもののみ
]
```

### 時系列分割

```python
TRAIN_END = '2021-12-31'
VALID_END = '2022-12-31'
# TEST: 2023-01-01以降
```

### 5シードアンサンブル

```python
SEEDS = [42, 43, 44, 45, 46]
FOLDS = 3
# 全シード・全フォールドの予測平均を最終スコアとする
```

---

## 標準コマンド

```bash
# 特徴量生成
python pure_rank/src/create_features.py

# 単一シード学習（動作確認）
python pure_rank/src/train.py

# 5シードアンサンブル学習（本番）
python pure_rank/src/train.py --ensemble

# 精度評価
python pure_rank/src/evaluate.py

# 市場情報混入チェック
grep -rn "odds\|popularity\|market_log_odds\|init_score" pure_rank/src/ --include="*.py"
```

---

## コーディング規約

1. **設定値はハードコードしない** — `pure_rank/config/train_config.json` に集約する
2. **型ヒントを付ける** — 関数シグネチャに `pd.DataFrame`, `np.ndarray` 等を明記する
3. **コメントは「なぜ」を書く** — 「何をしているか」はコードが示す
4. **features_*.parquet を上書きする前に必ずバックアップを作る**
5. **実験は1パラメータずつ変更する** — 複数同時変更は効果の分離が不可能

---

## よくある問題と対処

| 症状 | 根本原因 | 対処エージェント |
|------|---------|----------------|
| Top-1 > 40% | データリーク疑い | evaluator → implementer（shift確認） |
| 学習Top-1 >> テストTop-1 | 過学習 | evaluator → implementer（正則化強化） |
| 特定馬場・コースで低精度 | 条件特徴量不足 | evaluator → planner（仕様追加） |
| group配列エラー（LambdaRank） | 頭数集計のバグ | implementer（group生成修正） |
| categorical_feature 未指定 | 精度低下の原因 | implementer（lgb.Datasetに追加） |
| オッズ・人気の混入 | 誤実装 | refactorer → implementer（即時削除） |

---

## エージェント呼び出しのタイミング

### `@planner` を使う場面

- 新Phaseを開始するとき
- 特定条件での精度が低く、特徴量の追加を検討するとき
- 評価基準・除外条件を変更するとき

```
@planner Phase 3（血統特徴量）の実装仕様書を作成してください。
現在のPhase 2完了時点でTop-1=27.8%です。
```

### `@implementer` を使う場面

- plannerの仕様書に基づいてコードを書くとき
- 特徴量生成スクリプト・学習スクリプトを実行するとき
- バグを修正するとき

```
@implementer plannerが作成した仕様書に従い、
course_code と weather_code の特徴量を create_features.py に追加してください。
```

### `@evaluator` を使う場面

- モデル学習後に精度を評価するとき
- 市場情報の混入チェックを実施するとき
- 過学習・データリークを疑うとき

```
@evaluator phase2モデルのTop-1・NDCG@3・Spearmanを計算し、
Phase 7ベースラインと比較してください。
```

### `@refactorer` を使う場面

- Phaseが完了してコードを整理するとき
- 市場情報の混入チェックをしたいとき
- バグや重複コードを探したいとき

```
@refactorer Phase 2完了後のコード整理をお願いします。
市場情報の混入チェックと重複ロジックの統合を行ってください。
```

### `@orchestrator` を使う場面

- プロジェクト全体の状態を把握したいとき
- 次に何をすべきか判断したいとき
- 複数エージェントを順番に呼び出す必要があるとき

```
@orchestrator 現在の状態を確認して、次のステップを教えてください。
```

---

## 禁止事項（全エージェント）

1. **オッズ・人気を特徴量に使う**（排除の核心）
2. **`init_score` に市場オッズ由来の値を使う**（RaceAI_var2.0.0のアーキテクチャ）
3. **ROI・回収率で合否を判定する**（このプロジェクトの評価軸ではない）
4. **Top-1>40% の結果を「合格」とする**（リーク疑い。必ず診断する）
5. **テストデータで閾値・特徴量を後付け調整する**（後出しじゃんけん禁止）
6. **200レース未満のテストで「有意な改善」と主張する**
7. **Phase 7ベースライン（Top-1=28.5%）を下回る変更をリリースする**
8. **Phase 6（JRAマイニング）をユーザー承認なしに開始する**
9. **`features_*.parquet` をバックアップなしに上書きする**
10. **APIキー・JV-Link認証情報をコードにハードコードする**

---

## 関連プロジェクト

| プロジェクト | 場所 | 概要 |
|------------|------|------|
| **RaceAI_var1.0**（本プロジェクト） | `C:\Users\syugo\AI\RaceAI_var1.0\` | 市場情報なし・純粋能力LambdaRank |
| RaceAI_var2.0.0 | `C:\Users\syugo\AI\RaceAI\` | 市場残差学習（init_score + binary）|
| RaceAI_var3.0 | `C:\Users\syugo\AI\RaceAI_var3.0\` | — |

> このプロジェクトが Top-1>30% を達成した場合、RaceAI_var2.0.0 の残差学習と組み合わせた最終系への発展を検討する。
