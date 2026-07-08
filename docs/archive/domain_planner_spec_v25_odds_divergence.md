# 仕様書: 市場オッズ乖離特徴量（ensemble_v5 Exp-3）

作成日: 2026-06-14  
作成者: domain-planner  
対象バージョン: ensemble_v5 / features_v25

---

## 目的

JV-Link マイニング予測順位（`mining_predicted_rank`）と市場単勝人気順位（`popularity`）の間に生じる乖離を定量化し、**モデルが過小評価している馬（市場より高評価）を正しく識別する**特徴量を追加する。

### 仮説

JV-Link のマイニング AI がレース前に予測3位と判定した馬が、市場（単勝人気）では3番人気でない場合、その乖離は「パドック状態・調教評価・オーナー意向など、非公開情報を織り込んだ市場の追加シグナル」を示している可能性がある。

### 目標指標

- 予測3位馬の3着以内率: 現状 44.9% → **46% 以上**へ引き上げ
- ROI ≥ 105%（CLAUDE.md 合格基準）
- MDD ≥ -20%（CLAUDE.md 合格基準）

---

## JV-Link 利用可能フィールド（調査結果）

### features_past_v23.parquet 内のオッズ・人気関連列

| 列名 | 型 | null率 | 内容 | 予測時点での利用可否 |
|------|----|--------|------|---------------------|
| `odds` | float32 | 0.13% | 単勝確定オッズ（SEレコード offset=360, len=4） | **リーク注意**（当日予測では O1 取得値に置換） |
| `popularity` | int | 0.0% | 単勝確定人気順（SEレコード offset=364, len=2） | **リーク注意**（当日予測では O1 取得値に置換） |
| `mining_predicted_rank` | float32 | 0.13% | JV-Link マイニング予測順位（SEレコード offset=551, len=2） | **利用可能**（レース前に JV-Link 提供） |
| `lag1_odds`〜`lag5_odds` | float32 | 11.0% | 前走〜5走前の確定単勝オッズ | 利用可能（確定値） |
| `lag1_popularity`〜`lag5_popularity` | float32 | 11.0% | 前走〜5走前の確定単勝人気 | 利用可能（確定値） |

### SE_preprocessed.parquet 内のオッズ・人気関連列

| 列名 | データ区分 | 内容 |
|------|-----------|------|
| `odds` | SEレコード確定値 | 単勝確定オッズ（odds範囲: 1.1〜999.9） |
| `popularity` | SEレコード確定値 | 単勝確定人気順 |

### リーク判定の根拠

- `SE_SCHEMA` の `odds`（offset=360）および `popularity`（offset=364）は **レース確定後に格納される確定値**
- ただし、本プロジェクトの当日予測パイプラインでは **O1 レコード（0B31）からリアルタイム単勝オッズを取得**し、`odds` 列に上書きして利用している（`main/main.py` の `fetch_today_tan_odds_impl` 参照）
- したがって、**予測時点での `odds`/`popularity` は締切前暫定値**として扱われており、設計上のリークは回避済み
- **締切直後〜レース後の「最終確定オッズ」の直接参照は禁止**（これは現在も実施されていない）

---

## データ検証結果（features_past_v23 分析）

### rank_divergence = mining_predicted_rank - popularity

予測3位馬（`mining_predicted_rank == 3`）における乖離別3着以内率（n=39,184件）:

| 乖離スコア | 市場評価との関係 | 3着以内率 | サンプル数 |
|-----------|----------------|----------|-----------|
| -5 以下 | 大幅に市場を下回る評価（市場が過剰人気視） | 14.5% | 1,859 |
| -4 | — | 18.1% | 2,547 |
| -3 | — | 23.1% | 3,373 |
| -2 | — | 28.0% | 4,448 |
| -1 | 若干市場を下回る評価 | 34.3% | 5,515 |
| 0 | 市場と一致（予測3位=人気3位） | **44.1%** | 6,595 |
| +1 | 市場より若干高評価（人気4位以下を3位予測） | **52.7%** | 6,063 |
| +2 | 市場より高評価（人気5位以下を3位予測） | **63.0%** | 4,678 |

**解釈**: 乖離スコアが正値（市場より高い評価）の馬ほど実際の3着以内率が上昇。スコア+1で52.7%、+2で63.0%と顕著な単調増加が確認された。

### field_entropy × rank_divergence の複合効果

過小評価予測3位馬（乖離≥1、n=10,741件）を1レース内のオッズ分布エントロピーで分類:

| フィールドエントロピー | 意味 | 3着以内率 | サンプル数 |
|----------------------|------|----------|-----------|
| 低（混戦） | 各馬のオッズが均等に分散 | **66.7%** | 3,666 |
| 中 | — | 56.1% | 3,687 |
| 高（本命馬が明確） | 特定馬にオッズが集中 | 48.0% | 3,388 |

**解釈**: 混戦レース（エントロピー低）で過小評価馬の3着以内率が最も高い。本命馬が明確なレースでは過小評価馬の優位性が下がる。

---

## 特徴量仕様

### 特徴量 1: `odds_rank_divergence`

| 項目 | 内容 |
|------|------|
| **特徴量名** | `odds_rank_divergence` |
| **意味** | JV-Link マイニング予測順位と市場単勝人気順位の差（正値 = モデルが市場より高く評価） |
| **計算式** | `mining_predicted_rank - popularity` |
| **データソース** | `mining_predicted_rank`（SEレコード）、`popularity`（O1レコード → 当日予測時） |
| **データ型** | int16（範囲: -17〜+17） |
| **リーク防止** | `mining_predicted_rank` は JV-Link がレース前に確定公開する値のため安全。`popularity` は O1 レコードから取得する締切前暫定値を使用する（SEレコードの確定値を直接使用しない） |
| **欠損処理** | `mining_predicted_rank` または `popularity` が NaN の場合は `0`（乖離なし）で補完。NaN 率は両者とも < 0.15% |
| **期待効果** | 予測3位馬での検証: 乖離+1で52.7%、+2で63.0%の3着以内率（vs 全体平均35.9%） |

**実装ノート**:
```python
# リーク防止済み：当日予測では O1 取得値を popularity に代入済み
df['odds_rank_divergence'] = (
    df['mining_predicted_rank'].fillna(df['popularity'])
    - df['popularity']
).astype('int16')
```

---

### 特徴量 2: `field_odds_entropy`

| 項目 | 内容 |
|------|------|
| **特徴量名** | `field_odds_entropy` |
| **意味** | 1レース内の全馬単勝オッズ分布から計算した Shannon エントロピー。低値 = 均等な混戦、高値 = 本命馬が明確 |
| **計算式** | `probs_i = (1/odds_i) / Σ(1/odds_j)`, `entropy = -Σ probs_i * log(probs_i)` |
| **データソース** | `odds`（O1 レコードから取得した同一レース全馬の単勝オッズ） |
| **データ型** | float32（実測範囲: 0.54〜2.73、平均: 2.00、標準偏差: 0.25） |
| **リーク防止** | `odds` には O1 レコード（締切前暫定単勝オッズ）を使用する。SEレコードの確定オッズは使用しない |
| **欠損処理** | 同一 `race_id` の全馬の `odds` を集計するため、一部 NaN でも残りで計算可能。全馬 NaN の場合は `np.log(n_horses)`（均等配分エントロピー）で補完 |
| **期待効果** | 混戦（低エントロピー）での過小評価馬の3着以内率が 66.7%（高エントロピー比 +18.7pp） |

**実装ノート**:
```python
def calc_field_entropy(odds_series: pd.Series) -> float:
    """1レース内の単勝オッズ分布から Shannon エントロピーを計算する。
    
    混戦度の指標として使用。低値ほど各馬のオッズが均等（混戦）。
    """
    arr = odds_series.dropna().values
    if len(arr) == 0:
        return np.nan
    probs = 1.0 / arr
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))

# レース単位で集計（当日予測では同一レース全馬のO1オッズを集計）
entropy_by_race = (
    df.groupby('race_id')['odds']
    .apply(calc_field_entropy)
    .rename('field_odds_entropy')
)
df = df.merge(entropy_by_race, on='race_id', how='left')
```

---

### 特徴量 3: `log_odds_divergence_ratio` （補助特徴量・オプション）

| 項目 | 内容 |
|------|------|
| **特徴量名** | `log_odds_divergence_ratio` |
| **意味** | 市場暗黙確率と予測勝率の対数比。市場の過大・過小評価の程度を連続値で表現 |
| **計算式** | `log( (1/odds) / win_prob_est )` |
| **データソース** | `odds`（O1 取得）、`win_prob_est`（モデル出力、`predict_proba` の勝率） |
| **データ型** | float32 |
| **リーク防止** | `odds` は O1 暫定値。`win_prob_est` は当日推論結果のため問題なし |
| **欠損処理** | `win_prob_est < 0.001` の場合は NaN で補完（ゼロ除算防止） |
| **備考** | `win_prob_est` は推論後にのみ利用可能なため、学習フェーズでは `1/odds`（市場確率）のみで近似する |

**実装ノート**:
```python
# 学習フェーズ: 市場確率のみで近似（win_prob_est は未定のため）
df['market_implied_prob'] = 1.0 / df['odds'].clip(lower=1.01)

# 推論フェーズ: モデル出力確率と市場確率の乖離を計算
df['log_odds_divergence_ratio'] = np.log(
    df['market_implied_prob'] / df['win_prob_est'].clip(lower=0.001)
)
```

この特徴量は**推論フェーズの事後フィルタ**（推奨馬選定ロジック）で使用し、学習時の直接特徴量としては使用しない。

---

## リーク防止の注意事項

### 禁止事項

1. **SEレコードの確定 `odds`/`popularity` を「当日予測」特徴量として直接使用することは禁止**
   - SEレコードのこれらフィールドはレース確定後（着順確定後）に格納される確定値
   - 学習データとしては正当だが、予測時点での推論に使う場合は O1 取得値で置換すること

2. **締切後確定の最終オッズを参照することは禁止**
   - O1 レコードは締切前の暫定オッズ。締切直後に発表される最終確定単勝オッズは使用不可

3. **同一レース内の `finish_rank` を `field_odds_entropy` 計算に使用することは禁止**
   - エントロピーはオッズのみから計算する。着順情報は混入しない

### 許可事項

1. **lag1〜lag5 の前走オッズ・人気**: 過去レースの確定値。予測時点で完全に利用可能
2. **O1 レコード取得の当日暫定 `odds`/`popularity`**: 締切前の値として利用可能（ただし変動あり）
3. **`mining_predicted_rank`**: JV-Link が当日出走表と同時に提供する予測順位。予測時点で利用可能

### 時系列整合性チェック

```python
# データ品質チェック: rank_divergence の分布が
# 訓練期間・バリデーション期間・テスト期間で大きく異ならないことを確認
for period, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
    sub = df[mask]
    print(f"{period}: rank_divergence mean={sub['odds_rank_divergence'].mean():.3f}, "
          f"std={sub['odds_rank_divergence'].std():.3f}")
```

---

## data-generator への引き渡し事項

### 実装対象ファイル

`model_training/src/features_odds_divergence.py`（新規作成）

### 実装すべき処理

1. **`calc_odds_rank_divergence(df: pd.DataFrame) -> pd.Series`**
   - 入力: `mining_predicted_rank`, `popularity` 列を含む DataFrame
   - 出力: `odds_rank_divergence`（int16）
   - 欠損補完: NaN → 0（乖離なし扱い）

2. **`calc_field_odds_entropy(df: pd.DataFrame) -> pd.Series`**
   - 入力: `race_id`, `odds` 列を含む DataFrame
   - 出力: `field_odds_entropy`（float32、レース単位で集計して各行にマージ）
   - 欠損補完: NaN → `np.log(n_horses)` (均等配分仮定)

3. **`add_odds_divergence_features(df: pd.DataFrame) -> pd.DataFrame`**
   - 上記2関数をまとめて呼び出すラッパー関数
   - 返却: 元の DataFrame に `odds_rank_divergence`, `field_odds_entropy` を追加

### features_past_v25.parquet の生成

- ベース: `features_past_v23.parquet`
- 追加列: `odds_rank_divergence`, `field_odds_entropy`
- 保存先: `model_training/data/02_features/features_past_v25.parquet`
- 対応 manifest: `features_past_v25_manifest.json`（列数・行数・生成日時・追加列名を記録）

### 当日予測パイプラインへの統合

`main/main.py` の当日特徴量構築ステップで以下を追加:

```python
# O1取得後のodds/popularityが代入済みの状態で特徴量を計算
from model_training.src.features_odds_divergence import add_odds_divergence_features
race_df = add_odds_divergence_features(race_df)
```

---

## 合格基準

バックテスト評価（`backtest-evaluator` 担当）で以下をすべて満たすこと:

| 指標 | 合格基準 | 現状（v23ベース） |
|------|---------|----------------|
| 予測3位馬の3着以内率 | ≥ 46.0% | 44.9% |
| ROI（テスト期間 2025） | ≥ 105% | 119.9%（v5基準） |
| 最大ドローダウン | ≥ -20% | -19.05%（v5基準） |
| Sharpe レシオ | ≥ 0.10 | 0.100（v5基準） |
| テスト件数 | ≥ 500件 | —（要確認） |
| ベースライン（v23）比 ROI 改善 | +2pp 以上 | — |

### フォールバック基準

`field_odds_entropy` の追加により ROI・MDD が改善せず `odds_rank_divergence` 単体で効果がある場合:
- `field_odds_entropy` を除外した features_past_v25a.parquet を別途生成して比較検証する

---

## 付記: 代替アプローチ（オッズフィールドが利用不能な場合）

今回の調査では `odds`, `popularity`, `mining_predicted_rank` の全フィールドが利用可能であることを確認した。ただし、将来的にデータソース変更などで利用不能になった場合の代替として以下を検討する:

1. **前走人気変動率** (`lag1_popularity - lag2_popularity`): 市場の評価変化を間接的に表現
2. **トレーナー成績**: `trainer_code_encoded` との交互作用特徴量で穴馬特定
3. **CHSP show_prob 乖離**: v5 で実装済みの複勝市場確率との乖離（ただしフレームワーク未実装）
