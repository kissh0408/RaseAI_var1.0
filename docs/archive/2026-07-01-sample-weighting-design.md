# 実装仕様書: EV サンプル重み付き LambdaRank — 2026-07-01

## 前提条件

**この仕様書は `2026-07-01-wide-odds-ev-integration-design.md` の実装が完了した後に着手する。**
真の EV（WideOdds 事前オッズ使用）が計算できる状態になってから実施すること。

## 禁止特徴量の確認

- [x] WideOdds（オッズ）を `features_*.parquet` にマージしないことを確認した
- [x] EV 計算用の重みを特徴量として学習時 DataFrame に含めないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] 人気順位を含まないことを確認した

---

## 1. 目的と背景

### なぜサンプル重み付けが必要か

現在の LambdaRank（NDCG 最適化）はすべてのレースを均等に評価する。
しかし ROI 改善に直接寄与するのは「EV > 1.0 になるベット」であり、
これはテストセット全体の約 4% に過ぎない（EV > 1.0 フィルタ後ベット数 / 全レース数）。

均等学習では EV が低い多数のレースが目的関数を支配し、
EV が高いレースでの精度向上が抑制される可能性がある。

サンプル重み付けにより、EV が高いレース（= 収益に貢献するレース）での
着順予測精度を向上させ、ROI > 100% を達成する。

### 制約: Top-1 の維持

EV サンプル重み付けは「EV 高レースでの2着・3着予測」を改善する手法である。
Top-1 的中率（1着予測）を犠牲にすることは許容しない。

**合格条件**: Top-1 > 29.0%（Phase 7 ベースライン 28.5% 超かつ現状 30.18% の 95%）

---

## 2. 2段学習プロセスの詳細設計

### 全体フロー

```
Step A: ベースモデル（v30_relative 学習済みモデル）で学習データ全件の EV を予測
        └ WideOdds CSV から各ペアの事前オッズを取得
        └ Harville 確率 × 事前オッズ / 100 = EV

Step B: EV に基づくサンプル重みを計算
        └ シグモイド重み: w(EV) = 1 + sigmoid(k × (EV - 1.0))
        └ k = 5, 10, 20 で感度分析（Section 5 参照）

Step C: レース内重み正規化
        └ weight_normalized[i] = weight[i] / sum(weight_in_race)
        └ 目的: レース間の公平性を保つ

Step D: 重み付き LambdaRank を学習
        └ lgb.Dataset(..., weight=weight_normalized) に正規化済み重みを渡す
        └ その他のハイパーパラメータは変更しない
        └ 5 seeds × 3 folds のアンサンブル（標準構成を維持）
```

### Step A: 学習データの EV 計算

ベースモデル（現行 v30_relative モデル）を使って学習データ全件の EV を予測する。

対象データ:
```
TRAIN_END = '2021-12-31'
学習データ: race_date <= 2021-12-31 の全行
```

EV 計算の手順（1レース分）:
1. アンサンブル予測スコアを取得（`ensemble_predict()`）
2. 温度パラメータ T_opt で Softmax 変換（`softmax_with_temperature()`）
3. Harville 公式で各ペアの P_wide を計算（`compute_race_probabilities()`）
4. Harville 最大 P_wide のペアを選択（`_best_wide_pair()`）
5. WideOdds CSV から選択ペアの事前オッズを取得
6. `EV = P_wide × prior_odds / 100.0`

EV が取得できないレース（WideOdds に該当ペアのオッズが存在しない場合）は
`ev = NaN` とし、Step B では `weight = 1.0`（デフォルト）を割り当てる。

### Step B: サンプル重みの計算

重みはレース単位ではなく**馬単位**で計算する。
1レース内の全馬に同じレースの EV に基づく重みを割り当てる。

```python
def compute_ev_weight(ev: float, k: float) -> float:
    """EV をシグモイド関数で重みに変換する。

    Parameters
    ----------
    ev : 当該レースの最大 EV ペアの EV 値
    k  : シグモイドの急峻さパラメータ（感度分析: k=5, 10, 20）

    Returns
    -------
    weight in [1.0, 2.0]
        - EV << 1.0 → weight ≈ 1.0
        - EV == 1.0 → weight = 1.5
        - EV >> 1.0 → weight ≈ 2.0
    """
    if np.isnan(ev):
        return 1.0  # EV 不明 → デフォルト重み
    return 1.0 + 1.0 / (1.0 + np.exp(-k * (ev - 1.0)))
```

重みの範囲は [1.0, 2.0]。EV < 1.0 のレースも学習から除外しない（最低重み = 1.0）。

### Step C: レース内重み正規化

レース頭数の違いによる学習への過大影響を防ぐため、レース内で正規化する。

```python
# df_train は学習対象の全行（race_id, ev, 各行のweight を含む）
df_train["weight_raw"] = df_train["race_ev"].map(
    lambda ev: compute_ev_weight(ev, k=K)
)

# レース内正規化
race_weight_sum = df_train.groupby("race_id")["weight_raw"].transform("sum")
df_train["weight_normalized"] = df_train["weight_raw"] / race_weight_sum
```

正規化後の重みの性質:
- 1レース内の重みの合計 = 1.0（均等重みでも同じ）
- 頭数が多いレース（18頭）と少ないレース（5頭）が同等の学習影響力を持つ
- EV が高いレースでは上位馬がより高い重みを受け取る

### Step D: 重み付き LambdaRank の学習

`train.py` の `train_lambdarank()` 関数の `lgb.Dataset` に `weight` を追加する:

```python
lgb_train = lgb.Dataset(
    X_train[feature_cols],
    label=y_train,
    group=group_train,
    weight=weight_array,          # 追加: shape = (n_samples,)
    categorical_feature=valid_cat,
    free_raw_data=False,
)
```

`weight_array` は `df_train` の `weight_normalized` 列を numpy 配列として渡す。
学習順序が `group_train` と一致するように、`df_train` のソート順を `get_group_sizes()` と同一にすること。

---

## 3. データリーク防止

### 学習データ内の自己参照を防ぐ

Step A の EV 計算に使うモデルは**現行の v30_relative 学習済みモデル**である。
このモデルは `TRAIN_END = '2021-12-31'` のデータで学習されている。

重みを計算する対象も同じ学習データ（2021年以前）であるため、
モデルが「自分が学習したデータ」を EV 計算に使うことになる。
これは意図的な設計であり、以下の理由でリークにはならない:

- EV 計算の目的は「このレースが期待値プラスか否か」の判定（ラベル生成）ではない
- EV はサンプルの重要度を示す「重み」であり、ラベル（finish_rank）は変更しない
- `is_win`、`finish_rank` 等の真のラベルを EV 計算に使っていない

ただし以下は禁止する:

| 禁止事項 | 理由 |
|---------|------|
| Step D の重み付き学習モデルを Step A の EV 計算に再利用する | 循環参照 |
| テストデータ（2023年以降）の EV を学習データの重み計算に使用する | テストリーク |
| 重みを特徴量として `features_*.parquet` に追加する | EV は特徴量ではない |

### テストデータの保護

Step A〜C は**学習データ（2021年以前）のみ**に適用する。

```
学習データ: race_date <= 2021-12-31  → EV 計算 → 重み付き学習
バリデーション: 2022-01-01 〜 2022-12-31  → 重みなし（評価のみ）
テスト: 2023-01-01 以降  → 重みなし（評価のみ）
```

---

## 4. 実装ファイルと変更箇所

### `pure_rank/src/train.py` の変更

**変更箇所 1: `train_lambdarank()` の引数追加**

```python
def train_lambdarank(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    group_valid: list[int],
    feature_cols: list[str],
    cat_features: list[str],
    params_cfg: dict,
    training_cfg: dict,
    seed: int,
    weight_train: np.ndarray | None = None,  # 追加
) -> lgb.Booster:
```

`lgb.Dataset` の `weight` パラメータ:
```python
lgb_train = lgb.Dataset(
    X_train[feature_cols],
    label=y_train,
    group=group_train,
    weight=weight_train,  # None の場合は均等重み（後方互換）
    categorical_feature=valid_cat,
    free_raw_data=False,
)
```

**変更箇所 2: `main()` に EV 重み計算の呼び出しを追加**

新規ヘルパー関数 `compute_train_weights()` を `train.py` に追加し、
`main()` から `--use-ev-weight` フラグが立っている場合に呼び出す。

```python
# main() の引数に追加
parser.add_argument(
    "--use-ev-weight",
    action="store_true",
    help="EV ベースのサンプル重み付きで学習する（Phase 2: sample weighting）",
)
parser.add_argument(
    "--ev-weight-k",
    type=float,
    default=10.0,
    help="シグモイド重みの急峻さパラメータ（感度分析: 5, 10, 20）",
)
```

### `pure_rank/src/create_features.py` への変更

**変更しない**。EV は特徴量ではないため、`features_*.parquet` には含めない。

EV 計算のヘルパー関数は `train.py` 内（または新規 `ev_weights.py`）に配置し、
`create_features.py` から独立させる。

### 新規ブランチ

```
feature/sample-weighting
```

`pure_rank/models/` は既存モデルを保持したまま、重み付きモデルを別名で保存する:

```
pure_rank/models/  # 既存（v30_relative, 均等重み）
    lambdarank_fold1_seed42.txt
    ...

pure_rank/models_weighted_k{k}/  # 新規（重み付き）
    lambdarank_fold1_seed42.txt
    ...
```

`train_config.json` に以下を追加（implementer が値を記入）:

```json
"sample_weighting": {
    "enabled": false,
    "k": 10,
    "k_candidates": [5, 10, 20],
    "weight_range": [1.0, 2.0],
    "models_dir_weighted": "pure_rank/models_weighted_k10"
}
```

---

## 5. k パラメータの感度分析

k=5, 10, 20 の3パターンで独立して学習・評価する。

### 各 k の挙動

| k | EV=0.8 の重み | EV=1.0 の重み | EV=1.2 の重み | 特性 |
|---|--------------|--------------|--------------|------|
| 5 | 1.18 | 1.50 | 1.82 | 緩やか。EV の違いが重みに緩く反映される |
| 10 | 1.07 | 1.50 | 1.93 | 標準。EV=1.0 前後で急峻に変化する |
| 20 | 1.02 | 1.50 | 1.98 | 急峻。EV=1.0 より上下でほぼ 2.0 / 1.0 に分離される |

### 報告形式

```
k=5:
  Top-1:      ??.??%（基準: >29.0%）
  NDCG@3:     0.???
  Spearman:   0.???
  Wide ROI (EV>1.0): ??.??%（変更前: ??.??%）
  EV>1.0 ベット数:   ???（変更前: ???）

k=10（推奨試験値）:
  ...

k=20:
  ...
```

最良の k を採用する基準:
1. Top-1 >= 29.0%（必須条件）
2. Wide ROI（EV>1.0）が最も高い k を採用

---

## 6. 評価基準

### 合否判定

| 指標 | 合格 | 差し戻し条件 |
|------|------|------------|
| Top-1 的中率 | > 29.0% | < 29.0%（実装バグの疑い。均等重みに戻す） |
| NDCG@3 | > 0.525 | < 0.525（重み付けが過剰な疑い） |
| Spearman | > 0.48 | < 0.48 |
| Wide ROI（EV>1.0） | > 変更前の ROI | < 変更前 ROI（重み付けの効果なし） |
| EV>1.0 ベット数 | > 変更前の件数（±10%） | < 変更前の 90%（ペア選択が大きく変化） |
| Top-1 > 40% | 即座に実装停止 | データリークの強い疑い |
| Spearman > 0.6 | 即座に実装停止 | データリークの強い疑い |

### 差し戻しプロトコル

**Top-1 < 29.0% の場合（差し戻し）:**
- implementer: 重みなし（`--use-ev-weight` なし）で再学習し、Top-1 が回復することを確認
- 回復した場合: サンプル重みの計算・正規化ロジックにバグがある
- 回復しない場合: 特徴量生成や学習データに問題がある（planner に差し戻し）

**Wide ROI が改善しない場合（要確認）:**
- EV=NaN（オッズ未取得）のレース割合を確認する
- `n_ev_na / n_races_total > 0.05`（5%以上）の場合は WideOdds 統合に問題がある可能性
- k の値を変えて再試行する

---

## 7. 注意事項と設計判断の記録

### なぜレース内正規化が必要か

LambdaRank の `group` 配列はレース単位のグループを定義する。
LightGBM は各グループの損失を合計するため、頭数が多いレースほど損失への寄与が大きくなる。

重みをレース内正規化することで:
- 18頭立てレースと5頭立てレースが同等の影響力を持つ
- EV が高い馬が多数出走するレース（例: 重賞）が過剰に評価されない

### なぜ重みの上限を 2.0 にするか

シグモイド関数の出力範囲は (0, 1) であり、`1.0 + sigmoid(k*(EV-1.0))` の範囲は (1.0, 2.0) となる。
EV > 1.0 のレースの重みが均等重みの最大 2 倍になる設計である。

上限を設ける理由:
- 極端な重みによる過学習・不安定な学習を防ぐ
- EV が非常に高いレースでも均等重みの 2 倍以上にはならない

3倍以上に拡張する場合は planner に差し戻して設計変更を承認する。

### `features_*.parquet` の不変性

- `features_v30_relative.parquet` は変更しない
- 重みはメモリ内で計算し、`lgb.Dataset` に直接渡す
- バックアップ対象外（Parquet を変更しないため）

### Phase 2（サンプル重み）の実施タイミング

Phase 1（WideOdds 統合）が完了し、以下が確認された後に着手すること:

1. WideOdds ローダーが正常に動作している（`n_ev_na` が全体の 5% 未満）
2. EV > 1.0 フィルタ後の Wide ROI が計測されている
3. 計測された ROI を改善目標として設定できる状態になっている

---

## 8. 禁止事項（サンプル重み付け固有）

| 禁止事項 | 理由 |
|---------|------|
| テストデータの EV を重み計算に使う | テストリーク |
| 重みを `features_*.parquet` に含める | EV は特徴量ではない |
| バリデーション精度を見ながら k を調整する | 後出しじゃんけん禁止 |
| k=5, 10, 20 以外のパラメータを事後追加する | 1パラメータずつ変更の原則に違反 |
| 重み付きモデルと均等重みモデルを混在させたアンサンブルを組む | 効果の分離が不可能 |
