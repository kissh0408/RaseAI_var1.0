# 実装仕様書: EV サンプル重み付き LambdaRank — 2026-06-30

## 前提条件（実施順序）

**この仕様書は以下が完了した後に着手すること。**

1. `2026-06-30-quinella-ev-fix-design.md` の実装完了
   （QuinellaOdds 事前オッズで馬連 EV が正しく計算できる状態）
2. `2026-07-01-wide-odds-ev-integration-design.md` の実装完了
   （WideOdds 事前オッズで wide EV が正しく計算できる状態）
3. `simulate_ev.py` の EV >= 1.0 Wide ROI が計測済み
   （この値が改善目標ベースラインとなる）

上記が完了する前に本仕様書に着手した場合、Step A で使う EV が不正確となり
重みが意味をなさない。

## 禁止特徴量の確認

- [x] WideOdds / QuinellaOdds を `features_*.parquet` にマージしないことを確認した
- [x] EV 計算用の重みを特徴量として学習時 DataFrame に含めないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] 人気順位を含まないことを確認した

---

## 1. 目的と背景

### なぜサンプル重み付けが必要か

現在の LambdaRank（NDCG 最適化）はすべてのレースを均等に評価する。
しかし ROI 改善に直接寄与するのは「EV >= 1.0 になるベット」であり、
これはテストセット全体の数 % に過ぎない（WideOdds 事前オッズで計測後に確定）。

均等学習では EV が低い多数のレースが目的関数を支配し、EV が高いレースでの
精度向上が抑制される可能性がある。

サンプル重み付けにより、EV が高いレース（収益に貢献するレース）での
着順予測精度を向上させ、ROI の改善を目指す。

### 制約: Top-1 の維持

EV サンプル重み付けは「EV 高レースでの2着・3着予測」を改善する手法である。
Top-1 的中率（1着予測）を犠牲にすることは許容しない。

**必須条件**: Top-1 >= 29.0%（Phase 7 ベースライン 28.5% 超かつ現状 30.18% の 95%）

---

## 2. 2段学習プロセスの詳細設計

### 全体フロー

```
Step A: 現行モデル（features_v29_fixed + models/lambdarank_fold*_seed*.txt）で
        学習データ全件の EV を予測
        └ 学習データ = race_date <= valid_end（2024-12-31）の全行
        └ WideOdds CSV から各ペアの事前オッズを取得
        └ Harville 確率 × 事前オッズ = EV（Wide）

Step B: EV に基づくサンプル重みを計算
        └ シグモイド重み: w(EV) = 1 + (max_weight - 1) × sigmoid(k × (EV - 1.0))
        └ k = 10（デフォルト）、感度分析: k=5, 10, 20
        └ max_weight = 2.0（デフォルト）、感度分析: 1.5, 2.0, 3.0

Step C: レース内重み正規化
        └ weight_norm[i] = weight[i] / sum(weight[same_race])
        └ 目的: 頭数の多いレース（G1等）が過剰に評価されることを防ぐ

Step D: 重み付き LambdaRank を学習
        └ lgb.Dataset(..., weight=weight_norm)
        └ 5 seeds × 3 folds = 15 モデルのアンサンブル（構成は変更しない）
```

### Step A: 学習データの EV 計算

現行モデル（`pure_rank/models/` 内の全 15 モデル）を使って、学習プール
（`race_date <= 2024-12-31`）全件の EV を予測する。

```python
# EV 計算対象: train_pool 全行
cfg = load_config()
valid_end_ts = pd.Timestamp(cfg["training"]["valid_end"])  # 2024-12-31
df_train_pool = df[df["race_date"] <= valid_end_ts].copy()

# アンサンブル予測スコアを取得
models = load_models(models_dir)
preds = ensemble_predict(models, df_train_pool[feature_cols])
df_train_pool["pred_score"] = preds
```

EV 計算の手順（1レース分）:
1. アンサンブル予測スコアを取得（`ensemble_predict()`）
2. 温度パラメータ T_opt で Softmax 変換（`softmax_with_temperature(T_opt=0.76)`）
3. Harville 公式で各ペアの P_wide を計算（`compute_race_probabilities()`）
4. Harville 最大 P_wide のペアを選択（`_best_wide_pair()`）
5. WideOdds CSV から選択ペアの事前オッズを取得（学習プール対象年の全 CSV を読み込む）
6. `EV = P_wide × prior_odds`（NaN なら EV = NaN → weight = 1.0）

学習プール対象年（WideOdds CSV を読み込む年）:
```python
train_years = sorted(df_train_pool["race_date"].dt.year.unique().tolist())
# → 例: [2015, 2016, ..., 2024]
```

**自己参照に関する注意**: 現行モデルは `df_train_pool` のサブセット（各 fold の train 部分）で
学習されている。そのモデルを `df_train_pool` 全件に適用することは「自分が学習したデータ」を
予測することになるが、**EV はラベルではなく重みであるためリークにならない**。
`finish_rank` や `is_win` の真のラベルを EV 計算に使っていないことが重要。

### Step B: サンプル重みの計算

重みはレース単位ではなく**馬単位**（行単位）で計算する。
1レース内の全馬に同じレースの EV に基づく重みを割り当てる。

```python
def compute_ev_weight(ev: float, k: float, max_weight: float = 2.0) -> float:
    """EV をシグモイド関数で重みに変換する。

    Parameters
    ----------
    ev         : 当該レースの最大 EV ペアの EV 値（NaN の場合は 1.0 を返す）
    k          : シグモイドの急峻さパラメータ（感度分析: 5, 10, 20）
    max_weight : 重みの上限（感度分析: 1.5, 2.0, 3.0）

    Returns
    -------
    weight in [1.0, max_weight]
        - EV << 1.0 → weight ≈ 1.0
        - EV == 1.0 → weight = (1.0 + max_weight) / 2
        - EV >> 1.0 → weight ≈ max_weight
    """
    if np.isnan(ev):
        return 1.0  # EV 不明（オッズ未取得）→ デフォルト重み
    return 1.0 + (max_weight - 1.0) / (1.0 + np.exp(-k * (ev - 1.0)))
```

各 k・max_weight での重みの挙動:

| k / max_weight | EV=0.8 | EV=1.0 | EV=1.2 |
|----------------|--------|--------|--------|
| k=5, max=1.5   | 1.13   | 1.25   | 1.37   |
| k=10, max=2.0  | 1.07   | 1.50   | 1.93   |
| k=20, max=3.0  | 1.02   | 2.00   | 2.98   |

重みの上限を設ける理由: 極端な重みによる過学習・不安定な学習を防ぐ。
`max_weight` を 3.0 より大きくしたい場合は planner に差し戻して承認を得ること。

**各レースへの EV の割り当て:**

```python
# df_train_pool に race_ev 列を追加する
# race_ev = そのレースで Harville が推奨した最大 EV ペアの EV 値
# （全馬に同じ値を割り当てる）
race_ev_map = {}  # race_id -> EV

for race_id, grp in df_train_pool.groupby("race_id"):
    scores = grp.sort_values("pred_score", ascending=False)["pred_score"].values
    probs = compute_race_probabilities(scores, T_opt)
    horse_nums = grp.sort_values("pred_score", ascending=False)["horse_num"].astype(int).values
    wi, wj = _best_wide_pair(probs["wide_matrix"])
    wide_key = _norm_pair(int(horse_nums[wi]), int(horse_nums[wj]))
    prior = wide_odds_lookup.get(str(race_id), {}).get(wide_key, None)
    p_wide = float(probs["wide_matrix"][wi, wj])
    ev = (p_wide * prior) if prior is not None else float("nan")
    race_ev_map[race_id] = ev

df_train_pool["race_ev"] = df_train_pool["race_id"].map(race_ev_map)
```

### Step C: レース内重み正規化

```python
df_train_pool["weight_raw"] = df_train_pool["race_ev"].apply(
    lambda ev: compute_ev_weight(ev, k=K, max_weight=MAX_WEIGHT)
)

# レース内正規化: 各レース内で重みの合計を 1 にスケールする
race_weight_sum = df_train_pool.groupby("race_id")["weight_raw"].transform("sum")
df_train_pool["weight_norm"] = df_train_pool["weight_raw"] / race_weight_sum
```

正規化後の重みの性質:
- 1レース内の重みの合計 = 1.0（均等重みでも同じ。LambdaRank の group 正規化と整合）
- 18頭立てと5頭立てが同等の学習影響力を持つ

### Step D: 重み付き LambdaRank の学習

`train.py` の `train_lambdarank()` に `weight_train` 引数を追加して渡す。

```python
lgb_train = lgb.Dataset(
    X_train[feature_cols],
    label=y_train,
    group=group_train,
    weight=weight_train,          # 追加。None の場合は均等重み（後方互換）
    categorical_feature=valid_cat,
    free_raw_data=False,
)
```

`weight_train` の重要な制約: **`group_train` と同じ行順序で並んでいなければならない**。
`get_group_sizes()` は `sort=False` で呼ぶため、`df_train_pool` のソート順と
`weight_norm` の順序を一致させること。

---

## 3. 実装ファイルと変更箇所

### `pure_rank/src/train.py` の変更

**変更 1: `train_lambdarank()` の引数追加**

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
    weight_train: np.ndarray | None = None,  # 追加（デフォルト None = 均等重み）
) -> lgb.Booster:
```

**変更 2: `main()` の CLI 引数追加**

```python
parser.add_argument(
    "--use-ev-weight",
    action="store_true",
    help="EV ベースのサンプル重み付きで学習する",
)
parser.add_argument(
    "--ev-weight-k",
    type=float,
    default=10.0,
    help="シグモイド重みの急峻さパラメータ（感度分析: 5, 10, 20）",
)
parser.add_argument(
    "--ev-weight-max",
    type=float,
    default=2.0,
    help="サンプル重みの上限（感度分析: 1.5, 2.0, 3.0）",
)
```

**変更 3: `main()` に EV 重み計算の呼び出しを追加**

`--use-ev-weight` フラグが立っている場合にのみ重みを計算する:

```python
weight_array: np.ndarray | None = None
if args.use_ev_weight:
    weight_array = compute_train_weights(
        df=df_train_pool,
        feature_cols=feature_cols,
        models_dir=models_dir,
        odds_dir=PROJECT_ROOT / "common" / "data" / "output" / "odds",
        T_opt=T_opt,
        k=args.ev_weight_k,
        max_weight=args.ev_weight_max,
        cfg=cfg,
    )
    print(f"  EV weight range: {weight_array.min():.4f} - {weight_array.max():.4f}")
```

**変更 4: モデル保存先の変更（重み付きモデル）**

```python
if args.use_ev_weight:
    k_label = str(int(args.ev_weight_k)) if args.ev_weight_k == int(args.ev_weight_k) else str(args.ev_weight_k)
    max_label = str(args.ev_weight_max).replace(".", "")
    models_dir = PROJECT_ROOT / f"pure_rank/models_weighted_k{k_label}_max{max_label}"
else:
    models_dir = PROJECT_ROOT / cfg["data"]["models_dir"]
models_dir.mkdir(parents=True, exist_ok=True)
```

既存モデル（`pure_rank/models/`）は上書きしない。重み付きモデルは別ディレクトリに保存する。

### 新規ヘルパー関数: `compute_train_weights()` の配置

`train.py` 内（`train_lambdarank()` の前）に配置する。
または大きくなりすぎる場合は `pure_rank/src/ev_weights.py` に分離してもよい。

```python
def compute_train_weights(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    odds_dir: Path,
    T_opt: float,
    k: float,
    max_weight: float,
    cfg: dict,
) -> np.ndarray:
    """学習データ全行に対する EV サンプル重みを計算して返す。

    Parameters
    ----------
    df          : 学習プール全行（race_date <= valid_end）
    feature_cols: 学習に使う特徴量列
    models_dir  : 現行モデルの保存ディレクトリ
    odds_dir    : WideOdds CSV のディレクトリ
    T_opt       : Softmax 温度パラメータ（train_config.json から）
    k           : シグモイド急峻さ（感度分析パラメータ）
    max_weight  : 重みの上限（感度分析パラメータ）
    cfg         : train_config.json の内容

    Returns
    -------
    np.ndarray: shape = (len(df),)。df の行順と一致すること。
    """
```

この関数は `simulate_ev.py` の関数群（`load_models`, `ensemble_predict`,
`compute_race_probabilities`, `_best_wide_pair`, `_norm_pair`,
`softmax_with_temperature`, `_build_wide_odds_lookup`）を **インポートして再利用する**。
コードの重複は禁止。

### `pure_rank/src/create_features.py` への変更

**変更しない**。EV は特徴量ではないため `features_*.parquet` には含めない。

### `train_config.json` への追加

```json
"sample_weighting": {
    "enabled": false,
    "k": 10,
    "k_candidates": [5, 10, 20],
    "max_weight": 2.0,
    "max_weight_candidates": [1.5, 2.0, 3.0],
    "weight_range_note": "sigmoid output range = [1.0, max_weight]",
    "models_dir_template": "pure_rank/models_weighted_k{k}_max{max_weight}"
}
```

---

## 4. データリーク防止

### EV による重みはラベルではない

EV は「学習サンプルの重要度」であり、学習対象のラベル（`finish_rank`、`lr_label`）ではない。
現行モデルが「自分が学習したデータ」に対して EV を計算しても、リークにならない理由:

- モデルが EV 計算に使うのは「スコア（相対強さの予測）」であり、結果（着順）ではない
- `finish_rank` や `is_win` の真のラベルを EV 計算には使わない
- EV が高いレースが「実際に当たりやすかったレース」と一致するとは限らない

### テストデータの完全な保護

```
学習データ（重み計算対象）: race_date <= 2024-12-31
テストデータ（評価のみ）  : race_date >= 2025-01-01
```

テストデータの EV を重み計算に使ってはならない。

### 禁止事項

| 禁止事項 | 理由 |
|---------|------|
| テストデータ（2025+）の EV を重み計算に使う | テストリーク |
| 重みを `features_*.parquet` に含める | EV は特徴量ではない |
| バリデーション精度を見ながら k・max_weight を後付け調整 | 後出しじゃんけん禁止 |
| k=5, 10, 20 以外のパラメータを事後追加 | 1パラメータずつ変更の原則に違反 |
| 重み付きモデルと均等重みモデルを混在させたアンサンブルを組む | 効果の分離が不可能 |

---

## 5. 感度分析の計画

### 実施すべき組み合わせ（最小限）

| 実験 | k | max_weight | 目的 |
|------|---|------------|------|
| baseline（均等重み） | — | — | 比較基準（現行モデル）。再学習不要 |
| 推奨試験値 | 10 | 2.0 | 既存仕様書と同一設定。まず試す |
| k 感度 (low) | 5 | 2.0 | 穏やかな重み付け |
| k 感度 (high) | 20 | 2.0 | 急峻な重み付け |
| max_weight 感度 (low) | 10 | 1.5 | 控えめな最大重み |
| max_weight 感度 (high) | 10 | 3.0 | 強い最大重み（慎重に実施） |

**推奨実施順序**: まず `k=10, max_weight=2.0` を実施して Top-1 が維持できるか確認する。
その後 k 感度を試す。max_weight=3.0 は Top-1 が 29.0% 以上を維持できた場合のみ実施する。

### 報告フォーマット

```
[k=10, max_weight=2.0]
  Top-1:              ??.??%（基準: >29.0%）
  NDCG@3:             0.???  （基準: >0.525）
  Spearman:           0.???  （基準: >0.480）
  Wide ROI (EV>=1.0): ??.??%（ベースライン: ???%）
  EV>=1.0 ベット数:   ???件  （ベースライン: ???件）
  weight range:       ?.???–?.???

[k=5, max_weight=2.0]
  ...
```

---

## 6. 評価基準

### 合否判定

| 指標 | 合格 | 差し戻し条件 |
|------|------|------------|
| Top-1 的中率 | >= 29.0% | < 29.0%（均等重みに戻し原因調査） |
| NDCG@3 | > 0.525 | < 0.525（重み付けが過剰な疑い） |
| Spearman | > 0.480 | < 0.480 |
| Wide ROI（EV>=1.0） | > ベースライン ROI | <= ベースライン（重み付けの効果なし） |
| EV>=1.0 ベット数 | ベースラインの ±10% 以内 | > ±10%（ペア選択が大きく変化） |

### リーク停止閾値

```
Top-1 > 40% または Spearman > 0.6 → 即座に実装停止して evaluator へ報告
```

### 差し戻しプロトコル

**Top-1 < 29.0% の場合:**
1. `--use-ev-weight` なしで再学習し、Top-1 が 30.18% に回復することを確認
2. 回復すれば: サンプル重み計算・正規化ロジックにバグがある（implementer に差し戻し）
3. 回復しなければ: 特徴量生成や学習データに問題がある（planner に差し戻し）

---

## 7. 実装コマンド

```bash
# 均等重みモデル（ベースライン。既存モデルを使う。再学習不要）
# → 現行の pure_rank/models/ を使って evaluate.py を実行するだけでよい

# 重み付き学習（k=10, max_weight=2.0）
cd C:\Users\syugo\AI\RaceAI_var1.0
python pure_rank/src/train.py --ensemble --use-ev-weight --ev-weight-k 10 --ev-weight-max 2.0

# 重み付き学習（k=5, max_weight=2.0）
python pure_rank/src/train.py --ensemble --use-ev-weight --ev-weight-k 5 --ev-weight-max 2.0

# k=10, max_weight=3.0（Top-1 維持が確認できた後のみ実施）
python pure_rank/src/train.py --ensemble --use-ev-weight --ev-weight-k 10 --ev-weight-max 3.0

# 精度評価（重み付きモデル）
# evaluate.py の models_dir を pure_rank/models_weighted_k10_max20 に向けて実行
python pure_rank/src/evaluate.py --models-dir pure_rank/models_weighted_k10_max20

# EV シミュレーション（重み付きモデル）
python pure_rank/src/simulate_ev.py --models-dir pure_rank/models_weighted_k10_max20
```

---

## 8. implementer への引き渡し事項

以下の順序で実装すること（ブランチ: `feature/sample-weighting`）。

1. `pure_rank/src/train.py` に `compute_train_weights()` ヘルパー関数を追加する
   - `simulate_ev.py` の `_build_wide_odds_lookup`・`compute_race_probabilities` 等を
     インポートして再利用する
   - 関数シグネチャは Section 3 の仕様に従う

2. `train_lambdarank()` に `weight_train: np.ndarray | None = None` 引数を追加し、
   `lgb.Dataset` に渡す（Section 3 変更1）

3. `main()` に `--use-ev-weight`・`--ev-weight-k`・`--ev-weight-max` 引数を追加し、
   `compute_train_weights()` を呼ぶ（Section 3 変更2・3）

4. モデル保存ディレクトリを `--use-ev-weight` フラグによって切り替える（Section 3 変更4）

5. `train_config.json` に `sample_weighting` セクションを追加する

6. まず `k=10, max_weight=2.0` で学習を実行し、evaluator に報告する:
   ```bash
   python pure_rank/src/train.py --ensemble --use-ev-weight --ev-weight-k 10 --ev-weight-max 2.0
   python pure_rank/src/evaluate.py  # models_dir を weighted に向けること
   ```

7. 市場情報混入チェック:
   ```bash
   grep -rn "odds\|popularity\|market_log_odds\|init_score" \
       C:/Users/syugo/AI/RaceAI_var1.0/pure_rank/src/train.py --include="*.py"
   ```
   `odds_dir` の変数名が引っかかるが、WideOdds を特徴量に使っていなければ問題なし。
