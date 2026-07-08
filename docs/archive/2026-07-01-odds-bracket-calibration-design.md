# 実装仕様書: オッズ帯別キャリブレーション再設計 — 2026-07-01

## 禁止特徴量の確認

- [x] WideOdds を `features_*.parquet` にマージしないことを確認した
- [x] キャリブレーションパラメータを学習の特徴量として含めないことを確認した
- [x] `init_score` に市場オッズ由来の値を使わないことを確認した
- [x] 人気順位を含まないことを確認した

---

## 1. 目的と背景

### 現在のキャリブレーション問題

`calibration_wide.json` の実データ（テストセット 4775 レース）を見ると、
p_wide の過大評価が **全ビンで一貫して** 観測されている:

| ビン | 予測確率 | 実的中率 | 差 |
|------|---------|---------|-----|
| 0 | 0.1475 | 0.1234 | -0.0241 |
| 4 | 0.3194 | 0.2469 | -0.0726 |
| 9 | 0.7235 | 0.5711 | -0.1524 |

平均絶対誤差: 0.0794、最大絶対誤差: 0.1524。
高 p_wide 帯で誤差が拡大するパターン（Harville の大本命バイアス）。

### 前回の失敗: 全件一括 Isotonic の問題

前回（`_collect_bets_with_calibration()` の手法3）で試みた Isotonic 回帰は:
- 全ビンの p_wide 値をフラットに並べて単調増加回帰を学習
- 大量の低確率ペア（全 C(n,2) ペアの大半）がモデルを支配
- 高 EV 帯（実際にベットする帯）での補正が不十分

### 前回の失敗: 全件一括 Platt の問題

Platt スケーリング（LogisticRegression）の失敗:
- 線形変換 p_cal = sigmoid(a * p_win + b) は非線形誤差に対応できない
- 全馬 p_win の平均誤差を最小化するので、ベット対象ペアの誤差は改善されない

### 本仕様のアプローチ

**オッズ帯ごとに独立して Isotonic 回帰を学習する**。

理由:
1. 低オッズペア（p_wide 高、実ベット対象）と高オッズペア（p_wide 低、ベットしない）で
   Harville バイアスの挙動が異なる
2. 全件一括モデルは多数派の低確率ペアに引きずられる
3. オッズ帯内では p_wide の範囲が狭まり、Isotonic が安定して収束する

---

## 2. オッズ帯の定義

### 使用するオッズ: WideOdds の prior_odds

キャリブレーションの帯分けには WideOdds の事前オッズ（decimal multiplier）を使う。
HR 払戻（事後データ）ではなく事前オッズで帯を決める理由:
- 事後払戻は外れレースで 0 になり参照不能
- 事前オッズはベット時点で観測可能（実運用時も利用できる）

### 帯の設定（候補 A: 固定境界）

WideOdds の分布を考慮した分割:

| 帯番号 | 条件 | 性質 |
|--------|------|------|
| 0 | odds < 3.0 | 最人気ペア（2頭で3着圏ほぼ確実視） |
| 1 | 3.0 <= odds < 8.0 | 中間ペア（実際のベット対象の中心） |
| 2 | 8.0 <= odds < 20.0 | やや穴ペア |
| 3 | odds >= 20.0 | 大穴ペア（ベット対象外が多い） |

### 帯の設定（候補 B: バリデーション分位点）

2024 年バリデーションセットの WideOdds 分布から 25/50/75 パーセンタイルを計算し境界にする。

**採用: 候補 A（固定境界）**

理由: 固定境界は再現性が高く、テストリークのリスクがない。
分位点境界は 2024 年データで決まるが、解釈が難しく実運用時に境界が変わる問題がある。

---

## 3. Harville バイアス補正

### 問題の説明

Harville 公式は「各馬の勝率 p_win が独立な指数分布に従う」と仮定する。
この仮定の元では、低勝率馬（p_win < 0.05）の 2 着・3 着確率が過大評価される。

具体例（18 頭立て、p_win の最小馬が 0.01）:
- Harville: p2 ≈ 0.01 / (1 - ε) ≈ 0.01 ずつ 17 頭に分配
- 実際: 最低人気馬が 2 着に来る確率は p_win より低い傾向（人気馬が差し返す）

これにより、大穴馬を含むワイドペアの p_wide が過大評価される。
結果として帯 2・3（やや穴〜大穴）の Harville 確率が系統的に高すぎる。

### 補正方針: 帯別 Isotonic 回帰で経験的に補正

Stern(1990) の数学的補正（累積ハザード関数の修正）は実装が複雑。
代わりに、帯別 Isotonic 回帰が経験的にこのバイアスを吸収する:
- 帯 0〜1（低オッズ): Harville 過大評価が比較的小さい → 軽微な収縮
- 帯 2〜3（高オッズ): Harville 過大評価が大きい → 大きな収縮

Isotonic がデータドリブンでバイアスを補正するため、明示的な Stern 補正は不要。
ただし以下を監視: 帯 3 のサンプル数が少ない場合は Isotonic が不安定になる。

### 帯 3 の最小サンプル数チェック

```python
MIN_SAMPLES_PER_BRACKET = 100  # 100 ペア未満の帯は Isotonic を学習しない
```

帯 3 がサンプル不足の場合: 帯 2 の Isotonic をフォールバックとして使う。

---

## 4. 実装設計

### 4.1 学習データ生成

```python
def collect_wide_pair_data_with_odds(
    df: pd.DataFrame,
    models: list[lgb.Booster],
    feature_cols: list[str],
    hr_df: pd.DataFrame,
    wide_odds_lookup: dict[str, dict[PAIR_KEY, float]],
    T: float,
) -> pd.DataFrame:
    """
    各レースの全ペア (i, j) について Harville p_wide・WideOdds・is_wide_hit を返す。

    predict.py の `_collect_wide_pair_data()` を拡張し、WideOdds を追加する。

    Returns
    -------
    pd.DataFrame:
        p_wide_harville: float（Harville 生確率）
        prior_odds     : float（WideOdds decimal multiplier。NaN = 未取得）
        hit            : int（0 or 1）
        odds_bracket   : int（0〜3、prior_odds が NaN の場合は -1）

    制約
    ----
    - prior_odds が NaN のペアは bracket=-1 とし、キャリブレーション学習から除外する
    - 学習データは 2024 年バリデーションセットのみ使用（テストリーク禁止）
    """
```

### 4.2 帯割り当て関数

```python
def assign_odds_bracket(odds: float) -> int:
    """
    WideOdds の decimal multiplier からブラケット番号を返す。

    Parameters
    ----------
    odds : float（WideOdds decimal multiplier。NaN の場合は -1 を返す）

    Returns
    -------
    int: 0〜3（-1 = 未分類）
    """
    if odds != odds:  # NaN check
        return -1
    if odds < 3.0:
        return 0
    elif odds < 8.0:
        return 1
    elif odds < 20.0:
        return 2
    else:
        return 3
```

### 4.3 帯別 Isotonic 学習

```python
def fit_bracket_isotonic(
    df_pairs: pd.DataFrame,
    min_samples: int = 100,
) -> dict[int, "IsotonicRegression"]:
    """
    ブラケット別に Isotonic 回帰を学習する。

    Parameters
    ----------
    df_pairs    : collect_wide_pair_data_with_odds() の出力
    min_samples : 最小サンプル数（これを下回る帯は学習をスキップ）

    Returns
    -------
    dict[int, IsotonicRegression]: {bracket_id: fitted_model}
        - bracket=-1 は除外
        - サンプル不足の帯は辞書に含まない

    学習設定
    --------
    IsotonicRegression(out_of_bounds="clip", increasing=True)
    X: p_wide_harville
    y: hit
    """
    from sklearn.isotonic import IsotonicRegression

    models: dict[int, "IsotonicRegression"] = {}
    for bracket in [0, 1, 2, 3]:
        subset = df_pairs[df_pairs["odds_bracket"] == bracket]
        n = len(subset)
        if n < min_samples:
            print(f"  [bracket {bracket}] samples={n} < {min_samples}, skip")
            continue
        X = subset["p_wide_harville"].values
        y = subset["hit"].values
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(X, y)
        models[bracket] = iso
        hit_rate = y.mean()
        print(
            f"  [bracket {bracket}] n={n:,} "
            f"hit_rate={hit_rate:.4f} "
            f"p_wide range=[{X.min():.4f}, {X.max():.4f}]"
        )
    return models
```

### 4.4 推論時の補正適用

```python
def apply_bracket_isotonic(
    p_wide_harville: float,
    prior_odds: float,
    bracket_models: dict[int, "IsotonicRegression"],
) -> float:
    """
    ペアの Harville p_wide をブラケット別 Isotonic で補正する。

    Parameters
    ----------
    p_wide_harville: float（補正前の Harville 確率）
    prior_odds     : float（WideOdds decimal multiplier。NaN の場合はパスルー）
    bracket_models : fit_bracket_isotonic() の出力

    Returns
    -------
    float: 補正後の p_wide。モデル未発見の場合は入力値をそのまま返す。

    フォールバック優先順位:
    1. 当該帯のモデルが存在 → そのモデルを使用
    2. 帯 3 のモデルが存在しない → 帯 2 のモデルを使用
    3. いずれも存在しない → p_wide_harville をそのまま返す
    """
    import math
    if math.isnan(prior_odds):
        return p_wide_harville

    bracket = assign_odds_bracket(prior_odds)
    if bracket == -1:
        return p_wide_harville

    model = bracket_models.get(bracket)
    if model is None and bracket == 3:
        model = bracket_models.get(2)  # 帯 3 フォールバック
    if model is None:
        return p_wide_harville

    return float(model.predict([p_wide_harville])[0])
```

### 4.5 キャリブレーション保存・読み込み

```python
def save_bracket_calibration(
    models_dir: Path,
    bracket_models: dict[int, "IsotonicRegression"],
    meta: dict,
) -> None:
    """
    帯別キャリブレーションモデルを models/calibration/bracket_isotonic/ に保存する。

    Parameters
    ----------
    bracket_models : {bracket_id: IsotonicRegression}
    meta           : 境界値・学習年など（train_config.json の calibration セクションに保存）

    保存先:
        models/calibration/bracket_isotonic_0.joblib  (帯0)
        models/calibration/bracket_isotonic_1.joblib  (帯1)
        models/calibration/bracket_isotonic_2.joblib  (帯2)
        models/calibration/bracket_isotonic_3.joblib  (帯3、存在する場合)
        models/calibration/bracket_meta.json           (境界値・学習年)
    """

def load_bracket_calibration(models_dir: Path) -> tuple[dict, dict]:
    """
    保存済み帯別キャリブレーションを読み込む。

    Returns
    -------
    tuple[dict[int, IsotonicRegression], dict]:
        第1要素: bracket_models
        第2要素: meta（境界値・学習年）
    """
```

### 4.6 `train_config.json` への calibration セクション追加

```json
"calibration": {
  "method": "bracket_isotonic",
  "valid_year": "2024",
  "bracket_boundaries": [3.0, 8.0, 20.0],
  "min_samples_per_bracket": 100,
  "fallback_bracket_for_3": 2,
  "fitted": false
}
```

`fitted: true` に更新された時点でモデルが学習済みを示す。

---

## 5. 評価設計: 帯別キャリブレーションの効果測定

### 5.1 キャリブレーション誤差の比較

補正前後で帯別の MAE（Mean Absolute Error）を計算:

```python
def evaluate_bracket_calibration_error(
    df_pairs: pd.DataFrame,  # p_wide_harville, hit, odds_bracket
    bracket_models: dict[int, "IsotonicRegression"],
) -> dict[int, dict]:
    """
    各ブラケットのキャリブレーション誤差（前後比較）を返す。

    Returns
    -------
    dict[bracket_id, {"mae_before": float, "mae_after": float, "n": int}]
    """
```

### 5.2 ROI への効果測定

補正前（Harville そのまま）と補正後（帯別 Isotonic 適用）で EV を再計算し、
`compare_calibration_methods()` に組み込む:

- 手法4として `bracket_isotonic` を追加（手法1〜3 は既存）
- テストセット（2025+）での ROI・的中率・ベット数を比較

### 5.3 合否基準

| 指標 | 合格条件 |
|------|---------|
| 補正後 MAE < 補正前 MAE | 全帯で改善（最低でも帯 0〜2 で改善） |
| テスト ROI（EV>=1.0） | ベースライン（86.0%@EV>=1.3）以上 |
| n_bets（EV>=1.0） | 100 件以上 |

---

## 6. 前回失敗との差分まとめ

| 項目 | 前回（全件 Isotonic） | 今回（帯別 Isotonic） |
|------|---------------------|---------------------|
| 学習単位 | 全 C(n,2) ペアをまとめて 1 つの Isotonic | 帯ごとに独立した Isotonic |
| 問題 | 大量の低確率ペアが支配 | 帯内で p_wide 範囲が絞られる |
| Harville バイアス対応 | なし（単調増加制約が全帯に共通） | 帯ごとに独立した単調性 |
| オッズ情報の使用 | なし（p_wide のみ） | WideOdds で帯を決定 |
| 実装難易度 | 低（`_collect_wide_pair_data()` 流用） | 中（`collect_wide_pair_data_with_odds()` が新規） |

---

## 7. 実装制約

| 制約 | 内容 |
|------|------|
| 学習データ | 2024 年バリデーションセットのみ（テストリーク禁止） |
| テストデータ | 2025 年以降（変更なし） |
| WideOdds の扱い | EV 計算・帯分けのみ。特徴量に含めない |
| パラメータ管理 | `train_config.json` の `calibration` セクションに一元管理 |
| 既存手法との共存 | 手法 1〜3（Platt・ROI-T・Isotonic 全件）を削除しない |

---

## 8. implementer への引き渡し事項

1. `predict.py` に `collect_wide_pair_data_with_odds()` を追加（Section 4.1）
   - `_collect_wide_pair_data()` を拡張する形で実装
2. `predict.py` に `assign_odds_bracket()` を追加（Section 4.2）
3. `predict.py` に `fit_bracket_isotonic()` を追加（Section 4.3）
4. `predict.py` に `apply_bracket_isotonic()` を追加（Section 4.4）
5. `predict.py` に `save_bracket_calibration()` / `load_bracket_calibration()` を追加（Section 4.5）
6. `train_config.json` に `calibration` セクションを追加（Section 4.6）
7. `simulate_ev.py` の `compare_calibration_methods()` に手法4として `bracket_isotonic` を追加
8. バリデーション 2024 年で学習: `python pure_rank/src/predict.py --fit-bracket-calibration`
9. テストセット 2025+ で評価: `python pure_rank/src/simulate_ev.py --compare-calibration`

実行コマンド（新規追加フラグ）:
```bash
cd C:\Users\syugo\AI\RaceAI_var1.0
python pure_rank/src/predict.py --fit-bracket-calibration
python pure_rank/src/simulate_ev.py --compare-calibration
```
