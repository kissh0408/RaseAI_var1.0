# 実装仕様書: 複勝(top3)確率 直接予測実験 — 2026-07-09

**ステータス**: **完了（2026-07-09 evaluator サインオフ済み）— verdict = `not_superior_no_reattempt`**
**担当**: implementer（本仕様に従い実装）→ evaluator（合否判定）
**実験ディレクトリ**: `pure_rank/experiments/place_direct/`（この配下に完結させる）

---

## 完了記録（2026-07-09 — evaluator 独立検証・サインオフ）

> 隔離実験の記録。`evaluation/reports/gate_summary.json` 等の本番合否には反映しない。

**結果（TEST 2025+、4,775レース / 66,020頭、`reports/place_direct_comparison.json`）**:

| 系列 | logloss | 較正誤差最大(pp) |
|------|---------|----------------|
| (a) Stern 逆算 | **0.4003** | 4.53 |
| (b) Harville 逆算 | 0.4090 | 13.81 |
| (c) 直接予測 raw | 0.4193 | **3.49** |
| (d) 直接予測 norm | 0.4177 | 6.93 |

- **判定**: `not_superior_no_reattempt`。§6.2 の事前登録基準（logloss と較正誤差の両方で Stern に改善）を (c)(d) とも満たさず。ブートストラップ 95% CI（evaluator が別シードで独立再計算し整合確認）で direct は Stern に logloss 有意に劣後（CI がゼロを含まない）。**Stern 逆算が現状最良と確定**。
- **リーク**: なし（`any_leak_suspected=false`、direct は Stern 比 −4.7%/−4.3% とむしろ悪化。市場情報混入 grep も evaluator が独立実行し、オッズ使用は (a)(b) 比較対象算出のみで L1 特徴量への混入なし）。テスト 14/14 合格。本番資産（models/config/reports/parquet）非接触を確認。
- **evaluator 注記**: (a) Stern の p_win は formal 融合値 α=0, β=1.034（実質市場確率ベース）由来のため、本比較は「市場込み逆算 vs 市場なし直接予測」でもある。verdict は事前登録設計どおり有効だが、direct_raw が市場情報なしで較正誤差最小（3.49pp）を達成した点は較正後処理（isotonic 等）の余地を示す参考情報。
- **次アクション**: §12「優位でない」側 — λ2/λ3 再フィットや isotonic 後処理の検討へ。1・2・3着個別の直接予測への拡張は非推奨（後述の orchestrator 判断参照）。

---

## 1. 目的

複勝（3着以内）確率を LightGBM **binary 分類で直接予測**し、既存の
「単勝確率 p_win → Harville / Stern 式で逆算した複勝確率」と
**logloss・較正誤差・Brier score** で比較する。

### 背景（確定事実）

- L2 融合 OOS 測定で **α=0** が確定（`evaluation/reports/fusion_oos_fold2.json`）。
  L1 単勝スコアの情報は市場に完全に織り込まれている。単勝軸での優位は望めない。
- 一方、複勝・ペア確率は p_win からの**逆算**であり、較正誤差が実測されている
  （`betting/analysis/pair_probability_model_comparison.json`:
  ワイド較正誤差 Stern 8.3pp / Harville 13.5pp、馬連 Stern 28.6pp / Harville 16.0pp）。
- 複勝市場ベースライン（`evaluation/reports/place_baseline_oos.json`, TEST 2025+ 4,775レース）:
  1番人気の複勝的中率 65.5%、モデル top1 は 61.6%。
- 逆算式の実装: `prob_fusion/src/place_prob.py`（Stern。`place_prob_from_p_win()`）。

**仮説**: top3 という target を直接学習すれば、順序モデル経由の逆算より
複勝確率の較正が良くなる可能性がある（逆算は p_win の較正誤差と
順序モデル（λ2, λ3）の仮定誤差を両方引き継ぐため）。

**この実験が答える問い**: 「複勝的中を最もよく予測するのは、
直接 binary 予測か、Stern/Harville 逆算か」。市場超えの判定ではない。

---

## 2. 禁止特徴量の確認（プロジェクト憲法）

- [ ] オッズ系（`odds`, `win_odds`, `exp_win_odds` 等）を一切含まない
- [ ] 人気順位（`popularity`, `ninki`, `exp_popularity`）を含まない
- [ ] `market_log_odds` / `exp_market_log_odds` を含まない
- [ ] `init_score` を使わない（binary でも同様に禁止）

検証コマンド（実装完了時に実行し、実行ログを残す）:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/place_direct/ --include="*.py"
```

ヒットが許されるのはコメント・チェックコード内のみ。特徴量リスト・学習コードでのヒットは不合格。

**注意**: `pure_rank/data/02_features/features_v39_course_slim.parquet` をそのまま読む場合、
`market_leak_diagnostic` 実験が追加した `exp_*` 列が混在していないことを列名で検証すること
（同実験は別ファイルに出力しているはずだが、実装時に assert で確認する）。

---

## 3. モデル設計

### 3.1 target 定義

```
target_place = 1 if finish_rank <= 3 else 0
```

- `finish_rank` は当該レースの確定着順。**target は当該レース結果で正しい**
  （shift 不要。shift が必須なのは特徴量側のみ）。
- 除外条件適用後のデータで定義する（abnormal_code 除外により無効着順は入らないが、
  `finish_rank > 0` フィルタも既存標準どおり適用する）。

#### 7頭以下レースの扱い（top3 固定の妥当性と限界 — 明記事項）

JRA の複勝は出走頭数 7 頭以下では **2着まで払戻**（5〜7頭）、4頭以下は発売なしだが、
本実験では **target を top3 固定で統一**する。理由と限界:

- **理由**: 本実験の第一目的は「top3 到達確率の直接予測 vs 逆算」という
  *確率モデルの較正比較* であり、券種の払戻ルール再現ではない。
  比較対象の Stern/Harville 逆算も `p_win + p_2nd + p_3rd`（top3）を返すため、
  target を top3 に固定することで両者が**同一の事象を予測する公平な比較**になる。
- **限界**: 5〜7頭レースでは「複勝的中（2着以内）」と target（3着以内）が一致しないため、
  この実験の確率をそのまま少頭数レースの複勝 EV 計算に使ってはならない。
  betting 層へ接続する場合は将来の別仕様（頭数条件付き target または top2 モデル併設）が必要。
- **診断**: 評価時に頭数区分別（≤7頭 / 8頭以上）の指標も参考出力する（判定には使わない。§6.3）。

### 3.2 LightGBM パラメータ

`pure_rank/config/train_config.json` の値を読み込み、以下のみ上書きする:

```python
params = {
    "objective": "binary",       # lambdarank → binary
    "metric": "binary_logloss",  # ndcg → binary_logloss
    # 以下は本番 config から継承（変更しない）
    "num_leaves": 63,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "learning_rate": 0.05,
    "n_estimators": 800,         # early_stopping(50) が実効制御
}
# lambdarank 固有の label_gain / ndcg_eval_at / group は使わない
```

- ハイパーパラメータ探索は**行わない**（binary 向けチューニングで勝った場合、
  逆算側との比較が不公平になる。本番 LambdaRank と同一の複雑度制約で比較する）。
- 実験用 config は `pure_rank/experiments/place_direct/config.json` に置き、
  本番 `train_config.json` は読み取り専用参照とする（書き換え禁止）。

### 3.3 特徴量

- **v39_course_slim の特徴量セットをそのまま使用**
  （`pure_rank/data/02_features/features_v39_course_slim.parquet`）。
  特徴量の追加・削除・再生成は行わない。
- カテゴリ特徴量は本番 config `features.categorical` の 8 列をそのまま
  `lgb.Dataset(categorical_feature=...)` に渡す:
  `surface_code, track_condition_code, surface_condition, course_code, grade_code, distance_category, sex_code, weather_code`
- id 列（`race_id, ketto_num, race_date, finish_rank, is_win, lr_label`）は
  特徴量から除外する（本番 `features.id_cols` と同一）。

### 3.4 除外条件（本番標準と同一）

```python
df = df[
    (~df['grade_code'].isin([8, 9])) &
    (~df['abnormal_code'].isin([1, 3, 4])) &
    (df['horse_count'] >= 5) &
    (df['finish_rank'] > 0)
]
```

（features parquet が既にフィルタ済みならその旨をログで確認し、二重適用は無害なので適用してよい。）

### 3.5 fold2 限定・時系列分割（`market_leak_diagnostic` プロトコル踏襲）

| 区分 | 期間 | 用途 |
|------|------|------|
| train | race_date < 2023-01-01 | 学習 |
| early stopping | 2023-01-01 〜 2023-12-31 | early_stopping(50) のみ（弱汚染として明記） |
| 完全未見 | 2024-01-01 〜 | 学習・選択に一切使わない |
| **TEST** | **2025-01-01 〜** | 評価（既存 OOS と同一の 4,775 レース集合） |

- fold2 のみ学習する（3 フォールドの他 fold は作らない）。
- 5 シード（42, 43, 44, 45, 46）で学習し、**予測確率の単純平均**を最終予測とする。
- 参照実装パターン: `pure_rank/experiments/market_leak_diagnostic/`
  （build_features.py → train_fold2.py → export_scores.py の構成を踏襲。
  ただし本実験は特徴量追加がないため build_features 相当は「本番 parquet の読込＋target付与＋列検証」に縮退してよい）。

---

## 4. 確率化と正規化

binary 出力は既に確率だが、レース内制約「複勝(top3)確率の合計 = 3」
（正確には min(3, 頭数) だが、5頭未満除外済みなので常に 3）を自動では満たさない。
以下の **2 変種を両方**評価する:

- **(c) raw**: 5 シード平均の生予測確率 `p_raw`
- **(d) normalized**: レース内で合計 3 に正規化

```python
p_norm = p_raw * 3.0 / p_raw.groupby(race_id).transform('sum')
```

- 正規化後に `p_norm > 1` となる馬が出うる（強い馬が p_raw 高 & レース合計が 3 未満のとき）。
  この場合は **1.0 に clip し、超過分は同レースの他馬へ比例再配分**して合計 3 を保つ
  （反復: clip → 残余を未 clip 馬へ比例配分 → 収束まで最大 10 回、と仕様で固定）。
  clip 発生件数はレポートに記録する。
- logloss 計算時は数値安定化のため `eps=1e-12` で `[eps, 1-eps]` に clip する。

---

## 5. 比較対象（4 系列）

すべて **TEST 2025-01-01 以降、同一 4,775 レース集合、同一の馬集合**で per-horse に比較する。

| ID | 系列 | 算出方法 |
|----|------|---------|
| (a) | Stern 逆算 | fold2 OOS L1 スコア（`prob_fusion/data/scores_v39_course_slim_fold2_oos.parquet`）→ p_win（既存 OOS プロトコルと同一の確率化。`prob_fusion/src/oos_protocol.py` 準拠、λ2/λ3 は `fusion_oos_fold2.json` の formal 値: λ2=0.6018, λ3=0.6381）→ `place_prob_from_p_win()` |
| (b) | Harville 逆算 | 同 p_win → Harville（λ2=λ3=1.0、すなわち `place_prob_from_p_win(p_win, 1.0, 1.0)`。`betting/analysis/compare_pair_probability_models.py` の Harville 定義と整合させる） |
| (c) | 直接予測 raw | 本実験 binary モデル 5 シード平均 |
| (d) | 直接予測 normalized | (c) を §4 の方式でレース内合計 3 に正規化 |

**注**: (a)(b) の p_win 算出は既存 fold2 OOS 測定と完全に同一のコードパスを再利用すること
（新たに書き直すと比較が「実装差」を含んでしまう）。λ2/λ3 は TEST を見ずに
fit 済みの既存値を使う（後出し禁止）。

---

## 6. 評価プロトコル

### 6.1 指標（per-horse、TEST 全馬）

| 指標 | 定義 |
|------|------|
| logloss | `-mean(y*ln(p) + (1-y)*ln(1-p))`、y = target_place |
| Brier score | `mean((p - y)^2)` |
| 較正誤差（最大乖離pp） | 予測確率をビン分割し、各ビンの `\|mean(p) - mean(y)\|` の最大値（pp）。ビン方式は `betting/analysis/compare_pair_probability_models.py`（`pair_probability_model_comparison.json` を生成したもの）と**同一のビン境界・方式**を再利用する |
| mean_pred vs actual_rate | 全体平均予測確率と実測複勝率の差（大域バイアス確認） |

### 6.2 判定基準（事前確定 — 後出し禁止）

> **主判定**: (c) または (d) が、(a) Stern 逆算に対して
> **logloss と較正誤差（最大乖離pp）の両方**で改善していれば「直接予測が優位」と結論する。
> 片方のみの改善は「優位とは言えない（要追加検討）」とし、パラメータ調整による再挑戦は行わない。

- 比較の主軸は (a) Stern（現行のワイド較正最良の逆算法）。(b) Harville は参考系列。
- (c) と (d) の間では、上記基準を満たした系列を採用候補とする。両方満たす場合は
  logloss が小さい方。
- 有意性の目安: logloss 差はレース単位ブートストラップ（1,000 回）で 95% CI を付す。
  CI が 0 を跨ぐ場合は「差なし」と報告する。
- **この実験に Top-1 40% / Spearman 0.6 のリーク閾値をそのまま適用しないが**、
  直接予測の per-horse logloss が異常に良い場合（目安: (a) 比で 20% 以上の改善）は
  リーク疑いとして停止し evaluator へ報告する。target 生成ミス
  （特徴量に finish_rank 系列が混入等）が典型原因。

### 6.3 参考出力（判定に使わない）

- 頭数区分別（5–7頭 / 8頭以上）の logloss・較正誤差（§3.1 の限界の定量化）
- モデル (c)/(d) の top1 馬の複勝的中率 vs 既存ベースライン
  （`place_baseline_oos.json`: モデル 61.6%、1番人気 65.5%）
- 較正曲線データ（ビン別 mean_pred / actual / n）を JSON に保存

---

## 7. 隔離（本番非接触）

- コード・config・モデル・スコア・レポートすべて `pure_rank/experiments/place_direct/` 配下:

```
pure_rank/experiments/place_direct/
├── README.md              # 目的・実行手順・既知の制約（market_leak_diagnostic に倣う）
├── config.json            # 実験用設定（本番 train_config.json は読み取りのみ）
├── build_dataset.py       # features parquet 読込 + target付与 + 禁止列検証 + fold2分割
├── train_fold2.py         # 5シード binary 学習（early_stopping=2023）
├── export_probs.py        # TEST 期間の (c)(d) 予測、(a)(b) 逆算確率の算出
├── evaluate_place.py      # 4系列の logloss / Brier / 較正誤差 + 判定
├── data/                  # 中間データ（features は再生成せず参照のみ）
├── models/                # place_direct_seed{42..46}.txt
├── scores/                # probs_place_direct_fold2_oos.parquet
├── reports/               # place_direct_comparison.json 等（評価出力はここのみ）
└── tests/                 # §8 のテスト
```

- **書き換え禁止**: `pure_rank/models/`, `pure_rank/config/train_config.json`,
  `evaluation/reports/`, `prob_fusion/data/`, `pure_rank/data/02_features/*.parquet`
- `evaluation/reports/gate_summary.json` の合否判定にはこの実験結果を反映しない。
- 既存モジュール（`prob_fusion/src/place_prob.py`, `oos_protocol.py`,
  `betting/analysis/compare_pair_probability_models.py` のビン関数）は
  **import して再利用**する（コピー禁止。比較の同一性担保のため）。
  import 都合で関数抽出リファクタが必要な場合は最小限とし、既存テストが通ることを確認する。

---

## 8. TDD（テスト先行）

実装前に `pure_rank/experiments/place_direct/tests/` に以下を書く。
実行: `python -m pytest pure_rank/experiments/place_direct/tests/ -v`

| テスト | 検証内容 |
|--------|---------|
| `test_target.py::test_target_top3` | finish_rank 1,2,3 → 1、4以上 → 0、境界値 3/4 |
| `test_target.py::test_target_no_shift` | target がレース内の実着順と一致（shift されていないこと） |
| `test_target.py::test_filters_applied` | grade 8/9・abnormal 1/3/4・5頭未満・finish_rank<=0 が除外される |
| `test_normalize.py::test_sum_to_three` | 正規化後レース内合計が 3±1e-9 |
| `test_normalize.py::test_clip_redistribution` | p_norm>1 発生ケースで clip+再配分後も合計 3 かつ全馬 ≤1 |
| `test_normalize.py::test_uniform_case` | 全馬同確率 n 頭 → 各 3/n |
| `test_calibration.py::test_bins_match_reference` | ビン境界が `compare_pair_probability_models.py` と同一 |
| `test_calibration.py::test_perfect_calibration` | 合成データ（予測=実測率）で較正誤差 ≈ 0 |
| `test_calibration.py::test_known_miscalibration` | 既知の偏りを入れた合成データで期待どおりの誤差 pp |
| `test_split.py::test_fold2_boundaries` | train < 2023-01-01、ES = 2023 年のみ、2024+ が学習系に不在 |
| `test_no_market.py::test_feature_columns_clean` | 使用特徴量列名に odds/popularity/ninki/market_log_odds/init_score/exp_ が含まれない |

---

## 9. リーク防止チェックリスト（implementer が実行・ログ提出）

- [ ] `grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/experiments/place_direct/ --include="*.py"` — 特徴量・学習コードでヒットなし
- [ ] 特徴量 DataFrame の列名 assert（`test_no_market.py` を実データ列名でも実行）
- [ ] **target は shift 不要である根拠の確認**: target_place は当該レースの結果変数（予測対象）であり、特徴量には一切含めない。逆に特徴量側に finish_rank / is_win / lr_label / target_place が漏れていないことを id_cols 除外 assert で確認
- [ ] fold2 分割検証: 学習データ max(race_date) < 2023-01-01、ES データが 2023 年のみ、TEST 予測に 2024 年データが混ざっていない（`test_split.py` + 実行ログの期間サマリ）
- [ ] TEST レース集合が既存 OOS の 4,775 レースと一致（race_id 集合の一致 assert）
- [ ] (a)(b) の p_win が既存 fold2 OOS スコアから既存コードパスで算出されている（新規実装でない）

---

## 10. タスクリスト（implementer 実行順）

1. **ディレクトリ・README 作成**: `pure_rank/experiments/place_direct/` を §7 構成で作成。README に目的・隔離宣言・実行手順を記載。
2. **テスト先行**: §8 の全テストを書き、target/normalize/calibration/split の純関数を実装してテストを通す（この時点で学習は走らせない）。
3. **build_dataset.py**: `features_v39_course_slim.parquet` 読込 → 禁止列検証 → 除外フィルタ → target_place 付与 → fold2 分割（train / es2023 / test2025）→ `data/` に保存。期間・件数サマリをログ出力。
4. **train_fold2.py**: §3.2 パラメータで 5 シード binary 学習（categorical_feature 指定、early_stopping(50) は 2023 データ）。モデルを `models/` に保存。各シードの best_iteration・valid logloss をログ。
5. **export_probs.py**: TEST 期間で (c) 5シード平均 raw と (d) 正規化版を算出。(a)(b) は `scores_v39_course_slim_fold2_oos.parquet` + 既存 oos_protocol / place_prob コードパスで算出。4 系列を 1 つの parquet（race_id, ketto_num, target_place, p_stern, p_harville, p_direct_raw, p_direct_norm）に保存。
6. **evaluate_place.py**: §6 の指標・ブートストラップ CI・頭数区分別参考値を計算し、`reports/place_direct_comparison.json` に出力。§6.2 の判定を機械的に適用した verdict フィールドを含める。
7. **リーク防止チェックリスト実行**（§9）。grep 結果・assert ログを提出物に含める。
8. **evaluator へ引き渡し**: comparison JSON・学習ログ・チェックリスト結果。evaluator は §6.2 基準で合否を独立判定する。

### 実行コマンド（想定）

```bash
python -m pytest pure_rank/experiments/place_direct/tests/ -v
python pure_rank/experiments/place_direct/build_dataset.py
python pure_rank/experiments/place_direct/train_fold2.py
python pure_rank/experiments/place_direct/export_probs.py
python pure_rank/experiments/place_direct/evaluate_place.py
```

---

## 11. この実験でやらないこと（スコープ外）

- ハイパーパラメータ探索・特徴量追加（比較の公平性を壊すため）
- 複勝 EV/ベッティング接続（払戻ルール（7頭以下 top2）未対応のまま接続禁止）
- 本番 config・モデル・gate_summary の変更
- LambdaRank スコアの platt/isotonic 再較正との比較（優位が出た場合の次段候補として planner に差し戻す）
- top2 target モデル（少頭数対応）— 本実験の結果を見てから別仕様で判断

## 12. 成功後 / 失敗後の次アクション（参考）

- **直接予測が優位**: planner が「betting 層のワイド/複勝確率ソースを直接予測に差し替える」仕様を別途作成（ペア確率への拡張含む）。少頭数 top2 対応もその際に設計。
- **優位でない**: 逆算（Stern）が現状最良と確定。較正改善は λ2/λ3 の再フィットや isotonic 後処理の検討へ。
