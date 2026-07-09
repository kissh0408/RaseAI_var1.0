# 実装仕様書: Stern 複勝確率の較正改善（λ再フィット / isotonic 事後較正）— 2026-07-09

**ステータス**: 承認済み（ユーザー承認済みフェーズ）— 実装待ち
**担当**: implementer（本仕様に従い実装）→ evaluator（合否判定）
**実験ディレクトリ**: `pure_rank/experiments/place_calibration/`（この配下に完結させる）
**前フェーズ**: `docs/specs/2026-07-09-place-direct-prediction-spec.md`（完了・evaluator サインオフ済み、verdict = `not_superior_no_reattempt`）

---

## 1. 目的

複勝(top3)確率を返す **Stern 逆算式**（`prob_fusion/src/place_prob.py`）の較正を改善する。

### 背景（確定事実）

- 前フェーズで「直接 binary 予測は Stern に勝てない（logloss 有意劣後、ブートストラップ CI がゼロを含まない）」が確定。**Stern 逆算が現状最良**。
- ただし較正誤差は direct_raw 3.49pp < Stern 4.53pp であり、**Stern の較正には改善余地がある**。前フェーズの次アクションとして「λ2/λ3 再フィット手法の見直し」「isotonic 等の事後較正」が指定された。
- 現行 `fit_stern_lambda()`（`prob_fusion/src/place_prob.py` L80-107）は **Brier スコア最小化**（L-BFGS-B、bounds [0.1, 3.0]）で λ を推定している。評価軸は logloss なので、目的関数の不一致が第一の改善仮説。
- 現行 formal 値（比較ベースライン）: **lam2=0.6018, lam3=0.6381**（`evaluation/reports/fusion_oos_fold2.json`、fit=2023-01-01..2024-12-31、fit_n_races=6786）。
- 現行 Stern の TEST 実測（前フェーズ (a) 系列、`pure_rank/experiments/place_direct/reports/place_direct_comparison.json`）: **logloss=0.4003、calibration_max_error_pp=4.53**。

**この実験が答える問い**: 「logloss 目的の λ 再フィット、または isotonic 事後較正は、現行 Stern（Brier fit λ 固定）を logloss と較正誤差の両方で改善するか」。市場超えの判定ではない。

---

## 2. 禁止特徴量・市場情報境界の確認（プロジェクト憲法）

本実験は新しい特徴量を一切作らない（p_win → 較正のみ）。それでも以下を確認する:

- [ ] 実験コードが L1 特徴量にオッズ系（`odds`, `win_odds`）・人気（`popularity`, `ninki`）・`market_log_odds`・`init_score` を追加しない
- [ ] isotonic / λ fit の入力は **p_win（既存 fold2 OOS コードパス由来）と複勝実績（finish_rank<=3）のみ**。オッズ・人気を較正の説明変数に使わない
- [ ] p_win が formal 融合（α=0, β=1.034）由来で市場確率 q を含むのは **L2 の条件付きロジット統合として許容済み**（CLAUDE.md 市場情報境界）。本実験はその出力を消費するだけであり、q・オッズを新規に取り込まない

検証コマンド（実装完了時に実行し、実行ログを残す）:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/place_calibration/ --include="*.py"
```

ヒットが許されるのはコメント・チェックコード内のみ。較正の入力変数・fit コードでのヒットは不合格。

---

## 3. 設計

### 3.0 共通: p_win の算出（全系列で同一）

- **前フェーズ (a) 系列と完全同一のコードパス**を再利用する:
  `prob_fusion/data/scores_v39_course_slim_fold2_oos.parquet`（fold2 OOS L1 スコア）
  → `prob_fusion/src/oos_protocol.py` 準拠の確率化 → p_win。
- 新規実装・コピーは禁止（比較に「実装差」を混入させないため）。
- レース集合・除外条件も前フェーズと同一（TEST 2025+ は 4,775 レース / 66,020 頭。
  実装時に race_id 集合の一致を assert する）。

### 3.1 アプローチ A: λ2/λ3 再フィット手法の見直し

**A1（最有力候補）: logloss 目的の λ 再フィット**

- `fit_stern_lambda()` と同型のオプティマイザ（L-BFGS-B、bounds [0.1, 3.0]、初期値は現行 formal 値）で、目的関数を Brier から **per-horse logloss** に変える:

```
loss(λ2, λ3) = -mean over fit horses of [ y*ln(p_place) + (1-y)*ln(1-p_place) ]
y = 1 if finish_rank <= 3 else 0、p_place は [eps, 1-eps] に clip（eps=1e-12）
```

- Stern 式本体（`place_prob_from_p_win()`）は変更しない。パラメータ数: 2。
- fit 関数は実験ディレクトリ内に実装する（`place_prob.py` は import のみ、書き換え禁止）。

**A2: 頭数帯別 λ の logloss フィット**

- 頭数帯を **2 帯に事前固定**する: **5–7頭 / 8頭以上**（後出し変更禁止）。
  根拠: 前フェーズ §3.1 と同じ区分（複勝払戻ルールの境界かつ、Stern の順序仮定誤差が
  少頭数で変質しやすいという学習期間の一般知見）。3 帯以上への細分化は行わない。
- 各帯で A1 と同じ logloss 目的フィット。パラメータ数: 4（λ2/λ3 × 2帯）。上限内。
- TEST 適用時は当該レースの頭数で帯を選ぶ（レース単位で一意に決まり、リーク要素なし）。

### 3.2 アプローチ B: isotonic 事後較正

- **入力**: 現行 Stern 出力確率（λ は formal 値 0.6018/0.6381 に**固定**。再フィットと直交させる）。
- fit 期間の per-horse ペア `(p_stern, y_place)` に対し
  `sklearn.isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")` を学習し、TEST に適用する。
- **sklearn の依存確認は implementer のタスク**（§10-0）。プロジェクト依存に既に含まれるはずだが、実行環境で import を確認しログに残す。
- isotonic 後はレース内合計=3 が崩れる。以下の 2 変種を**事前登録**し両方評価する:

**B1: isotonic raw** — isotonic 出力をそのまま使う（正規化なし）。

**B2: isotonic normalized** — レース内で合計 3 に正規化し、p>1 は
前フェーズ §4 と**同一方式**で処理する: 1.0 に clip → 超過分を未 clip 馬へ比例再配分 →
収束まで最大 10 回反復。clip 発生件数をレポートに記録。実装は前フェーズの正規化関数を
**import 再利用**する（`pure_rank/experiments/place_direct/` 内の該当純関数。コピー禁止。
import 都合の最小リファクタは可、その場合 place_direct のテストが通ることを確認）。

### 3.3 やらないこと（系列の複雑化防止）

- A×B の組み合わせ（再フィット λ + isotonic）は本フェーズでは登録しない（採用系列が出た場合の次段候補。§12）
- λ の bounds・オプティマイザ変更、3 帯以上の頭数区分、距離/馬場別 λ、spline/Platt 較正
- 直接予測モデルの再学習（前フェーズで決着済み）

---

## 4. 比較系列（事前登録 — 全 5 系列、追加・変更禁止）

すべて TEST 2025-01-01 以降、同一 4,775 レース / 66,020 頭で per-horse 比較。

| ID | 系列 | λ2/λ3 | fit 目的関数 | fit 期間 | 事後較正 | 正規化 | パラメータ数 |
|----|------|-------|------------|---------|---------|--------|------------|
| **S0** | 現行 Stern（基準） | 0.6018 / 0.6381 固定 | Brier（既存 formal） | 2023–2024（既存） | なし | なし | 0（新規なし） |
| **A1** | logloss λ 再フィット | 再fit（global） | logloss | 2023–2024 | なし | なし | 2 |
| **A2** | 頭数帯別 λ 再フィット | 再fit（5–7頭 / 8頭以上） | logloss | 2023–2024 | なし | なし | 4 |
| **B1** | isotonic raw | 0.6018 / 0.6381 固定 | —（isotonic を 2023–2024 で fit） | 2023–2024 | isotonic | なし | ノンパラ |
| **B2** | isotonic normalized | 0.6018 / 0.6381 固定 | 同上 | 2023–2024 | isotonic | 合計3（clip+再配分） | ノンパラ |

**S0 の基準値（事前固定）**: logloss=0.4003、calibration_max_error_pp=4.53。
実装時に S0 を同一パイプラインで再計算し、この値と一致（logloss ±0.0005、較正誤差 ±0.05pp）
することを検証してから A/B 系列を評価する（パイプライン検証）。

---

## 5. fit 期間の定義と比較の公平性（明確化事項）

- **全再フィット系列（A1/A2/B1/B2）の fit 期間は 2023-01-01..2024-12-31 に統一する。**
  これは現行 formal λ の fit 期間（`fusion_oos_fold2.json`: fit_n_races=6786）と**同一**である。
- **公平性の論拠**: 比較基準 S0 の λ は 2023–2024 で fit されている。新系列だけ 2024 単年
  （約半分のデータ）で fit すると、「手法の差」と「fit データ量の差」が分離できない。
  fit 期間を S0 と揃えることで、差分は純粋に目的関数・較正手法に帰属する。
- **Rule 3 との整合**: Rule 3（後出し禁止）の本質は「TEST(2025+) を見て fit・調整しない」こと。
  2023・2024 はいずれも TEST 外であり、既存 formal fit も同じ期間を使用済み。
  なお 2023 は fold2 の early stopping に使用された弱汚染期間だが、これも S0 の formal fit と
  同条件であり、比較の公平性を損なわない（`fusion_oos_fold2.json` の 2024 単年感度分析で
  λ の期間感度が小さいことも確認済み: lam2=0.5987/lam3=0.6505）。
- **感度分析（参考出力・判定に使わない）**: 採用候補となった系列についてのみ、
  fit=2024 単年での再フィット値と TEST 指標を参考出力する（Rule 3 の「VALID=2024 のみ」
  解釈との整合確認用。判定には一切使わない）。
- fit 期間の複勝実績・p_win は §3.0 と同一コードパスで取得する。fold2 OOS スコア parquet が
  fit 期間（2023–2024）をカバーしていることを実装時に assert する（formal λ fit が同じ
  ソースで行われた以上カバーされているはずだが、件数 6,786 レースとの整合を確認しログに残す）。
- **TEST(2025+) での評価は一度だけ**。TEST 結果を見た後の λ 再調整・ビン変更・系列追加は禁止。

---

## 6. 評価プロトコル

### 6.1 指標（per-horse、TEST 全馬 — 前フェーズ §6.1 と同一）

| 指標 | 定義 |
|------|------|
| logloss | `-mean(y*ln(p) + (1-y)*ln(1-p))`、y = 複勝実績（finish_rank<=3）、p は `[1e-12, 1-1e-12]` に clip |
| Brier score | `mean((p - y)^2)`（参考） |
| 較正誤差（calibration_max_error_pp） | `betting/analysis/compare_pair_probability_models.py` の `calibration_max_error_pp` を **import 再利用**（ビン境界・方式を完全同一にする。再実装禁止） |
| mean_pred vs actual_rate | 大域バイアス確認（参考） |

### 6.2 判定基準（事前登録 — 後出し禁止）

> **主判定**: 系列 A1/A2/B1/B2 のうち、S0（logloss=0.4003 / calibration_max_error_pp=4.53）に対し
> **logloss と calibration_max_error_pp の両方**で改善した系列のみ「採用候補」。
> **片方のみの改善は不採用・パラメータ調整による再挑戦なし。**
> 複数系列が両方改善した場合は **logloss 最小の系列を採用候補**とする。

- 有意性: 採用候補系列と S0 の logloss 差について**レース単位ブートストラップ（1,000 回）で
  95% CI** を付す。CI が 0 を跨ぐ場合は「差なし」と報告し、採用しない
  （両指標の点推定改善 + logloss 差 CI がゼロを含まない、が採用の必要条件）。
- 全系列の指標・CI・fit された λ 値・isotonic の入出力ノット数をレポート JSON に記録する。
- **リーク/異常停止**: いずれかの系列の logloss が S0 比 **10% 以上**改善した場合
  （目安 0.360 未満）は較正のみでは説明困難な異常として実装を停止し evaluator へ報告する
  （典型原因: fit/TEST 分割ミス、TEST データの fit 混入、y 定義ミス）。
- 本実験に Top-1 40% / Spearman 0.6 のリーク閾値は直接適用しない（順位は不変。
  isotonic は単調変換、λ 変更も p_win の順位を変えないため Top-N は S0 と同一になるはず。
  **これ自体を検証項目とする**: 各系列の top1 選択馬が S0 と一致することを assert し、
  不一致なら実装バグとして停止）。
  注: A2 はレース間で λ が異なるがレース内順位は保存される。isotonic の平坦区間で
  タイが生じた場合は元の p_stern 順で安定ソートし順位保存を保証する。

### 6.3 参考出力（判定に使わない）

- 頭数区分別（5–7頭 / 8頭以上）の logloss・較正誤差（全系列。A2 の帯別効果の定量化）
- 較正曲線データ（ビン別 mean_pred / actual / n）を全系列分 JSON に保存
- 採用候補系列の fit=2024 単年感度分析（§5）
- 前フェーズ direct_raw の較正誤差 3.49pp との位置関係（文脈情報）

---

## 7. 隔離（本番非接触）

- コード・config・レポートすべて `pure_rank/experiments/place_calibration/` 配下:

```
pure_rank/experiments/place_calibration/
├── README.md               # 目的・隔離宣言・実行手順
├── config.json             # 系列定義・fit期間・S0基準値（ハードコード禁止の受け皿）
├── build_dataset.py        # fold2 OOSスコア読込 → p_win算出（既存コードパス）→ fit/TEST分割
├── fit_calibrators.py      # A1/A2 logloss λfit + B1/B2 isotonic fit（fit期間のみ使用）
├── export_probs.py         # TEST 5系列の p_place を parquet 出力
├── evaluate_calibration.py # 指標・ブートストラップCI・判定 verdict を JSON 出力
├── data/                   # 中間データ
├── models/                 # fitted λ / isotonic（joblib等）
├── reports/                # place_calibration_comparison.json 等（評価出力はここのみ）
└── tests/                  # §8 のテスト
```

- **書き換え禁止**: `prob_fusion/src/place_prob.py`、`prob_fusion/src/oos_protocol.py`、
  `evaluation/reports/`、`prob_fusion/data/*.parquet`、`pure_rank/data/02_features/*.parquet`、
  `pure_rank/models/`、`pure_rank/config/train_config.json`
- `evaluation/reports/gate_summary.json` の合否判定にはこの実験結果を反映しない。
- 既存モジュールは **import して再利用**（コピー禁止）:
  `prob_fusion/src/place_prob.py`（Stern 式本体）、`prob_fusion/src/oos_protocol.py`（p_win 確率化）、
  `betting/analysis/compare_pair_probability_models.py`（`calibration_max_error_pp`）、
  `pure_rank/experiments/place_direct/` の正規化純関数（B2 用）。
  import 都合で関数抽出リファクタが必要な場合は最小限とし、既存テストが通ることを確認する。

---

## 8. TDD（テスト先行）

実装前に `pure_rank/experiments/place_calibration/tests/` に以下を書く。
実行: `python -m pytest pure_rank/experiments/place_calibration/tests/ -v`

| テスト | 検証内容 |
|--------|---------|
| `test_lambda_fit.py::test_logloss_objective_value` | 小さな合成レース集合で目的関数値が手計算 logloss と一致 |
| `test_lambda_fit.py::test_recovers_known_lambda` | 既知 λ で生成した合成データから logloss fit が λ を近似回復（許容誤差を明記） |
| `test_lambda_fit.py::test_bounds_respected` | fit 結果が bounds [0.1, 3.0] 内 |
| `test_lambda_fit.py::test_band_split` | A2 の頭数帯割当: 5–7頭 / 8頭以上の境界（7頭→帯1、8頭→帯2） |
| `test_isotonic.py::test_monotonic` | isotonic 出力が入力順序に対し単調非減少 |
| `test_isotonic.py::test_output_range` | 出力が [0, 1]、out_of_bounds="clip" の挙動（fit 範囲外入力） |
| `test_isotonic.py::test_rank_preserved` | 較正後のレース内 top1 が較正前と一致（タイは元順で安定） |
| `test_normalize.py::test_sum_to_three` | B2 正規化後レース内合計 3±1e-9（place_direct の関数 import で担保） |
| `test_normalize.py::test_clip_redistribution` | p>1 ケースで clip+再配分後も合計 3 かつ全馬 ≤1 |
| `test_split.py::test_fit_test_boundaries` | fit ⊆ 2023-01-01..2024-12-31、TEST ⊇ 2025-01-01、重複ゼロ |
| `test_split.py::test_test_race_set` | TEST race_id 集合が既存 OOS の 4,775 レースと一致 |
| `test_no_market.py::test_calibration_inputs_clean` | 較正 fit の入力 DataFrame 列に odds/popularity/ninki/market_log_odds/init_score が含まれない |
| `test_baseline.py::test_s0_reproduction` | S0 の再計算が基準値（logloss 0.4003±0.0005、較正誤差 4.53±0.05pp）と一致 |

---

## 9. リーク防止チェックリスト（implementer が実行・ログ提出）

- [ ] `grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" pure_rank/experiments/place_calibration/ --include="*.py"` — 較正入力・fit コードでヒットなし（コメント・チェックコードのみ許容）
- [ ] fit/TEST 分割検証: fit データ max(race_date) <= 2024-12-31、TEST min(race_date) >= 2025-01-01、race_id 重複ゼロ（`test_split.py` + 実行ログの期間・件数サマリ）
- [ ] TEST レース集合が既存 OOS の 4,775 レース / 66,020 頭と一致（assert）
- [ ] p_win が `scores_v39_course_slim_fold2_oos.parquet` + 既存 `oos_protocol.py` コードパスで算出されている（新規実装でない）
- [ ] fit 期間レース数が formal λ fit（6,786 レース）と整合することを確認しログに残す
- [ ] S0 再現検証合格（§4）**の後に** A/B 系列を評価している（実行順のログ）
- [ ] 全系列で top1 選択馬が S0 と一致（順位保存 assert。§6.2）
- [ ] TEST 評価は一度だけ実行し、結果閲覧後の fit・系列変更を行っていない

---

## 10. タスクリスト（implementer 実行順）

0. **依存確認**: `sklearn`（`sklearn.isotonic.IsotonicRegression`）と `scipy.optimize` の import 可否を実行環境で確認しログに残す。不可の場合は実装前に orchestrator へ報告（勝手に requirements を変更しない）。
1. **ディレクトリ・README・config 作成**: §7 構成。config.json に系列定義・fit 期間・S0 基準値を記載。
2. **テスト先行**: §8 の全テストを書き、λfit 目的関数・帯割当・isotonic ラッパ・正規化・分割の純関数を実装してテストを通す（この時点で実データ fit は走らせない）。
3. **build_dataset.py**: fold2 OOS スコア読込 → 既存コードパスで p_win 算出 → y_place 付与 → fit(2023–2024) / TEST(2025+) 分割 → `data/` に保存。期間・件数サマリをログ出力。
4. **S0 再現検証**: TEST で S0 を計算し §4 の許容誤差内で基準値と一致することを確認（`test_baseline.py` を実データで実行）。不一致ならここで停止し原因調査（A/B 評価に進まない）。
5. **fit_calibrators.py**: fit 期間のみで A1（global logloss λ）、A2（2 帯別 logloss λ）、B1/B2（isotonic）を fit。fitted λ・isotonic ノット数を `models/` とログに保存。
6. **export_probs.py**: TEST で 5 系列の p_place を算出し 1 つの parquet（race_id, ketto_num, y_place, p_s0, p_a1, p_a2, p_b1, p_b2）に保存。
7. **evaluate_calibration.py**: §6 の指標・レース単位ブートストラップ 1,000 回 CI・頭数区分別参考値を計算し、`reports/place_calibration_comparison.json` に出力。§6.2 の判定を機械的に適用した verdict フィールド（`adopted_series` または `no_improvement_no_reattempt`）を含める。
8. **感度分析（採用候補が出た場合のみ）**: 当該系列の fit=2024 単年版を参考出力（§5）。
9. **リーク防止チェックリスト実行**（§9）。grep 結果・assert ログを提出物に含める。
10. **evaluator へ引き渡し**: comparison JSON・fit ログ・チェックリスト結果。evaluator は §6.2 基準で合否を独立判定する。

### 実行コマンド（想定）

```bash
python -m pytest pure_rank/experiments/place_calibration/tests/ -v
python pure_rank/experiments/place_calibration/build_dataset.py
python pure_rank/experiments/place_calibration/fit_calibrators.py
python pure_rank/experiments/place_calibration/export_probs.py
python pure_rank/experiments/place_calibration/evaluate_calibration.py
```

---

## 11. この実験でやらないこと（スコープ外）

- A×B 組み合わせ（logloss λ + isotonic の重ね掛け）— 採用系列が出た場合の次段候補
- λ bounds・オプティマイザの変更、3 帯以上・条件別（距離/馬場）λ
- Platt / spline / beta calibration 等の他の較正手法
- 直接 binary 予測の再学習（前フェーズで決着済み）
- 複勝 EV / ベッティング接続（7 頭以下 top2 払戻ルール未対応のまま接続禁止）
- 本番 `place_prob.py`・`fusion_config.json`・`evaluation/reports/`・本番 parquet の変更
- ワイド・馬連ペア確率への較正適用（採用系列が出た場合の別仕様）

## 12. 成功後 / 失敗後の次アクション（参考）

- **採用系列あり**: planner が「本番 `place_prob.py` / fusion 設定への昇格仕様」を別途作成
  （formal λ の更新 or isotonic モデルの本番組込み、ワイド較正への波及検証、
  A×B 組み合わせの追加検証を含む）。昇格前に refactorer の市場情報チェックと
  evaluator の再現確認を必須とする。
- **採用系列なし**: Stern（現行 Brier fit λ）の較正 4.53pp が現状の到達点と確定。
  複勝確率較正の改善はクローズし、リソースを他レイヤー（データ拡張・α>0 ゲート系）へ移す。
