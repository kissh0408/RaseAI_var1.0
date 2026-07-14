# 実装仕様書: P1 セグメント条件付き α の異質性検定 — 2026-07-09

**作成者**: planner
**承認済み提案**: `docs/specs/2026-07-09-next-performance-improvement-proposal.md` §3 P1 / §5 / §6
**実装先**: `pure_rank/experiments/alpha_segments/`（隔離実験・本番非接触）
**実装担当**: implementer（本仕様書はコードを含まない）

---

## 1. 目的

fold2 OOS 正式測定で α=0（L1 スコアは市場に情報を追加しない）が確定した。これは
「**全レース平均**」の結論であり、「あらゆる部分集合で成立する」ことは意味しない。
本 Phase では、市場の情報集約が構造的に弱いと**ドメイン知識のみから事前登録した
5 セグメント**について、セグメント内で条件付きロジット（α, β）を再推定し、
α > 0 の LRT 検定を行う。

- 成功時: 当該セグメント限定で L2 統合 → L3 EV が初めて意味を持つ（初の正 EV 候補領域）。
- 全滅時: 「市場効率性はセグメント横断的に成立（fold2 OOS 基準）」という事実が確定し、
  P2（Track B 調教時系列）へ移行する。**セグメントの後出し追加・再定義による延長は禁止。**

---

## 2. 禁止特徴量の確認

- [x] セグメント判定に使う列（§4 参照: `hist_last_rank`, `horse_count`,
  `track_condition_code`, `course_code`, `race_condition_code`）はいずれも
  オッズ・人気・市場由来ではないことを確認した（レース属性 + shift 済み過去走属性のみ）。
- [x] 市場確率 q（`ln_market_q`）は **L2 条件付きロジットの統合変数としてのみ**使用する
  （プロジェクト憲法で許容済みの範囲。LightGBM 等の特徴量には一切入れない）。
- [x] z の二重使用なし（`pure_score_z` は α·z の一箇所のみ）。

---

## 3. 事前登録判定基準（提案書 §6 の転記。TEST を見る前に固定・変更禁止）

| 項目 | 固定内容 |
|---|---|
| 使用スコア | `scores_v39_course_slim_fold2_oos.parquet`（fold2 のみ。15 モデル平均は使用禁止） |
| 検定プロトコル | αゲートと同一: fit=2023 → eval=2024。TEST(2025+) は生き残り候補のみ各 1 回 |
| セグメント定義 | §3 P1 の 5 案を基に、ステップ 1 の n 集計後に確定・本書へ追記してから実装。以後の追加・変更・削除は禁止 |
| 最小サンプル | fit 期間 eval 年（2024）で n≥300 レース未満のセグメントは検定対象外（事前除外） |
| 一次判定（成功） | セグメント内 α の LRT p < 0.01/K（K=確定セグメント数、Bonferroni）かつ ΔLL/race > 0 |
| 二次判定（TEST、1 回のみ） | 当該セグメントの TEST logloss が市場を下回る、かつセグメント内 Top-1 がリーク閾値（>40%）に達しない |
| 失敗判定 | 全セグメントで補正後 p ≥ 0.01/K → 「市場効率性はセグメント横断的に成立（fold2 OOS 基準）」と `gate_summary.json` に記録し P1 終了。セグメントの後出し追加・再定義による延長は禁止 |
| リーク・停止条件 | いずれかのセグメントで Top-1 > 40% または Spearman > 0.6 → 即停止・evaluator 報告（合格ではなく危険信号） |
| 失敗時の次アクション | P2（Track B、事前登録済み 5 候補）へ移行。P1 の結果いかんに関わらず P2 は実施する |

---

## 4. セグメント定義（確定。列名・コード値は実データで検証済み）

判定は全て**レース単位**（レース内の全馬に同一のセグメントフラグを付与する）。

| ID | セグメント | 使用列（ソース） | 条件式（レース単位） | 根拠（ドメイン知識） |
|----|-----------|----------------|--------------------|--------------------|
| S1 | 新馬・未出走馬中心レース | `hist_last_rank`（features, shift 済み過去走） | レース内の「過去走ゼロ馬」比率 ≥ 0.5。過去走ゼロ = `hist_last_rank` が NaN | 過去走情報が市場にも乏しい（調教・血統の相対価値が高い） |
| S2 | 少頭数レース | `horse_count`（features/RA） | `horse_count <= 8` | 流動性が低く市場集約が粗い |
| S3 | 重・不良馬場 | `track_condition_code`（features/RA） | `track_condition_code in (3, 4)`（3=重, 4=不良。0=未設定は非該当） | 当日変化する条件で市場の反応が遅れる可能性 |
| S4 | ローカル開催場 | `course_code`（features/RA） | `course_code in (1, 2, 3, 4, 7, 10)` = 札幌・函館・福島・新潟・中京・小倉（中央4場 = 東京5・中山6・京都8・阪神9 **以外**） | 投票参加者の情報水準が中央場より低い可能性 |
| S5 | 低クラス条件戦 | `race_condition_code`（`RA_preprocessed.parquet` から race_id で merge） | `race_condition_code in (703, 5)` = 未勝利(703)・1勝クラス(005) | 注目度が低く市場参加者の分析投資が薄い |

### 定義に関する固定済みの設計判断（fit 期間のドメイン知識のみで決定）

- **S1**: `hist_last_rank` の NaN はデータ期間（2015〜）内に JRA 過去走が無いことを意味する。
  地方・海外からの転入馬が「過去走ゼロ」として混入しうるが許容ノイズとする。
  閾値 0.5 は「レースの過半数が初出走」という自然な定義（新馬戦は比率 1.0 になる）。
- **S4**: 「中央4場以外」を採用（中京を含む）。福島・新潟・小倉のみの狭い定義は不採用
  （事前に決定済み。実装後の切替は後出しとなるため禁止）。
- **S5**: `features_v39_course_slim.parquet` の `grade_code==1` は新馬〜3勝クラスを全部
  含み広すぎるため、JV 競走条件コード（#2007）を持つ `RA_preprocessed.parquet` の
  `race_condition_code` で 未勝利+1勝クラス に限定する。
  参考: fit 期間の分布は 701=新馬(604), 703=未勝利(2458), 5=1勝(1867), 10=2勝(936),
  16=3勝(424), 999=OP(621)。
- セグメントは相互に**重複してよい**（S1 は S5 とほぼ包含関係）。各セグメントを独立に
  検定し、多重性は Bonferroni（p < 0.01/K）で制御する。

### 参考: fit 期間の暫定レース数（planner が features parquet 上で確認。TEST 非接触）

| ID | 2023 | 2024 | n≥300（2024）見込み |
|----|------|------|--------------------|
| S1 | 307 | 311 | ボーダー（merge 後の脱落で 300 を割れば除外） |
| S2 | 180 | 220 | **除外見込み**（n<300） |
| S3 | 538 | 336 | 通過見込み |
| S4 | 1,506 | 1,516 | 通過 |
| S5（703+005） | ≈2,160/年 | ≈2,160/年 | 通過 |

**正式な n 集計（Stage 1）は、merge 後のゲート用データセット（scores ∩ features ∩ RA ∩
オッズ付与成功行）に対して implementer が再実行し、その結果で K を確定する。**
上表は暫定値であり、確定値が優先する。

---

## 5. データと既存 API（実測確認済み）

### 5.1 入力ファイル

| データ | パス | 備考 |
|---|---|---|
| fold2 OOS スコア | `C:\Users\syugo\AI\RaceAI_var1.0\pure_rank\data\03_scores\scores_v39_course_slim_fold2_oos.parquet` | 列: `race_id, race_date, ketto_num, horse_num, horse_number, course_code, finish_rank, pure_score, pure_score_z`。158,180 行 / 11,561 レース |
| 特徴量（セグメント列） | `pure_rank\data\02_features\features_v39_course_slim.parquet` | 132 列。使用列: `race_id, horse_num, race_date, finish_rank, hist_last_rank, horse_count, track_condition_code, course_code` |
| 競走条件コード | `pure_rank\data\01_preprocessed\RA_preprocessed.parquet` | 使用列: `race_id, race_condition_code`（S5 用。race 単位） |
| 市場オッズ→q | `evaluation/odds_loader.py::attach_odds_from_se_parquet` → `prob_fusion/src/market_prob.py::attach_market_q` | `alpha_gate.py::_ensure_gate_inputs` と同一経路。L2 統合変数としてのみ使用 |

### 5.2 再利用する既存 API（コピー禁止、import で再利用）

`prob_fusion/src/fit_fusion.py`:

- `build_race_tuples(df: pd.DataFrame, x_col: str | None = None) -> list[RaceTuple]`
  — `race_id, horse_num, finish_rank, pure_score_z, ln_market_q` を持つ df から
  `(z, ln_q, winner_idx)` タプル列を構築（x_col は本 Phase では使わない）。
- `fit_fusion_mle(races, *, alpha_bounds=(0.0,5.0), beta_bounds=(0.0,3.0), gamma_bounds=(0.0,5.0), market_only=False, gamma_fixed_zero=False) -> FusionParams`
  — H1（α, β 自由）は既定呼び出し、H0（α=0, β のみ）は `market_only=True`。
- `likelihood_ratio_test(races, fitted, *, alpha_bounds=(0.0,5.0), beta_bounds=(0.0,3.0)) -> dict`
  — **α の LRT**（H0: α=0 vs H1: α 自由）。返り値: `h0_nll, h1_nll, lr_statistic, p_value, h0_beta`。
  ※ γ 用の `gamma_likelihood_ratio_test` は本 Phase では使わない（検定対象は α）。
- `mean_logloss(df, alpha, beta, *, x_col=None, gamma=0.0) -> float`
- `top1_hit_rate(df, alpha, beta, *, x_col=None, gamma=0.0) -> float`

`evaluation/alpha_gate.py`:

- `split_alpha_gate_cv(df, *, race_date_col="race_date", fit_year=2023, eval_year=2024) -> (fit_df, eval_df)`
- `_ensure_gate_inputs(df)` 相当の入力整備（`race_id` を str 化、`horse_num` 補完、
  オッズ付与 → `attach_market_q`）。private 関数のため、同等処理を実験側ラッパーで
  公開 API（`attach_odds_from_se_parquet` / `attach_market_q`）を直接呼んで構成する。

`prob_fusion/src/oos_protocol.py`:

- `split_oos_periods(df) -> (fit_df, test_df)`（FIT=2023-01-01..2024-12-31 / TEST=2025-01-01..）
  — **Stage 3（TEST 二次判定）でのみ使用**。

### 5.3 α LRT の統計的注記

`alpha_bounds=(0.0, 5.0)` の下限境界上の検定であるため、χ²(df=1) ベースの p 値は
**保守的**（真の分布は 0.5·χ²₀ + 0.5·χ²₁ の混合）。保守側に倒れるのは本 Phase の
目的（偽陽性の抑制）と整合するため、既存実装のまま使用し、レポートに注記する。

---

## 6. 実装構成（隔離実験・先例パターン準拠）

```
pure_rank/experiments/alpha_segments/
├── README.md               # 目的・隔離宣言・市場情報境界・実行手順（place_calibration の README に準拠）
├── config.json             # セグメント定義（列名・コード値・閾値）、期間定数、n_min=300、
│                           #   LRT 基準 p=0.01、リーク停止閾値（top1>0.40, spearman>0.60）
├── segments_lib.py         # 純関数のみ: セグメントフラグ付与（レース単位）、debut比率計算、
│                           #   n≥300 除外判定、Bonferroni 閾値計算。市場情報列に一切触れない
├── build_dataset.py        # scores + features + RA + オッズ付与 → ゲート用データセット parquet
├── run_stage1_counts.py    # Stage 1: fit 期間セグメント別 n 集計 → results/stage1_counts.json、K 確定
├── run_stage2_lrt.py       # Stage 2: 確定セグメント別 α/β MLE + LRT + ΔLL/race → results/alpha_segments.json
├── run_stage3_test.py      # Stage 3: 一次通過セグメントのみ TEST 1 回（存在しなければ実行しない）
├── data/                   # 実験内データセット（gate_dataset.parquet 等）
├── results/                # stage1_counts.json / alpha_segments.json / alpha_segments_test.json
└── tests/                  # §9 の TDD テスト
```

### 隔離宣言（README に明記すること）

本実験は上記ディレクトリに完結する。以下は**書き換えない**:
`pure_rank/models/`、`pure_rank/data/02_features/*.parquet`、`pure_rank/data/03_scores/`、
`prob_fusion/data/`、`prob_fusion/src/`、`evaluation/reports/gate_summary.json`、
`pure_rank/config/train_config.json`。
`gate_summary.json` への結果反映（合格・全滅いずれの場合も）は **evaluator の判定後に
別タスクとして** `evaluation/update_gate_summary.py` 経由で行い、本実験のスクリプトは
実験ディレクトリ内の `results/` のみに出力する。

---

## 7. 手順（3 ステージ。Rule 3 遵守）

### Stage 1: セグメント確定（TEST 非接触）

1. `build_dataset.py`: fold2 OOS スコア × features（セグメント列・`race_date`・`finish_rank`）
   × RA（`race_condition_code`）を `race_id`（+`horse_num`）で inner merge し、
   `attach_odds_from_se_parquet` → `attach_market_q` で `ln_market_q` を付与。
   **TEST 期間（2025+）の行も dataset には含めてよいが、Stage 1/2 のスクリプトは
   2024-12-31 以前しか読み込まないこと**（date フィルタを io 直後に適用）。
2. `run_stage1_counts.py`: レース単位セグメントフラグを付与し、2023 / 2024 の
   セグメント別レース数を集計。**2024 年 n≥300** のセグメントのみ「確定」とし、
   `results/stage1_counts.json` に `{segment, n_2023, n_2024, confirmed}` を出力。
   K = 確定セグメント数。Bonferroni 閾値 = 0.01/K。
3. **確定結果（K と除外セグメント）を本仕様書 §12 の追記欄に記録**してから Stage 2 に進む。

### Stage 2: セグメント別 α LRT（fit=2023 → eval=2024、αゲートと同一プロトコル）

確定セグメントごとに:

1. `split_alpha_gate_cv` で fit(2023) / eval(2024) に分割し、セグメントフィルタを適用。
2. fit races（2023）で H1: `fit_fusion_mle(fit_races)`、
   LRT: `likelihood_ratio_test(fit_races, fitted)` → `p_value`（H0 の β も得られる）。
3. eval(2024) セグメント df で
   `delta_ll_per_race = mean_logloss(eval_df, 0.0, h0_beta) - mean_logloss(eval_df, fitted.alpha, fitted.beta)`。
4. 監視指標: `top1_hit_rate(eval_df, fitted.alpha, fitted.beta)` と、レース内 Spearman
   （fusion 確率の順位 vs `finish_rank`、`scipy.stats.spearmanr` をレース毎に計算し平均）。
   **Top-1 > 0.40 または Spearman > 0.60 → 即停止し evaluator へ報告**（結果 json に
   `leak_stop: true` を記録し、後続ステージへ進まない）。
5. 一次判定: `p_value < 0.01/K` **かつ** `delta_ll_per_race > 0` → 一次通過。
6. 全セグメント分を `results/alpha_segments.json` に出力
   （segment, n_fit, n_eval, alpha, beta, h0_beta, lrt_statistic, lrt_p, bonferroni_threshold,
   delta_ll_per_race, eval_top1, eval_spearman, primary_pass, leak_stop）。

### Stage 3: TEST 二次判定（一次通過セグメントのみ、各 1 回。事前登録済み手順）

一次通過セグメントが 0 の場合、**Stage 3 は実行しない**（TEST 完全非接触のまま終了）。
実行する場合の手順は以下に固定する（TEST を見る前に本仕様書で確定）:

1. `split_oos_periods` で fit(2023-01-01..2024-12-31) / TEST(2025-01-01..) に分割し、
   セグメントフィルタ適用。
2. fit 期間セグメント races で H1（α, β）と H0（`market_only=True` の β）を**再フィット**。
3. TEST セグメント df で `test_logloss_fusion = mean_logloss(test_df, α, β)`、
   `test_logloss_market = mean_logloss(test_df, 0.0, β_h0)` を各 1 回だけ計算。
4. 二次判定合格 = `test_logloss_fusion < test_logloss_market` かつ TEST セグメント
   Top-1 ≤ 0.40（かつ Spearman ≤ 0.60）。
5. `results/alpha_segments_test.json` に出力。判定は evaluator が行う。

### 全滅時の終了処理

全確定セグメントで補正後 p ≥ 0.01/K の場合、`results/alpha_segments.json` の
`verdict` に `"market_efficiency_holds_across_segments"` を記録して P1 終了。
gate_summary への反映は evaluator 判定後の別タスク。次アクションは P2（Track B）。

---

## 8. 市場情報混入チェック手順

1. **セグメント定義層（最も厳格）** — `segments_lib.py` はオッズ・市場列に触れないこと:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score\|market_q\|ln_market" \
  pure_rank/experiments/alpha_segments/segments_lib.py
# → 0 件であること
```

2. **実験全体** — 市場情報の使用は `attach_odds_from_se_parquet` / `attach_market_q` の
   import 経由（L2 統合変数）のみに限定されていること:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/alpha_segments/ --include="*.py"
# → ヒットは build_dataset.py / run_stage*.py の import 行と
#   ln_market_q 列参照（build_race_tuples への入力）のみであること。
#   セグメント条件式・フラグ計算に market 系列が現れたら即修正
```

3. **本番非接触の確認**:

```bash
git status --short  # 変更が pure_rank/experiments/alpha_segments/ と docs/specs/ に限られること
```

---

## 9. TDD テスト項目（テストファースト。合成データのみ、実データ不要で走ること）

`pure_rank/experiments/alpha_segments/tests/` に実装:

1. **セグメントフィルタの正しさ**（`segments_lib.py` の各関数）
   - S1: 合成レース（NaN 比率 0 / 0.4 / 0.5 / 1.0）で境界含め正しくフラグが立つ
     （比率 0.5 ちょうどは該当 = `>= 0.5`）。
   - S2: horse_count 8 は該当、9 は非該当。
   - S3: track_condition_code 3, 4 のみ該当（0, 1, 2 は非該当）。
   - S4: course_code {1,2,3,4,7,10} 該当、{5,6,8,9} 非該当。
   - S5: race_condition_code {703, 5} 該当、{701, 10, 16, 999} 非該当。
   - レース単位付与: 同一 race_id の全行に同一フラグが付くこと。
2. **n<300 除外ロジック**: n_2024=299 → excluded、300 → confirmed。
3. **Bonferroni 閾値計算**: K=4 → 0.0025、K=1 → 0.01。K=0（全滅）で Stage 2 が
   空実行終了すること。
4. **α 検出力（陽性コントロール）**: 合成レース群（勝者を `softmax(α_true·z + β·ln q)` から
   サンプリング、α_true=0.8、n=500 レース、seed 固定）で `fit_fusion_mle` が α>0 を回復し
   `likelihood_ratio_test` の p < 0.01 となること。
5. **α=0（陰性コントロール）**: α_true=0 の合成データで p が大きい（> 0.05）こと、
   および ΔLL/race が 0 近傍（負も許容）であること。
6. **ΔLL/race の符号**: 陽性コントロールの eval 分割で
   `mean_logloss(H0) - mean_logloss(H1) > 0` となること。
7. **リーク停止フラグ**: eval_top1=0.41 または spearman=0.61 を注入したレポート組立で
   `leak_stop=true` になり Stage 3 対象から外れること。
8. **市場列ガード**: `segments_lib.py` の SEGMENT_COLUMNS 定数が事前登録ホワイトリスト
   `{hist_last_rank, horse_count, track_condition_code, course_code, race_condition_code}`
   と一致し、`pure_rank/src/common.py` の `FORBIDDEN_MARKET_COLS` /
   `SUSPICIOUS_MARKET_NAME_PATTERN` に一切マッチしないことをテストで自動検証。
9. **再現性**: 合成データテストは seed 固定で決定的に通ること。

---

## 10. implementerへの引き渡し事項（順序付きタスクリスト)

1. `pure_rank/experiments/alpha_segments/` を作成し、README（隔離宣言・市場情報境界・
   実行手順）と `config.json`（§4 のセグメント定義・§3 の閾値を全てここに集約。
   ハードコード禁止）を書く。
2. **テストを先に書く**（§9 の 1〜9。`python -m pytest pure_rank/experiments/alpha_segments/tests/ -v`）。
3. `segments_lib.py`（純関数）を実装しテストを通す。
4. `build_dataset.py` を実装（§7 Stage 1-1。merge 統計 — 行数・レース数・オッズ付与
   成功率 — を stdout と dataset メタに記録）。
5. `run_stage1_counts.py` を実行し `results/stage1_counts.json` を生成。
   確定セグメントと K を orchestrator/planner に報告し、本仕様書 §12 に追記されるのを
   待ってから次へ（**TEST には触れない**）。
6. `run_stage2_lrt.py` を実装・実行し `results/alpha_segments.json` を生成。
   リーク停止条件に該当したら即停止し evaluator へ報告。
7. §8 の市場情報混入チェック（grep 3 種）を実行しログを残す。
8. evaluator へ引き渡し（一次判定の検証）。一次通過セグメントが存在し evaluator が
   承認した場合のみ、`run_stage3_test.py` を実装・実行（TEST 各 1 回）。
9. 全結果を evaluator の最終判定に回す。gate_summary への反映は evaluator 判定後の
   別タスク（本実験スクリプトからは書き込まない）。

---

## 11. 評価基準（evaluator 向けサマリ）

- 一次: LRT p < 0.01/K（fit=2023 races で計算）かつ ΔLL/race > 0（eval=2024）。
- 二次（TEST 1 回のみ）: セグメント TEST logloss（fusion）< 市場 logloss、
  かつ Top-1 ≤ 40%・Spearman ≤ 0.60。
- リーク停止: どの段階でも Top-1 > 40% または Spearman > 0.6 → 即停止・evaluator 報告。
- 本 Phase の評価軸は logloss / LRT であり、ROI では判定しない（L3 接続は合格後の別 Phase）。

---

## 12. Stage 1 確定結果（2026-07-09 追記。出典: `pure_rank/experiments/alpha_segments/results/stage1_counts.json`。追記後は変更禁止）

| ID | n_2023 | n_2024 | confirmed |
|----|--------|--------|-----------|
| S1 新馬・未出走馬中心 | 307 | 311 | **確定** |
| S2 少頭数 | 180 | 220 | 除外（n_2024 < 300） |
| S3 重・不良馬場 | 538 | 336 | **確定** |
| S4 ローカル場 | 1,506 | 1,516 | **確定** |
| S5 低クラス条件戦 | 2,162 | 2,163 | **確定** |

K = 4 / Bonferroni 閾値 0.01/K = **0.0025**

---

## 13. 最終結果（2026-07-09、P1 終了。出典: `pure_rank/experiments/alpha_segments/results/alpha_segments.json`、evaluator PASS 2026-07-09）

確定 4 セグメント（S1/S3/S4/S5）の全てで α = 0.0（下限張り付き）、LRT p ≈ 1.0
（S1: 0.999999 / S3: 0.999994 / S4: 1.0 / S5: 0.999995、いずれも閾値 0.0025 に遠く及ばず）、
ΔLL/race はゼロ近傍（最大 +1.3e-08、S5 は -1.8e-09 と負）で**一次判定は全滅**。
リーク停止条件（Top-1 > 40% / Spearman > 0.6）への該当なし（eval Top-1 は 34.4〜36.0%、
Spearman は 0.533〜0.575）。一次通過セグメントが存在しないため **Stage 3 は不実行 =
TEST(2025+) は完全非接触のまま終了**。verdict =
`"market_efficiency_holds_across_segments"`（市場効率性はセグメント横断的に成立、
fold2 OOS 基準）。事前登録どおりセグメントの後出し追加・再定義による延長は行わず
**P1 を終了**する。次アクションは提案書 §6 のとおり P2（Track B 調教時系列 5 候補）へ移行。

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-09 | 初版（P1 仕様確定。セグメント定義・2 段階手順・TDD 項目・混入チェックを事前登録） |
| 2026-07-09 | §12 Stage 1 確定結果を追記（K=4、S2 除外）。§13 最終結果を追記（4 セグメント全滅・α=0・verdict=market_efficiency_holds_across_segments・evaluator PASS・Stage 3 不実行=TEST 非接触）。**P1 終了** |
