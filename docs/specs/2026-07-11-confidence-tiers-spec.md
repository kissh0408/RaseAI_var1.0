# 実装仕様書: 自信度階層別の対1番人気ROI優位検定（confidence tiers） — 2026-07-11

**作成者**: planner
**承認**: ユーザー承認済み（可変サイジングを直感で実装する前に、自信度と市場優位の関係を事前登録・OOS規律で検証することに合意）
**実装先**: `betting/experiments/confidence_tiers/`（隔離実験・本番非接触）
**実装担当**: implementer（本仕様書はコードを含まない）
**先例書式**: `docs/specs/2026-07-09-p1-alpha-segments-spec.md`（事前登録・Bonferroni・段階手順）、
`docs/specs/2026-07-10-p4-cross-pool-divergence-spec.md`（クラスタブートストラップ・危険信号）、
`docs/specs/2026-07-10-loss-minimization-implementation-spec.md`（flat top-1 の測定土俵）

---

## 1. 目的・背景

flat top-1 戦略（モデル予測1位への定額単勝ベット）は TEST(2025+) で
本番ROI 83.35% vs 1番人気ROI 78.35%、**+4.99pp の点推定優位**（95%CI [-2.16, +10.18]pp、
verdict = `pass_point_only`。`evaluation/reports/betting_backtest_oos_flat.json`）を示した。

ユーザーから「勝率期待度が高い馬券には大きく賭けてもいいのでは」という可変サイジング提案が
あった。本 Phase はその前提仮説を検証する:

> **仮説**: flat top-1 戦略で実際に賭けている馬券を、モデルの自信度で階層化したとき、
> 自信度が高い階層ほど対1番人気ROI優位（Δ = モデルROI − 同一レース集合の1番人気ROI）が
> 統計的に大きい。

### orchestrator が事前に示した懸念（本仕様の設計に反映済み）

1. **優位性の源泉**: モデルの点推定優位（+4〜5pp）は「過剰人気馬への集中投票の回避」に
   由来する可能性が高い。
2. **希釈リスク**: 自信度が高い予測ほど市場の人気馬に近い可能性があり、自信度で重み付けすると
   優位性の源泉（人気馬回避）を薄めるおそれがある。→ §5.4 の記述的診断
   （階層別の1番人気一致率・平均オッズ）で直接観測する。
3. **ケリー不適用**: ケリー的比例サイジングは正の期待値を前提とするが、現状の期待値は
   **負**（控除率の壁。fold2 OOS 実測で元本の約18%の期待損失）。よって本 Phase の結論が
   どうであれ、導入検討対象は「負の期待値下での相対的損失最小化の配分調整」であり、
   ケリー基準そのものは数学的前提を欠くため採用しない（§7.4）。

本 Phase は測定のみを行う。**可変サイジングの実装は行わず**、本番設定
（`betting/config/betting_config.json` の凍結値 `stake_fraction=0.001`）には一切触れない。

---

## 2. 市場情報境界・禁止事項の確認

- [x] **L1（`pure_rank/src/`）に一切変更を加えない**。本検証は L1 特徴量に何も追加せず、
  既存 flat top-1 戦略の出力（fold2 OOS スコア由来の `pure_score_z`）とオッズ
  （L3 許可範囲: 除外条件・決済・ベースライン算出のみ）を階層化するだけである。
- [x] **自信度指標（§3）はオッズ・人気・市場由来の列を一切使わない**（`pure_score_z` のみ）。
  階層割当関数が市場列に触れないことを §11 のテストで機械的に担保する。
- [x] オッズの使用箇所は (i) `select_top1_bets` の min/max 除外、(ii) 決済
  （payout = stake×odds）、(iii) 1番人気ベースラインの特定、の3箇所のみ（すべて L3 許可範囲）。
- [x] z の二重使用なし（L2 統合は本 Phase に登場しない）。
- [x] **「黒字化」を主張・示唆する表現をコード・ログ・JSON・レポートに一切書かない**。
  `betting/src/flat_top1.py::DISCLAIMER` を結果 JSON の `disclaimer` キーに埋め込む。
- [x] ROI は本 Phase の測定対象だが、いかなる階層の ROI が 100% を超えても
  「黒字」とは記述せず、§9 の危険信号として扱う。

---

## 3. 自信度指標の定義（事前登録。後出し追加・変更禁止）

### 3.1 採用指標: margin（L1 スコアの1位−2位差）

```
margin(r) = pure_score_z(レースr内の1位馬) − pure_score_z(レースr内の2位馬)
```

- `pure_score_z` は `scores_v39_course_slim_fold2_oos.parquet` の既存列
  （L1 のレース内 z スコア。市場情報を含まない）。
- 1位・2位の特定は `betting/src/flat_top1.py::select_top1_bets` と同一の決定的順序
  （スコア降順 → 同値は馬番昇順）に従う。2位馬のスコアも同じソート順の2行目から取る。
- 同値タイ（margin = 0）は定義上そのまま 0 として扱う（最下位階層に入る）。
- margin は「モデルがレース内でどれだけ1位馬を突出させているか」という
  **レース内相対の自信度**であり、レース間比較可能（z 正規化済みスコアの差）。

### 3.2 候補 (a) win_prob_est（融合確率）を検定対象として不採用とする根拠

fold2 OOS 正式測定（`evaluation/reports/fusion_oos_fold2.json`）で **α = 0** が確定している。
このとき融合確率は p ∝ exp(β·ln q) = q^β（β≈1.034）となり、**市場確率 q の単調変換に一致する**。
したがって win_prob_est による階層化は:

1. 「モデルの自信度」ではなく「モデル1位馬に対する**市場の**自信度（＝その馬のオッズ帯）」を
   測ることになり、本 Phase の仮説（モデルの自信度）に答えられない。
2. 実質的にオッズ帯セグメント分析であり、既に適用しているオッズ除外
   （min_odds=2.0 / max_odds=50.0）と条件が絡み合う別仮説になる。
3. 市場情報を階層化の条件変数に使うことになり、「モデル固有の情報で配分を変えられるか」
   という可変サイジングの前提検証として不適切。

以上のドメイン論理のみに基づき（実データの結果は未参照）、**検定対象は margin の単一指標**
とする。ただし懸念2（自信度が高いほど市場人気馬に近い可能性）を直接観測するため、
§5.4 の**記述的診断**（検定なし・判定に不使用）として階層別の市場近接度を報告する。
win_prob_est を用いた別仮説の事後追加は禁止（本 Phase 終了後に新 Phase として
事前登録し直す場合のみ可）。

---

## 4. 階層設計（事前登録）

### 4.1 階層数と境界の決定手順

- **K = 4（四分位）**。階層 T1（margin 最小 = 低自信）〜 T4（margin 最大 = 高自信）。
- **境界値は fit 期間 2023 年の margin 分布のみで決定する**:
  2023-01-01〜2023-12-31 の「実際にベット対象となったレース」
  （§5.1 の選定・オッズ除外適用後）の margin の 25/50/75 パーセンタイル
  （`numpy.quantile`, `method="linear"`）を境界 [b1, b2, b3] とする。
- 割当規則: `tier = 1 + np.searchsorted([b1, b2, b3], margin, side="right")`
  （境界値ちょうどは**下位階層**に入る。§11 でテスト）。
- 境界は Stage 1 実行後に §13 に追記し、**以後変更禁止**（TEST はもちろん、
  VALID 2024 の結果を見た後の再決定も禁止）。

### 4.2 期間プロトコルと P1 先例との整合

| 段階 | 期間 | 用途 |
|---|---|---|
| Stage 1（境界決定） | 2023 年 | margin 四分位の算出のみ（着順・ROI は見ない） |
| Stage 2（一次判定） | **VALID = 2024 年** | 階層別 Δ の測定・検定・一次判定 |
| Stage 3（二次判定） | TEST = 2025-01-01 以降 | 事前登録条件成立時のみ **1回** |

- P1 先例（fit=2023 → eval=2024 → TEST 1回）と同一の年次分割。
- **fold2 OOS スコアの 2023 年は early-stopping に使われた弱汚染年**
  （`fusion_oos_fold2.json` protocol.caveat）である。ただし Stage 1 で 2023 年から得るのは
  **margin 分布の四分位のみ**（着順・払戻・ROI を一切参照しない outcome-blind な統計量）
  であり、汚染がもたらしうるのはスコア分布のわずかな楽観化のみで、判定（2024/2025 の
  完全 OOS 年で実施）へのリークにはならない。この注記を結果 JSON の `caveats` に記載する。
- 期間定数は `run_backtest_oos_flat.py` / `derive_flat_fraction.py` と同じソース
  （`prob_fusion/src/oos_protocol.py::TEST_START`、VALID = 2024-01-01..2024-12-31）を参照する。

### 4.3 最小サンプル規則

- VALID(2024) で各階層のベット数 n(t) ≥ **200**（憲法: ROI 有意性主張の最低標本数）。
  200 未満の階層は当該階層の検定を「判定保留」とし、順序仮説（§6.2）も実行しない
  （4階層すべての Δ が必要なため）。
- 2023 四分位を 2024 に適用するため階層占有率は厳密に 25% ずつにはならない。
  Stage 2 で階層別 n を必ず報告する（参考: VALID 2024 の総ベット数は
  `flat_fraction_valid_2024.json` 実測で 2,608 → 1階層あたり期待 ~650）。

---

## 5. 測定式（`run_backtest_oos_flat.py` と同一の土俵。事前固定）

### 5.1 ベット集合の構築

- データ: `pure_rank/data/03_scores/scores_v39_course_slim_fold2_oos.parquet` +
  `pure_rank/data/02_features/features_v39_course_slim.parquet` +
  `evaluation/odds_loader.py::attach_odds_from_se_parquet`（`betting/src/backtest.py::load_scored_odds_frame`
  を再利用してよい）。標準除外フィルタ（grade_code / abnormal_code / horse_count / finish_rank）は
  既存ロード経路のものをそのまま使う。
- 選定: `betting/src/flat_top1.py::select_top1_bets` を **import 再利用（コピー禁止）**。
  cfg は `min_odds=2.0 / max_odds=50.0`（本番と同値）。
- **オッズ除外: 適用する**（事前登録）。理由: 仮説は「**実際に賭けている**馬券」の階層化で
  あり、本番運用と同じベット集合で測らなければ可変サイジング導入判断の材料にならない。
- ステーク: 100円均等（flat）。ROI = Σpayout / Σstake。決済は
  `flat_top1.py::settle_win_bets` を再利用。

### 5.2 階層別の測定量

各階層 t ∈ {T1..T4}、対象レース集合 R(t) = 「モデル1位馬（＝ベット対象）の margin が
階層 t に入る、オッズ除外を通過したレース」について:

```
ROI_model(t) = Σ_{r∈R(t)} payout_model(r) / (100 × |R(t)|)
ROI_fav(t)   = Σ_{r∈R(t)} payout_fav(r)   / (100 × |R(t)|)
Δ(t)         = ROI_model(t) − ROI_fav(t)
```

- **1番人気ベースラインは階層ごとに再定義しない**。同一レース集合 R(t) 内で、
  1番人気（レース内単勝オッズ最小、同オッズは馬番昇順の決定的タイブレーク。
  `run_backtest_oos_flat.py::_favorite_flat_roi` / `compute_favorite_baseline` と同一定義）へ
  同額 flat bet した場合の ROI と**ペア比較**する。
- 1番人気側にはオッズ除外を適用しない（本番 TEST 測定 `_favorite_flat_roi` と同一。
  ベースラインは「同じレースで市場に従った場合」の対照であり、除外はモデル側の
  ベット集合の定義にのみ作用する）。
- モデル1位馬と1番人気が同一馬のレースでは payout が一致し Δ への寄与は 0
  （ペア比較の自然な性質。除外しない）。

### 5.3 再現性アンカー（reproduction gates）

- **VALID**: 全階層合算（T1〜T4 の和）の n_bets・的中率・ROI_model が
  `evaluation/reports/flat_fraction_valid_2024.json` の実測
  （2,608ベット・的中率24.2%・ROI 79.1%）と一致すること（±0.1pp / n は完全一致）。
  不一致は選定・階層割当のバグとして修正対象（Rule 3 違反ではない）。
- **TEST**（Stage 3 実行時のみ）: 全階層合算が `betting_backtest_oos_flat.json` の
  本番設定（n_bets=3,758・ROI_model 83.35%・ROI_fav 78.35%）と一致すること（同上）。

### 5.4 記述的診断（検定なし・判定に不使用。懸念2の直接観測）

各階層について以下を報告する（orchestrator 懸念「高自信＝市場人気馬に近い」の検証）:

- 1番人気一致率: R(t) 内でモデル1位馬 = 1番人気であるレースの比率
- モデル1位馬の平均・中央値オッズ
- モデル Top-1 的中率と1番人気 Top-1 的中率（同一 R(t) 内のペア）
- 階層内 margin の平均・範囲

これらは解釈の補助であり、一次・二次判定には一切使わない（後出しの判定基準化を禁止）。

---

## 6. 統計検定（事前固定。事後の方式変更禁止）

### 6.1 階層別検定（H1〜H4）

- **レース単位ペアドクラスタ・ブートストラップ**: R(t) の race_id を復元抽出
  （B = 10,000、seed = 42）し、各リサンプルで Δ*(t) を再計算。モデル payout と
  1番人気 payout は同一レースでペアのままリサンプルする（P4 の
  `betting/experiments/cross_pool_divergence/divergence_lib.py::cluster_bootstrap_p_value` /
  `_percentile_two_sided_p` のパターンを再利用。ペア差分列に対して適用）。
- 仮説は方向付き（Δ(t) > 0）のため **片側 percentile p 値**（p = P(Δ* ≤ 0)）を用いる。
  95% percentile CI も併記する。

### 6.2 順序仮説（H_ord。事前登録により「含める」と決定）

各階層独立検定だけでは「自信度が高い**ほど**大きい」という本来の仮説に答えられないため、
順序仮説を**含める**。ただし4点間の Spearman は粒度が粗く検定として不安定なため、
事前登録する順序統計量は**最上位−最下位コントラスト**とする:

```
H_ord: C = Δ(T4) − Δ(T1) > 0
```

- 検定: R(T4) ∪ R(T1) のレースをクラスタ単位で復元抽出し（各階層内で層別リサンプル、
  B = 10,000、seed = 42）、C* の片側 percentile p 値（p = P(C* ≤ 0)）と 95%CI。
- 補助として「Δ(T1) ≤ Δ(T2) ≤ Δ(T3) ≤ Δ(T4) が全て成立するか」（単調性フラグ）を
  記述的に報告する（検定・判定には使わない。4点の完全単調はノイズで壊れやすく、
  これを判定基準にすると偽陰性過多になるため）。

### 6.3 多重比較補正

- 事前登録仮説数 = **5**（H1〜H4 + H_ord）。Bonferroni 閾値 = **0.01 / 5 = 0.002**。
- 判定保留階層（n < 200）が出た場合も閾値は 0.002 のまま固定する
  （保留による実効仮説数減少で閾値を緩める後出しを禁止。保守側に倒れるのは許容）。

---

## 7. 判定基準・解釈の事前固定（TEST を見る前に確定。変更禁止）

### 7.1 一次判定（VALID 2024 のみ。Rule 3）

| 判定 | 条件 |
|---|---|
| **一次通過** | H4（最高自信度階層 Δ(T4) > 0）**または** H_ord（C > 0）の少なくとも一方が p < 0.002 で有意 |
| **一次不通過（全滅）** | H1〜H4・H_ord のいずれも p ≥ 0.002 |
| **判定保留** | いずれかの階層で n < 200（§4.3）→ 検定自体を縮退させず、そのまま「保留 = 不通過扱い」で終了（データ追加による延長は別 Phase として事前登録し直す） |

### 7.2 二次判定（TEST 2025+、1回のみ。一次通過時のみ実行）

- **実行条件**: 一次通過、かつ evaluator が一次判定を承認した場合のみ。
  一次不通過なら **Stage 3 は実行しない = TEST 完全非接触のまま終了**。
- 二次判定合格 = TEST で (i) Δ(T4) > 0（点推定）かつ (ii) C = Δ(T4) − Δ(T1) > 0（点推定）
  かつ (iii) 全階層合算の再現性アンカー（§5.3）成立。
  TEST では有意性を要求しない（1回きりの確認測定であり、TEST 結果への
  検定の後付け最適化を避けるため点推定の符号のみを事前登録する）。
- **TEST を見た後の指標・境界・検定・判定基準の変更は一切禁止**。

### 7.3 「自信度可変サイジングを導入すべきか」への解釈（事前固定）

| 結果 | 解釈（これ以外の解釈の後付けを禁止） |
|---|---|
| 一次通過 + 二次合格 | 「自信度と対市場優位の正の関連に統計的証拠あり」。可変サイジング設計を**別 Phase として planner が新規事前登録する資格**が得られる。本 Phase 内でのサイジング実装は行わない。設計する場合もケリーではなく有界な段階的配分（例: 階層別固定倍率）とし、負の期待値下の損失最小化の枠内に留める |
| 一次通過 + 二次不合格 | VALID の有意性は再現せず。可変サイジングは**導入しない**。flat 継続 |
| 一次不通過（全滅） | 「自信度は対市場優位を予測しない（VALID 2024 基準）」を確定記録。可変サイジングは**導入しない**。flat（stake_fraction=0.001 凍結値）継続 |
| Δ(T4) が点推定で負、または Δ(T4) < Δ(T1)（VALID） | 懸念2（希釈リスク）の実証: 自信度加重は優位性の源泉を薄める方向。可変サイジングは**導入しない**とともに、その旨を verdict に明記（`confidence_weighting_would_dilute_edge`） |

いずれの結果でも、本 Phase は「賭けて勝てる」ことを一切主張しない。全体の期待値は負
（fold2 OOS 実測で元本の約18%の期待損失）であり、測っているのは損失の相対差である。

### 7.4 ケリー基準の不採用（事前明記）

ケリー的比例サイジング（stake ∝ エッジ/オッズ）は正の期待値を前提とした資金成長率の
最適化であり、期待値が負の本戦略には数学的前提が成立しない（負のエッジではケリー最適解は
「賭けない」）。本 Phase の結論が最良でも、導入検討対象は有界な階層別配分に限られる。

---

## 8. 手順（3 ステージ。Rule 3 遵守）

### Stage 1: 境界決定（2023 のみ。着順・ROI 非参照）

1. `build_dataset.py`: scores + features + オッズ付与 → `select_top1_bets` 適用済みの
   ベット候補データセット（margin 列付き）を `data/` に生成。全期間を含んでよいが、
   **Stage 1/2 のスクリプトは io 直後に期間フィルタを適用**し、Stage 1 は 2023 年のみ、
   Stage 2 は 2024 年のみを読む（TEST 行に触れない）。
2. `run_stage1_boundaries.py`: 2023 年ベット集合の margin 四分位 [b1, b2, b3] を算出し
   `results/stage1_boundaries.json` に出力（n_2023、margin 分布統計を含む。
   **2023 年の ROI・的中率は計算・出力しない** — 境界決定を outcome-blind に保つため）。
3. 境界値を本仕様書 §13 に追記してから Stage 2 に進む。

### Stage 2: VALID 2024 一次判定

1. `run_stage2_valid.py`: 2024 年ベット集合に凍結境界で階層を割当て、階層別に
   §5.2 の測定・§5.4 の記述的診断・§6 の検定を実行。
2. 再現性アンカー（§5.3 VALID）を検証。不成立なら判定に進まずバグ調査。
3. 出力 `results/tiers_valid.json`: 階層ごとに
   `{tier, n_races, boundaries_used, roi_model, roi_fav, delta, bootstrap_p, ci95,
   hit_rate_model, hit_rate_fav, favorite_agreement_rate, mean_odds, median_odds,
   margin_mean}`、全体フィールド
   `{K_hyp: 5, bonferroni_threshold: 0.002, ordering_contrast: {c, p, ci95},
   monotonicity_flag, primary_pass, verdict, reproduction_gate, caveats, disclaimer,
   protocol（seed, B, 期間, 境界出典）}`。
4. §9 の危険信号チェック。該当すれば即停止・evaluator 報告。
5. evaluator へ引き渡し（一次判定の独立検証）。

### Stage 3: TEST（一次通過 + evaluator 承認時のみ、1回）

1. `run_stage3_test.py`: TEST(2025+) ベット集合に**同じ凍結境界**で階層割当て、
   §5.2 の測定と §7.2 の判定材料を1回だけ算出。検定は行わず点推定と記述統計のみ
   （ブートストラップ CI は参考値として同一 seed で1回算出してよいが判定には使わない）。
2. 再現性アンカー（§5.3 TEST: 合算が 83.35% / 78.35% / n=3,758 と一致）を必ず含める。
3. 出力 `results/tiers_test.json`。判定は evaluator が行う。

### 全滅時の終了処理

一次不通過の場合、`results/tiers_valid.json` の `verdict` に
`"confidence_does_not_predict_market_edge"`（Δ(T4)<0 または Δ(T4)<Δ(T1) の場合は
`"confidence_weighting_would_dilute_edge"`）を記録して終了。
`gate_summary.json` への反映は evaluator 判定後の別タスク（§10 隔離宣言）。

---

## 9. リーク停止・危険信号（事前登録）

1. **階層別 Top-1 的中率 > 40%**: 自信度上位階層では条件付けにより的中率が全体平均
   （~30%）を大きく上回ることが**正当に**起こりうる（1番人気も同階層内では同様に上振れ
   するはず）。よって機械的な即不合格ではなく、**即時停止 → evaluator 報告 → 検証**の
   手順とする。evaluator は同一階層内の1番人気 Top-1 的中率と比較し、
   (i) 1番人気側も同水準に上振れしていれば条件付け効果として続行可、
   (ii) モデル側のみ突出していればリーク疑いとして implementer へ差し戻す。
   本判定手順自体を事前登録とし、結果を見てからの基準変更を禁止する。
2. **Spearman > 0.6 相当の扱い**: 本 Phase は新しいランキングモデルを作らないため
   Spearman は直接の監視対象ではないが、margin 計算のバグ等で実質的に未来情報が混入した
   場合は階層別的中率に現れる。上記 1 がその検出器を兼ねる。
3. **ROI > 100% の階層**: 「黒字」とは記述せず、まずデータ結合バグ（race_id 正規化・
   払戻重複計上）を疑い、即停止して evaluator に検証を依頼する（P4 §10 パターン）。
   検証には payout 集中度ゲート（`top1_payout_share ≤ 0.30` かつ `n_hits ≥ 10`、
   `divergence_lib.py::payout_concentration_gate` 再利用）を階層別診断として必ず適用し、
   ゲート違反の ROI は無効（winner's curse の再演）として記録する。
4. **|Δ(t)| > 20pp の階層**: 全体 Δ（+4〜5pp）から大きく外れるためバグ優先で検証。
5. いずれの報告にも「本測定は確定/前日水準オッズベースであり、購入時点のオッズで
   優位が縮小しうる」限界と DISCLAIMER を併記する。

---

## 10. 実装構成（隔離実験・P1/P4 パターン準拠）

```
betting/experiments/confidence_tiers/
├── README.md                 # 目的・隔離宣言・市場情報境界・実行手順（P4 README 書式準拠）
├── config.json               # K=4・n_min=200・B=10000・seed=42・Bonferroni基準0.01/5、
│                             #   min_odds/max_odds（本番と同値2.0/50.0）、期間定数参照、
│                             #   境界値（Stage 1後に凍結追記）— ハードコード禁止の受け皿
├── tiers_lib.py              # 純関数のみ: margin計算・四分位境界・階層割当・Δ/ペアド
│                             #   ブートストラップ・順序コントラスト・最小サンプル判定・
│                             #   Bonferroni・危険信号フラグ。市場列は odds（Δ計算・
│                             #   ベースライン特定用の引数）以外に触れない
├── build_dataset.py          # scores + features + オッズ → ベット候補parquet（data/）
├── run_stage1_boundaries.py  # 2023 margin四分位 → results/stage1_boundaries.json
├── run_stage2_valid.py       # VALID 2024 測定・検定・一次判定材料 → results/tiers_valid.json
├── run_stage3_test.py        # 一次通過+承認時のみ TEST 1回 → results/tiers_test.json
├── data/                     # bets_dataset.parquet
├── results/
└── tests/                    # §11 の TDD テスト
```

### 隔離宣言（README に明記すること）

本実験は上記ディレクトリに完結する。以下には**書き込まない**:
`betting/config/betting_config.json`（凍結 stake_fraction=0.001 を変更しない）、
`pure_rank/models/`、`pure_rank/data/`（読み取りのみ）、`prob_fusion/`、
`betting/src/`（`flat_top1.py` は import のみ）、
`evaluation/reports/gate_summary.json`、`evaluation/reports/betting_backtest_oos_flat.json`。
`gate_summary.json` への結果反映は **evaluator の判定後に別タスクとして**
`evaluation/update_gate_summary.py` 経由で行う。本実験のスクリプトは
`data/` と `results/` のみに出力する。

---

## 11. TDD テスト項目（テストファースト。合成データのみで走ること）

`betting/experiments/confidence_tiers/tests/` に実装:

1. **margin 計算**: 合成レース（z = [2.0, 1.5, 0.0]）で margin = 0.5。
   1位・2位の順序が `select_top1_bets` と同一のタイブレーク（スコア同値→馬番昇順）に
   従うこと。全馬同値レースで margin = 0。
2. **階層割当の境界テスト**: 境界 [b1,b2,b3] = [0.2, 0.5, 0.9] に対し
   margin = 0.19→T1、0.2→T1（境界値ちょうどは下位）、0.21→T2、0.5→T2、0.9→T3、
   0.91→T4。決定的であること。
3. **四分位境界の再現性**: 固定合成配列に対し `numpy.quantile(method="linear")` の
   期待値と一致。境界決定関数が着順・払戻列を受け取らないシグネチャであること
   （outcome-blind の構造的担保）。
4. **ペア比較の整合**: 合成レース集合で ROI_model(t)・ROI_fav(t) が手計算と一致。
   モデル1位=1番人気のレースで当該レースの Δ 寄与が 0 になること。
5. **ブートストラップ陽性コントロール**: Δ = +8pp を注入した合成データ（500レース、
   seed固定）で H4 の p < 0.002。**陰性コントロール**: Δ = 0 の合成データで p > 0.05。
   seed=42 固定で決定的に再現すること。
6. **順序コントラスト**: Δ(T4)−Δ(T1) = +10pp 注入で H_ord p < 0.002、
   差 0 注入で p > 0.05。単調性フラグが正しく立つ/立たないこと。
7. **最小サンプル**: n=199 → 判定保留、200 → 検定実施。保留時に H_ord が実行されないこと。
8. **Bonferroni**: 閾値が常に 0.01/5 = 0.002 であること（保留発生時も不変）。
9. **危険信号フラグ**: 階層 Top-1=0.41 注入 → `leak_review_required=true`。
   ROI=1.01 注入 → `danger_roi_gt_100=true` かつ集中度ゲート適用。
   top1_payout_share=0.31 または n_hits=9 → `diagnosis_valid=false`。
10. **市場情報混入の静的検査**: `tiers_lib.py` の margin・境界・階層割当関数群の
    ソースに `odds` / `popularity` / `ninki` / `market_log_odds` / `init_score` /
    `market_q` / `ln_market` / `win_prob` が現れないこと（Δ計算・ベースライン関数は
    odds 引数を持つため検査対象から除外し、除外対象関数名をテスト内に明示列挙する。
    禁止トークン定数は文字列結合でエンコードし自己ヒットを回避）。
11. **選定ロジックの import 再利用**: 実験モジュールが
    `betting.src.flat_top1.select_top1_bets` を import しており、同等ロジックの
    再実装（コピー）が存在しないことの静的検査（`def select_top1` 等の定義が
    実験ディレクトリ内に存在しないこと）。
12. **L1 特徴量非追加の担保**: 実験コードが読み取る scores/features は既存 parquet のみで、
    `pure_rank/src/` の import が無いこと（本検証が既存 flat_top1 出力＋オッズの
    階層化に閉じていることの機械的担保）。
13. **再現性**: 全合成テストが seed 固定で決定的に通ること。

---

## 12. implementer への引き渡し事項（順序付きタスクリスト）

1. `betting/experiments/confidence_tiers/` を作成し、README（隔離宣言・市場情報境界・
   実行手順）と `config.json`（§3〜§7 の全定数を集約。ハードコード禁止）を書く。
2. **テストを先に書く**（§11 の 1〜13。
   `python -m pytest betting/experiments/confidence_tiers/tests/ -v`）。
3. `tiers_lib.py`（純関数）を実装しテストを通す。ブートストラップは
   `betting/experiments/cross_pool_divergence/divergence_lib.py` のパターンを再利用
   （import できる形なら import、パスの都合で不可なら最小限の同型実装＋出典コメント）。
4. `build_dataset.py` を実装（`load_scored_odds_frame` + `select_top1_bets` 再利用。
   build ログに行数・レース数・オッズ付与成功率・skip 内訳を記録）。
5. `run_stage1_boundaries.py` を実行し境界を確定 → orchestrator/planner に報告し、
   本仕様書 §13 に追記されるのを待ってから次へ（**2024/2025 の結果には触れない**）。
6. `run_stage2_valid.py` を実装・実行 → `results/tiers_valid.json`。
   再現性アンカー不成立ならバグ調査（Rule 3 違反ではないが理由をログに残す）。
   危険信号該当時は即停止し evaluator へ報告。
7. 市場情報混入チェックをコマンドラインでも実行しログを残す:
   ```bash
   grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
     pure_rank/src/ --include="*.py"   # → 増分ゼロであること
   git status --short                   # → 変更が betting/experiments/confidence_tiers/ と
                                        #    docs/specs/ に限られること
   ```
8. evaluator へ引き渡し（一次判定の独立検証）。
9. 一次通過かつ evaluator 承認の場合のみ `run_stage3_test.py` を実装・実行
   （**TEST 1回のみ。再実行禁止**。バグ修正による再実行はレポートに理由明記の場合のみ可）。
10. 全結果を evaluator の最終判定に回す。`gate_summary.json` への反映は判定後の別タスク。

---

## 13. Stage 1 確定結果（2026-07-11 追記。出典: `betting/experiments/confidence_tiers/results/stage1_boundaries.json`。追記後は変更禁止）

- **境界（凍結）**: b1 = 0.11895233392715454 / b2 = 0.28122442960739136 / b3 = 0.5161097198724747
- n_2023 = 2,714 ベット（outcome-blind: 2023 年の着順・ROI・的中率は未計算・未出力）
- margin 分布（2023）: min≈0.0001, max≈2.074, mean≈0.357, std≈0.306
- build 統計: 158,180 行 / 11,561 レース、オッズ付与成功率 100%、`select_top1_bets` 通過 9,080 ベット、skip 2,481（全て odds_below_min）
- **§4.1 の記法訂正（判定内容の変更なし）**: 本文の `side="right"` は §11-2 の事前登録済み判定例
  （境界値ちょうど→下位階層: 0.2→T1, 0.5→T2, 0.9→T3）と矛盾するため、判定例を正とし
  実装は `np.searchsorted(boundaries, margin, side="left")` とする。テスト 37 件全パスで確認済み。

---

## 14. 評価基準（evaluator 向けサマリ)

- 一次（VALID 2024 のみ）: H4 または H_ord が片側ブートストラップ p < 0.002（0.01/5）。
  各階層 n ≥ 200。再現性アンカー（合算 = flat_fraction_valid_2024.json）成立。
- 二次（TEST、1回のみ・一次通過時のみ）: Δ(T4) > 0 かつ Δ(T4) − Δ(T1) > 0（点推定）、
  再現性アンカー（合算 = betting_backtest_oos_flat.json: 83.35% / 78.35% / n=3,758）成立。
- 解釈は §7.3 の表に固定（可変サイジング導入は一次通過+二次合格の場合に限り
  「別 Phase の事前登録資格」が生じるのみ。本 Phase では実装しない）。
- 危険信号（§9）該当時は合格ではなく停止・検証。「黒字化」表現はいかなる結果でも禁止。

---

## 15. 最終結果（Stage 2 実行後に追記。追記後は変更禁止）

**記録日: 2026-07-11**

### Stage 2（VALID 2024、K_hyp=5、Bonferroni閾値0.002）

再現性アンカー成立（合算n=2,608、的中率24.233%、ROI 79.099% — `flat_fraction_valid_2024.json`と完全一致）。

| 階層 | n | ROI_model | ROI_fav | Δ | 片側p | 1番人気一致率 | 平均オッズ |
|---|---|---|---|---|---|---|---|
| T1（自信度最低） | 666 | 82.85% | 80.98% | +1.88pp | 0.4053 | 37.4% | 4.64倍 |
| T2 | 677 | 74.05% | 84.90% | **-10.86pp** | 0.9465 | 48.9% | 4.16倍 |
| T3 | 641 | 78.55% | 77.66% | +0.89pp | 0.4509 | 54.1% | 3.63倍 |
| T4（自信度最高） | 624 | 81.14% | 77.52% | +3.62pp | 0.2408 | **74.2%** | 3.26倍 |

**H1〜H4・順序仮説（C=Δ(T4)−Δ(T1)=+1.74pp, p=0.4171）全て不通過**（最良のH4でもp=0.2408、閾値0.002に遠く及ばず）。単調性フラグもfalse（T2で崩れる）。

**verdict = `confidence_does_not_predict_market_edge`**

### 記述的診断（判定不使用、解釈補助）

自信度が高い階層ほど「モデル予測1位が1番人気と一致する割合」が明確に上昇する（T1: 37.4% → T4: 74.2%）。平均オッズも単調に低下する（4.64倍 → 3.26倍）。これは着手前に懸念していた仮説——**「自信度で重み付けすると、モデルの優位性の源泉（過剰人気馬回避）そのものを薄めてしまう」**——を記述的に裏付ける結果である。

### 結論

自信度（`pure_score_z`の1位-2位差）が対1番人気ROI優位の大きさを予測するという仮説は**支持されなかった**。全階層n≥200、payout集中度ゲート全通過（winner's curseの兆候なし）、危険信号（Top-1>40%等）該当なし。一次判定不通過のため、事前登録通り**TEST(2025+)には一切接触せず終了**。

### 可変サイジングへの推奨

**導入を推奨しない。** 自信度と市場優位の相関が統計的に確認できなかった上、記述的診断は「自信度で重み付けすると人気馬への回帰が進み、モデルの数少ない優位性の源泉を弱める」という逆方向のリスクを示唆している。現行の定額（flat, f=0.001固定）サイジングを維持すべきである。

### 独立検証

implementerの報告値（reproduction_gate, 各階層のROI/p値/信頼区間、verdict）を`results/tiers_valid.json`から直接読み込み再計算し、完全一致を確認した。テストスイート37/37件パス。市場情報混入チェック: `pure_rank/src/`への変更差分ゼロ、L1特徴量への市場情報混入なし（oddsの参照は全てbetting層のROI/ベースライン計算内のみ）。本番資産（`betting_config.json`の凍結値、`pure_rank/models/`、`evaluation/reports/gate_summary.json`）は本フェーズで変更なし。

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-07-11 | 初版（自信度指標 = margin の単一事前登録・win_prob_est 不採用根拠・K=4 四分位・境界 = 2023 のみで決定・ペアドクラスタブートストラップ 5 仮説 Bonferroni 0.002・順序仮説 = T4−T1 コントラスト・判定と解釈の事前固定を登録） |
| 2026-07-11 | §15 最終結果追記。全仮説不通過（verdict=confidence_does_not_predict_market_edge）、TEST非接触のまま終了。可変サイジング導入は非推奨と結論 |
