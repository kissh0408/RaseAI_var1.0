# alpha_segments — セグメント条件付き α の異質性検定（P1）

**目的**: fold2 OOS 正式測定で確定した α=0（全レース平均で L1 スコアは市場に情報を
追加しない）が、事前登録した 5 セグメント（S1〜S5）のいずれかで破れるかを検定する。

**答える問い**: 「ドメイン知識のみから事前登録した部分集合の中に、市場効率性が
構造的に弱く L1 スコアが有意な追加情報を持つ領域があるか」。

**仕様書**: `docs/specs/2026-07-09-p1-alpha-segments-spec.md`

## 隔離宣言（本番非接触）

本実験は `pure_rank/experiments/alpha_segments/` 配下に完結する。以下は書き換えない:
`pure_rank/models/`、`pure_rank/data/02_features/*.parquet`、`pure_rank/data/03_scores/`、
`prob_fusion/data/`、`prob_fusion/src/`、`evaluation/reports/gate_summary.json`、
`pure_rank/config/train_config.json`。
`gate_summary.json` への結果反映は evaluator の判定後に別タスクとして
`evaluation/update_gate_summary.py` 経由で行う。本実験のスクリプトは
`results/` のみに出力する。

## 市場情報境界（プロジェクト憲法遵守）

- セグメント判定に使う列（`hist_last_rank`, `horse_count`, `track_condition_code`,
  `course_code`, `race_condition_code`）はいずれもレース属性 + shift 済み過去走属性
  であり、オッズ・人気・市場由来ではない。
- 市場確率 q（`ln_market_q`）は L2 条件付きロジット統合の変数としてのみ使用する
  （`fit_fusion.py::build_race_tuples` 経由。`segments_lib.py` では一切参照しない）。
- z の二重使用なし（`pure_score_z` は α·z の一箇所のみ）。

検証コマンド:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score\|market_q\|ln_market" \
  pure_rank/experiments/alpha_segments/segments_lib.py
# → 0 件であること

grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/alpha_segments/ --include="*.py"
# → build_dataset.py / run_stage*.py の import 行と ln_market_q 列参照のみであること
```

## セグメント定義（確定。後出し変更禁止）

| ID | セグメント | 条件式（レース単位） |
|----|-----------|--------------------|
| S1 | 新馬・未出走馬中心レース | レース内の `hist_last_rank` が NaN の馬の比率 ≥ 0.5 |
| S2 | 少頭数レース | `horse_count <= 8` |
| S3 | 重・不良馬場 | `track_condition_code in (3, 4)` |
| S4 | ローカル開催場 | `course_code in (1, 2, 3, 4, 7, 10)`（中央4場=5,6,8,9 以外） |
| S5 | 低クラス条件戦 | `race_condition_code in (703, 5)`（未勝利・1勝クラス） |

## 実行手順

```bash
# 0. テスト（TDD。合成データのみ、実データ不要）
python -m pytest pure_rank/experiments/alpha_segments/tests/ -v

# 1. データセット構築（fold2 OOS scores + features + RA + odds/q）
python pure_rank/experiments/alpha_segments/build_dataset.py

# 2. Stage 1: セグメント別 n 集計、K 確定（TEST 非接触）
python pure_rank/experiments/alpha_segments/run_stage1_counts.py

# 3. Stage 2: 確定セグメント別 α LRT（fit=2023 → eval=2024）
python pure_rank/experiments/alpha_segments/run_stage2_lrt.py

# 4. Stage 3（一次通過セグメントが存在し evaluator が承認した場合のみ、TEST 各1回）
python pure_rank/experiments/alpha_segments/run_stage3_test.py
```

## ディレクトリ構成

```
pure_rank/experiments/alpha_segments/
├── README.md
├── config.json             # セグメント定義・期間定数・閾値（ハードコード禁止の受け皿）
├── segments_lib.py         # 純関数のみ。市場情報列に一切触れない
├── build_dataset.py
├── run_stage1_counts.py
├── run_stage2_lrt.py
├── run_stage3_test.py
├── data/                   # gate_dataset.parquet
├── results/                # stage1_counts.json / alpha_segments.json / alpha_segments_test.json
└── tests/
```

## 既存モジュールの import 再利用（コピー禁止）

- 条件付きロジット MLE / LRT: `prob_fusion/src/fit_fusion.py`
  （`build_race_tuples`, `fit_fusion_mle`, `likelihood_ratio_test`, `mean_logloss`, `top1_hit_rate`）
- fit/eval 分割（αゲートと同一）: `evaluation/alpha_gate.py::split_alpha_gate_cv`
- オッズ付与・市場 q: `evaluation/odds_loader.py::attach_odds_from_se_parquet`,
  `prob_fusion/src/market_prob.py::attach_market_q`
- TEST 分割（Stage 3 のみ）: `prob_fusion/src/oos_protocol.py::split_oos_periods`

## 事前登録判定基準（要約。詳細は仕様書 §3, §7, §11）

- 最小サンプル: fit 期間 eval 年（2024）で n≥300 レース未満のセグメントは検定対象外。
- 一次判定: LRT p < 0.01/K（K=確定セグメント数、Bonferroni）かつ ΔLL/race > 0（2024 eval）。
- 二次判定（TEST、1回のみ）: TEST logloss（fusion）< 市場、かつ Top-1 ≤ 40%・Spearman ≤ 0.60。
- リーク停止: どの段階でも Top-1 > 40% または Spearman > 0.6 → 即停止・evaluator 報告。
- 全滅時: `verdict = "market_efficiency_holds_across_segments"` を記録して P1 終了。
  セグメントの後出し追加・再定義による延長は禁止。次アクションは P2（Track B）。
