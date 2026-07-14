# track_b_training — 調教時系列 5 候補の αゲート審査（P2 Track B）

**目的**: HC（坂路、511万行）・WC（ウッド、74万行）の生時系列から個体内の
**変化・頻度・形状**（水準は使わない）に基づく事前登録済み 5 候補（B-1〜B-5）を
1 本ずつ生成し、稼働済み αゲート（`evaluation/alpha_gate.py`、γ の LRT）で
市場条件付き尤度改善を審査する。

**仕様書**: `docs/specs/2026-07-10-p2-track-b-training-spec.md`

## 隔離宣言（本番非接触）

本実験は `pure_rank/experiments/track_b_training/` 配下に完結する。以下は書き換えない:
`pure_rank/models/`、`pure_rank/data/02_features/*.parquet`、`pure_rank/data/03_scores/`、
`pure_rank/config/train_config.json`、`prob_fusion/data/`、`prob_fusion/src/`、
`evaluation/reports/`（`gate_summary.json` を含む）。
`gate_summary.json` への結果反映は evaluator の判定後に別タスクとして
`evaluation/update_gate_summary.py` 経由で行う。本実験のスクリプトは
`results/` のみに出力する（`run_alpha_gate` は必ず `out_dir=results/` で呼ぶ）。

## 市場情報境界（プロジェクト憲法遵守）

- 候補生成の入力は HC/WC（JRA 公式計測の調教タイム）とレースキー
  （`race_id, horse_num, ketto_num, race_date`）のみ。オッズ・人気・払戻・
  `market_log_odds`・`init_score` は候補生成コードで一切参照しない。
- 市場確率 q（`ln_market_q`）が現れるのは `evaluation/alpha_gate.py` 内部
  （L2 条件付きロジット統合の変数）のみ。候補生成側 (`training_lib.py`,
  `build_candidates.py`) には現れない。
- z の二重使用なし（`pure_score_z` は αゲート内部の α·z の一箇所のみ）。

検証コマンド:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score\|market_q\|ln_market" \
  pure_rank/experiments/track_b_training/training_lib.py \
  pure_rank/experiments/track_b_training/build_candidates.py
# → 0 件であること

grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/track_b_training/ --include="*.py"
# → run_gate.py の alpha_gate import・コメントのみであること
```

## 候補定義（B-1〜B-5。確定。後出し変更・符号反転禁止）

| ID | 名前 | 仮説 | 符号 |
|----|------|------|------|
| B-1 | `b1_intensity_trend` | 直近30日の坂路4Fタイムの個体内傾き | タイム短縮=正 |
| B-2 | `b2_freq_change` | 前走間隔の調教本数 vs 個体基準（中央値）比 | 本数多い=正 |
| B-3 | `b3_accel_profile` | 直近3本の坂路加速度 - キャリア平均 | 直近上振れ=正 |
| B-4 | `b4_fade_trend` | ラスト200m失速率（hc_200/(hc_3f/3)）の個体内傾き | 失速率低下=正 |
| B-5 | `b5_wc_switch` | 直近30日のWC比率 - キャリアWC比率 | ウッド比率増=正 |

詳細な計算式・NaN 規約・却下条件は仕様書 §5 を参照（本 README は要約のみ）。

## 実行手順

```bash
# 0. テスト（TDD。合成データのみ、実データ不要）
python -m pytest pure_rank/experiments/track_b_training/tests/ -v

# 1. 候補生成（1実行1候補）
python pure_rank/experiments/track_b_training/build_candidates.py --candidate b1
python pure_rank/experiments/track_b_training/build_candidates.py --candidate b2
python pure_rank/experiments/track_b_training/build_candidates.py --candidate b3
python pure_rank/experiments/track_b_training/build_candidates.py --candidate b4
python pure_rank/experiments/track_b_training/build_candidates.py --candidate b5

# 2. αゲート実行（各候補ごと。fit=2023 -> eval=2024）
python pure_rank/experiments/track_b_training/run_gate.py --candidate b1
python pure_rank/experiments/track_b_training/run_gate.py --candidate b2
python pure_rank/experiments/track_b_training/run_gate.py --candidate b3
python pure_rank/experiments/track_b_training/run_gate.py --candidate b4
python pure_rank/experiments/track_b_training/run_gate.py --candidate b5
```

## ディレクトリ構成

```
pure_rank/experiments/track_b_training/
├── README.md
├── config.json          # 窓幅・最小本数・WC_START・符号規約・判定閾値
├── training_lib.py       # 純関数のみ。市場情報列に一切触れない
├── build_candidates.py   # HC/WC + レースキー -> data/cand_b{n}_*.parquet + meta.json
├── run_gate.py           # run_alpha_gate(out_dir=results/) + eval Spearman 補完
├── data/                 # cand_b{n}_*.parquet / *.meta.json
├── results/               # alpha_gate_b{n}_*.json / track_b_summary.json
└── tests/                 # TDD テスト（合成データのみ）
```

## 既存モジュールの import 再利用（コピー禁止）

- αゲート本体: `evaluation/alpha_gate.py::run_alpha_gate`
- 条件付きロジット MLE / LRT: `prob_fusion/src/fit_fusion.py`
- TEST 分割（二次判定のみ）: `prob_fusion/src/oos_protocol.py::split_oos_periods`

## 判定基準（要約。詳細は仕様書 §3, §7, §11）

- 一次判定: γ の LRT p < 0.01（fit=2023）かつ ΔLL/race > 0（eval=2024）。Bonferroni なし。
- Top-1 退行防止: eval_top1 ≥ 0.2999。
- リーク停止: Top-1 > 40% / Spearman > 0.6 / |r| ≥ 0.7 → 即停止・evaluator 報告。
- 二次判定（TEST、1回のみ、一次通過候補のみ）: TEST logloss(H1) < TEST logloss(H0) かつ
  TEST Top-1 ≤ 40%・Spearman ≤ 0.60。
- 5 本全滅時: `verdict = "training_data_no_market_signal_at_current_granularity"` を記録して
  P2 終了（後出し追加禁止）。
