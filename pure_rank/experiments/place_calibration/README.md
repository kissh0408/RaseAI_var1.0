# place_calibration — Stern 複勝確率の較正改善実験（λ再フィット / isotonic）

**目的**: 複勝(top3)確率を返す Stern 逆算式の較正を改善する。
logloss 目的の λ2/λ3 再フィット（A系列）と isotonic 事後較正（B系列）を、
現行 Stern（Brier fit λ 固定 = S0）と logloss・較正誤差の両軸で比較する。

**答える問い**: 「logloss 目的の λ 再フィット、または isotonic 事後較正は、
現行 Stern を logloss と較正誤差の両方で改善するか」。市場超えの判定ではない。

**仕様書**: `docs/specs/2026-07-09-stern-calibration-spec.md`
**前フェーズ**: `docs/specs/2026-07-09-place-direct-prediction-spec.md`（verdict = `not_superior_no_reattempt`、Stern が現状最良と確定）

## 隔離宣言（本番非接触）

本実験は `pure_rank/experiments/place_calibration/` 配下に完結する。以下は書き換えない:
`prob_fusion/src/place_prob.py`、`prob_fusion/src/oos_protocol.py`、`evaluation/reports/`、
`prob_fusion/data/*.parquet`、`pure_rank/data/02_features/*.parquet`、`pure_rank/models/`、
`pure_rank/config/train_config.json`。
`evaluation/reports/gate_summary.json` の合否判定にはこの実験結果を反映しない。

## 市場情報境界（プロジェクト憲法遵守）

- 本実験は新しい特徴量を一切作らない（p_win → 較正のみ）。
- 較正 fit の入力は **p_win（既存 fold2 OOS コードパス由来）と複勝実績（finish_rank<=3）のみ**。
  オッズ・人気を較正の説明変数に使わない。data/ の parquet にも odds/market_q 列を保存しない。
- p_win が formal 融合（α=0, β=1.034）由来で市場確率 q を含むのは L2 の条件付きロジット統合
  として許容済みの範囲（本実験はその出力を消費するだけ）。

検証コマンド:

```bash
grep -rn "odds\|popularity\|ninki\|market_log_odds\|init_score" \
  pure_rank/experiments/place_calibration/ --include="*.py"
```

## 比較系列（事前登録 5 系列。追加・変更禁止）

| ID | 系列 | λ2/λ3 | fit 目的 | 事後較正 | 正規化 |
|----|------|-------|---------|---------|--------|
| S0 | 現行 Stern（基準） | 0.6018/0.6381 固定 | Brier（既存formal） | なし | なし |
| A1 | logloss λ 再フィット | 再fit（global） | logloss | なし | なし |
| A2 | 頭数帯別 λ 再フィット | 再fit（5–7頭/8頭以上） | logloss | なし | なし |
| B1 | isotonic raw | S0固定 | —（isotonic fit） | isotonic | なし |
| B2 | isotonic normalized | S0固定 | 同上 | isotonic | 合計3（clip+再配分） |

fit = 2023-01-01..2024-12-31（formal λ fit と同一期間・同一 6,786 レース）。
TEST = 2025-01-01..（既存 OOS と同一 4,775 レース / 66,020 頭）。TEST 評価は一度だけ。

## 実行手順

```bash
# 0. テスト（TDD。純関数テストは実データなしで走る。統合テストは build 後に有効化）
python -m pytest pure_rank/experiments/place_calibration/tests/ -v

# 1. fold2 OOS スコア → p_win 算出（前フェーズ(a)と同一コードパス）→ fit/TEST 分割
python pure_rank/experiments/place_calibration/build_dataset.py

# 2. S0 再現検証（§4 のゲート。合格してから 3 へ進む）
python -m pytest pure_rank/experiments/place_calibration/tests/test_baseline.py -v

# 3. fit 期間のみで A1/A2/B1/B2 を fit
python pure_rank/experiments/place_calibration/fit_calibrators.py

# 4. TEST で 5 系列の p_place を算出
python pure_rank/experiments/place_calibration/export_probs.py

# 5. 評価レポート生成（指標・ブートストラップCI・順位保存検証・verdict）
python pure_rank/experiments/place_calibration/evaluate_calibration.py
```

## ディレクトリ構成

```
pure_rank/experiments/place_calibration/
├── README.md
├── config.json              # 系列定義・fit期間・S0基準値（ハードコード禁止の受け皿）
├── calib_lib.py             # 純関数（λ logloss fit / 帯割当 / isotonic / 順位保存）
├── build_dataset.py
├── fit_calibrators.py
├── export_probs.py
├── evaluate_calibration.py
├── data/                    # fit/test parquet・TEST 5系列確率
├── models/                  # lambda_fit.json / isotonic_b.joblib
├── reports/                 # place_calibration_comparison.json
└── tests/                   # 仕様書 §8 のテスト
```

## 既存モジュールの import 再利用（コピー禁止）

- Stern 式本体: `prob_fusion/src/place_prob.py`（`place_prob_from_p_win`）
- p_win 確率化・fit/TEST 分割: `prob_fusion/src/oos_protocol.py`（`split_oos_periods`, FIT/TEST 定数）
- 較正誤差ビン: `betting/src/pair_probs.py` の `calibration_max_error_pp`
  （`betting/analysis/compare_pair_probability_models.py` が使うものと同一関数）
- B2 正規化・logloss: `pure_rank/experiments/place_direct/place_lib.py`
  （`normalize_place_probs`, `place_logloss`）

## 既知の制約・注記

- 同一オッズの2頭は p_win が完全一致し Stern place 確率も数学的に同値になるため、
  argmax による top1 は float 総和順序（~1e-16）で揺れる。順位保存検証は
  tie_tol=1e-9 の数値タイを許容し、真の順位反転のみを実装バグとして扱う
  （TEST 2025+ で該当12〜16レースを実測確認済み）。
- 5〜7頭レースの複勝は2着まで払戻のため、本実験の確率をそのまま少頭数レースの
  複勝 EV 計算に使ってはならない（前フェーズと同じ制約。betting 接続はスコープ外）。
